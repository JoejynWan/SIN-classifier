import os
import tempfile
import multiprocessing as mp

import numpy as np
from speciesnet import SpeciesNet, SpeciesNetEnsemble
from speciesnet.utils import ModelInfo, prepare_instances_dict
from speciesnet.geofence_utils import (
    should_geofence_animal_classification,
    roll_up_labels_to_first_matching_level,
)
from speciesnet.taxonomy_utils import get_ancestor_at_level

__all__ = ["SpeciesNetInference"]

## Taxonomy levels the roll-up walks, coarsest last. Mirrors the defaults
## SpeciesNet's own ensemble uses in ensemble_prediction_combiner.py.
ROLLUP_LEVELS = ["genus", "family", "order", "class", "kingdom"]
ROLLUP_SCORE_THRES = 0.65

## Score a single species must reach to be reported as-is. Below it the
## prediction is rolled up to a coarser taxonomic level instead. Not a
## keep-vs-discard gate: every detection still gets an answer either way.
SPECIES_SCORE_THRES = 0.8


def get_by_key(lst, key, value):
    normalized_value = os.path.normpath(str(value))
    return next((item for item in lst if os.path.normpath(str(item.get(key))) == normalized_value), None)


def softmax(logits):
    """Softmax over a 1D sequence of logits, shifted for numerical stability."""
    arr = np.asarray(logits, dtype=np.float64)
    shifted = np.exp(arr - arr.max())
    return shifted / shifted.sum()


class SpeciesNetInference:
    """
    Adapter between MegaDetector detection output and the SpeciesNet classifier.

    Converts PytorchWildlife normalized xyxy coordinates into the xywh bboxes
    SpeciesNet expects, and flattens SpeciesNet's nested predictions back into
    per-detection dicts suitable for supervision's Detections class.

    When a country is supplied, classification is restricted to the species that
    country's geofence permits. SpeciesNet applies geofencing in its ensemble
    component, which is never built when only the classifier is loaded, so the
    restriction is applied here instead: the permitted labels are passed as
    target species, and the classifier then reports logits for all of them on
    every prediction regardless of their global rank. Without this the top-5 the
    classifier returns is chosen from all ~2500 global labels, and a locally
    plausible species ranking outside it can never be recovered.
    """
    def __init__(self, version='v4.0.3a/1', run_mode='multi_thread', geofence=True,
                 country=None):

        self.model_url = 'kaggle:google/speciesnet/pyTorch/{}'.format(version)
        self.run_mode = run_mode
        self.geofence = geofence
        self.country = country

        self.progress_bars = False
        self.target_species_path = None

        try:
            mp.set_start_method('spawn')
        except RuntimeError as e:
            if "context has already been set" in str(e):
                pass  # Context is already set, so skip silently
            else:
                raise

        ## Taxonomy and geofence maps ship with the weights. Loading the ensemble
        ## component pulls in only those two files, not the classifier weights.
        self.ensemble = SpeciesNetEnsemble(self.model_url, geofence=self.geofence)
        self.taxonomy_map = self.ensemble.taxonomy_map
        self.geofence_map = self.ensemble.geofence_map

        model_info = ModelInfo(self.model_url)
        with open(model_info.classifier_labels, mode='r', encoding='utf-8') as fp:
            all_labels = [line.strip() for line in fp if line.strip()]

        self.target_labels = self.permitted_labels(all_labels) if self.geofenced else None
        if self.target_labels:
            self.target_species_path = self.write_target_species(self.target_labels)

        self.model = SpeciesNet(
            self.model_url,
            components='classifier',
            geofence=self.geofence,
            target_species_txt=self.target_species_path,
            multiprocessing=(self.run_mode == "multi_process"),
        )

        ## Copy rather than alias: the roll-up labels appended below would
        ## otherwise be injected into the classifier's own label dict.
        self.id_to_label = dict(self.model.classifier.labels)
        self.label_to_id = {v: k for k, v in self.id_to_label.items()}
        if self.target_labels:
            self.register_rollup_labels()

    @property
    def geofenced(self):
        """Geofencing is active only when a country is configured."""
        return bool(self.country) and self.geofence

    def permitted_labels(self, all_labels):
        """Labels the configured country's geofence rules allow."""
        return [label for label in all_labels
                if not should_geofence_animal_classification(
                    label, self.country, None, self.geofence_map, True)]

    def write_target_species(self, labels):
        """
        Write the permitted labels for SpeciesNet to read as target species.

        Entries must be the full 'uuid;class;order;family;genus;species;common_name'
        strings; SpeciesNet silently drops any line it cannot match against its own
        labels, which would quietly degrade to no filtering at all.
        """
        handle = tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False,
                                             encoding='utf-8')
        with handle as fp:
            fp.write("\n".join(labels))
        return handle.name

    def register_rollup_labels(self):
        """
        Give every reachable roll-up label an id.

        Roll-up can return ancestors that are not themselves classifier labels
        (genus- and family-level entries in particular), so they need ids of their
        own before results_generation can look them up. Ids continue on from the
        classifier's own so that the caller's convention of treating the last
        entry as "unknown" still holds once it appends that entry.
        """
        next_id = max(self.id_to_label) + 1
        for label in self.target_labels:
            for level in ROLLUP_LEVELS:
                ancestor = get_ancestor_at_level(label=label, taxonomy_level=level,
                                                 taxonomy_map=self.taxonomy_map)
                if ancestor and ancestor not in self.label_to_id:
                    self.id_to_label[next_id] = ancestor
                    self.label_to_id[ancestor] = next_id
                    next_id += 1

    def detections_dict_generation(self, det_results):
        detections_dict = {}

        for det in det_results:
            det['filepath'] = det['img_id']
            det['detections'] = [{'bbox' : [b[0], b[1], b[2] - b[0], b[3] - b[1],]}
                                 for b in det['normalized_coords']]
            detections_dict[det['filepath']] = det

        return detections_dict

    def resolve(self, labels, scores):
        """
        Turn a score vector over `labels` into (label, score, source).

        Takes the strongest label, falling back to the most specific taxonomic
        level that clears ROLLUP_SCORE_THRES when nothing individual is
        convincing. A weak label is still returned when roll-up finds nothing, so
        the caller always gets an answer and its own threshold decides whether to
        call it unknown.

        There is deliberately no confidence gate above this: roll-up is itself
        the answer to "not sure enough for a species", so callers should take
        what comes back rather than discarding it. Degrading to a coarser rank
        keeps information that a threshold would throw away entirely.
        """
        best = int(np.argmax(scores))
        if scores[best] >= SPECIES_SCORE_THRES:
            return labels[best], float(scores[best]), 'classifier'

        rolled = roll_up_labels_to_first_matching_level(
            labels=list(labels),
            scores=[float(x) for x in scores],
            country=self.country,
            admin1_region=None,
            target_taxonomy_levels=ROLLUP_LEVELS,
            non_blank_threshold=ROLLUP_SCORE_THRES,
            taxonomy_map=self.taxonomy_map,
            geofence_map=self.geofence_map,
            enable_geofence=True,
        )
        if rolled:
            return rolled

        return labels[best], float(scores[best]), 'classifier'

    def best_classification(self, classifications):
        """Pick a label for a single frame's prediction."""
        if not self.geofenced or 'target_logits' not in classifications:
            return (classifications['classes'][0],
                    classifications['scores'][0],
                    'classifier')

        return self.resolve(classifications['target_classes'],
                            softmax(classifications['target_logits']))

    def resolve_smoothed(self, all_confs):
        """
        Re-decide a classification from a temporally smoothed score vector.

        Per-frame scores are noisy: a species can be confident on one frame and
        absent the next, and deciding species-vs-roll-up frame by frame throws
        that instability straight into the output. Smoothing the whole
        distribution first and resolving once means the decision is made on
        evidence pooled across the track. Expects a vector ordered like
        self.target_labels, which is what results_generation emits.
        """
        scores = np.asarray(all_confs, dtype=np.float64).ravel()
        label, score, source = self.resolve(self.target_labels, scores)
        return {
            'class_id': self.label_to_id[label],
            'prediction': label.split(';')[-1],
            'confidence': score,
            'prediction_source': source,
        }

    def results_generation(self, predictions_dict, det_results):
        clf_results = []
        for pred in predictions_dict['predictions']:
            det = get_by_key(det_results, 'img_id', pred['filepath'])
            clf = pred['classifications']
            label, score, source = self.best_classification(clf)

            ## Full per-class vector, for smoothing across frames. Ordered like
            ## target_labels so resolve_smoothed can map positions back to labels.
            all_confidences = None
            if self.geofenced and 'target_logits' in clf:
                probs = softmax(clf['target_logits'])
                all_confidences = [[self.label_to_id[lbl], float(p)]
                                   for lbl, p in zip(clf['target_classes'], probs)]

            for _ in range(len(det['normalized_coords'])):
                result = {
                    'img_id': pred['filepath'],
                    'class_id': self.label_to_id[label],
                    'prediction': label.split(';')[-1],
                    'confidence': score,
                    'prediction_source': source
                }
                if all_confidences is not None:
                    result['all_confidences'] = all_confidences
                clf_results.append(result)
        return clf_results

    def single_image_classification(self, file_path, det_results=None, country=None):

        instances_dict = prepare_instances_dict(
            filepaths=[file_path],
            country = country if country is not None else self.country,
        )

        predictions_dict = self.model.classify(
            instances_dict=instances_dict,
            detections_dict=self.detections_dict_generation([det_results]) if det_results else None,
            run_mode=self.run_mode,
            batch_size=1,
            progress_bars=self.progress_bars,
        )
        return self.results_generation(predictions_dict, [det_results])

    def batch_image_classification(self, data_path, batch_size=8, det_results=None, country=None):

        instances_dict = prepare_instances_dict(
            folders=[data_path],
            country=country if country is not None else self.country,
        )

        predictions_dict = self.model.classify(
            instances_dict=instances_dict,
            detections_dict=self.detections_dict_generation(det_results) if det_results else None,
            run_mode=self.run_mode,
            batch_size=batch_size,
            progress_bars=self.progress_bars,
        )

        return self.results_generation(predictions_dict, det_results)

    def __del__(self):
        ## Module globals can already be torn down when this runs at interpreter
        ## exit, so os itself may be None. Failing to remove a temp file is never
        ## worth an exception here.
        try:
            if getattr(self, 'target_species_path', None):
                os.unlink(self.target_species_path)
        except Exception:
            pass
