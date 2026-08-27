import torch
import torch.nn as nn
import torch.nn.functional as F

from PytorchWildlife.data import transforms as pw_trans
from PytorchWildlife.models.classification import CustomWeights
from PytorchWildlife.models.classification.resnet_base.base_classifier import (
    PlainResNetClassifier,
)

__all__ = ["SINClassifierInference"]

## Scale LDAMLoss applies to the cosine logits before cross-entropy
## (PW_FT_classification/src/models/losses.py: F.cross_entropy(self.s*output, ...)).
## It is a call-site literal there, not recorded in the checkpoint, so it is
## mirrored here. Only LDAM applies it; CE and Focal use the raw output.
LDAM_LOGIT_SCALE = 30.0

## Keys a training checkpoint carries that inference has no home for.
RSG_DROP_PREFIXES = ("net.feature.RSG.",)
RSG_DROP_KEYS = frozenset({"net.criterion_cls.weight"})

## The RSG backbone never builds an fc layer (see rsg_resnet.ResNet.__init__),
## while upstream's torchvision-derived ResNetBackbone always does. It is unused
## by forward, so the gap is expected rather than a mismatch.
RSG_EXPECTED_MISSING = frozenset({"net.feature.fc.weight", "net.feature.fc.bias"})


class NormedLinear(nn.Module):
    """
    Cosine-similarity head, vendored from PW_FT_classification/src/models/rsg_resnet.py.

    Weight is (in_features, out_features), transposed relative to nn.Linear, and
    there is no bias, so the checkpoint's shapes only load against this.
    """

    def __init__(self, in_features, out_features):
        super(NormedLinear, self).__init__()
        self.weight = nn.Parameter(torch.Tensor(in_features, out_features))
        self.weight.data.uniform_(-1, 1).renorm_(2, 1, 1e-5).mul_(1e5)

    def forward(self, x):
        return F.normalize(x, dim=1).mm(F.normalize(self.weight, dim=0))


class SINClassifierInference:
    """
    Adapter between a PW_FT_classification checkpoint and PytorchWildlife inference.

    Handles both architectures the training tool produces. They share a backbone at
    inference, since RSGResNet's _forward_impl skips the RSG module entirely when
    phase_train is False and reduces to the same flatten(avgpool(layer4(...))) as
    upstream's ResNetBackbone. They differ only in the classifier head:

      Plain  nn.Linear, weight (num_cls, 2048) with bias. Loads into upstream's
             CustomWeights as-is, strict, with nothing left over.
      RSG    NormedLinear, weight (2048, num_cls), no bias. Needs the head swapped
             and the training-only keys filtered out.

    Confidence scale is not comparable between the two. A NormedLinear head emits
    cosine similarities in [-1, 1], and this repo's RSG checkpoint trained with
    loss_type CE, which applies no scale, so its softmax cannot exceed ~0.17 over
    36 classes. Rescaling it would be post-hoc calibration rather than a fix, and
    is deliberately not done: the scale is read from the checkpoint's loss_type so
    a future LDAM model gets its 30x and a CE model does not. This matters beyond
    reporting, because sc_utils/smoother.py averages the confidence vector across
    frames and then takes argmax, and averaging does not commute with a softmax
    temperature change, so the wrong scale changes which class wins.
    """

    IMAGE_SIZE = 224

    def __init__(self, weights, device="cpu", verbose=True):
        self.device = device
        self.verbose = verbose

        checkpoint = torch.load(weights, map_location=device, weights_only=False)
        state_dict = checkpoint["state_dict"]
        hparams = checkpoint.get("hyper_parameters", {}) or {}

        ## Index order is authoritative and must not be sorted: labels 0-34 happen
        ## to be alphabetical but the 36th was appended out of order, so sorting
        ## silently shifts one class.
        id_to_labels = hparams["id_to_labels"]
        self.CLASS_NAMES = [id_to_labels[i] for i in range(len(id_to_labels))]
        self.num_cls = len(self.CLASS_NAMES)

        self.architecture = self.detect_architecture(state_dict, hparams)
        self.loss_type = hparams.get("loss_type")
        self.logit_scale = LDAM_LOGIT_SCALE if self.loss_type == "LDAM" else 1.0

        if self.architecture == "Plain":
            self.model = CustomWeights(weights=weights,
                                       class_names=self.CLASS_NAMES,
                                       device=device)
            self.transform = self.model.transform
        else:
            self.model = None
            self.net = self.build_rsg_net(state_dict)
            self.transform = pw_trans.Classification_Inference_Transform(
                target_size=self.IMAGE_SIZE)

        self.log_load()

    def detect_architecture(self, state_dict, hparams):
        """
        Decide Plain vs RSG from the weights themselves.

        The state_dict is the ground truth: config_classify.yaml records the settings
        of the most recent edit, not of the run that produced this file, and the two
        already disagree on loss_type. hyper_parameters is used only to cross-check,
        and a disagreement is raised rather than guessed past, because loading the
        wrong head produces plausible-looking nonsense.
        """
        has_rsg = any(k.startswith(RSG_DROP_PREFIXES) for k in state_dict)
        weight = state_dict.get("net.classifier.weight")
        ## NormedLinear stores (in_features, out_features), so a 2048-row weight
        ## with no matching bias is the RSG head.
        transposed = (weight is not None
                      and weight.shape[0] == 2048
                      and "net.classifier.bias" not in state_dict)

        from_weights = "RSG" if (has_rsg or transposed) else "Plain"

        recorded = hparams.get("model_name")
        expected = {"RSGResNet": "RSG", "PlainResNetClassifier": "Plain"}.get(recorded)
        if expected is not None and expected != from_weights:
            raise ValueError(
                "Checkpoint disagrees with itself: state_dict looks like {} but "
                "hyper_parameters model_name is {!r}.".format(from_weights, recorded))

        return from_weights

    def build_rsg_net(self, state_dict):
        """
        Build the RSG model and load the checkpoint into it.

        strict=False is unavoidable here, so the dropped and missing key sets are
        asserted explicitly instead. Otherwise a genuine mismatch, such as a renamed
        layer or a different depth, would load silently and quietly degrade.
        """
        holder = nn.Module()
        holder.net = PlainResNetClassifier(num_cls=self.num_cls, num_layers=50)
        holder.net.classifier = NormedLinear(2048, self.num_cls)

        dropped = {k for k in state_dict
                   if k.startswith(RSG_DROP_PREFIXES) or k in RSG_DROP_KEYS}
        filtered = {k: v for k, v in state_dict.items() if k not in dropped}

        result = holder.load_state_dict(filtered, strict=False)

        unexpected = set(result.unexpected_keys)
        missing = set(result.missing_keys)
        if unexpected:
            raise ValueError(
                "Unexpected keys after filtering: {}".format(sorted(unexpected)))
        if missing != RSG_EXPECTED_MISSING:
            raise ValueError(
                "Missing keys {} do not match the expected {}.".format(
                    sorted(missing), sorted(RSG_EXPECTED_MISSING)))

        self.dropped_keys = sorted(dropped)
        self.missing_keys = sorted(missing)

        holder.eval()
        holder.net.to(self.device)
        return holder.net

    def log_load(self):
        if not self.verbose:
            return
        print("SINClassifier: {} head, {} classes, loss_type={!r}, logit scale {}x".format(
            self.architecture, self.num_cls, self.loss_type, self.logit_scale))
        if self.architecture == "RSG":
            print("  dropped {} training-only keys, tolerated {} unused: {}".format(
                len(self.dropped_keys), len(self.missing_keys), self.missing_keys))
            print("  max attainable confidence ~{:.3f}, cosine logits are not "
                  "comparable to the Plain head".format(self.max_confidence()))

    def max_confidence(self):
        """
        Ceiling on softmax confidence for a cosine head, so a capped scale shows up
        in the log rather than being mistaken downstream for an unconfident model.

        Returns None for the Plain head, which is an unbounded nn.Linear and so has
        no ceiling to report -- the [-1, 1] bound this computes is a property of
        NormedLinear alone, and quoting it for Plain would understate a head that
        routinely returns confidences above 0.9.
        """
        if self.architecture != "RSG":
            return None
        best = torch.full((self.num_cls,), -1.0)
        best[0] = 1.0
        return torch.softmax(best * self.logit_scale, dim=0).max().item()

    def forward(self, img):
        feats = self.net.feature(img)
        return self.net.classifier(feats)

    def results_generation(self, logits, img_ids, id_strip=None):
        probs = torch.softmax(logits * self.logit_scale, dim=1)
        preds = probs.argmax(dim=1)
        confs = probs.max(dim=1)[0]

        results = []
        for i, (pred, img_id, conf) in enumerate(zip(preds, img_ids, confs)):
            results.append({
                "img_id": str(img_id).strip(id_strip),
                "prediction": self.CLASS_NAMES[pred.item()],
                "class_id": pred.item(),
                "confidence": conf.item(),
                ## Same [label, score] pairing CustomWeights emits, so the Plain and
                ## RSG paths stay interchangeable for callers.
                "all_confidences": [[name, score] for name, score
                                    in zip(self.CLASS_NAMES, probs[i].tolist())],
            })
        return results

    def single_image_classification(self, img, img_id=None, id_strip=None):
        if self.model is not None:
            return self.model.single_image_classification(img, img_id=img_id,
                                                          id_strip=id_strip)

        from PIL import Image
        img = Image.open(img) if isinstance(img, str) else Image.fromarray(img)
        img = self.transform(img)
        logits = self.forward(img.unsqueeze(0).to(self.device))
        return self.results_generation(logits.cpu(), [img_id], id_strip=id_strip)[0]
