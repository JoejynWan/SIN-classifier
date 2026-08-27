# Restore the SIN classifier path in PytorchWildlife_Vid.py

## Context

[general/PytorchWildlife_Vid.py:238](general/PytorchWildlife_Vid.py#L238) calls
`pw_classification.SINClassifier(...)`, a class that exists in no installed package — it lives
only on CameraTrap branch `MDv6_beta` and was never carried across the Pytorch-Wildlife
migration. The line is guarded by `if config.CLS_WEIGHTS_PATH:`, currently `NULL`, so the
script imports fine and the crash is latent. Setting a weights path triggers an immediate
`AttributeError`.

The SpeciesNet path is now complete and committed (`53d861a`). This restores the second,
complementary classifier: SpeciesNet is global and taxonomic, the SIN classifier is
Singapore-specific and trained on local camera-trap footage.

Everything below was verified by loading both checkpoints and reading the training source in
`CameraTrap2\PW_FT_classification`. Three claims in the previous revision of this plan turned
out to be wrong; they are corrected inline and listed under [Corrections](#corrections).

## The two candidate checkpoints

Both are already staged in `models/`, both from training run `SINClassifierV2_20250604`, both
**36 classes** with identical `id_to_labels`:

| | Plain | RSG |
|---|---|---|
| path | `models/Plain_SINClassifierV2_20250604/version_5/…epoch=181-valid_mac_acc=57.98.ckpt` | `models/RSG_SINClassifierV2_20250604/version_14/…epoch=196-valid_mac_acc=34.56.ckpt` |
| `model_name` | `PlainResNetClassifier` | `RSGResNet` |
| `algorithm` / `train_rule` | `Plain` / `None` | `RSG` / `DRW` |
| `loss_type` | `CE` | **`CE`** |
| head | `nn.Linear`, weight `(36, 2048)` + bias | `NormedLinear`, weight `(2048, 36)`, **no bias** |
| macro val acc | **57.98** | 34.56 |
| loads into upstream `CustomWeights` | **yes, `strict=True`, 0 missing / 0 unexpected / 0 shape mismatches** | no — needs a custom path |

**Recommendation: make Plain the default.** It is 23 points more accurate *and* free of the
confidence problem below. Support RSG, but do not point `CLS_WEIGHTS_PATH` at it by default.

Older weights remain dead ends: `SIN_SpClassifier_v1.pt` is a TorchScript archive with no
`state_dict` and 11 coarse classes; the 20240717 checkpoint is superseded (35 classes vs 36).

## Corrections

1. **RSG has no `net.feature.fc.*` keys at all.** The previous plan said `fc` "differs in shape
   (`(num_cls, 2048)` for RSG vs `(1000, 2048)` for Plain) … so it must be dropped rather than
   matched." In fact `rsg_resnet.ResNet.__init__` never builds an `fc`, so for RSG these are
   **missing** keys that must be tolerated, not extra keys to drop. (Plain does carry `fc` at
   `(1000, 2048)`, unused by `forward` — that part was right.)
2. **The key-filter list was incomplete.** Besides `net.feature.RSG.*` (11 tensors), the RSG
   checkpoint carries **`net.criterion_cls.weight`** `(36,)` — the DRW class-weight buffer
   attached to the loss module. It must also be dropped. The measured unexpected-key set is
   exactly those 12.
3. **`loss_type` is `CE` on *both* checkpoints, so the `s=30` scale does not apply.** The rule
   ("read `loss_type` from the checkpoint rather than hardcoding") was correct but assumed
   LDAM. `classify/config_classify.yaml` says `loss_type: LDAM`; the checkpoints say `CE`. The
   checkpoint wins. Confirmed in source: only `LDAMLoss.forward` applies
   `F.cross_entropy(self.s*output, …)`; `CE` and `Focal` apply nothing.

Claims that **held up**: the backbones are equivalent at inference
(`_forward_impl(phase_train=False)` skips RSG entirely and returns
`flatten(avgpool(layer4(…)))`, identical to upstream's `ResNetBackbone`); `NormedLinear` is
cosine similarity with a transposed, bias-free weight; preprocessing matches
(`Resize((224,224))` + `ToTensor` + ImageNet `Normalize`, and `IMAGE_SIZE = 224`).

## The RSG confidence problem

Because this RSG model trained with **CE on unscaled cosine logits**, its outputs are confined
to `[-1, 1]` before softmax, so confidence is structurally capped. Measured over 36 classes:

| logit pattern | max softmax |
|---|---|
| absolute best case: one class `+1`, all others `−1` | **0.174** |
| realistic peak: one class `+0.6`, others ~`0` | **0.050** |
| *(if `s=30` were applied — it must not be)* | 1.000 |

So RSG confidences sit around 3–17% and never approach 1.0, while Plain's are ordinary softmax
probabilities. Consequences:

- **Any absolute confidence threshold is meaningless on RSG** and cannot be shared with the
  Plain model or compared against SpeciesNet's scores.
- **The scale is not merely cosmetic.** [sc_utils/smoother.py:111](sc_utils/smoother.py#L111)
  averages `all_confs` across frames and *then* takes `argmax`. Averaging does not commute with
  a nonlinear softmax temperature change, so applying the wrong scale changes **which class
  wins**, not just the reported number. (By contrast, `get_video_class`'s single argmax is
  rank-preserving and unaffected — the risk is specific to the smoothed path.)
- Applying `s=30` anyway to "fix" the numbers would be **incorrect** — the model was not trained
  that way. Any rescaling of the CE-trained RSG head is post-hoc calibration and must be
  labelled as such, not silently applied.

## Plan

### 1. Add `sc_utils/sin_classifier.py`

A `SINClassifierInference` that inspects a checkpoint and builds the matching model, following
the "adapter lives in the consumer repo" precedent of
[sc_utils/speciesnet_classifier.py](sc_utils/speciesnet_classifier.py).

- **Detect architecture from the `state_dict`**, not from config: RSG iff `net.feature.RSG.*`
  keys are present, or `net.classifier.weight` is `(2048, N)` rather than `(N, 2048)`.
  Cross-check against `hyper_parameters['model_name']` and fail loudly on disagreement rather
  than guessing.
- **Class names from the checkpoint** — `hyper_parameters['id_to_labels']`, a 36-entry dict.
  Never hardcode. **Never sort it**: entries 0–34 are alphabetical but `Porcupines
  (Hystricidae)` was appended at index 35, so sorting silently mislabels 1 class in 36. Index
  order is authoritative. Expose as `self.CLASS_NAMES`, indexable by `class_id`, which
  [get_video_class](general/PytorchWildlife_Vid.py#L112) requires.
- **Plain path:** delegate to `pw_classification.CustomWeights(weights=…, class_names=…,
  device=…)`. Verified to load with zero key or shape discrepancies.
- **RSG path:** reuse upstream's `PlainResNetClassifier` backbone, swap in a vendored
  `NormedLinear` head, drop the 12 unexpected keys (`net.feature.RSG.*` +
  `net.criterion_cls.weight`), and tolerate the 2 missing ones
  (`net.feature.fc.{weight,bias}`). Load with `strict=False` **only after** asserting that the
  missing set is exactly those 2 and the dropped set exactly those 12 — anything else is a real
  mismatch and should raise. (Not 3: `NormedLinear` has no bias, so the model never expects
  `net.classifier.bias` either and it is absent from both sides.)
- **Scale:** read `hyper_parameters['loss_type']`. Apply `s=30` only for `LDAM`; apply nothing
  for `CE`/`Focal`. Log which branch was taken and the resulting max achievable confidence, so
  the ~17% ceiling is visible in the run log rather than discovered downstream.
- **Emit the shape the callback already expects:** `{img_id, prediction, class_id, confidence,
  all_confidences}` so [PytorchWildlife_Vid.py:78](general/PytorchWildlife_Vid.py#L78) needs no
  change. `all_confidences` is `[[class_name, conf], …]` — note this is **names, not ids**, which
  is what `CustomWeights.results_generation` actually emits (`self.CLASS_NAMES[i]`), so the Plain
  and RSG paths stay interchangeable. It differs from `SpeciesNetInference`, which pairs ids; only
  element `[1]` is ever read once step 3 removes `all_class_id`, so the two conventions coexist.

### 2. Wire it into `PytorchWildlife_Vid.py`

Replace the `pw_classification.SINClassifier(...)` call at line 238 with
`SINClassifierInference(weights=config.CLS_WEIGHTS_PATH, device=DEVICE)`. No parallel
`SINClassifier_Vid.py`: that file's callback, `ClassificationSmoother` usage and
`get_video_class` are already built for exactly this classifier.

Point `CLS_WEIGHTS_PATH` in [general/config.yaml](general/config.yaml) at the **Plain**
checkpoint, with a comment naming the RSG one as the alternative and noting its confidence
scale differs.

### 3. Remove the dead `all_class_id`

[PytorchWildlife_Vid.py:79](general/PytorchWildlife_Vid.py#L79) builds `all_class_id` from
`all_confidences[i][0]`; nothing reads it — only `all_confs` is consumed, by
[sc_utils/smoother.py:111](sc_utils/smoother.py#L111). Drop it so no future reader assumes it
is meaningful.

## Remaining gaps

Previously-open gaps now **closed**: an RSG checkpoint exists and is testable; the 36th class is
`Porcupines (Hystricidae)` — a real taxon, not an "unknown"/"other" bucket, so `get_video_class`
needs no special handling; and `loss_type` is present on both checkpoints, so no defaulting is
needed.

Still open:

1. **`s=30` is a call-site literal**, not recorded in the checkpoint. Irrelevant for these two
   CE models, but a future LDAM run would depend on it. Worth recording `s` in checkpoints.
2. **RSG's 34.56 macro accuracy may make it not worth shipping.** The work to support it is
   modest and already specified, but consider whether the second code path earns its
   maintenance cost given Plain scores 57.98.
3. **`PW_FT_classification` is not installed**, so `classify/train_model.py` cannot run in this
   env. Out of scope, but it means the RSG source of truth lives in `CameraTrap2` and should
   eventually come from `microsoft/MegaDetector-Classifier`.
4. **Eventual unification** of `SpeciesNet_Vid.py` and `PytorchWildlife_Vid.py` is a separate
   refactor — they differ in more than the classifier, and the SIN classifier's flat class list
   has no taxonomy to roll up or break ties on.

## Verification

1. **Plain loads:** `SINClassifierInference(weights=<plain ckpt>)` constructs, reports `Plain`,
   exposes 36 `CLASS_NAMES` equal to the checkpoint's `id_to_labels` **in index order**, with
   `CLASS_NAMES[35] == 'Porcupines (Hystricidae)'`.
2. **RSG loads:** same for the RSG checkpoint, reporting `RSG`, with the assertion in step 1 of
   the plan confirming exactly 12 dropped and 2 missing keys.
3. **Inference shape:** `single_image_classification` on one frame returns all five keys, with
   `len(all_confidences) == 36` and confidences summing to ~1.0, for both models.
4. **Confidence ceiling is real, not a bug:** assert RSG's max confidence on the sample clips is
   below ~0.18 and Plain's is not. This turns the finding into a regression test.
5. **Scale branch:** confirm no scale is applied for these `CE` checkpoints, and that forcing
   `LDAM` would apply 30× — proving the branch is live and reading the checkpoint.
6. **End-to-end:** set `CLS_WEIGHTS_PATH` to the Plain checkpoint and run
   `python general\PytorchWildlife_Vid.py` against `data/example_test_set` (`CT7_20211119`,
   `CT11_20211119`; note `general/config.yaml` currently points `SOURCE_DIR` at a
   `C:\TempDataForSpeed\` copy).
7. **No regression to SpeciesNet:** `python general\SpeciesNet_Vid.py` still reproduces the
   committed results.

### Blockers found while running this

1. **`speciesnet` env — fixed.** Its editable install pointed at the pre-rename
   `D:\CNN_Animal_ID\CameraTrap` via a dangling `PytorchWildlife.egg-link`. Replaced with a real
   `pip install -e D:\CNN_Animal_ID\Pytorch-Wildlife --no-deps`; `PytorchWildlife` now resolves
   to the fork and every dependency was already satisfied.
2. **`single_image_detection(verbose=...)` — outstanding, not ours.**
   [PytorchWildlife_Vid.py:54](general/PytorchWildlife_Vid.py#L54) passes `verbose=False`, but the
   fork's `YOLOV8Base.single_image_detection` has no such parameter — it was a CameraTrap-side
   customization (`917231a6`) that did not migrate. The fix exists locally on another desktop and
   is unpushed, so the call site is deliberately left unchanged. Until it lands, end-to-end runs
   on this machine need a shim that forwards `verbose` onto `self.predictor.args.verbose`.
   Without it ultralytics logs one line per frame.
3. **Upstream bug in the fork, unrelated but latent.** `megadetectorv6.py:49` sets
   `MODEL_NAME = "MDV6b-rtdetr-c.pt"` while `yolov8_base.py:63` tests for `'MDV6b-rtdetrl.pt'`
   when choosing `RTDETRPredictor` over `DetectionPredictor`. The strings do not match, so
   `DET_VERSION: 'MDV6-rtdetr-c'` would silently use the wrong predictor. Not hit by the current
   config (`MDV6-yolov9-c`).

## Risks

- **RSG predictions are scale-sensitive through the smoother**, as above. If RSG output ever
  looks subtly wrong, suspect the scale branch before the weights.
- **`strict=False` can hide genuine mismatches** — mitigated by asserting the exact expected
  missing/dropped key sets rather than trusting the flag.
- **Both models are trained on 2025-06 data**; accuracy on current footage is unknown. Compare
  against SpeciesNet on the same clips before concluding a wiring bug.
- **`get_video_class` selects by maximum confidence**, so one overconfident frame decides a whole
  video — worth watching, though it is rank-preserving and so unaffected by the scale question
  itself.
