# Two-model aorta engineering run

**Branch:** `fakect.26.09.16`. **Started:** 2026-09-28 local Wahab time.
**Current status:** all 192 native pairs and their companion change masks are
prepared and verified. Both models completed five training epochs, all 32
held-out scenarios were tested in both stages, and the HTML report is complete.
Job `6855976` finished with exit `0:0` in **47 minutes 20 seconds**, with peak
host RSS **1,988,804 KiB (1.90 GiB)**. The pipeline executes correctly, but this
first learned baseline is not reliable for pathology interpretation.

Open the [complete HTML report](../../../outputs/experiments/coa/two-stage-r2/report.html)
or the [machine-readable review](../../../outputs/experiments/coa/two-stage-r2/review-summary.json).
The first attempt remains preserved as a TensorFlow runtime diagnostic.

The [input](../../../configs/models/coa-two-stage.ini) specifies two compact
3D U-Nets: attenuation-to-observed-aorta and observed-mask-to-reference-mask.
Their derived differences identify estimated added and removed regions.
The [guide](../../TWO_STAGE_TRAINING.md) documents scope and commands.

| Job | Work | Resources / dependency |
| --- | --- | --- |
| `6855939` | Twelve frozen native paired preparations | CPU array; up to six concurrent tasks, each 1 CPU / 8 GB / 6 h |
| `6855944` | Verify pairs and write shared manifest/change companions | 1 CPU / 8 GB / 1 h; after all preparation tasks succeed |
| `6855949` | Initial TensorFlow 2.17 GPU fixture | Failed at unsupported deterministic XLA max-pooling gradient; preserved diagnostic |
| `6855953` | Corrected GPU fixture | Passed on Tesla V100-SXM2-16GB; both models fit/reloaded/tested |
| `6855966` | Tiny training-only overfit check | Passed; both models learn the known synthetic deformation |
| `6855955` | Initial real-data attempt (`two-stage-r1`) | Stopped before epoch 1: per-batch graph retracing and unbounded host memory |
| `6855975` | Fixed training graph GPU check | Passed: one graph per model through 32 updates with actual 64-cubed patches, save/reload and oracle/chained tests |
| `6855976` | Corrected real-data run (`two-stage-r2`) | Completed both fits, held-out tests and HTML; 1 GPU / 4 CPUs / 16 GB allocation |

The correction uses stride-2 3D convolutions for downsampling and explicitly
disables XLA JIT; deterministic TensorFlow operations remain enabled. The GPU
fixture used TensorFlow 2.17.0, hid test payloads during fitting, reloaded both
models on GPU and checked exact reference/change-mask identities. Its result
is at `outputs/experiments/coa/two-stage-gpu-smoke-r2/result.json`.

All **50 focused two-stage tests pass** with
`python3 -m unittest discover -s tests -p 'test_two_stage_*.py' -v`.
These cover native pair bindings/change masks, model/config locks, family split
isolation, real TensorFlow 2.20 CPU fitting and save/reload, chained output
identities, corruption rejection, completed-run reuse and HTML rendering.
The subsequent metadata/history checksum hardening was covered by the CPU
suite; the GPU fixture proves the same network/downsampling/training operations
in the TensorFlow 2.17 deployment environment.

After fixing the batch loop, all **12 learning tests passed again in the
TensorFlow 2.17 container** (81.6 seconds), including bounded graph counts with
varying batch sizes, both models' fit/save/reload/test, held-out payload isolation
and integrity checks. A separate repeat under login-host TensorFlow 2.20 timed
out during library import/source-file I/O, before reaching the backend; it is
not counted as a pass for the corrected loop.

The first full run reached 22,902,100 KiB host RSS before completing an epoch.
The installed TensorFlow 2.17 Keras implementation creates a new generator on
each `train_on_batch` call, triggering a new graph. A fresh-process probe
confirmed 40 graphs after 40 batches; the fixed tensor-signature step stayed
at two graphs with flat memory. The full-size GPU check then used one graph per
model; 16 warm batches took 0.63/0.56 seconds. The loop calls the same public
Keras `train_step`, with unchanged optimizer and loss. Epoch histories now
record trace counts and training/validation times, and reject excessive tracing.
The first full corrected epoch completed in 270.3 seconds with one graph,
512 training patches and 1,920 complete-crop validation patches.

The separate overfit fixture used two 16-cubed examples and 100 updates per
model, without reading any real-cohort image or held-out result. Model 1 Dice
improved from **0.1417 to 0.9877**. Model 2 reference Dice improved from **0.1598
to 1.0**, with exact recovery of all 48 added and 48 removed voxels and no false
changes on the control. Save/reload predictions matched. The full settings and
history are in `outputs/experiments/coa/two-stage-overfit-r1/result.json`.
These numbers establish learnability on a tiny fixture, not generalization.

Complete data: 192 native scenarios, including twelve unmodified
references and 180 edited siblings. The companion manifest is
`outputs/datasets/coa/two-stage-radius115-r1/dataset-manifest.json`.
It references original native arrays by checksum and adds compressed change
masks; it does not alter frozen generation or relabel provisional families as
verified. Generated arrays and models are ignored by Git.

The native pairs occupy **1,115,281,347 bytes (1.04 GiB)**; the compressed change
sidecars add **1,145,743 bytes (1.1 MiB)**. All 180 edited scenarios contain both
nonzero net additions and removals. Full source-payload verification ran during
manifest creation, followed by an independent metadata/plan check. Manifest
SHA256: `da53eca72b738b03cbc4765dd347bd244ad9d63cac5ca19bfaca294023fedff0`.

The locked design uses 128 training, 32 validation and 32 test scenarios:

- Training: 260613, 260614, 260615, 260616, 260618, 260619, 260620, 260621.
- Validation: 260602, 260611.
- Test: 260612, 260617.

This is a pediatric-to-adult validation and infant-held-out engineering design,
with known duplicate anatomy kept together. The selected ROI lineage defines
the target; complete thoracic coverage and whole-CT localization are not yet
validated. Current images are attenuation-copy proxies. Results must not be
interpreted as demonstrated clinical generalization.

The eight training case IDs contain only **five distinct original-mask
geometries**: the 1-, 5- and 10-year pairs duplicate their corresponding
scenarios. The 128 training rows contain 80 distinct final masks and images.
They remain in the same split; sample counts must not be interpreted as counts
of independent anatomical subjects. Each split includes one unchanged control
per case ID (8 training, 2 validation and 2 test controls).

Five epochs per model use every training scenario, with four sampled patches
per scenario per epoch. Physical model spacing is 0.6 mm, patches are 64 cubed,
and validation/test cover complete crops. The report will separate segmentation,
reference reconstruction with exact masks, and chained predictions; retain the
identity comparator, unchanged controls and segmentation-error overlap.

A read-only normalization check used only baseline and edited samples from
260616 (training) and 260602 (validation), on native and model grids. All image
values were finite and inside the configured 0–0.5 cm^-1 clipping interval;
there was no clipping. Native aortic attenuation was about 0.183805 cm^-1 in
both cases, mapping to 0.367610 normalized, with matching model-grid target
medians. Target occupancy was only 2.11–2.81% of the full model crops. Other
tissue shares the same intensity, so the selected-aorta task requires spatial
context beyond scalar thresholding. No test payloads or hyperparameters were
used or changed by this check.

Run outputs: `outputs/experiments/coa/two-stage-r2/` (corrected loop).
The failed attempt remains under `two-stage-r1/`, including its locked input,
code hashes and matching source snapshots in `diagnostic-source/src/`.
Preparation logs: `outputs/cohorts/coa/radius115/preparation-logs/`.
GPU log: `logs/fakect_two_stage.6855976.out` and matching `.err`.
Model 1 completed five epochs and selected epoch 5 by validation loss
(1.023218), with voxel-pooled validation Dice 0.245529. Every epoch visited 512
training patches and 1,920 validation patches with one training graph. No test
payload was read during fitting. This modest validation result is a baseline
for improvement, not an accepted segmentation model.

Model 2 selected epoch 4 by validation loss 0.041282, with validation reference
Dice 0.955550. Epoch 5 had higher Dice but worse selection loss; it was not used
for testing. Both models maintained exactly one training graph throughout.

## Held-out results

The table averages scenarios within each of the two test case IDs, then averages
the cases within their single provisional infant family. All scores use the
0.6 mm model grid and include the two unchanged controls unless stated otherwise.

| Measurement | Mean test Dice |
| --- | ---: |
| Model 1: observed aorta | 0.4221 |
| Model 2: reference from exact observed mask | 0.9183 |
| Identity baseline: keep observed mask as reference | **0.9799** |
| Model 2: expansion / narrowing, exact input | 0.2218 / 0.3491 |
| Chained reference: Model 1 input to Model 2 | 0.4274 |
| Chained expansion / narrowing versus true edits | 0.0011 / 0.1102 |

On the **30 edited scenarios only**, oracle-input expansion/narrowing Dice is
0.2366/0.3724, with precision 0.1487/0.2394 and recall 0.8771/0.8692. Reference
Dice is 0.9188 versus identity 0.9785. The low precision demonstrates substantial
overcorrection. Interpret pathology recall on edited cases: the empty-control
convention assigns recall 1 when there is no true lesion and can inflate a
pooled mean, especially for the chained expansion predictions.

On the **two unchanged controls**, exact-input Model 2 incorrectly predicts mean
expansion/narrowing volumes of **248.94/127.66 mm³**, versus zero for identity.
Chaining raises those means to **296.03/377.68 mm³**. Across the test set,
**85.4% of chained expansion predictions overlap Model 1 segmentation errors**.
High whole-reference overlap does not establish successful lesion recovery.

An independent review checked all 64 saved prediction bundles: sample IDs,
JSON and prediction checksums, before/after net masks, reference/change Boolean
identities, probability formulas, TP/FP/FN, Dice and false-positive volumes all
matched. Model 1 measurements match the same predictions inside the chained
evaluation. Representative exact-input and chained slices were inspected; the
original portable HTML contains 14 embedded figures (1.62 MB). No fitting or threshold
changes were made after inspecting test results.

The report was subsequently extended with an offline interactive 3D viewer of
all 32 held-out scenarios. It shows exact saved model-grid surfaces with no
pooling, smoothing or mesh decimation, and separates exact-mask, chained and
segmentation-error comparisons. Each label can be hidden or shown with 15%,
45% or 80% opacity. `report.html` now requires the adjacent `report-3d/` directory
(approximately 111 MiB); the original self-contained report remains
`report-2d.html`. This is a visualization revision only: prediction arrays,
thresholds, learned weights and scores are unchanged. Firefox control events
and local-file scene loading were checked; the headless login node could not
validate actual WebGL rasterization because it has no suitable graphics driver.

The [phantom mask workflow](../../PHANTOM_MASK_WORKFLOW.md) now provides
an atlas-based alternative to Model 1 and a separate Model 2 runner. Broader
automatic masks change the target definition and require anatomical review,
a new dataset release and retraining; the old scores must not be interpreted
as whole-thoracic performance.

## Interpretation and next development

The data preparation, independent fitting, held-out testing, artifact integrity
and report generation are implemented and exercised. The first models do not
yet support dependable segmentation or pathology localization. Model 1 needs
better anatomical discrimination; a larger spatial context and a carefully
reviewed target extent are candidate experiments. Model 2 needs stronger
preservation of unchanged anatomy and objective selection that explicitly
evaluates true edits and false corrections, rather than relying on overall
reference overlap. These are proposed experiments, not verified remedies.

Keep this run as the frozen baseline. Use training/validation evidence to design
new runs and a prospective family evaluation protocol before making broader
performance claims. The clinical sibling-population hypothesis still requires
controlled training-cohort comparisons and independent clinical testing.
