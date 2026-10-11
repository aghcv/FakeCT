# CoA multiresolution engineering pilot

> Historical generated artifacts are recorded as paths below; they are not included in this source checkout.

## Active learning-rate control and poster figures

The fresh single-factor control in
[coa-focused-pilot-lr-control.ini](../configs/models/coa-focused-pilot-lr-control.ini)
was submitted as GPU job **6902259**. Adam learning rate changes from
**0.001 to 0.0003**; architecture, losses, fresh initialization, the 32/8 panel,
family partitions, frozen locator, 50 epochs and 800 updates remain the same.
Minimum validation BCE again selects the checkpoint. The new experiment freezes
**47 runtime sources and 43 dependencies**, including the completed parent.
All **19 TensorFlow 2.17 tests passed**. Its own GPU/cache smoke passed with
the complete 4,860 MiB synthetic cache resident, peak host RSS **6,765 MiB**
and GPU allocation **1,535 MiB**; all 40 real crop locations replayed exactly.
The 40 real image samples were cached before training and the first epoch
completed with 16 balanced updates, matching the reference's initial weights
and training-pair order. The infant test family remains excluded. See the
control report (`outputs/experiments/coa/focused-pilot-lr-control-r1/report.html`)
and startup checks (`outputs/experiments/coa/focused-pilot-lr-control-r1/startup-validation.json`).

The completed parent's
poster figure gallery (`outputs/poster/coa/pilot-r3/report.html`) contains
six source-checked figures with vector PDF/SVG and 300 dpi PNG exports, plus
underlying CSV/JSON: training curves, empirical ROC, validation errors,
detection/severity behavior, family partition and model architecture.
Seven figure-metric/integrity tests passed and all 24 plot/data/report artifacts
reproduced byte-for-byte. The
[poster and population study plan](COA_POSTER_AND_POPULATION_STUDY.md) defines
family rotations and independent population-size/diversity comparisons;
their performance curves require additional experiments.

```bash
python3 scripts/research/train_coa_pilot_control.py --config archive/research/configs/models/coa-focused-pilot-lr-control.ini --stage report
squeue -j 6902259
sacct -j 6902259 --format=JobID,State,ExitCode,Elapsed
```

## Completed 32/8 development pilot

The development experiment uses
[coa-focused-pilot.ini](../configs/models/coa-focused-pilot.ini), submitted on
**2026-10-03** as GPU job **6902162** on a Tesla V100. It completed with
exit `0:0` in **18 minutes 25 seconds**, finishing all **50 epochs**:
16 balanced, same-phantom batches per epoch, up to 800 optimizer updates and
1,600 training image presentations. The architecture, learning rate, loss,
seed, frozen locator and reviewed image-derived crop origins inherit the
passed 200-epoch diagnostic below. Its trained classifier weights are not
used to initialize this pilot.

The minimum validation BCE selected **epoch 37**. Training accuracy was
**96.875%**, versus **62.5%** validation accuracy. Validation detected one of
four positive scenarios, with no false positives; AUROC was **0.875**.
Edited-only validation accuracy was **50%**. These are development results
from two adult phantoms in one provisional anatomy family, not independent
test performance. The training learning gate passed, while generalization
and severity prediction remain unresolved.

The selected severity head predicted nearly zero local volume loss for every
validation example, with MAE **18.26 percentage points**. Its constant
training-mean comparison has MAE **12.53 points**. After epoch 37, validation
BCE deteriorated to **6.41** at epoch 50. A separate fresh lower-learning-rate
control is the next experiment; additional epochs alone are not supported by
this history. Poster assets and the population-comparison protocol are described
in [the poster study plan](COA_POSTER_AND_POPULATION_STUDY.md).

The panel contains **32 training scenarios from eight phantoms** and
**eight validation scenarios from two phantoms**, including original,
edited-negative, lower-positive and higher-positive examples for each phantom.
Training covers four provisional anatomy families; validation uses the adult
family. Siblings retain their family partition, and the reserved infant-family
test images remain excluded. **Minimum validation BCE selects the checkpoint.**
The training learning gate and validation performance are separate measurements;
severity regression remains a secondary engineering measurement.

All 40 normalized image samples occupy **4,860 MiB** within the **5,120 MiB**
RAM-cache budget. A separate production-size GPU smoke process physically
touches the complete cache, performs a nonzero-gradient model update while it
remains resident and verifies checkpoint reload before fit starts. Fit replays
all 40 locator proposals and preloads every selected image before its first
gradient. Shared native attenuation and immutable edit crops are reused;
additional whole-body sibling images are not exported.

The production TensorFlow **2.17.0** smoke passed with all 40 synthetic cache
samples resident. Including checkpoint reload, peak host RSS was **6,753 MiB**
and TensorFlow GPU allocation peaked at **1,538 MiB**, below both 12 GiB
application guards. The resident update had a nonzero gradient and retained
one train/predict trace. All 40 real development crop locations subsequently
replayed exactly; floating-point locator-box differences did not change their
native integer crop origins.

The first real training epoch completed with **16 paired updates** and all
40 samples still cached. Its classifier initialization fingerprint matches
the passed diagnostic, and checkpoint selection equals the eight-sample
validation BCE. Peak training host RSS was **7,607 MiB** and GPU allocation
**1,573 MiB**, within the configured guards. This verifies startup, rather than
completed-model performance; see the
startup validation (`outputs/experiments/coa/focused-pilot-r1/startup-validation.json`).

**Seventeen tests passed in the production TensorFlow 2.17 environment**,
including a complete 32/8 synthetic training cycle, 16 paired updates,
validation-BCE selection, unchanged initialization and locator, checkpoint
reload, parent immutability, full-cache proof and reserved-test exclusion.
The pilot freezes **43 runtime sources and 32 artifact dependencies**. See its
report (`outputs/experiments/coa/focused-pilot-r1/report.html`),
validation record (`outputs/experiments/coa/focused-pilot-r1/validation.json`),
submission record (`outputs/experiments/coa/focused-pilot-r1/submission.json`)
and GPU/cache smoke measurements (`outputs/experiments/coa/focused-pilot-r1/runtime-smoke.json`).
The configuration and earlier experiment artifacts are immutable; changes
require a separate output revision.

```bash
python3 scripts/research/train_coa_pilot.py --config archive/research/configs/models/coa-focused-pilot.ini --stage report
squeue -j 6902162
sacct -j 6902162 --format=JobID,State,ExitCode,Elapsed
```

## Completed 200-epoch budget control

The controlled revision in
[coa-focused-paired-v2.ini](../configs/models/coa-focused-paired-v2.ini) completed
on **2026-10-03** as GPU job **6902017** (`COMPLETED`, exit `0:0`, **13 minutes
39 seconds**). It finished 200 epochs and selected epoch 196. **All original
training-diagnostic gates passed.** Its separate output directory is
`outputs/experiments/coa/focused-paired-r2/`. See the
completion review (`outputs/experiments/coa/focused-paired-r2-review-r1/report.html`),
numerical review (`outputs/experiments/coa/focused-paired-r2-review-r1/review.json`)
and training report (`outputs/experiments/coa/focused-paired-r2/report.html`).

It started fresh with the same initial weights, four training siblings, crop
geometry, architecture, optimizer, learning rate and BCE/Huber loss. Only the
diagnostic budget increased from 100 to **200 epochs**: 400 paired optimizer
updates and 800 individual image presentations. This did not restore the
previous optimizer or begin the 32/8 development pilot. The per-case correct
classification, probability-gap and BCE gates remain unchanged.

| Selected training checkpoint | 100-epoch control | 200-epoch control |
| --- | ---: | ---: |
| Selected epoch | 98 | 196 |
| Accuracy at probability 0.5 | 75% | 100% |
| Binary cross-entropy | 0.65108 | 0.0004959 |
| Positive-minus-negative probability, case 260613 | 0.04146 | 0.99846 |
| Positive-minus-negative probability, case 260616 | 0.04652 | 0.99956 |
| Local-volume-loss MAE, percentage points | 18.76 | 6.634 |

The first epoch satisfying every original gate was **135** (270 paired
updates); every epoch from **147 through 200** passed. Classification stayed
stable over the final 20 epochs, with BCE between 0.000496 and 0.000709. The
four-sample training feasibility check has therefore passed; no validation or
reserved-test data were evaluated, and these results do not establish
generalization to other anatomy.

Severity MAE improves on the constant training-mean baseline of **20.923
percentage points**, but regression is less stable than classification. Its
best MAE was 3.296 at epoch 192, versus 6.634 at the selected checkpoint and
10.661 at the final epoch. Across the last 20 epochs it ranged from 3.296 to
17.985 percentage points. Checkpoint selection minimizes classification BCE;
the selected checkpoint is not optimized for severity error.

At the matched 100-epoch budget, the new run still failed the gates: BCE was
0.65246 and the mean probability gap was 0.04598, compared with 0.65630 and
0.05452 at epoch 100 of the reference. Initialization and the first 100 epochs'
pair order match. Tiny initial floating-point differences accumulate later, so
the two runs have controlled settings rather than an identical numerical
trajectory. The longer budget was needed to demonstrate reliable separation.

The version-two schema verifies the completed 100-epoch reference and rejects
other normalized setting changes. The learner differs from the frozen preceding
learner only in its experiment-module binding. The new experiment pins **39
runtime sources and 23 direct artifact dependencies**, including the reference
history, predictions, initialization and selected checkpoint. All 39 sources and
23 dependencies were verified after completion, and older runtime source pins
remain intact. **Twenty-two TensorFlow 2.17 tests passed**: twelve
paired-learning regression tests and ten new revision tests, including reference
integrity, frozen snapshots, failure gating and real synthetic training with
matching initialization, frozen locator, image-only inputs and checkpoint reload.

The job ran a new full-size batch-two GPU check in a separate process before
its training diagnostic. Its four normalized samples reused the same
approximately 486 MiB RAM cache and shared native images.

The batch-two GPU check passed on a Tesla V100, with peak
host RSS 1,932 MiB and GPU allocation 1,574 MiB. All four crop origins replayed
exactly and the initial classifier weight hash matches the completed reference.
The locator remained frozen, selected-checkpoint reload reproduced predictions
exactly, and training retained one train/predict trace. The completed diagnostic
peaked at approximately 3,309 MiB host RSS and 1,537 MiB TensorFlow GPU allocation.

These results motivated the fresh **32-training/8-validation development
pilot** above. Its 50-epoch budget provides 800 paired updates, compared with
160 updates under the preceding 10-epoch setting and 270 updates before the
diagnostic first passed. The new 5,120 MiB cache can retain the entire
4,860 MiB panel; the preceding 2,048 MiB budget would repeatedly evict and
reload images. Production memory checks enforce the existing 12 GiB host and
GPU limits before fit.

```bash
python3 scripts/research/train_coa_paired_v2.py --config archive/research/configs/models/coa-focused-paired-v2.ini --validate-only
python3 scripts/research/train_coa_paired_v2.py --config archive/research/configs/models/coa-focused-paired-v2.ini --stage report
sacct -j 6902017 --format=JobID,State,ExitCode,Elapsed
```

## Completed 100-epoch control and preceding findings

The paired optimization control in
[coa-focused-paired.ini](../configs/models/coa-focused-paired.ini) completed as
GPU job **6899862** on **2026-10-03** (`COMPLETED`, exit `0:0`, **9 minutes
44 seconds**). It finished 100 epochs and selected epoch 98. **The training
diagnostic improved, but its learning gate still failed.** See the separate
completion review (`outputs/experiments/coa/focused-paired-r1-review-r1/report.html`)
and numerical review (`outputs/experiments/coa/focused-paired-r1-review-r1/review.json`).

The control reuses the same four training images, focused architecture, 0.6 mm
crops, initialization, learning rate and BCE/Huber loss. A mild/severe sibling
pair is evaluated at the same weights, their losses are averaged, then their
combined gradient is clipped and applied once. Its 100 epochs contain 200
optimizer updates and 400 image presentations, compared with 160 updates and
160 presentations in the preceding 40-epoch run. The final gain therefore cannot
be attributed entirely to pairing.

| Selected training checkpoint | Previous single-image updates | Paired updates |
| --- | ---: | ---: |
| Selected epoch | 38 | 98 |
| Accuracy at probability 0.5 | 50% | 75% |
| Binary cross-entropy | 0.70968 | 0.65108 |
| Positive-minus-negative probability, case 260613 | 0.000166 | 0.04146 |
| Positive-minus-negative probability, case 260616 | 0.000097 | 0.04652 |
| Local-volume-loss MAE, percentage points | 20.92 | 18.76 |

At the paired checkpoint, both severe siblings are positive, but the mild
260616 sibling remains a false positive (probability 0.50844). Both per-case
probability gaps remain well below the required 0.30, and BCE remains above the
0.30 limit. Severity MAE now modestly improves on the constant training-mean
baseline of 20.923 percentage points, but severity estimates remain compressed:
case 260613 changes from 8.38% to 52.91% true volume loss while predictions change
only from 34.30% to 38.05%. These four examples are all training cases; no
validation or reserved-test data were evaluated.

At matched exposure, epoch 40 of the paired run has the same 160 image
presentations as the prior run, but only 80 optimizer updates. Its BCE is
0.69260 and accuracy reaches 100%, yet the mean probability gap is only
0.000549. Accuracy alone would therefore give a misleading pass. The gap keeps
growing near the end of training, from 0.0060 at epoch 80 to 0.0545 at epoch 100,
while shared prediction shifts make threshold decisions unstable: all four
samples fall below 0.5 at epoch 99 and above it at epoch 100. The selected epoch
98 balances this growing response with lower BCE.

These findings motivated the active **200-epoch** control above, preserving
architecture, initialization, data, paired updates and optimization settings.
It uses a new parser revision and output directory because the preceding pinned
parser caps its diagnostic at 100 epochs. The existing per-case probability and
BCE gates are retained. Only after the new diagnostic passes should the
32-training/8-validation pilot start from fresh weights; further cohort
expansion is premature.

The paired control pins 35 runtime sources and 21 dependencies, including the
prior frozen diagnostic, and refuses to train if its starting weights differ
from that reference. **Twelve new TensorFlow 2.17 tests passed**, including real
paired training, cancellation of identical background gradients, safe checkpoint
reload and independent inference for a single scan. The new batch-two GPU smoke
check passed before cohort gradients, with peak host RSS 1,937 MiB and peak
TensorFlow GPU allocation 1,569 MiB. All four crop origins replayed exactly, and
the initial classifier weight hash matches the reference. The locator remained
unchanged, selected-checkpoint reload reproduced predictions exactly, and
training retained one train/predict trace. Follow the
paired classifier report (`outputs/experiments/coa/focused-paired-r1/report.html`).

```bash
python3 scripts/research/train_coa_paired.py --config archive/research/configs/models/coa-focused-paired.ini --validate-only
python3 scripts/research/train_coa_paired.py --config archive/research/configs/models/coa-focused-paired.ini --stage report
# A later pilot requires the paired training diagnostic to pass:
sbatch scripts/research/slurm/fakect_coa_paired_gpu.sh archive/research/configs/models/coa-focused-paired.ini fit
```

The preceding first focused diagnostic used the classifier in
[coa-focused-classifier.ini](../configs/models/coa-focused-classifier.ini), with
output `outputs/experiments/coa/focused-classifier-r1/`. It reuses the reviewed
epoch-99 locator and two native-resolution 0.6 mm blocks. The new classifier
retains a stride-four feature grid (2.4 mm spacing), an explicit attenuation
amplitude path, local evidence/attention, and ordered 6×6×6 feature bins. Its
inputs are detail attenuation and acquisition validity. There is no overview or
crop-coordinate classifier branch in this diagnostic, and class labels remain
loss targets.

Submitted on **2026-10-03** as GPU job **6899856** after **30 targeted tests
passed in TensorFlow 2.17**, including real diagnostic training, training-only
sample reads, frozen-locator replay and checkpoint reload parity. The experiment
pins 30 runtime source files and 14 inherited dependencies. Its production-size
GPU check passed with eight gradient updates, one train/predict trace, unchanged
RSS after warm-up, safe checkpoint reload, peak host RSS 1,869 MiB and peak
TensorFlow GPU allocation 724 MiB. All four cohort crop origins replayed exactly.
The job completed normally in 3 minutes 49 seconds; its 40-epoch diagnostic
**failed**, selecting epoch 38 with BCE 0.70968, accuracy 50%, and mild/severe
probability gaps of 0.000166 and 0.000097. Native and encoded differences survive
through the local evidence map and pooled features. Early per-sample gradients
are large and opposing sibling updates swing predictions, motivating the paired
optimization control above. That mechanism remains a hypothesis to test.
Follow the
focused classifier report (`outputs/experiments/coa/focused-classifier-r1/report.html`).

The initial job runs a production-size synthetic GPU check in a separate
process, followed by a 40-epoch training diagnostic on four matched edited
siblings from training cases 260613 and 260616. Each case contributes its mild
negative and strongest positive, with identical companion parameters and
image-derived crop origins. A bounded RAM cache stores approximately 486 MiB of
normalized inputs without exporting additional datasets. The report tracks
input and layer differences before and after training, prediction separation,
severity error, resource use and safe checkpoint reload.

Every diagnostic case must classify both siblings correctly at probability 0.5,
separate their probabilities by at least 0.30, and achieve binary cross-entropy
no greater than 0.30. This demonstrates training feasibility; it does not measure
generalization or establish which architectural change explains the earlier
failure. The later 32/8 development pilot starts fresh weights, requires a passed
diagnostic, and selects its checkpoint using validation BCE. Test phantoms remain
excluded throughout both stages.

```bash
python3 scripts/research/train_coa_focused.py --config archive/research/configs/models/coa-focused-classifier.ini --validate-only
python3 scripts/research/train_coa_focused.py --config archive/research/configs/models/coa-focused-classifier.ini --stage plan
sbatch scripts/research/slurm/fakect_coa_focused_gpu.sh archive/research/configs/models/coa-focused-classifier.ini
python3 scripts/research/train_coa_focused.py --config archive/research/configs/models/coa-focused-classifier.ini --stage report
# After a successful diagnostic:
sbatch scripts/research/slurm/fakect_coa_focused_gpu.sh archive/research/configs/models/coa-focused-classifier.ini fit
```

The classifier pilot was submitted on **2026-10-03** as GPU job **6899715**
after **27 targeted tests passed in TensorFlow 2.17**. Its frozen plan contains
25 runtime source files and 14 inherited artifact dependencies. The job completed
normally (`COMPLETED`, exit `0:0`) in **38 minutes 34 seconds**, with all 10 epochs
and 320 optimizer updates. **The classifier learning gate failed.** See the
classifier report (`outputs/experiments/coa/classifier-pilot-r1/report.html`)
and completion review (`outputs/experiments/coa/classifier-pilot-r1/completion-review.json`).

| Selected epoch 10 | Training (32 scenarios) | Validation (8 scenarios) |
| --- | ---: | ---: |
| Accuracy / balanced accuracy | 50% | 50% |
| Sensitivity / specificity | 100% / 0% | 100% / 0% |
| Binary cross-entropy | 0.69755 | 0.69356 |
| Local-volume-loss MAE, percentage points | 17.25 | 13.27 |
| Constant training-mean estimate MAE, percentage points | 15.93 | 12.53 |

Every sample is classified positive at the fixed 0.5 probability threshold.
Both training and validation accuracy stayed at 50% throughout all ten epochs.
Binary cross-entropy is near the balanced constant-probability baseline
(`ln(2) = 0.69315`); its improvement from initialization mainly reflects a
less-biased prediction, without useful class separation. Severity estimates
underperform a constant computed from training labels alone.

Predictions differ by phantom, but barely respond to the imposed disease within
each phantom: the mean strongest-sibling versus original probability change is
0.00000337 in training and 0.000000231 in validation. Within-case positive versus
negative ordering is correct at tiny separations, so the signal is not literally
zero. Validation AUROC 0.75 should not be treated as practical success: the eight
samples come from only two phantoms in one family and the tiny score differences
do not yield useful decisions at the fixed threshold. No validation samples lie
in the 18–22% near-threshold band; four training samples do.

The run's engineering checks passed: all 40 image-derived crops replayed exactly,
the locator and overview encoder remained unchanged, selected-checkpoint reload
reproduced predictions exactly, and input/source/dependency/result hashes still
match. Peak sampled host RSS was 2,471 MiB and TensorFlow GPU allocation was
594 MiB, with one training and one prediction trace. Test phantoms remain unread.

The input-signal audit (`outputs/experiments/coa/classifier-pilot-r1/input-signal-review.json`)
confirms that attenuation changes are present at every removed voxel in all 24
edited training siblings. Exact detail-reader comparisons for training phantoms
260613 and 260616 confirm that the overlays survive normalization without
clipping or changed validity/crop origins. Differences occupy only 0.0344% and
0.2315% of their upper detail blocks, respectively; lower blocks are unchanged.

The next step is a training-only layer-sensitivity diagnostic and tiny
same-phantom overfit experiment before increasing the cohort. Isolate the
fine-image path from the coarse features and identify where sibling sensitivity
is lost. The current fine encoder downsamples by 16 (9.6 mm feature-grid
spacing from 0.6 mm inputs) and globally pools each large block; signal dilution
and shallow channel normalization are hypotheses to test, not established
causes. A revised encoder should first demonstrate reliable separation of edited
negatives and strong positives in a small controlled training panel. Retain the
successful locator, fixed family split and reserved tests during this work.

This run uses
[coa-classifier-pilot.ini](../configs/models/coa-classifier-pilot.ini) to train
only the classifier. It inherits the reviewed epoch-99 locator, frozen coarse
encoder, 32 training / 8 validation scenarios, image scaling, and padded detail
geometry. Its new output is `outputs/experiments/coa/classifier-pilot-r1/`.
Parent sources, checkpoint and numerical review are verified and pinned. Before
any classifier gradients, all image-derived proposals must reproduce the
reviewed native crop origins exactly. Test payloads are rejected before readers
or TensorFlow initialization.

Classifier initialization and sample shuffling use a fresh recorded seed. This
is a new classifier stage, not restoration of the interrupted optimizer or RNG.
After each epoch it records both training and validation predictions, checks
that coarse encoder weights remain identical, and selects the checkpoint by
validation binary cross-entropy. Reloaded checkpoint predictions must match the
selected epoch. Numerical results are written before the HTML report, and any
rendering failure is recorded separately.

The portable report includes learning curves, AUROC and confusion counts,
predicted probability versus achieved local-volume loss, regression error in
percentage points, all scenario predictions, and separate original/edited
negative/threshold-near groups. These remain development measurements on two
validation phantoms in one family; no reserved test inference or clinical
severity validation is performed.

```bash
python3 scripts/research/train_coa_classifier.py --config archive/research/configs/models/coa-classifier-pilot.ini --validate-only
python3 scripts/research/train_coa_classifier.py --config archive/research/configs/models/coa-classifier-pilot.ini --stage plan
sbatch scripts/research/slurm/fakect_coa_classifier_gpu.sh archive/research/configs/models/coa-classifier-pilot.ini
python3 scripts/research/train_coa_classifier.py --config archive/research/configs/models/coa-classifier-pilot.ini --stage report
```

The fit stage refuses an already-started output directory. Changed classifier
settings require a new revision, while the same reviewed locator and shared
images may be reused. Larger virtual-cohort comparisons follow a successful
small-panel learning check; the classifier architecture and family partitions
should then remain fixed across cohort-size comparisons.

The selected revision-three locator is **ready for the CoA classifier** after
checkpoint recovery evaluation on 2026-10-01. See the
recovered report (`outputs/experiments/coa/multiscale-pilot-r3-review-r1/report.html`).
The original job **6872594** completed all 120 locator epochs, then failed while
rendering list-valued detail-block coordinates. It did not train a CoA classifier.
Evaluation-only job **6875975** recovered the saved checkpoint without retraining
or changing the frozen run; it completed successfully in 2 minutes 21 seconds.

| Development measurement | Revision two | Revision three, selected epoch 99 |
| --- | ---: | ---: |
| Mean training center error | 24.77 mm | 2.84 mm |
| Mean validation center error | 29.63 mm | 8.36 mm |
| Scenarios passing original coverage gates | 29/40 | 40/40 |

The recovered detail blocks contain **100% of before/after target anatomy and
all added/removed voxels** in every selected development scenario. All 60 fixed
translation probes retain native anatomy coverage. Validation mean relative
translation-response errors are 0.0145, 0.0092 and 0.0104 for k,j,i, comfortably
below the predefined 0.5 threshold. Matched real-image center error is 8.28 mm
versus 604.35 mm with blank attenuation, retaining validity; the image-dependence
screen passes. These are development screens used alongside checkpoint
selection, not reserved-test or clinical performance claims.

The original epoch-99 checkpoint reproduces recorded geometry within 0.01 mm.
Keep it: the final epoch improved training error but worsened validation error
to 12.83 mm. Training peaked near 2 GiB host memory with one train/predict trace.
The classifier-only continuation described above uses this verified locator
with its own immutable metadata and explicit deterministic classifier shuffle
seed. It does not restore the interrupted augmentation RNG state. Train the
existing binary CoA and continuous achieved local-volume-loss heads only in a
new revision when addressing the failed learning check above. The fixed-model
small-versus-expanded virtual-cohort comparison follows a successful learning
diagnostic. All reserved test families remain untouched.

Recovery uses the standalone `fakect_multiscale_preview.py` renderer; five actual
rendering tests cover list/array inputs and off-plane/out-of-FOV blocks. The
evaluation writes numerical results before optional rendering so a figure
failure cannot discard the completed audit. Original v3 source and failure
artifacts remain pinned. For a new recovery review directory:

```bash
sbatch scripts/research/slurm/fakect_multiscale_review_gpu.sh archive/research/configs/models/coa-multiscale-pilot-v3.ini outputs/experiments/coa/NEW-REVIEW-DIRECTORY
```

Revision three is configured by
[coa-multiscale-pilot-v3.ini](../configs/models/coa-multiscale-pilot-v3.ini).
It addresses the completed r2 run: native coverage passed29/40 development
scenarios, so the CoA classifier did not train. A fixed-center geometry audit
increased coverage to32/40 with9.6mm transverse padding;260611 and260616 still
needed better localization. The selected r2 checkpoint was epoch49, with
mean center errors24.8mm training and29.6mm validation. Peak host RSS was
1,844MiB and train/predict functions retained one trace each.

The r3 locator uses a fully convolutional spatial heatmap with stride-four
features. The center no longer comes from a flattened dense regression head.
Only the extent initialization uses original training-mask means. Training
views translate, scale and truncate the overview while consistently transforming
the supervised box and retaining its full extent plus context. Views are
generated on demand and do not duplicate native images.

`margin_mm_ijk=9.6,9.6,0` adds a symmetric physical margin to the existing detail
shape. Outward rounding maintains voxel-grid alignment and the padded input
budget is checked explicitly. At0.6mm this yields216×192×192 blocks. The report
shows actual blocks in pink alongside cyan predictions and yellow supervision.

Checkpoint selection uses equal weights for original validation loss and the
mean loss under six fixed translations. After selection, each validation axis
must have mean relative translation-response error≤0.5, with no individual
probe>1.0. Real-image mean center error must be≤75% of the matched blank-image
error retaining the acquisition-validity channel. These are predefined
engineering screens on two development cases, not independent test evidence.
Transformed predictions are inverse-mapped before native-mask coverage QA, and
all validation probes must pass the before/after anatomy coverage floors too.
Original coverage gates are unchanged; both coverage and robustness must pass
before the CoA classifier trains.

The GPU job first tests synthetic batches at the real overview AND padded
detail dimensions, including gradient updates, stable traces and checkpoint
reload. It then starts the cohort run in a fresh process. Existing12GiB
application memory caps,16GiB host request and four-hour limit remain.

```bash
python3 scripts/research/train_multiscale_coa_v3.py --config archive/research/configs/models/coa-multiscale-pilot-v3.ini --validate-only
python3 scripts/research/train_multiscale_coa_v3.py --config archive/research/configs/models/coa-multiscale-pilot-v3.ini --stage plan
sbatch scripts/research/slurm/fakect_multiscale_v3_gpu.sh archive/research/configs/models/coa-multiscale-pilot-v3.ini
```

Status and measurements appear in
the r3 report (`outputs/experiments/coa/multiscale-pilot-r3/report.html`).
Submitted as job **6872594** on2026-10-01 UTC after **30 targeted tests passed
in TensorFlow2.17**, including staged training/checkpoint tests and independent
coverage/robustness rejection checks. The complete target plus context fits all
60 fixed translation probes across the development originals. This confirms
probe eligibility, not trained prediction quality.
The production-size synthetic GPU check passed on the V100:32 locator and32
classifier updates, one trace per function, no measured RSS growth after
warm-up, peak host RSS1,695MiB and peak TensorFlow GPU allocation488MiB. Both
detail branches used216×192×192 inputs. All40 development samples were prepared
and all120 locator epochs completed. Recovered coverage results are recorded
above; the separate classifier continuation and its failed learning check are
documented at the start of this file.
Use a new output directory for each changed experiment. Source snapshots and
data for earlier revisions remain pinned.

Revision two uses [coa-multiscale-pilot-v2.ini](../configs/models/coa-multiscale-pilot-v2.ini)
and writes `outputs/experiments/coa/multiscale-pilot-r2/`. The original revision
and its source snapshots remain unchanged.

Revision two was submitted as **6870408**. Eighteen targeted checks passed in
the actual TensorFlow 2.17 container. Its separate production-shape GPU warm-up
also passed: 32 synthetic overview batches retained one training/inference
trace each, resident memory did not grow after step eight, and the process
peaked at 1,596 MiB host RSS. This checks the runtime fix; actual localization
and classification performance remain separate results in the
revision-two report (`outputs/experiments/coa/multiscale-pilot-r2/report.html`).

Run 6866879 completed successfully as a scheduler job but failed the anatomy
localization gate. The CoA classifier was therefore never trained. Predicted
boxes had zero overlap and the patches covered none of the aorta in all 40
development scenarios. Mean center error was 308.8 mm in training and 386.8 mm
in validation. A constant box calculated from the eight training originals
gave 48.0 and 56.7 mm errors respectively: useful as an initialization/control,
but insufficient for image-based localization. Correctly placed patches covered
all targets and edits, so the patch dimensions are retained.

The longitudinal output had saturated near the top of the scan. Disjoint boxes
made the IoU term constant, while the remaining coordinate gradient was weak.
Revision two initializes from the **training-only** mean box and uses stable
supervision on raw box parameters plus stronger coordinate regression. The
report compares its learned result with that prior. Neither validation cases
nor held-out labels contribute to the prior.

Preparation peaked at 324 MiB host RSS, but training reached approximately
46.6 GiB, confirmed independently by Slurm. The installed Keras 3.4.1 batch
wrapper and repeated retracing warnings point to retained training graphs as
the likely cause; a causal isolation experiment was not performed. Revision
two uses fixed TensorSpec GradientTape functions instead of `train_on_batch`,
following the [TensorFlow tracing guidance](https://www.tensorflow.org/guide/function).
Tests in the actual TensorFlow 2.17 container verify one graph per function
across changing input batches. Runtime telemetry checks this during the run.

The new scheduler entry first runs 32 synthetic batches at the real overview
dimensions on the GPU, checks memory growth after warm-up, and tests checkpoint
reload. It then starts the real pilot in a fresh process. Host memory is sampled
around batches with a 12 GiB stopping threshold; this guard cannot prevent a
single transient allocation between samples. The request remains 16 GiB host
memory. GPU allocation remains capped at 12 GiB.

Anatomy coverage of at least 90% is now required for **every** scenario, including
originals. The first revision incorrectly passed empty-change controls even when
they missed the aorta. Full erosion coverage and the no-cancellation proof remain
mandatory for modified cases. Projected boxes outside the displayed slice are
dashed in revised previews.

```bash
python3 scripts/research/train_multiscale_coa_v2.py --config archive/research/configs/models/coa-multiscale-pilot-v2.ini --validate-only
python3 scripts/research/train_multiscale_coa_v2.py --config archive/research/configs/models/coa-multiscale-pilot-v2.ini --stage plan
sbatch scripts/research/slurm/fakect_multiscale_v2_gpu.sh archive/research/configs/models/coa-multiscale-pilot-v2.ini
```

Use a new output revision for changed inputs. The descriptions below record the
shared workflow and original run; their revision-one commands remain available
for inspecting the preserved first experiment.

The first implementation combines a streamed whole-body overview with two
fine image patches. It tests whether this pipeline can localize the relevant
anatomy and learn the existing provisional CoA labels across several phantoms.
It does not yet perform the final virtual-population comparison.

The first frozen run is `multiscale-pilot-r1`, submitted to Wahab as job
**6866879** on 2026-09-30. Its runtime status and results belong to the report
and scheduler logs, rather than this static document. Before submission,
32 tests passed, including miniature TensorFlow training and checkpoint reload.

The [input file](../configs/models/coa-multiscale-pilot.ini) is independent of
the morphology recipes. It references the frozen 192-scenario context cohort
and the separate 20% local-volume-loss annotation revision. The user's decision
to proceed is recorded as `annotation_use = accepted_engineering_proxy` in
the experiment contract; the original draft labels are preserved.

## Stages and commands

Run these commands from the `fakect.26.09.16` worktree:

```bash
python3 scripts/research/train_multiscale_coa.py --config archive/research/configs/models/coa-multiscale-pilot.ini --validate-only
python3 scripts/research/train_multiscale_coa.py --config archive/research/configs/models/coa-multiscale-pilot.ini --stage plan
mkdir -p logs
sbatch scripts/research/slurm/fakect_multiscale_gpu.sh archive/research/configs/models/coa-multiscale-pilot.ini
```

`--validate-only` checks metadata, annotation joins and scores, provenance,
family splits and input allocation bounds. It does not claim that optimization
will succeed. `--stage plan` freezes the INI and source files into a new output
revision; it refuses to overwrite an existing experiment.

The GPU job runs `--stage all`: prepare overviews and review images, train the
locator, audit predicted patch coverage, and train the classifier only if the
coverage gate passes. Individual `prepare`, `fit` and `report` stages are also
available. Re-running `prepare` can reuse a complete verified preparation;
`fit` never implicitly resumes or overwrites a model directory.

The report is
outputs/experiments/coa/multiscale-pilot-r1/report.html (`outputs/experiments/coa/multiscale-pilot-r1/report.html`).
It includes original anatomy supervision, predicted bounding boxes, per-sample
coverage, training histories, resource measurements and the final gate result.
Regenerate it while a job runs with:

```bash
python3 scripts/research/train_multiscale_coa.py --config archive/research/configs/models/coa-multiscale-pilot.ini --stage report
```

## Data and partitions

The family-preserving 8/2/2 case split reserves the infant family for future
testing. Four developmental families supply eight training phantoms; the two
adult references supply validation. Within each development case the pilot
selects four scenarios: original, lowest below-threshold edited negative,
lowest positive and highest positive. All three edited selections use the
same lowest dilation setting and companion parameters. Thus there are
**32 training and 8 validation scenarios, balanced by class**. Test images and
targets are never loaded or predicted by this runner.

The binary target is achieved CoA local target-volume loss of at least 20%.
The continuous head predicts the same loss fraction. This is not minimum lumen
area or clinical severity. Near-threshold flags remain available. Below-threshold
edited negatives still have mild narrowing, arch erosion and dilation.

## Shared storage and physical sampling

`fakect_multiscale_volume.build_overview` streams native attenuation into a
maximum 160×96×96 padded grid. Integer box averaging is an explicit antialias
filter. Partially occupied boundary cells are averaged over their actual
native support, with a separate fractional validity channel. Physical spacing
and padded extent are recorded; anatomy is not stretched to fill the array.

One checksummed base cache is shared across siblings. The loader streams only
the crop difference into the affected pooled cells, then clips/scales the
composed attenuation for the network. Learned features are not cached during
training. Native images remain memory-mapped, and source reads are bounded by
four million voxels. Whole-body sibling images are not exported.

A production check on the adult 260602 source (1.046 billion voxels) used about
153 MiB peak process RSS, created its initial overview in 49.95 seconds, and
processed a cached sibling in 0.115 seconds. Seventy-seven independent native
box means agreed within 7.22×10⁻⁹ cm⁻¹. These are one-case CPU loader measurements,
not GPU training or whole-cohort performance claims.

## Models and leakage checks

The locator is a compact 3D encoder followed by a spatial bounding-box head.
It receives attenuation and validity, and learns thoracic-aorta bounds from
original training masks. Validation bounding-box loss chooses its checkpoint.
After freezing it, every selected sibling receives its own image-derived
prediction.

Two patches are centered at 25% and 75% of the predicted longitudinal extent.
Each is 216×160×160 at 0.6 mm, covering 129.6×96×96 mm. The classifier uses a shared
fine 3D encoder, mean/max features, and spatial coarse features sampled at the
actual fine-patch coordinates. Its outputs are a binary logit and a continuous
loss fraction. This is the classification version of the platform; a U-Net
segmentation decoder remains a later optional head.

The network uses logits with binary cross-entropy, following the
[TensorFlow API contract](https://www.tensorflow.org/api_docs/python/tf/keras/losses/BinaryCrossentropy),
and gradient clipping. Normalization does not depend on other batch members.
Layer activations are recomputed for each input, and the coarse encoder is
frozen during classifier training.

Mask-derived boxes are supervision and review overlays only. They never move
the predicted input patches. All coordinate transformations use voxel faces
and the padded overview extent, including anisotropic grids.

## Gates before larger studies

1. **Condition coverage:** accept only the supported one-dilation/two-erosion
   recipe and verify that net additions/removals equal the sums of accepted
   step counts. This proves that canceled edits are not hidden. Require 100%
   coverage of native net-removed voxels and at least the configured fraction
   of all changes. Under that proof, full erosion coverage includes CoA
   removals. Missing coverage stops classification and generates a report;
   it does not substitute a known target box. Empty original changes are N/A.
2. **Multi-case learning:** the selected classifier must improve training BCE
   by at least 20% from initialization, distinguish edited negatives from
   positives with both sensitivity and specificity ≥0.75, and have a mean
   positive-minus-negative probability gap ≥0.15. This is an engineering
   readiness check, not evidence of generalization. Validation metrics are
   reported separately.
3. **Resources:** cap TensorFlow GPU allocation at 12 GiB by default and measure
   observed GPU/host peaks. The scheduler job is bounded to four hours, one GPU,
   four CPUs and 16 GiB host memory. No automatic retry or escalation launches
   additional training.

Next revisions need independently varied CoA-off companion controls, achieved
centerline-normal area/length measurements, architecture efficiency comparisons,
and a fixed-model small-versus-expanded population study. Family cross-validation
and segmentation/localization heads remain planned. See the
[platform plan](VIRTUAL_POPULATION_PLATFORM_PLAN.md) for the three workstreams.
