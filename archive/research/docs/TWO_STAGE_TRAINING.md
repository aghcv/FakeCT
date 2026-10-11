# Training the segmentation and reference-shape models

For the current labeled-phantom work, use the newer
[atlas extraction and mask-only workflow](PHANTOM_MASK_WORKFLOW.md). It replaces
learned segmentation with anatomical labels upstream of Model 2. The original
two-stage experiment below remains available for comparison. Its current HTML
report includes 3D overlays of all 32 held-out scenarios with independent label
visibility and 15%, 45%, or 80% opacity. Share the HTML together with its entire
`report-3d/` folder. The original single-file report is retained as `report-2d.html`.

The [two-stage INI](../configs/models/coa-two-stage.ini) controls a synthetic
engineering pilot. Model 1 maps attenuation to the observed aorta mask. Model 2
maps that binary mask to an estimated unedited reference. Comparing the input
and reconstructed masks produces possible enlargement and narrowing regions.

The source targets are the reviewed ROI-selected aorta and branch junctions,
including descendants outside the selection ROI. They are **not yet validated
complete thoracic-aorta labels or a whole-scan localization dataset**. The pilot
therefore operates on supplied crops with the recorded anatomical extent.

## Data preparation

All twelve inputs in [the project](../../../configs/case-studies/coa/project.json)
are frozen. Extract each case's sixteen accepted scenarios with the existing
preparation command; do not re-freeze or edit those inputs:

```bash
python3 scripts/prepare_cohort.py --config configs/case-studies/coa/phantoms/260602.ini --stage prepare
```

The supplied `scripts/research/evaluation/coa_prepare_pairs_array.sh` runs all twelve on
Slurm. Preparation refuses to overwrite complete or partial output directories.
Native arrays, models, annotations and reports live under ignored output/data
directories, so they do not enter Git. Back up these artifacts independently.

Once all cases are prepared, build the shared manifest and change-mask companions:

```bash
python3 scripts/research/train_two_stage.py --config archive/research/configs/models/coa-two-stage.ini --stage prepare
python3 scripts/research/train_two_stage.py --config archive/research/configs/models/coa-two-stage.ini --validate-only
python3 scripts/research/train_two_stage.py --config archive/research/configs/models/coa-two-stage.ini --stage plan
```

The manifest pins the source `before` mask once per phantom, and each variant's
attenuation image and `after` mask. The companions store net changes:

```text
added   = after AND NOT before
removed = before AND NOT after
```

Before/after images and masks remain in the immutable native preparation; the
adapter references them by path and checksum. It never reruns morphology.
Changes canceled by later edits do not appear in net masks. Original controls
have empty added/removed targets. Reusing a prepared companion requires the
same project and export settings; changing either needs a new data revision.

## Models and physical grid

Both backends use a compact 3D U-Net with paired 3x3x3 convolution blocks,
stride-2 convolutional downsampling, skip connections, upsampling and a single
sigmoid foreground output. XLA JIT is disabled: the initial deterministic GPU
test found an unsupported max-pooling gradient in TensorFlow 2.17, so learned
strided convolutions provide the downsampling instead.
This follows the volumetric encoder/decoder pattern of
[3D U-Net](https://arxiv.org/abs/1606.06650), implemented using TensorFlow
[Conv3D layers](https://www.tensorflow.org/api_docs/python/tf/keras/layers/Conv3D).
The initial settings use two downsampling levels and 4/8/16 feature channels, with
64-cubed patches and a batch size of one. It is an engineering baseline whose
accuracy and context adequacy must be measured.

The training loop traces one tensor-signature function around Keras `train_step`.
This avoids TensorFlow 2.17's per-call Python-generator retracing in
`train_on_batch`, which caused unbounded host-memory growth in the first full
attempt. Epoch histories record graph counts and training/validation durations;
the loss, Adam optimizer and model architecture are unchanged by this fix.

The common grid is 0.6 mm isotropic. Native arrays are unchanged: model-time
image interpolation is linear, masks use the same nearest-neighbor coordinates,
and added/removed masks are re-derived on that grid. The report records native
and model-grid change counts. Initial metrics are model-grid measurements.
Patch physical coverage is 38.4 mm per axis; a larger context may be needed for
reference-shape estimation across a complete arch.

Model 1 uses fixed attenuation clipping and BCE plus Dice. Model 2 receives only
the observed mask and learns baseline occupancy, emphasizing changed voxels in
its BCE term. Unedited controls and an identity comparator measure inappropriate
correction of normal variation. The original phantoms are reference-normal under
the study assumption, not clinically certified normal anatomy.

At inference, derive changes from a single thresholded reference prediction:

```text
estimated_added   = input_mask AND NOT estimated_reference
estimated_removed = estimated_reference AND NOT input_mask
```

This guarantees exact consistency between the reconstructed mask and the two
change maps. Removed voxels lie outside the surviving input aorta and remain
inside the reconstruction field of view. A plausible reference is not a unique
recovery of the patient's historical anatomy.

## Training and testing

Both tasks share an immutable experiment lock and whole-family assignment.
The first run uses all 192 scenarios, partitioned into 128 training, 32
validation and 32 test samples. With seed 42, the eight training cases are the
1-, 5-, 10- and 15-year groups; adults 260602/260611 provide validation, and
infants 260612/260617 are reserved for testing. This is age/size extrapolation.
Known duplicates remain together. These six groups are explicitly provisional;
the existing verified-registry experiment requirements are unchanged.

Each epoch samples four patches per training variant: change-centered,
foreground and uniform context, with controls retained. The supplied INI uses
no batch cap, so every training variant is visited. Validation covers every
voxel in nonoverlapping tiles and selects the lowest validation-loss checkpoint.
Test inference uses overlapping tiles and averages their probabilities.

Train either task independently in the TensorFlow environment:

```bash
python3 scripts/research/train_two_stage.py --config archive/research/configs/models/coa-two-stage.ini --stage fit-segmentation
python3 scripts/research/train_two_stage.py --config archive/research/configs/models/coa-two-stage.ini --stage fit-reconstruction
python3 scripts/research/train_two_stage.py --config archive/research/configs/models/coa-two-stage.ini --stage test
python3 scripts/research/train_two_stage.py --config archive/research/configs/models/coa-two-stage.ini --stage report
```

On Wahab, submit the combined sequence after native extraction:

```bash
mkdir -p logs
sbatch scripts/research/slurm/fakect_two_stage_gpu.sh archive/research/configs/models/coa-two-stage.ini
```

The wrapper requests one GPU, four CPUs, 16 GB RAM and two hours. It checks that
the GPU is visible in `tensorflow-gpu/2.17`, using `crun -p ~/envs/fakect`.
Its `all` stage prepares/reuses the companion, plans/reuses an identical lock,
fits both models, tests and writes the HTML report. Completed models can be
reused only with matching settings, partitions and checksums; partial output is
preserved for inspection. Changed model code or inputs require a new run revision.

Report these results separately:

1. Model 1: predicted aorta versus the exact observed mask.
2. Model 2 with exact input masks: reference reconstruction and true edit
   localization, including the identity baseline.
3. Full pipeline: Model 1 output passed into Model 2. Derived differences can
   reflect segmentation mistakes as well as imposed morphology. Report both
   true-edit agreement and restoration-residual accuracy, with their overlap
   with Model 1 errors.

Metrics include Dice, precision, recall and false-positive volume, with
case/family summaries and controls shown separately. Five epochs are a bounded
first run, not an assertion of convergence. No clinical thresholds, clinical
generalization or intermediate/severe classification are inferred from it.

The [recorded longer-term plan](integration/2026-09-28/pathology-voxel-pilot.md)
retains per-condition history, centerline coordinates and geometric severity
annotations for subsequent development.

## Focused CT change localization: revised Model 1

The clarified pipeline on 2026-10-05 is CT -> existing thoracic-aorta locator ->
Model 1 predicts baseline-relative net enlargement and narrowing -> Model 2
receives focused CT plus predicted abnormality maps and classifies intermediate
versus severe anomalies. Model 1 does not need to distinguish CoA from arch
changes. This naming describes the new pipeline; the earlier segmentation and
mask-reconstruction models documented above remain separate historical pilots.

The additive `archive/research/configs/models/aorta-focused-change.ini` validates Model 1 first.
It reuses the five frozen development-family locators and exact predicted focus
crops. Every fresh three-level 3D U-Net receives normalized sibling attenuation
and acquisition validity only. Targets are mutually exclusive unchanged, net
added and net removed voxels from original/final aorta masks on the same grid.
All edits contribute to these targets. Baseline and true change maps are never
network inputs. One unchanged baseline per case measures false abnormality.
Controls use deterministic paired focus coordinates from the first sibling;
this is declared fixed framing, not a fresh baseline localization experiment.

Each fold fits 56 samples (48 edited siblings and eight controls) and evaluates
14 held-out samples (12 siblings and two controls), with the related anatomy
family wholly excluded from weight fitting. Infants 260612/260617 stay closed.
The first trial declares 64 epochs, four patches per training sample and a
96-cubed patch (57.6 mm on the inherited 0.6-mm grid). Class-weighted CE and
foreground Dice emphasize sparse changes; training samples changed voxels,
while inference traverses every frozen focus tile independently of truth.
Tile/proposal probabilities are averaged on a common grid, scoring each voxel
once even when focus crops overlap. Dense probability maps are saved for later
classifier experiments without copying the whole-body CT.

```bash
python3 scripts/research/train_focused_change.py --config archive/research/configs/models/aorta-focused-change.ini --validate-only
python3 scripts/research/train_focused_change.py --config archive/research/configs/models/aorta-focused-change.ini --stage plan
python3 scripts/research/train_focused_change.py --config archive/research/configs/models/aorta-focused-change.ini --stage list --ids-only
python3 scripts/research/train_focused_change.py --config archive/research/configs/models/aorta-focused-change.ini --stage report
```

GPU smoke must pass before fitting. The Slurm wrappers are
`scripts/research/slurm/fakect_change_gpu.sh` and `scripts/research/slurm/fakect_change_report.sh`;
jobs use the existing TensorFlow 2.17 container/environment. Each fold publishes
change Dice, precision/recall, false-positive volume on controls and an offline
3D overlay with independent baseline, sibling and true/predicted change toggles.
The final epoch is declared in advance; held-out results do not choose weights.

The INI's initial engineering gates require mean case Dice >=0.60 and precision
and recall >=0.50 for each nonempty change class, plus predicted change on every
control <=1% of its reference aorta volume and passing crop coverage. These
thresholds are development decisions, not validated clinical requirements.
Empty-target overlap metrics remain undefined; their false positives are
reported separately. No downstream classifier starts automatically.

After successful localization, train Model 2 with predictions from Model 1
generated without fitting on that anatomy family. Supplying true change masks
to classifier training but estimated masks at inference would conceal an input
distribution mismatch. Severity target definitions and Model 2 evaluation are
separate next-stage work; no severity classifier is trained by this experiment.

Submitted focused-change validation on 2026-10-05: GPU smoke `6922315`
completed successfully in 61 seconds; localization array `6922316` runs five
folds with concurrency two, and CPU report `6922317` depends on array completion.
All 39 new TensorFlow 2.17 checks passed. Smoke peak memory was about 353 MiB
on GPU and 1,410 MiB host RSS, with true added and removed targets exercised.
Run metadata, gates and reports live in
`outputs/experiments/aorta/focused-change-r1/`; the root `report.html` links to
per-family metrics and 3D views once available. Performance is pending.


### Completed focused-change results evaluated on 2026-10-06

All five folds and the report completed successfully with exit code `0:0`.
Each fold fitted exactly 64 epochs / 14,336 optimizer updates. Source-lock
checks (62 files), parent metadata and all five checkpoint hashes passed.
No reserved infant data or downstream classifier was evaluated.

| Held-out family | Enlargement Dice | Narrowing Dice | Unchanged controls passing |
| --- | ---: | ---: | ---: |
| 1 year | 0.000 | 0.128 | 2/2 |
| 5 years | 0.209 | 0.212 | 0/2 |
| 10 years | 0.208 | 0.373 | 2/2 |
| 15 years | 0.388 | 0.313 | 1/2 |
| Adult | 0.067 | 0.141 | 2/2 |

Averaging cases within each family, then families, enlargement Dice is 0.174,
precision 0.341 and recall 0.144; narrowing Dice is 0.234, precision 0.565 and
recall 0.175. Undefined precision when no positive voxels were predicted is
assigned zero for the declared gate. All five localization gates failed.
Coverage passed for every edited scenario, with both change classes fully
covered in the saved crop audits. Seven of ten unchanged baselines pass the
1% reference-aorta-volume false-change limit. Failing controls are 260614
(1.694%), 260619 (1.517%) and 260616 (1.051%).

Training losses fall from 0.906–0.986 to 0.226–0.286 and were still improving
near epoch 64. These are sampled training-patch losses, not full-volume training
Dice. Full-focus training performance has not yet been measured. Therefore it
is premature to decide that the failure is solely overfitting, insufficient
epochs or an unobservable target.

The next diagnostic is inference on representative TRAIN scans and unchanged
controls using the existing frozen final checkpoints and exactly the held-out
whole-focus tiling. Good training localization with weak held-out localization
would point to generalization; poor training localization would motivate
checking fitting, tile boundaries, spatial context and input visibility before
a larger budget. The 57.6-mm input-patch context and uniform tile-boundary
averaging are hypotheses to test, not established causes. Model 2 remains
unstarted because the localization validation has not passed.

These are five related-anatomy development folds with one seed, not 60
independent patients. The consolidated review is
`outputs/experiments/aorta/focused-change-r1/review-r1/report.html`; per-fold
reports retain the saved 3D overlays. The additive evaluation receipt is
`checks/results-evaluation.json`. No new training was submitted in this review.


### Broader morphology population initiated on 2026-10-06

The user requested more digital siblings with longer CoA/arch narrowings and
varied asymmetry and ascending enlargement. Two separate friendly inputs now
control this expansion:

- `archive/research/configs/cohorts/aorta-diversity.ini` generates native paired siblings.
- `archive/research/configs/models/aorta-diverse-change.ini` trains Model 1 on that population.

The first population requests 256 new recipes for each of ten development
phantoms: 2,560 edited siblings. Each anatomy has 85 enlargement-only, 85
narrowing-only and 86 mixed requests covering all seven active-step patterns.
Independent seeded strata vary location, affected length, requested distance,
Gaussian concentration/window and inner/outer/symmetric direction. Angular
sectors span 45–180 degrees; symmetric edits use `all` (360 degrees). CoA
lengths span 8–40%, arch lengths 10–45% and ascending enlargement 12–50% of the
physical parent path, within their respective allowed location intervals.
Distance budgets, main ROI, crop, anatomical protection, tissue stiffness and
original connectivity safeguards remain inherited. Native final trunk
continuity is checked. Symmetric arch erosion uses a conservative half-peak
scale. Bone and skin constraints and the existing label-4 interior-fill proof
remain valid within the unchanged dilation envelope.

```bash
python3 scripts/research/prepare_sibling_population.py --config archive/research/configs/cohorts/aorta-diversity.ini --validate-only
python3 scripts/research/prepare_sibling_population.py --config archive/research/configs/cohorts/aorta-diversity.ini --stage list --ids-only
python3 scripts/research/train_change_population.py --config archive/research/configs/models/aorta-diverse-change.ini --validate-only
```

The model validation command becomes available after the complete population
has been collected. Generation input/code is already frozen; use fresh output
revisions for changes. Do not edit queued model sources/settings: the delayed
GPU launch checks the submitted checksum inventory before freezing/fitting.

All 2,560 requests pass metadata halo validation. Thirty-three production
TensorFlow 2.17 checks pass. Seven-pattern trials on 260602, 260613 and 260614
passed all 21 native executions with continuity and immutable source checks.
One request produced baseline-equivalent geometry and is explicitly recorded;
requested count is not a claim that every result is a distinct altered anatomy.
Variant/case manifests record actual changes, duplicates, no-ops and safeguards.
Every case must succeed completely before collection; failures are preserved.
Only compact native crops and mask companions are written, sharing original
whole-body ATN. Generated outputs are ignored by Git. The initial 10-case
resource estimate is about 3.5–6.5 hours per case for one native generation
pass; 24-hour CPU allocations accommodate longer edits and checks.

Each training fold uses 2,048 new edited siblings plus eight original controls,
compared with 48 edited siblings plus eight controls previously. The exact
old 60 held-out edited scans and ten controls, family exclusions, locator
weights, U-Net, loss, resolution, seed and total 14,336 optimizer updates remain
fixed. New TRAIN crops are predicted from their CT by the frozen fold locator;
coverage QA never recenters them. Single-edit and serial cancellation cases
use net-mask consistency, avoiding the old condition-specific three-step proof.

The sampler runs 64 reporting blocks of 224 updates, not 64 complete passes
through the new cohort. Every new sibling gets six patch visits, and each
original control gets 256, preserving the previous control exposure and 1/7
control fraction. This first experiment isolates broader TRAIN morphology
without also multiplying the optimization budget roughly 37-fold. A larger
training budget can follow as its own control if warranted by results.

Native trial array 6924565 completed. Full generation array 6924589 runs at
most four cases concurrently; collection 6924595 depends on its success.
GPU planning/smoke 6924599 depends on collection; five-fold fitting array
6924600 (concurrency two) depends on smoke; report 6924601 follows fitting.
The preliminary trial 6924561 exited before plan publication and created no
native data; it was replaced by the successful trial array. Full submission
history is `outputs/cohorts/aorta/diversity-r1/submission.json`.

The population is not yet complete and Model 1 retraining is queued behind
generation/collection. Model 2 remains unstarted. More siblings add controlled
morphology diversity; they do not add independent baseline anatomy. Infants
260612/260617 remain unopened. Later classifier training must also exclude
its outer-held-out family from every upstream model used to produce training
maps, not only from the classifier itself.


### Generation blockage and selective repair on 2026-10-07

The larger native generation attempted all 2,560 requests and succeeded for
2,549. Eleven configurations failed the additional selected-lineage trunk
continuity check: two each in 260613, 260616 and 260618; one in 260620; four in
260621. Five entire cases released successfully and five correctly remained
INCOMPLETE. Collection 6924595 was DependencyNeverSatisfied, so GPU smoke and
Model 1 training never began. Obsolete pending jobs 6924595, 6924599, 6924600 and
6924601 were cancelled rather than left permanently waiting.

The representative replay of 260613:615fc995 showed a single disconnected
ROI-selected carotid voxel while the broader thoracic anatomy remained
connected. A failed conservative check is not proof of a clinically severed
aorta. The rule is nevertheless preserved. Per-pass callbacks identify the
first failing erosion, reduce only that step's distance by 0.5 and replay from
the immutable baseline, with at most eight reductions per erosion. Dilation,
locations, lengths, angular directions, stiffness and existing safeguards remain
unchanged. In that pilot, arch distance 1.945934 ->0.972967 mm restored the strict
check while still removing 30 native target voxels.

`archive/research/configs/cohorts/aorta-diversity-repair.ini` publishes
`outputs/cohorts/aorta/diversity-r2`: it references all 2,549 successful payloads
without copying them and regenerates only the eleven rejected requests. New
accepted-content IDs preserve explicit requested-ID mappings, every rejected
attempt, requested/effective parameters and achieved step outcomes. New complete
per-case receipts authorize these references; failed r1 receipts and diagnostics
are preserved. No repairs are silently dropped, and exhaustion prevents release.
Original duplicates and baseline-equivalent scenarios remain explicitly counted.

A separate launch bug was caught before training: legacy prepared JSON metadata
lacks checksum sidecars. The new population experiment now checks its pinned
content hash directly; signed files still require their sidecar as well. No
original metadata was modified. The originally submitted 68-file model source
bundle is archived under `diversity-r1/checks/submitted-model-source`. Thirteen
repair/scheduling/metadata contract checks pass, including full freeze/inspect.

Replacement repair array 6930574 runs at most four case tasks (five merely
publish references). Collection 6930579 follows successful repair; GPU
planning/smoke 6930585 follows collection, and Model 1 array 6930586 follows
smoke with concurrency two. The new input is
`archive/research/configs/models/aorta-diverse-change-r2.ini`; five family splits, fixed old
held-out panels, architecture and 14,336-update comparison remain unchanged.
Model 1 is pending repair and validation; Model 2 remains unstarted.


### Expanded-cohort Model 1 results evaluated on 2026-10-07

All eleven selective repairs succeeded. Collection published all 2,560 edited
siblings plus ten reused original baselines. Production smoke, all five fits
and the report completed with exit code `0:0`. Each fresh model received exactly
14,336 updates, including 12,288 edited patches and 2,048 controls; each new
sibling received six patches and each original control 256. Source/dependency
checks (67 files/48 dependencies) and all five checkpoint hashes passed.

| Mean across development families | Small cohort | Expanded cohort |
| --- | ---: | ---: |
| Enlargement Dice | 0.1743 | 0.0000 |
| Narrowing Dice | 0.2336 | 0.0136 |
| Enlargement recall | 14.39% | 0.00% |
| Narrowing recall | 17.53% | 3.01% |
| Unchanged controls passing false-change limit | 7/10 | 2/10 |

Held-out payloads, target counts, focus crops, architecture, loss, normalization
and seed match. All five localization gates fail. Zero enlargement Dice does
not mean zero enlargement predictions: one family produces false enlargement
regions, but there are no enlargement true positives in the held-out panel.
Narrowing false positives increase substantially. Model 2 remains unstarted.

The expanded experiment changed the foreground distribution as well as the
number of siblings. Achieved class patterns are 804 both, 898 enlargement-only,
820 narrowing-only and 38 no-ops. The old selected panel contains both classes
in every edited scan. Positive change volumes are generally smaller, though
large changes are also present. Requesting an absent class falls back to a
uniform patch: the expected genuine change-centred share of edited patches
falls from 75% to 48.72% (all updates: 64.29% to 41.76%, including controls).
Thus matched optimizer/control budgets do not imply matched positive-voxel
exposure or loss balance. This does not establish that virtual-cohort diversity
is inherently harmful; this training protocol did not benefit from the expansion.

A read-only TensorFlow 2.17 check of six purposive change-centred TRAIN patches
from the 10-year holdout fold found zero enlargement true positives in all three
patches containing enlargement. One adult narrowing patch recovered 1,274 of
1,927 removed voxels but had Dice 0.340 due to false positives; two small-child
narrowing patches had zero true positives. These six patches are a diagnostic,
not whole-cohort training performance. They show that the observed failure is
not confined to unseen anatomy. Weights were not updated.

Next retain the full dataset, choose added/removed patch modes only when that
class actually exists, balance achieved small/moderate/large change burdens
within training cases, and audit absent-target versus present-target loss
contributions. Require a bounded training-only localization exercise to learn
both classes before another full five-fold run. A larger update budget can then
be tested separately; merely adding more siblings or assuming 200 epochs will
solve the collapse is not supported by this result.

The consolidated comparison with figures and links to per-fold 3D overlays is
`outputs/experiments/aorta/diverse-change-r2/review-r1/report.html`.
`checks/train-patch-diagnostic.json` and `checks/results-evaluation.json` retain
the diagnostic/evaluation evidence. Original outputs and frozen sources were
preserved; no new training was submitted during this results evaluation.


### Class-present sampling diagnostic initiated on 2026-10-07

The user approved the recommended sampler correction and small learning check.
The additive `fakect_change_sampling.py` balances TRAIN cases, actual present
added/removed classes and within-case/class low/middle/high achieved burden
strata. An absent requested class now raises explicitly, including classes
lost on nearest-neighbor resampling; it never becomes a uniform positive draw.
Declared uniform context remains distinct. Zero-net edited siblings are listed
outside positive queues; original healthy controls retain 1/7 of updates.
The audit of full TRAIN rosters gives 6,144 added requests, 6,144 removed requests
and 2,048 control updates per fold, before the declared context-draw Bernoulli.

The first bounded exercise uses
`archive/research/configs/models/aorta-change-sampling-pilot.ini`: source TRAIN cases 260602 and
260613 from the completed 10-year family fold. Six edited siblings per case
cover three achieved burden strata for each class, with single/mixed examples;
two original controls are included. The fixed cache has twelve verified
change-centred patches plus four healthy proposal-centred patches, all96cubed.
Each of64epochs performs24edited and4healthy updates:1,792total. This deliberately
concentrated, supervised TRAIN memorization exercise is not a causal comparison
of generalization or a full pass through the2,560-sibling cohort. Baseline,
labels and burden bins are supervision only; predictor inputs remain CT+validity.
Network, learning rate, loss and resolution are unchanged, with fresh weights.

All41 TensorFlow2.17 tests pass. The actual source panel independently passes
same-grid truth, positive-class-survival and unchanged-control checks. The
smallest child narrowing patch contains29model voxels. Source files and older
results remain unchanged. GPU smoke6975386 passed (about354MiBGPU,1,520MiBhost;
stable single traces and safe reload). Diagnostic6976154 runs after smoke;
report6976551 follows it.

Initial and final predictions are measured on exactly the same fixed patches.
Whole-focus inference on these14TRAIN scans is reported separately, using the
usual image-predicted crops and tiling. Whole-focus original controls enforce
false change/original aorta volume≤1%; patch background is not the denominator.
The gate requires both classes to detect voxels in both anatomies and meet
Dice/precision/recall requirements. Even a pass does not establish held-out or
clinical reliability, or start Model2/full-five-fold training automatically.
The patch manifest and all cached arrays are checksum-bound to the GPU smoke.

```bash
python3 scripts/research/train_change_sampling.py --config archive/research/configs/models/aorta-change-sampling-pilot.ini --validate-only
python3 scripts/research/train_change_sampling.py --config archive/research/configs/models/aorta-change-sampling-pilot.ini --stage report
```

Run metadata, initial/final 3D overlays and the resulting gate are published at
`outputs/experiments/aorta/change-sampling-pilot-r1/report.html`. No validation,
held-out or reserved image payloads are opened by this exercise.


### Sampling diagnostic result and export recovery on 2026-10-07

Diagnostic 6976154 completed all 64 epochs and 1,792 updates (final sampled loss
0.26844), but its export check failed before publishing results. It compared
traced inference with eager inference at a strict logit tolerance. The original
failed outputs remain preserved. Independent CPU checks found 30 weight arrays
and 91,559 parameters unchanged through serialization. Read-only GPU recovery
6980813 then verified exact weights and bitwise identical graph roundtrip
outputs on all 16 fixed patches. GPU graph versus eager paths differed by at
most 0.000122 logits and 0.000000507 probability, with zero changed voxel
classes. Recovery performed no optimizer updates. The departed process's
pre-save variables were unavailable; these checks establish the saved archive's
consistency rather than claiming verification of that missing state.

The derived report is
`outputs/experiments/aorta/change-sampling-pilot-r1/recovery-r1/report.html`.
On the same fixed TRAIN patches, equal-case enlargement Dice is 0.6674,
precision 0.7849 and recall 0.5952. Narrowing Dice is 0.5415, precision 0.5932
and recall 0.5147. Both classes detect voxels in both adult and child anatomies.
The declared gate still fails the narrowing Dice floor of 0.60 and the
whole-focus healthy-control check. False change divided by original aorta
volume is 0.568% for the adult control and 6.641% for the child control.

Whole-focus TRAIN inference remains weak on edited scans: adult enlargement
Dice ranges from 0.012 to 0.155 on scenarios containing that class; child
enlargement Dice is zero, and narrowing Dice is zero throughout this 14-scan
panel. These are TRAIN results, not held-out generalization. Centred-patch
memorization does not yet transfer to the current whole-focus tiling. No full
five-fold training or Model 2 was started.

Next test class-present sampling with varied positive positions and actual
inference tile positions on the small TRAIN panel. Keep fixed-patch and
whole-focus scores separate and retain unchanged controls. Investigate tile
boundaries and context before attributing the gap to anatomy generalization.
Loss changes and larger budgets remain separate subsequent controls. All
original sources, data, trained weights and failed export artifacts are
preserved; the recovery is a separately versioned derived evaluation.

### Baseline-relative CT exploration before Models 1 and 2

`archive/research/configs/analysis/aorta-ct-clusters.ini` defines an unsupervised descriptive
analysis of the completed `aorta/diversity-r2` development population. All
2,560 edited siblings enter fitting; the ten original baselines are projected
as zero-difference reference points. Reserved infants remain excluded.

Only paired attenuation differences enter the feature matrix. The original
main aorta centreline and identified thoracic slab/crop define the fixed
analysis region. Coordinates use normalized physical arc length, distance
divided by the original median aortic radius, and a transported angular frame.
Each signed attenuation difference is normalized by the original arterial
attenuation and integrated with physical voxel volume over 32 longitudinal,
four radial and eight angular bins. Positive and negative channels give 2,048
features. This removes native voxel-count and anatomy-size units while
retaining spatial pattern and relative extent. Finite voxel sampling still
introduces discretization error; the representation is not an exact claim of
invariance to arbitrary resampling. Coverage outside the fixed domain is
reported. CT-identical anatomical changes cannot be recovered from CT alone.

PCA mean-centres without normalizing each sibling to unit length. K-means
uses retained components, rather than only the first two plotted components.
Candidate K=2–10 is assessed with silhouette, restart agreement and minimum
cluster size. The best partition and the diagnostic suggested count are
reported separately: partitions of a continuum do not establish disease types.
Phantom-coloured and extent-coloured views check residual anatomy effects.
Mask counts appear only as posthoc annotations and never define the features.

```bash
python3 scripts/research/cluster_sibling_ct.py --config archive/research/configs/analysis/aorta-ct-clusters.ini --stage plan
fakect_ct_feature_job=$(sbatch --parsable --array=0-9%4 scripts/research/slurm/fakect_ct_clusters_cpu.sh archive/research/configs/analysis/aorta-ct-clusters.ini)
sbatch --dependency="afterok:${fakect_ct_feature_job}" scripts/research/slurm/fakect_ct_clusters_report.sh archive/research/configs/analysis/aorta-ct-clusters.ini
```

The plan freezes source metadata, inputs and code; new revisions require a new
output directory. Extracted rows, feature ordering, checksums and zero controls
are checked before publication. The offline report and PNG/PDF/SVG exports
are saved under `outputs/analysis/aorta/ct-delta-clusters-r1/review-r1/`.
No model is trained here. If these descriptors later guide training selection
or an evaluated predictor, PCA and clustering must be refit within each TRAIN
partition; the all-development-cohort fit is an exploratory result only.

The first completed analysis (extraction array 7007413, report 7007777) used
all 2,560 siblings and ten projected original controls. All controls have
exactly zero features; every sibling's CT-difference energy was entirely
captured within the fixed domain. Thirty-eight siblings have no CT difference.
All 27 feature, PCA/clustering and reporting tests passed. Report hashes and
feature integrals were independently checked after publication.

K=2 had the highest sampled silhouette, 0.7145, and identical assignments
across all four restarts (adjusted Rand agreement 1.0). Cluster 1 contains
2,359 siblings; cluster 2 contains 201. Both contain all ten phantoms.
Cluster 2 has strong distal CT-decrease energy: its mean longitudinal location
is 74.0% of the baseline path and 93.6% lies beyond two-thirds of the path.
Cluster 1 contains the remainder, including prominent CT-increase examples.
These are exploratory partitions, not established clinical disease classes.
The scatter has a tapered continuous shape without an obvious empty gap.

PC1 and PC2 explain 44.56% and 10.08% of total variance. Clustering uses
twenty components, which retain 88.59%, below the requested 90% target.
Both plotted components place approximately 99.92% of their squared loading
weight in the CT-decrease channel; PC1 correlates with CT-decrease energy
at r=0.936. Thus this first plot mainly displays distal decrease extent and
location. It does not comprehensively separate enlargement and narrowing
patterns. A future comparison can balance the two channels using fixed
population-wide weights while retaining variation in sibling extent; this
would be a separate derived analysis, rather than replacing the physical
CT-amplitude result. Full coordinates and annotations are in `siblings.csv`;
signed longitudinal cluster profiles and verification are recorded in
`outputs/analysis/aorta/ct-delta-clusters-r1/interpretation-r1.json`.

### Pretrained aorta benchmark initiated on 2026-10-08

The next application tests whether controlled virtual morphology can improve
an established CT segmentor. Begin with fixed pretrained inference before
comparing matched baseline-only and baseline-plus-sibling fine-tuning. Current
PCA partitions are descriptive CT-change patterns, not intermediate/severe
clinical labels. Existing TensorFlow runs and frozen population files remain
intact. This additive integration uses TotalSegmentator 2.18.0 in a separate
PyTorch 2.8.0 CUDA environment.

Input: `archive/research/configs/analysis/aorta-pretrained-benchmark.ini`.
Output: `outputs/experiments/aorta/pretrained-totalseg-r1`.
The initial development panel contains sixteen scans from 260602, 260613,
260616 and 260621: one original, a lower removed-only burden, the maximum
removed-only burden and the maximum added-only burden per case. These roles
are deterministic engineering descriptions, not validated disease grades.

XCAT calibration and orientation were confirmed rather than assumed wrong.
All ten development cases have logged water attenuation 0.1614 cm^-1 at
120 keV photon energy, consistent per-pixel scaling and native +i left,
+j posterior, +k superior axes. Five named anatomical landmark comparisons
per case corroborate those axis directions. The export applies the standard
water-normalized HU formula and encodes the original voxel-centre geometry
in a RAS NIfTI qform/sform, including spacing and crop offset. The logged water
value is rounded; no additional precision is invented. Unit conversion does
not simulate a different scanner acquisition or reconstruction.

Inputs retain full native X/Y coverage and a fixed atlas thoracic z slab with
50 mm context on each end. This is a region-assisted benchmark, not a test of
automatic thorax detection. Native source ATN is read in bounded slabs and
immutable sibling crops are overlaid before conversion; full-body siblings
are not duplicated. Image and target are not resampled at export. The official
pretrained `total` task uses 1.5 mm internally and a 6 mm rough crop model;
returned predictions must exactly match the native export grid and affine.
The official API downloads the default task checkpoint set, even though only
the cardiac part is inferred for the requested aorta class. All checkpoints
and environment versions are checksum-bound to the frozen benchmark.

Reference targets use the existing atlas thoracic **core**, excluding branch
segments and previously excluded disconnected label islands. Accepted sibling
edits are transferred through original-core voxel ancestry; growth can extend
beyond the old tight atlas crop. A full-X/Y thoracic evaluation domain includes
false positives away from the target. Provisional atlas boundaries are not
silently promoted to clinically reviewed ground truth. Native changed masks
define fixed local evaluation regions only, never model inputs or recentering.

Reports provide native whole-core and change-local Dice, precision/recall,
area-weighted voxel-face surface Dice/HD95 and six-connected component
statistics. Artificial domain caps are excluded from surface scores. Offline
3D overlays independently toggle truth, prediction, errors and known changes
with 15/45/80% opacity; axial views show whole-source native coordinates.
These component statistics do not establish a validated centreline or clinical
minimum-lumen measurement. Those remain subsequent measurement work.

All 34 evidence, export/lineage, panel-selection and report tests passed in the
production benchmark environment, including actual adapter-to-NIfTI-to-report
interoperability. Read-only baseline/largest-sibling transfer checks also pass
all ten development cases. Heavy exports, model weights, environments and
predictions remain ignored by Git; code, inputs and tests are tracked normally.

```bash
python3 scripts/research/benchmark_pretrained_aorta.py --config archive/research/configs/analysis/aorta-pretrained-benchmark.ini --validate-only
# The frozen revision already exists; use a new output directory to plan again.
python3 scripts/research/benchmark_pretrained_aorta.py --config archive/research/configs/analysis/aorta-pretrained-benchmark.ini --stage list --ids-only
```

Preparation array 7261020 produces the native sixteen-scan panel. GPU inference
array 7262089 follows successful preparation, one phantom at a time. The CPU
review 7262659 follows successful completion of all inference tasks. These jobs perform
no model fitting. Package/model acquisition receipts and tests are stored under
`outputs/environments/totalseg-r1/`, with model checksums under
`outputs/pretrained/totalseg-2.18.0/model-lock.json`.

After reviewing the pretrained baseline, use the same checkpoint initialization
for baseline-only and baseline-plus-sibling adaptation, match optimizer updates
and preserve anatomy-family splits. Sampling should cover achieved narrowing,
enlargement, lesion length and asymmetry, not only the decrease-dominated PCA
groups. Plot held-out local accuracy versus achieved morphology and sibling
count, with phantom/family-level summaries. Improvement on unseen synthetic
anatomies is distinct from clinical generalization, which requires independently
labelled clinical CT evaluation.

Primary model documentation:
[TotalSegmentator](https://github.com/wasserth/TotalSegmentator),
[nnU-Net transfer](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/pretraining_and_finetuning.md).

Independent export replay is saved in
`outputs/experiments/aorta/pretrained-totalseg-r1/input-audit-r1/receipt.json`.
The reusable script is `scripts/research/evaluation/audit_pretrained_exports.py`. All
64 headers/hashes, streamed mask counts, four exact baseline core masks and
32 direct native ATN-to-HU comparisons passed. Maximum sampled HU difference
was 0.0001802. The receipt includes the sampled voxel coordinates and expected
values, and limits voxel equality to those samples rather than claiming a
whole-image equality check. Baseline branch exclusion is checked explicitly.

A read-only transfer feasibility check reconstructed the official Dataset293
cardiac model with nnU-Net's plans and strictly loaded every checkpoint tensor:
31,209,279 parameters, 19 output channels, aorta at local channel 2, and five
pretrained deep-supervision heads. Ordinary nnU-Net `-pretrained_weights`
deliberately skips segmentation heads; it is unsuitable unchanged for retaining
this aorta predictor. `load_checkpoint` also resumes old optimizer/epoch state.
A subsequent external trainer should strictly initialize network weights only,
create a fresh optimizer, and preserve all head shapes. Aorta-versus-rest loss
can aggregate the other 18 logits with logsumexp, respecting unknown regions
outside the reviewed domain rather than inventing labels for other organs.
Preserve the same inference extractor in every comparison: the official
19-way argmax followed by aorta extraction differs from a fixed probability
threshold of 0.5. Preprocessing order and numeric parity also require checking.
This trainer has not yet been implemented or run.

Before choosing the fine-tuning grid, audit target loss under internal 1.5 mm
sampling. Native-resolution output cannot restore erased thin lumens. If a
0.6 mm focused refinement is needed, apply it equally to both population arms:
the same 128-cubed patch then covers 76.8 mm, rather than the current 192 mm.
Do not attribute a changed physical field of view or resolution to the effect
of additional siblings.

### Completed pretrained results on 2026-10-08

Preparation, all four inference tasks and CPU review 7262659 completed with
exit code 0. All sixteen native predictions passed grid/affine roundtrip. The
review is `outputs/experiments/aorta/pretrained-totalseg-r1/review-r1/report.html`,
with complete native and lesion-local metrics, axial views and sixteen exact
interactive 3D scenes. No surfaces were omitted. Thirty-nine published artifact
hashes were independently verified. This run performs no model fitting.

| Phantom | Original whole-core Dice | Higher-narrowing whole-core Dice | Higher-narrowing change-local Dice |
| --- | ---: | ---: | ---: |
| 260602 | 0.9264 | 0.9210 | 0.9336 |
| 260613 | 0.0000 | 0.0000 | 0.0000 |
| 260616 | 0.8105 | 0.7819 | 0.3032 |
| 260621 | 0.6007 | 0.6100 | 0.7824 |

The panel includes one adult, one one-year reference and two fifteen-year
references, not four independent age families. Scores are descriptive,
region-assisted synthetic-domain results. The strong adult result supports
input/model compatibility. It does not establish general reliability. In
260616 the higher-narrowing local precision is 0.988 and recall only 0.179,
so the main issue is missed remaining vessel. Whole-core Dice obscures much
of this failure. Child 260613 has just 81 false-positive voxels in the original
scan and zero overlap with its 78,609-voxel truth; the prediction is not a
correct abdominal aorta hidden outside the evaluation slab.

The unchanged-head localization diagnostic uses
`archive/research/configs/analysis/aorta-pretrained-focus-diagnostic.ini` and
`scripts/research/evaluation/probe_pretrained_focus.py`. GPU 7277779 supplies a solid
rectangle from the frozen atlas crop plus 40 mm X/Y context and the full
already-exported thoracic Z range, bypassing the model's automatic 6 mm crop.
It uses the same official 1.5 mm cardiac weights, preprocessing and aorta
extraction without optimizer updates. This is an explicitly privileged
known-region diagnostic, not a blind CT localization result.

The diagnostic report is
`outputs/experiments/aorta/pretrained-totalseg-focus-r1/review-r1/report.html`.
For original 260613, Dice rises from 0.0000 to 0.3514 and recall to 0.2275;
for original 260621, Dice rises from 0.6007 to 0.7415 and recall to 0.6204.
Context/crop dependence is evident, but it does not explain all remaining
segmentation errors. No phantom calibration or orientation correction was
needed beyond the verified standard unit/header conversion. The next controlled
adaptation should retain this distinction between localization and fine
segmentation, quantify the sampling ceiling, then compare matched baseline-only
and sibling-augmented training. Neither result is a fine-tuning outcome.

### Matched pretrained adaptation and burden profiles

`archive/research/configs/models/aorta-pretrained-finetune.ini` now defines the controlled
comparison. All three arms use one fixed atlas-assisted region, canonical RAS
model tensors, 1.5 mm sampling, the original cardiac CT normalization and the
same Gaussian-overlap inference. The untuned comparator is regenerated under
this common protocol; old automatic-crop scores are context, not the comparator.
Both training arms strictly retain all 19 output channels and every pretrained
head, initialize a fresh AdamW optimizer and receive 2,000 batch-one 128-cubed
patch updates. Aorta-versus-rest supervision uses CT only as input and ignores
unannotated context beyond the thoracic slab. Original source/checkpoint files
remain immutable.

Five conservative anatomy-family folds reuse the existing eight-TRAIN/two-test
assignments. Each TRAIN phantom contributes its original in the baseline arm,
or that original plus sixteen distinct siblings in the population arm. The
sixteen cover four addition-only, four removal-only and eight mixed examples,
with burden extremes and intermediate values. Each held-out phantom supplies
one original plus twelve different, preregistered burden-spanning siblings.
Across the five folds this gives 130 unique evaluation scans per arm. No
held-out early stopping/checkpoint selection is performed. Source phantoms and
their related family partners stay together; these are development folds,
not independent clinical validation.

Case order and baseline-derived physical patch positions are identical between
training arms. The population arm retains originals for 25% of updates; its
other images differ through sibling morphology. Baseline model volumes are
stored once per phantom; exact changed CT/target values are sparse model-grid
deltas. Complete whole-body sibling volumes are never created. Compact native
evaluation crops are shared by all model arms. Native target/reference exports
must match the exact core census and preserve the verified NIfTI geometry.

The reusable core census covers all 2,560 edited siblings and ten originals.
Core-only changed-count bins contain 39 zero-edit siblings, 241 with 1–100,
785 with 101–500, 442 with 501–1,000, 855 with 1,001–5,000 and 198 above 5,000
modified voxels. Maximum core additions/removals are 15,117/6,177. These differ
from earlier broad-mask counts because branches are excluded. Counts describe
net geometry versus each original, not cumulative serial edit events or
validated clinical intermediate/severe grades.

The completed original/max-removal geometry audit has minimum whole-reference
roundtrip Dice 0.911 and local Dice 0.890, no lost target coverage and no
substantial new trunk breaks. Preparation additionally records all 130
evaluation-mask roundtrips; inspect those before study fits. Roundtrip fidelity
is a resolution diagnostic, not a mathematically strict bound on every learned
prediction. A production-size permitted TRAIN gradient/reload smoke precedes
fits. The configured limits are 14,000 MiB allocated GPU and 24,576 MiB host RSS.
All 32 new data/learning/study/comparison tests pass in the production environment.

The comparison report shows the SAME current CT and selected planes in four
columns: atlas reference, pretrained, original-only fine-tuning, and sibling
fine-tuning. Zooms use frozen known-change support for edited cases, with native
source coordinates and RAS millimeters. Three example identities are chosen
before inference using the original/control and upper-quartile edit burdens,
not prediction improvement. Axial/coronal/sagittal views and transparent 3D
comparisons are available. Removed and added voxel counts receive separate
histograms and performance profiles, with global/local Dice, recall, surface
metrics, per-phantom views and physical mm3 adjuncts. Average within phantom
before combining related siblings; do not interpret native voxel counts as
resolution-independent clinical disease severity. PNG/PDF/SVG exports support
poster figures. Unfinished arms are explicitly pending.

```bash
python3 scripts/research/finetune_aorta.py --config archive/research/configs/models/aorta-pretrained-finetune.ini --validate-only
python3 scripts/research/finetune_aorta.py --config archive/research/configs/models/aorta-pretrained-finetune.ini --stage list
# Use a fresh output revision for any frozen input/code change.
```

Output: `outputs/experiments/aorta/pretrained-finetune-r1`.
Preparation array 7289927 creates baseline/sparse-delta caches and native
references. TRAIN-only production smoke 7289938 and the explicitly pending
preparation review 7289939 follow it. Study fitting starts only after successful
preparation, resampling review and gradient/reload/memory verification. Full
training/inference submission and final report receipts are recorded in the
experiment directory; no unfinished model is presented as a completed result.

The production smoke passed two actual 128-cubed TRAIN updates, using about
2,764 MiB allocated GPU and 2,379 MiB host RSS. Saved/reloaded predictions
matched exactly. All 130 evaluation-mask roundtrips retain target coverage;
minimum global/local roundtrip Dice is 0.911/0.871, and the minimum largest
component fraction is 0.999823. No substantial new trunk discontinuity was
introduced by the fixed grid. The signed resolution review is stored with the
experiment.

Unmodified reference evaluation array 7289948 and matched training/evaluation
array 7289951 completed all five folds and all ten fine-tuned models.
Initialization and patch-schedule hashes match across both arms in every fold.
Each model received 2,000 updates; no smoke weights initialized study fits.
All 390 arm/scan predictions are complete: 130 identical held-out scans in each
of the three arms. The earlier partial interim review 7289964 remains a
historical snapshot of the first completed 10-year family. Final comparison
report 7289956 calculates full surface metrics after all arrays complete.

The initial report/histograms are in
`outputs/experiments/aorta/pretrained-finetune-r1/review-preparation-r1/report.html`.
Interim and final comparisons use `review-interim-r1/` and
`review-comparison-r1/`. Both histories and figure inputs are retained; later
reports never replace earlier reviews. The exact source/model/split identities,
job dependency chain and first-pair parity are recorded in `submission.json`.

The first interim snapshot contains only the completed 10-year held-out family
for the two fine-tuning arms (26 scans), while pretrained predictions cover all
130. Comparisons must use the same 26 completed scans, rather than mixing the
different available sample totals. Independent native-overlap checks give
whole-core Dice 0.91480 pretrained, 0.93388 original-only fine-tuning and
0.93404 sibling fine-tuning. For the 24 edited scans, change-local Dice is
0.92708, 0.93884 and 0.93904. This supports adaptation to these synthetic data,
not a demonstrated population-augmentation advantage: sibling minus baseline
fine-tuning is only 0.00017 whole-core / 0.00020 local in this first family.
The other folds subsequently completed; the following all-family overlap
readout replaces that interim snapshot for the main comparison.

A real zoomed comparison is available at
`outputs/experiments/aorta/pretrained-finetune-r1/zoom-example-r1/example-00.png`
(also PDF/SVG). It uses the already-preregistered 15-year example
`260616:sibling-9b51bb01b6dc1d3e`, with 2,556 added core voxels and no removals.
All three completed models see exactly the same current modified CT; the
four columns show its reference and three predictions in axial, coronal and
sagittal planes. Shape differences are modest and this illustration does not
establish an average improvement. Input/output hashes, source code and the
frozen example identity are retained beside the figure.

### Completed five-family comparison and final presentation

The complete matched native-overlap analysis gives:

| Mean native Dice | Pretrained | Baseline-only fine-tuning | Sibling fine-tuning |
| --- | ---: | ---: | ---: |
| Whole thoracic aorta, 130 scans | 0.71894 | 0.90087 | 0.90631 |
| Change-local region, 120 edited scans | 0.68671 | 0.83445 | 0.82666 |

Both fine-tuning arms substantially improve phantom-domain segmentation.
Adding siblings improves whole-aorta Dice by 0.00545 over original-only
fine-tuning, but lowers mean change-local Dice by 0.00779. Four of the five
families improve locally; the one-year family declines by 0.06327. Its local
recall decreases from 0.39818 to 0.33812 while precision increases from 0.80528
to 0.83057: the problem is missed remaining aorta. This is a mixed augmentation
result, not evidence of a uniform improvement in occlusive-disease segmentation.
Do not omit the one-year family after observing its result. The equal 26 scans
per family make these means family-balanced, while the independent anatomical
information still comes from five families, not 130 independent patients.

The final scientific review `review-comparison-r1` completed successfully in
job 7289956, with all 130 scans paired across three arms and no pending/failed
predictions. Its native overlap averages exactly match the independent readout
above. Mean whole/local surface Dice is 0.68985/0.65460 pretrained,
0.92821/0.84242 original-only, and 0.93879/0.84110 sibling fine-tuning; surface
tolerance is 1 mm. Whole/local HD95 is 21.458/7.770 mm pretrained,
1.388/2.429 mm original-only, and 1.343/2.267 mm sibling fine-tuning.
Surface results therefore also support a mixed augmentation effect rather
than resolving the local recall deficit. Local HD95 averages have different
denominators: pretrained 112/120, original-only 120/120, siblings 117/120. Eight
pretrained and three sibling local prediction surfaces are empty, so their
distances are undefined. The apparently lower sibling local HD95 excludes
three failures and must not be presented as an unconditional improvement.
Dice, recall and surface Dice retain those failures. The completed report includes
original/control and addition-only, removal-only and mixed readouts, with
Dice, precision, recall and surface curves against exact native added and
removed voxel counts. The full 2,560-sibling census supplies the histogram;
mixed siblings appear in both addition and removal distributions. Physical
volume and per-phantom views complement counts because native spacing differs.
Counts remain synthetic burden, not validated clinical disease grades.

The final CT/3D presentation is already available at
`outputs/experiments/aorta/pretrained-finetune-r1/zoom-prespecified-r2/report.html`.
It contains all three examples frozen before inference, an unmodified original
and two edited siblings, each shown with identical CT/slices/window across
reference, pretrained, baseline-only and sibling predictions. Per the requested
presentation style, the reference is cyan (`#00ffff`), predictions are magenta
(`#ff00ff`), and CT contours are 2.2 points wide. The offline 3D viewer retains
independent visibility and opacity controls. PNG/PDF/SVG files, coordinate
metadata and verified checksums accompany the figures. A separate
`young-child-diagnostic-r1` figure is explicitly post-hoc failure inspection,
chosen by maximum held-out removal burden within the one-year family; it is
not a preregistered representative example.

A complementary post-hoc removal-stress example is available at
`outputs/experiments/aorta/pretrained-finetune-r1/stress-removal-example-r1/index.html`.
Its identity, `260615:sibling-422b68a0c1aa1a73`, was chosen by maximum removal
in the fixed evaluation panel (6,177 core voxels), independently of prediction
scores. Saved local Dice is 0.830 pretrained and approximately 0.922 in both
fine-tuning arms. It illustrates successful adaptation under narrowing stress,
without an appreciable sibling-versus-original-only benefit in that scan.

Presentation job 7290033 also completed successfully, publishing
`review-presentation-r2` from saved scientific metrics and styled prespecified
figures. The final user-facing review is
`outputs/experiments/aorta/pretrained-finetune-r1/review-presentation-r4/report.html`.
It retains thicker cyan/magenta contours, uses distinct gray/cyan/magenta model
curves at 2.5 points, constrains burden axes to nonnegative counts and separates
bin-count labels for readability, with the undefined-distance counts disclosed
explicitly. No presentation reruns models or
recalculates metrics; both preserve the earlier scientific review. All output
checksums were verified. The final publisher source is saved beside its report,
with the earlier publisher archived separately. The command is reproducible
through `scripts/research/style_pretrained_review.py`; the signed submission receipt is
`presentation-submission.json`. PNG/PDF/SVG versions of every figure and the
390-row metrics CSV are available beside the final HTML report.

Before increasing architecture size or sibling counts, inspect added-voxel
recall, removed-voxel rejection and retained-aorta recall using the existing
predictions. Then audit TRAIN exposure to edited regions. If exposure is sparse,
compare identical lesion-focused patch schedules in both arms while preserving
model initialization and update budget. Repeated seeds and nested sibling doses
can subsequently test whether an observed population effect is stable.

### Simplified pretrained versus fine-tuned presentation

`review-pretrained-finetuned-r1/report.html` displays only **Pretrained** and
**Fine-tuned**, where Fine-tuned is the model trained using original phantoms
plus digital siblings. CT panels have three columns: Reference, Pretrained,
Fine-tuned. Curves, family summaries, paired plots, supplemental illustrations
and the offline 3D model selector use the same two labels. The cyan/magenta
contours and 2.2-point borders are retained. All 130 displayed scans and their
measurements are unchanged; CSV contains 260 model/scan rows. The earlier full
controlled comparison remains preserved in its original report directories.

The presentation is generated by `scripts/research/publish_aorta_two_model_review.py`,
with CT assets from `scripts/research/evaluation/aorta_two_arm_ct.py`. Native slices,
crop coordinates, HU windows and retained 3D meshes are verified against the
original assets. No model fitting, inference or metric replay is performed.
The self-contained local package is
`outputs/exports/fakect-aorta-pretrained-vs-finetuned-2026-10-08.zip`.
Extract it fully and open its top-level `index.html`. The ZIP includes
PNG/PDF/SVG exports, the two-model CSV/JSON, supplemental examples, study
metadata, offline JavaScript and checksums. Its reproducible packaging command
is `python3 scripts/research/package_aorta_two_model_review.py --date 2026-10-08`.

### Branch-inclusive target correction

The original adaptation used `ThoracicCoreProvider`, which intentionally read
only the atlas `core_mask` and marked connected branches as background. Both
models already had identical physical CT boxes; the mismatch was the definition
of foreground. The official cardiac checkpoint also gives the brachiocephalic,
subclavian and common-carotid arteries their own output labels, whereas the
original evaluator extracted channel 2 alone. The earlier core-only results and
ZIP remain unchanged and should be interpreted with that narrower target.

`archive/research/configs/models/aorta-pretrained-finetune-branches-v3.ini` defines the corrected
study in `outputs/experiments/aorta/pretrained-finetune-branches-r3`. It retains
the exact original cases, family folds, sibling rosters, example identities,
native CT boxes and 1.5 mm grids. Branch-inclusive labels start with the verified
atlas aortic component and follow connected atlas-qualified systemic,
cerebrovascular and coronary arteries within that common box and the same
thoracic slab. Pulmonary labels and disconnected same-ID components are
excluded. Branches continue to the common spatial boundary, rather than stopping
at the former three-radius reach inside the box. The template transfers edited
membership through exact source-component ancestry and retains untouched
branches outside the edit ROI/crop. No prediction is intersected with the
ground-truth foreground.

Both pretrained and adapted models now use the identical winner-class union
`{2,4,5,6,7,8}`: aorta, brachiocephalic trunk, right/left subclavians and right/left
common carotids. The masked binary loss marginalizes the same foreground group.
All 19 vendor channels and pretrained heads remain intact; grouped semantics are
recorded separately in model/checkpoint metadata. Coronary segments may belong
to this atlas-defined target, but the original cardiac head has no dedicated
coronary channel. Their detection is custom adaptation, not existing pretrained
coronary coverage. Separate native core/branch reference masks and positive-voxel
recalls accompany combined-target metrics.

The published anatomical audit is
`outputs/experiments/aorta/pretrained-finetune-branches-r3/annotation-review-r1/report.html`.
All ten baseline targets retain the original core and old proximal branches,
exclude pulmonary IDs, and are six-connected. The recovered baseline branch
total is 266,288 native voxels, compared with 56,926 in the old proximal selection
and zero branch positives in the old learning target. This includes 209,362
continuation voxels beyond the former internal reach cut. Cyan core/magenta
branch CT previews and independent-opacity native 3D views cover the three
predetermined example cases. The audit contains reference anatomy, not updated
model predictions.

Preparation 7292975 completed all ten cases and 130 evaluation references. Every
baseline normalized CT input exactly matches its original cache; no target
leaves the shared model crop, and all 130 native target masks are connected.
The bounded native audit allows 64 million voxels for the wider branch reference;
legacy core audits retain their 32-million default. ACT decoding accepts exact
signed integral IDs, preserves signed atlas eligibility, and rejects fractional
or overflowing labels without rounding. Earlier failed preparation attempts are
retained in separate revisions; no model used them.

All 58 integrated target/data/learning/study/report tests pass. Grouped production
smoke 7292991 passes gradient and checkpoint-reload parity at about 2,696 MiB GPU
and 2,440 MiB host RSS. Native branch-to-1.5mm round-trip Dice ranges from 0.716
to 0.835; smaller branches remain resolution-limited. Minimum combined whole/local
round-trip Dice is 0.863/0.890. Under 26-neighbor model-grid connectivity, baseline
largest-component fractions range from 0.980 to 0.997; native references remain
fully connected. Sampling fidelity is a diagnostic, not a strict learned-model
accuracy bound.

Corrected original-weight evaluation 7292994 and paired fixed-budget retraining
7292995 are submitted. Each training model starts fresh from the original
checkpoint and receives 2,000 updates. Report 7293002 depends on both arrays;
two-model presentation 7293009 follows the completed report. New model scores
are pending until those jobs finish. The final reader-facing report will be
`outputs/experiments/aorta/pretrained-finetune-branches-r3/review-pretrained-finetuned-r1/report.html`.
Only Pretrained/Fine-tuned appear there, with the prior three-column CT style.
The signed resolution review and submission receipts record the corrected
target, class group, input identity and job chain.

An explicitly partial `coverage-preview-r1` audit uses the first three completed
held-out families (10-year, 15-year, one-year): six cases and 78 matched scans.
With the corrected identical foreground group and native reference frames,
mean connected-branch recall is 0.3394 pretrained / 0.6167 sibling fine-tuning;
core recall is 0.6701 / 0.8833. Branch recall by family is 0.4674/0.7646,
0.4605/0.7883 and 0.0904/0.2972, respectively. This verifies increased positive
branch coverage after correction. It does not measure specificity, isolate the
effect of siblings from adaptation, or replace the pending complete 130-scan
evaluation. Young-child branch coverage remains limited.
