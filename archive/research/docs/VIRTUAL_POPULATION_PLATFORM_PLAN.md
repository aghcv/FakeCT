# Virtual population platform and CoA implementation plan

> Historical generated artifacts are recorded as paths below; they are not included in this source checkout.

This plan separates the reusable AI platform, the FakeCT generator and data
registry, and the CoA research study. The immediate deliverable is a reviewable
classification of the existing virtual cohort. The next software milestone is
a TensorFlow model that combines a low-resolution anatomical overview with
selected high-resolution regions under explicit memory limits.

**Date:** 2026-09-30. **Branch:** `fakect.26.09.16`.
**Status:** draft for scientific review. The label audit is implemented
separately; the multiresolution architecture below is proposed, not trained.
This document supersedes the immediate development priorities in the original
[thoracic aorta project outline](THORACIC_AORTA_PROJECT.md), whose earlier
results and implementation history remain available.

**Implementation update, 2026-09-30:** the label convention has been accepted
for an engineering pilot. A streamed overview loader, explicit cohort/annotation
join, family-preserving pilot plan, image-based bounding-box locator, and fused
coarse/fine classifier are now implemented. The original draft label artifact
remains unchanged. See [the pilot workflow](MULTISCALE_COA_PILOT.md) for commands,
coverage gates and current scope. Successful cohort learning and the later
population-benefit experiment are still to be established.

**First pilot outcome:** run 6866879 stopped before classifier training because
the locator missed every development target. Host training memory also grew
well beyond its request. The next milestone is revision-two localization and
runtime correction, preserving the same cohort, family partitions and patch
dimensions. See [the current pilot findings and commands](MULTISCALE_COA_PILOT.md).
The cohort-benefit experiment remains gated on successful multi-case learning.

## Three development directions

| Direction | Responsibility | Outputs and success criteria |
| --- | --- | --- |
| AI training platform | Consume immutable cohort references; load multiple resolutions; find relevant anatomy from images; train interchangeable TensorFlow task heads; evaluate accuracy, memory and runtime. | A reusable physical-coordinate loader, overview and detail network, model INI, learning checks, inference contract and reproducible accuracy versus compute benchmarks. |
| FakeCT generation and data organization | Edit phantoms, preserve source identities, assign labels and attenuation, store compact changes, export geometry and condition provenance, and freeze reviewed dataset revisions. | Shared source volumes, native crop overlays, optional pyramid changes, condition annotations and immutable manifests. Reconstructed inputs must match the explicitly composed reference. |
| CoA research study | Define the condition and severity targets, review labels and controls, choose family partitions and evaluation endpoints, and test the value of virtual siblings. | A versioned CoA label policy, a cohort containing positive and negative cases, anatomy and image robustness experiments, and a family-level research report. |

Keep these directions in one version-controlled development branch initially,
with separate modules, configurations and artifacts. The generator should not
depend on a trained model. A model experiment should reference a reviewed cohort
and label revision rather than silently regenerate anatomy. Task definitions
should not be embedded in the generic volume loader or U-Net implementation.

The AI platform itself is a product deliverable. Its contribution is efficient
access to related virtual anatomies, consistent geometry across resolutions,
bounded resource use, interchangeable tasks and auditable evaluation. CoA is
the first application used to test that platform.

## Current verified foundation

The current context manifest contains **192 scenarios from 12 phantom IDs**:
12 original references and 180 edited siblings. The IDs belong to six
provisional anatomy families. Related originals and all descendants must stay
in the same model partition. Generating siblings increases morphological
coverage without increasing the number of independent originating anatomies.

The data loader already combines a shared whole-body attenuation source with
a native edited crop and compact target masks. It does not store a separate
whole-body scan per sibling. Registry identities, hashes, physical coordinates,
bounded volume caches and family-aware split planning can be reused.

The completed single-resolution comparison produced empty test predictions in
both arms. The saved models also fail on selected positive training patches;
the earlier one-patch diagnostic still reproduces Dice 0.97787. This motivates
a stronger multi-case learning check before expensive experiments. It does
not establish whether synthetic siblings help a functioning model. See the
completed evaluation (`outputs/experiments/coa/context-comparison-r1/evaluation-review.html`).

## Immediate provisional CoA classification

The user defines all 12 original phantoms as non-CoA references for this study.
Record this as a study assumption, not an independent clinical diagnosis.
For the existing edited scenarios, a usable measurement is already recorded
by the CoA erosion safeguard:

```text
local_volume_loss = 1 - retained_target_voxels / baseline_target_voxels
```

Here, `baseline_target_voxels` is the selected target within the **positive
longitudinal support of the CoA step**, immediately before that step. It
includes the full selected cross-section before angular weighting and follows
earlier recipe edits. It is not the volume of the entire long aortic ROI.
For this cohort, CoA is the last step and runs once. All 180 records pass the
count, volume, ratio, accepted-step and connectivity consistency checks.

**Proposed review threshold:** `local_volume_loss >= 0.20` is benchmark-positive.
An edited scenario below 0.20 is benchmark-negative or below threshold; it must
not be described as unedited or clinically normal. Keep `coa_edit_applied`
separate from `coa_benchmark_positive` and retain the continuous loss value.
The 20% value is an explicit engineering convention chosen for discussion,
without optimizing any model outcome. It is not a minimum-lumen-area threshold
or a validated definition of CoA.

| Local volume loss threshold | Benchmark positive | Below threshold or reference |
| --- | ---: | ---: |
| 10% | 156 | 36 |
| **20% proposed** | **132** | **60** |
| 30% | 99 | 93 |

At 20%, the 60 negatives comprise 12 originals and 48 mildly edited scenarios.
All 12 phantom IDs have scenarios on both sides of the threshold. Fifteen
edited scenarios fall within the proposed **18–22% review band**; the binary
rule remains explicit, while these cases receive an additional review flag.
Changing the threshold creates a new label revision; no source data or frozen
cohort manifest is changed.

The separate label review uses
[coa-label-review.ini](../configs/studies/coa-label-review.ini). Its output
contains a checksummed label manifest, row-level CSV (`outputs/labels/coa/provisional-volume-r1/labels.csv`)
and HTML distribution/case report (`outputs/labels/coa/provisional-volume-r1/report.html`)
under `outputs/labels/coa/provisional-volume-r1/`. Missing or inconsistent
measurements must fail validation rather than silently become negative labels.
The label revision is a draft for discussion and is not authorization to launch
a long model run.

To validate this configuration and its source metadata from the development
worktree:

```bash
python3 scripts/research/evaluation/classify_coa_cohort.py \
  --config archive/research/configs/studies/coa-label-review.ini --validate-only
```

To export a revised threshold, change the threshold and choose a new output
directory in the INI, then run the same command without `--validate-only`.
Existing exports are immutable and cannot be overwritten.

This proxy measures the **amount of local geometry changed**, not the narrowest
lumen. The same volume loss can represent a long mild narrowing or a short
severe narrowing. Until true area and length measurements are available, any
regression target using it must be named `local_volume_loss`, not clinical
severity. Current curvilinear plots explicitly show voxel volume divided by
path-bin length; their mm² units do not make them vessel-normal areas.

## AI platform architecture

The proposed model observes the whole anatomy cheaply, then examines selected
regions in detail. Its primary image input remains attenuation. Original labels,
edit masks and disease metadata supply supervision and QA; they are not image
channels available to the deployed predictor.

```text
Shared native source + sibling image overlay
                  |
       Streamed image pyramid
                  |
     Low-resolution whole-body overview
                  |
      Broad anatomical locator
                  |
    Bounded high-resolution proposals
                  |
  Fine 3D features + aligned coarse features
                  |
 Task heads: CoA probability / location / scalar score
             or another anatomy-specific task
```

A lightweight overview encoder learns broad anatomy and physical context. A
locator proposes the thoracic aorta/arch region in both original and modified
scans. It should select anatomy likely to contain the target rather than zoom
only where it already predicts disease, since a subtle missed lesion would
otherwise receive no fine examination.

A compact fine-resolution encoder or U-Net processes a bounded number of these
regions. Spatial coarse features are sampled at the fine regions' physical
coordinates and fused with local features. A U-Net decoder can provide anatomy
or lesion-location supervision; pooled features and separate heads provide
scan-level classification and regression. Segmentation need not be the primary
research endpoint.

The first implementation should train the coarse locator separately, freeze
it, and then train the fine model. This avoids debugging a changing locator,
discrete patch selection and the detector simultaneously. Joint fine-tuning is
a later option. During evaluation and inference, proposals must come from the
image pathway. Atlas or edit coordinates may supervise locator training, but
must not select held-out patches. A broad, uniformly defined thoracic field of
view may be used for a declared early pilot; that is not yet whole-scan
automatic localization.

Multiresolution processing and coarse-to-fine cascades are established design
patterns: [DeepMedic](https://arxiv.org/abs/1603.05959) uses multiple spatial
scales, and the [official nnU-Net training guide](https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/how-to/train-models.md)
describes low-resolution and full-resolution cascade stages. Those sources
motivate this design; they do not validate the proposed FakeCT implementation
or imply that the architecture alone will resolve the earlier training failure.
Implementation remains in TensorFlow to fit the existing Wahab workflow.

## Resolution and memory budgets

Start with explicit, configurable limits rather than unrestricted adaptive
refinement. Candidate budgets for the pilot are an overview of at most roughly
two million voxels and two fine proposals processed in bounded microbatches.
Retain the verified 0.6 mm sampling for the first CoA fine-grid comparison;
profile alternative patch extents rather than changing resolution and task at
the same time. The earlier 216×160×160 block spans 129.6×96×96 mm and provides a
measured reference point, not a mandatory final architecture.

Choose overview spacing from physical scan extent and the voxel budget, retain
the resulting transform and spacing, and pad where needed. Do not silently
resize every anatomy into identical physical dimensions. A coarse grid is for
context and localization; thin-lumen measurements and fine supervision remain
native or at a separately audited resolution. Coronary applications will need
their own sampling audit; the CoA setting is not a universal vessel resolution.

The initial engineering target is to fit a 16 GiB GPU with a proposed peak
allocation budget of 12 GiB, leaving operating margin. Measure host memory,
activation memory, latency and native bytes read. Input-array size alone does
not bound neural-network memory. Profile coarse and fine branches separately,
then together; enforce the configured number of proposals and fallbacks.

The current `ContextVolume.read_block` reads the native bounding slab underlying
a requested block. Requesting a very coarse whole-body block can therefore
still allocate a huge native slab and exceed its 64-million-voxel guard.
Whole-body pyramid construction needs a new streamed downsampling path, with
an explicit anti-alias filter, bounded chunks and the complete filter halo.

## Shared data and correct reuse

Cache one base image pyramid per phantom and store or compute only affected
coarse tiles for each sibling. For a fixed linear filtering/resampling operator
`D`, `D(base + delta) = D(base) + D(delta)` permits linear decomposition of the
pyramid. Separately computed floating-point values need not be bitwise equal:
freeze numerical tolerances and verify them against full recomposition on
representative cases. Numerical image tolerance alone does not prove identical
network predictions. Include the full filter support around native changes.
Apply nonlinear clipping or normalization after composition, because these do not generally commute with
adding image differences. The overview must represent the final sibling, not
always the unedited source.

Cache keys must include source and overlay identity, physical grid, filter and
boundary policy, units, dtype and code version. Normalized/augmented caches
also need those transformations and random seeds. Coarse and fine inputs,
supervision and proposal coordinates must share the same physical augmentation.
Train-derived intensity statistics must be fitted within the training partition.

Preserve the distinction between three different kinds of reuse:

| Reuse | Initial policy |
| --- | --- |
| Native images, overlays and image pyramids | Reuse immutable inputs with complete provenance; do not duplicate whole-body siblings. |
| Learned features | Recompute while encoder weights change. Optional frozen deterministic feature caches must include the checkpoint hash and complete input/transform identity. Training gradients, augmentation and normalization state can make reuse invalid. |
| Predictions and evaluation statistics | The existing local-tile proof is insufficient after global context is fused. A sibling edit may change coarse features and therefore predictions in locally unchanged patches. Recompute unless every conditioning input is proven equivalent. |

Initially optimize image loading and pyramid composition; avoid complex
activation caching in joint training. This preserves correctness while still
reducing data storage and high-resolution compute. The new model must not
inherit the old evaluator's 14.6-fold validation reuse claim without a new proof
and benchmark.

## Generator and registry extensions

Preserve the current source-plus-native-crop format and registry. Add optional
pyramid references as derived, replaceable caches rather than replacing the
immutable source. Existing datasets and models remain usable under their frozen
implementations; develop new capabilities in new modules.

Condition-specific geometry needs more provenance than net before/after masks.
The recipe callback already exposes accepted per-pass changes. Export each
condition's additions/removals, pre-step reference and final support into a
versioned annotation revision. Net changes cannot recover edits that overlap
or cancel. Record clipping, protected tissues, safeguard backoff and unresolved
reassignment so model labels reflect achieved anatomy rather than requested
`distance_mm`.

Keep three independently versioned contracts:

1. **Cohort revision:** immutable baseline sources, recipes, image overlays,
   masks, native geometry, atlas and family provenance.
2. **Annotation revision:** source cohort hash, condition definitions, continuous
   measurements, binary thresholds and review decisions. Several studies may
   label the same images differently without copying those images.
3. **Model experiment:** cohort and annotation hashes, family partitions,
   preprocessing, architecture, budgets, seeds, optimizer and evaluation policy.

A new model input can reference the first two contracts. Its reader must join
on unique sample IDs and reject missing, duplicated or incompatible records.
Generic caching must not silently turn a draft annotation into a reviewed label.

## CoA research endpoint and controls

The first pilot asks whether attenuation can predict the **provisional local
volume-loss class and continuous value**. The eventual anatomical severity
endpoint should use achieved minimum lumen-area reduction and lesion length,
measured on reviewed centerline-normal planes. Validate the centerline itself:
an editing tube is not automatically a lumen centerline. Branch intersections,
other nearby vessels, clipped planes and near-resolution lumens need flags.

For condition attribution, retain both pre-CoA/post-CoA measurements and matched
CoA-on/CoA-off siblings with other edits fixed. This separates the marginal CoA
effect from arch narrowing that preceded it. The existing 192 images can
support the provisional label review now; true area measurements and additional
condition controls require later development and new artifact revisions.

All current altered scenarios combine nonzero CoA erosion, arch erosion and
dilation. Thresholding creates edited examples on both sides of a chosen loss
cutoff, but below-threshold cases still contain some CoA-like modification.
This is different from having truly CoA-off companion-anomaly controls.

The next generation revision should vary independently:

- CoA off/on and achieved strength, followed by length, position and asymmetry.
- Arch narrowing off/on and strength.
- Enlargement off/on and strength.

Include unedited references, isolated CoA, companion abnormalities without CoA,
and combined conditions. Start with a small reviewed panel rather than a large
factorial expansion. Use the same image-processing and reassignment policy
across classes; inspect whether edit boundaries or intensity patterns reveal the
label without the intended anatomy. Imaging perturbations should be applied
independently of the class.

## Separate the software and population experiments

First compare **overview only, fine detail only, and fused overview/detail** on
the same task, data, family partitions and optimization budget. Report task
accuracy alongside peak GPU/host memory, inference time, storage, native bytes
read, proposal coverage and fraction of the anatomy processed at high
resolution. Select the platform configuration using training and validation
evidence, and keep whole-pipeline failure rates visible when a locator misses.

Then fix the model and compare a **small class-complete cohort** with a
**sibling-expanded class-complete cohort**. Both need positives and negatives;
normal-only images are not a conventional supervised CoA-classifier baseline.
Match optimizer exposure, use the same ordinary augmentation, and add a
sample-count-matched diversity comparison when the claim concerns diversity
independently of dataset size. This prevents attributing an architecture change
or extra optimization to the virtual population.

Evaluate sensitivity/specificity and precision-recall behavior, calibration,
continuous-score error, location error, performance near the labeling boundary,
and response to companion abnormalities. Report proxy-volume error initially;
reserve area/diameter severity error for a separately validated geometric label.
Threshold sensitivity for the label policy is distinct from tuning a model's
decision threshold. Freeze both policies before held-out testing.

Keep families intact. The existing generic evaluation-design module supports
family holdout and leave-one-family-out/nested designs; the new multiscale runner
still needs to integrate them. A pilot holdout establishes basic learning.
Subsequent family cross-validation and repeated seeds assess variability, with
uncertainty summarized at family level rather than treating siblings as
independent subjects. Use inner validation for architecture/threshold choices.
Current data establish synthetic-domain evidence; clinical CT generalization
requires a later external clinical evaluation.

## Configuration and code boundaries

Maintain the readable INI convention: a short `NOTE` reference beside each
parameter and numbered explanations below. Separate generation, labeling and
model experiments. The label-review INI is implemented now. A future model INI
should expose the following sections; these are design names, not currently
accepted parser inputs:

| Proposed section | User controls |
| --- | --- |
| `[data]` | Cohort manifest and annotation revision |
| `[input.overview]` | Physical sampling policy and maximum overview voxels |
| `[input.detail]` | Fine spacing, patch extent and maximum proposals |
| `[proposal]` | Image-derived anatomical locator and coverage policy |
| `[model]` | TensorFlow encoders, feature fusion and task heads |
| `[task.coa]` | Class field, continuous score field and optional localization target |
| `[sampling]` | Case/class balance and foreground/background curriculum |
| `[cache]` | Image-pyramid cache and bounded loaded-volume count |
| `[budget]` | Host/GPU targets and microbatch limits |
| `[evaluation]` | Family design, seeds, metrics and fixed decision policy |

Likely new modules are `fakect_multiscale_volume`, `fakect_anatomy_proposal`,
`fakect_multiscale_learning` and a task-definition module. Reuse the current
registry, `ContextVolume` conventions and evaluation-design helpers through
explicit interfaces. Preserve code pinned by completed experiments.

## Milestones and completion checks

| Milestone | Direction | Concrete deliverable | Required check before advancing |
| --- | --- | --- | --- |
| M0 Review current labels | CoA study | Draft threshold INI, 192-row labels, histogram and per-case HTML | Counts and denominators verified; user reviews the meaning of a below-threshold edited case and the proposed cutoff. |
| M1 Freeze contracts | All three | Separate cohort, annotation and model schemas; physical input/proposal contract | Missing labels, split leakage, unknown inference inputs and incompatible revisions are rejected. |
| M2 Build multiscale data access | Generator and AI platform | Streamed base pyramid, sibling updates and aligned fine patches | Toy exact-composition tests; production float parity within frozen tolerances; anti-alias/halo tests; measured bounded memory. |
| M3 Demonstrate learning | AI platform | Small multi-case positive/negative pilot with a fixed overview and detail budget | Correct input/label overlays; train/validation metrics and probability diagnostics; working save/reload; no oracle evaluation crops. A one-patch fit alone is insufficient. |
| M4 Improve condition geometry | Generator and CoA study | Accepted-step annotations, reviewed normal-plane areas and independent condition controls | Geometry measurements match analytic phantoms and visual review; final image/target consistency passes; new labels remain versioned. |
| M5 Evaluate platform efficiency | AI platform | Overview-only, detail-only and fused comparison | Report both performance and compute on identical data; proposal misses counted; resource limits hold on largest/smallest cases. |
| M6 Evaluate virtual populations | CoA study | Fixed-model small-versus-expanded cohort study, then family CV | Both classes represented, families isolated, budgets matched, label and model thresholds frozen before testing. |
| M7 Reuse for another anatomy | All three | Coronary or another task plugin | New tissue/geometry labels, proposal coverage, fine resolution and image realism reviewed; reuse platform contracts without inheriting aortic severity thresholds. |

M0 is the current review deliverable. M1 and M2 can be developed in parallel
after the label discussion; M3 is required before any new long training job.
M4 can proceed alongside platform development, and its geometric endpoint is
required before claiming anatomical severity performance. M5 and M6 answer
different questions and should be reported separately.

## Decisions for the next discussion

Review whether the 20% local-volume-loss rule is useful for the initial software
pilot, whether the 18–22% band should be withheld or explicitly retained in the
first fitting set, and whether the first reported endpoint should remain that
proxy or wait for true cross-sectional measurements. These are scientific
choices to record in a new annotation revision, not permission to change the
frozen images. The shared low-resolution overview plus selective fine-detail
architecture is the proposed platform direction in either case.
