# FakeCT project outline: controlled anatomy to anatomy-specific AI

**First study:** thoracic aorta segmentation across synthetic aortic arch
hypoplasia, coarctation-like narrowing, and combined geometries.  
**Date:** 2026-09-17. **Development branch:** `fakect.26.09.16`.  
**Status:** the first staged training-study prototype now supports aorta preview,
parameter planning, paired-data preparation and a 2D segmentation trainer.
Start with the [practical study guide](TRAINING_STUDIES.md) and
[aorta INI](../configs/studies/thoracic-aorta.ini). The actual aorta input remains
unreviewed: only its preview is generated; no aorta cohort or GPU training has
been launched. The broader multi-anatomy benchmark, `[test]` and `[generate]`
methods below remain planned work.

## 1. Objective and first research question

Build one reproducible workflow that edits a labeled phantom, creates paired
image/segmentation datasets, trains anatomy-specific models, evaluates them on
reserved virtual anatomies, and eventually applies validated models to new scans.

The first question is: **can controlled variation in thoracic aortic geometry
improve attenuation-to-aorta segmentation, including localized and extended
narrowing, on anatomies and scenarios excluded from model development?**

The first deliverable is a synthetic-domain research benchmark. Phantom labels
give exact identity relative to the simulated anatomy, but anatomical realism,
image realism and population representativeness must be evaluated separately.
Generating more variants does not create more independent subjects.

## 2. Four methods with one configuration framework

| Method | Responsibility | Main outputs |
| --- | --- | --- |
| `[edit]` | Apply a reviewed operation or an explicit ordered recipe to a phantom. Start with erosion/dilation; later add combined lesions, bifurcation operations and other geometric models. | Original and edited labels, changed-voxel masks, image-generation provenance, achieved geometry and an HTML review report. |
| `[train]` | Plan a virtual cohort, call the edit/image-generation stages, construct image–target-mask pairs, freeze their split manifest, and train selected TensorFlow segmentation models. Permit dataset preparation without training and reuse of an existing frozen dataset. | Versioned paired dataset, split/QA manifests, training logs, checkpoints and a model metadata record. |
| `[test]` | Evaluate frozen models on predefined held-out scenarios and anatomy families. Compare errors by anatomy, lesion and imaging condition. | Per-volume and per-region metrics, failure cases, uncertainty summaries and comparison reports. |
| `[generate]` | Run a selected, validated segmentation model on new attenuation/CT volumes and export predictions in the source geometry. | Probability volume, binary segmentation, preprocessing record, quality flags and inference report. |

Here, `[generate]` means **generate predicted segmentations**, while synthetic
cohort generation belongs to the data-preparation stage of `[train]` and reuses
`[edit]`. This keeps model inference distinct from creating new phantom anatomy.

Retain the current readable INI convention: one parameter per line, a short
`NOTE` reference beside it, and detailed numbered comments below. Reuse the
existing study, input, selection, ROI, preview and output sections. Add explicit
cohort, split, normalization and model settings when their implementations exist.
An explicit method selector should run one method; merely including several
sections must not silently run all of them.

Current `fakect.edit/1` supports one crop-level edit. A future schema should
reference an ordered, human-editable recipe for multiple operations, including
overlap rules and operation order. Every independent cohort member starts from
the immutable source; operations within a combined-lesion recipe act in the
declared order. Preserve existing preview/edit inputs through versioned adapters.

## 3. What we can reuse and what must be added

| Available now | Needed for this project |
| --- | --- |
| Audited XCAT sources, signed organ labels and DPI atlas tissue grouping | A reviewed thoracic-aorta selector, anatomical landmarks and complete target coverage |
| Sphere/tube ROIs, native slice views and portable HTML reports | Aorta-specific centerlines, vessel-normal area measurements and lesion-specific QA |
| Bounded erosion/dilation, spatial Gaussian profiles, protected tissue rules and deterministic reassignment | Ordered multi-operation recipes, geometry calibration and cohort sweeps |
| Original/edited crop labels, transition masks and a separate attenuation-copy proxy | Accepted image formation/reconstruction for training pairs and consistent volumetric export |
| Staged paired-data prototype, 2D segmentation baseline and Wahab TensorFlow 2.17 launch wrapper | GPU validation, independent-anatomy evaluation and a production inference path |

The existing `fakenoise.py` is a **mask-to-image reconstruction** experiment, not
an image-to-mask segmentation trainer. Its audit also found train/validation
case overlap and shape-based exclusions. Reuse its cluster launch experience,
but implement an explicit segmentation task and new manifests rather than
relabeling its reconstruction metrics as segmentation performance.
See the [training audit](integration/2026-09-16/TRAINING_AUDIT.md).

## 4. First case and anatomical definition

Use **case 260602, frame 1 as the proposed engineering pilot**, since its source
geometry and current crop workflow have already been exercised. Source label
2922 (`dias_aorta`) is now located through native scans and corroborating
dictionary/atlas/surface evidence; see the
[localization record](integration/2026-09-17/aorta-localization.json).
The provisional report covers an arch crop, with the complete defined target
inside that crop shown separately from the edit tube. Target scope, landmarks
and ROI still require anatomical review; generic artery contacts are not
automatically included as aortic continuations.
The available cases are 260602 and 260611–260621 under
`/home/aghorban/slurm/xcat/`; use their `.par`/log metadata and matching non-averaged
`act`/`atn` volumes. A later age-specific study may require a different seed case.

The first anatomy review should:

1. Resolve aortic source IDs through the DPI hierarchy, source dictionaries and
   case/frame-specific surface evidence. Review generic artery labels and signed
   labels; do not assume the coarse `artery` category isolates the aorta.
2. Define the target precisely: lumen versus outer vessel envelope, included
   ascending/arch/descending portions, proximal/distal endpoints, and branch
   inclusion/exclusion. Confirm that the XCAT labels support that definition.
3. Mark native geometry, centerline, arch/isthmus landmarks and branch origins;
   verify orientation and spacing before exporting physical measurements.
4. Save one reviewed aorta selector and ROI input, with source/atlas hashes and
   a baseline report. Keep this pilot anatomy out of the final unseen-anatomy test.

The eventual pulmonary-artery task should use the same machinery with a different
reviewed anatomical selector. Existing DPI records include case-scoped corrections
where names such as `arteries_lung` refer to internal thoracic arteries; a filename
or coarse tissue name alone is not a reliable pulmonary target definition.

## 5. Synthetic lesion families and controlled experiments

Start with four study families, defined as **synthetic geometry recipes** rather
than diagnosed patient phenotypes:

| Family | Initial geometric construction | Parameters and checks |
| --- | --- | --- |
| Baseline/control | Original anatomy plus explicit no-op variants | Exact identity, unchanged branches, distinct imaging realizations if used |
| Arch hypoplasia-like geometry | Extended narrowing over a reviewed arch segment | Segment extent, smooth entry/exit, achieved area/diameter profile, branch preservation |
| Coarctation-like geometry | Localized narrowing at a reviewed location | Location, lesion length, concentration, achieved minimum area and residual connectivity |
| Combined geometry | Extended narrowing plus a localized lesion | Explicit operation order, shared/overlapping support, cumulative effect and branch QA |

Dilation can provide controlled enlargement and broader geometry variation.
Later versions can add eccentric narrowing, branch-origin/bifurcation lesions,
and wall-aware operations. A symmetric erosion is a first approximation; it does
not reproduce all wall remodeling, eccentricity or developmental anatomy.

The existing `distance_mm` is a spacing-weighted six-neighbor grid distance.
Calibrate requested changes against achieved geometry using centerline-normal
cross-sections. Record `A(s)`, minimum area, lesion length and, where useful,
equivalent diameter `D_eq(s) = 2 sqrt(A(s)/pi)`. Store ratios to the unchanged
source at corresponding positions. These are geometric descriptors, not clinical
diagnostic thresholds. Establish admissible ranges through anatomy review before
large sweeps; do not infer clinical severity directly from an erosion setting.

Vary anatomy, lesion geometry and image appearance as separate experimental
factors. Balance the four families for an initial controlled study, and record
that this balance is an experimental choice rather than a population prevalence.

## 6. Paired dataset and image formation

For each accepted variant, store:

- **Image:** edited-anatomy attenuation or simulated reconstructed CT, with units,
  image-generation method, acquisition assumptions and normalization metadata.
- **Target:** binary mask for the complete defined thoracic-aorta target within
  the exported field of view, derived from the final anatomical labels.
- **Evidence:** original and edited fine labels, edit support, transition masks,
  unresolved/blocked counts, geometry measurements and HTML review.
- **Identity:** variant ID, source case/frame, base-anatomy-family ID, parent
  phantom/model identity, recipe/seed, atlas/policy/config/code hashes and split.

The binary target must include unchanged aorta outside the lesion. The edit ROI
and changed-voxel mask are separate artifacts and must not become the aorta mask.
Record every image/mask transform and preserve their exact alignment. Export
native geometry through a suitable volume format and metadata; use bounded patches
for training, with a fixed field-of-view policy and full-volume reconstruction for
evaluation. Never provide a ground-truth aorta mask or lesion ROI as an input to a
segmentation model unless explicitly studying a prompted task available at inference.

If the pilot uses crops positioned from known aorta labels, describe its task as
segmentation within a supplied ROI. Full-volume inference requires a localization
or tiling procedure that does not use the held-out target mask; a manual ROI is
also possible if it is explicitly part of the intended inference workflow.

Image fidelity is a prerequisite for useful segmentation training:

1. **Engineering baseline:** use the existing attenuation-copy proxy only for
   pipeline/overfit checks, flagging it as a proxy. Do not count it as realistic CT.
2. **Research dataset:** establish a documented image-generation path, such as
   validated material/attenuation assignment with a forward acquisition and
   reconstruction model, or an available XCAT regeneration route that actually
   incorporates the edited geometry. Verify feasibility before choosing a route.
3. **Appearance experiments:** vary supported contrast, resolution, blur, noise,
   reconstruction and partial-volume conditions while retaining aligned labels.
   Noise on ideal attenuation alone does not establish CT realism.

If learned scalar recovery becomes part of the image generator, train/fix that
generator using development data only. Keep final test anatomy out of its training
as well, and record generator version in every segmentation pair. Evaluate
sensitivity to image-generation artifacts and avoid a segmentation benchmark that
merely learns a particular generator's boundary signatures.

## 7. Virtual population and leakage-resistant splits

Freeze base-anatomy lineage **before** creating variants, frames or patches.
Every descendant of the same underlying anatomy belongs to one partition.
Different case directory names are insufficient proof of independent anatomy;
audit their source phantom/model parameters and relationships first.

Use three explicitly different evaluations:

| Evaluation | Held out | What it establishes |
| --- | --- | --- |
| One-anatomy engineering pilot | Some lesion recipes/parameter combinations | Code and scenario behavior on a known base anatomy; not unseen-subject performance |
| Main synthetic benchmark | Entire base-anatomy families | Generalization across the available independent synthetic anatomies |
| Stress benchmark | Predeclared geometry ranges/combinations or imaging conditions | Robustness under specified synthetic distribution shifts |

Use training data for fitting and a separate anatomy-grouped validation partition
for architecture, threshold and hyperparameter selection. Compare candidates on
validation; freeze the chosen model and preprocessing before final test evaluation.
Do not repeatedly choose the best model from final test results.

A roughly 70/15/15 family split is an initial planning option only if the lineage
count supports it. With few independent families, use grouped development folds
and reserve whole families for a final test, reporting the small sample limitation.
If only one family is available, stop claims at the engineering/scenario pilot.
Stratify by age/size and anatomy where feasible; distinguish within-age from
cross-age performance. Report base families, variants and image realizations as
separate counts, and aggregate uncertainty at family level rather than treating
slices or descendants as independent patients.

Begin with a manageable deterministic pilot, for example **32 geometry variants**
across the four families with one imaging realization each. This is a proposed
debugging budget, not a statistical sample-size justification. Expand only after
QA, storage/runtime measurement and a learning-curve study.

## 8. TensorFlow segmentation experiments on Wahab

Extend the initial 2D segmentation model and bounded loader into a model registry.
Proposed comparison sequence: simple threshold/region-growing baseline, 2D U-Net, 2.5D U-Net using
physical through-plane context, and a compact patch-based 3D U-Net when memory
permits. TensorFlow provides a U-Net segmentation tutorial; the original 3D U-Net
paper provides a volumetric architectural reference. These are starting points
for this study, not evidence of performance on these aortic cases.
([TensorFlow segmentation](https://www.tensorflow.org/tutorials/images/segmentation),
[3D U-Net paper](https://arxiv.org/abs/1606.06650))

For the first learned baseline, use a binary output and a declared loss such as
Dice plus binary cross-entropy. Include vessel boundaries, hard negatives
(pulmonary vessels, veins and adjacent tissues), and target-absent patches;
evaluate full volumes so patch sampling does not inflate the reported performance.
Apply spatial augmentation identically to image and label, with appropriate
label interpolation, and specify context/patch dimensions in physical units.

The segmentation entry point and SLURM wrapper now use the confirmed
`container_env tensorflow-gpu/2.17` and `crun -p ~/envs/fakect` environment pattern;
the separate fakenoise wrapper still runs a reconstruction model. The initial
CPU synthetic-fixture fit/reload check is separate from demonstrating one GPU
batch, a tiny-set overfit, and GPU save/reload prediction agreement. Record
GPU placement, versions, host/GPU peak memory, throughput, seed and preprocessing.

Use bounded loading, shuffle sample metadata before expanding volume patches,
and tune caching/prefetch against measured memory. TensorFlow documents these
input-pipeline mechanisms; its example buffer sizes are not a resource budget
for XCAT volumes. ([TensorFlow data performance](https://www.tensorflow.org/guide/performance/datasets))

## 9. Testing, model comparison and acceptance

Each `[test]` report should include volumetric Dice, precision/recall, surface
distances in millimetres, and failures/empty predictions. Define empty-mask handling
and surface tolerances before evaluation. Report whole-target and lesion-region
results separately, plus arch/isthmus/descending regions where the target definition
supports them. Include continuity, false bridges, branch spillover, area-profile
error and minimum-area location error; overlap alone can hide a poor local lesion.

Compare baseline-only versus lesion-augmented training, uniform versus Gaussian
edits, and the candidate architectures under the same frozen data/split rules.
Use repeated training seeds when affordable. Measure per-family performance,
failure distributions and calibration if probabilities are used. Reserve recipe
metadata for evaluation strata; do not expose lesion parameters as model inputs.

Before training the main benchmark, publish acceptance criteria for geometry,
pair alignment, exclusions, segmentation quality and failure rates. Numerical
segmentation thresholds should be set after baseline/validation work and domain
review, before accessing final test results. Require transparent reporting of
rejected variants and unmet geometry targets rather than silently dropping them.

## 10. `[generate]`: later transfer to patient-specific images

Package each selected model with its target definition, label vocabulary,
supported modality/protocol, image units, spatial transforms, normalization,
decision threshold and tested input range. Reproduce exactly that preprocessing
at inference, predict with suitable tiling, and map probabilities and labels back
to the original volume geometry. Preserve the input and save predictions separately.

The XCAT audit describes `atn` values as attenuation per pixel. Their conversion
to comparable inverse-length units must use source metadata; this is not an
automatic conversion to patient CT HU. Establish scanner/protocol-aware units
and calibration where needed. Fit population normalization on training data only;
any per-scan normalization must be fixed and reproducible at inference.

Normalization is one component of transfer, not evidence that synthetic and
patient distributions match. While segmented patient data are unavailable,
patient predictions are exploratory outputs with quality flags, not validated
accuracy claims. Later acquire an independently annotated patient evaluation set,
separate any adaptation cohort from final testing, and assess performance across
relevant scanners/protocols and disease presentations. Other modalities require
their own image model, preprocessing, training/adaptation and external evaluation;
CT intensity normalization alone is not a cross-modality bridge.

## 11. Milestones and concrete deliverables

| Milestone | Deliverable | Completion evidence |
| --- | --- | --- |
| A0 — Aorta definition | Reviewed case/frame, target IDs, landmarks, centerline and baseline INI/report | Correct target coverage and exclusions; lumen/envelope meaning and source geometry documented |
| A1 — Geometry pilot | Baseline, extended narrowing, focal narrowing and combined recipes | Achieved profiles, connectivity/branch checks, protected-label invariants and repeatability; no-op exact |
| A2 — Paired-data pilot | Small versioned image/mask set with frozen lineage and QA manifest | Exact pairing, preserved labels/units/transforms, explicit image-fidelity status and measured resource use |
| A3 — TensorFlow baseline | Segmentation loader, baseline model and GPU launcher | One GPU batch, tiny-set overfit, save/reload agreement, bounded memory and no split leakage |
| A4 — Synthetic benchmark | Multiple independent families, frozen splits and model comparison | Predeclared held-out evaluation, regional/family metrics, failures and uncertainty reported |
| A5 — Reproducible inference | `[generate]` model bundle and native-geometry prediction export | Input/output roundtrip, fixed preprocessing and synthetic held-out inference tests |
| A6 — Patient/modality validation | Annotated external evaluation and any separate adaptation studies | Evidence supporting the specific intended patient/modality use; synthetic results alone are insufficient |

The practical dependency order is **A0 → A1/A2 → A3 → A4/A5 → A6**. Interface
scaffolding and resource profiling can proceed alongside anatomy/image work.
Do not scale cohort generation before the image-pair and lineage checks pass.

## 12. Immediate next work package

1. Review the located aorta and provisional ROI in case 260602/frame 1; choose a
   different case if its labels, resolution or intended age group are unsuitable.
2. Iterate the dedicated aorta INI/report and record the accepted anatomy/ROI,
   retaining fine anatomical identity.
3. Produce one control, one extended narrowing, one focal narrowing and one
   combined example; measure achieved geometry and review branch preservation.
4. Export a small aligned image/binary-mask set with its image-fidelity label.
5. Validate the segmentation baseline on Wahab after review, extending the CPU
   synthetic-fixture fit/reload check to a small accepted set and GPU overfit.
6. Audit all available phantom lineages and freeze the development/test allocation
   before the main virtual population is generated.

The implemented study input is `configs/studies/thoracic-aorta.ini`; its preview,
future prepared pairs and model outputs use separate versioned directories under
`outputs/studies/thoracic-aorta/`. The initial aorta dataset/model directories are
planned destinations, not generated cohorts or fitted models. Keep large
arrays/model weights outside ordinary Git tracking;
version the INIs, manifests, hashes and reports. Preserve the current dataset
while the synthetic benchmark is developed.

Existing implementation references: [ROI reports](ROI_PREVIEWS.md),
[bounded morphology](MORPHOLOGY.md), [atlas mapping](TISSUE_MAPPING.md),
[integration milestones](integration/2026-09-16/PLAN.md), and
[latest edit evaluation](integration/2026-09-17/MORPHOLOGY_EVALUATION.md).
