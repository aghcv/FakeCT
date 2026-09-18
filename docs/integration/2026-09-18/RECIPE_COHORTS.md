# Recipe-based CoA cohort integration — 2026-09-18

The new `fakect.recipe-study/1` schema combines ordered named edits with a
bounded Cartesian sweep, metadata validation, native preflight, paired export,
and the existing TensorFlow fitting interface. The complete
[cohort guide](../../RECIPE_COHORTS.md) includes commands and target semantics.

## Preserved reviewed recipe

The starter [thoracic-aorta-cohort.ini](../../../configs/studies/thoracic-aorta-cohort.ini)
is a separate copy of the user's current curvature recipe. It retains the main
tube, 33 mm crop margin, category-based artery selection, centerline settings,
15–55% dilation interval, 25–55% inner-arch interval, 65–75% CoA interval,
Gaussian profiles, directions, distances, iterations and stiffness values.
The inner-arch safeguard stays disabled; CoA retains its user-set 10% local
volume floor and connectivity check. All four existing user-edited INIs were
preserved byte-for-byte.

The only necessary edit-policy difference is enabling normal tissue assignment
for both erosions. Diagnostic released markers carry placeholder attenuation
and remain invalid for paired image/mask exports.

The grid requests outer expansion at 3, 6 and 9 mm crossed with CoA narrowing
at 2, 4, 6, 8 and 10 mm: 15 variants plus an unchanged baseline. The inner arch
remains fixed at 7 mm. Every variant starts from the original phantom; steps
within one variant execute sequentially. Parameter axes use explicit lists or
inclusive decimal start:stop:step syntax. No guard is enabled by sweeping.

## Normal-reassignment pilot

The [native pilot audit](cohort-reassignment-pilot.json) compared the exact
current diagnostic recipe with normal reassignment using unchanged stiffness.
It verified the saved native recipe-v12 crop against the current input/catalog,
source size/mtime, crop hashes, selected labels and ROI mask. The crop is
168 × 126 × 87 k,j,i voxels at 1 mm spacing; source channels are each 4.185 GB.

| Edit / mode | Accepted budget (mm) | Proposed removal | Applied removal | Unresolved |
| --- | ---: | ---: | ---: | ---: |
| Inner arch, diagnostic | 7 | 754 | 754 | 0 |
| Inner arch, normal assignment | 7 | 754 | 59 | 695 |
| CoA, diagnostic | 5 | 1,418 | 1,418 | 0 |
| CoA, normal assignment | 10 | 2,390 | 845 | 1,545 |

The chest-surface label is currently treated as skin with stiffness 0.95.
At a 3 mm recipient search this leaves 0.15 mm reach, below one native voxel
step. Thus normal assignment can substantially reduce erosion, independently
of the manual safeguard. Normal CoA recipients in this trial were muscle
(136 voxels), lung (592), and esophagus (117). Existing atlas identities were
recorded as observations; this work did not relabel chest surfaces or loosen
bone/skin constraints. Fixing and validating that tissue interpretation remains
a geometry/image-realism milestone before scaling the cohort.

## Target and validation scope

The original selected target contains 52,576 artery voxels: 50,485 labeled
`dias_aorta` and 2,091 mapped to other artery/branch labels, including coronary,
arch and some pulmonary artery voxels. `selected_lineage` tracks exactly this
reviewed selection and its surviving descendants. It does not make a complete
or anatomically exclusive thoracic-aorta annotation. Binary masks, original
labels, the catalog and ancestry are all available for inspection/refinement.

Image arrays use the current attenuation-copy proxy in inverse centimetres;
they are not HU, learned background recovery or physical CT reconstructions.
All scenarios share one source anatomy. Whole realized geometry groups stay
together in train/validation/test, and baseline-equivalent variants remain in
training. Scenario holdouts cannot measure unseen-anatomy generalization.

## Implementation verification

All **378 tests passed** (`python3 -m unittest discover -s tests`). New checks
cover strict decimal grids and limits, deterministic scenario identities,
recipe halo validation, native preflight without pairs, immutable source reuse,
selected ancestry outside the original ROI, per-step safeguard independence,
failed-variant reporting, duplicate-mask grouping, complete manifest publication,
and loading the paired archives through the existing patch data loader without
TensorFlow fitting. Legacy study inputs and their workflow remain covered.

Metadata validation passed all 16 requested combinations without reading source
voxel payloads. Native execution and paired-export results are recorded below
and in the [companion cohort audit](cohort-artifact-audit.json). No model fitting
or GPU job was launched by this integration evaluation.

## Native grid outcome

All **16/16** planned cases completed successfully in native preflight, with
**16 distinct target masks**, no duplicates and no failures. Final geometry-group
splits are **10 training / 3 validation / 3 test**, including the unchanged
baseline in training. The metadata-only planned split counts were provisional;
actual membership was assigned after native mask hashes were known.

| Requested dilation (mm) | Added artery voxels |
| ---: | ---: |
| 3 | 77 |
| 6 | 318 |
| 9 | 635 |

| Requested CoA erosion (mm) | Removed artery voxels | Unresolved proposals |
| ---: | ---: | ---: |
| 2 | 290 | 106 |
| 4 | 631 | 449 |
| 6 | 751 | 935 |
| 8 | 774 | 1,303 |
| 10 | 845 | 1,545 |

The fixed inner-arch edit removes 59 voxels and leaves 695 proposals unresolved
in every edited case. All CoA distances pass the configured safeguards with
normal assignment; none needs distance backoff in this grid. Achieved removal
still differs substantially from the requested proposal because of recipient
eligibility and stiffness. Source voxels are 1 mm³, so these counts also equal
physical volumes in mm³.

Review the [native preflight report](../../../outputs/studies/thoracic-aorta/cohort-preflight-v1/preflight.html),
[per-pass CSV](../../../outputs/studies/thoracic-aorta/cohort-preflight-v1/preflight.csv),
and [full JSON audit](../../../outputs/studies/thoracic-aorta/cohort-preflight-v1/preflight.json).

The [distance-response chart](cohort-distance-response.png) ([vector PDF](cohort-distance-response.pdf))
plots these achieved changes and the final selected-target volume for all 15
edited cases. Reproduce it with:

```bash
python3 scripts/evaluation/plot_cohort_response.py \
  --preflight outputs/studies/thoracic-aorta/cohort-preflight-v1/preflight.json \
  --output docs/integration/2026-09-18/cohort-distance-response.png
```

## Completed paired export

The [dataset manifest](../../../outputs/studies/thoracic-aorta/cohort-pairs-v1/dataset-manifest.json)
indexes **16 paired native crops**, with **10 training / 3 validation / 3 test**
cases. The complete output occupies approximately 23 MiB on disk. Its data
fingerprint is identical to preflight, and all image hashes, mask hashes,
per-step outcomes and split assignments match the corresponding preflight case.
The completed directory has no `INCOMPLETE` marker.

The [artifact audit](cohort-artifact-audit.json) verified all nine preflight
artifact hashes and 26 dataset artifact hashes. Every pair has finite float32
attenuation, a binary uint8 target matching selected ancestry, complete labels,
and changes confined to the recorded edit region. Source file metadata and all
four original user-edited INIs remain unchanged. The existing patch loader read
a training batch of shape `4 × 64 × 64 × 1` without fitting a model or using test
payloads for fitting. Engineering integrity checks inspected all exported pairs;
they were not model-performance evaluations.

The first CLI export process terminated with exit code 143 after 13 successful
cases, without a reported simulation failure. Its partial output remains at
`cohort-pairs-v1-interrupted-20260918T1851`, marked `INCOMPLETE` and without a
dataset manifest. A fresh run called the same `prepare_recipe_dataset` backend
using the frozen input, retaining its full metadata/native validation while
avoiding the CLI's duplicate planning pass. It completed all 16 cases in
376.6 seconds. No partial cases were reused in the completed dataset.
