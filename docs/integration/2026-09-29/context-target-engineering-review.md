# Context aorta targets: engineering review, 2026-09-29

The current targets are suitable for an **engineering comparison of baseline-only versus baseline-plus-sibling training**, with the same provisional target definition in both arms. This review found no evident wrong-vessel selection, empty target, crop-edge truncation, or residual restriction to the edit ROI that should block that comparison. It does **not** establish clinically correct aortic boundaries or expert anatomical approval. `anatomical_extent_reviewed` remains `false` throughout the immutable dataset.

## Evidence reviewed

- Dataset: `outputs/datasets/coa/context-aorta-r1/dataset-manifest.json`, SHA-256 `3ec87c228c482691df413ad6afc35380498271dd5bcd67af510353ec6571b0e5`: 12 cases, 192 baseline/sibling scenarios.
- [Native attenuation overlays and interactive surfaces](../../../outputs/reports/context-aorta-r1/anatomy/index.html), including all 12 `slices-<case>.png` images and their `review-provenance.json`.
- Each case's immutable `anatomy/selection.json`, native `mask.npz`, ACT crop, and baseline row metadata.
- [Native target checks](../../../outputs/qa/context-target-review/native-target-checks.json): a new, read-only verification against the above artifacts. The only new outputs are this review and its QA record; no targets, dataset status, or source phantoms were changed.
- [Production sampling guard](../../../outputs/qa/context-aorta-r1-production-grid-guard.json) and [loader sampling parity](../../../outputs/qa/context-aorta-r1-loader-grid-parity-r2.json), retained as separate evidence for the 0.6 mm training grid.

## Objective checks

All 12 baseline targets have one face-connected native component. Their core and branch masks are disjoint and their union exactly equals the selected target. Native NPZ, selection manifest, report image and recorded ACT crop hashes agree with provenance; selected voxel counts agree across the report, masks and selection records. Every core voxel reread from ACT belongs to the atlas aortic anchor, `dias_aorta` (ID 2922); all included branch IDs have eligible systemic, coronary or cerebrovascular atlas paths. No selected label has a pulmonary hierarchy path.

The automatic crop leaves at least three empty voxels between target and every array face. The target extends beyond the previous editing crop in **every** case: 9.7–18.7% of the baseline target lies outside that crop. This directly confirms that the new segmentation target is not bounded by the smaller edit crop. It does not assert that every voxel in the entire biological thoracic aorta is correctly classified.

The native core retains the largest aortic component within the explicit landmark slab. The excluded secondary aortic islands contain only 0–5 voxels per case. Proximal branches account for 6.1–9.2% of target voxels. There is no empty or unexpectedly branch-dominated target.

For all cases, the inferior slab coordinate equals the midpoint of the recorded T12 surface minimum/maximum z coordinates; the superior coordinate equals the maximum T1 surface z coordinate. All target voxel centers lie inside those bounds. Branch reach is three times the median verified aortic RAW ring radius, ranging from 7.15 to 22.13 mm. The crop is computed from the anatomical surface bounds, slab and branch reach; old ROI bounds and selected IDs are explicitly ignored when the surface inventory cache is reused.

| Case | Provisional family | Core voxels | Branch voxels | Target volume, mL | Slab z, mm | Outside old edit crop |
| --- | --- | ---: | ---: | ---: | --- | ---: |
| 260602 | adult | 66,708 | 5,892 | 72.600 | 1219.1–1502.6 | 11.7% |
| 260611 | adult | 50,103 | 3,493 | 53.596 | 1120.7–1390.5 | 10.7% |
| 260612 | infant | 51,608 | 4,352 | 3.581 | 307.4–407.5 | 16.7% |
| 260613 | 1yr | 78,609 | 6,012 | 10.578 | 464.3–621.6 | 18.7% |
| 260614 | 5yr | 85,168 | 7,259 | 19.964 | 763.8–941.4 | 18.6% |
| 260615 | 10yr | 66,297 | 6,713 | 37.381 | 974.6–1186.7 | 9.7% |
| 260616 | 15yr | 58,876 | 4,587 | 46.265 | 1138.5–1425.7 | 15.6% |
| 260617 | infant | 51,717 | 4,327 | 3.587 | 307.4–407.5 | 16.7% |
| 260618 | 1yr | 78,609 | 6,012 | 10.578 | 464.3–621.6 | 18.7% |
| 260619 | 5yr | 85,168 | 7,259 | 19.964 | 763.8–941.4 | 18.6% |
| 260620 | 10yr | 66,297 | 6,713 | 37.381 | 974.6–1186.7 | 9.7% |
| 260621 | 15yr | 46,109 | 2,986 | 35.790 | 1070.7–1362.9 | 17.2% |

Voxel counts are not directly comparable between ages because native spacing varies from 0.4 to 1.0 mm. Physical volumes in the table account for this spacing.

## Visual engineering observations

All 12 tri-planar overlays place the selected core on a consistent course from the central cardiac/ascending region through the arch and along the descending vessel anterior to the spine. Proximal branch patches are visible near the arch and cardiac root region. The small structures sharing blood attenuation around the heart and lungs are not indiscriminately included. No gross displacement of the masks from vessel-shaped attenuation regions was evident.

One fixed sagittal or coronal plane can miss a curved segment even when the three-dimensional component is continuous. Apparent gaps within an individual view therefore should not be treated as a missing ascending segment; the independent native connectivity and core-label checks support the broader interpretation. These views nevertheless cannot establish exact annular/root coverage, fine ostial correctness, or complete coverage of every named branch.

## Boundaries and uncertainty retained

1. **Inferior thoracic cutoff:** T12 midpoint is a reproducible approximation. It is not a traced diaphragm or aortic hiatus. The target can include an abdominal-adjacent segment or omit a distal thoracic segment depending on anatomy. Both comparison arms must retain this same definition. A diaphragm-based policy would require a new target revision.
2. **Aortic root:** The proximal boundary is inherited from the generator's `dias_aorta` label. This review found a plausible selected cardiac/ascending region, but did not independently identify the valve annulus or prove complete sinus/root extent.
3. **Branch endpoints and names:** Branch inclusion uses face connectivity and Euclidean distance to the selected core, not along-vessel length. It includes proximal coronary as well as arch-region branches. Generic `arteries` IDs remain generic; the thoracic occurrence of the atlas name `internal_carotid_right` should not be presented as independently verified named-branch anatomy. Contact is geometric evidence, not proof of physiological connection.
4. **Orientation/registration:** Native i/j/k coordinates are preserved. RAW/BIN alignment rests on the content-identified generator and parameter-file contract. All frozen cases have equal x/y dimensions, so bounds alone cannot independently rule out an x/y permutation. The reviewed overlays support coarse alignment but do not replace independent registration or laterality validation. Do not add clinical left/right labels on this evidence alone.
5. **Sampling:** The existing 0.6 mm audit passes all 192 scenarios under its documented volume and connectivity thresholds. This does not mean every resampled target is perfectly one-component: the three previously problematic infant scenarios retain tiny secondary fragments, approximately 1.08 and 0.216 mm³, while the main vessel remains connected under the guard. At 1 mm those same cases have material breaks, so the engineering comparison should keep 0.6 mm sampling.
6. **Independence:** Three pairs have identical native target-array content: 260613/260618, 260614/260619 and 260615/260620. The two infant targets are also very similar. Twelve case IDs are therefore not evidence of twelve independent anatomical origins. Keep the existing conservative family grouping, all sibling variants and related baseline cases in one partition; report the number of held-out families alongside case/scenario counts.

The automated and visual evidence supports proceeding under this stated target policy. Target review does not establish model generalization, disease realism, independent clinical accuracy, or the hypothesis that digital siblings improve clinical performance. The baseline-versus-sibling comparison should use matched held-out family assignments, model settings and training-update budgets, and describe these targets as provisional atlas-derived thoracic aorta/proximal-branch masks.
