# Recipe-v8: growth outside the selection ROI

Captured input: [recipe-v8/input.ini](../../../outputs/studies/thoracic-aorta/recipe-v8/input.ini).
Report: [recipe-v8/report.html](../../../outputs/studies/thoracic-aorta/recipe-v8/report.html).
Editable input: [thoracic-aorta-curvature.ini](../../../configs/studies/thoracic-aorta-curvature.ini).
Methods: [selection and growth regions](../../SELECTION_GROWTH.md).
Evidence: [growth-v8-evaluation.json](growth-v8-evaluation.json).

The starting point was the user's modified curvature INI, retaining its artery
category with blank `source_ids`, original tube coordinates and radii, 28 mm
crop width, stride 3, curvature settings and all tissue resistance factors.
The normalized input differs only by adding `[recipe] roi_role=selection` and
choosing the fresh output directory `recipe-v8`; explanatory comments were updated.

| Step | Operation | Parent selection | Distance per pass | Local Gaussian window | Passes |
|---|---|---|---|---|---|
| arch_outer_expand | Outward dilation | 3–20% | 9 mm | 0.1–0.8 | 1 |
| arch_inner_narrow | Inward erosion | 25–35% | 7 mm | 0.3–0.9 | 1 |

Step names were preserved as working labels. Both use shape concentration 6,
180-degree full angular sectors, and sequential processing. Their original
selection intervals are disjoint, so the second region does not inherit the
first region's ancestor displacement budget. Existing crop context is sufficient:
required halos are 13 mm and 11 mm beyond the corresponding original envelopes.

## Comparison using the same parameters

The comparison uses the user's revised parameters in both modes. It does not
compare against the earlier v7 trial's different distances and ranges.

| Quantity | Previous boundary role | New selection role |
|---|---:|---:|
| Added target voxels | 690 | 1,710 |
| Added outside the original main ROI | 0 | 1,020 |
| Removed target voxels | 0 | 0 |
| Blocked dilation proposals | 16 | 102 |
| Unresolved erosion proposals, preserved | 35 | 35 |

The native grid is 1 mm isotropic, so these voxel counts also equal volumes in
mm³. The new dilation starts with 9,375 original target voxels inside an 11,726
voxel selector. Its automatically derived footprint contains 50,981 voxels.
It accepts 1,710 of 1,812 dilation proposals. The growth can be seen beyond the
orange selector and inside the dashed blue edit footprint in the per-pass slices.
The final main selected lineage increases from 52,576 to 54,286 voxels, including
offspring outside that selector.

Every addition inherits a selected original target ancestor. Unselected current
artery voxels stay unchanged and cannot seed or carry growth. Sixteen additions
contact eight unselected target voxels, all carrying the catalog name
`dias_aorta`. A shared catalog name does not establish a specific anatomical
relationship; contacts are reported rather than interpreted as validated branches.

## Why this erosion is still an identity result

The second step selects 4,946 original target voxels. Its distance/profile/
direction produce 41 release requests before target stiffness and 35 after it.
None of those 35 can be reassigned under the configured recipient search.

The non-target recipient shell comprises 44 `chest_surface` voxels. This label
has coarse category `soft_tissue` but a fine skin classification, so the existing
precedence rule gives it stiffness 0.95. With a 3 mm recipient search, its initial
penalty is 2.85 mm and its remaining reach is 0.15 mm, below the first 1 mm edge.
The other neighboring voxels are remaining target artery, which cannot serve as
non-target recipients. Unresolved voxels therefore retain their labels and scalar
values. No skin factor or classification was changed to force erosion.

The growth-region change removes ROI clipping; it does not resolve the separate
skin-classification question. These interior-adjacent `chest_surface` voxels need
anatomical/material review before being treated as physical skin or having that
protection changed.

## Verification

- 289 tests passed. New cases cover growth across the ROI wall, repeated passes,
  later erosion of outside-ROI offspring, multiple original roots with partially
  overlapping selectors, same-ID neighboring arteries, empty original selections,
  cumulative context checks, and conservative overlap rejection before editing.
- The full source crop is preserved. Independent read-only memory mapping matches
  captured original labels and attenuation; source file sizes and modification
  times remain unchanged. Full source-volume hashes were not calculated.
- All 34 manifest entries and 19 generating-code hashes match. The captured INI
  matches the updated editable input; all starting numerical and tissue settings
  were preserved. The three other previously modified INIs remain unchanged.
- Per-pass labels, scalars and ancestor indices form an exact chain. Every accepted
  addition copies a selected input seed's label/scalar and inherits its original
  ancestor. The final selected mask includes surviving ancestors and offspring
  outside the original main ROI.
- Independently computed physical envelopes match the saved edit footprints.
  Actual changes have positive, reliable directional support and remain inside
  those footprints. Stiffness-1 barriers remain unchanged. All outside-selector
  counts agree with the native saved masks.
- Static per-pass slices and the final before/after surface overlay were inspected.
  Report tests verify the selection/growth descriptions, area accounting, image
  embedding and visible outside-ROI metrics. No new interactive WebGL pixel
  validation was performed for this trial.

For another trial, update the same INI and choose a fresh output directory such
as `recipe-v9`. The original selector can stay narrow; crop context, actual tissue
resistance and selection of the intended vessel surface still matter.
