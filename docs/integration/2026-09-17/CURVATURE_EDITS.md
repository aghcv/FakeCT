# Recipe-v7: curvature frames and category-based artery editing

Input: [thoracic-aorta-curvature.ini](../../../configs/studies/thoracic-aorta-curvature.ini).
Output: [recipe-v7/report.html](../../../outputs/studies/thoracic-aorta/recipe-v7/report.html).
Methods and controls: [curvature editing guide](../../CURVATURE_EDITS.md).
Machine-readable evidence: [curvature-v7-evaluation.json](curvature-v7-evaluation.json).

Case 260602, frame 1, native 1 mm isotropic crop `(158,116,77)` in k,j,i order.
The parent centerline and radii are from the reviewed input captured in recipe-v6.
This new example selects `tissue=artery`, with `source_ids` blank and no original-ID
stiffness overrides. Existing user INIs were left untouched. Bone remains 1 and
skin 0.95; the other category factors remain permissive as configured.

## Achieved edits

Both steps use the same parent tube over 25–65% of original physical path length,
3 mm peak distance, one pass, Gaussian concentration 6, and a 180-degree full
angular sector. Outward dilation executes first, then inward erosion.

| Pass | Proposed | Achieved | Preserved / blocked |
|---|---:|---:|---:|
| Outer dilation | 990 added voxels | 583 added | 407 blocked |
| Inner erosion | 565 released voxels after target resistance | 49 removed | 516 unresolved, preserved |

All 632 actual changed voxels have positive angular weights, reliable local
directions, and lie in the requested arc interval and main ROI. Actual outward
changes span parent 43.05–58.78%; inward changes span 38.82–58.11%. The taper,
grid, ROI, and reassignment policy limit the achieved extent.

The union of artery labels has 27 six-neighbor components in the crop before
and after; within the main ROI it has 7 before and after. Most ROI voxels belong
to one component (52,569 before; 53,103 after), with seven other voxels distributed
among the remaining six components. Unchanged component counts do not establish
smooth junction geometry, anatomical tree identity, or absence of local defects.

## Branch coverage and limits

The artery category resolves to 644 catalog IDs; 99 occur in this crop. Nine
occur in the main ROI, and six occur in the selected edit interval. Counts below
are from the original phantom, with names reproduced from the catalog. They are
not independent anatomical adjudications.

| Catalog label | Main ROI voxels | 25–65% range voxels |
|---|---:|---:|
| dias_aorta | 50,485 | 19,349 |
| internal_carotid_right | 311 | 311 |
| arteries (first original label) | 265 | 265 |
| arteries (second original label) | 657 | 657 |
| right_pulmonary_arts | 76 | 0 |
| left_pulmonary_arts | 116 | 116 |
| dias_pul_art | 495 | 69 |
| dias_rca1 | 59 | 0 |
| dias_lad1 | 112 | 0 |

Coronary-named labels are present in the main ROI and displayed in the crop,
but this arch-only interval does not edit them. Selecting a different parent
percentage can target a more proximal region without entering IDs.

All five non-aorta labels in the edit interval share native voxel faces with the
aorta label. This includes pulmonary labels. The source grid therefore contains
contacts that make simple connected-component selection insufficient to isolate
the aortic tree. Category selection alone also cannot distinguish these artery
types. Inspect the native Local view and tube placement around those contacts;
generic `arteries` names do not establish which named arch branch is represented.
The evidence JSON preserves contact counts and example i,j,k coordinates.

## Why inward erosion is limited

The nonarterial face-neighbors of the unresolved erosion set are exclusively
`chest_surface`. Some unresolved voxels lie one layer deeper and do not directly
touch that label. The current catalog assigns `chest_surface` to the fine
`11_Integumentary/Skin` hierarchy via the `region_surface_skin` rule, with medium
confidence, although its coarse tissue category is `soft_tissue`.

Fine skin identity takes precedence over the soft-tissue factor. Therefore these
recipient seeds use stiffness 0.95. With a 3 mm search their initial penalty is
2.85 mm, leaving only 0.15 mm of travel, less than one 1 mm grid edge. Of 458
recipient seeds, 432 carry this label. The other seeds are muscle, trachea and
bronchi; these permit the 49 achieved aorta releases. The 516 preserved proposals
comprise 509 aorta voxels and 7 left-pulmonary-label voxels.

A diagnostic that retained the proposed-release domain but removed seed penalties
could reach 509 of those 516 voxels within 3 mm. This was a read-only calculation,
not an edited output or a recommended stiffness setting. Every unresolved release
component has at least one eligible seed, so absence of nearby tissue is not the
primary cause.

This is evidence to review the **meaning of interior `chest_surface` voxels and
the skin mapping** before treating them as physical skin or weakening protection.
The v7 trial retains the requested skin factor and preserves unresolved voxels.
It does not establish seamless junction changes or calibrated erosion severity.

## Frame and validation evidence

The 204.1933 mm original path yields a fitted length of 203.5153 mm with 205
approximately equally spaced spline samples. At smoothing 1 mm, control-point
RMS displacement is 1.000006 mm (within FITPACK's numerical tolerance) and maximum
displacement is 1.54062 mm. There are 203 reliable frames; the two endpoints are
excluded. No low-curvature or inflection ambiguity was flagged for this fitted
path. No voxel in either selected edit ROI was skipped for unreliable curvature.
These flags describe this sampled reference and do not establish anatomical fit.

- 265 tests passed, including analytic curves, direction reversal, anisotropic
  spacing, straight-path skipping, actual asymmetric morphology, and complete
  report/NPZ orchestration.
- All 34 output manifest entries and all 18 generating code hashes match. The
  captured INI equals the new input, including blank `source_ids`.
- Independent read-only memory mapping confirms that source crop labels and
  attenuation equal the saved originals. Full source sizes and modification
  times are unchanged. Full 4.185 GB source volumes were not hashed.
- Per-pass labels and attenuation form an exact chain; changed voxels remain
  inside supported directions and ROI/range bounds. Stiffness-1 barriers remain
  unchanged. Frame T/N/B vectors are unit and orthogonal, with B = T × N.
- Static frame, curvature, and before/after figures were inspected. Firefox
  verified report tabs, captured category selection, frame embedding and legend
  visibility states. Headless WebGL was unavailable; its notice intercepted
  initial legend clicks. For control checks only, notice pointer interception
  was disabled in the isolated browser. Interactive 3D scene pixels were not
  validated. The report includes static fallbacks.

Next review: confirm the skin mapping near the arch, refine the tube to manage
pulmonary contacts, and choose any proximal interval needed for coronary ostia.
Use a fresh output directory for the next parameter trial, such as recipe-v8.
