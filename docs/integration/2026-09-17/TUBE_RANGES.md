# Reuse a tube through per-edit physical ranges

The aorta input now defines its centerline and radii once. Each edit references
`roi = main` and a `path_percent = start,end` interval. The first listed point
is 0%; the last is 100%. Percentages measure physical centerline distance,
including anisotropic voxel spacing, rather than fractions of the point count.
Endpoint coordinates and radii interpolate along the parent segments.

The alternative `point_range = 2,8` selects original points 2 through 8 exactly,
with one-based inclusive endpoint numbering. Supply either selector or neither;
omitting both selects the complete referenced ROI. Existing explicit named
spheres/tubes remain supported, including subranges of named tubes. `main` is
reserved for the top-level `[roi]` section.

Ranges constrain uniform and Gaussian operations alike. The effective mask is
the intersection of the subpath tube, parent tube, selected parent arc interval,
and main ROI. A voxel's closest parent centerline position determines its arc
coordinate; ties retain the first supplied segment. Round subpath caps cannot
extend edits beyond an internal range limit. At bends, closest-point partitions
need not be planar. The parent path remains fixed through all recipe passes.

For `profile_axis = tube`, the selected interval is remapped to local u=0…1.
The Gaussian's `shape_window` is relative to that interval. For parent range
30–75%, local u=0.5 is parent 52.5%. This implements longitudinal localization
and taper; circumferential/eccentric asymmetry remains future work.

## Aorta evaluation

Open the [new report](../../../outputs/studies/thoracic-aorta/recipe-v6/report.html)
and [captured compact input](../../../outputs/studies/thoracic-aorta/recipe-v6/input.ini).
The parent tube has 14 points and physical length **204.193311 mm**. The report's
main tube point table includes native i,j,k, radius, cumulative millimeters, and
percentage. Region rows and per-pass details identify the parent, range, resolved
geometry, and local-coordinate meaning. Colored native close-ups and the shared
before/after overlay remain included.

| Edit | Path (%) | Approximate parent points | Requested edit | Changed voxels |
| --- | --- | --- | --- | ---: |
| ascending_expand | 7.38–51.28 | 2–8 | 10 mm dilation, one pass | 4,600 added |
| descending_narrow | 66.54–87.63 | 10–13 | 10 mm erosion, one pass | 2,038 removed |
| arch_refine | 20.84–60.23 | 4–9 | Inactive planning interval | 0 |

The ascending interval incorporates the extra arch-side points in the user's
latest example. Percentages were rounded to two decimal places, so endpoints
approximately match those point numbers; use `point_range` when exact controls
are desired. The main path/radii, 10 mm edit strengths, profiles, tissue factors,
28 mm crop half-width, and context stride 3 were retained. The three duplicated
named ROI definitions were removed. The new output is `recipe-v6`; prior trial
inputs and results remain available for comparison.

## Verification

All **233 tests passed**. New coverage includes strict parser alternatives,
uneven/anisotropic physical paths, interpolation and direction reversal, uniform
round-cap leakage prevention, independent ranges on one parent, overlap policy,
sequential label/attenuation state, local Gaussian profiles, source/config
immutability, legacy equivalence, compact end-to-end artifacts, and report tables.

For the real report, every changed voxel's parent arc coordinate was independently
calculated from the saved masks: ascending changes lie within 9.8174–48.9104%,
and descending changes within 68.0184–86.1414%, both inside their input intervals.
The inactive step changed no voxels. All 33 artifact hashes and 15 generating-code
hashes match. Original crop arrays and source file stats were verified; whole
source files were not hashed. Static range and final surface previews were
visually inspected.

Firefox headless checks verified Edits tab activation, all 14 point-reference
rows, physical percentage values, and the captured compact input. WebGL pixels
were not validated in this environment. Machine-readable evidence is saved in
[tube-range-results.json](tube-range-results.json).

For the next trial, edit percentages, choose a new output such as `recipe-v7`,
and run the usual `scripts/preview_roi.py --config ...` command. The new ranges
do not alter tissue policy, introduce a training sweep, or infer anatomical
direction from the array axes.
