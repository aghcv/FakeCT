# Crop context and display budget for 10 mm aorta edits

The user's revised aorta recipe requested 10 mm ascending dilation and 10 mm
descending erosion, with revised named paths. Its 24 mm crop half-width and
display stride 2 could not satisfy the crop and rendering checks together.

The required context is 14 mm beyond each active named ROI envelope:
10 mm editing distance + 3 mm reassignment search + 1 mm maximum voxel spacing.
The ascending ROI had only 12 mm at the lower i crop face. Increasing the crop
alone then exceeded the separate 650,000-cell display budget.

The working input now uses `roi.crop_half_width_mm = 28` and
`preview.volume_stride = 3`. Only those two parameters changed; numbered notes
were expanded to explain their relationship. The user's ROI coordinates,
radii, edit distances, profiles, stiffness factors, and `recipe-v5` output
choice were preserved. The other edited INI files were untouched.

The [completed report](../../../outputs/studies/thoracic-aorta/recipe-v5/report.html)
and [captured input](../../../outputs/studies/thoracic-aorta/recipe-v5/input.ini)
contain the complete trial. The native crop is 158 × 116 × 77 (k,j,i), or
1,411,256 voxels. Stride 2 would require 797,040 estimated display cells;
stride 3 requires 252,560. Editing, native slices, and the before/after overlay
retain source resolution. Native overlay stride remains 1.

The final recipe adds 3,340 target voxels and removes 2,038. These counts cannot
be compared as a pure edit-distance experiment with recipe-v4 because the user
also changed named ROI paths. The complete before/after overlay and per-step
comparisons are included for review.

Errors now identify the responsible edit and ROI, give the halo calculation
and limiting crop face, and distinguish unavailable source context from a crop
that can be enlarged. Display errors give the estimated cell count, limit, and
smallest valid stride. If no stride preserves at least two display blocks along
each axis, the message explains that constraint. Geometry and resource limits
remain enforced.

Metadata validation, full report generation, and all 205 unit tests passed.
The original arrays
were preserved; 33 artifact hashes, 14 generating-code hashes, and source file
stats were verified. Entire source volumes were not hashed. The static overlay
was visually inspected. The parser regression tests now use a fixed fixture
instead of the user's actively edited geometry. See the
[machine-readable results](preview-context-results.json).

For the next trial, change `[output] directory` to a fresh location such as
`outputs/studies/thoracic-aorta/recipe-v6`, then run from this checkout:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-recipe.ini --validate-only
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-recipe.ini
```

The successful `recipe-v5` directory is now populated and will not be overwritten.
