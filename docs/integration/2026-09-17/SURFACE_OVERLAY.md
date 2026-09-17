# Before and after surfaces in one view

The Edits / recipe tab now begins with a registered surface overlay. Blue shows
the original selected anatomy; orange shows the final result of the complete
recipe. Each surface has its own Show checkbox and 15%, 45%, and 80% opacity
presets. Changing opacity preserves visibility, the other surface, and the
camera. The legend also toggles individual surfaces. A static overlay remains
available in an expandable section below the interactive view.

Open the [updated aorta report](../../../outputs/studies/thoracic-aorta/recipe-v4/report.html#surface-overlay).
The [saved input](../../../outputs/studies/thoracic-aorta/recipe-v4/input.ini)
captures the user's current 5 mm ascending dilation and 5 mm descending erosion,
each with one pass. Descending Gaussian concentration is 8; ascending is 6.
The arch step remains inactive. A temporary copy changed only `output.directory`
to `recipe-v4`; the user's three edited INI files and previous reports were
preserved.

## Geometry and observed changes

Both meshes use native voxel boundaries in the same coordinate frame, without
pooling or decimation. They include all selected anatomy within the crop rather
than clipping surfaces to the editing ROI. In this example the native crop is
146 × 104 × 65 in k,j,i order, with 1 mm spacing. The overlay includes 5,329
unchanged target voxels outside the main ROI as surrounding anatomical context.

| Measurement | Native count |
| --- | ---: |
| Before target voxels in crop | 55,814 |
| After target voxels in crop | 54,454 |
| Added target voxels | 1,889 |
| Removed target voxels | 3,249 |
| Unchanged target voxels | 52,565 |
| Before surface triangles | 39,984 |
| After surface triangles | 40,188 |

Additions and removals match the recipe's final native masks. Surface counts
cover the full crop; other recipe metrics may cover only the main ROI.
Overlapping colors are a qualitative geometry comparison, not a quantitative
change map. The surfaces display selected labels; they do not establish CT
attenuation recovery or anatomical orientation. Crop-edge closures are display
boundaries.

## Validation and use

`python3 -m unittest discover -s tests` passed all 202 tests. Coverage includes
native one-voxel changes despite coarse context settings, anisotropic coordinate
registration, unchanged source masks, empty and identical masks, bounded mesh
construction, standalone embedding, and both single-edit and recipe workflows.
The editable recipe test fixture now normalizes its own expected edit levels
instead of depending on the user's current trial values.

Firefox 140.12.0 ESR headless checks passed on both a small standalone overlay
and the actual embedded report: visibility controls, all six opacity choices,
hidden-surface preservation, independence of the other surface, camera
preservation, and direct navigation to the overlay's report tab. This environment
reports WebGL unsupported, so those checks validate JavaScript state and UI
behavior rather than interactive rendered pixels. The native static overlay
was generated and visually inspected. Interactive 3D requires WebGL in the
viewing browser.

All 33 artifact hashes and 14 generating-code hashes were verified. Overlay mask
hashes and counts were independently checked against saved original and final
arrays; input hashes, original crop hashes, and source file stats also match.
Entire source volumes were not hashed. See the
[machine-readable evaluation](surface-overlay-results.json).

New edit and recipe reports include the overlay automatically; there are no
additional INI parameters. Choose a fresh output directory, such as
`outputs/studies/thoracic-aorta/recipe-v5`, and run from the integrated checkout:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-recipe.ini
```

The native overlay accepts up to 2,000,000 crop voxels and 600,000 combined
surface triangles. It requests a smaller crop when these bounds are exceeded.
Ordinary no-edit previews and older reports remain supported without an overlay.
