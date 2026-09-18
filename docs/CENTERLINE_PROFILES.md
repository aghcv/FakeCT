# Profiles along the ROI centerline

Tube ROI reports now plot edit profiles against **curvilinear distance in mm
from the first original ROI centerline point**. An upper axis shows the matching
percentage used by `path_percent`. This happens automatically; no new INI
setting is needed. The main recipe comparison uses the main ROI. Each individual
edit uses its full referenced parent ROI, with its selected interval shaded.
Selecting a subrange does not reset the distance to zero.

The old `k` axis is the native axial slice index: an array coordinate, not
distance along the aorta. A curved vessel can intersect one axial slice more
than once. The old plot remains available under **Native axial diagnostic
(k slices)**. Sphere ROI reports continue to use that axial coordinate because
a sphere supplies no ordered centerline.

## What the curves measure

Before, after and change masks are measured on the same original physical
polyline. Each native voxel center is assigned to its nearest point on that
polyline, accounting for voxel spacing. Its volume contributes once to a
physical arc-length bin. The displayed quantity is **volume per unit path
length**, in mm³/mm (equivalently mm²). It is not a plane-sampled vessel-normal
lumen area, diameter, or stenosis percentage. Branches and self-approaching
paths can share bins; a nearest-path assignment does not identify anatomy.

The bins are equal in length, approximately twice the largest native voxel
spacing, with at most 512 bins. Native voxel-center sampling can produce small
steps in the curves. Voxels projected to either terminal endpoint have no
resolved longitudinal position beyond that endpoint; their volumes are
reported separately instead of producing misleading edge peaks. Interior-bin
volumes plus the two endpoint totals reproduce each original mask volume.

Tracked growth outside the selector is included at its **current position**.
Offspring are not plotted at an ancestor's location. The strength curve shows
the largest requested spatial distance in each bin within the permitted edit
footprint, including directional taper and before tissue resistance. The final
recipe uses the largest per-pass request, not the sum of requests or an
achieved displacement. Empty bins are zero.

The reference follows straight segments between the original ordered control
points, matching the editing engine and `path_percent`. Smoothing the
direction-reference spline does not move this coordinate. Point ordering sets
the start; patient proximal/distal orientation is not inferred.

## Outputs and rerunning

`edit-profile.png` is the new default chart. `edit-profile-axial.png` retains
the axial diagnostic for tube ROIs. `edit-profile.csv` contains the plotted
interior-bin values. Full bin volumes, endpoint totals, node distances and
semantics are in the figure's `arc_profile` metadata in `preview-report.json`
and the saved NPZ metadata. Each recipe pass has the same files in its step
directory. The HTML embeds both charts for portable review.

The original centerline-profile evaluation is `outputs/studies/thoracic-aorta/recipe-v9`.
The current input adds [diagnostic erosion](DIAGNOSTIC_EROSION.md) in `recipe-v10`.
After that directory has been generated, select a new `[output] directory`
(for example `recipe-v11`) and rerun from the integration worktree:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-curvature.ini
```

This changes reporting only. ROI masks, morphology, tissue stiffness and
reassignment rules retain their existing behavior.
