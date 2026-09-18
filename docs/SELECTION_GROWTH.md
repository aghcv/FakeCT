# Selecting anatomy separately from its growth region

The [current curvature input](../configs/studies/thoracic-aorta-curvature.ini)
uses the user's revised parameters: 9 mm outward dilation over parent 3–20%,
Gaussian window 0.1–0.8, followed by 7 mm inward erosion over 25–35%, window
0.3–0.9. Each step runs once. Tube coordinates, radii, spacing, tissue category,
direction settings and stiffness factors are preserved.

The one new setting is:

```ini
[recipe]
roi_role = selection
```

The ROI and path interval select original artery voxels. Their edited offspring
may extend beyond both the tube wall and the main ROI. You do not need to widen
the selector to provide growth space. The original evaluation is
[recipe-v8/report.html](../outputs/studies/thoracic-aorta/recipe-v8/report.html);
[recipe-v9](../outputs/studies/thoracic-aorta/recipe-v9/report.html) retains the
same edits with [centerline-distance profiles](CENTERLINE_PROFILES.md).
Omitting `roi_role`, or setting it to `boundary`, retains the earlier behavior
that clips every edit to its ROI.

## How selection persists

Each original target voxel has an internal ancestor index corresponding to its
native location. An accepted dilation voxel inherits the original ancestor of
the winning current target seed. Erosion removes that ancestry when tissue is
reassigned. These indices are internal provenance; users still select the
`artery` category and ROI coordinates without entering XCAT anatomical IDs.

Every iteration and later edit selects the surviving ancestors inside its
**original** ROI and their offspring, wherever those offspring currently lie.
For example, dilating and then eroding the same original region includes the
first step's growth outside the selector. An initially empty selector cannot
adopt tissue that later grows through it from a different original selection.
Different ranges can act on overlapping original ancestors in the configured
recipe order.

Nearby artery voxels outside the initial selection are never new seeds, even
if they share the same anatomical ID. They cannot carry a dilation path and
remain unchanged. Growth can touch those voxels; the software does not infer
whether a contact is a branch junction or a different vascular tree. The
report records contact counts. The selector must include the target surface
you intend to expand: unselected portions of the same vessel also retain their
original identity and are not silently added to selection.

## Available growth space and achieved movement

Each dilation pass builds a physical Euclidean envelope within `distance_mm`
of its current tracked target. This bounds the engine's weighted six-neighbor
paths and supplies space beyond the ROI. The envelope does not change labels
by itself or guarantee that a voxel can move that distance.

The actual local budget still combines the distance, longitudinal profile,
curvature-relative angular factor and donor stiffness. Accepted paths obey
tissue eligibility and local budgets. Erosion applies only to the tracked
target, using the full target category to locate true tissue boundaries so a
selection cut does not become an artificial vessel wall. Recipient search and
stiffness rules remain unchanged.

`path_percent` limits original target selection. In selection mode, uniform
dilation can grow beyond the original interval's end faces. Gaussian strength
still uses the current voxel's spatial parent arc coordinate and vanishes
outside the configured local `shape_window`. Curvature directions likewise
remain referenced to the full original tube and are evaluated throughout the
edit region. The fit does not move or enlarge the selection tube.

Repeated passes grow from tracked offspring rather than repeatedly selecting
all current artery voxels inside a bigger spatial mask. Conservative context
validation accounts for prior growth inherited through overlapping selectors
and the full requested iteration count. `crop_half_width_mm` provides loaded
context only. A context error requires a larger crop or smaller edit budget;
increasing the ROI radius would change the selected anatomy. This input's
existing crop and display stride suffice for its two one-pass edits.

`overlap = sequential` permits ordered interactions. `overlap = error` checks
both original selectors and conservative growth envelopes for distinct edit
regions before the first pass. It can reject envelopes whose actual changes
would not overlap after profiles and resistance; it is a conservative check.

## Reviewing and rerunning

The report's per-pass figures distinguish the fixed orange selector, dashed
blue edit footprint and tracked target. Plot bounds include outside-ROI growth.
Target area plots include the full tracked result; separate metadata retains
the area still intersecting the original selector. The final before/after
surface overlay continues to show the complete artery context.

Per-pass `step.npz` files preserve `selection_roi_mask`, `edit_region_mask`,
`original_selected_mask`, `target_seed_index`, `target_origin_before_index` and
`target_origin_index`, together with labels, scalars, profiles and change masks.
Original ancestor indices are zero-based C-order flat indices into the native
k,j,i crop; -1 indicates a non-target voxel. The saved geometry converts them
to native coordinates. Final counts include offspring outside the main ROI;
`changed_outside_selection_roi` identifies those changes explicitly.

After editing the INI, choose a fresh output directory such as `recipe-v11`, then
run from the integrated worktree:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-curvature.ini --validate-only
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-curvature.ini
```

The original phantom stays unchanged. The existing `chest_surface` skin mapping
can still restrict borrowing near the aorta; separating the growth region does
not change that classification or the requested skin protection. Geometry,
branch contacts and reassignment remain subjects for review.
