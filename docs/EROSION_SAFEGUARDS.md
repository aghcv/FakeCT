# Local erosion safeguards

Erosion can now retry a rejected distance before publishing any changed labels.
Configure the checks independently on each erosion in the same INI:

```ini
[edit.coa_narrow]
roi = main
path_percent = 65,75
operation = erosion
distance_mm = 10
iterations = 1
profile = gaussian
profile_axis = tube
shape_k = 8
shape_window = 0.05,0.95
direction = all
assign_surrounding_tissue = false
min_volume_ratio = 0.50
preserve_connectivity = true
backoff_factor = 0.5
max_backoff_steps = 8
```

These settings also work on a standalone `[edit]`. The usual source, selection,
ROI, reassignment and output sections are still required. The current complete
example is [thoracic-aorta-curvature.ini](../configs/studies/thoracic-aorta-curvature.ini),
with detailed numbered comments in NOTEs 30–32. Its tissue selection remains
`artery`; no original anatomical ID is needed.

## Measurement reference

At the start of a named edit, capture the currently selected target where the
longitudinal profile is positive. For a ranged tube this excludes the rest of
the parent ROI and the zero-strength Gaussian window ends. The example above
uses parent arc positions **65.5–74.5%**, derived from its 65–75% selection and
local 0.05–0.95 window. Native voxel centers use the same closest-parent-polyline
coordinate as the edit, including physical spacing.

The reference includes the full **selected** cross-section within this active
segment, before angular weights, tissue resistance or the erosion-distance
threshold. It is not a mask of only the voxels proposed for removal. A narrow
selector can exclude target from the denominator; inspect the selector and the
saved measurement mask. In selection mode, previously grown offspring of the
selected ancestors participate too.

The retained fraction is:

```text
remaining target voxels at reference positions / original reference target voxels
```

All native crop voxels have equal physical volume, so the volume factor cancels.
The report also records both volumes in mm³. This baseline is fixed across all
iterations of the named step, including retries. It is captured after earlier
recipe steps; a different named edit receives a new reference even if its range
overlaps. This is a per-step floor, not a cumulative whole-recipe floor.

An empty reference causes a no-op; its retained ratio is reported as 1 by
convention. No tissue is invented to satisfy a floor.

## Connectivity and retry policy

For every pre-step target component intersecting the measurement mask, the
connectivity check requires its surviving target voxels to form exactly one
component connected through voxel faces (six neighbors). It evaluates the full
loaded target crop, including connections outside the selected segment.
Pre-existing separate components are checked individually. A split cannot be
hidden by the disappearance of another component, and complete loss fails.

Every pass tries the requested `distance_mm` first. If either enabled check
fails, it retries from the **same pass input** with the distance multiplied by
`backoff_factor`. With a factor of 0.5, a 10 mm request tries 10, 5, 2.5 mm, etc.
`max_backoff_steps` limits reductions after the first attempt. The first passing
trial is accepted; this is a bounded geometric search, not an optimization for
the largest feasible distance. If none passes, the original pass input is kept
and accepted distance is reported as zero. Small positive distances can also
leave the discrete voxel geometry unchanged.

Rejected results never update labels, released markers, attenuation, target
ancestry, activity masks, per-pass artifacts or the following pass. Only their
measurement diagnostics enter the audit. Requested values in the INI stay
unchanged. Each iteration starts by trying the requested value again, against
the same fixed step baseline. Conservative crop/halo checks still validate the
full request before any retry.

| Setting | Default when omitted | Allowed values |
| --- | --- | --- |
| `min_volume_ratio` | `0` (disabled) | Finite 0–1; 0.5 means retain at least 50% |
| `preserve_connectivity` | `false` | `true` or `false` |
| `backoff_factor` | `0.5` | Finite, strictly between 0 and 1 |
| `max_backoff_steps` | `8` | Integer 0–20 |

Both checks must pass when both are enabled. Backoff settings alone do not
enable a check. These fields require `operation = erosion` in user input;
remove them when converting that edit to dilation or an inactive placeholder.
Existing inputs without the fields retain their previous editing behavior.

For single-edit training sweeps, enabled policies are retained and recorded in
erosion variants. Generated baseline/dilation variants omit the erosion-only
fields. Diagnostic released markers still block training-pair generation until
normal tissue assignment is restored.

## Reviewing and rerunning

From this checkout:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-curvature.ini --validate-only
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-curvature.ini
```

Choose a fresh `[output] directory` for each run; populated reports are preserved.
The Edits tab shows requested and accepted distance, baseline and retained local
volume, the floor, component results, and each trial's rejection reasons.
Per-pass profiles describe the accepted trial. Each `steps/.../step.npz` saves
`erosion_guard_scope_mask` and `erosion_guard_retained_mask`; `step_json` records
the same audit. Source data, catalog metadata, original inputs and code hashes
remain available through the report's provenance artifacts.

The 50% floor in the example is an initial debugging setting. These checks do
not constrain minimum cross-sectional area or diameter: a one-voxel bridge can
pass connectivity, and retained volume can be unevenly distributed along the
segment. They do not repair pre-existing gaps or infer connections outside the
crop. Continue inspecting local slices and the before/after surface overlay.
