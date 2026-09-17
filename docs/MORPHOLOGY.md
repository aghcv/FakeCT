# Erosion and dilation through the same INI workflow

Use [xcat-edit.ini](../configs/examples/xcat-edit.ini) to run a bounded label edit
and inspect before/after results in the same standalone HTML report. It retains
the existing input, tissue selection, sphere/tube ROI, preview and output settings.
Schema `fakect.edit/1` adds two explicit sections:

```ini
[edit]
operation = erosion
distance_mm = 1.5
profile = gaussian
profile_axis = tube
shape_k = 10
shape_window = 0.25, 0.75

[reassignment]
allowed_tissues = soft_tissue, muscle, adipose
max_distance_mm = 3
unresolved = preserve
```

These are excerpts: use the complete example for a runnable input. Its numbered
NOTES explain each parameter. The example retains the user's most recent tube
radii, `4.7,4.7,4.5`, and leaves `source_ids` blank. Every selected arterial label
inside the ROI participates, so review the original-ID table when deciding
whether the tube includes the intended anatomy.

From the integration checkout:

```bash
cd /home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16
python3 scripts/preview_roi.py --config configs/examples/xcat-edit.ini --validate-only
python3 scripts/preview_roi.py --config configs/examples/xcat-edit.ini
```

Choose a fresh `output.directory` before each run. Existing populated directories
are refused. Each trial starts from the original XCAT input rather than the last
trial's edited labels. For dilation, change `operation = dilation` and choose a
fresh directory. Set `operation = none` for ordinary selection review. Existing
`fakect.preview/1` and `/2` inputs remain preview-only; adding edit sections to
those schemas is rejected. To upgrade an existing INI, change its schema to
`fakect.edit/1` and add **all** fields from both sections above.

## What the controls mean

- `distance_mm` is the maximum weighted six-neighbor distance for a boundary
  change. Each i/j/k step costs that axis's voxel spacing. This axis-biased grid
  metric extends the phantom tool's six-neighbor morphology to anisotropic data;
  it is not an exact Euclidean offset or measured vessel stenosis percentage.
- `profile = uniform` applies that budget throughout the ROI. A Gaussian profile
  uses the phantom normalized Gaussian formula spatially, with zero at the ends
  of `shape_window` and peak one at its midpoint. `shape_k` controls concentration.
  `shape_window = 0.25,0.75` affects the middle half of the specified path/axis.
- `profile_axis = tube` measures normalized physical arc length along the ordered
  control points. Voxels project to the nearest segment; equal-distance segments
  use the first in path order. The alternative axes `i`, `j`, `k` normalize over
  the ROI envelope along that physical axis. For a sphere with Gaussian profile,
  choose one of those three axes. A uniform profile ignores the profile axis.
- Voxel labels are discrete: a nonzero requested distance can produce no change
  if it does not reach a neighbor at the native spacing. Inspect achieved counts
  and profiles instead of assuming the request was fully attained.

The new engine is `weighted_6_neighbor_mm_v1`. It reuses the phantom topology and
Gaussian formula, with a new spatial profile and explicit label ownership. The
old GUI's `scale_factor` changes an integer spherical ROI radius and weights
morphology iterations; it does not implement a calibrated vessel-diameter ratio.
Audits found no-op strength ranges, nonmonotonic erosion and global bridge changes.
That routine is retained for the historical GUI, but its scale and global bridge
are not exposed as equivalent controls in this adapter.

## Neighboring labels and released voxels

The unchanged ROI is the edit boundary. Source signed IDs are authoritative;
coarse tissue groups select candidates and define eligibility without discarding
the original anatomical labels.

For **dilation**, the operation proposes growth from selected target voxels.
It can borrow only voxels of `allowed_tissues` inside the ROI. Accepted growth
propagates through eligible voxels, preserving protected barriers. Each new voxel
receives the original ID of its winning target seed. The propagation distance
must fit the local strength profile; blocked proposals are counted and displayed.

For **erosion**, boundary distances use the full candidate mask in the context
crop, avoiding artificial erosion where the vessel crosses the ROI boundary.
Only selected voxels inside the ROI can be released. Eligible original neighbors
in `allowed_tissues` supply recipient labels. Propagation travels only through
proposed release voxels and is limited by `max_distance_mm`; original eligible
neighbors may be outside the ROI because they are read as context, not changed.

Ownership is deterministic: shortest accepted physical six-edge path, then lowest
signed original ID, then earliest source k,j,i position. No arbitrary distant
tissue or empty-air fill is used. All unlisted tissues and unknown classifications
are protected; the target tissue cannot also be an allowed surrounding tissue.
An empty allowlist permits no borrowing or reassignment.

`unresolved = preserve` retains erosion voxels that lack an eligible recipient
and reports them. `unresolved = error` refuses such a trial before writing output.
This setting concerns erosion reassignment; blocked dilation proposals always
remain unchanged and are reported. The operation can reduce or disconnect a
target, so before/after connected-component counts are diagnostics, not a
connectivity guarantee.

## Review and exported data

The report includes before/after tissue views, accepted/blocked/unresolved change
overlays, axial area curves, requested profile, original-to-new label transitions,
and separate before/after interactive 3D views. Axial areas are ROI intersections;
they are not measured vessel-normal lumen areas. Native arrays determine counts;
the 3D anatomy display may use coarser occupancy sampling.

`crop.npz` retains the original preview arrays and catalog. `edit.npz` additionally
contains original and edited labels, original and edited tissue groups, before/
after target masks, proposal and accepted-change masks, the spatial strength field,
and a separately named `attenuation_proxy_per_pixel`. That proxy copies the
winning original target/recipient voxel value on accepted changes. Original
attenuation remains separately available. This copy is **not TensorFlow recovery,
physical CT synthesis, or training-ready recovered CT**.

The INI, machine report and artifact hashes capture the complete trial. Crops
are bounded to two million voxels for this prototype, with edit/search distances
at most 50 mm and an explicit context halo around the ROI. Insufficient crop or
source context is rejected before editing. Source volumes, the original dataset,
and the user's preview INI remain unchanged.

The next integration steps are calibrated diameter/area targets, multiple ordered
edits and independent parameter sweeps, scalar recovery, full-volume export, and
cohort generation. They are separate from this executable crop-level edit path.
