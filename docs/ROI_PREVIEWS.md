# Edit the ROI in a plain-text input

Open [configs/examples/xcat-roi.ini](../configs/examples/xcat-roi.ini). This is the
executable authoring format for the current preview workflow. Each setting has a
short comment pointing to a detailed numbered NOTE at the bottom. Generated JSON
is retained for provenance; users do not need to edit it. The older cohort JSON
is a historical design draft for operations that have not yet been implemented.

Start with these settings:

```ini
[selection]
tissue = artery                    # NOTE 7
source_ids = 1185                   # NOTE 8: internal_carotid_left

[roi]
center_ijk = 404, 363, 1584          # NOTE 9: relocate the sphere
radius_mm = 12                     # NOTE 10: physical radius
crop_half_width_mm = 40            # NOTE 11: close-up extent
```

These excerpts illustrate edits within the complete file; keep its other sections.
The initial position was found from actual signed source ID 1185 in case 260602,
frame 1. It is a candidate location for visual review. It differs from the earlier
whole-body crosshair at `(375,375,1264)`, which was not on this carotid segment.

From the integration checkout:

```bash
python3 scripts/preview_roi.py --config configs/examples/xcat-roi.ini --validate-only
python3 scripts/preview_roi.py --config configs/examples/xcat-roi.ini
```

The evaluated example already produced `outputs/roi/carotid-v2/`. Before running
your next iteration, set `output.directory` to a new location such as
`outputs/roi/carotid-v3`. Existing populated output directories are refused.
All relative paths inside the INI resolve from the repository containing the
script, regardless of where the INI itself is saved.

## Read and adjust the previews

- **[ROI close-ups](integration/2026-09-16/xcat-roi-closeups.png):** attenuation and
  tissue views in all three native array planes. Orange fill is the sphere's true
  intersection with that plane; cyan outlines the selected original IDs. The
  indices stay tied to the full source volume when the crop moves.
- **[Neighboring z levels](integration/2026-09-16/xcat-roi-z-stack.png):** five axial
  planes across the sphere, with fixed k and relative z in millimetres. This helps
  choose which part of the vessel the ROI should cover.
- **Interactive 3D:** open `outputs/roi/carotid-v2/roi-volume.html` in a browser.
  Drag to rotate, scroll to zoom, click legend entries to hide/show the selected
  vessel, bone, veins or ROI, and use the opacity slider. The HTML embeds its
  JavaScript and requires no internet connection; rendering uses browser WebGL.
- **[Static 3D](integration/2026-09-16/xcat-roi-surfaces.png):** selected vessel and
  ROI alone beside the same crop with transparent context. This is a surface
  rendering; the interactive HTML uses binary-occupancy volume traces.

To relocate the ROI, change `center_ijk`. To inspect another plane while keeping
the ROI fixed, change `preview.slice_ijk` from `roi` to an explicit i,j,k triple.
To select a different structure, change `tissue` and `source_ids`; a blank ID field
selects all original IDs assigned to that group. A selected label absent from the
crop still produces discovery views and an explicit relocation warning.

The sphere radius and crop half-width use millimetres with each axis's own spacing.
Source axes are i=column, j=row, k=slice; anatomical orientation/physical origin
remain unverified. `coordinate_reviewed` records the current placement review; it
does not authorize geometry changes. Configuration and catalog snapshots are
captured at the start, so a saved preview records the inputs actually used even
if the editable file changes while rendering.

## Resolution and retained data

2D views, source IDs, target masks and sphere membership use full native resolution.
For manageable 3D rendering, the default stride 2 combines each small voxel block
by maximum occupancy. Thin selected labels survive, but displayed support can widen
and apparent gaps can close. Inspect the native 2D slices before judging fine
geometry. Static transparency is illustrative and 3D views do not validate topology.
Use smaller crops with stride 1 for native occupancy; the loader rejects requests
that exceed the bounded crop/display budgets with instructions to reduce them.

`crop.npz` retains original signed labels, grouped labels, original attenuation,
the selected mask, sphere mask, geometry and the complete catalog snapshot.
`input.ini`, `resolved-config.json` and `preview-report.json` retain the exact
settings, hashes, source identities and measured counts. Unknown labels stay 255
in the grouped view and retain their original values in `original_labels`.

These arrays are a useful input for adapting the notebook algorithms, but this
command performs preview only. Geometry edits still need the shared bounded
donor/recipient rules and original-organ ownership for new voxels. The unknown
group must not silently become a donor or a background replacement. No source
volume, label map or scalar data is modified by this workbench.

The ROI preview uses NumPy, Matplotlib, Plotly and scikit-image, plus Python's
standard-library INI parser. It does not require a graphical session on Wahab.

## Atlas identity

The [currency check](integration/2026-09-16/DPI_ATLAS_CURRENCY.md) compares the pinned
documents to the current DPI source and the frozen v6 campaign's actual case
provenance. They match. `1.0.0-proposed` names **FakeCT's grouping policy**, while
the downstream DPI cardiovascular atlas is **1.1.0**. V6 names a connectivity
recovery workflow, not a replacement source atlas. Figures now distinguish the
grouping-policy version from the DPI source-atlas hash. Current source limitations
remain reviewable in the detailed metadata beneath each group.
