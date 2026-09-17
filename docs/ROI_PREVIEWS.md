# One HTML report for tissue and ROI review

The primary output is now **`report.html`**. It combines the selection summary,
ROI definition, native-resolution close-ups, neighboring z levels, interactive
3D volume, static 3D views, selected anatomical labels, component counts, captured
input, and provenance. Images and Plotly JavaScript are embedded: copy this one
file to open the report elsewhere without a server or internet connection.
Browser WebGL is required for interactive 3D; static views remain in the report.
Source volumes remain at their recorded paths; reversible crop arrays are saved
in the output directory.

Use the [tube input](../configs/examples/xcat-roi-tube.ini) for tissue-only
selection, or the [legacy sphere input](../configs/examples/xcat-roi.ini).
Both use readable INI settings with brief parameter hints and numbered detailed
NOTES at the bottom. The tube file uses schema `fakect.preview/2`; existing `/1`
sphere inputs remain accepted.

## Select a tissue without knowing its original IDs

```ini
[selection]
tissue = artery                    # NOTE 7: candidate tissue group
source_ids =                       # NOTE 8: no original-ID restriction

[roi]
shape = tube                       # NOTE 18
center_ijk = 404,360,1544 ; 404,363,1584 ; 401,362,1624 # NOTE 9
radius_mm = 2.5, 2.5, 2.3           # NOTE 10: one radius in mm per point
crop_half_width_mm = 40             # NOTE 11: context around the full path
```

These are excerpts from the complete example; retain its other sections.
**Final selection = tissue candidates ∩ ROI.** All original IDs mapped to the
chosen tissue are candidates when `source_ids` is blank. Specifying IDs remains
an optional additional restriction. Context can include `artery` itself to show
neighboring arterial voxels outside the selected region.

The center points are native **i,j,k indices**, separated by semicolons. They are
connected in the order written; no coordinate sorting or spline fitting occurs.
Give exactly one positive physical radius per point. Centers and radii interpolate
linearly along each segment, and the ROI is the union of those moving balls,
including rounded caps. Distances account for each axis's voxel spacing.
Decimal center coordinates are accepted in `/2`; image slice indices remain
integers. For a sphere in `/2`, set `shape = sphere`, one center triple and one
radius. A crop surrounds the complete path, so longer tubes increase memory/I/O.

A narrow tube can exclude a nearby artery even if both vessels share an original
ID. It includes every matching tissue voxel it intersects, so a branch or
adjacent vessel inside a wide tube is also selected. The report lists selected
original IDs and six-neighbor components to help review this. One component is
not proof of one vessel, and multiple original IDs are not proof of multiple
vessels. There is no automatic connectivity-based branch removal.

## Generate and iterate

From `/home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16`:

```bash
python3 scripts/preview_roi.py --config configs/examples/xcat-roi-tube.ini --validate-only
python3 scripts/preview_roi.py --config configs/examples/xcat-roi-tube.ini
```

The evaluated example is at `outputs/roi/carotid-tube-v3/report.html`. Set
`output.directory` to a new directory such as `outputs/roi/carotid-tube-v4` before
your next run; existing populated directories are refused. Paths inside the INI
resolve relative to the repository containing the script, wherever the INI itself
is saved. The shell's `--config` path follows the usual working-directory rules.

Open the HTML and move through its sections:

1. Check candidate, ROI and selected-voxel counts and any component warnings.
2. Inspect the orange native ROI mask and cyan final-selection boundary in 2D.
   Faint dotted boundaries show surrounding candidate tissue. The five axial
   levels are representative samples; use `slice_ijk` to inspect other locations.
3. Rotate embedded 3D, toggle context groups and adjust opacity. The static
   comparison shows the selected region alone and with its surroundings.
4. Review actual original IDs selected, including small unexpected fragments.
5. Edit the captured INI in the report, download it, and run it with a new output
   directory. Editing the text does not change existing figures: rerun the
   command to produce the next report.

`slice_ijk = roi` uses the rounded sphere center or middle tube control point as
the crosshair. Explicit i,j,k slice indices move the inspection planes without
moving the ROI. Empty tissue–ROI intersections still produce a discovery report.

## Evaluated example and limits

In case 260602/frame 1, the wider tube with radii `4,4,3.5` selected 1,746 voxels:
1,664 of `internal_carotid_left` (1185) and 82 of a generic `arteries` label (1547).
They formed one connected component. This is why the original-ID table remains
useful even when the input selects only tissue.

The narrower example above selects 1,352 arterial voxels, all from original ID
1185, without specifying that ID in the input. It excludes the generic arterial
fragment, but also excludes some peripheral carotid voxels. The ROI defines a
segment of interest; it does not promise complete vessel segmentation.

The source-axis interpretation remains native i=column, j=row, k=slice with
unverified anatomical orientation/physical origin. `coordinate_reviewed` records
placement review only. Anatomical grouping and attenuation material identity
remain separate; the [DPI currency check](integration/2026-09-16/DPI_ATLAS_CURRENCY.md)
still applies.

## Resolution, retained data and compatibility

2D views, ROI membership and selection counts use full native resolution. The
3D anatomy display uses stride-2 block-maximum occupancy by default: thin labels
survive but their displayed support can widen and gaps can appear closed. Tube
ROI surfaces use the full native ROI mask, independent of that anatomy display
stride. These are visualization effects, not edits to labels.

`crop.npz` retains original signed labels, tissue groups, original attenuation,
`candidate_mask`, `roi_mask`, `selected_mask`, geometry and the complete catalog.
For current report schema `fakect.roi-preview/2`, `selected_mask` is explicitly
`candidate_mask & roi_mask`. Earlier report schema `/1` exports used the full
crop candidate mask under the name `selected_mask`; retain their accompanying
metadata when comparing historical runs. No source voxels are reassigned here.

The captured INI, resolved JSON, machine report and artifact hashes accompany the
HTML. Input/catalog snapshots are captured before rendering. Derived images are
retained for export, but the review entry point is `report.html`. Notebook
morphology still needs bounded donor/recipient and ownership rules before
geometry changes become production edits.
