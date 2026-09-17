# Find and refine an ROI without anatomical IDs

Choose a tissue category and place an ROI. XCAT's anatomical IDs are optional
filters; the pipeline obtains the underlying labels from the catalog for you:

```ini
[selection]
tissue = artery
source_ids =
```

You may also omit the `source_ids` line. This works for preview, single-edit and
named-recipe inputs. `[stiffness.labels]` is also optional in a recipe: omit that
section and use the category factors in `[stiffness]`.

The [aorta exploration input](../configs/studies/thoracic-aorta-explore.ini)
starts with the reviewed overall tube but has no anatomical-ID filter and applies
no geometry edits. Run it from the integrated checkout:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-explore.ini
```

Open [the exploration report](../outputs/studies/thoracic-aorta/explore-v1/report.html).
New reports contain separate **Global view** and **Local view** tabs. The report
is portable and its navigation works offline. Earlier reports are immutable
snapshots; rerun an input into a fresh output folder to obtain the new tabs.

## Global view: locate the region

The global navigator samples the **whole phantom**, independently of the local
crop. Browse axial, coronal and sagittal planes with the linked slice controls,
or click an image to move the crosshair. Coordinate readouts use native **i, j,
k voxel indices**. They do not require knowing an anatomical label ID.

Switch between attenuation and tissue display to locate an anatomical region.
The configured main ROI and any named recipe regions provide context. A movable
temporary sphere guide helps estimate a center and physical radius; copy the
coordinates into the input when useful. The guide is a positioning aid, and does
not overwrite an existing tube or change the saved INI.

Global images are sampled for bounded memory and report size. The report states
the sample spacing; thin vessels can be missed between samples. Use the global
view to find the neighborhood, then native local views to refine the selection.

## Local view: refine the boundary

Local view shows the captured ROI at native resolution, including the transparent
ROI overlay, tissue candidates, selected tissue, neighboring axial slices and
3D context. Native indices are displayed on the slice axes.

To move or reshape the actual ROI, edit the input:

```ini
[roi]
shape = tube
center_ijk = 387,360,1352 ; 378,365,1363 ; 373,362,1374
radius_mm = 10,10,10
crop_half_width_mm = 22
coordinate_reviewed = false

[preview]
slice_ijk = roi
```

Keep all other required sections from the complete example. Centers are native
indices; radii and crop padding are millimetres. Each tube node has one radius,
and semicolons separate the ordered nodes. For a sphere, use one center triple
and one radius. `slice_ijk = roi` follows the ROI automatically; a numeric triple
chooses inspection planes inside the local crop.

Change `[output] directory` to a fresh folder, such as
`outputs/studies/thoracic-aorta/explore-v2`, save and rerun the same command. Open
the new report and switch to Local view. The Input tab can also edit, copy or
download the INI text; save that text to the file passed to the command.

Global slider/guide changes update only the browser view. Native local images
and edit results correspond to the captured input until a new run is generated.
They are not silently recomputed from a temporary browser guide.

## Separate nearby vessels by geometry

With `tissue = artery` and no ID filter, the selected mask is:

```text
all artery voxels intersected with the ROI
```

If two arteries enter that ROI, both are selected. A narrower tube following one
vessel can distinguish them even if they share the same XCAT ID. Position the
path and tune its radii using the local overlays and multiple slices. Connected
components are spatial diagnostics, not automatic identification of an aorta
or another named vessel.

For multiple regional edits, transfer refined coordinates into `[roi.NAME]`
sections of a [named recipe](EDIT_RECIPES.md). Its main `[roi]` remains the outer
edit boundary, and every named ROI is intersected with it. Leave `source_ids`
blank or omitted there too. The report records actual selected original IDs for
provenance, but entering them is not a prerequisite for ROI placement or edits.

Training segmentation targets have their own definition, separate from this
edit selection. The current anatomy-specific study trainer still uses explicit
`train.target_source_ids` for full-crop ground truth; clearing an edit's
`source_ids` does not change that training target. See [training studies](TRAINING_STUDIES.md).
