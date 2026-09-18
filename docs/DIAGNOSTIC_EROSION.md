# Inspect erosion before assigning surrounding tissue

Set this optional flag on an erosion edit:

```ini
[edit.arch_inner_narrow]
operation = erosion
assign_surrounding_tissue = false
```

The flag also works in a single `[edit]` section. Omission or `true` retains
the existing surrounding-tissue assignment. The current curvature example sets
it to `false` and writes `outputs/studies/thoracic-aorta/recipe-v10`.

With `false`, every proposed release that passes the target distance,
stiffness, directional and profile constraints becomes the bright-pink
**released** diagnostic category. The recipient search is skipped, so skin-like
classification of neighboring `chest_surface` voxels cannot veto that release.
The nearby tissues are not relabeled, and their atlas mapping and stiffness
settings remain unchanged. This mode does not resolve whether `chest_surface`
is a correct material identity in this region.

The report provides native before/after close-ups focused on releases and a
table of the labels sharing a voxel face with them. Counts are unique neighbor
voxels, not contact-face counts, and include remaining target tissue. These are
the existing classifications, not newly verified material assignments.

In the overlapping 3D view, each present nonbackground tissue category has its
own visibility and 15%, 45%, or 80% opacity controls. Context categories show
the **final** labels. Ordinary categories begin hidden to keep the before/after
comparison visible; released markers begin visible at 80%. Selected before
and after surfaces and the released-marker surface retain native resolution.
Other context layers may use bounded occupancy pooling; their metadata records
the display stride and native voxel counts.

## Labels, attenuation and sequential passes

Released is a synthetic output label, not an anatomical tissue or an XCAT ID.
The derived signed label ID is `-2147483648` and its uint8 category is `254`.
Reserved identity collisions are rejected. The original catalog remains intact;
`derived-label-catalog.json` appends the diagnostic mapping and is recorded in
the report provenance and final NPZ metadata.

Attenuation at released positions is **retained from the input state only as a
placeholder**. It is not copied from a neighboring tissue or reconstructed.
`attenuation_unassigned_mask` identifies every such voxel. The category and
mask therefore support geometry debugging, not completed CT training pairs.
Training preparation rejects this diagnostic mode and existing marker voxels.

The saved arrays also include `diagnostic_released_mask` (new releases during
this pass, or their union for the recipe) and `released_mask` (all current
markers). Original before-label arrays preserve which anatomy was removed.
In selection mode, removal clears target ancestry normally.

Markers remain protected in later passes; they cannot seed filling, supply
dilation material, or become anatomical edit targets. Repeated diagnostic
erosion can remove additional target layers where the remaining target/profile
allows it. To evaluate normal material assignment again, set the flag to `true`
and rerun from the original phantom into a new directory.

`max_distance_mm` and `unresolved` still apply to ordinary reassignment, but
do not control diagnostic marker creation. The existing conservative crop halo
is retained. False is accepted only for erosion in the INI.

After the current output exists, choose a fresh output directory such as
`recipe-v11` and rerun:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-curvature.ini
```
