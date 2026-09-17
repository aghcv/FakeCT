# Aorta study integration: preview, paired data and segmentation training

Date: 2026-09-17. Branch: `fakect.26.09.16`.

The first aorta study is ready for ROI and parameter review. Its actual XCAT run
produced a source preview and proposed sweep; **no aorta cohort, real aorta model
or Slurm job was launched**. Pair preparation and fitting were exercised with
small synthetic fixtures.

## Review artifacts

- [Editable study INI](../../../configs/studies/thoracic-aorta.ini)
- [Standalone HTML report](../../../outputs/studies/thoracic-aorta/preview-v1/report.html)
- [Preview verification and provenance](aorta-study-preview.json)
- [Aorta localization evidence](aorta-localization.json)
- [Practical workflow guide](../../TRAINING_STUDIES.md)

The HTML embeds four PNG figures and one interactive 3D frame, including a
separate source-image/full-target comparison. It is about 17.8 MB and can be
copied alone. The first draft remains under `preview-initial-draft`; `preview-v1`
is the final report with explicit `--stage preview` in its rerun command.

## Actual source evaluation

Case **260602**, frame **1** uses native `k,j,i` arrays of shape
`1860,750,750`, with `1,1,1` mm spacing. Source ID **2922**, `dias_aorta`, is
supported by the organ dictionary, DPI hierarchy and a named heart surface.
Surface block ordinal 25 is naming evidence only, not a voxel ID. Neighboring
generic arterial labels were recorded as contacts and were not added to the
aorta target.

The locator found 83,743 aorta voxels in its refined scan range, with six-neighbor
components of 83,738 and 5 voxels. The unreviewed 14-node arch tube produces:

| Quantity | Result |
| --- | ---: |
| Crop lower `i,j,k` | `351,338,1314` |
| Crop upper `i,j,k`, exclusive | `416,442,1434` |
| Crop shape `k,j,i` | `120,104,65` |
| Crop voxels | 811,200 |
| Aorta target throughout crop | 51,688 |
| Aorta target inside editable tube | 44,874 |
| Aorta target outside editable tube | 6,814 |
| Selected six-neighbor components | 44,873 + 1 |
| Unknown-group voxels inside tube | 4,142 |

The small original fragment remains preserved; it was not silently removed.
Unknown tissue assignments remain visible and protected by the reassignment
policy. The crop omits much of the descending aorta. It therefore supports an
arch-crop engineering experiment, not complete thoracic segmentation. Anatomical
orientation, vessel boundary interpretation, endpoints and intervention support
still need review.

Native image/mask figures and the static 3D view were visually inspected. All
11 listed output artifact hashes, captured input bytes and generating-code hashes
matched. Source sizes and modification times matched their recorded values.
The bounded crop hashes are recorded; full multi-gigabyte sources were not hashed.
Embedded-asset checks passed. Interactive WebGL execution in a browser was not
tested in this environment.

## Implemented stages and pairing contract

`fakect.study/1` shares the existing ROI, edit and reassignment settings. Its
strict `[train]` and `[model]` sections select one explicit stage:

1. `preview`: current source, optional single edit, binary-target view and draft
   parameter table in HTML. It never prepares the cohort or starts TensorFlow.
2. `plan`: validates metadata and all planned edit halos without reading voxel
   payloads; captures input, normalized settings, plan and hashes.
3. `prepare`: independently applies each variant to the original crop; exports
   aligned float32 image / uint8 full-crop binary-mask pairs, original signed
   label identities, change masks, source snapshot and frozen split manifest.
4. `fit`: verifies the existing dataset configuration and artifacts, then fits
   a compact TensorFlow 2D U-Net. Validation selects a `.keras` checkpoint; test
   arrays are not decoded or used for training or selection. Integrity checks
   can read reserved archive bytes to verify checksums.

The initial Cartesian sweep is erosion/dilation at `0.5,1,1.5,2` mm with Gaussian
`shape_k=10`, window `0.25,0.75`, plus unchanged baseline: **nine requested
variants**, provisionally split 7 train / 1 validation / 1 test. This is a draft
plan, not nine demonstrated distinct anatomies. Small physical grid distances
can produce no change. Exact duplicate masks stay together; baseline-equivalent
masks stay in training. A resulting empty validation split prevents fitting.

The binary target is membership in edited ID 2922 across the entire crop,
including outside the editable ROI. Every variant preserves its original-ID
metadata. The scalar image method is explicitly `attenuation_copy_proxy`,
converted to inverse centimetres; it is neither learned background recovery nor
scanner reconstruction/HU. `scenario_only` holdouts share one source anatomy and
cannot establish independent-anatomy or patient generalization.

## Verification

The expanded suite contains **134 tests**. Its full run passed 133 tests; the
TensorFlow subprocess reached the original 180-second timeout. The isolated
CPU training/save/reload/inference test subsequently passed in **17.740 seconds**
with the timeout extended to accommodate variable initialization time. The
preview-stage override regression was checked again after its command fix.

Coverage includes exact input capture; strict INI validation; unchanged sources;
signed IDs; whole-crop masks; independent edits; image/label alignment; bounded
plans; geometry-duplicate split handling; artifact corruption and stale settings;
padding excluded from loss and metrics; fixed normalization; and test arrays
excluded from fitting. Synthetic end-to-end tests cover every CLI stage, with
the fitting route mocked separately from the real TensorFlow fixture test.

The actual CPU fixture used **TensorFlow 2.20.0**. The new Wahab wrapper follows
the supplied `container_env tensorflow-gpu/2.17` and `~/envs/fakect` pattern;
its shell syntax passed, but TensorFlow 2.17 GPU execution remains untested.

## Next review

Open the HTML and edit the INI's tube nodes, radii and inspection crosshair.
Choose `preview-v2` as the next output directory. A single `erosion` or `dilation`
in `[edit]` adds a representative before/after trial while stage remains
`preview`. Then refine the sweep and permitted recipient/donor tissues from
realized geometry before preparing pairs. Clinical lesion measurements,
multi-operation recipes, independent-family splits, and general `[test]` /
`[generate]` commands remain later milestones in the project outline.
