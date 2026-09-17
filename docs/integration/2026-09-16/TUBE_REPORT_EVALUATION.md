# Tissue-only tube selection and a portable review report

The preview command now produces one `report.html` containing the selection
summary, ROI control points, native 2D overlays, representative axial levels,
interactive 3D volume, static 3D comparison, selected original-label counts,
captured editable INI, and provenance. Its images and Plotly JavaScript are
embedded. Downloading an edited INI does not regenerate figures; rerun the CLI
with that input and a fresh output directory.

The evaluated report is `outputs/roi/carotid-tube-v3/report.html`, generated from
[the tube example](../../../configs/examples/xcat-roi-tube.ini). See
[the workflow guide](../../ROI_PREVIEWS.md) for commands and interpretation, and
[the machine results](tube-report-results.json) for counts, hashes and checks.

## Selection and geometry

Blank `source_ids` considers every original ID mapped to the selected tissue.
The final selected mask is the intersection of those candidates and the ROI.
Original labels and attenuation remain available in the reversible crop.

Tube controls are ordered i,j,k triples, with one physical radius per point.
Centers and radii interpolate linearly along each segment; membership uses the
exact union of those balls, including rounded caps. Axis-specific voxel spacing
is applied before measuring distances. Points are not automatically sorted or
fitted to a spline. The mask does not apply connected-component filtering.

Schema `fakect.preview/2` adds explicit `shape = sphere` or `shape = tube` and
supports fractional control coordinates. Legacy `/1` sphere inputs still load.
Machine report `/2` stores `candidate_mask` separately and defines `selected_mask`
as candidate AND ROI. Historical report `/1` crops used `selected_mask` for the
entire crop's candidates; their recorded schema must be considered when comparing.

## Real-case comparison

Case 260602, frame 1 uses 1 mm spacing on all axes and control points
`404,360,1544 ; 404,363,1584 ; 401,362,1624`. Neither comparison restricts original
IDs. The 161 x 84 x 84 native crop contains 1,136,016 voxels and 8,073 artery
candidates. Its i,j,k bounds are [361,320,1504] through [445,404,1665], exclusive.

| Measurement | Wider tube: 4,4,3.5 mm | Current tube: 2.5,2.5,2.3 mm |
| --- | ---: | ---: |
| ROI voxels | 3,921 | 1,605 |
| Selected artery voxels | 1,746 | 1,352 |
| ID 1185, internal_carotid_left | 1,664 | 1,352 |
| ID 1547, generic arteries | 82 | 0 |
| Six-neighbor components | 1 | 1 |

The narrower tube excludes the generic arterial fragment, while also excluding
some peripheral carotid voxels. It defines a segment of interest rather than a
complete vessel segmentation. The wider result demonstrates that a single
connected component does not establish a single anatomical vessel. Original-ID
counts remain useful when selecting by tissue, but IDs alone also do not prove
vessel identity.

There are 125,104 unknown-category voxels in the context crop and zero within
the current ROI. Unmapped dictionary IDs in the crop are -830 and -816; these
remain visible in the report. The existing DPI atlas currency and classification
limits continue to apply; no atlas or tissue policy was changed in this step.

## Verification

- All 66 tests passed with `python3 -m unittest discover -s tests`.
- A complete synthetic workflow includes two parallel arteries sharing original
  signed ID -7. Tissue-only selection produces 18 candidate voxels; the tube
  selects nine from one artery. Source bytes remain unchanged, and the saved
  selection equals the intersection of the saved candidate and ROI masks.
- Geometry checks cover anisotropic spacing, fractional controls, round caps,
  bends, preserved order, reversal, varying radii and steep tapers. Invalid or
  inconsistent controls are rejected; legacy sphere behavior remains covered.
- Native crop content hashes match the exported arrays. Independent source reads
  check eight channel/coordinate combinations, including the focus and crop
  corners. Source sizes and modification times match the captured records.
- The captured INI matches the example byte for byte. Catalog, audit, atlas,
  policy, six code files and all ten artifact hashes were checked. The embedded
  PNGs match their source images, and embedded 3D HTML matches the standalone file.
- Close-up and static 3D images were visually inspected. HTML structure,
  escaping, embedded assets, missing-selection behavior and overwrite refusal
  are tested. Browser/WebGL interaction was not executed in this environment.

Reading the native crop required 161 complete axial planes from each source
channel: 724,500,000 requested bytes total. Full source volumes were not hashed
or validated. The interactive anatomy display uses stride-2 block-maximum
occupancy and can enlarge apparent support; the orange tube surface uses the
native ROI mask. The 2D overlays and counts use native resolution.

These changes provide selection review and reproducible inputs. They do not yet
perform dilation/erosion, voxel reassignment, AI recovery, job submission, cohort
generation or replacement of the training dataset.
