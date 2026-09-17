# Bounded INI morphology evaluation

The same `scripts/preview_roi.py --config PATH.ini` command now accepts
`fakect.edit/1`, adding erosion/dilation and explicit tissue reassignment to the
existing ROI workflow. See [the complete input](../../../configs/examples/xcat-edit.ini),
[usage guide](../../MORPHOLOGY.md), and [machine results](morphology-results.json).

## Relationship to the phantom tools

The imported `src/fakect.py` remains unchanged from `origin/phantom` commit
`c6a5c2c9c47ebcbee116c8ade18269eef4ac09ff`. The new engine reuses its six-neighbor
morphology topology and normalized Gaussian formula. Its physical grid distances,
spatial profile, tube support and label ownership are new behavior, named
`weighted_6_neighbor_mm_v1`; it does not claim identical output to the legacy GUI.

The inherited `apply_roi_scale` uses an integer sphere, converts ROI radius to an
iteration count, and weights iteration progress with a Gaussian. That is not a
measured vessel-diameter ratio. Prior audits found nonmonotonic erosion, no-op
strength ranges, and changes outside the ROI from the global bridge. The new
adapter uses a fixed physical ROI, explicit millimetre grid-distance budget,
spatial profile, and protected-tissue rules. It does not invoke the global bridge.

Notebook donor/recipient concepts inform the policy, but the original notebook
algorithms are not imported as production code. Accepted dilation propagation
cannot cross protected barriers; 3D erosion repair traverses only proposed release
voxels from eligible original neighbors. Original signed labels remain the source
of identity. Ties use physical path distance, signed ID and original source index.

## Executed real-case trials

Both runs used case 260602/frame 1, the user's current ordered tube points
`404,360,1544 ; 404,363,1584 ; 401,362,1624`, and radii `4.7,4.7,4.5` mm. The existing
user-edited preview INI was preserved. The crop is 161 x 84 x 84 native voxels,
with 1 mm spacing and 8,073 artery candidates. Initial tissue-only selection
contains 1,892 voxels: 1,706 from internal_carotid_left (1185) and 186 from generic
arteries (1547). Both IDs participate in these trials; this is not a single-ID edit.

Both use distance 1.5 mm, Gaussian shape k=10, tube arc-length profile with window
[0.25,0.75], allowed surrounding tissues soft_tissue/muscle/adipose, and erosion
recipient search limit 3 mm with unresolved voxels preserved.

| Measurement | Erosion | Dilation |
| --- | ---: | ---: |
| Target voxels before | 1,892 | 1,892 |
| Target voxels after | 1,679 | 2,150 |
| Proposed releases | 213 | 0 |
| Reassigned releases | 213 | 0 |
| Proposed additions | 0 | 272 |
| Accepted additions | 0 | 258 |
| Blocked additions | 0 | 14 |
| Unresolved releases | 0 | 0 |
| Six-neighbor components before/after | 1 / 1 | 1 / 1 |

All 14 blocked dilation proposals belonged to vein tissue, which the allowlist
protects. Erosion changed 193 carotid and 20 generic-artery voxels; dilation added
239 carotid and 19 generic-artery voxels. Counts also equal mm³ here because the
native voxels are 1 mm³. They do not measure a clinical stenosis percentage.

The reports are:

- `outputs/edits/carotid-erosion-v1/report.html` (38,074,615 bytes).
- `outputs/edits/carotid-dilation-v1/report.html` (38,061,315 bytes).

Each is a self-contained HTML file with six embedded PNGs and two embedded Plotly
documents, covering before/after geometry, affected tissue, profiles and provenance.
The captured INI is editable for subsequent runs. The dilation input differs only
in operation, study name and output path and is captured in its output directory.

## Verification

`MPLCONFIGDIR=/tmp/fakect-mpl-cache python3 -m unittest discover -s tests`
completed with **93 tests passing**, including 17 new morphology cases and an
end-to-end edit workflow. Coverage includes:

- Weighted anisotropic steps, exact crop/full equivalence and monotone accepted
  masks as distance increases with fixed ROI/profile/policy.
- Full-target context during erosion, preventing artificial cut-face erosion
  at ROI boundaries; zero-ended spatial profiles and ordered tube projections.
- Protected barriers and unknown labels, explicit background eligibility,
  deterministic signed-ID ownership and scalar-copy source ties.
- Empty allowlists, unresolved preservation/error, source-array identity,
  original-label preservation outside the ROI, and balanced transition counts.
- Strict INI parsing and legacy preview compatibility, new report assets,
  standalone embedding, escaping and no-overwrite behavior.

For both real cases, independent saved-array checks confirmed:

- `changed_mask` exactly equals original-versus-edited label differences.
- Outside-ROI and protected labels are unchanged; scalar proxies outside accepted
  changes equal the original scalar exactly.
- Every proxy value at a changed voxel exists in the original scalar values of
  its assigned original label. The proxy is not presented as reconstructed CT.
- Proposed/applied/blocked/unresolved counts and transition tables balance.
- Original crop content hashes match captured source hashes, while source size
  and modification time remain unchanged. Full source binaries were not hashed.
- Config, catalog, atlas, policy, audit, all eight code files and all artifacts
  match recorded hashes. Embedded PNG/3D bytes match their separate artifacts.

Before/after comparison images and the achieved/profile chart were visually
inspected. Browser/WebGL execution was not available for automated verification.
Each trial read 724,500,000 source bytes in complete axial planes while retaining
only the native crop. No source files, source DPI data, training dataset, or jobs
were modified or submitted.

## Scope remaining

These are derived crop-level label edits. The separately named attenuation-copy
proxy is not learned recovery, XCAT material simulation or a production CT volume.
Clinical diameter/area calibration, GUI parity, multiple-edit composition, sweeps,
full-volume export and training/cohort integration remain separate milestones.
