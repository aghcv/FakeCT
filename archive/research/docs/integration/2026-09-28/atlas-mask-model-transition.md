# Atlas masks and independent reference reconstruction

This development replaces learned segmentation upstream of Model 2 for labeled
XCAT experiments. It also adds interactive 3D review of existing predictions
and of a first broader anatomical target. No models were retrained, thresholds
changed, or frozen scientific results replaced.

## Completed artifacts

| Artifact | Result |
| --- | --- |
| Existing experiment `two-stage-r2/report.html` | Interactive 3D comparison of all 32 held-out scenarios; exact-mask, chained, and segmentation-error modes |
| `two-stage-r2/report-2d.html` | Preserved original self-contained report |
| `outputs/anatomy/thoracic-auto/260602-r1/selection.json` | ROI-free native selection: 66,708 aortic-core + 5,892 proximal branch voxels; nine original source IDs |
| `outputs/anatomy/thoracic-auto/metadata-validation-all12.json` | Metadata/registered RAW inventory validation passes for all twelve cases; this is not twelve completed native extractions |
| `outputs/datasets/coa/thoracic-mask-preview/260602-r2/preview-manifest.json` | All sixteen frozen edit scenarios transfer successfully to the broader reference; zero incompatible edits |
| `outputs/datasets/coa/thoracic-mask-preview/260602-r2/review/index.html` | Seventeen native-grid scenes: anatomical selection plus sixteen before/after/add/remove comparisons |
| `archive/research/configs/models/coa-mask-model.ini` | Independent Model 2 runner validates the existing 192-pair manifest with 128/32/32 samples and no Model 1 |

The new native-mask viewer contains approximately 54.5 MiB of HTML/JS assets,
with at most 134,446 triangles per scene. All requested surfaces fit the display
budget: no pooling, smoothing, mesh decimation or omission was needed. The
sixteen compressed draft mask bundles total 626,635 bytes. Viewers load one
scene at a time from local files. Preserve the whole viewer directory when
sharing it. Generated arrays, model weights and report assets remain Git-ignored.

The atlas selector uses the same original ACT revision as the frozen pairs.
Its original aorta label includes the abdominal aorta, so the first explicit
extent policy is T12 vertebral z-midpoint through the upper T1 surface bound.
The core is the largest native aortic component; eligible proximal branches
must connect to it after pulmonary labels are excluded. Branch reach defaults
to three times the median verified aortic ring radius. This is a reviewable
approximation, not a verified whole-thoracic/diaphragm boundary or a promise
of complete named branches. The old ROI and its selected IDs are not reused
for inclusion; its full surface inventory can optionally accelerate RAW reads.

For 260602 the selection uses native 1 mm spacing, crop origin
`i,j,k = 339,321,1216`, shape `k,j,i = 290,131,100`, a RAW-z slab of
1219.061–1502.551 mm, and 22.132 mm branch reach. All original selected voxels
belong to the broader reference, and no old lineage additions were already
present in the broader reference. These checks hold for this pilot case only.

Draft transfer verifies the selector manifest/mask checksums, case/frame,
crop geometry, spacing, native bounds and original ACT provenance. It also
checks frozen pair hashes and dilation ancestry. Anatomy outside the existing
edit deltas is preserved, and added/removed masks are recomputed from the new
before/after pair. The draft manifest explicitly records `training_ready=false`
and cannot be used as the old training release.

## Verification and scientific scope

All 62 targeted tests passed: 11 selector, 17 pair-transfer/export, 14 mask-model,
11 3D-helper and 9 report tests. The existing experiment lock still verifies its
original scientific code/configuration/data hashes. Its review summary now
records separate original-2D and updated-3D report provenance. Browser control
events and local-file loading were checked; the headless Wahab node lacks a
working WebGL graphics driver, so actual mesh rasterization needs a browser
with WebGL support.

The current mask-model INI still trains on the existing ROI-selected lineage
dataset. Its validated input scope is explicit. The new broader reference
needs consistent anatomical review across cases, a separate training-release
adapter and retraining. No improved performance follows merely from bypassing
Model 1: the prior exact-mask Model 2 result already excludes segmentation
errors and has reference Dice 0.918 versus 0.980 for identity, with added/removed
Dice 0.222/0.349. False changes on unchanged controls remain a central failure.

The model estimates a reference and net expansion/narrowing regions. Specific
coarctation, arch-hypoplasia and ascending-dilation attribution and severity
remain later evaluation tasks. A new clinical case could supply an independently
segmented aorta, but the deterministic XCAT selector itself requires native
anatomical labels and does not segment attenuation-only patient CT.

See [the workflow and commands](../../PHANTOM_MASK_WORKFLOW.md) for review,
configuration changes and the independent mask-model interface.
