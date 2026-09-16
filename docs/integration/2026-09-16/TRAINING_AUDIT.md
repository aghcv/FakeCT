# FakeCT branch, training, and morphology audit

Read-only audit of cached Git branches, existing Wahab launchers/logs, dataset headers/manifests, saved Keras ZIP metadata, and small synthetic morphology arrays. No training, GPU job, installation, checkout, or source changes were performed by this audit. Audit artifacts are this report, `scripts/evaluation/audit_training.py`, and `outputs/evaluation/training-results.json`; source and data remain unchanged. Historical measurements below were read from existing output files; they are not newly benchmarked results.

## Integration sources

- `origin/phantom` / `origin/sujay` at `c6a5c2c`: current VTI/GUI/multilayer morphology source; no TensorFlow training module.
- Local `main` at `183e003`: older `src/fakenoise.py` training implementation, equivalent to historical `origin/stenosis` training.
- Local `stenosis` at `d3b7a62923959abd7f6f983526367829ffcf0c65`: one commit ahead of historical cached `origin/stenosis` (`ca30a3a`), adds improved evaluation and context-tagged output names. No changes to model architecture, leakage, binary masks, or memory design.
- The original checkout has pre-existing modifications to `environment.yml` and untracked `requirements-tf115.txt`, `requirements_cluster_backup.txt`; this audit did not touch them.

`git diff main..stenosis -- src/fakenoise.py` shows the newer version adds `--tol-step`, per-context output paths, per-slice previews, tolerance curves, intensity-bin error, and PSNR/SSIM. It also avoids the old preview shape mismatch when another case has different dimensions. The newer version remains an experimental mask-to-grayscale model, with no load-model/inference command, full-volume prediction export, HU calibration, mask-edit compositing, or source-background preservation contract.

## Wahab launchers and historical execution evidence

`/home/aghorban/slurm/fakenoise_train_gpu.sh:2-9` requests one GPU, eight CPUs, 8G host memory, and 28 hours. Lines 16-18 load `container_env tensorflow-gpu/2.17` and run `crun -p ~/envs/fakect python .../src/fakenoise.py --mode train --csv .../pairs.csv --context 15 --context-step 4`.

Peers are inconsistent historical launchers:

- `fakenoise_train_cpu.sh:10-14`: TensorFlow CPU 1.15.0 module but direct `/usr/bin/python3` execution.
- `fakenoise_train_gpu_heavy.sh:16-21`: TensorFlow GPU 1.15.0, GCC/CUDA modules, direct `/usr/bin/python3`, context 60/step 1.
- `fakenoise_train_cpu_heavy.sh:9-11`: python3 container module but direct Python; context 60/step 1 and misleading pair-job output naming.
- `fakenoise_pair.sh:9-11`: `crun -p ~/.conda/envs/fakect`; this differs from the current GPU environment path.

There are six saved Keras models and six metric CSVs under `data/dataset/paired_datasets`. Completion stdout exists for jobs `5575321`, `5575375`, `5575594`, `5575595`, `5575596`, and `5575597`. However, stderr for all six explicitly reports missing CUDA libraries/drivers and that GPU will not be used. Therefore the files demonstrate completed training, but not successful GPU acceleration or the current TF2.17 launcher configuration. Script contents are current; the saved output logs are historical and do not persist the exact command or source SHA.

Representative errors: `fakenoise_train.5553531.err` GLIBC_2.27 import failure; `5553674.err` missing NumPy; `5554111.err` and `5575071.err` missing numpy.typing; early jobs hit time limits; `5575112.err` and `5575202.err` finish training then fail preview because a 512x512 slice is passed to a 512x666 model. These are concrete reasons to retain the working container route and add an explicit GPU preflight.

Saved model `metadata.json` says Keras 3.10.0, saved April 11-13, 2026. Input shapes are `(None,512,666,7/31/61/121)`, confirming historical shape restriction and context-channel expansion. This metadata does not by itself establish TensorFlow version. The untagged `fakenoise_train/fakenoise_model.keras` has seven channels, so it must not be labeled a single-slice baseline; its context step is not recorded in that path.

## Historical metrics (normalized intensity, not HU)

Source: each respective `test_metrics*.csv`, row 2. The context-tagged values are consistent with output folders but lack full run provenance.

| Output setting | MAE | RMSE | Mean slice PSNR (dB) | Mean slice SSIM |
| --- | ---: | ---: | ---: | ---: |
| Untagged seven-channel model | 0.21243756 | 0.35638806 | unavailable | unavailable |
| context 3, step 2 | 0.21184087 | 0.35981068 | 8.93683 | 0.336756 |
| context 15, step 4 | 0.19933224 | 0.34756330 | 9.25064 | 0.353644 |
| context 30, step 1 | 0.20407440 | 0.35139132 | 9.15201 | 0.344319 |
| context 30, step 2 | 0.19883706 | 0.34413415 | 9.33760 | 0.341808 |
| context 60, step 1 | 0.19947372 | 0.34582189 | 9.29446 | 0.346090 |

For context15/step4, `tolerance_curve_ctx15_step4.csv` reports 65.73565% pixels within 0.05 normalized absolute error and 69.54869% within 0.10. Its intensity-bin CSV shows MAE 0.02708 for ground truth [0,.1), but 0.58581 for [.6,.7), 0.63044 for [.7,.8), and 0.68594 for [.9,1]. Thus global MAE is heavily helped by dark pixels; reconstruction of higher intensity regions is substantially worse. These metrics are not evidence of clinically accurate CT recovery, multilabel reconstruction, or unseen-patient generalization.

## Data/split/geometry findings

The existing `data/dataset/paired_datasets/pairs.csv` contains 22,753 rows, 56 case IDs, 56 image/mask pairs, zero duplicate `(image,x_index)` entries, and no missing image/mask files. IDs are constructed from paths (`src/fakenoise.py:148-151`), not a verified patient identity registry. Different scans of one patient would become different IDs.

All 56 mask headers are three-dimensional and describe exactly one segment with LabelValue 1: 45 names `aorta`, ten `Segment_1`, and one `Segment_3`. This header check establishes single-segment metadata, not an exhaustive unique-voxel-value scan. There is no evidence here of training with distinct organ/tissue classes. Training itself collapses all nonzero labels with `mask > 0` (main lines 404,412; stenosis equivalent lines 410,418), so future multilabel inputs would lose their identities. Four-dimensional Slicer segment arrays would be rejected by `_load_nrrd` (lines 30-36).

Reproduction using current manifest order and seed42, exactly following main lines353-368 / stenosis359-374:

| Split | Nominal rows / cases | Rows / cases actually yielded |
| --- | ---: | ---: |
| Train | 15,972 / 44 | 1,991 / 14 |
| Validation | 1,774 / 44 | 235 / 14 |
| Test | 5,007 / 12 | 482 / 3 |

The first shuffled training row is D3, so input shape becomes 512x666. Seventeen cases have this shape; 39 have 512x512. The generator reads/caches each volume and silently skips nonmatching shapes (`main:387-395`, `stenosis:393-401`). Consequently 20,045/22,753 rows are excluded, and test metrics concern only D11, D4, and D8. The retained validation cases are the same fourteen cases as training. All 44 nominal training cases also occur in nominal validation. Test case IDs do remain disjoint. Neighbor-mask contexts can overlap across training and validation slices, making validation especially optimistic.

Input iteration is unsorted (`Path.rglob` at line156); a recreated CSV can alter ID order, seed42 shuffle, and selected shape. Only NumPy/dataframe shuffling is seeded; TensorFlow initialization/global random state is not. No split manifest, exclusions manifest, data hash, Git SHA, or normalization metadata is saved.

Geometry is not validated beyond array dimensions. NRRD `space directions` and `space origin` are discarded in training. Sample D4 header is LPS with directions approximately diag(.751953,.751953,2.988826), sizes(512,666,180). R9 is diag(.744141,.744141,.625). K1 is diag(.810547,.810547,.5). `nrrd.read` defaults to header-axis/F ordering in this environment, whereas code names `vol.shape` as z,y,x and calls axis2 sagittal/X. For these sampled diagonal-LPS files, axis2 is the physical superior-inferior direction; the sagittal label is misleading. Context15/step4 spans +/-60 index slices, about +/-179mm for D4 versus +/-37.5mm for R9, if used without resampling. All coordinate/axis conventions must be explicit before joining to phantom's z,y,x arrays and i,j,k GUI coordinates.

Each grayscale volume is independently percentile-normalized using its own p1/p99, including test target images (`_load_volume_cached:266-277`). This is not necessarily model-training leakage, but it makes evaluation depend on ground-truth-derived scale and loses HU meaning. A mask-only inference path cannot reproduce unknown target-specific intensity scaling without an explicit fixed calibration policy.

## Scalability

Whole image and mask volumes are retained indefinitely in dictionaries. Header-derived uncompressed image+mask storage for all56 cases is about **17.539GiB**, before percentile temporaries, TensorFlow, prefetched data, activations, and predictions. Even excluded-shape volumes are loaded before their shape check. This is an estimated resident array requirement, not measured process RSS.

The `tf.data` pipeline shuffles 256 already expanded float32 mask stacks and gray targets (`main:421-429`). At512x666:

| Context | Channels | 256-sample shuffle buffer, GiB | Batch8 input tensor, GiB |
| --- | ---: | ---: | ---: |
| 3 | 7 | 2.602 | 0.071 |
| 15 | 31 | 10.406 | 0.315 |
| 30 | 61 | 20.162 | 0.620 |
| 60 | 121 | 39.674 | 1.230 |

The context15 shuffle buffer alone exceeds current Slurm `--mem=8G`. Actual TensorFlow retention/allocation and job accounting must be measured; historical completion does not validate the present 8G configuration. Prefer shuffling compact sample indices, bounded volume caching/patch loading, explicit prefetch limits, batched inference, and capped preview counts. New stenosis predicts every preview slice individually, then evaluates all slices again, adding substantial avoidable inference and small-file I/O.

## Phantom morphology and terminal reuse

`origin/phantom:src/fakect.py` reusable core functions:

- `_roi_mask_vox:1831`, `_bitwise_dilate6:1843`, `_bitwise_erode6:1853`, `_gaussian_shape:1822`, `proximity_bridge:1863`, `apply_roi_scale:1901`.
- Input layer and metadata construction: `_make_mask_layer:595`, `load_vti_label_layers:847`, `load_vti_directory_label_layers:939`.
- Export helpers: `mask_to_trimesh:1545`, `save_mask_stl:1562`.

The GUI can select a target label and perform ROI morphology. `apply_exclusive_conflicts:3146` removes target occupancy from other layers in the same exclusive group. Batch snapshots at3161 start from each original layer; GUI Apply at3711 operates on current state. The JSON/YAML contract must state whether operations are cumulative or independent from a source snapshot. Eroded voxels are not automatically reassigned to another tissue; a background/neighbor resolution policy is needed.

`export_batch_masks:3557-3634` derives spherical ROI center/radius from two points, sweeps scale and shape_k, saves all layer masks with spacing/origin and ROI metadata, and optionally exports target STL. It is nested inside Dash and depends on GUI state. Extract this orchestration and exclusive-group handling into UI-independent functions; retain a thin Dash adapter and add a terminal adapter.

CLI at4390-4449 supports STL/VTI/VTI-directory conversion, export, sampling limits, and `--no-show`; it does **not** expose ROI morphology, batch sweeps, label target selection, JSON/YAML configuration, or AI prediction. The existing `json` import only decodes Dash callback IDs. `--no-show` currently converts/exports the input and does not reproduce interactive edits. NRRD input is not handled by phantom's pipeline.

Suggested modules: geometry-aware volume I/O; schema/validation; pure morphology+label conflict operations; batch pipeline+provenance; model/data preprocessing; training/evaluation; inference/compositing; optional GUI adapter. A schema should include version, input formats/paths, array or label mapping, coordinate space+units, explicit target label, ROI, operation sequence, scale/shape/bridge parameters, conflict policy, output resolution, optional model+normalization, seed, and output paths. Preserve native data for exports and separate browser decimation from processing resolution; default VTI maximum dimension is160.

## Bounded morphology runtime findings

Executed AST-extracted unmodified phantom functions with NumPy/SciPy/scikit-image on small arrays, without importing Dash or reading data volumes. These are behavioral tests, not tests of a new implementation.

For a49-cube array with a rectangular vessel `mask[10:39,21:28,21:28]=1` (1,421 voxels), center(24,24,24), default shape_k10/window(.25,.75), bridge0:

| Radius | Scale | Result voxels | Changed voxels | Changes outside initial ROI |
| --- | ---: | ---: | ---: | ---: |
| 8 | .5,.8,1,1.2,1.5 | 1,421 each | 0 | 0 |
| 8 | 2 | 2,049 | 628 | 264 |
| 12 | .5 | 971 | 450 | 0 |
| 12 | 1.5 | 2,235 | 814 | 186 |
| 16 | .5 | 819 | 602 | 0 |
| 16 | .8 | 675 | 746 | 0 |
| 16 | 1.2 / 1.5 | 2,331 each | 910 | 0 |

Identity passes. Dilation/erosion can work, but moderate edits on a small ROI can be no-ops. More strongly, .5 leaves more voxels than .8 in this example: the parameter is not a reliable monotonic diameter scale. Gaussian weighting, integer radius steps, and moving ROI boundaries make the nominal scale materially different from measured vessel diameter. Dilation's expanding ROI intentionally permits changes beyond the initial sphere; the supported edit region needs an explicit contract.

A separate33-cube mask has a central vessel, center(16,16,16),r8,scale1.5, plus isolated voxels at array indices(2,2,2) and(2,2,4). With bridge0 the remote gap stays zero. With default bridge2, voxel(2,2,3) becomes1, far outside every intended ROI. `apply_roi_scale:1988-1989` calls `proximity_bridge` over the entire merged volume, so even a locally ineffective morphology request can alter unrelated anatomy. Restrict bridge changes to declared edit support before using it in an augmentation pipeline.

## Proposed acceptance gates

These are engineering gates for integration, not clinical acceptance thresholds.

1. Data integrity: zero undocumented exclusions; every manifest pair checked for shape, dtype, affine/directions/origin, label-map consistency, and valid bounds. Record exact inclusion/exclusion counts and reasons. Make a documented normalization/resampling policy handle both existing image shapes.
2. Leakage: immutable train/validation/test manifests with **zero verified subject overlap** between every pair of splits; group repeated scans, derived phantoms, and all augmented descendants with the source subject. Report exact case and slice counts. At minimum, zero case-ID overlap where true subject metadata is unavailable, explicitly recording that limitation.
3. Geometry: round-trip landmarks and affine metadata at native resolution; axis permutation/physical-position tests including anisotropic, non-cubic, and oriented volumes. Require exact label-ID preservation and no non-target edits unless the configured conflict policy permits them.
4. Morphology: scale1 identity; predictable erosion/dilation on analytic fixtures; zero changed voxels outside declared edit support including bridging; quantified diameter/area change instead of assuming nominal scale equals stenosis severity. Decide monotonicity requirements and test them. GUI and terminal operations using the same request must produce byte-identical label arrays.
5. GPU readiness: preflight under the exact `crun`/module environment records Python, TensorFlow/Keras, visible GPU name/device, data/model SHA, and source commit; fail a required-GPU run if no GPU exists. Execute a tiny batch before full training and require a finite loss plus save/reload parity. No GPU job was submitted in this audit.
6. Resource control: measure peak host/GPU memory and examples/sec for context0 and15; use an explicit host memory ceiling below the Slurm allocation (for8G, target <=6GiB RSS for headroom). No unbounded cache or expanded-sample shuffle; deterministic reporting of skipped samples; finite dataset cardinality; bounded previews. Increase the requested memory only after measurement justifies it.
7. Baselines: regenerate one-channel/context0 and31-channel/context15-step4 models with identical leakage-free splits and normalization, plus simple zero/mean or tissue-intensity baselines. Do not call the existing untagged seven-channel model context0. Fix the test set once and select hyperparameters with validation only.
8. Quality reporting: case-level MAE/RMSE, PSNR/SSIM, per-tissue/mask/background/edited-region scores, intensity-bin errors, and confidence intervals across subjects; include full-volume seam/continuity and geometry fidelity. For an initial improvement gate, require held-out subject MAE better than the best simple baseline, with a paired bootstrap interval excluding no improvement and no loss of the geometry/background-preservation guarantees. Historical .1993 MAE/.3536 SSIM are reference observations, not valid universal thresholds.
9. Reproducible inference: exported model sidecar records label vocabulary, axes/spacing, context/range in mm, normalization, training split/data hashes, source SHA, and software versions. Save/reload must agree within declared floating-point tolerance. Preserve untouched original CT outside the chosen edit support exactly when doing background recovery/compositing; where no source CT is given, clearly describe the output as synthetic generation.
10. Terminal configuration: validate JSON/YAML before any output writes; provide dry-run with resolved config, planned outputs, memory estimate, and data checks; reject unknown labels, coordinate ambiguity, invalid windows/scales, and unsafe output collisions. CLI and GUI call the same validated operation API and emit provenance for every augmented volume.

## Reproduction artifacts

`python3 scripts/evaluation/audit_training.py` regenerates `outputs/evaluation/training-results.json`. Options select repository, dataset root, Slurm directory, morphology Git ref, and output JSON. It uses existing NumPy/pandas/pynrrd/SciPy/scikit-image, reads only NRRD headers (not volume pixels), reads bounded logs, hashes source/manifest/models, inspects Keras ZIP metadata without TensorFlow, and runs the exact small morphology fixtures above. The JSON includes software versions, source SHA, metric rows, retained split IDs/counts, memory estimates, launcher snapshots, diagnostic log lines, and voxel-change measurements. No package installation is needed in the audited environment.

The current launchers must not be equated with the commands that produced historical logs. Missing-CUDA messages establish that those recorded runs lacked GPU use; no current TF2.17 job was run during this audit.
