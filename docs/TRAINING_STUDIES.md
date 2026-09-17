# Review an aorta study, prepare pairs, then train

Start with [thoracic-aorta.ini](../configs/studies/thoracic-aorta.ini) and
[its first HTML review report](../outputs/studies/thoracic-aorta/preview-v1/report.html).
The initial study is **unreviewed** and defaults to `train.stage = preview`.
Only its source-anatomy preview and proposed parameter combinations have been
generated; no aorta cohort or GPU training has been launched.

The `fakect.study/1` INI extends the existing ROI/edit input with `[train]` and
`[model]`. Short comments beside parameters refer to numbered notes at the bottom.
Relative paths inside the INI are resolved from the integrated FakeCT checkout.
The original input text is captured with outputs.

## What each stage does

| Stage | Action | Destination |
| --- | --- | --- |
| `preview` | Read one bounded source crop; produce the HTML report, native slice/3D views, full binary training-target figure, and proposed variants. | `[output] directory` |
| `plan` | Save a metadata-only parameter plan; no volume processing or training. | `[output] directory` |
| `prepare` | Generate the reviewed variants, aligned image/mask pairs, provenance and frozen split manifest. | `[train] dataset_directory` |
| `fit` | Train the segmentation model from an existing validated dataset manifest. | `[train] model_directory` |

`--stage` overrides `train.stage` for that invocation. `--validate-only` checks
input/source metadata and all planned edit bounds without reading voxel payloads
or creating products. It does not verify an existing paired dataset; that full
integrity check runs when `fit` starts. Preparation and fitting are separate
commands; fitting does not silently generate a new cohort.

## The initial target and image definition

The pilot uses **case 260602, frame 1, signed source ID 2922 (`dias_aorta`)**.
The dictionary, DPI hierarchy, named surface and native label scans support this
identity; see [aorta localization evidence](integration/2026-09-17/aorta-localization.json).
Neighboring generic `arteries` IDs are recorded as contacts, not assumed to be
aortic continuations.

The current field of view covers an **aortic arch crop**. Its binary target is
every voxel carrying the specified aorta ID throughout that crop, including
unchanged aorta outside the edit ROI. The orange tube limits geometry edits;
it is not the segmentation mask. The preview shows the full target separately
so that misplaced or overly narrow edit support is visible. This is not yet a
complete whole-thorax or whole-aorta target definition.

Prepared images use `image_method = attenuation_copy_proxy`: accepted edits copy
the original target/recipient attenuation value selected by the reassignment
engine, then convert XCAT per-pixel attenuation to `cm^-1` using source spacing.
This is an engineering proxy, not AI recovery, scanner reconstruction, or CT HU.
The source volumes remain unchanged. Fixed clipping for the model also uses
`cm^-1`; it does not turn phantom attenuation into patient CT.

## Iteration workflow

Run commands from the integrated checkout. The supplied report already occupies
`preview-v1`; choose a fresh `[output] directory`, such as
`outputs/studies/thoracic-aorta/preview-v2`, before rerunning a preview:

```bash
cd /home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16
python3 scripts/train_study.py --config configs/studies/thoracic-aorta.ini --validate-only
python3 scripts/train_study.py --config configs/studies/thoracic-aorta.ini --stage preview
```

1. **Review anatomy first.** Open the generated `report.html`. Check full target
   coverage, original ID, slice coordinates, nearby vessels, tube placement,
   radii and the full-target binary figure. Keep `coordinate_reviewed = false`
   while adjusting the provisional input.
2. **Relocate the edit support.** Update ordered `roi.center_ijk` nodes and their
   matching `radius_mm` values. The path follows the listed order; an aortic arch
   must not be sorted by k. Adjust `preview.slice_ijk` to inspect another location.
   Give `[output] directory` a new name, such as
   `outputs/studies/thoracic-aorta/preview-v2`, and rerun `preview`.
3. **Review the planned sweep.** Defaults propose erosion and dilation at
   `0.5, 1, 1.5, 2` mm, `shape_k = 10`, a Gaussian tube profile with window
   `0.25, 0.75`, plus one unchanged baseline: **nine requested variants**.
   These distances are weighted six-neighbor path budgets, not a percentage
   stenosis or a guaranteed diameter change. Small settings can yield identical
   geometry. Review allowed reassignment tissues as well as the nominal settings.
4. **Record the review before preparing data.** Once the anatomy, ROI and proposed
   ranges are accepted, set `coordinate_reviewed = true`. Keep the reviewed INI
   as the input snapshot. Use new dataset/model directories for a new experiment.
   The initial shipped input stays unreviewed until that decision is made.
   The flag records the review; it does not itself validate anatomical correctness.

A separate plan can be saved without generating variants. First choose a fresh
`[output] directory`, for example `outputs/studies/thoracic-aorta/plan-v1`:

```bash
python3 scripts/train_study.py --config configs/studies/thoracic-aorta.ini --stage plan
```

After the review step, prepare the paired dataset in the configured
`outputs/studies/thoracic-aorta/pairs-v1` directory:

```bash
python3 scripts/train_study.py --config configs/studies/thoracic-aorta.ini --stage prepare --validate-only
python3 scripts/train_study.py --config configs/studies/thoracic-aorta.ini --stage prepare
```

Inspect the resulting manifest and achieved edits before fitting. Each pair has
an aligned float32 attenuation image and uint8 binary aorta mask, with preserved
fine-label/edit evidence and source/configuration hashes. The model input is the
image; neither the target mask nor the lesion ROI is an input channel.
`dataset-manifest.json` records the frozen split and checksums; `source.npz`
preserves the source crop, and `variants/` contains the aligned pairs.

```bash
python3 scripts/train_study.py --config configs/studies/thoracic-aorta.ini --stage fit --validate-only
python3 scripts/train_study.py --config configs/studies/thoracic-aorta.ini --stage fit
```

The model directory contains the best-validation `model.keras`, epoch history
and `model-metadata.json`, including normalization and dataset provenance.

## Splits and training scope

This first runner uses **`split_mode = scenario_only`**: every variant shares one
base anatomy. Its validation/test scenarios cannot establish generalization to
new subjects. All patches and slices inherit their variant's split. Identical
target geometries remain in one split; the baseline and edits that reproduce it
stay in training. Requested fractions can therefore differ from the final counts.
Fitting rejects an empty training or validation partition instead of moving
duplicate geometry across partitions. Test images/masks are not decoded or used
for fitting, normalization or model selection. Dataset integrity checks may
read the held-out archive bytes to verify checksums.

The initial model is a compact **2D U-Net** using axial image patches, fixed
normalization, and a segmentation loss that excludes padded pixels. All native
slices are tiled, including background-only patches; one decoded volume is
cached at a time. This is segmentation within the supplied crop. Whole-volume
localization, independent-anatomy evaluation, `[test]` and `[generate]` commands
remain later work.

## Wahab fitting

The new [segmentation Slurm wrapper](../scripts/slurm/fakect_segmentation_gpu.sh)
targets the confirmed `container_env tensorflow-gpu/2.17` launch pattern and
`~/envs/fakect`. Submit only after reviewing the prepared dataset and selecting a
new model directory:

```bash
mkdir -p logs
sbatch scripts/slurm/fakect_segmentation_gpu.sh \
  /home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16/configs/studies/thoracic-aorta.ini
```

Submit from the integrated checkout, or set `FAKECT_CHECKOUT` to it. The wrapper
installs no packages. The CPU smoke environment uses TensorFlow **2.20.0**;
that synthetic-fixture check is separate from validating a TensorFlow 2.17 GPU
job on Wahab. No real aorta fitting or cluster job has been run for this input.

The broader milestones and research questions remain in the
[thoracic-aorta project outline](THORACIC_AORTA_PROJECT.md).
