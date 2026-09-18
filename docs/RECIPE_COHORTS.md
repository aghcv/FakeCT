# CoA recipe cohorts

The staged training runner now accepts `fakect.recipe-study/1`: a complete ordered
recipe plus named parameter sweeps. The editable starter is
[thoracic-aorta-cohort.ini](../configs/studies/thoracic-aorta-cohort.ini).
It copies the reviewed geometry and manual safeguard choices from the current
curvature recipe. The interactive curvature file remains separate and unchanged.

## Define the discrete parameter space

```ini
[sweep.arch_outer_expand]
distance_mm = 3:9:3

[sweep.coa_narrow]
distance_mm = 2:10:2
```

`start:stop:step` includes both endpoints and must land exactly on the stop.
These ranges mean `3,6,9` and `2,4,6,8,10`. Comma-separated lists are equally
valid, including uneven samples such as `2,3,5,8`.

All axes form a Cartesian product. The example generates 15 complete recipes,
plus one unchanged baseline when `train.include_baseline = true`. Each variant
starts from the original phantom. Only steps within that variant consume the
preceding edited result, in `[recipe] steps` order.

Each `[sweep.NAME]` references `[edit.NAME]`. Supported axes are `distance_mm`,
`shape_k`, and `iterations`. Unspecified settings retain their base values.
For example, `arch_inner_narrow` stays at its configured 7 mm without erosion
safeguards; `coa_narrow` retains its current 10% local floor and connectivity
check. A sweep never enables a safeguard automatically. Distances are requested
budgets in mm, not percentages of stenosis or promised surface displacements.

Zero in a distance sweep disables that step for the corresponding variant.
The baseline disables all steps. An active base `[edit.NAME]` still requires a
positive distance. Iteration values must be integers from 1 to 10; every
resolved recipe must also meet the total pass and crop limits.

`train.max_variants` prevents accidental large grids before opening voxel data.
The current implementation has a hard ceiling of 1,000 variants. This workflow
checks the chosen discrete points, not every real-valued point between them.

## Validate and generate

Run from the integration checkout:

```bash
cd /home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16
python3 scripts/train_study.py --config configs/studies/thoracic-aorta-cohort.ini --validate-only
python3 scripts/train_study.py --config configs/studies/thoracic-aorta-cohort.ini --stage preflight
python3 scripts/train_study.py --config configs/studies/thoracic-aorta-cohort.ini --stage prepare
```

| Stage | Checks or output |
| --- | --- |
| `--validate-only` | Metadata checks for every complete recipe: ranges, tissue policy, crop/halo, pass count. No voxel payloads or outputs. |
| `--stage plan` | Frozen input and JSON inventory of all requested combinations. |
| `--stage preview` | Full interactive report for the current base edit values, with the cohort plan and binary target definition. |
| `--stage preflight` | Native execution of every variant; HTML, CSV and JSON audit; no training-pair NPZs. |
| `--stage prepare` | Independent native executions and paired image/mask NPZs; complete manifest published only after checks pass. |
| `--stage fit` | Existing TensorFlow U-Net runner using a validated prepared dataset. |

Metadata validation cannot predict native target presence, accepted geometry,
recipient availability, or duplicate masks. Inspect `preflight.html` for those
outcomes. CSV rows give each variant's per-pass requested/accepted distance,
proposed and actual changes, and unresolved voxels. An accepted safeguard
distance remains a budget; tissue resistance and reassignment may further
reduce the actual edit. Native execution success does not establish realism.

Preparation repeats native checks; it does not require an earlier preflight
artifact and never trains a model implicitly. A failed preparation keeps its
`INCOMPLETE` marker and diagnostic report without publishing a usable dataset
manifest. Existing directories are never overwritten.

The relevant output paths are independent:

```ini
[output]
directory = outputs/studies/thoracic-aorta/cohort-plan-v1

[train]
preflight_directory = outputs/studies/thoracic-aorta/cohort-preflight-v1
dataset_directory = outputs/studies/thoracic-aorta/cohort-pairs-v1
model_directory = outputs/studies/thoracic-aorta/cohort-model-v1
```

Choose a new directory for the stage you rerun, for example `cohort-preflight-v2`
and `cohort-pairs-v2`. Changing only a stage or output directory does not change
the data fingerprint. Changing data-generating recipe settings or code does.

## What the pairs represent

`train.target_scope = selected_lineage` requires `recipe.roi_role = selection`.
The source target is the selected tissue inside the reviewed main ROI. The
binary target after editing contains its surviving original voxels and added
descendants, including growth outside the ROI. Nearby unselected arteries stay
negative. Original XCAT IDs are optional; this target uses `tissue = artery`
and the reviewed ROI.

This is an ROI-defined aorta-region target, including whichever branches or
neighboring artery voxels intersect that ROI. It does not automatically provide
a complete or anatomically exclusive thoracic-aorta annotation. Source labels,
the catalog hierarchy and ancestry remain available for later refinement.

Each variant archive contains `image` (float32 attenuation in cm^-1), `mask`
(uint8 binary target), original-ID `edited_labels`, change masks, target ancestry
and metadata. Original labels and attenuation are preserved in `source.npz`.
Metadata records every resolved recipe and its accepted per-pass outcomes;
source/catalog/config/code fingerprints support reproduction.

The current image method is `attenuation_copy_proxy`. Accepted edits copy
current-state target or recipient attenuation. It is not learned background
recovery, a physical CT reconstruction, or HU data. Active erosion therefore
requires `assign_surrounding_tissue = true`: diagnostic released labels carrying
placeholder attenuation cannot form these pairs. Normal assignment can produce
less erosion than diagnostic mode when the surroundings resist reassignment.
Preflight makes that difference explicit; it does not loosen stiffness values.

## Splits and the next model experiment

All variants here share one phantom. The explicit `scenario_only` protocol
tests held-out synthetic changes of that anatomy. It does not test generalization
to independent XCAT subjects or patients. Whole volumes and all their slices
remain in one split. Identical realized target masks are assigned as groups
after native execution, including duplicates caused by safeguard backoff;
baseline-equivalent masks remain in training. Preparation requires nonempty
requested holdouts before publishing a completed dataset.

The model architecture and normalization are specified in `[model]`. Once the
paired crops and splits have been reviewed, fitting uses the existing runner:

```bash
python3 scripts/train_study.py --config configs/studies/thoracic-aorta-cohort.ini --stage fit
```

The existing Wahab GPU wrapper also accepts this INI:

```bash
mkdir -p logs
sbatch scripts/slurm/fakect_segmentation_gpu.sh configs/studies/thoracic-aorta-cohort.ini
```

No GPU submission or model fitting is part of validation, preflight or preparation.
The next validation milestone is to repeat reviewed ROIs across multiple source
anatomies and split by anatomy family before assessing broader model performance.
