# Selecting a model evaluation from the INI

The separate
[two-model pathology pilot](integration/2026-09-28/pathology-voxel-pilot.md):
CT-to-aorta segmentation followed by mask-to-reference-shape reconstruction.
Added/removed regions come from comparing the reconstructed and input masks.
It defers centerline/severity targets. Use the
[two-stage training guide](TWO_STAGE_TRAINING.md) and `train_two_stage.py` for
that implemented engineering pilot. The commands below retain the independent
binary-aorta evaluation interface and its verified-family requirements.

Use [coa-cross-validation.ini](../configs/models/coa-cross-validation.ini) to choose a split strategy, training repeats and a fixed prediction threshold. It consumes published image/target pairs. Dataset preparation and phantom editing remain separate.

The current backend evaluates **binary aorta segmentation**. The [condition-classification study plan](integration/2026-09-28/coa-family-cross-validation-plan.md) describes the additional labels, model outputs and severity-response analysis still needed. The latest radius115 pairs are not published, so only provisional split preview is available for that real cohort until preparation and family verification are complete.

| INI strategy | What runs |
| --- | --- |
| `holdout` | One exact whole-family split with the requested distinct-case counts |
| `leave_one_out` | Every family becomes the outer test family once; a separate development family supplies validation for checkpoint selection |
| `nested_leave_one_out` | Every family becomes the outer test family; all other families rotate through inner validation, followed by a fresh fit on all development families |

Choose one of these `[evaluation]` blocks. Keep the shared `[experiment]`, `[split]`, `[preprocess]` and `[model]` sections from the supplied input.

```ini
[evaluation]
strategy = holdout
case_counts = 8,2,2
seeds = 42
threshold = 0.5
planning_project = configs/cohorts/coa/project-radius115.json
```

```ini
[evaluation]
strategy = leave_one_out
validation_families = 1
seeds = 42
threshold = 0.5
planning_project = configs/cohorts/coa/project-radius115.json
```

```ini
[evaluation]
strategy = nested_leave_one_out
seeds = 42
threshold = 0.5
planning_project = configs/cohorts/coa/project-radius115.json
```

`case_counts` means training, validation, testing **base case IDs**. Every descendant and related case remains in its family. Counts that require splitting a family are rejected rather than approximated silently. The current twelve cases form six provisional two-case groups, so an 8/2/2 assignment is possible if those group identities are verified. The split planner counts repeated receipts for one case only once.

`seeds = 42,1729,2026` repeats the entire experiment with three training seeds. It preserves the same family assignments, controlled separately by `[split] seed`. During an evaluation run the old validation/test fractions are not used for its final assignments. Existing model INIs without `[evaluation]` retain their original single-split behavior.

Nested evaluation currently selects the **epoch budget**, with other model settings fixed by the INI. For each outer fold and seed, it takes the median best epoch from the inner validation fits, rounds up, initializes a new model and trains on every development family for that many epochs. Outer test data never select a checkpoint or epoch count. This is not an architecture or hyperparameter grid search.

For the six provisional families, the fit counts per training seed are one for holdout, six for leave-one-out, and thirty inner plus six final fits for nested leave-one-out. During nested selection the counts are eight/two/two; final refitting uses ten development and two test case IDs. Provisional age/template groups do not establish six independent biological subjects.

Preview the design without loading datasets or running TensorFlow:

```bash
python3 scripts/model_experiment.py --config configs/models/coa-cross-validation.ini --preview-splits
```

The output explicitly says `planning_only=true`, `families_verified=false` and `ready=false`. It shows the proposed assignments and fit budget. The inventory is only a preview source; real execution requires reviewed family provenance in the registry.

Once paired datasets have been prepared and registered, validate and freeze the experiment:

```bash
python3 scripts/model_experiment.py --config configs/models/coa-cross-validation.ini --validate-only
python3 scripts/model_experiment.py --config configs/models/coa-cross-validation.ini --stage plan
```

Validation reports all missing/unverified datasets and creates no output directories. Planning checks payloads once and saves `input.ini`, `evaluation-lock.json` and its checksum in `[experiment] output_directory`. Fold locks retain every source dataset, assign complete families, and pin receipts and model-time preprocessing. A change to input or evaluation implementation requires a new experiment revision.

Run fitting in the configured TensorFlow environment. The command itself does not submit a Slurm job:

```bash
python3 scripts/model_experiment.py --config configs/models/coa-cross-validation.ini --stage fit
python3 scripts/model_experiment.py --config configs/models/coa-cross-validation.ini --stage test
```

`fit` trains the requested folds and seeds without reading their test payloads. Completed models can be reused only when their locks, settings and checkpoints match. An interrupted nonempty model directory is reported explicitly; there is no automatic checkpoint-resume claim. `test` first requires every outer model, then evaluates only test samples and creates:

- `folds/outer-*/seed-*/evaluation.json`: per-variant held-out metrics and model/input checksums.
- `evaluation-summary.json`: case, family, seed and overall results.
- `evaluation-report.html`: a portable table of family results and the overall average.

Rebuild the report from complete, unchanged results with:

```bash
python3 scripts/model_experiment.py --config configs/models/coa-cross-validation.ini --stage summarize
```

The report averages variants within each case, cases within each family, repeated seeds within family, and finally gives every family equal weight. Metrics are Dice, IoU, precision and recall from full evaluated volumes; padded pixels never enter counts. Empty-mask handling is explicit in the saved metadata. No confidence interval treats siblings or repeated seeds as independent patients. Broad severity grids and deliberately oversampled boundary stress tests will require separately specified summaries once those annotations are available.

The `threshold` field is the binary segmentation probability cutoff. It does not define intermediate/severe morphology. The runner does not yet export the three pathology probabilities, lesion-specific masks, calibrated clinical risk or severity-response curves. These belong to the next condition-annotated dataset/model extension, using the same fold and provenance infrastructure.
