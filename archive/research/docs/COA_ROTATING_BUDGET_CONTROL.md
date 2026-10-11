# CoA classifier: fixed 200-epoch control

This experiment tests whether the short final refit budgets contributed to the
weak generalization in `rotating-evaluation-r2`. The user selected 200 epochs
before this run. It is a development cross-validation control; the previous
outer outcomes have already informed development.

Input: `archive/research/configs/models/coa-rotating-budget200.ini`.
Output: `outputs/experiments/coa/rotating-budget200-r1`.

## Fixed comparison

- Five primary seed-42 family folds; the same 60 selected edited scenarios from
  ten development phantoms.
- Each fold uses 48 training images and 12 held-out images. All other siblings
  of the held-out anatomy family remain excluded.
- Each completed fold's locator and exact image-derived crop coordinates are
  reused. Atlas masks do not become classifier channels or recenter test crops.
- The classifier and Adam optimizer start fresh. Its initial weight fingerprint
  must match the previous fold, and its initial predictions must agree with the
  same reviewed inputs.
- The graph, physical resolution, learning rate `0.001`, batch of two matched
  siblings, regression weight `1.0`, gradient clipping and shuffle sequence stay
  unchanged. Only the final classifier epoch budget changes from 8–29 to 200.
- Each fold performs 24 optimizer updates per epoch, totaling 4,800 updates.
- Original infant cases `260612` and `260617` remain unopened for model fitting
  and prediction.

## Checkpoints and evaluation

Training metrics and runtime memory checks are recorded every epoch. Model
archives at epochs 50, 100, 150 and 200 preserve weights; they do not include a
resumable Adam state. The declared performance checkpoint is epoch 200.

No intermediate held-out prediction or outer early stopping is performed.
Held-out inference starts only after the final checkpoint and receipt are
fixed. Coverage failures would remain visible in the evaluated panel.

The new report compares the previous final budget with epoch 200 using exactly
the same scenarios. It distinguishes pooled AUROC from mean family AUROC and
retains the descriptive severity proxy. Mean ROC curves preserve the full
ordered empirical coordinates, correcting the interpolation issue in the
previous report. Family spread is descriptive; siblings are not independent
patients. A larger training budget may improve fitting without improving
generalization.

## Commands

Run from the integrated worktree:

```bash
python3 scripts/research/train_coa_budget.py --config archive/research/configs/models/coa-rotating-budget200.ini --validate-only
python3 scripts/research/train_coa_budget.py --config archive/research/configs/models/coa-rotating-budget200.ini --stage plan
python3 scripts/research/train_coa_budget.py --config archive/research/configs/models/coa-rotating-budget200.ini --stage list --ids-only
python3 scripts/research/train_coa_budget.py --config archive/research/configs/models/coa-rotating-budget200.ini --stage report
```

Planning creates a new immutable directory. Once frozen, change the output
revision before changing parameters. The training CLI rejects existing folds,
changed parent crops/checkpoints, changed code, or changed input metadata.

The Slurm entrypoint is `scripts/research/slurm/fakect_coa_budget_gpu.sh`. The production
sequence is a synthetic full-size GPU/memory smoke, a five-task fit array
dependent on that smoke, and a CPU report dependent on completion of the array.
The array may run at most two folds concurrently. Each fit requests one GPU,
four CPUs, 24 GiB host memory and a four-hour time limit. Existing receipts
suggest roughly 75–85 minutes per fold; this is an estimate, not a guarantee.

Submission IDs and dependency receipts will be saved in the experiment's
`submission.json`. No additional volumetric training datasets are generated.

## Submitted on 2026-10-04

- GPU smoke `6907420`: completed successfully (`0:0`, 54 seconds).
- Classifier array `6907424`, tasks 0–4, at most two concurrent: depends on
  successful smoke completion. The first two folds started successfully.
- CPU report `6907428`: depends on completion of every array task, including
  unsuccessful tasks, so any fitting failures remain visible.

All 28 new tests passed in the production TensorFlow 2.17 environment, with
targeted checks rerun after the final assertions/guard edits. Test receipts are
stored under the experiment's `checks/` directory. The full-size smoke verified
single train/predict traces, saved-model reload agreement and a physically
resident 50-sample cache during an optimizer update. Its peaks were about
7.8 GiB host RSS and 1.4 GiB GPU memory, within the inherited guards.

Monitor with `squeue -j 6907424,6907428`. The experiment's `report.html` redirects
to the latest immutable report revision and will update after the report job.

## Results evaluated on 2026-10-05

All five tasks and the CPU report completed with exit code `0:0`. Each
classifier completed exactly 200 epochs and 4,800 optimizer updates. All 20
milestone archives and final checkpoint hashes verified; source, parent input,
initialization and report provenance checks passed. The completed comparison is
in `outputs/experiments/coa/rotating-budget200-r1/review-r2/report.html`.

| Measure | Previous 8–29-epoch refits | Fixed 200 epochs |
|---|---:|---:|
| Training accuracy across folds | 50.0–95.8% | 100% in all five |
| Equal-family mean AUROC | 0.4889 | 0.5556 |
| Pooled out-of-fold AUROC | 0.5144 | 0.5311 |
| Held-out balanced accuracy | 50% in every fold | 50% in every fold |
| Sensitivity / specificity | 60% / 40% | 50% / 50% |
| Probability-clipped BCE | 0.8137 | 5.4175 |
| Severity proxy MAE | 26.54 pp | 21.04 pp |
| Locator coverage checks | 60/60 passed | Same 60/60 passed |

The 200-epoch run resolves poor training fit but does not provide a reliable
unseen-anatomy classifier. Perfect training accuracy was sustained from epochs
37–90 onward. Held-out outputs became very confident; their probability error
increased considerably. The report clips probabilities using float32 epsilon.
Computing BCE stably from the saved raw logits instead yields 6.3896 at epoch
200, versus 0.8137 previously, strengthening the overconfidence finding.

Held-out family AUROCs at epoch 200 are 1yr: 0.000; 5yr: 0.833; 10yr: 1.000;
15yr: 0.500; adult: 0.444. The 10-year fold ranks positives above negatives but
assigns every scenario a probability below 0.000003. The 5-year fold labels
every scenario positive. Four individual phantoms have reversed ranking
(AUROC zero). Only 17/30 matched pairs give the positive sibling a higher
logit. Thus adjusting one global probability threshold cannot resolve all
observed failures.

The aggregate severity MAE reduction is driven by recovery from the previous
adult outlier: adult MAE improves from 77.20 to 21.55 percentage points, while
all four younger-family MAEs worsen. The 10yr, 15yr and adult folds predict
almost zero severity on held-out scenarios. Their final training MAEs also
match the zero-output baseline almost exactly. These observations suggest
saturation of the sigmoid severity output; the mechanism has not been proved
by a gradient analysis or an ablation.

The matched control verifies equal starting weights within reload tolerance
and identical training-pair order. Numerical training trajectories differ
after small initial rounding differences, so they are not described as
bit-for-bit repeats. This remains a single-seed development comparison across
five provisional anatomy families. It does not establish virtual-cohort benefit
or clinical generalization; the reserved infant pair remains unopened.

The next recommended experiment is a fresh classification-only loss control
with `regression_weight = 0`, retaining the five folds, reviewed crops, seed,
architecture, learning rate and fixed 200-epoch budget. Its severity output
would be untrained and must not be evaluated as a severity prediction. This
would test whether the auxiliary task affects classification. If necessary,
test image-derived anatomy-scale normalization or scale augmentation as a
separate subsequent control. No further training was submitted during this
results evaluation.

## Classification-only control authorized on 2026-10-05

The next control uses `archive/research/configs/models/coa-classification-only200.ini` and a
separate output revision, `outputs/experiments/coa/classification-only200-r1`.
It compares directly with the completed `rotating-budget200-r1` rather than
the earlier short-budget rotation. Only `regression_weight` changes, from 1
to 0, with a fresh classifier and optimizer for each of the same five folds.

The graph retains its auxiliary head to preserve starting weights and optimizer
slots. Its outputs are untrained: training metrics and held-out exports omit
severity performance. Raw initialization outputs remain only as parity evidence.
TensorFlow tests check that different finite severity targets produce identical
classification-only updates, and that the auxiliary output layer weights remain
unchanged while the shared representation learns.

Each fold still uses 48 training and 12 held-out images, the reviewed locator
proposals, seed 42, learning rate 0.001, the same paired ordering and 200 epochs
(4,800 updates). The new report compares training BCE/accuracy, pooled and mean
family AUROC, sensitivity/specificity, and balanced accuracy. Held-out inference
uses only the declared final epoch; the infant pair remains reserved.

The implementation is additive: the completed experiments' frozen source files
and results are preserved. Both the original rotation ancestry and completed
200-epoch comparison metadata are checksum-pinned in the new lock. Storage reuses
existing sibling images; no new whole-body arrays are generated.

```bash
python3 scripts/research/train_coa_classification.py --config archive/research/configs/models/coa-classification-only200.ini --validate-only
python3 scripts/research/train_coa_classification.py --config archive/research/configs/models/coa-classification-only200.ini --stage plan
python3 scripts/research/train_coa_classification.py --config archive/research/configs/models/coa-classification-only200.ini --stage list --ids-only
python3 scripts/research/train_coa_classification.py --config archive/research/configs/models/coa-classification-only200.ini --stage report
```

Submission follows a production-size GPU smoke, five-task GPU array with at most
two concurrent tasks, and CPU report dependent on array completion. Submission
IDs and checks are recorded in this run's `submission.json` and `checks/`.

Submitted jobs on 2026-10-05:

- GPU smoke: `6916265`.
- Five-fold classifier array: `6916266`, with concurrency two and an
  `afterok:6916265` dependency.
- CPU comparison report: `6916267`, with an `afterany:6916266` dependency.

All 29 new tests passed in TensorFlow 2.17 before submission. The frozen run
includes 62 source files and 101 parent/comparison dependency hashes. Its initial
report is pending at `outputs/experiments/coa/classification-only200-r1/report.html`.
The CPU job publishes a new immutable report revision after the array finishes.

The GPU smoke completed successfully (`0:0`, 75 seconds). Peak host RSS was
about 7.8 GiB and peak GPU memory about 1.4 GiB; train and predict each retained
one trace and saved-model reload parity passed. Array tasks 0 and 1 then started;
the other three await the concurrency limit. No model performance is available
for this control until fitting and final held-out inference complete.

```bash
squeue -j 6916265,6916266,6916267
```

## Completed classification-only results evaluated on 2026-10-05

All five GPU folds and the dependent report completed with exit code `0:0`.
The completed comparison is `outputs/experiments/coa/classification-only200-r1/review-r2/report.html`.
Each fold completed 200 epochs and 4,800 updates. The final epoch was used
without selecting a checkpoint on held-out performance. The two reserved infant
phantoms remain unevaluated.

| Metric | Joint loss, 200 epochs | Classification only, 200 epochs |
| --- | ---: | ---: |
| Final training accuracy, every fold | 100% | 100% |
| Pooled held-out AUROC | 0.5311 | 0.4722 |
| Mean family AUROC | 0.5556 | 0.4278 |
| Balanced accuracy at probability 0.5 | 50.0% | 53.3% |
| Sensitivity | 50.0% | 46.7% |
| Specificity | 50.0% | 60.0% |
| Correct scenarios | 30/60 | 32/60 |
| Locator coverage passes | 60/60 | 60/60 |

| Held-out family | Joint AUROC | Classification-only AUROC |
| --- | ---: | ---: |
| 1 year | 0.000 | 0.000 |
| 5 years | 0.833 | 0.139 |
| 10 years | 1.000 | 0.750 |
| 15 years | 0.500 | 0.750 |
| Adult | 0.444 | 0.500 |

The net two additional correct scenarios come from phantom 260616. Nine of ten
held-out phantom IDs still receive a constant classification across all six
siblings. In 30 matched same-phantom positive/negative pairs, correct ordering
fell from 17 to 11. Removing auxiliary severity loss therefore did not produce
consistent improvement, and does not support that loss as the main explanation
for poor generalization. Perfect training fit with poor unseen-anatomy performance
continues despite successful crop coverage. These results describe five correlated
family folds with one seed; the 60 siblings are not 60 independent patients.

Next, audit whether changes in attenuation and local caliber remain discernible
in the actual resampled network inputs, especially for small anatomy. Also assess
whether the baseline-relative label (at least 20% local target-volume loss) has
an identifiable after-only image signature across baseline calibers. This is an
observability question, not evidence that the labels are wrong. Then evaluate
image-derived anatomy-scale normalization or scale augmentation as a separate
matched control, keeping family separation and the reserved infant pair intact.
Another larger epoch budget is not justified by this result. No further training
was submitted during this evaluation.

The additive receipt `checks/results-evaluation.json` records the final-history,
coverage and per-phantom prediction checks. Existing frozen model and report
artifacts were preserved.

## Revised learning objective: net enlargement and narrowing

The user's clarified objective on 2026-10-05 is to localize regions that are
larger or narrower than the original reference anatomy, without distinguishing
CoA from arch changes. Existing aggregate net-change targets are appropriate;
per-condition masks and per-step spatial replay are not required for this task.

For aligned binary reference mask `B` and final sibling mask `A`:

```text
changed = A XOR B
added   = changed AND A  # enlargement relative to the reference
removed = changed AND B  # narrowing relative to the reference
```

These are exactly the net added/removed semantics already exported by the
two-stage dataset adapter. Changes canceled by subsequent edits are absent,
as appropriate for evaluating the final anatomy. CT intensity differences
remain a separate possible signal; their sign does not determine anatomical
addition versus removal.

The proposed deployable task is focused sibling CT -> predicted added/removed
voxel maps, using true paired differences as training targets. A baseline-assisted
reference experiment may receive the true differences, with that input advantage
declared separately. At inference without a baseline, the model must estimate
the maps; it cannot calculate the true XOR directly. The crop must include removed
voxels outside the surviving aorta. No new training is submitted by this scope
clarification.
