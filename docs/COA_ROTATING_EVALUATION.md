# Focused CoA rotating evaluation

**Date:** 2026-10-04. **Status:** implementation and metadata validation;
no rotating model-performance results yet. The first production memory smoke
and an isolated engineering fold are required before the remaining fits.

The new [input file](../configs/models/coa-rotating-evaluation-v2.ini) separates
model-selection validation from final predictions. It references the existing
frozen cohort and annotation revision; it does not regenerate images. The
classifier receives attenuation and image-validity channels. Every stage fits
a **fresh image-based locator and focused classifier** on permitted training
phantoms. Previously trained locator weights and predictions are not reused
to choose held-out crops.

## Partitions and selected scenarios

Twelve phantom IDs form six **provisional** age/sex groups. Template names
identify adult age 50 and ages 1, 5, 10 and 15 years. Infant age is unspecified;
the infant pair, 260612/260617, remains outside all development fitting,
validation and final predictions. Age grouping alone does not prove shared
anatomical ancestry or independence.

Each of the ten development phantoms contributes six edited scenarios: three
distinct negative/positive pairs. Each pair has identical ascending-dilation
and arch-edit settings, varying CoA. The negative is the closest achieved loss
below 20%; positives span low, middle and high loss across three sorted
companion configurations. Original phantoms provide separate locator
references. **All other siblings of a held-out phantom remain held out**, even
when they are absent from the selected six-scenario panel.

The reference class is achieved local CoA target-volume loss ≥20%, an
engineering proxy. The prediction decision threshold is separately fixed at
0.5. Neither is a validated clinical severity threshold.

| Protocol | Planned stages, seed 42 | Training / validation / final-test phantom counts | Scope |
| --- | ---: | --- | --- |
| `family_holdout`, inner selection | 20: four inner runs for each of five outer development families | 6 / 2 / 2 | Primary; outer cases produce no predictions during inner selection |
| `family_holdout`, fresh outer refit | 5 | 8 / 0 / 2 | Primary final predictions; budget selected only from inner validation |
| `same_age_other_sex` | 10 | 8 / 1 / 1 | Secondary unseen-phantom analysis; V selects models, P supplies final predictions |
| `nearest_age_same_sex` | 10 | 8 / 1 / 1 | Secondary comparison with the same P scenarios |

Two additional infant IDs are excluded in every row. Refits use the rounded
median selected locator/classifier epochs across their four completed inner
runs, then start fresh and retain the final fixed-budget checkpoint. They do
not early-stop on the outer test family.

Nearest available same-sex age uses minimum absolute year difference, then
younger age, then case ID. At age 10, the 5-year and 15-year candidates tie, so
the configured rule selects age 5. Adult age 50 selects age 15; a 35-year gap
is reported rather than described as a clinically close comparison. Reserved
infants are not candidates and are not assigned an invented numerical age.

## Evidence that changes execution eligibility

The current original target-mask metadata contains three identical-digest
pairs: **260613/260618 (1 year), 260614/260619 (5 years), and 260615/260620
(10 years)**. This does not establish that the whole CT volumes or all
anatomical tissues are identical. The execution policy nevertheless prohibits
an identical original target mask, or a shared attenuation source, from
crossing training, validation and final-test roles.

All 25 primary stages satisfy this policy. Four same-age secondary stages and
two nearest-age secondary stages are eligible. The other **14 secondary
stages are explicitly excluded**, leaving **31 eligible fits from 45 planned
stages**. They are not silently dropped or reported as failed predictions.
Both secondary protocols currently have eligible final predictions for the
two adult targets; additional age comparisons require independently reviewed
source anatomy or a revised family-preserving protocol.

These checks strengthen the conservative grouping with actual target-mask
evidence. They do not change completed pilot results or certify every remaining
phantom as anatomically independent. The plan and report preserve provisional
ancestry status and every exclusion reason.

## Commands

Run from `/home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16`.

Validate metadata, annotations, isolation and source references without fitting:

```bash
python3 scripts/train_coa_rotations.py \
  --config configs/models/coa-rotating-evaluation-v2.ini --validate-only
```

Freeze the input, source code and exact stage manifest; create the initial
review report:

```bash
python3 scripts/train_coa_rotations.py \
  --config configs/models/coa-rotating-evaluation-v2.ini --stage plan
```

List exact stage IDs and partitions:

```bash
python3 scripts/train_coa_rotations.py \
  --config configs/models/coa-rotating-evaluation-v2.ini --stage list
```

`--ids-only` prints one ID per line for orchestration; the complete frozen
manifest/report records eligibility. Never submit excluded stages. Fitting
requires the experiment's own successful production-size GPU memory smoke:

```bash
sbatch scripts/slurm/fakect_coa_rotations_gpu.sh \
  configs/models/coa-rotating-evaluation-v2.ini smoke
```

After that check passes, submit one exact eligible inner stage. For example:

```bash
sbatch scripts/slurm/fakect_coa_rotations_gpu.sh \
  configs/models/coa-rotating-evaluation-v2.ini \
  family_holdout/xcat_reference_1yr/seed-42/inner-xcat_reference_10yr
```

The equivalent stage command inside the TensorFlow GPU environment is:

```bash
python scripts/train_coa_rotations.py \
  --config configs/models/coa-rotating-evaluation-v2.ini --stage fit \
  --run-id family_holdout/xcat_reference_1yr/seed-42/inner-xcat_reference_10yr
```

An outer refit refuses to start until all four source inner runs have completed
without evaluating the outer cases. Preserve started/failed stage folders;
changed settings or retries belong to a new experiment revision.

Refresh the review report after completed or failed stages:

```bash
python3 scripts/train_coa_rotations.py \
  --config configs/models/coa-rotating-evaluation-v2.ini --stage report
```

The expected review location is
`outputs/experiments/coa/rotating-evaluation-r2/report.html`. Planned, running,
failed, completed and excluded stages remain distinguishable. Inner validation
receipts support checkpoint selection but do not become final-test ROC points.

## Runtime and reporting

The production Slurm script requests one GPU, four CPUs, 24 GiB host RAM and
eight hours. Runtime guards are 16 GiB host and 12 GiB GPU; the image cache is
bounded at 6 GiB. Cached data remain attenuation/validity arrays, with shared
whole-body source images and immutable sibling crop overlays. These limits
must pass the largest-stage smoke before substantial fitting.

The initial settings are 120 locator epochs, at learning rate 0.0003, and up to
50 classifier epochs, at learning rate 0.001. Classification and the secondary
local-volume-loss proxy retain the current focused model; no improvement is
claimed before the rotating fits complete.

Final reports show each held-out family or phantom, empirical ROC, fixed-cutoff
sensitivity/specificity, errors and predictions versus achieved morphology.
Only completed final-prediction stages enter those measurements. Family curves
use a common FPR grid and equal-family averaging; pooled AUROC is a separate
quantity. Seeds describe optimization variation, and siblings are not counted
as independent originating anatomies. With five development families, show
individual results and descriptive variability rather than interpreting a
scan-level confidence interval as anatomy-generalization uncertainty.

The current architecture and task protocol were chosen using earlier
development work, including adult validation. Rotating these same development
families therefore produces **development cross-validation**, not a new
untouched confirmation study. The infant pair remains a candidate for later
confirmation after the protocol is frozen; clinical CT confirmation requires
separate data.

Population-size/diversity arms and the poster study remain defined separately
in [the poster protocol](COA_POSTER_AND_POPULATION_STUDY.md). This rotating
implementation supplies the isolation and reporting foundation for those
comparisons without claiming that their performance effects have been measured.


## Submitted development workflow — 2026-10-04

The reviewed input is `configs/models/coa-rotating-evaluation-v2.ini`, frozen in
`outputs/experiments/coa/rotating-evaluation-r2`. The earlier r1 directory is a
preserved preliminary planning snapshot; no training was submitted from it.

The final production TensorFlow 2.17 suite passed **40 tests**. Its log and
validation receipt are in the experiment's `checks/` directory. The independent
full-size GPU smoke, job **6902347**, completed successfully: 6,075 MiB of
physically touched cache remained resident during a gradient update; host peak
was 8,003.12 MiB and GPU peak 1,473.54 MiB. Both models demonstrated nonzero
gradients, fixed train/predict trace counts and save/reload parity.

| Job or array | Purpose | Gate |
| --- | --- | --- |
| 6902348 | First inner-family engineering fit | Successful GPU smoke |
| 6902349, indices 0–18, at most two concurrently | Remaining 19 inner-family fits | Successful first fit |
| 6902350, indices 0–5, one concurrently | Six eligible secondary age/sex fits | Successful first fit |
| 6902351, indices 0–4, one concurrently | Five fresh outer-family refits and predictions | All 20 inner fits successful |
| 6902352, CPU only | Assemble reports after completion or failure | All fitting arrays terminal |

`submission.json` pins stage IDs, dependencies and job IDs. The array gate
was verified against scheduler membership and [official Slurm array dependency
semantics](https://slurm.schedmd.com/job_array.html): `afterok` on an array ID
requires every task to finish successfully. Failed prerequisite jobs cancel
dependent fitting jobs rather than starting a broader run.

The report entry is `outputs/experiments/coa/rotating-evaluation-r2/report.html`.
Every refresh creates a new `review-rN` artifact, preserving earlier reports.
Its optional secondary exclusions do not block the primary family workflow.
No final ROC curve or population benefit is claimed before measured outer
predictions exist. The infant pair remains outside this development study.
