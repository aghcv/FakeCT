# Context segmentation comparison: launch record

The matched comparison was frozen and submitted on **2026-09-29** from branch
`fakect.26.09.16`. Slurm array **6862667** runs the two arms sequentially:

- **6862667_0**: `baseline_only`, 8 training baselines.
- **6862667_1**: `baseline_plus_siblings`, 128 training scenarios.

At launch, task 0 was running on `d2-w4140a-03`; task 1 was waiting on the
one-task-at-a-time array limit. No held-out segmentation result was available.
This record documents launch, not completion. Consult Slurm and regenerate
the report for the current state.

## Inputs and outputs

- Editable input: `configs/models/coa-context-comparison.ini`.
- Immutable input/implementation/split: `outputs/experiments/coa/context-comparison-r1/experiment-lock.json` and `source/`.
- Report: `outputs/experiments/coa/context-comparison-r1/comparison-report.html`.
- Per-arm checkpoints/history/metrics: `<experiment>/<arm>/model/` and `evaluation/`.
- Logs: `logs/fakect_compare.6862667_0.{out,err}` and `logs/fakect_compare.6862667_1.{out,err}`.
- Full protocol and commands: [controlled comparison](../../../docs/CONTEXT_AORTA_COMPARISON.md).

Both arms use 20 epochs × 512 updates, matched starting weights, patch locations,
augmentation seeds and hyperparameters. The 8/2/2 case split preserves 4/1/1
provisional families. Validation uses 32 adult scenarios; testing uses 32 infant
scenarios, only after each arm finishes fitting. The checkpoint is selected by
validation family-macro Dice at epochs 5,10,15,20.

## Evidence permitting launch

1. All 12 native targets passed the engineering mask/provenance/crop review.
   The anatomical boundary and branch policy remains provisional; the explicit
   engineering opt-in is retained and no review status was changed.
2. All 192 scenarios pass the 0.6 mm sampling guard. The full patch sees
   129.6×96×96 mm with attenuation as its only input.
3. Matched-budget, isolation, immutability, loader and learning regression tests
   pass. The context suite run logged 103 tests with four opt-in backend skips;
   the final cached-evaluation checks (16 tests) and reporter checks (10 tests)
   passed separately after their last additions.
4. The exact evaluator covers all 29,696 validation tiles using 2,036 actual
   forwards, with no equality-proof fallback. Source rounding endpoints are
   conservatively protected. Toy and real TensorFlow checks confirm that
   unchanged tiles preserve image/mask/prediction values and that changed tiles
   are recomputed.
5. **GPU benchmark 6862665 completed 0:0** in 2m04s on a V100-SXM2-16GB,
   TensorFlow 2.17.0. Both tiny comparison arms trained, saved, reloaded and
   evaluated successfully; initialized weights, update counts and physical
   sampling hashes matched. Direct and cached scores matched exactly.
6. Benchmark source hashes match the frozen comparison lock. The earlier
   successful diagnostic lock still verifies; its implementation was unchanged.

Benchmark artifacts are in `outputs/qa/context-comparison-benchmark-r2/`.
The initial benchmark attempt `6862639` stopped when an evaluator source
hardening change occurred during its tiny fit; the provenance guard correctly
rejected finalization. The source-stable rerun above is the accepted evidence.

## Runtime and storage

Thirty-two stratified real validation tiles averaged **1.624 seconds** each:
1.094 seconds loading, 0.147 seconds forward computation and 0.329 seconds
metrics, plus preprocessing. Eight timed updates on a training-only baseline
averaged **2.011 seconds**; optimizer computation averaged 0.423 seconds.

Estimated training plus four validation rounds is **9.39 hours per arm**,
or **11.74 hours with a 25% margin**, before test evaluation, crop/proof setup
and checkpoint I/O. These are projections, not guarantees; filesystem load
and allocation can change them. Timing disabled augmentation while production
enables it. The production jobs each have a 28-hour limit and 16 GiB host
allocation. Benchmark TensorFlow GPU peak was **3.16 GiB** and process MaxRSS
was **2.29 GiB**.

No new whole-body sibling copies are generated. Existing compact masks and
edited crops are referenced; full-body probability output remains disabled.
Generated files stay ignored by Git. No training data or previous models were
deleted.

## Current interpretation

No comparison accuracy or augmentation benefit is claimed at launch. The
single-patch diagnostic remains a resubstitution check. This new experiment
can measure the effect of these synthetic siblings under a matched training
budget and a fixed synthetic split; it cannot establish clinical CT
generalization or population-level significance from one held-out family.

```bash
squeue -j 6862667
python3 scripts/report_context_comparison.py --config configs/models/coa-context-comparison.ini
```

The HTML can show provisional history during fitting; completed metrics require
checksum and provenance checks. It is regenerated automatically after each
successful arm. Changed code or parameters require a new experiment revision.
