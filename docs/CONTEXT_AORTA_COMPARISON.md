# Controlled baseline-versus-sibling segmentation experiment

The next experiment tests whether synthetic anatomical variation improves
attenuation-to-aorta segmentation on unseen **phantom families**. It builds on
the successful 0.6 mm, 216×160×160 training-patch diagnostic (Dice 0.97787),
which establishes that the network can learn a patch, not that it generalizes.

The input is [coa-context-comparison.ini](../configs/models/coa-context-comparison.ini).
The original diagnostic implementation and outputs are preserved. New modules
implement the matched comparison and exact evaluation reuse separately.

## Experimental controls

| Setting | Baseline only | Baseline plus siblings |
| --- | --- | --- |
| Training source cases | Same 8 | Same 8 |
| Training scenarios | 8 baselines | 128, including the 8 baselines |
| Training anatomy families | Same 4 provisional families | Same 4 provisional families |
| Updates per epoch | 512 | 512 |
| Updates per case per epoch | 64 baseline patches | 4 patches for each of 16 scenarios |
| Epochs | 20 | 20 |
| Initial weights / physical patches / augmentation seeds | Matched | Matched |
| Validation | Same 32 scenarios from adult family | Same 32 scenarios from adult family |
| Test | Same 32 scenarios from infant family | Same 32 scenarios from infant family |

Both arms use the same 459,377-parameter 3D U-Net, Adam at 0.001, stable
logit-based cross entropy plus soft Dice, 0.6 mm sampling, and a
129.6×96×96 mm physical patch. Inputs are attenuation only. Every training
patch location is drawn from the same baseline-based schedule in both arms;
half are target-centered and half span the full scan. The edited lesion masks
do not determine sampling. Augmentation uses paired i-axis flips and intensity
scales between 0.9 and 1.1.

Validation covers the full scans at epochs 5, 10, 15 and 20. Maximum validation
family-macro Dice selects the checkpoint; ties keep the earliest. Test images
are loaded only after fitting. Both runs use all held-out variants. Report
case/family-level scores alongside scenario averages, since siblings are
correlated. The default split is 8/2/2 case IDs but **4/1/1 provisional anatomy
families**; it is not a 12-independent-patient study.

## Targets and inputs

The frozen context dataset references 12 original whole-body ATN volumes and
192 existing native edited-crop/target pairs. It does not create 192 new
whole-body image files. The targets cover the atlas-defined aortic core and
proximal branches, extending beyond the earlier editing ROI. The independent
[engineering review](integration/2026-09-29/context-target-engineering-review.md)
found no evident blocker, but the T12-midpoint/T1-upper limits, generator root
boundary and branch reach remain provisional. `anatomical_extent_reviewed`
remains false. The explicit INI opt-in permits this engineering comparison.

All 192 targets pass the 0.6 mm connectivity/volume guard. The coarser 1 mm
grid materially breaks three severe infant coarctations and must not be used
for this experiment. Source images are attenuation-copy proxies in cm^-1,
not HU and not reconstructed clinical CT.

## Exact whole-scan evaluation reuse

Siblings share the source volume but differ within compact edited regions.
For each model and case, the evaluator first scans the baseline over every
tile in the full field of view. A sibling can reuse a tile's additive counts
and loss statistics only if both image and target are proven unchanged there.
The proof compares native attenuation overlays, target masks and original
reference masks at their physical crop origins; image-change support includes
the linear-interpolation halo. A conservative numerical margin protects
floating-point boundary cases. Tiles overlapping changes are inferred anew.
Unprovable sharing falls back to a complete scan.

No cache crosses cases, model weights, epochs or calls. No ground-truth bounding
box restricts inference. Far-field false positives therefore still count.
The current validation sharing plan represents 29,696 tiles using 2,036
forwards, about 14.6 times fewer, with no proof fallbacks. This is an exact
computation optimization, not a different evaluation metric or cropped test.

## Commands on Wahab

From the `fakect.26.09.16` worktree:

```bash
python3 scripts/compare_context_aorta.py --config configs/models/coa-context-comparison.ini --validate-only
python3 scripts/compare_context_aorta.py --config configs/models/coa-context-comparison.ini --stage plan
mkdir -p logs
sbatch scripts/slurm/fakect_context_comparison_gpu.sh configs/models/coa-context-comparison.ini
```

The plan freezes the INI, dataset identity, split and implementation once.
The Slurm array runs the two arms serially (`0-1%1`), one GPU each, through
the confirmed TensorFlow 2.17 container. Each arm fits then tests its selected
checkpoint. Existing arm directories are never overwritten or implicitly
resumed. If code or inputs change, use a new output revision and plan.

Monitor using `squeue -u "$USER"`, the corresponding
`logs/fakect_compare.<array>_<task>.out` files, and each arm's `model/history.json`.
The first final metrics appear only after a checkpoint has completed test
evaluation. A running history is provisional.

```bash
python3 scripts/report_context_comparison.py --config configs/models/coa-context-comparison.ini
```

The report can be regenerated during the run. Completed results require
valid checksums and matching model/lock provenance. When both arms finish,
the report checks matching initial weights, update counts and physical patch
event hashes. It is also generated automatically after each successful arm.

Output layout:

```text
outputs/experiments/coa/context-comparison-r1/
  input.ini
  experiment-lock.json
  source/
  baseline_only/model/{initialization,history,training}.json
  baseline_only/model/checkpoint.keras
  baseline_only/evaluation/results.json
  baseline_plus_siblings/model/...
  baseline_plus_siblings/evaluation/results.json
  comparison-report.html
```

Model outputs, source binaries and generated NPZ files remain excluded from
Git. Editable code, INIs, tests and documentation are version-controlled.

## Interpretation and next decision

Compare held-out Dice, precision and recall, per-case errors, and whether
performance deteriorates for severe narrowing or expansion. The first result
answers whether adding these siblings helps this model under a matched update
budget and this synthetic family split. It does not isolate every possible
effect of data size or architecture, and a single held-out infant family cannot
support population-level significance or clinical generalization.

Use this result to decide whether to improve sampling/architecture or expand
the experiment to held-out-family cross-validation and repeated seeds. Test
performance must not be used to select an epoch, threshold or hyperparameter
within this frozen comparison. Clinical CT evaluation remains a separate
future validation step.
