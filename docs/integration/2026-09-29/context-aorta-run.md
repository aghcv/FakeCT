# Larger-context aorta development — 2026-09-29

## Purpose

Investigate the weak initial attenuation-to-aorta segmentation by first making target coverage consistent and increasing physical 3D context. This development does not change Model 2. The prior `two-stage-r2` source fingerprints and frozen dataset checksum were verified unchanged.

## Implemented

- Broader atlas-derived targets independent of the edit ROI, with frozen synthetic additions/removals transferred to the new reference.
- Shared source whole-body attenuation plus existing sibling edit overlays, loaded in bounded blocks without whole-body copies.
- INI-controlled volumetric U-Net, larger physical blocks, family-grouped holdout assignment, immutable run records, validity-masked BCE + Dice, and consistent whole-volume validation/test tiling.
- Separate fixed-training-patch learning diagnostic, baseline anatomical review report and prediction/error diagnostic report.

Primary instructions: [CONTEXT_AORTA_SEGMENTATION.md](../../CONTEXT_AORTA_SEGMENTATION.md).

## Validation

The related regression suite passed 142 tests with one optional TensorFlow test skipped. That optional loss/gradient/checkpoint test separately passed with real TensorFlow enabled. Four anatomical-report tests also passed. The loader reproduced existing native edited attenuation exactly in tested real samples. Final real-data preparation and diagnostic measurements are recorded below when available.

No files were deleted for storage management. New datasets, models and reports remain under ignored `outputs/` paths.

## Prepared data and geometry

All **12 cases / 192 scenarios** passed broader-mask transfer and release integrity checks. The new data directory occupies approximately **27 MiB** because attenuation references and frozen edit crops are shared rather than duplicated. Source whole-body ATN/ACT files remain unchanged.

Both model plans resolve to 128 training, 32 validation and 32 test siblings (8/2/2 case IDs, grouped into provisional anatomy families). The 128×96×96 mm patch requires a worst-case native interpolation slab of 73,728,000 bytes; this bound excludes model activations and caches. Complete-volume evaluation requires 29,696 validation tiles and 1,440 test tiles at this configuration, so full-cohort fitting should be budgeted separately from the diagnostic.

The baseline geometry audit found native targets connected in all 12 cases. Resampling to 1 mm changes total target volume by −0.794% to +0.578%, and branch volume by −3.058% to +0.466%. Six cases acquire small disconnected fragments (1–4 mm³ each); their largest components still retain at least 99.804% of target volume. This is a small-volume but real sampling artifact. The audit is at `outputs/qa/context-aorta-r1.json`.

In additional native-grid checks, baseline and first edited 260613 samples reproduced all 2,003,468 frozen attenuation-crop voxels exactly, with maximum error zero. Independent nearest-neighbor mask sampling matched the production loader.

## Initial diagnostic exposed a numerical failure

Slurm job `6861416` completed in 4m44s (1,343,196 KiB maximum reported host RSS). The 459,377-parameter model used TensorFlow 2.17 on a V100. Its initial Dice was 0.01669; after 320 updates it predicted background everywhere (Dice 0). The fixed patch contains 9,799 target voxels among 1,179,648 valid voxels.

The diagnostic identified a numerical failure in probability-clipped BCE: clipping an extremely wrong saturated sigmoid prediction makes its BCE gradient zero, while the sigmoid Dice gradient is also negligible. The plateau loss was exactly consistent with this failure. The failed run and its report remain at `outputs/experiments/coa/context-diagnostic-r1/` for comparison. A new revision replaces probability-clipped BCE with stable logits-based cross-entropy while keeping the image, target, network capacity, seed and learning rate unchanged.

## Severe-narrowing sampling check

The all-scenario audit found three native-connected 260612 siblings that separate into substantial pieces at 1 mm: `recipe-802c366a0de61096`, `recipe-271e7c54a43e9f0c` and `recipe-a633c86e0630ed40` (see exact sample IDs in `outputs/qa/context-aorta-r1-all-scenarios.json`). Each includes coarctation distance 2.961 mm; their native grid is 0.4 mm. Roughly 37–38% of their resampled target lies outside the largest connected component, despite less than 1% volume drift. Total volume preservation alone is therefore insufficient.

A cohort planning guard rejects material resampling-induced fragmentation. The final cohort grid is selected only after a finer-spacing comparison across all scenarios; the learning diagnostic remains on the nonsevere 260613 baseline.

The stable-loss regression covers logits ±40 and ±1000: wrong saturated predictions retain corrective gradients, invalid padding has zero gradient, and an actual saturated network head updates after one optimizer step. The public saved model continues to output sigmoid probabilities. The repeat diagnostic changes this numerical loss implementation while retaining the original 1 mm patch and learning rate.

The anatomical report is at `outputs/reports/context-aorta-r1/anatomy/index.html`: 12 scenes, no omitted surfaces, native slice overlays, independent visibility and 15/45/80% opacity. Firefox verified local loading, case switches, opacity controls and synchronized slice images. WebGL rasterization could not be verified on the login node, which lacks a usable WebGL context; adult and infant slice overlays were visually inspected separately.

## Selected cohort grid

The final independent audit uses exactly the production loader's floating-point operation order. At **0.6 mm, all 192 scenarios pass** the connectivity criteria. Maximum foreground outside the largest component is 0.0491%; the largest secondary component is 2.592 mm³. Total target-volume drift is −0.7929% to +0.3972%. The production guard reproduces the component counts exactly; physical-volume arithmetic agrees within floating-point tolerance.

The final cohort INI uses **216×160×160 voxels at 0.6 mm**, giving **129.6×96×96 mm** context. All three severe-case masks were independently compared with actual loader output at this grid: 1,132,970 voxel positions per case, zero differences. Evidence is under `outputs/qa/context-aorta-r1-finer-grids-r2.json`, `context-aorta-r1-loader-grid-parity-r2.json`, and `context-aorta-r1-production-grid-guard.json`.

A foreground-sampling boundary regression also corrected candidate verification to use the loader's exact coordinate arithmetic and field-of-view validity. This does not change loaded image or target arrays. The repeat diagnostic's input patch is compared directly with the first run to verify its numerical comparison remains controlled.

## Stable-loss repeat and checkpoint compatibility

The repeat (`context-diagnostic-r2`, Slurm `6861500`) reached best training-patch Dice **0.972583** at epoch 38. Its finalization stopped because TensorFlow 2.17 / Keras 3.4.1 reloads a single model output as a one-element symbolic list. The auxiliary probability/logit view introduced an extra dimension. This was an interface-shape failure; it was not nonfinite weights or invalid probabilities.

GPU probe `6861582` loaded the preserved checkpoint without retraining. Canonical flat input/output tensors restored the expected shape, finite values and the same **0.972583** Dice on three identical repeated predictions. The probe record is `outputs/experiments/coa/context-r2-reload-probe/probe.json`. The original run remains marked incomplete rather than rewriting its provenance. Its pending dependent grid job `6861511` was canceled before execution.

The interface fix retains the strict probability and finite-value checks. A fresh finer-grid diagnostic verifies the complete fitting, checkpoint reload and evaluation path in Wahab's actual TensorFlow environment.

The numerical comparison used byte-identical attenuation, target and validity arrays and the same patch origin, confirmed in `outputs/qa/context-r1-r2-input-parity.json`.

The final cohort `--validate-only` result is `outputs/qa/context-segmentation-final-plan.json`: ready=true, all192 targets accepted, 129.6×96×96 mm physical blocks, 75,505,300-byte worst-case native float32 slab (excluding caches and neural-network memory). The related regression suite reached 163 tests, with the three opt-in TensorFlow checks exercised separately.

The final backend passed **23 tests inside Wahab's TensorFlow 2.17 container**, including a complete real tiny fit → save → reload → evaluate cycle. After the compatibility fix, 77 context-specific regression tests passed locally (four opt-in TensorFlow tests were exercised separately in the container).

The fresh final-grid diagnostic is Slurm `6861698`, configured by `configs/models/coa-context-grid-diagnostic.ini`, output `outputs/experiments/coa/context-grid-diagnostic-r2/`. Its measurements are recorded below after completion. The general 1 mm diagnostic example points to a fresh `context-diagnostic-r3` directory for any future rerun; no result is claimed for that unexecuted revision.

## Completed final-grid diagnostic

`context-grid-diagnostic-r2` completed **fit, checkpoint reload and evaluation** successfully. Slurm job `6861698` finished in **6m03s**, with maximum reported host RSS **1,739,272 KiB**. The model has 459,377 parameters; 40 epochs × 8 updates = 320 optimizer steps. The selected checkpoint was epoch 37.

On the fixed **training** patch (260613 baseline), Dice was **0.977870**, precision **0.987554**, and recall **0.968373**. It contained 45,721 target voxels; prediction had 44,275 true positives, 558 false positives and 1,446 false negatives. Peak TensorFlow GPU allocation was **3,361,564,672 bytes (3.13 GiB)** on a V100. Recorded fitting time was 168.5 seconds; the batch wall time also includes source verification and container startup.

The frozen experiment lock was rechecked after completion. Only `260613:baseline` was loaded by this fitting/evaluation diagnostic; test image payloads were not used for learning. These scores demonstrate successful fitting and checkpoint execution, **not unseen-family performance**.

The completed interactive report is `outputs/experiments/coa/context-grid-diagnostic-r2/report/index.html`. It includes learning curves, three orthogonal attenuation/error overlays and independently toggleable 3D reference, prediction and error surfaces. The slice overlays were visually inspected after generation.

## Development assessment and next experiment

The data preparation, physically consistent loading, model-grid checks, TensorFlow learning and saved-model evaluation now work together. Full-cohort training has **not** been launched. The next experiment should confirm anatomical target coverage, measure full-volume validation throughput, then fit/evaluate with the existing family-disjoint 8/2/2 case assignment. Validation covers 29,696 tiles per pass, so diagnostic runtime should not be extrapolated directly to a whole-cohort job.

To test the digital-sibling hypothesis, compare baseline-only training against baseline-plus-sibling training while keeping target definition, architecture, split and evaluation protocol fixed. Report unseen-family full-volume Dice, precision/recall, and branch/lesion coverage. Independent annotated clinical CT data will still be required to assess clinical generalization. Model 2 shape recovery remains outside this completed development step.
