# Larger-context attenuation-to-aorta segmentation

The new workflow separates the **editing region** from the **segmentation target**. Edits still come from the frozen CoA recipes; the target now covers an atlas-defined thoracic aorta and its proximal connected arterial branches. The network receives attenuation only. Anatomy labels, edit masks and ground-truth crops are not input features.

The implementation is on `fakect.26.09.16`. Existing datasets and previous two-stage models are retained. Storage preflight found no need for deletion; see [the storage audit](STORAGE_PREFLIGHT.md).

The completed 0.6 mm training-patch diagnostic reached **Dice 0.9779**, with a **3.13 GiB** TensorFlow GPU peak and successful saved-model reload/evaluation. This is a known-patch fit check, not unseen-case accuracy. See [the run record](integration/2026-09-29/context-aorta-run.md) and `outputs/experiments/coa/context-grid-diagnostic-r2/report/index.html`. Engineering target review and validation-throughput profiling now pass. The [matched baseline-versus-sibling cohort comparison](CONTEXT_AORTA_COMPARISON.md) was submitted on September 29 as Slurm array **6862667**; its held-out performance is pending.

## Data preparation

```bash
cd /home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16
python3 scripts/research/prepare_context_cohort.py --config archive/research/configs/cohorts/coa/context-aorta.ini --validate-only
python3 scripts/research/prepare_context_cohort.py --config archive/research/configs/cohorts/coa/context-aorta.ini --stage prepare --resume
```

The current 12-case dataset is already prepared. `--resume` verifies/reuses a matching completed revision; it does not overwrite it. The preparation input references the existing 192-sample, 12-phantom frozen cohort. It selects anatomy independently of the editing ROI and transfers the already-generated additions/removals onto that broader reference. A transfer that would change anatomy outside the selected reference is reported; failed samples cannot silently disappear from the release. `--resume` only reuses completed cases with matching inputs and implementation fingerprints. Changed inputs require a new output revision.

`outputs/datasets/coa/context-aorta-r1/dataset-manifest.json` contains shared source ATN/ACT references, compact mask files, native edited-crop references, units, geometry, family identities and checksums. The release remains explicitly provisional for anatomical review. Its current definition uses T12 midpoint through T1 upper extent, with connected proximal arterial branches within a radius-scaled reach. This is an operational boundary policy to inspect, not an established clinical thoracic-aorta definition.

A loader reads only the needed source slab and overlays the frozen sibling's native edited attenuation before interpolation. It returns attenuation in cm^-1, a binary target and a validity mask. Image interpolation is linear; target sampling uses nearest neighbor. Global grid alignment uses the lower voxel face, preventing patch-dependent coordinate shifts. Validity excludes padded voxel centers outside the scan. No full-body sibling copies are written.

## Anatomical review

The existing report is `outputs/reports/context-aorta-r1/anatomy/index.html`. To regenerate into a **new** report revision:

```bash
python3 scripts/research/report_context_cohort.py --manifest outputs/datasets/coa/context-aorta-r1/dataset-manifest.json --output outputs/reports/context-aorta-r2/anatomy
```

Open `outputs/reports/context-aorta-r1/anatomy/index.html`. A single case selector controls the native 3D aortic-core/branch surfaces and the corresponding axial, coronal and sagittal attenuation overlays. Surface labels have independent visibility and opacity controls. Slice axes show native coordinates. This report deliberately displays reference anatomy rather than predictions.

## Learning diagnostic first

```bash
python3 scripts/research/train_context_aorta.py --config archive/research/configs/models/coa-context-diagnostic.ini --validate-only
mkdir -p logs
sbatch scripts/research/slurm/fakect_context_gpu.sh archive/research/configs/models/coa-context-diagnostic.ini
```

The diagnostic repeatedly fits **one fixed training patch** from case 260613. It uses no augmentation and no validation/test image blocks. Its purpose is to establish that target/image alignment, loss, gradients, checkpoint reload and learning work together. Its Dice must not be interpreted as unseen-patient performance. Forty epochs with eight updates per epoch give 320 optimizer updates.

The model is a 3D U-Net with three downsampling levels, eight initial filters, paired 3D convolutions, skip connections and a sigmoid output. Loss is valid-voxel BCE computed stably from logits plus soft Dice; Adam uses a fixed configured learning rate. A 128×96×96 block at 1 mm spacing provides 128×96×96 mm of physical context, compared with the previous 38.4 mm cube. This is a conventional larger-context network; a separate low-resolution locator/high-resolution cascade is not yet implemented.

The experiment freezes the INI, implementation, dataset identity and family assignment. Changed inputs/code require a fresh output directory. The command stages are `plan`, `fit`, `test`, and `all`; `--validate-only` checks without freezing. `all` is intended for a fresh experiment. If fitting already completed but evaluation has not, use `--stage test` rather than repeating `all`.

After completion:

```bash
python3 scripts/research/report_context_diagnostic.py --experiment outputs/experiments/coa/context-diagnostic-r3
```

The report combines learning curves, orthogonal attenuation/error overlays and a 3D viewer with independent reference, prediction, false-positive and false-negative visibility/opacity. The model/evaluation artifacts record the exact patch, clipping, spacing, versions, checkpoint checksum and loaded sample IDs.

## Finer-grid GPU diagnostic

The current 0.6 mm run uses `outputs/experiments/coa/context-grid-diagnostic-r2/`. Its report is generated once in that directory. To rerun, first copy the INI and choose an unused `output_directory`; do not resubmit a completed immutable revision.

`archive/research/configs/models/coa-context-grid-diagnostic.ini` fits the same training baseline on the cohort's 0.6 mm grid and larger voxel block. This separately checks GPU memory and learning at the final physical sampling before a cohort run:

```bash
# Use your copied INI after selecting a new output_directory:
sbatch scripts/research/slurm/fakect_context_gpu.sh path/to/new-grid-diagnostic.ini
python3 scripts/research/report_context_diagnostic.py --experiment path/to/new-experiment-directory
```

TensorFlow allocator current/peak bytes are recorded in `model/training.json`; these cover TensorFlow-managed GPU allocations, not all driver/container memory. As with the first diagnostic, these scores measure fitting a known patch only.

## Cohort experiment

`archive/research/configs/models/coa-context-segmentation.ini` is the separate cohort input. It uses a 216×160×160 block at 0.6 mm, giving 129.6×96×96 mm context. This finer grid passed the connectivity audit for all 192 targets; 1 mm broke three severe infant narrowings. The separate numerical diagnostic uses 1 mm on a known nonsevere baseline; the grid diagnostic uses the cohort’s 0.6 mm sampling. Validate the cohort INI to see family splits, geometry, memory bounds, target-grid checks and complete-volume evaluation tile counts:

```bash
python3 scripts/research/train_context_aorta.py --config archive/research/configs/models/coa-context-segmentation.ini --validate-only
```

Related phantom IDs and all their siblings stay in one split. The requested 8/2/2 case counts are resolved over the provisional anatomy families; these are not 12 independent patients. Identical original reference-mask identities cannot cross partitions. Family identity itself still needs independent confirmation.

Training mixes foreground-centered and spatially uniform whole-scan patches. Paired i-axis flips and bounded attenuation scaling are configurable; the superior/inferior axis is not flipped. Validation and test both scan the **entire available image** using the same nonoverlapping tiling protocol, including background outside the aorta neighborhood. No true target box is supplied to inference. Padding is excluded from scoring. Checkpoints are selected by validation family-macro Dice. Reports distinguish sample-, case- and family-level averages.

Cohort planning audits every target on the requested global model grid. It blocks empty targets, a greater-than-one-percentage-point increase in volume outside the largest component, a newly split fragment over 5 mm³, or loss of a native component over 5 mm³. These are explicit engineering checks, not clinical thresholds. Smaller sampling artifacts remain in the audit for review; masks are never repaired silently.

Full-volume validation is substantially more expensive than the fixed-patch diagnostic. The plan prints tile counts before launch. The supplied Slurm wrapper requests two hours for the diagnostic; use an appropriate `sbatch --time=...` override for a full cohort run after measuring throughput. Full prediction volumes are not stored; metrics accumulate block by block. A separate overlap/blending inference option, truncation augmentation and full-volume interactive prediction export are future work.

Before an outcome study, review anatomical coverage across all cases and complete the learning diagnostic. Then run the cohort experiment and a controlled comparison that keeps targets, split and evaluation protocol fixed while changing physical context. The previous crop-based model scores are not a valid direct context ablation because the target definition and evaluated field of view differ.

## Limits of the current experiment

- Source images are XCAT attenuation with copied attenuation in edited regions, not reconstructed clinical CT or HU images. Synthetic held-out performance cannot demonstrate clinical generalization.
- Broader targets remain provisional until boundary and branch inclusion are reviewed; metadata never marks this review complete automatically.
- One-patch memorization can establish basic learning but cannot prove whole-volume localization or unseen-family robustness.
- Anatomy-selection errors, small-vessel sampling artifacts on the chosen model grid and tile-edge effects should be inspected before architecture or cohort size is increased further.
- Model 2 shape recovery is unchanged by this development. The current work isolates the attenuation-to-aorta segmentation problem first.
