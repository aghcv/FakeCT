# Aorta masks and reference-shape learning for labeled phantoms

For labeled XCAT phantoms, anatomical selection can supply the observed aorta mask directly. A learned CT segmentation model is unnecessary for this experiment. The reference-shape model then receives only that binary mask and estimates an unedited reference. Comparing the two produces candidate expansion and narrowing regions.

This development adds an automatic anatomical selector, a broader paired-mask preview, an independent mask-model runner, and interactive 3D reviews. **No new model was trained during this development.** The existing trained model and its dataset remain unchanged.

The current [results report](../outputs/experiments/coa/two-stage-r2/report.html) includes an [interactive 3D viewer](../outputs/experiments/coa/two-stage-r2/report-3d/index.html). Rotate and zoom the view; show or hide individual surfaces and set their opacity. Compare observed anatomy, original reference, predicted reference, and true/predicted additions and removals. Start with the exact-mask reconstruction views to isolate Model 2 from segmentation errors. Unchanged controls are useful for spotting false changes.

All commands below run from the integration checkout:

```bash
cd /home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16
```

1. **Extract a provisional thoracic anatomical mask without an edit ROI.**

   Edit [thoracic-aorta-260602.ini](../configs/anatomy/thoracic-aorta-260602.ini), validate its inputs, then run:

   ```bash
   python3 scripts/select_phantom_anatomy.py --config configs/anatomy/thoracic-aorta-260602.ini --validate-only
   python3 scripts/select_phantom_anatomy.py --config configs/anatomy/thoracic-aorta-260602.ini
   ```

   This uses native XCAT anatomical ACT labels, the atlas hierarchy, and registered RAW surface metadata. It discovers source IDs through those inputs; no user-supplied source ID or target ROI coordinates are required. An optional earlier surface inventory is a cache of all registered surfaces: its old ROI and old selected IDs do not define the new mask.

   The aortic label extends into the abdomen, so the atlas alone does not establish the thoracic endpoint. The current explicit policy places the inferior cut at the midpoint of the T12 vertebral surface z bounds and the superior cut at the upper T1 surface bound. **These are provisional landmarks, not a reviewed diaphragm/hiatus boundary.** The mask keeps the largest aorta component inside that slab, then adds connected eligible arterial branches. Pulmonary labels are excluded before connectivity testing.

   Branch extent is limited by physical distance from the aorta core. By default, the reach is three times the median verified aortic ring radius; `branch_reach_mm` can override it. These are **proximal branches**, not guaranteed complete named vessels. Generic branch labels retain their generic names. Native face contact is useful geometric evidence, but does not prove physiological connectivity. Inspect the distal cutoff, arch branches, coronary origins, and disconnected pieces before accepting the extent.

   The output contains native masks, geometry, selected-label provenance and checksums under `outputs/anatomy/thoracic-auto/260602-r1/`. Existing output directories are preserved; choose a new revision when changing the selection.

2. **Transfer the existing synthetic edits onto the broader mask for review.**

   Use [260602-mask-pairs.ini](../configs/anatomy/260602-mask-pairs.ini). Set its output directory to a fresh revision; the current review command below uses `260602-r2`.

   ```bash
   python3 scripts/preview_mask_pairs.py --config configs/anatomy/260602-mask-pairs.ini
   ```

   This reuses frozen edit deltas and native lineage provenance; it does not rerun morphology. Removed voxels and dilation ancestors must belong to the new reference mask. Mismatches are marked for review. Unchanged broader anatomy is retained, and the crop can grow to include borrowed voxels. Before/after/add/remove masks are recomputed together: an old lineage addition that was already part of the broader reference is no longer counted as a new addition.

   The output uses `fakect.mask-pair-preview/1`, with `training_ready=false`. Passing the transfer checks means ready for visual review, not approved anatomical coverage or training release.

3. **Review the selection and broader pairs together.**

   ```bash
   python3 scripts/review_phantom_masks.py \
     --selection outputs/anatomy/thoracic-auto/260602-r1/selection.json \
     --pairs outputs/datasets/coa/thoracic-mask-preview/260602-r2/preview-manifest.json \
     --output outputs/datasets/coa/thoracic-mask-preview/260602-r2/review
   ```

   Open the [draft anatomical and paired-mask review](../outputs/datasets/coa/thoracic-mask-preview/260602-r2/review/index.html). Use transparent overlays to check selection extent and transferred changes. Adjust the anatomical INI and regenerate into new output revisions when needed. Repeat this review across all phantoms before freezing a broader cohort. Copy the entire review directory when sharing it offline.

4. **Use the independent mask-model interface on the existing released pairs.**

   [coa-mask-model.ini](../configs/models/coa-mask-model.ini) points to the current immutable 192-pair manifest. Its explicit `target_scope = selected_lineage` means the reviewed ROI-selected aorta lineage, including the branch junctions already present in those pairs. It does not claim full thoracic coverage. The input contains the observed binary mask; the reference and change masks are supervision. Shared files still contain attenuation arrays, but they are not network input features for this model.

   ```bash
   python3 scripts/train_mask_model.py --config configs/models/coa-mask-model.ini --validate-only
   ```

   This validation passed for all 192 pairs, preserving the earlier family-disjoint split: 128 training, 32 validation, and 32 test samples, corresponding to 8/2/2 phantoms. Related anatomy and all its synthetic siblings remain together. These are provisional engineering family groups.

   To train a future revision, allocate a Wahab GPU first. For an interactive allocation:

   ```bash
   srun -p gpu --gres=gpu:1 --cpus-per-task=4 --mem=16G --time=04:00:00 --pty bash
   module load container_env tensorflow-gpu/2.17
   export TF_NUM_INTRAOP_THREADS=4
   export TF_NUM_INTEROP_THREADS=1
   export OMP_NUM_THREADS=4
   crun -p ~/envs/fakect python scripts/train_mask_model.py \
     --config configs/models/coa-mask-model.ini --stage all
   ```

   The same command can run inside a GPU Slurm script. `all` freezes configuration, dataset identity, partitions and implementation snapshots; fits only the reference-shape model; tests it; and writes the report. Individual stages are `plan`, `fit`, `test`, and `report`. Completed matching outputs can be reused; changed inputs or partial results require inspection and a new experiment revision. Do not use the older two-stage Slurm wrapper for this command, since that wrapper explicitly launches both models.

   Model 1 is never fitted or loaded by this interface. Existing backend keys named `oracle_*` refer to exact supplied observed masks. They do not imply a learned segmentation step. Generated arrays, checkpoints and reviews remain outside Git.

5. **Release broader training data before retraining for broader anatomy.**

   The additional anatomical labels change the target definition. First review the automatic extent and transferred edits consistently across all phantoms. Then implement a training-release adapter for the broader before/after/change masks, including geometry, source-label provenance, checksums, unchanged controls and family assignments. Freeze that as a new dataset revision, select an appropriate bounded model grid, and retrain in a new experiment directory. The draft preview manifest intentionally cannot masquerade as the current training schema.

The earlier exact-mask test already isolates the reference-shape problem: reference Dice was approximately **0.918**, compared with **0.980** for simply leaving the input unchanged; added-region Dice was **0.222** and removed-region Dice **0.349**. Removing Model 1 avoids its additional segmentation errors, but does not improve those existing exact-mask results by itself. Review false changes on controls and missed edited regions alongside the overall reference Dice.

The model currently predicts a plausible reference and net added/removed regions. It does not yet produce validated condition-specific diagnoses for coarctation, arch hypoplasia or ascending dilation. Those interpretations require further validation and anatomical attribution. The deterministic selector also requires labeled XCAT inputs; attenuation-only patient CT would still need a separately validated segmentation process.
