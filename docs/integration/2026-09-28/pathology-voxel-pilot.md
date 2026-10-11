# First pathology AI pilot: segment the aorta, then estimate its reference shape

**Decision date:** 2026-09-28. **Branch:** `fakect.26.09.16`.
**Status:** the two-model engineering implementation now exists. See the
[training guide](../../TWO_STAGE_TRAINING.md) and
[execution record](two-stage-training-run.md) for the paired-data extraction,
TensorFlow validation and real-cohort job status. The detailed geometry/severity
phase below remains deferred.

Start with two independently trainable models and voxel localization of their
shape differences. Defer centerline fields,
normal-plane area measurements, severity regression and intermediate/severe
classification until the image-to-mask experiment works. This ordering refines
the [project outline](../../THORACIC_AORTA_PROJECT.md) and
[family evaluation protocol](coa-family-cross-validation-plan.md).
Manual condition annotation and severity thresholds are not prerequisites.

## 1. Two-model workflow

| Model | Input at inference | Training target | Result |
| --- | --- | --- | --- |
| 1: aorta segmentation | CT/attenuation image | Final aorta mask | Observed aorta mask |
| 2: reference-shape reconstruction | Observed aorta mask only | Original unedited mask, with derived added/removed supervision | Estimated reference aorta and enlargement/narrowing difference maps |

Designate the twelve unedited phantoms as the study's reference-normal anatomy.
This is the experiment's baseline assumption, not a clinical normality finding.
Each synthetic sibling retains its own source phantom as the reconstruction
target. A held-out family and its baseline are never used to train either model.

Model 1 learns the aorta as it appears, including the imposed changes. Model 2
learns to estimate an unedited counterpart from shape alone. At inference it
does not receive the original mask, edit settings, patient identity or CT.
Its reconstruction is a plausible reference learned from training anatomy;
the exact historical anatomy is not uniquely determined by an abnormal shape.

For notation, let `B` be the original binary mask, `A` the true edited mask,
and `S` the binary segmentation predicted by Model 1. Model 1 trains on
`image -> A`. Start Model 2 with `A -> B`, then evaluate `S -> B_estimate`.

## 2. Paired data and consistent outputs

The original target is already exported in `source.npz` as `target_mask`;
each prepared variant contains its final `mask`. These aligned masks are enough
to derive the two new targets without replaying edits or modifying the frozen
generator. Publish a versioned annotation companion that pins source/sample
hashes and geometry, leaving existing image/mask payloads unchanged. The latest
radius115 paired arrays and change companions have now been extracted and
verified for all 192 scenarios; see the [run record](two-stage-training-run.md).

```python
original = source["target_mask"].astype(bool)
final = sample["mask"].astype(bool)
added = final & ~original
removed = original & ~final
```

Use net target-membership changes. `changed_mask` records fine-label changes;
`scalar_changed_mask` records attenuation changes. Neither is an interchangeable
target. Changes canceled by later recipe steps disappear from these net masks.
The union `added | removed` provides a derived "any synthetic change" result.

Added voxels overlap `A`; removed voxels are outside its surviving lumen.
Keep both in the same crop, with enough margin to reconstruct the original
mask. Do not clip reconstruction or removed masks to the input aorta.

The smallest consistent Model 2 has **one learned baseline probability head**.
Given binary input `I` (`A` for isolated testing, `S` for the full pipeline)
and its predicted baseline probability `pB`, derive:

```text
enlargement_score = I * (1 - pB)
narrowing_score   = (1 - I) * pB

B_estimate = pB >= a validation-selected threshold
added_estimate   = I AND NOT B_estimate
removed_estimate = B_estimate AND NOT I
```

Thus `B_estimate = (I minus added_estimate) union removed_estimate` by
construction. Model 2 delivers all three requested results without independent
heads that could contradict one another. The added/removed targets are
mathematically determined by `A` and `B`; exporting them helps inspection,
change-focused training and evaluation, rather than supplying new information.

Use a baseline reconstruction loss such as BCE plus Dice. With exact input
`A`, optionally add losses on the derived change scores or weighting around
actual edits so an unchanged majority does not encourage an identity solution.
Include `B -> B` controls and measure overcorrection of unchanged anatomy.
Compare with the identity prediction `B_estimate = I` as a minimal baseline.

The corrupted-input/original-target training pattern is illustrated in the
[TensorFlow denoising autoencoder tutorial](https://www.tensorflow.org/tutorials/generative/autoencoder#second_example_image_denoising).
Applying it to synthetic aorta edits is the proposed experiment, not a result
established by that example.

## 3. Keep segmentation error separate from imposed morphology

First evaluate Model 2 using exact held-out edited masks `A`. Its added/removed
outputs can then be scored against the known synthetic edit masks.

Next evaluate the full pipeline using Model 1's held-out prediction `S`.
Differences between `S` and a reconstructed reference include both imposed
morphology and segmentation errors. Save and report them as reference-shape
differences, not automatically as pathology. Evaluate baseline reconstruction
against `B`; evaluate restoration residuals against `S AND NOT B` and
`B AND NOT S`. Separately report agreement with the original synthetic edit
masks and overlap with the segmentation error mask `S XOR A`.

For robustness training, use controlled mask perturbations or Model 1
predictions generated without fitting on the same anatomy family, using only
development families. Train these noisy inputs toward `B`. If supervising
derived residual heads, their targets must use the same noisy input `S` and
`B`; enforcing `A`-based change labels through `S`-gated outputs can be
inconsistent. Separating segmentation corrections from morphology as distinct
inference outputs would need an additional task beyond this minimal model.

## 4. Implementation and first evaluation

1. Publish original/final mask references and optional derived-mask companions
   with geometry and checksums. Validate mask identities, disjoint changes,
   no-op controls and serial cancellations. Keep frozen generation unchanged.
2. Train Model 1 with the existing binary segmentation task. Verify that its
   target covers the agreed thoracic-aorta extent, not merely a small edit
   interval; current exported masks follow the reviewed ROI-selected lineage.
   A crop-based pilot requires a declared crop at inference; whole-scan use
   additionally requires localization or tiling without ground-truth labels.
3. Add a separate Model 2 experiment/loader contract for mask-to-mask training,
   baseline reconstruction, derived differences and aligned spatial transforms.
   Preserve spacing, physical scale and a common grid; avoid independently
   aligning or rescaling before/after anatomy in a way that erases the edits.
   Use enough surrounding vessel context to infer reference shape. A small
   2D model can check plumbing; a compact volumetric or multi-slice model is
   a candidate for shape reconstruction after memory/fit checks.
4. Run fixture tests and a tiny overfit exercise, including save/reload and
   exact reconstruction/difference identities. Then evaluate Model 1, Model 2
   with exact masks, and the full pipeline as three separate results.
5. Report reconstruction and change-mask Dice, IoU, precision and recall,
   alongside false-positive change volume on controls. Summarize positive and
   empty-change cases separately. Show unchanged anatomy preservation, since
   the model must not turn unfamiliar healthy variation into an abnormality.
6. Use the same whole-anatomy-family partitions for both models. Keep related
   phantoms, all siblings, source baselines and intermediate predictions
   together. Begin with a holdout pilot; family cross-validation can follow.
   Twelve phantom IDs are not proof of twelve independent anatomy families.

The current combined-lesion cohort can support this limited pilot. Separate
arch/CoA recognition requires independently varied single and combined lesions;
all currently edited scenarios include all three operations. Current
attenuation-copy images support an engineering study; clinical generalization
remains a subsequent question.

The runnable [shared INI](../../../configs/models/coa-two-stage.ini) contains
separate segmentation/reconstruction settings and uses separate checkpoints
with the same pinned dataset and family split. The previous binary model
interface remains available. This two-model design replaces the earlier
proposal for one CT model with three independent output channels.

## 5. Deferred condition, location and severity annotations

Retain the detailed proposal as the next phase:

- Capture actual per-step added/removed masks through `apply_recipe`'s existing
  `on_step` callback, keyed by `arch_outer_expand`, `arch_inner_narrow`,
  `coa_narrow` and iteration. Keep accepted changes separate from proposals,
  blocked voxels and no-ops. Preserve overlapping event history separately
  from final-state targets; define final condition attribution when later
  edits undo earlier changes. Old aggregate masks cannot recover this history;
  use saved step artifacts or replay with verified final hashes if needed.
- Export nearest-centerline distance `s_mm` and normalized position `s/L`,
  with validity masks. Keep existing INI percentages tied to original-parent
  path length and name true smoothed-spline arc length separately. Comparable
  positions across phantoms require consistent anatomical endpoints.
- Measure target area on planes perpendicular to the smoothed reference path,
  using the same planes before/after each step and in final geometry. Existing
  plots show voxel volume per path-bin length, not normal-plane areas. Prefer
  mask integration over circular radius estimates for asymmetric sections;
  flag branch, competing-vessel, invalid-tangent and clipped-plane cases.
- Record local final/original area ratios and per-step after/before ratios.
  Prefer local original area as the reference; an unaffected reference segment
  is optional. Keep undefined baseline sections invalid. Area divided by ROI
  length alone is not dimensionless and is not the default severity measure.
- Store compact profile tables and a voxel-to-path map, materializing dense
  area/severity channels during loading to limit storage. Preserve optional
  normal/binormal angular coordinates for asymmetric changes.
- Add final affected-vessel condition targets, achieved-severity regression
  and predeclared synthetic intermediate/severe thresholds, followed by the
  condition-specific response experiments in the family evaluation protocol.

These additions build on the mask pilot; they do not block it.
