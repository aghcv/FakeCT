# CoA sibling-population study: family cross-validation and severity response

**Development order updated 2026-09-28:** first implement the
[two-model voxel-mask pilot](pathology-voxel-pilot.md): CT-to-aorta segmentation,
then mask-to-reference-shape reconstruction with derived added/removed regions.
The geometry/severity annotations and
condition-specific benchmark below are a subsequent phase, not prerequisites
for that pilot. The existing binary model backend supports the segmentation
task; reconstruction needs separate data, training and evaluation contracts.

**Status: study design, not an executed cohort experiment.** This refinement records the user's proposed repeated held-out-anatomy analysis. The [INI evaluation interface](../../MODEL_EVALUATION.md) now supports fixed holdout, leave-one-family-out and nested leave-one-family-out execution for the existing binary segmentation backend, including independent testing and family-balanced averages. Nested fitting currently selects an epoch budget with one configured architecture. Condition classification, per-condition localization and severity-response analysis still require the annotated dataset/model extension below. The current 192-scenario release is frozen for extraction, but its paired arrays have not been exported. Preserve that release while developing a new condition-annotated revision. See the [current cohort review](radius115-cohort-review.md) and [project outline](../../THORACIC_AORTA_PROJECT.md).

The long-term hypothesis is: **incorporating anatomically diverse digital sibling populations improves model robustness and generalization on previously unseen clinical CT.** The first experiment measures performance on held-out synthetic anatomy families. Independent clinical testing will address the clinical part of the hypothesis later; synthetic holdouts alone cannot establish it. Current images are attenuation-copy proxies, and clinical image realism remains a separate question. This distinction follows [CLAIM's guidance on internal and external testing](https://pubs.rsna.org/doi/10.1148/ryai.240300).

**Unit of separation.** Keep all case IDs, motion frames, synthetic siblings, patches, image realizations and dataset revisions derived from the same underlying anatomy in one family. The project currently declares six provisional groups. These are protective grouping assumptions, not verified evidence of six independent anatomies. Review template ancestry, exact and near-duplicate geometry, and source provenance before locking folds; merge related groups if necessary. Age/sex or different filenames alone do not establish independence.

| Outer fold | Provisional held-out family | Test case IDs | Development cases |
| --- | --- | --- | --- |
| 1 | `xcat_reference_10yr` | 260615, 260620 | All other 10 |
| 2 | `xcat_reference_15yr` | 260616, 260621 | All other 10 |
| 3 | `xcat_reference_1yr` | 260613, 260618 | All other 10 |
| 4 | `xcat_reference_5yr` | 260614, 260619 | All other 10 |
| 5 | `xcat_reference_adult` | 260602, 260611 | All other 10 |
| 6 | `xcat_reference_infant` | 260612, 260617 | All other 10 |

The [draft fold manifest](coa-family-folds.draft.json) enumerates all six outer folds and thirty inner splits from [the project inventory](../../../configs/cohorts/coa/project-radius115.json). It is a planning artifact, not a runnable model INI or an experiment lock. It does not mark any family verified. Regenerate it if reviewed family membership or the project inventory changes. Each family has one outer test turn: this is leave-one-group-out cross-validation, with one fold per group. [Method definition](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.LeaveOneGroupOut.html).

**Work within an outer fold.**

1. Reserve the complete held-out family and its predefined severity trajectories for outer testing. Its images and predictions do not guide model selection, preprocessing, calibration or decision thresholds.
2. Rotate each of the five remaining families through inner validation. Each inner fit uses four families: provisionally eight case IDs for fitting, two for validation, and two reserved for outer testing.
3. Select architecture, hyperparameters and an epoch schedule using inner validation only. Fit learned normalization on the applicable training partition. Use inner out-of-fold predictions if fitting a probability calibrator or choosing a probability decision threshold; document how these are applied to the final refit.
4. Initialize a new model and refit on all five development families with the selected settings and fixed epoch schedule. Do not use the outer test family for early stopping.
5. Evaluate the frozen model on every predeclared, QC-passing sibling of the held-out family. Save predictions before inspecting aggregate outcomes. Retain generation failures, exclusions and unattainable severity levels in an audit rather than silently dropping them.
6. Repeat for all families and a prespecified set of training seeds. Every scored sample must be predicted only by models whose fitting and tuning excluded its family. An ensemble used for an outer test family must likewise contain only eligible models.

Nested selection separates tuning from outer performance estimation. Once outer results have influenced changes to the architecture, generator or protocol, a rerun is exploratory; it cannot be presented as a fresh untouched test. [Nested cross-validation guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

For six families, a search over H candidates with R repeats requires `R * (30 * H + 6)` fits per experimental arm: thirty inner fits per candidate and six final outer refits per repeat. For one candidate and three training seeds this is 108 fits per arm. This is a planning count, not a submitted job budget; runtime and GPU memory require a small engineering pilot. Final refits use ten development case IDs, while inner selection reproduces the eight/two/two design.

**Independent lesions and severity labels.** The frozen release has one baseline and fifteen variants per case; every edited variant combines dilation, fixed arch erosion and CoA erosion. A new revision must include independent absence/presence and severity variation for each condition, with isolated and combined lesions. Confirm the anatomical extent of the enlargement and arch targets, since current editing intervals overlap. Maintain three condition labels rather than one mutually exclusive pathology label.

Define continuous severity from achieved geometry after resistance, recipient limits and safeguards. Keep added/removed mm3, local normalized change, vessel-normal cross-sectional area measurements when implemented, and all QA flags. A simple perturbation score is `achieved changed volume / baseline target volume in the predefined lesion segment`; record the sign/operation and keep that reference segment fixed across variants. Record both per-step effects and the final achieved geometry when operations overlap. A fixed hypothetical clinical severity should not be inferred from requested millimetres or raw voxel counts. Label the bins as synthetic perturbation classes until clinically calibrated; an unedited control is not automatically proof of normal clinical anatomy.

Freeze a separate intermediate/severe cutoff `tau` and boundary width `delta` for each condition before outer evaluation. These geometry-label definitions are distinct from the model's probability decision threshold. If data are needed to estimate a cutoff, use a separate development reference or the outer-development families under a declared rule; never estimate it from the held-out family's outcomes. Prefer common prespecified geometry definitions so class meaning remains comparable across folds.

**Test severity response on an unseen family.** Keep a broad, predeclared severity grid and an additional boundary-focused grid around `tau`. Vary one lesion while holding the other two fixed for interpretable response curves; also test predefined combinations to measure interactions. Use achieved severity on the horizontal axis, not the edit request. Record whether the requested boundary levels are attainable at native resolution; repeated masks or missing levels are reported as such. Geometric generation may target prespecified achieved levels without consulting model predictions.

| Question | Saved output and assessment |
| --- | --- |
| Does the model recognize the condition as it increases? | Condition probability and predicted continuous severity versus achieved severity for each held-out case, including absent controls and response reversals |
| Can it distinguish intermediate from severe near the cutoff? | Predictions versus signed distance `severity - tau`, confusion matrices, severe-class sensitivity and specificity within the predeclared boundary band; report denominators and unavailable classes |
| Does it find the correct region? | Per-condition affected-vessel localization and overlap/distance errors; keep edit-provenance masks separate from the final vessel-region masks |
| Does performance depend on base anatomy? | Case and family summaries, equal-family aggregate, worst-family results, and age/size/spacing annotations |
| Does randomness explain apparent differences? | Repeat the same folds and samples with paired training seeds; report seed variability separately from family variability |
| Do coexisting lesions alter predictions? | Response surfaces or stratified curves across the other lesions' prespecified levels |

Report broad-grid and boundary-grid results separately: denser sampling near a threshold must not silently reweight the main score. Use only out-of-family predictions. Save continuous scores even when reporting binary/ordinal classes. Check calibration with the sampling distribution explicitly stated; probabilities calibrated on this artificial grid are not established clinical risk probabilities. Do not assume a smooth or monotonic response—measure violations, jumps and transition locations. Boundary misclassification reflects a synthetic class definition and finite image resolution as well as model behavior.

**What variability this identifies.** Sweeping severity within one fixed phantom isolates sensitivity to a controlled synthetic geometric intervention. Comparing held-out families characterizes variation across the available templates, but age, shape, size, spacing and imaging properties may change together. Outer-fold differences also reflect overlapping but different training sets. Report these as synthetic anatomy/domain sensitivity; they do not independently quantify all biological variability or prove a clinical mechanism. Add source families and imaging realizations to address those factors. If attribution to training-set composition is needed, predeclare a secondary experiment with a fixed outer test family and controlled omissions within its development families, keeping settings fixed and controlling training size; this addresses a different question from the primary outer-fold estimate.

Compute case summaries within family and then aggregate with equal family weights; predeclare within-case scenario weights. Repeated siblings and training seeds do not increase the number of independent subjects. Outer models share training data, so their scores are not independent repetitions. With only six provisional families, show every family result and interpret any family-level bootstrap interval as exploratory; do not use an ordinary independent-fold t-test or a voxel/slice bootstrap to claim population precision.

**Comparison needed for the hypothesis.** Use identical outer families, test trajectories, architecture-selection rules, optimizer-update budgets and paired seeds for each arm. Compare a limited-sibling cohort against an expanded anatomically diverse sibling cohort. Add a size-matched comparison so diversity can be distinguished from merely more examples. Include absence, intermediate and severe examples in every feasible training arm. The current source templates remain identical across these arms unless the protocol explicitly tests number of independent source families; distinguish added geometric sibling diversity from added independent anatomy. Report paired per-family improvements and failures, not only pooled accuracy. An unmodified-only cohort can be a segmentation baseline, but has no positive induced-lesion labels for the proposed supervised pathology classifier.

**Implementation sequence.**

1. Review family provenance and register verified identities; retain a draft manifest until this is complete.
2. Create a new condition-annotated cohort revision with independent lesions, achieved severity, per-edit provenance and final affected-vessel targets. Freeze geometry definitions, QC/exclusion rules, severity cutoffs, boundary grids and metrics.
3. Extract and checksum-pin image/target pairs; update model dataset receipts to that revision. Preserve the existing frozen release and older registered preparation.
4. Use the implemented explicit outer/inner family assignments, isolated fold outputs, refitting and saved held-out metrics; extend their current binary segmentation contract to multi-condition objectives, probabilities, lesion masks and severity trajectories. An architecture/hyperparameter search remains separate from the implemented nested epoch selection.
5. Verify data/model plumbing on engineering fixtures, then run the frozen synthetic benchmark with the agreed compute budget and comparators. No training jobs are launched by this document or its draft manifest.
6. Freeze a subsequent model-development procedure and evaluate on independent clinical CT with appropriate reference annotations. Keep this separate from exploratory synthetic-fold iteration.
