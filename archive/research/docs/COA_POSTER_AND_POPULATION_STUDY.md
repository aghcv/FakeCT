# CoA poster figures and virtual-population study

> Historical generated artifacts are recorded as paths below; they are not included in this source checkout.

**Date:** 2026-10-03. **Branch:** `fakect.26.09.16`.
**Status:** poster evidence and experiment plan. The first development pilot is
complete. Rotating-family evaluation and population-size/diversity comparisons
have not been trained. The lower-learning-rate control was submitted as GPU job **6902259** in a
new experiment revision; its performance results are not yet available. See
[its configuration](../configs/models/coa-focused-pilot-lr-control.ini) and
report (`outputs/experiments/coa/focused-pilot-lr-control-r1/report.html`).

The poster should show a reproducible route from editable digital phantoms to
an image-based CoA benchmark, followed by measured learning and the planned
population experiment. Its research hypothesis is that anatomically diverse
digital sibling populations improve robustness on unseen anatomies. **The
current pilot demonstrates training-set learning, while the benefit of virtual
population expansion remains to be measured.** Clinical CT generalization is a
later endpoint requiring clinical evaluation data.

This plan extends the [platform and population plan](VIRTUAL_POPULATION_PLATFORM_PLAN.md)
and [current pilot workflow](MULTISCALE_COA_PILOT.md). Frozen datasets, annotation
revisions and completed experiment outputs remain unchanged.

## What is measured now

The source cohort contains 192 scenarios from 12 phantom IDs, grouped into six
provisional anatomy families. Two related phantom IDs belong to each family;
12 IDs therefore do not represent 12 independent originating anatomies. The
focused pilot selected four scenarios per development phantom: an original,
an edited negative, a lower positive and a higher positive.

| Partition | Anatomy families | Phantom IDs | Selected scenarios | Positive / negative |
| --- | --- | --- | ---: | ---: |
| Training | 1-, 5-, 10-, 15-year reference families | 260613–260616, 260618–260621 | 32 | 16 / 16 |
| Validation | Adult reference family | 260602, 260611 | 8 | 4 / 4 |
| Reserved final test | Infant reference family | 260612, 260617 | Not evaluated | Not evaluated |

The full image pipeline observes attenuation, uses an image-trained coarse
locator, and examines two fine 3D crops at 0.6 mm spacing. The focused classifier
uses normalized attenuation and acquisition-validity channels, with a shared
3D encoder and classification/regression heads. Segmentation masks, edit masks,
atlas IDs and disease metadata provide supervision and QA; they are not
classifier image channels. Describe this implemented network as a focused 3D
classifier, rather than calling the current experiment a U-Net segmentor.

The completed job `6902162` ran 50 epochs and 800 paired optimizer updates.
Minimum validation binary cross entropy selected epoch 37.

| Selected-checkpoint measurement | Training | Validation |
| --- | ---: | ---: |
| AUROC, all selected scenarios | 1.000 | 0.875 |
| Accuracy at probability threshold 0.5 | 96.9% | 62.5% |
| Sensitivity / specificity | 100% / 93.8% | 25% / 100% |
| TP / TN / FP / FN | 16 / 15 / 1 / 0 | 1 / 4 / 0 / 3 |
| AUROC, edited scenarios only | 1.000 | 0.750 |
| Local-volume-loss MAE | 20.80 percentage points | 18.26 percentage points |

These are development measurements from four training families and one
validation family. They are not an independent test estimate. The regression
predictions collapsed close to zero, so reliable severity prediction is not a
current result. Present the continuous target and this failure visibly rather
than labeling a near-zero prediction as successful severity estimation.

Evidence: results JSON (`outputs/experiments/coa/focused-pilot-r1/pilot/results.json`),
sample-level predictions (`outputs/experiments/coa/focused-pilot-r1/pilot/classifier-predictions.json`)
and review report (`outputs/experiments/coa/focused-pilot-r1/report.html`).

## Label and prediction conventions

For this benchmark, the original phantoms are assumed non-CoA references.
An edited scenario is positive when its **achieved local CoA target-volume loss
is at least 20%**. The denominator is the target in the longitudinal support of
the CoA edit, immediately before that step, after any preceding recipe edits.
This is an engineering proxy for induced change, not a clinically validated
CoA diagnosis, minimum-lumen-area reduction or severity grade. Below-threshold
edited cases remain altered anatomies.

Keep these two thresholds separate in every caption: **20%** defines the
reference class from achieved morphology; **0.5** is the current decision
threshold applied to a predicted probability. Preserve the continuous loss,
CoA-edit-applied flag, companion edits and 18–22% review flag. Requesting a large
erosion is not sufficient to assign a positive label when safeguards accepted
a smaller change. The annotation revision (`outputs/labels/coa/provisional-volume-r1/labels.json`)
records the achieved values.

## Poster panels

Use a consistent family color scheme and minimal text. Export vector PDF/SVG
for plots and a 300 dpi PNG for convenient poster assembly. Each plot should
retain its row-level CSV and source hashes. Proposed panel titles and scope:

| Panel | Figure and purpose | Available evidence / remaining work |
| --- | --- | --- |
| A. Controlled anatomical siblings | One original plus mild/strong narrowing and a companion-anomaly example; matched views and a before/after surface overlay. | Generator reports exist. Select reviewed cases, retain identical view and scale, and show achieved change. |
| B. Population and protected partitions | Family × partition diagram, scenario counts and achieved-change distribution. | Current manifest and label metadata; distinguish six families from 12 IDs. |
| C. Efficient image-based prediction | Coarse image locator → two fine crops → CoA probability and local-volume-loss proxy. Include physical crop sizes and image channels. | Implemented pipeline; high-resolution processing covers selected regions rather than copying whole-body siblings. |
| D. Training and validation behavior | BCE and balanced-accuracy traces; selected epoch marker; lower-rate control added only after completion. | Completed 50-epoch pilot. Training and validation must have separate labels. |
| E. Detection performance | Validation ROC, precision–recall, confusion counts at 0.5, and prediction versus achieved-change scatter. Include edited-only results alongside all scenarios. | Eight adult validation scenarios. ROC FPR increments are 0.25 because there are only four negatives. |
| F. Generalization across anatomy | Thin ROC curves for each held-out development family; equal-family mean curve; family-level AUROC dots and paired changes between arms. | Planned rotating-family experiments; no averaged curve exists yet. |
| G. Virtual-population benefit | Two plots: performance versus distinct scenarios at fixed families; performance versus family diversity at fixed scenario count. | Planned controlled comparisons, not extrapolations from training accuracy. |
| H. Resource use, optional inset | Compact source/overlay storage, cache use, peak host/GPU memory and runtime. | Pilot image cache 4,860 MiB; measured host peak 7,719.84 MiB and GPU peak 1,572.68 MiB. Use the measured pilot scope, not an unmeasured scaling claim. |

Panels A–E can support a development poster now. Panels F–G are the main
experiments needed for the stated population-benefit hypothesis. Planned
panels should appear as a study-design diagram until results exist; do not draw
illustrative performance curves that could be mistaken for measured outcomes.

Reuse the existing pixel-neighborhood methods figure (`outputs/illustrations/bitwise-2d-neighbors/bitwise-morphology-2d-neighbors.pdf`)
(SVG (`outputs/illustrations/bitwise-2d-neighbors/bitwise-morphology-2d-neighbors.svg`),
PNG (`outputs/illustrations/bitwise-2d-neighbors/bitwise-morphology-2d-neighbors.png`))
and cohort workflow neighborhood panel (`outputs/cohorts/coarctation-workflow-v2/panels/H-bitwise-neighborhood.pdf`).
The adult locator preview (`outputs/experiments/coa/multiscale-pilot-r3-review-r1/previews/260602-locator.png`)
provides an image-based localization example. The [methods asset record](integration/2026-09-28/morphology-illustrations.md)
describes these exports.

## Immediate controls before the population experiment

| Control | Change from completed pilot | Hold fixed | Question |
| --- | --- | --- | --- |
| Lower learning rate | Fresh run at 0.0003 instead of 0.001 | Selected 32/8 images, architecture, locator, preprocessing, seed, 50 epochs and 800 updates | Does optimization become more stable? |
| Classification-only | Separate revision disabling the regression loss | Same images, split, classifier capacity and update budget | Does joint proxy regression impair detection? |
| Rotating held-out families | New family-specific training and evaluation partitions | Selected task and fixed protocol | Does performance depend strongly on originating anatomy? |

Keep the first two as development controls; their training curves do not by
themselves estimate a cohort-size benefit. Choose the model and loss protocol
before the controlled population comparison, or use inner validation within
each outer fold. Preserve unsuccessful controls in the study record.

## Controlled population comparisons

All arms need positives and negatives. A normal-only training arm cannot serve
as the conventional supervised CoA-classifier baseline. Choose scenario lists
from generator/annotation metadata before opening evaluation outcomes; maintain
comparable achieved-loss and companion-edit coverage. Use nested subsets where
possible, the same model initialization seed for paired arms, and the same
held-out evaluation cases in every arm.

Select **matched edited-negative/positive pair blocks with identical ascending
dilation and arch-edit parameters**, varying the CoA step. An original-versus-
combined-pathology pair is useful as a secondary reference comparison but can
confound CoA with companion abnormalities. It should not replace those matched
blocks in the primary population study. The completed pilot contains both a
matched edited pair and an original/lower-positive pair, so the new all-matched
panel is a new selection rather than an identical repeat of that pilot.

A metadata audit found three companion configurations per development phantom.
Most have one edited negative and four positives per configuration; the 5-year
family has two negatives and three positives. Consequently the common existing
budget is **three distinct matched pairs, or six scenarios per phantom**.
An original provides an additional negative but lacks those companion edits.
The following grid makes the availability limit explicit; both related phantom
IDs remain included in each selected family.

| Experiment | Training families | Matched scenarios per phantom | Total training scenarios | Availability |
| --- | ---: | ---: | ---: | --- |
| Size S2 | 4 | 2: one negative, one positive | 16 | Existing metadata supports selection |
| Size S4 | 4 | 4: two negative, two positive | 32 | Existing metadata supports new all-matched selection |
| Size S6 | 4 | 6: three negative, three positive | 48 | Existing metadata supports selection |
| Size S8 | 4 | 8: four negative, four positive | 64 | Requires additional reviewed matched pair(s) |
| Initial diversity D1-small | 1 | 4: two negative, two positive | 8 | Existing metadata supports selection |
| Initial diversity D2-small | 2 | 2: one negative, one positive | 8 | Existing metadata supports selection |
| Diversity D1 | 1 | 8: four negative, four positive | 16 | Requires additional pair(s) for most eligible families |
| Diversity D2 | 2 | 4: two negative, two positive | 16 | Existing metadata supports selection |
| Diversity D4 | 4 | 2: one negative, one positive | 16 | Existing metadata supports selection |

The size series holds originating families fixed. The diversity series holds
total count and class balance fixed. Both hold the evaluation panel, achieved-
change sampling policy and optimization budget fixed. Generate CoA-off or low-
CoA siblings with the same companion edits when a fourth negative is needed,
review them, and freeze a new cohort revision. Do not silently duplicate a
negative image and call it new morphology coverage.

Start with the ready eight-scenario D1-small/D2-small comparison while the
fourth matched pair is being prepared. For each diversity arm, use all eligible
family subsets or prespecified balanced subsets;
do not select the families that perform best on the adult validation panel.
Plot those subset results as repeated configurations, not independent patient
replicates. These adult-panel comparisons are useful development studies;
rotating-family evaluation is needed to avoid resting the conclusion on one
adult family.

The primary comparison should match **optimizer updates**, proposed at 768
balanced batch-of-two updates, rather than matching epochs while giving a
larger cohort more optimization. Report distinct scenarios, optimizer updates,
sample presentations and family counts separately. An epoch-matched comparison
can be a named sensitivity analysis. Repeated presentations increase training
exposure; they do not increase distinct cohort size.

At 768 updates and 1,536 image presentations, the 16-, 32-, 48- and 64-scenario
arms receive 96, 48, 32 and 24 complete paired epochs respectively. The
optimization controls retain their original 800-update budget, which would
give 100, 50 and 25 complete epochs for 16, 32 and 64 scenarios. These are
separate experiment groups; compare equal budgets within each group. The
current pilot runner requires a specific 40-image panel
resident in RAM; it does not already run these population arms. A 64-training /
8-validation panel would need 8,748 MiB just for the normalized images, above
the present 5,120 MiB cache. Add bounded loading/cache support before that arm,
without changing the predictor inputs or counting streamed rereads as new data.

In a nested rotating-family fold, only three families are available for inner
training after excluding the outer held-out family and one inner validation
family. The four-family table cannot be copied into that partition. Start with
size arms over those three families and the ready eight-scenario D1-small/
D2-small comparison, followed by the 16-scenario D1/D2 comparison when
additional matched pairs exist. Expand the grid only when scenario availability
and balance pass validation. Freeze exact sample IDs for every fold and arm.

For a claim about the entire image pipeline, train both locator and classifier
using each arm's allowed families. A cheaper classifier-only comparison can
share a locator trained within the fold, but must state that its population
effect concerns the fine classifier alone. Count locator misses as pipeline
failures rather than omitting the missed cases.

## Rotating-family evaluation and aggregation

Use the **five development families** for outer leave-one-family-out evaluation;
keep the infant family reserved until the protocol is frozen. In each outer
fold, hold out both related phantom IDs and all their siblings. Perform model,
budget, checkpoint and any threshold selection using only remaining families.
If these choices vary, use inner family validation, then refit with the selected
fixed budget on the four outer development families. Every train-derived
preprocessing step and the locator must obey these partitions. The current
frozen locator cannot simply be reused in folds where it has seen a held-out
family. This follows [grouped validation principles](https://scikit-learn.org/stable/modules/cross_validation.html#cross-validation-iterators-for-grouped-data)
and [nested model-selection guidance](https://scikit-learn.org/stable/auto_examples/model_selection/plot_nested_cross_validation_iris.html).

The repository's `fakect_evaluation_design` already plans family-preserving
holdout and nested leave-one-out assignments. The focused pilot does **not**
yet execute this rotating training schedule. New fold-specific experiments
and their frozen manifests are required.

For each held-out family, retain sample probabilities, labels, achieved loss,
roles, fold, training-arm IDs and model seed. Calculate family ROC/AUROC and
operating-point sensitivity/specificity, with an edited-only companion table.
With training repeats, average a metric across seeds within each family first;
show seed spread separately as optimization variability. Then average equally
across the five held-out families.

For the average ROC, interpolate each family/seed curve onto the same
prespecified FPR grid, average repeats within family and then average families
equally. Show faint individual-family curves and an explicitly **descriptive**
SD band. Report mean family AUROC alongside the curve. The area beneath an
interpolated mean curve and AUROC from pooled out-of-fold probabilities are
different summaries; label them separately and do not substitute pooled AUROC
for the family mean. The [official ROC example](https://scikit-learn.org/stable/auto_examples/model_selection/plot_roc_crossval.html)
illustrates common-grid averaging; the [cross-validation documentation](https://scikit-learn.org/stable/modules/cross_validation.html#obtaining-predictions-by-cross-validation)
also distinguishes fold summaries from pooled predictions.

Show every family as a dot, and compare arms through paired per-family metric
differences on identical evaluation scenarios. Siblings, paired phantom IDs,
seeds and overlapping folds are not additional independent anatomies. With one
current validation family, do not report an anatomy-generalization confidence
interval. Five development families still support only a small descriptive
study. If enough independent families are added later, uncertainty resampling
should use whole families, preserving all their cases and siblings, rather than
resampling individual scans or seeds as independent units.

## Reproducible figure bundle and next deliverables

Generate current measured plots from the frozen experiment in the development
worktree:

```bash
python3 scripts/research/evaluation/build_coa_poster_figures.py \
  --experiment outputs/experiments/coa/focused-pilot-r1 \
  --output outputs/poster/coa/pilot-r3
```

The bundle contains PDF/SVG/PNG plots, the underlying CSV tables, a source
hash manifest and an HTML contact sheet. Its captions identify validation
data, selected epoch, class definition and sample/family counts. The current
`pilot-r3` destination is frozen; choose a new output directory when regenerating.
Population-size
and rotating-family figures are added only when their experiment artifacts
exist. New comparisons get new output directories so the current poster bundle
remains reproducible.

The current bundle's contact sheet (`outputs/poster/coa/pilot-r3/report.html`)
collects the following exports, each with PDF, SVG and 300 dpi PNG versions:

| Export stem | Content |
| --- | --- |
| `fig01-training-validation` | Epoch traces for BCE, AUROC, accuracy and proxy MAE |
| `fig02-empirical-roc` | Empirical training and development-validation ROC; edited-only metrics appear in the table above |
| `fig03-validation-errors` | Confusion counts and all eight exact validation probabilities |
| `fig04-detection-and-severity` | Detection probability versus achieved loss and failed regression parity |
| `fig05-cohort-partition` | Six provisional families and protected partitions; reserved-family metadata only |
| `fig06-model-architecture` | Image locator, fine crops, encoder and prediction heads |

Underlying tables are `training-curves.csv`, `roc-points.csv`,
`selected-predictions.csv` and `cohort-partitions.csv`. `figure-data.json` and
`provenance.json` preserve measurements, source hashes and export identities.
Figure generation reads metadata and saved development predictions; it does
not run TensorFlow inference or open image, checkpoint or reserved-test payloads.

Before the population runs, freeze an experiment registry containing cohort and
annotation hashes; every fold's family/sample assignments; training arms and
subset-selection rules; seeds; locator/classifier source hashes; update budgets;
checkpoint and decision-threshold policies; and all failed or missing runs.
The immediate deliverables are the current measured figure bundle and the
lower-rate control. Next come classification stability, family rotation, and
the matched size/diversity study. Clinical validation remains a separate future
study.
