The CoA cohort revision enlarges every ROI tube diameter by exactly **15%** across all **12 phantoms**, retaining the original centerline points. The inputs live in [configs/case-studies/coa/phantoms](../../../../../configs/case-studies/coa/phantoms/), with [project-radius115.json](../../../../../configs/case-studies/coa/project.json) listing the cases. All **192 native scenarios have now completed successfully**, and all **12 case inputs and native audits are accepted and frozen** for paired extraction. The combined release report (`outputs/cohorts/coa/radius115/review/index.html`) and [final acceptance manifest](radius115-final-review.json) contain the per-case review decisions and freeze fingerprints. At this native-release checkpoint, paired extraction and model fitting had not run. **Follow-up:** all 192 native pairs and their added/removed companions are now prepared and verified; the [two-model run record](two-stage-training-run.md) gives the subsequent training/test status. The native acceptance evidence below remains unchanged.

The executed cohort contains **192 scenarios**: 16 per phantom, comprising one baseline and 15 combinations of three ascending/arch dilation distances and five coarctation erosion distances. Each edited scenario also applies the configured arch erosion. All lesions execute in the ordered recipe, starting from the original phantom for each scenario. Distances remain phantom-specific; the native audits preserve achieved voxel changes, recipient limits and safeguard results.

ROI selection uses artery tissue, the optional thoracic-aorta anatomy filter, native ACT connectivity, RAW surface evidence and the atlas. Pulmonary labels are excluded and protected. Diameter expansion required larger source crops; the largest contains **2,145,882 voxels**. [Input changes](radius115-input-changes.json) preserve original radii, exact scaling, crop adjustments and maximum-distance geometry checks.

The native ROI report (`outputs/cohorts/coa/radius115/roi-review-r1/report.html`) contains sagittal, coronal and axial views for every phantom, with transparent ROI overlays, selected-target contours, atlas aorta context and native coordinates. The [tracked ROI summary](radius115-roi-review-summary.json) retains paths and SHA256 hashes for the per-case JSON/images and the component supplement (`outputs/cohorts/coa/radius115/roi-review-r1/component-details.json`).

All 12 phantoms passed exact radius scaling and unchanged-point checks. Every centerline node lies inside the selected target, and every selected mask is a subset of the ROI and admitted anatomical IDs. Anatomy resolution retained **zero disconnected ROI voxels by whole-ID admission** across the cases. These are local selection checks; they do not measure whole-aorta coverage.

| Phantom | Selected native voxels | Selected component sizes, 6-neighbor |
| --- | ---: | --- |
| 260602 | 54,050 | 54,050 |
| 260611 | 40,254 | 40,254 |
| 260612 | 33,989 | 33,989 |
| 260613 | 49,032 | 49,030, 2 |
| 260614 | 58,832 | 58,832 |
| 260615 | 56,334 | 56,334 |
| 260616 | 45,036 | 45,036 |
| 260617 | 33,483 | 33,483 |
| 260618 | 49,032 | 49,030, 2 |
| 260619 | 58,832 | 58,832 |
| 260620 | 56,334 | 56,334 |
| 260621 | 35,023 | 35,023 |

Cases **260613 and 260618** each contain two baseline ROI-clipped `internal_carotid_right` voxels, original ID **1186**, at native `(i,j,k) = (334,322,1186)` and `(334,323,1186)`. Both connect to the main aortic anchor through native admitted labels outside the ROI. Their parent path positions are **39.18–39.48%**, placing them within the two arch selectors and outside the CoA selector. They are baseline selection fragments, not evidence of an edit-induced aortic discontinuity. The centerline was retained, and these fragments remain the documented baseline for interpreting edit-topology diagnostics.

The revised arch erosion now has an explicit connectivity safeguard: `preserve_connectivity = true`, `backoff_factor = 0.5`, and `max_backoff_steps = 8`. This protects arch/branch continuity without adding the CoA-specific retained-volume floor to the arch edit. CoA keeps its existing local volume and connectivity safeguards. The [manual adjustment record](radius115-manual-adjustments.json) captures these changes and the transition to fresh `r2` outputs.

An additional reassignment issue was identified for original ID **4**, `chest_surface`. Its coarse material category is soft tissue, while an atlas naming heuristic associates it with skin and therefore high stiffness. The [initial interior-fill check](radius115-internal-fill-review.json) is preserved as superseded evidence. The current [prospective interior proof, revision 2](radius115-internal-fill-review-r2.json), uses the **exact Euclidean distance-transform expansion of the native ROI voxel mask** by the maximum cumulative swept dilation budget, including varying-radius transitions. It subtracts the **full native voxel diagonal** from donor-to-background center distances to bound separation between the complete voxel cells. ACT background, internal cavities and padded crop boundaries are treated conservatively, and an envelope extending beyond the loaded crop is rejected.

All **12 cases pass the 3 mm margin** under this stronger test. The smallest conservative clearance is **5.634 mm**, in cases 260613/260618; every case has zero tested label-4 voxels within the forbidden margin. This updated result supports the recorded interior-only override envelope without relying on the superseded 6.07 mm estimate.

The per-case `[stiffness.labels]` override `4 = 0.05` is limited to these documented geometry/edit envelopes. It does not globally reclassify skin or establish tissue histology. The bone and skin defaults remain **1.0** and **0.95**. Any expanded/repositioned ROI or larger dilation budget requires renewed clearance evidence before relying on this exception.

The initial native sweep was interrupted after a shared-resource-limit inconsistency: metadata/morphology accepted enlarged crops, but the lineage helper still enforced the old two-million-voxel ceiling and failed with `candidates must be a bounded native Boolean crop`. Logs and partial `preflight-r1` outputs are preserved. [fakect_limits.py](../../../../../src/fakect_limits.py) now supplies one **3,000,000-voxel** native edit limit to morphology, direction, lineage and associated profile/surface previews; display budgets remain separate. Corrected trials use new output revisions.

Corrected minimum/maximum corner reviews completed for all 12 phantoms. The table uses `trial-r2` for nine cases, `trial-r3` for **260614/260619**, and `trial-r4` for **260617**. Every active edit changed native voxels in these accepted corner trials, and the accepted erosions passed their configured connectivity checks. The complete parameter grids were subsequently executed and reviewed; their results are summarized below.

The table records **minimum / maximum corner** voxel counts, in recipe order. These are achieved changes in each step, not percentages of stenosis, calibrated severity or final net volume change. Links open the native trial audit.

| Phantom | Dilation added | Arch erosion removed | CoA erosion removed | CoA maximum safeguard backoff |
| --- | ---: | ---: | ---: | --- |
| 260602 (`outputs/cohorts/coa/radius115/260602/trial-r2/trial.json`) | 1,055 / 5,163 | 561 / 561 | 396 / 1,194 | none |
| 260611 (`outputs/cohorts/coa/radius115/260611/trial-r2/trial.json`) | 879 / 4,473 | 286 / 286 | 290 / 902 | none |
| 260612 (`outputs/cohorts/coa/radius115/260612/trial-r2/trial.json`) | 158 / 1,258 | 225 / 225 | 206 / 1,338 | none |
| 260613 (`outputs/cohorts/coa/radius115/260613/trial-r2/trial.json`) | 565 / 3,079 | 196 / 196 | 269 / 1,698 | none |
| 260614 (`outputs/cohorts/coa/radius115/260614/trial-r3/trial.json`)* | 145 / 1,650 | 315 / 315 | 218 / 1,136 | none |
| 260615 (`outputs/cohorts/coa/radius115/260615/trial-r2/trial.json`) | 861 / 4,447 | 473 / 473 | 423 / 1,812 | none |
| 260616 (`outputs/cohorts/coa/radius115/260616/trial-r2/trial.json`) | 909 / 4,566 | 369 / 369 | 305 / 1,295 | none |
| 260617 (`outputs/cohorts/coa/radius115/260617/trial-r4/trial.json`) | 146 / 1,118 | 205 / 205 | 209 / 705 | none; revised maximum 1.4805 mm |
| 260618 (`outputs/cohorts/coa/radius115/260618/trial-r2/trial.json`) | 565 / 3,079 | 196 / 196 | 269 / 1,698 | none |
| 260619 (`outputs/cohorts/coa/radius115/260619/trial-r3/trial.json`)* | 145 / 1,650 | 315 / 315 | 218 / 1,136 | none |
| 260620 (`outputs/cohorts/coa/radius115/260620/trial-r2/trial.json`) | 861 / 4,447 | 473 / 473 | 423 / 1,812 | none |
| 260621 (`outputs/cohorts/coa/radius115/260621/trial-r2/trial.json`) | 922 / 4,609 | 359 / 359 | 305 / 1,147 | none |

\* For **260614 and 260619**, the original 0.571 mm minimum CoA request removed **zero voxels** in `trial-r2`. This was a native-grid no-op with 0.6 mm spacing and artery resistance, rather than a missing edit step. Raising only that sweep minimum to **0.8 mm** produced **218 removed voxels** in each `trial-r3` minimum trial; the other four CoA distances remain 1.141, 1.712, 2.282 and 2.853 mm. The five-level sweep size and 192-scenario total are unchanged. Both unsuccessful trials remain available in their original directories.

Case **260617** demonstrates an actual connectivity backoff: its original maximum CoA request of **2.961 mm** split or lost an affected component in the first attempt. The accepted **1.4805 mm** trial retained connectivity and removed 705 voxels. This reduction is distinct from incomplete reassignment. The revised input now requests CoA distances **0.592, 0.8, 1.0, 1.2 and 1.4805 mm**, and its base CoA edit is **1.4805 mm**. The sweep retains five levels, so the cohort contains 192 scenarios. All of this case's review, preflight, freeze and paired-output directories use **`r4`**. The replacement sweep produced 16 distinct masks, with strictly increasing achieved changes along both requested-distance axes and no safeguard backoff.

Only 260617's original revision-2 sweep was canceled for this range correction. Two intermediate revision-3 submissions were also canceled after an exact-text assertion caught an incorrectly transcribed prior value; their diagnostics remain preserved in the [manual adjustment record](radius115-manual-adjustments.json). The replacement revision-4 full-grid job **6855168** and corner-trial job **6855170** were launched from the corrected input. Both replacement runs completed; the revision-4 full grid passed all 16 scenarios. The canceled runs remain diagnostic history and are excluded from the accepted cohort.

The **260612 maximum CoA scenario** remains natively connected but was classified during manual review as **severe and limited by the native resolution**, with a nearly occluded cross-section. Its accepted trial removes 1,338 voxels and retains about 42.65% of the target volume within the guard's affected region; this volume ratio does not describe the minimum lumen area. Six-neighbor connectivity does not establish a well-resolved residual lumen. Retain this severity flag when reviewing the full cohort or selecting training/test scenarios.

Several accepted erosions remove fewer voxels than proposed because the recipient search remains limited to **3 mm** and `unresolved = preserve` keeps voxels without an eligible recipient. For example, the 260602 maximum CoA trial proposed 2,390 removals but completed 1,194; 1,196 unresolved voxels remained unchanged. Its requested distance was accepted without safeguard backoff. The corresponding arch erosion completed 561 of 832 proposed removals. Conversely, the corrected 260614/260619 minimum CoA trials completed 218 of 222 proposed removals, preserving four unresolved voxels. Such partial erosion must be evaluated using achieved geometry and the recorded unresolved counts; it is not evidence that the recipe skipped the erosion. The recipient-search limit has not been silently enlarged.

The final native outcome review (`outputs/cohorts/coa/radius115/outcome-review-final.html`) covers **192 / 192 successful scenarios**, comprising 12 baselines and 180 edited variants. Every phantom has **16 distinct target masks**. All **540 active edit passes** changed voxels; all **360 erosion-guard checks** accepted their requested distances without backoff under the corrected final inputs. All **96 distance-response sequences** showed strictly increasing achieved changes when the other sweep axis was held fixed. These are native execution and response checks, not calibrated clinical severity measurements.

The automated outcome review contains **zero rejection flags and 216 review flags**, which were retained and interpreted in the [manual acceptance manifest](radius115-final-review.json):

- **198 partial-release flags:** recipient-limited erosions preserve unresolved target voxels. Their achieved removals and unresolved counts remain in the native audit.
- **10 directional-support flags:** at the 3.362 mm maximum dilation in 260613/260618, **3 of 67,022 directional-envelope cells** have unreliable curvature directions. The engine assigns zero directional weight and asserts that those cells remain unchanged; the [directional-support evidence](radius115-direction-review.json) confirms unchanged labels and attenuation at all three cells in both cases. There are 3,079 target additions elsewhere. Each affected case has five CoA combinations at that dilation level.
- **6 case flags for matching target masks:** the 260613/260618, 260614/260619 and 260615/260620 pairs each share 16 scenario masks, producing **48 cross-case matching-mask groups**.
- **2 case flags for matching source crops:** only 260614/260619 share both complete native ACT and ATN crop hashes. This is a crop-level statement, not a claim about the entire phantoms.

Repeated anatomy remains identified by case provenance. The 12 case IDs are **not 12 independent anatomies**; keep related anatomy families together in later training/validation/test assignments to avoid leakage.

Manual maximum-edit review also flags **260602, 260611, 260616 and 260621** as dilations involving arch-branch junctions. The continuity checks preserve affected native components, but they do not certify absence of within-tree shortcuts, minimum lumen caliber or complete vascular topology. Together with the severe, nearly occlusive 260612 cases, these remain explicit scenario limitations. This release is a technically reviewed synthetic cohort; clinical realism has not been validated.

Verification passed the **507-test full suite**, followed by **20 updated focused checks**: six interior-fill, three visual-review and eleven outcome-review tests. The focused checks overlap the suite and are not an additional count of unique tests.

All **12 freeze records have been created and validated**. Each contains the reviewed input snapshot, native preflight audit and generation fingerprint. Output revisions are **`r2` for eleven cases** and **`r4` for 260617**. Final input changes were limited to `parameters_reviewed = true` and documentation references to the updated evidence; parsed generation settings were checked unchanged before freezing. The final manifest records original-review and frozen-input hashes alongside each preflight and freeze SHA256.

Native **preflight** executed every configured virtual scenario and retained outcome audits, provenance and diagnostic reports. It did **not** export a complete collection of paired NPZ samples. The freeze now locks the accepted input and audit for reproducible extraction. **Prepare** regenerates the accepted scenarios and writes image/target pairs into the case's configured `dataset_directory`.

The next command, from the integration worktree, extracts pairs for one already-frozen case:

```bash
python3 scripts/prepare_cohort.py --config configs/case-studies/coa/phantoms/260602.ini --stage prepare
```

Replace `260602` with each other case ID to extract that phantom's pairs. The command follows the current INI paths, including **`pairs-r4` for 260617** and **`pairs-r2` for the other eleven cases**. Do not rerun `--stage freeze` into these existing freeze directories. Later parameter changes require a new revision and renewed validation/review. Model training remains decoupled and consumes prepared cohort metadata after paired extraction.

At this release, native scenario generation, review and metadata freezing are complete. **Paired-array extraction and model training remain unperformed.**
