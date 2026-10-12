"""Exact ROC, identical panels, train-only gates and immutable metadata reports."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fakect_coa_budget_report import (analyze_records, checked_history, corrected_mean_roc,
                                    source_bundle, write_report)
from fakect_coa_rotation_report import rotation_manifest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n")
    return digest


def fixture(root, completed=5):
    parent, control = root / "parent", root / "control"
    families = ["family-" + c for c in "ABCDE"]
    inventory = [dict(case_id=f + str(i), anatomy_family=f) for f in families for i in range(2)]
    samples = []
    for case in inventory:
        for i in range(6):
            label = int(i >= 3)
            samples.append(dict(sample_id=case["case_id"] + ":" + str(i), case_id=case["case_id"],
                                annotation={"coa_benchmark_positive": bool(label),
                                            "coa_local_target_volume_loss_fraction": .1 + .4 * label}))
    stages = []
    for family in families:
        test = [r["case_id"] for r in inventory if r["anatomy_family"] == family]
        train = [r["case_id"] for r in inventory if r["anatomy_family"] != family]
        stages.append(dict(id="family_holdout/" + family + "/seed-42/outer_refit", protocol="family_holdout",
                           seed=42, stage="outer_refit", outer_fold_id=family, train_case_ids=train,
                           validation_case_ids=[], final_test_case_ids=test, parent_inner_stage_ids=[],
                           train_sample_ids=[s["sample_id"] for s in samples if s["case_id"] in train],
                           final_test_sample_ids=[s["sample_id"] for s in samples if s["case_id"] in test]))
    settings = dict(classifier_epochs=50, regression_weight=1., classifier_learning_rate=.001)
    old = dict(inventory=inventory, stages=stages, reserved_case_ids=["reserved-infant"],
               samples=samples, prediction_threshold=.5, config={"training": settings})
    parent_hash = write_json(parent / "experiment-lock.json", old)
    manifest = rotation_manifest(old)
    dependencies = {}
    baseline = []
    for i, stage in enumerate(stages):
        rows = []
        for sample in samples:
            if sample["case_id"] in stage["final_test_case_ids"]:
                label = int(sample["annotation"]["coa_benchmark_positive"])
                score = sample["annotation"]["coa_local_target_volume_loss_fraction"]
                rows.append(dict(protocol="family_holdout", fold_id=stage["outer_fold_id"], seed=42,
                                 sample_id=sample["sample_id"], case_id=sample["case_id"],
                                 outer_case=None, outer_family=stage["outer_fold_id"], family=stage["outer_fold_id"],
                                 split="final_test", label=label, probability=.5, score_label=score,
                                 score_prediction=.3, locator_coverage_passed=True, pipeline_failure=False))
        baseline.extend(rows)
        old_result = dict(status="completed", stage_id=stage["id"], protocol="family_holdout",
                          source_lock_sha256=parent_hash, final_test_evaluated=True, reserved_test_evaluated=False,
                          classifier={"fresh_initial_weights_sha256": str(i) * 64}, final_predictions=rows)
        path = parent / "stages" / stage["id"] / "results.json"
        dependencies[str(path)] = write_json(path, old_result)
    new_stages = [dict(s, expected_initial_weights_sha256=str(i) * 64) for i, s in enumerate(stages)]
    lock = dict(parent_directory=str(parent), parent_lock_sha256=parent_hash,
                stages=new_stages, reserved_case_ids=old["reserved_case_ids"], fold_manifest=manifest,
                unchanged_training_settings=settings, dependency_sha256=dependencies,
                config={"training": {"classifier_epochs": 200, "checkpoint_interval": 50}})
    own_hash = write_json(control / "experiment-lock.json", lock)
    controls = []
    for i, stage in enumerate(new_stages[:completed]):
        rows = [dict(r, probability=.9 if r["label"] else .1, score_prediction=r["score_label"])
                for r in baseline if r["fold_id"] == stage["outer_fold_id"]]
        controls.extend(rows)
        receipt = dict(selected_epoch=200, epochs_completed=200, validation_sample_ids=[], validation=None,
                       training_sample_ids=stage["train_sample_ids"], fresh_initial_weights_sha256=str(i) * 64,
                       optimizer_updates=4800, checkpoint_sha256="a" * 64)
        history = {"epochs": [], "selected_epoch": 200, "checkpoint_selection": "final fixed-budget epoch"}
        for epoch in range(1, 201):
            history["epochs"].append(dict(epoch=epoch, optimizer_updates=epoch * 24,
                                         validation=None, selection_bce=None,
                                         training_pair_order=[stage["train_sample_ids"][i:i+2] for i in range(0, 48, 2)],
                                         training={"bce": .7 / epoch, "accuracy": .9, "score_mae": .05}))
        result = dict(status="completed", stage_id=stage["id"], protocol="family_holdout", source_lock_sha256=own_hash,
                      parent_lock_sha256=parent_hash, classifier_selected_epoch=200,
                      classifier=receipt, final_predictions=rows, final_test_evaluated=True,
                      reserved_test_evaluated=False, locator_retrained=False, locator_proposals_changed=False)
        folder = control / "stages" / stage["id"]
        write_json(folder / "results.json", result)
        write_json(folder / "classifier/receipt.json", receipt)
        write_json(folder / "classifier/history.json", history)
        write_json(folder / "final-test/predictions.json", dict(predictions=rows, checkpoint_sha256="a" * 64, source_lock_sha256=own_hash))
    return control, lock, baseline, controls, manifest


class BudgetReportTests(unittest.TestCase):
    # Purpose: Verify paired budget comparisons retain exact tied-score metrics.
    # Inputs: Five-family fixture with tied baseline and perfect epoch-200 controls.
    # Outputs: Baseline AUROC 0.5, control AUROC 1, 60 matched pairs, and zero severity MAE.
    def test_exact_ties_and_paired_tables(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, baseline, controls, manifest = fixture(Path(tmp))
            analysis = analyze_records(baseline, controls, manifest)
            self.assertEqual(analysis["status"], "complete")
            self.assertEqual(analysis["protocols"]["baseline"]["pooled_scenario_metrics"]["auroc"], .5)
            self.assertEqual(analysis["protocols"]["epoch200"]["equal_family_metrics"]["auroc"]["mean"], 1.)
            self.assertEqual(analysis["paired_comparisons"][0]["matched_sample_seed_pairs"], 60)
            self.assertEqual(analysis["paired_comparisons"][0]["equal_family_delta"]["auroc"]["mean"], .5)
            self.assertEqual(analysis["protocols"]["epoch200"]["equal_family_metrics"]["severity_mae_percentage_points"]["mean"], 0.)

    # Purpose: Verify mean ROC interpolation preserves ordered vertical steps.
    # Inputs: P,N,N,P ROC path with AUROC 0.5.
    # Outputs: TPR 0.5 at FPR 0.75, exact area 0.5, and finite-grid area 0.50025.
    def test_ordered_vertical_roc_steps_avoid_optimistic_diagonal(self):
        # P,N,N,P gives [(0,0),(0,.5),(.5,.5),(1,.5),(1,1)].
        # Removing the lower endpoint at FPR=1 overstates the area to .75.
        roc = dict(fpr=[0., 0., .5, 1., 1.], tpr=[0., .5, .5, .5, 1.])
        result = corrected_mean_roc([dict(protocol="x", family="A", auroc=.5, roc=roc)], "x")
        self.assertEqual(result["mean_tpr"][750], .5)
        self.assertEqual(result["exact_mean_family_auroc"], .5)
        self.assertAlmostEqual(result["finite_grid_integral"], .50025)
        self.assertIsNone(result["confidence_interval"])

    # Purpose: Verify incomplete control runs remain partial.
    # Inputs: Five-family fixture with only one completed control stage.
    # Outputs: Partial analysis, pending paired comparison, four pending stages, and 72 records.
    def test_pending_control_has_no_full_paired_comparison(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lock, _, _, _ = fixture(Path(tmp), completed=1)
            analysis = source_bundle(root, lock)["analysis"]
            self.assertEqual(analysis["status"], "partial")
            self.assertEqual(analysis["paired_comparisons"][0]["status"], "pending")
            self.assertEqual(sum(s["status"] == "pending" for s in analysis["stage_status"]), 4)
            self.assertEqual(len(analysis["records"]), 72)

    # Purpose: Verify paired analyses enforce complete, finite, leakage-free panels.
    # Inputs: Duplicate/missing rows, NaN score, reserved case, wrong split/label, and fold overlap.
    # Outputs: ValueError for every invalid prediction panel or train/evaluation overlap.
    def test_leakage_nonfinite_duplicates_and_incomplete_panels_rejected(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, baseline, controls, manifest = fixture(Path(tmp))
            failures = [controls + controls[:1], controls[1:],
                        [dict(controls[0], probability=float("nan"))] + controls[1:],
                        [dict(controls[0], case_id="reserved-infant")] + controls[1:],
                        [dict(controls[0], split="validation")] + controls[1:],
                        [dict(controls[0], score_label=.999)] + controls[1:]]
            for rows in failures:
                with self.subTest(first=rows[0]), self.assertRaises(ValueError):
                    analyze_records(baseline, rows, manifest)
            changed = copy.deepcopy(manifest)
            changed["folds"][0]["training_case_ids"].append(changed["folds"][0]["evaluation_case_ids"][0])
            with self.assertRaises(ValueError):
                analyze_records(baseline, controls, changed)

    # Purpose: Verify fixed-budget selection uses only TRAIN history and final epoch.
    # Inputs: 200-epoch history altered with validation, wrong updates, epoch 50, or held-out pair.
    # Outputs: Valid history accepted and ValueError for each selection/update/leakage mutation.
    def test_train_only_final_epoch_gate(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lock, _, _, _ = fixture(Path(tmp), completed=1)
            stage = lock["stages"][0]
            folder = root / "stages" / stage["id"]
            history = json.loads((folder / "classifier/history.json").read_text())
            receipt = json.loads((folder / "classifier/receipt.json").read_text())
            self.assertEqual(len(checked_history(history, receipt, stage)), 200)
            for field, value in (("validation", {"bce": .1}), ("selection_bce", .1), ("optimizer_updates", 25)):
                changed = copy.deepcopy(history)
                changed["epochs"][0][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    checked_history(changed, receipt, stage)
            bad = dict(receipt, selected_epoch=50)
            with self.assertRaises(ValueError):
                checked_history(history, bad, stage)
            changed = copy.deepcopy(history)
            changed["epochs"][0]["training_pair_order"][0][0] = stage["final_test_sample_ids"][0]
            with self.assertRaises(ValueError):
                checked_history(changed, receipt, stage)

    # Purpose: Verify source bundles reject provenance and completion drift.
    # Inputs: Mutated parent pin, receipt epoch, result lock, predictions, or parent INCOMPLETE marker.
    # Outputs: ValueError for each altered dependency or incompatible completion artifact.
    def test_source_hash_receipt_and_lock_identity_gates(self):
        for mutation in ("parent_pin", "receipt", "own_result_lock", "prediction_artifact", "parent_INCOMPLETE"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root, lock, _, _, _ = fixture(Path(tmp), completed=1)
                stage = lock["stages"][0]
                folder = root / "stages" / stage["id"]
                if mutation == "parent_pin":
                    path = Path(lock["parent_directory"]) / "stages" / stage["id"] / "results.json"
                    old = json.loads(path.read_text());old["final_predictions"][0]["probability"] = .4
                    write_json(path, old)
                elif mutation == "parent_INCOMPLETE":
                    (Path(lock["parent_directory"]) / "stages" / stage["id"] / "INCOMPLETE").touch()
                else:
                    name = {"receipt": "classifier/receipt.json", "own_result_lock": "results.json", "prediction_artifact": "final-test/predictions.json"}[mutation]
                    path = folder / name
                    row = json.loads(path.read_text())
                    if mutation == "receipt":
                        row["selected_epoch"] = 50
                    elif mutation == "own_result_lock":
                        row["source_lock_sha256"] = "b" * 64
                    else:
                        row["predictions"][0]["probability"] = .2
                    write_json(path, row)
                with self.assertRaises(ValueError):
                    source_bundle(root, lock)

    # Purpose: Verify failed-stage markers are distinguished from pending stages.
    # Inputs: No completed controls, one hashed failure marker, then tampered marker bytes.
    # Outputs: Blocked-by-failure status, one failed/four pending stages, and checksum rejection.
    def test_failed_marker_checked_and_pending_not_failed(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lock, _, _, _ = fixture(Path(tmp), completed=0)
            folder = root / "stages" / lock["stages"][0]["id"]
            path = folder / "failure.json"
            write_json(path, {"message": "Fixture hardware failure"})
            report = source_bundle(root, lock)["analysis"]
            self.assertEqual(report["status"], "blocked_by_failed_stage")
            self.assertEqual(sum(s["status"] == "failed" for s in report["stage_status"]), 1)
            self.assertEqual(sum(s["status"] == "pending" for s in report["stage_status"]), 4)
            path.write_text('{"message":"altered"}')
            with self.assertRaisesRegex(ValueError, "checksum"):
                source_bundle(root, lock)

    # Purpose: Verify report revisions preserve prior artifacts and expose provenance.
    # Inputs: Complete five-family fixture rendered twice.
    # Outputs: Three figures per format, 120 records, verified hashes, and immutable review-r1 plus review-r2.
    def test_fresh_report_revisions_and_artifact_provenance(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lock, _, _, _ = fixture(Path(tmp))
            result = write_report(root, lock)
            out = Path(result["output_directory"])
            self.assertEqual(out.name, "review-r1")
            self.assertEqual(result["figure_count"], 3)
            self.assertEqual(result["records"], 120)
            for extension in ("pdf", "svg", "png"):
                self.assertEqual(len(list(out.glob("*." + extension))), 3)
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()}
            proof = json.loads((out / "provenance.json").read_text())
            self.assertFalse(proof["image_or_checkpoint_payloads_read"])
            self.assertFalse(proof["reserved_test_predictions_read"])
            self.assertFalse(proof["intermediate_outer_predictions_read"])
            for name, expected in proof["artifact_sha256"].items():
                self.assertEqual(before[name], expected)
            with (out / "family-metrics.csv").open() as handle:
                self.assertEqual(len(list(csv.DictReader(handle))), 10)
            result2 = write_report(root, lock)
            self.assertEqual(Path(result2["output_directory"]).name, "review-r2")
            self.assertIn("review-r2/report.html", (root / "report.html").read_text())
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in out.iterdir()})


if __name__ == "__main__":
    unittest.main()
