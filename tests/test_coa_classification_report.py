"""Matched classification control: isolation, disabled head and report integrity."""
import copy
import csv
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

from fakect_coa_classification_report import (BASELINE, CONTROL, LOCK_SCHEMA,
                                            analyze_records, checked_history,
                                            source_bundle, write_report)
from fakect_coa_rotation_report import rotation_manifest


def write_json(path, value):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, indent=2) + "\n")
    digest = hashlib.sha256(path.read_bytes()).hexdigest()
    Path(str(path) + ".sha256").write_text(digest + "\n")
    return digest


def fixture(root, completed=5):
    parent, comparison, control = (root / n for n in ("parent", "comparison", "control"))
    inventory = [dict(case_id=f + str(i), anatomy_family=f) for f in ("family-" + c for c in "ABCDE") for i in range(2)]
    samples = []
    for case in inventory:
        for i in range(6):
            samples.append(dict(sample_id=case["case_id"] + ":" + str(i), case_id=case["case_id"],
                                annotation={"coa_benchmark_positive": bool(i % 2),
                                            "coa_local_target_volume_loss_fraction": .1 + .4 * (i % 2)}))
    stages = []
    for i, family in enumerate(sorted({r["anatomy_family"] for r in inventory})):
        train = [r["case_id"] for r in inventory if r["anatomy_family"] != family]
        test = [r["case_id"] for r in inventory if r["anatomy_family"] == family]
        stages.append(dict(id="family_holdout/" + family + "/seed-42/outer_refit", protocol="family_holdout",
                           seed=42, stage="outer_refit", outer_fold_id=family, train_case_ids=train,
                           validation_case_ids=[], final_test_case_ids=test, parent_inner_stage_ids=[],
                           train_sample_ids=[s["sample_id"] for s in samples if s["case_id"] in train],
                           final_test_sample_ids=[s["sample_id"] for s in samples if s["case_id"] in test]))
    settings = dict(classifier_epochs=50, regression_weight=1., classifier_learning_rate=.001)
    original = dict(inventory=inventory, stages=stages, reserved_case_ids=["reserved-infant"],
                    samples=samples, prediction_threshold=.5, config={"training": settings})
    parent_hash = write_json(parent / "experiment-lock.json", original)
    stages = [dict(s, expected_initial_weights_sha256=str(i) * 64) for i, s in enumerate(stages)]
    manifest = rotation_manifest(original)
    prior = dict(schema_version="fakect.coa-budget-control/1", parent_directory=str(parent),
                 parent_lock_sha256=parent_hash, stages=stages, reserved_case_ids=original["reserved_case_ids"],
                 unchanged_training_settings=settings, fold_manifest=manifest,
                 config={"training": {"classifier_epochs": 200, "checkpoint_interval": 50}})
    comparison_hash = write_json(comparison / "experiment-lock.json", prior)
    dependencies = {}
    baseline, history_by_stage, initialization_by_stage, audits_by_stage = [], {}, {}, {}
    for stage in stages:
        all_rows = []
        for sample in samples:
            label = int(sample["annotation"]["coa_benchmark_positive"])
            all_rows.append(dict(protocol="family_holdout", fold_id=stage["outer_fold_id"], seed=42,
                                 sample_id=sample["sample_id"], case_id=sample["case_id"],
                                 family=next(r["anatomy_family"] for r in inventory if r["case_id"] == sample["case_id"]),
                                 outer_case=None, outer_family=stage["outer_fold_id"],
                                 split="train" if sample["case_id"] in stage["train_case_ids"] else "final_test",
                                 label=label, logit=0., probability=.5,
                                 score_label=sample["annotation"]["coa_local_target_volume_loss_fraction"], score_prediction=.3,
                                 locator_coverage_passed=True, pipeline_failure=False))
        rows = [r for r in all_rows if r["split"] == "final_test"]
        baseline.extend(rows)
        receipt = dict(selected_epoch=200, epochs_completed=200, validation_sample_ids=[], validation=None,
                       training_sample_ids=stage["train_sample_ids"], fresh_initial_weights_sha256=stage["expected_initial_weights_sha256"],
                       optimizer_updates=4800, checkpoint_sha256="a" * 64)
        history = dict(selected_epoch=200, checkpoint_selection="final fixed-budget epoch", epochs=[])
        for epoch in range(1, 201):
            history["epochs"].append(dict(epoch=epoch, optimizer_updates=epoch * 24, validation=None, selection_bce=None,
                                         training_pair_order=[stage["train_sample_ids"][i:i+2] for i in range(0, 48, 2)],
                                         training={"bce": .7 / epoch, "accuracy": .9, "score_mae": .05}))
        initial = dict(seed=42, weights_sha256=stage["expected_initial_weights_sha256"], fresh_weights=True,
                       predictions=[r for r in all_rows if r["split"] == "train"])
        result = dict(status="completed", stage_id=stage["id"], protocol="family_holdout", source_lock_sha256=comparison_hash,
                      parent_lock_sha256=parent_hash, classifier_selected_epoch=200, classifier=receipt, final_predictions=rows,
                      final_test_evaluated=True, reserved_test_evaluated=False, locator_retrained=False, locator_proposals_changed=False)
        previous = comparison / "stages" / stage["id"]
        for name, value in (("results.json", result), ("classifier/history.json", history), ("classifier/initialization.json", initial)):
            path = previous / name
            dependencies[str(path)] = write_json(path, value)
        audits = {}
        for name, role in (("proposal-audit.json", "train"), ("final-test/proposal-audit.json", "final_test")):
            audit = dict(passed=True, selection_allowed=False, samples=[dict(sample_id=r["sample_id"], split=role,
                         coverage_passed=True, origin_kji=[1, 2, 3]) for r in all_rows if r["split"] == role])
            for folder in (parent / "stages" / stage["id"], previous):
                path = folder / name
                dependencies[str(path)] = write_json(path, audit)
            audits[name] = audit
        history_by_stage[stage["id"]] = history
        initialization_by_stage[stage["id"]] = initial
        audits_by_stage[stage["id"]] = audits
    lock = dict(schema_version=LOCK_SCHEMA, parent_directory=str(parent), parent_lock_sha256=parent_hash,
                comparison_directory=str(comparison), comparison_lock_sha256=comparison_hash, stages=stages,
                reserved_case_ids=original["reserved_case_ids"], fold_manifest=manifest, dependency_sha256=dependencies,
                unchanged_training_settings=settings, intended_training_settings=dict(settings, classifier_epochs=200, regression_weight=0.),
                classifier_epochs=200, config={"training": {"classifier_epochs": 200, "checkpoint_interval": 50, "regression_weight": 0.}})
    own_hash = write_json(control / "experiment-lock.json", lock)
    controls = []
    for stage in stages[:completed]:
        rows = [dict(r, probability=.9 if r["label"] else .1, score_prediction=None) for r in baseline if r["fold_id"] == stage["outer_fold_id"]]
        controls.extend(rows)
        receipt = dict(selected_epoch=200, epochs_completed=200, validation_sample_ids=[], validation=None,
                       training_sample_ids=stage["train_sample_ids"], fresh_initial_weights_sha256=stage["expected_initial_weights_sha256"],
                       optimizer_updates=4800, checkpoint_sha256="b" * 64, regression_weight=0., severity_trained=False,
                       training={"score_mae": None})
        history = copy.deepcopy(history_by_stage[stage["id"]])
        for row in history["epochs"]:
            row["training"]["score_mae"] = None
        result = dict(status="completed", stage_id=stage["id"], protocol="family_holdout", source_lock_sha256=own_hash,
                      parent_lock_sha256=parent_hash, comparison_lock_sha256=comparison_hash, classifier_selected_epoch=200,
                      classifier=receipt, final_predictions=rows, final_test_evaluated=True, reserved_test_evaluated=False,
                      locator_retrained=False, locator_proposals_changed=False, regression_weight=0., severity_trained=False)
        folder = control / "stages" / stage["id"]
        for name, value in (("results.json", result), ("classifier/receipt.json", receipt), ("classifier/history.json", history),
                            ("classifier/initialization.json", initialization_by_stage[stage["id"]]),
                            ("final-test/predictions.json", dict(predictions=rows, checkpoint_sha256="b" * 64, source_lock_sha256=own_hash))):
            write_json(folder / name, value)
        for name, value in audits_by_stage[stage["id"]].items():
            write_json(folder / name, value)
    return control, lock, baseline, controls, manifest


class ClassificationReportTests(unittest.TestCase):
    # Purpose: Verify matched classification comparisons omit disabled severity outputs.
    # Inputs: Five-family tied baseline and perfect classification-only controls.
    # Outputs: AUROC 0.5 versus 1, 60 pairs, absent severity metrics, and null score predictions.
    def test_exact_classification_comparison_without_severity_metrics(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, baseline, controls, manifest = fixture(Path(tmp))
            result = analyze_records(baseline, controls, manifest)
            self.assertEqual(result["protocols"][BASELINE]["pooled_scenario_metrics"]["auroc"], .5)
            self.assertEqual(result["protocols"][CONTROL]["equal_family_metrics"]["auroc"]["mean"], 1.)
            self.assertEqual(result["paired_comparisons"][0]["matched_sample_seed_pairs"], 60)
            self.assertNotIn("severity_mae_percentage_points", json.dumps(result))
            self.assertTrue(all(r["score_prediction"] is None for r in result["records"]))
            self.assertFalse(result["severity_trained"])

    # Purpose: Verify unfinished controls retain complete baseline evidence.
    # Inputs: Five-family fixture with zero or one completed control.
    # Outputs: Pending/partial status, pending paired comparison, and all five baseline folds retained.
    def test_pending_and_partial_keep_complete_baseline_without_full_comparison(self):
        for completed in (0, 1):
            with self.subTest(completed=completed), tempfile.TemporaryDirectory() as tmp:
                root, lock, _, _, _ = fixture(Path(tmp), completed=completed)
                result = source_bundle(root, lock)["analysis"]
                self.assertEqual(result["status"], "partial" if completed else "pending")
                self.assertEqual(result["paired_comparisons"][0]["status"], "pending")
                self.assertEqual(result["protocols"][BASELINE]["completed_fold_runs"], 5)
                self.assertEqual(len(result["training_history"]), 1000 + completed * 200)

    # Purpose: Verify classification panels preserve identities, targets, and disabled-head state.
    # Inputs: Duplicate/missing records, NaN, reserved/train rows, wrong score labels, and score output.
    # Outputs: ValueError for every incompatible panel mutation.
    def test_exact_panels_targets_probability_and_untrained_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            _, _, baseline, controls, manifest = fixture(Path(tmp))
            failures = [controls + controls[:1], controls[1:],
                        [dict(controls[0], probability=float("nan"))] + controls[1:],
                        [dict(controls[0], case_id="reserved-infant")] + controls[1:],
                        [dict(controls[0], split="train")] + controls[1:],
                        [dict(controls[0], score_label=.99)] + controls[1:],
                        [dict(controls[0], score_prediction=.2)] + controls[1:]]
            for rows in failures:
                with self.subTest(first=rows[0]), self.assertRaises(ValueError):
                    analyze_records(baseline, rows, manifest)

    # Purpose: Verify fixed-budget histories exclude validation selection and severity training.
    # Inputs: 200-epoch control history mutated with validation, wrong updates, score MAE, or selected epoch.
    # Outputs: Valid history accepted and invalid selection/update/severity records rejected.
    def test_training_only_and_disabled_severity_gates(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lock, _, _, _ = fixture(Path(tmp), completed=1)
            stage = lock["stages"][0]; folder = root / "stages" / stage["id"]
            history = json.loads((folder / "classifier/history.json").read_text())
            receipt = json.loads((folder / "classifier/receipt.json").read_text())
            self.assertEqual(len(checked_history(history, receipt, stage)), 200)
            for field, value in (("validation", {"bce": .1}), ("selection_bce", .1), ("optimizer_updates", 25)):
                changed = copy.deepcopy(history); changed["epochs"][0][field] = value
                with self.subTest(field=field), self.assertRaises(ValueError):
                    checked_history(changed, receipt, stage)
            changed = copy.deepcopy(history); changed["epochs"][0]["training"]["score_mae"] = .2
            with self.assertRaisesRegex(ValueError, "severity"):
                checked_history(changed, receipt, stage)
            with self.assertRaises(ValueError):
                checked_history(history, dict(receipt, selected_epoch=100), stage)

    # Purpose: Verify classification control provenance matches its pinned comparison.
    # Inputs: Altered ancestor pins, lock, pair order, initialization, proposals, receipt, or predictions.
    # Outputs: ValueError for every provenance mutation and incomplete comparison stage.
    def test_pinned_ancestry_pair_order_initialization_proposal_and_result_identity(self):
        for mutation in ("parent_pin", "comparison_pin", "own_lock", "pair_order", "initialization", "proposal", "receipt", "prediction", "comparison_INCOMPLETE"):
            with self.subTest(mutation=mutation), tempfile.TemporaryDirectory() as tmp:
                root, lock, _, _, _ = fixture(Path(tmp), completed=1)
                stage = lock["stages"][0]; folder = root / "stages" / stage["id"]
                if mutation == "comparison_INCOMPLETE":
                    (Path(lock["comparison_directory"]) / "stages" / stage["id"] / "INCOMPLETE").touch()
                elif mutation in ("parent_pin", "comparison_pin"):
                    location = lock["parent_directory"] if mutation == "parent_pin" else lock["comparison_directory"]
                    path = Path(location) / "experiment-lock.json"
                    row = json.loads(path.read_text()); row["tampered"] = True; write_json(path, row)
                else:
                    name = {"own_lock": "results.json", "pair_order": "classifier/history.json", "initialization": "classifier/initialization.json",
                            "proposal": "proposal-audit.json", "receipt": "classifier/receipt.json", "prediction": "final-test/predictions.json"}[mutation]
                    path = folder / name; row = json.loads(path.read_text())
                    if mutation == "own_lock": row["source_lock_sha256"] = "c" * 64
                    elif mutation == "pair_order": row["epochs"][0]["training_pair_order"].reverse()
                    elif mutation == "initialization": row["predictions"][0]["probability"] = .7
                    elif mutation == "proposal": row["samples"][0]["origin_kji"] = [3, 2, 1]
                    elif mutation == "receipt": row["severity_trained"] = True
                    else: row["predictions"][0]["probability"] = .7
                    write_json(path, row)
                with self.assertRaises(ValueError):
                    source_bundle(root, lock)

    # Purpose: Verify failure markers block reporting while untouched stages stay pending.
    # Inputs: No completed controls, one hashed hardware-failure marker, then changed bytes.
    # Outputs: Blocked-by-failure analysis, four pending stages, and checksum mismatch rejected.
    def test_failed_receipt_checksum_and_pending_status(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lock, _, _, _ = fixture(Path(tmp), completed=0)
            path = root / "stages" / lock["stages"][0]["id"] / "failure.json"
            write_json(path, {"message": "Hardware failure"})
            analysis = source_bundle(root, lock)["analysis"]
            self.assertEqual(analysis["status"], "blocked_by_failed_stage")
            self.assertEqual(sum(r["status"] == "pending" for r in analysis["stage_status"]), 4)
            path.write_text('{"message":"changed"}')
            with self.assertRaisesRegex(ValueError, "checksum"):
                source_bundle(root, lock)

    # Purpose: Verify classification reports export auditable revisions without severity data.
    # Inputs: Complete control fixture rendered into two report revisions.
    # Outputs: Three figures per format, populated classification CSVs, verified hashes, and immutable prior revision.
    def test_exports_are_immutable_metadata_only_and_classification_only(self):
        with tempfile.TemporaryDirectory() as tmp:
            root, lock, _, _, _ = fixture(Path(tmp))
            result = write_report(root, lock); folder = Path(result["output_directory"])
            self.assertEqual((folder.name, result["figure_count"], result["records"]), ("review-r1", 3, 120))
            before = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()}
            proof = json.loads((folder / "provenance.json").read_text())
            for field in ("image_or_checkpoint_payloads_read", "reserved_test_predictions_read", "intermediate_outer_predictions_read", "severity_metrics_exported"):
                self.assertFalse(proof[field])
            for name, digest in proof["artifact_sha256"].items():
                self.assertEqual(before[name], digest)
            for suffix in ("pdf", "svg", "png"):
                self.assertEqual(len(list(folder.glob("classification*." + suffix))), 3)
            for name in ("family-metrics.csv", "training-history.csv", "predictions.csv"):
                with (folder / name).open() as handle:
                    reader = csv.DictReader(handle)
                    self.assertFalse(any("severity" in key or "score_" in key for key in reader.fieldnames))
                    self.assertTrue(list(reader))
            self.assertIn("Only the auxiliary regression weight changes", (folder / "report.html").read_text())
            self.assertEqual(Path(write_report(root, lock)["output_directory"]).name, "review-r2")
            self.assertEqual(before, {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in folder.iterdir()})


if __name__ == "__main__":
    unittest.main()
