"""Test OOF identity, leakage checks, equal-family aggregation and paired panels."""
import copy
import csv
import hashlib
import json
import math
from pathlib import Path
import tempfile
import unittest

from fakect_coa_rotation_report import (aggregate_records, paired_comparison,
                                      rotation_manifest, validate_manifest, write_rotation_report)


def fixture(protocols=("family-control",), seeds=(42,)):
    cases = ["A", "B", "C"]
    families = {case: "family-" + case for case in cases}
    folds, records = [], []
    for protocol in protocols:
        for seed in seeds:
            for case in cases:
                sample_ids = [case + ":" + str(i) for i in range(4)]
                folds.append(dict(protocol=protocol, fold_id="outer-" + case, seed=seed,
                                  outer_case=None, outer_family=families[case], grouping="family",
                                  evaluation_case_ids=[case], evaluation_sample_ids=sample_ids,
                                  training_case_ids=[c for c in cases if c != case], validation_case_ids=[]))
                for i, (label, score, probability) in enumerate(zip((0, 0, 1, 1), (0., .1, .3, .6), (.1, .2, .8, .9))):
                    records.append(dict(protocol=protocol, fold_id="outer-" + case, seed=seed,
                                        outer_case=None, outer_family=families[case],
                                        sample_id=sample_ids[i], case_id=case, label=label,
                                        probability=probability, score_label=score, score_prediction=score))
    return records, dict(folds=folds, family_by_case=families,
                         reserved_case_ids=["reserved-infant"], class_threshold=.5)


class RotationAggregationTests(unittest.TestCase):
    def test_perfect_exact_metrics_and_equal_family_means(self):
        rows, manifest = fixture(seeds=(42, 43))
        result = aggregate_records(rows, manifest)
        self.assertEqual(result["status"], "complete")
        self.assertEqual(len(result["per_case"]), 6)
        self.assertEqual(len(result["per_family"]), 6)
        summary = result["protocols"]["family-control"]
        self.assertEqual(summary["equal_family_metrics"]["auroc"]["mean"], 1.)
        self.assertEqual(summary["equal_family_metrics"]["auroc"]["units"], 3)
        self.assertEqual(summary["equal_family_metrics"]["sensitivity"]["mean"], 1.)
        self.assertEqual(summary["mean_roc"]["families"], 3)
        self.assertEqual(summary["mean_roc"]["confidence_interval"], None)
        expected_bce = -(math.log(.9) + math.log(.8) + math.log(.8) + math.log(.9)) / 4
        self.assertAlmostEqual(summary["equal_family_metrics"]["bce"]["mean"], expected_bce)

    def test_tied_scores_have_exact_half_credit(self):
        rows, manifest = fixture()
        for row in rows:
            row["probability"] = .5
        result = aggregate_records(rows, manifest)
        metrics = result["per_case"][0]
        self.assertEqual(metrics["auroc"], .5)
        self.assertEqual(metrics["sensitivity"], 1.)
        self.assertEqual(metrics["specificity"], 0.)
        self.assertEqual(metrics["balanced_accuracy"], .5)

    def test_correlated_seed_repeats_do_not_increase_family_units(self):
        rows, manifest = fixture(seeds=(42, 43))
        for row in rows:
            if row["case_id"] == "C":
                row["probability"] = 1 - row["probability"]
        result = aggregate_records(rows, manifest)
        summary = result["protocols"]["family-control"]["equal_family_metrics"]["auroc"]
        self.assertAlmostEqual(summary["mean"], 2 / 3)
        self.assertEqual(summary["units"], 3)
        self.assertIsNone(summary["confidence_interval"])

    def test_prediction_corruption_and_incomplete_panels_are_rejected(self):
        rows, manifest = fixture()
        failures = [rows + rows[:1], rows[1:], [dict(rows[0], probability=float("nan"))] + rows[1:],
                    [dict(rows[0], case_id="B")] + rows[1:],
                    [dict(rows[0], outer_family="family-B")] + rows[1:],
                    [dict(rows[0], split="validation")] + rows[1:],
                    [dict(rows[0], label=2)] + rows[1:]]
        for corrupted in failures:
            with self.subTest(first=corrupted[0]), self.assertRaises(ValueError):
                aggregate_records(corrupted, manifest)

    def test_train_case_family_and_reserved_leakage_are_rejected(self):
        _, manifest = fixture()
        changed = copy.deepcopy(manifest)
        changed["folds"][0]["training_case_ids"].append("A")
        with self.assertRaises(ValueError):
            validate_manifest(changed)
        changed = copy.deepcopy(manifest)
        changed["reserved_case_ids"] = ["A"]
        with self.assertRaises(ValueError):
            validate_manifest(changed)
        changed = copy.deepcopy(manifest)
        changed["family_by_case"]["B"] = "family-A"
        with self.assertRaises(ValueError):
            validate_manifest(changed)

    def test_whole_pending_folds_allowed_but_validation_receipts_excluded(self):
        rows, manifest = fixture()
        receipt = dict(protocol="family-control", fold_id="outer-A", seed=42,
                       validation_case_ids=[], selected_epoch=37,
                       predictions=[dict(probability=.99, label=1)])
        result = aggregate_records(rows[:4], manifest, validation_receipts=[receipt])
        self.assertEqual(result["status"], "partial")
        self.assertEqual(result["protocols"]["family-control"]["pooled_scenario_metrics"]["samples"], 4)
        self.assertEqual(result["receipt_performance_use"], "none")
        bad = dict(receipt, validation_case_ids=["A"])
        with self.assertRaises(ValueError):
            aggregate_records(rows, manifest, validation_receipts=[bad])

    def test_paired_controls_require_identical_panels_and_targets(self):
        rows, manifest = fixture(protocols=("left", "right"), seeds=(42, 43))
        for row in rows:
            if row["protocol"] == "right":
                row["probability"] = [.2, .8, .4, .9][int(row["sample_id"].split(":")[1])]
        result = aggregate_records(rows, manifest, comparisons=[dict(left_protocol="left", right_protocol="right")])
        comparison = result["paired_comparisons"][0]
        self.assertAlmostEqual(comparison["equal_family_delta"]["auroc"]["mean"], -.25)
        self.assertEqual(comparison["matched_sample_seed_pairs"], 24)
        mismatch = copy.deepcopy(result)
        mismatch["records"] = [r for r in mismatch["records"] if not (r["protocol"] == "right" and r["case_id"] == "A")]
        with self.assertRaises(ValueError):
            paired_comparison(mismatch, "left", "right")
        changed = copy.deepcopy(rows)
        next(r for r in changed if r["protocol"] == "right")["score_label"] = .05
        with self.assertRaises(ValueError):
            aggregate_records(changed, manifest)

    def test_pending_report_contains_no_invented_performance_figures(self):
        _, manifest = fixture()
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory) / "pending"
            result = write_rotation_report(out, [], manifest)
            self.assertEqual(result["status"], "pending")
            self.assertEqual(result["figure_count"], 0)
            self.assertIn("No rotating evaluation results", (out / "report.html").read_text())
            self.assertEqual(list(out.glob("*.png")), [])
            with self.assertRaises(FileExistsError):
                write_rotation_report(out, [], manifest)

    def test_locator_failures_remain_in_the_exact_performance_panel(self):
        rows, manifest = fixture()
        for row in rows:
            row.update(locator_coverage_passed=True, pipeline_failure=False)
        rows[0].update(locator_coverage_passed=False, pipeline_failure=True)
        result = aggregate_records(rows, manifest)
        pooled = result["protocols"]["family-control"]["pooled_scenario_metrics"]
        self.assertEqual(pooled["samples"], 12)
        self.assertEqual(pooled["pipeline_failure_samples"], 1)
        self.assertAlmostEqual(pooled["locator_coverage_pass_rate"], 11 / 12)
        self.assertEqual(pooled["auroc"], 1.)
        rows[0]["pipeline_failure"] = False
        with self.assertRaises(ValueError):
            aggregate_records(rows, manifest)

    def test_design_adapter_excludes_inner_selection_and_locator_baselines(self):
        import fakect_coa_rotation_design as design
        from test_coa_rotation_design import fixture as design_fixture
        _, _, _, inventory, panel = design_fixture()
        planned = design.build_rotation_design(inventory, panel)
        manifest = rotation_manifest(planned)
        self.assertEqual(len(manifest["folds"]), 25)
        self.assertEqual(len(manifest["stage_catalog"]), 45)
        self.assertEqual(len(manifest["sample_targets"]), 60)
        self.assertEqual(sum(f["grouping"] == "family" for f in manifest["folds"]), 5)
        self.assertEqual(sum(f["grouping"] == "case" for f in manifest["folds"]), 20)
        result = aggregate_records([], planned)
        self.assertEqual(result["status"], "pending")
        self.assertEqual(result["protocols"]["family_holdout"]["planned_fold_runs"], 5)
        inner = [s for s in planned["stages"] if s["stage"] == "inner_selection"]
        receipts = [dict(stage_id=s["id"], status="completed", validation_case_ids=s["validation_case_ids"])
                    for s in inner[:2]]
        result = aggregate_records([], planned, validation_receipts=receipts)
        self.assertEqual(len(result["validation_receipts"]), 2)
        self.assertEqual(len(result["records"]), 0)
        bad = dict(receipts[0], validation_case_ids=inner[0]["final_test_case_ids"])
        with self.assertRaises(ValueError):
            aggregate_records([], planned, validation_receipts=[bad])
        failure = dict(stage_id=inner[0]["id"], status="failed", error="Fold-local locator failed")
        result = aggregate_records([], planned, validation_receipts=[failure])
        self.assertEqual(result["status"], "blocked_by_failed_stage")
        self.assertEqual(len(result["failed_stages"]), 1)

    def test_known_sample_cannot_be_assigned_to_another_case_in_same_family(self):
        import fakect_coa_rotation_design as design
        from test_coa_rotation_design import fixture as design_fixture
        _, _, _, inventory, panel = design_fixture()
        planned = design.build_rotation_design(inventory, panel)
        manifest = rotation_manifest(planned)
        fold = next(f for f in manifest["folds"] if f["protocol"] == "family_holdout")
        rows = []
        for identity in fold["evaluation_sample_ids"]:
            target = manifest["sample_targets"][identity]
            rows.append({**target, "protocol": fold["protocol"], "fold_id": fold["fold_id"],
                         "outer_case": None, "outer_family": fold["outer_family"], "seed": fold["seed"],
                         "sample_id": identity, "probability": .8 if target["label"] else .2,
                         "split": "final_test"})
        aggregate_records(rows, manifest)
        a, b = fold["evaluation_case_ids"]
        next(r for r in rows if r["case_id"] == a)["case_id"] = b
        with self.assertRaisesRegex(ValueError, "frozen sample target"):
            aggregate_records(rows, manifest)

    def test_measured_export_retains_failure_rows_and_exact_source_fingerprints(self):
        rows, manifest = fixture()
        rows[0].update(locator_coverage_passed=False, pipeline_failure=True)
        with tempfile.TemporaryDirectory(prefix="fakect-rotation-fixture-") as directory:
            out = Path(directory) / "fixture-report"
            summary = write_rotation_report(out, rows, manifest, provenance={"data_scope": "unit-test fixture only"})
            self.assertEqual(summary["figure_count"], 3)
            self.assertEqual(summary["records"], 12)
            self.assertEqual(len(list(out.glob("*.pdf"))), 3)
            self.assertEqual(len(list(out.glob("*.svg"))), 3)
            self.assertEqual(len(list(out.glob("*.png"))), 3)
            with (out / "out-of-fold-predictions.csv").open() as handle:
                exported = list(csv.DictReader(handle))
            self.assertEqual(len(exported), 12)
            self.assertEqual(sum(r["pipeline_failure"] == "True" for r in exported), 1)
            proof = json.loads((out / "provenance.json").read_text())
            self.assertFalse(proof["reserved_test_predictions_read"])
            self.assertFalse(proof["validation_receipts_used_for_performance"])
            for name, expected in proof["artifact_sha256"].items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), expected)


if __name__ == "__main__":
    unittest.main()
