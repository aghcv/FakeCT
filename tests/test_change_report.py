"""Net-change scores exclude padding and report empty-control false changes."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
import fakect_change_report as report


class ChangeMetricsTests(unittest.TestCase):
    # Purpose: Verify change counts and physical volume exclude invalid padding.
    # Inputs: 2-cube truth and prediction with one invalid false addition and spacing 0.5, 1, 2.
    # Outputs: Seven valid voxels, correct class counts, added Dice 2/3 and removed false-positive volume 1.
    def test_counts_and_physical_volume_exclude_padding(self):
        labels = np.zeros((2, 2, 2), np.uint8)
        labels.flat[0] = 1; labels.flat[1] = 2
        predicted = labels.copy(); predicted.flat[1] = 1
        predicted.flat[2] = 2; predicted.flat[7] = 1
        valid = np.ones_like(labels); valid.flat[7] = 0
        metrics = report.change_metrics(labels, predicted, valid, [.5, 1., 2.])
        self.assertEqual(metrics["valid_voxels"], 7)
        self.assertEqual(metrics["classes"]["added"]["tp"], 1)
        self.assertEqual(metrics["classes"]["added"]["fp"], 1)
        self.assertAlmostEqual(metrics["classes"]["added"]["dice"], 2 / 3)
        self.assertEqual(metrics["classes"]["removed"]["fn"], 1)
        self.assertEqual(metrics["classes"]["removed"]["false_positive_mm3"], 1.)
        self.assertFalse(metrics["no_op_control"])

    # Purpose: Verify absent target classes report undefined overlap and control errors.
    # Inputs: Unchanged 2-cube target with one predicted removal and spacing 2, 3, 4.
    # Outputs: No added/removed Dice; control flagged with one false voxel and 24 mm3 false change.
    def test_absent_classes_have_no_overlap_score(self):
        labels = np.zeros((2, 2, 2), np.uint8)
        predicted = labels.copy(); predicted.flat[0] = 2
        metrics = report.change_metrics(labels, predicted, np.ones_like(labels), [2., 3., 4.])
        self.assertIsNone(metrics["classes"]["removed"]["dice"])
        self.assertIsNone(metrics["classes"]["added"]["dice"])
        self.assertTrue(metrics["no_op_control"])
        self.assertEqual(metrics["no_op_false_positive_voxels"], 1)
        self.assertEqual(metrics["no_op_false_positive_mm3"], 24.)

    # Purpose: Verify entirely invalid crops cannot count as successful controls.
    # Inputs: Zero truth, prediction and validity arrays.
    # Outputs: Control flag false and unchanged Dice undefined.
    def test_all_invalid_crop_does_not_become_a_successful_control(self):
        zeros = np.zeros((2, 2, 2), np.uint8)
        metrics = report.change_metrics(zeros, zeros, zeros, [1., 1., 1.])
        self.assertFalse(metrics["no_op_control"])
        self.assertIsNone(metrics["classes"]["unchanged"]["dice"])

    # Purpose: Reject malformed class grids and physical spacing.
    # Inputs: Zero or NaN spacing, wrong spacing length, class 3 and mismatched prediction shape.
    # Outputs: ValueError rejects every grid or spacing violation.
    def test_invalid_grids_and_spacing_rejected(self):
        zeros = np.zeros((2, 2, 2), np.uint8)
        for spacing in ([0, 1, 1], [1, 1], [float("nan"), 1, 1]):
            with self.assertRaises(ValueError):
                report.change_metrics(zeros, zeros, zeros, spacing)
        with self.assertRaises(ValueError):
            report.change_metrics(zeros + 3, zeros, zeros, [1, 1, 1])
        with self.assertRaises(ValueError):
            report.change_metrics(zeros, zeros[:, :, :1], zeros, [1, 1, 1])

    # Purpose: Verify empty controls cannot inflate nonempty overlap summaries.
    # Inputs: Perfect unchanged control and completely missed added voxel in another family.
    # Outputs: Added family Dice zero, one empty-target record and one no-op control.
    def test_empty_controls_do_not_inflate_nonempty_overlap(self):
        zeros = np.zeros((2, 2, 2), np.uint8)
        changed = zeros.copy(); changed.flat[0] = 1
        rows = [dict(sample_id="control", case_id="a", anatomy_family="family-a", baseline=True,
                     metrics=report.change_metrics(zeros, zeros, np.ones_like(zeros), [1, 1, 1])),
                dict(sample_id="changed", case_id="b", anatomy_family="family-b", baseline=False,
                     metrics=report.change_metrics(changed, zeros, np.ones_like(zeros), [1, 1, 1]))]
        summary = report.summarize(rows)
        self.assertEqual(summary["classes"]["added"]["family_macro"]["dice"], 0.)
        self.assertEqual(summary["classes"]["added"]["empty_target_records"], 1)
        self.assertEqual(summary["no_op_controls"]["records"], 1)

    # Purpose: Verify multiple focus crops count as one scenario.
    # Inputs: Two proposal indices for one sample, followed by a duplicated proposal row.
    # Outputs: One scenario, two proposals and one case; duplicate proposal rejected.
    def test_focus_crops_are_not_counted_as_distinct_scenarios(self):
        zeros = np.zeros((2, 2, 2), np.uint8)
        metric = report.change_metrics(zeros, zeros, np.ones_like(zeros), [1, 1, 1])
        rows = [dict(sample_id="a", proposal_index=i, case_id="case-a", anatomy_family="f", metrics=metric) for i in (0, 1)]
        summary = report.summarize(rows)
        self.assertEqual(summary["scenario_count"], 1)
        self.assertEqual(summary["proposal_count"], 2)
        self.assertEqual(summary["case_count"], 1)
        with self.assertRaisesRegex(ValueError, "unique"):
            report.summarize([rows[0], rows[0]])

    # Purpose: Verify family summaries weight cases equally despite sample imbalance.
    # Inputs: Five perfect samples from case a and one missed sample from case b.
    # Outputs: Sample Dice 5/6 and equally weighted family Dice 0.5.
    def test_family_mean_gives_cases_equal_weight(self):
        labels = np.zeros((2, 2, 2), np.uint8); labels.flat[0] = 1
        valid = np.ones_like(labels)
        perfect = report.change_metrics(labels, labels, valid, [1, 1, 1])
        missed = report.change_metrics(labels, np.zeros_like(labels), valid, [1, 1, 1])
        rows = [dict(sample_id=f"a:{i}", case_id="a", anatomy_family="family", metrics=perfect) for i in range(5)]
        rows.append(dict(sample_id="b:0", case_id="b", anatomy_family="family", metrics=missed))
        summary = report.summarize(rows)["classes"]["added"]
        self.assertAlmostEqual(summary["nonempty_sample_macro"]["dice"], 5 / 6)
        self.assertEqual(summary["family_macro"]["dice"], .5)


class ChangeReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory(); self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        self.zeros = np.zeros((3, 3, 3), np.uint8)
        self.after = self.zeros.copy(); self.after[1, 1, 1] = 1
        self.metrics = report.change_metrics(self.after, self.after, np.ones_like(self.zeros), [1., 1., 1.])
        self.row = dict(sample_id="<sample>", case_id="case", anatomy_family="f", proposal_index=0,
                        baseline=False, metrics=self.metrics)
        self.results = dict(status="completed", records=[self.row], classifier_started=False,
                            gate={"passed": True}, warnings=["<unsafe>"])
        self._write()

    def _write(self):
        (self.root / "results.json").write_text(json.dumps(self.results))

    # Purpose: Verify report text escaping and immutable results with learning plots.
    # Inputs: Completed single-sample results, unsafe text and two-epoch history.
    # Outputs: Escaped identifiers and warnings, learning plot file and unchanged JSON artifacts.
    def test_report_escapes_text_preserves_results_and_plots_history(self):
        history = dict(epochs=[dict(epoch=1, loss=.8, dice_loss=.6), dict(epoch=2, loss=.5, dice_loss=.3)])
        (self.root / "history.json").write_text(json.dumps(history))
        original = (self.root / "results.json").read_bytes()
        saved = report.write_report(self.root)
        document = Path(saved["report"]).read_text()
        self.assertIn("&lt;sample&gt;", document)
        self.assertIn("&lt;unsafe&gt;", document)
        self.assertIn("1 scenarios · 1 focus crops", document)
        self.assertIn("False", document)
        self.assertTrue(Path(saved["learning_plot"]).is_file())
        self.assertEqual(original, (self.root / "results.json").read_bytes())
        self.assertEqual(history, json.loads((self.root / "history.json").read_text()))

    # Purpose: Verify saved scenes expose true and predicted net-change layers.
    # Inputs: 3-cube NPZ containing one true and predicted addition.
    # Outputs: One added voxel in both layers, no removals and six scene layers.
    def test_exact_scene_reuses_true_and_predicted_layers(self):
        path = self.root / "prediction.npz"
        np.savez_compressed(path, before=self.zeros, after=self.after, labels=self.after,
            predicted=self.after, valid=np.ones_like(self.zeros), spacing_ijk_mm=[1., 1., 1.],
            sample_id=np.asarray(self.row["sample_id"]))
        scene = report._scene(path, self.row)
        self.assertEqual(scene["surfaces"]["added"]["voxels"], 1)
        self.assertEqual(scene["surfaces"]["predicted_added"]["voxels"], 1)
        self.assertEqual(scene["surfaces"]["removed"]["voxels"], 0)
        self.assertEqual(len(scene["modes"][0]["layers"]), 6)
        self.assertIn("focus 0", scene["sample_id"])

    # Purpose: Reject escaped prediction paths and incorrect payload checksums.
    # Inputs: Relative traversal, absolute path and local payload with wrong then correct SHA-256.
    # Outputs: Unsafe paths and checksum rejected; verified local prediction path returned.
    def test_saved_prediction_must_remain_inside_run_and_match_hash(self):
        for value in ("../outside.npz", "/tmp/prediction.npz"):
            with self.assertRaises(ValueError):
                report._prediction_path(self.root, dict(self.row, prediction_npz=value))
        path = self.root / "prediction.npz"; path.write_bytes(b"payload")
        with self.assertRaisesRegex(ValueError, "checksum"):
            report._prediction_path(self.root, dict(self.row, prediction_npz=path.name, prediction_sha256="0" * 64))
        sha = hashlib.sha256(path.read_bytes()).hexdigest()
        self.assertEqual(report._prediction_path(self.root, dict(self.row, prediction_npz=path.name, prediction_sha256=sha)), path)

    # Purpose: Reject scene truth inconsistent with before/after masks.
    # Inputs: NPZ with unchanged before/after masks but an added truth label.
    # Outputs: ValueError reports disagreeing net truth.
    def test_scene_rejects_inconsistent_net_truth(self):
        path = self.root / "prediction.npz"
        np.savez_compressed(path, before=self.zeros, after=self.zeros, labels=self.after,
            predicted=self.after, valid=np.ones_like(self.zeros), spacing_ijk_mm=[1., 1., 1.])
        with self.assertRaisesRegex(ValueError, "disagree"):
            report._scene(path, self.row)

    # Purpose: Verify complete reports publish a local viewer without modifying predictions.
    # Inputs: Hashed prediction NPZ referenced by completed results.
    # Outputs: One-scene viewer, expected opacity presets, report link and identical prediction checksum.
    def test_complete_report_writes_local_viewer_without_changing_prediction(self):
        path = self.root / "prediction.npz"
        np.savez_compressed(path, before=self.zeros, after=self.after, labels=self.after,
            predicted=self.after, valid=np.ones_like(self.zeros), spacing_ijk_mm=[1., 1., 1.],
            sample_id=np.asarray(self.row["sample_id"]))
        self.row["prediction_npz"] = path.name
        self.row["prediction_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        self._write()
        result = report.write_report(self.root)
        self.assertEqual(result["interactive_3d"]["scenarios"], 1)
        manifest = json.loads((self.root / "report-3d" / "manifest.json").read_text())
        self.assertEqual(manifest["opacity_presets"], [.15, .45, .8])
        self.assertIn("report-3d/index.html", Path(result["report"]).read_text())
        self.assertEqual(self.row["prediction_sha256"], hashlib.sha256(path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
