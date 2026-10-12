"""Matched fine-tuning review: count semantics, correlation and pending arms."""
import copy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fakect_pretrained_comparison_report as report
from fakect_pretrained_comparison_report import ARMS, burden_bins, comparison_status, select_examples

try:
    import nibabel as nib
except ImportError:
    nib = None


def metric(value):
    data = dict(dice=value, recall=value, precision=value, surfaces=dict(surface_dice=value, hd95_mm=1 - value))
    return dict(status="completed", metrics=dict(whole_thoracic_core=data, lesion_local=data))


class PretrainedComparisonTests(unittest.TestCase):
    def record(self, sample, case, value, family="family", removed=20, added=0):
        return dict(sample_id=sample, case_id=case, anatomy_family=family, baseline=False,
                    removed_voxels=removed, added_voxels=added,
                    change_kind="mixed" if removed and added else "removal_only",
                    predictions={arm: metric(value) for arm in ARMS})

    # Purpose: Verify burden summaries balance cases rather than sibling counts.
    # Inputs: Nine zero-score siblings in case A and one perfect sibling in case B.
    # Outputs: Ten siblings across two cases yield mean 0.5.
    def test_case_balancing_does_not_let_sibling_count_dominate(self):
        rows = [self.record("A:" + str(i), "A", 0.) for i in range(9)] + [self.record("B:0", "B", 1.)]
        result = burden_bins(rows, "removed_voxels")[0]
        self.assertEqual(result["siblings"], 10)
        self.assertEqual(result["cases"], 2)
        self.assertEqual(result["means"]["siblings"], .5)

    # Purpose: Verify burden summaries give equal weight to families.
    # Inputs: Two zero-score cases in one family and one perfect case in another.
    # Outputs: Mean pretrained score is 0.5.
    def test_family_balancing_is_explicit_and_independent_of_case_count(self):
        rows = [self.record("A", "A", 0., "same"), self.record("B", "B", 0., "same"), self.record("C", "C", 1., "other")]
        self.assertEqual(burden_bins(rows, "removed_voxels")[0]["means"]["pretrained"], .5)

    # Purpose: Preserve pending comparison arms without inventing zero scores.
    # Inputs: One completed sample with the siblings arm pending.
    # Outputs: Comparison is partial with zero paired samples and no burden bin.
    def test_pending_arm_not_zero_score_or_completed_comparison(self):
        row = self.record("A", "A", .9)
        row["predictions"]["siblings"] = dict(status="pending")
        state = comparison_status([row])
        self.assertEqual(state["status"], "partial")
        self.assertEqual(state["paired_samples"], 0)
        self.assertEqual(burden_bins([row], "removed_voxels"), [])

    # Purpose: Verify mixed changes contribute to both burden graphs and exclude controls.
    # Inputs: One mixed sibling with 100 removals/25 additions and a baseline control.
    # Outputs: Both graphs count one sibling and one mixed sibling.
    def test_mixed_native_counts_appear_in_both_graphs_and_controls_excluded(self):
        row = self.record("A", "A", .8, removed=100, added=25)
        control = self.record("A:baseline", "A", .9, removed=0); control["baseline"] = True
        for quantity in ("removed_voxels", "added_voxels"):
            result = burden_bins([row, control], quantity)[0]
            self.assertEqual(result["siblings"], 1)
            self.assertEqual(result["mixed_siblings"], 1)

    # Purpose: Verify prespecified examples ignore prediction performance.
    # Inputs: Baseline plus removal/addition burden panels with subsequently replaced scores.
    # Outputs: Example IDs stay identical, include the baseline, and total three.
    def test_prespecified_examples_do_not_depend_on_prediction_scores(self):
        rows = [dict(self.record("A:original", "A", .8, removed=0), baseline=True)]
        rows += [self.record("B:" + str(i), "B", i / 10., removed=i * 100) for i in range(1, 5)]
        rows += [self.record("C:" + str(i), "C", i / 10., removed=0, added=i * 100) for i in range(1, 5)]
        first = select_examples(rows)
        other = copy.deepcopy(rows)
        for row in other: row["predictions"] = {arm: metric(1.) for arm in ARMS}
        self.assertEqual(first, select_examples(other))
        self.assertEqual(len(first), 3)
        self.assertEqual(first[0], "A:original")


@unittest.skipUnless(nib is not None, "Requires production benchmark nibabel environment")
class PretrainedComparisonNiftiTests(unittest.TestCase):
    def fixture(self, root):
        shape = (25, 25, 25)
        before = np.zeros(shape, bool); before[5:20, 9:16, 9:16] = True
        after = before.copy(); after[12, 8, 10] = True; after[10:12, 9:11, 9:11] = False
        changed = before ^ after
        row = dict(sample_id="A:sibling", case_id="A", anatomy_family="family", baseline=False,
                   added_voxels=1, removed_voxels=8, export=dict(origin_ijk=[25, 40, 100]), predictions={})
        for name, array in (("input", after.astype(np.float32) * 125), ("truth", after),
                            ("domain", np.ones(shape, bool)), ("changed", changed)):
            path = root / (name + ".nii.gz")
            image = nib.Nifti1Image(array.astype(np.float32 if name == "input" else np.uint8).transpose(2, 1, 0),
                                   np.diag([-1., 1., 1., 1.]))
            image.header.set_xyzt_units("mm"); nib.save(image, path)
            row[name + "_nifti"] = str(path); row[name + "_nifti_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        for arm, array in (("pretrained", before), ("baseline_only", before), ("siblings", after)):
            path = root / (arm + ".nii.gz")
            nib.save(nib.Nifti1Image(array.astype(np.uint8).transpose(2, 1, 0), np.diag([-1., 1., 1., 1.])), path)
            row["predictions"][arm] = dict(status="completed", prediction_nifti=str(path),
                prediction_nifti_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        return row

    # Purpose: Verify native comparison report uses identical current CT across arms.
    # Inputs: Hashed native NIfTI fixture with one addition/eight removals and three prediction arms.
    # Outputs: Complete report preserves exact counts/origin, exports figures/viewer, and pins output hashes.
    def test_complete_native_report_same_ct_columns_and_exact_counts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = self.fixture(root)
            checked = []
            original_save = report._save
            def inspect_figure(fig, out, stem):
                if stem.startswith("example"):
                    self.assertEqual(len(fig.axes), 12)
                    for orient in range(3):
                        planes = [np.asarray(fig.axes[orient * 4 + col].images[0].get_array()) for col in range(4)]
                        for plane in planes[1:]: np.testing.assert_array_equal(plane, planes[0])
                    checked.append(stem)
                original_save(fig, out, stem)
            with patch.object(report, "_save", side_effect=inspect_figure):
                result = report.write_report(dict(samples=[row], example_sample_ids=[row["sample_id"]]), root / "review")
            self.assertEqual(result["status"], "complete")
            self.assertEqual(checked, ["example-00"])
            data = json.loads((root / "review/analysis.json").read_text())
            sample = data["samples"][0]
            self.assertEqual(sample["added_voxels"], 1); self.assertEqual(sample["removed_voxels"], 8)
            self.assertEqual(sample["change_kind"], "mixed")
            self.assertEqual(data["examples"][0]["source_origin_ijk"], [25, 40, 100])
            self.assertTrue(all(v["identical_current_ct_across_columns"] for v in data["examples"][0]["orientations"]))
            self.assertEqual(sample["predictions"]["siblings"]["metrics"]["whole_thoracic_core"]["dice"], 1.)
            viewer = json.loads((root / "review/report-3d/manifest.json").read_text())
            self.assertEqual(viewer["opacity_presets"], [.15, .45, .8])
            provenance = json.loads((root / "review/provenance.json").read_text())
            for name, digest in provenance["output_sha256"].items():
                self.assertEqual(digest, hashlib.sha256((root / "review" / name).read_bytes()).hexdigest())
            for ext in ("png", "pdf", "svg"):
                self.assertTrue((root / "review" / ("example-00." + ext)).is_file())
            with self.assertRaises(FileExistsError):
                report.write_report(dict(samples=[row]), root / "review")

    # Purpose: Reject false native counts and mismatched prediction affines before output.
    # Inputs: Added count changed to 99, then prediction affine shifted by 1 mm.
    # Outputs: Both report calls raise ValueError and create no output directories.
    def test_wrong_affine_and_false_core_counts_fail_without_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = self.fixture(root)
            bad = copy.deepcopy(row); bad["added_voxels"] = 99
            with self.assertRaises(ValueError): report.write_report(dict(samples=[bad]), root / "counts-review")
            path = Path(row["predictions"]["siblings"]["prediction_nifti"])
            image = nib.load(path); affine = image.affine.copy(); affine[0, 3] += 1.
            nib.save(nib.Nifti1Image(np.asanyarray(image.dataobj), affine), path)
            row["predictions"]["siblings"].pop("prediction_nifti_sha256")
            with self.assertRaises(ValueError): report.write_report(dict(samples=[row]), root / "affine-review")
            self.assertFalse((root / "counts-review").exists()); self.assertFalse((root / "affine-review").exists())

    # Purpose: Verify partial native reports retain missing arms as pending.
    # Inputs: Native sample with baseline-only and siblings predictions pending.
    # Outputs: Partial report has no paired metrics or burden curves and clearly marks pending arms.
    def test_partial_native_report_keeps_missing_arms_pending(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = self.fixture(root)
            row["predictions"]["baseline_only"] = dict(status="pending")
            row["predictions"]["siblings"] = dict(status="pending")
            result = report.write_report(dict(samples=[row]), root / "partial")
            self.assertEqual(result["status"], "partial")
            self.assertEqual(result["paired_samples"], 0)
            data = json.loads((root / "partial/analysis.json").read_text())
            self.assertEqual(data["arms"]["siblings"]["pending"], 1)
            self.assertEqual(data["burden_profiles"]["removed_voxels"]["lesion_local/dice"], [])
            self.assertNotIn("metrics", data["samples"][0]["predictions"]["siblings"])
            self.assertIn("PARTIAL", (root / "partial/report.html").read_text())


if __name__ == "__main__":
    unittest.main()
