"""PCA report retains all siblings and separates CT from mask annotations."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fakect_ct_delta_report import cluster_summary, write_report


class CTDeltaReportTests(unittest.TestCase):
    def fixture(self):
        rows = [dict(sample_id="A:zero", case_id="A", ct_relative_l1=0., ct_positive_l1=0., ct_negative_l1=0., added_voxels=0),
                dict(sample_id="A:one", case_id="A", ct_relative_l1=.02, ct_positive_l1=.02, ct_negative_l1=0., added_voxels=15),
                dict(sample_id="B:one", case_id="B", ct_relative_l1=.03, ct_positive_l1=0., ct_negative_l1=.03, removed_voxels=80)]
        return dict(pca={"explained_variance_ratio": [.65, .2]},
                    clustering={"candidate_best_k": 2, "suggested_k": 1, "evidence_passed": False,
                                "fallback_reasons": ["candidate_contains_small_cluster"],
                                "candidates": [dict(k=2, silhouette=.4)]}), rows, np.array([[0., 0.], [1., 1.], [-1., 1.]]), np.array([0, 0, 1])

    # Purpose: Verify offline PCA reports retain all siblings and export verified artifacts.
    # Inputs: Three sibling CT-delta records, PCA coordinates, fallback clustering and optional baseline.
    # Outputs: Offline HTML, PNG/PDF/SVG/CSV exports, matching output hashes and overwrite rejection.
    def test_offline_report_all_siblings_exports_and_hashes(self):
        analysis, rows, scores, labels = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            destination = Path(tmp) / "review"
            result = write_report(destination, analysis, rows, scores, labels,
                                  baseline_records=[dict(case_id="A")], baseline_scores=np.zeros((1, 2)))
            self.assertEqual(result["n_siblings"], 3)
            text = (destination / "report.html").read_text()
            self.assertIn("A:zero", text)
            self.assertIn("plotly.js", text)
            self.assertNotIn('src="https://cdn.plot.ly', text)
            self.assertIn("not validated clinical severity", text)
            self.assertIn("diagnostic suggested K=1", text)
            self.assertIn("candidate_contains_small_cluster", text)
            provenance = json.loads((destination / "report-provenance.json").read_text())
            self.assertFalse(provenance["anatomical_masks_used_as_features"])
            for name, digest in provenance["outputs_sha256"].items():
                self.assertEqual(hashlib.sha256((destination / name).read_bytes()).hexdigest(), digest)
            for ext in ("png", "pdf", "svg"):
                self.assertTrue((destination / ("pca-clusters." + ext)).is_file())
            csvtext = (destination / "siblings.csv").read_text()
            self.assertIn("A:zero", csvtext)
            self.assertIn("added_voxels", csvtext)
            with self.assertRaises(FileExistsError):
                write_report(destination, analysis, rows, scores, labels)

    # Purpose: Verify CT cluster summaries ignore anatomical mask counts.
    # Inputs: CT energy records with one added_voxels count changed to 100000.
    # Outputs: Identical summaries, cluster-one labeling and CT median 0.01.
    def test_ct_summary_does_not_use_mask_counts(self):
        _, rows, _, labels = self.fixture()
        first = cluster_summary(rows, labels)
        rows[1]["added_voxels"] = 100000
        self.assertEqual(first, cluster_summary(rows, labels))
        self.assertEqual(first[0]["ct_relative_l1"]["median"], .01)
        self.assertEqual(first[0]["cluster"], 1)
        self.assertEqual(first[0]["cluster_index"], 0)

    # Purpose: Verify native CT extractor energy fields are accepted.
    # Inputs: One record with normalized positive energy 0.2 and negative energy 0.3.
    # Outputs: Relative CT energy median 0.5 and positive median 0.2.
    def test_native_extractor_energy_names_supported(self):
        rows = [dict(sample_id="A:one", case_id="A", normalized_positive_ct_energy=.2,
                     normalized_negative_ct_energy=.3)]
        summary = cluster_summary(rows, [0])[0]
        self.assertAlmostEqual(summary["ct_relative_l1"]["median"], .5)
        self.assertAlmostEqual(summary["ct_positive_l1"]["median"], .2)

    # Purpose: Reject PCA coordinate and sibling identity mismatches.
    # Inputs: Wrong coordinate count, float cluster labels, NaN coordinates or duplicate sample IDs.
    # Outputs: ValueError for every malformed report input.
    def test_reject_coordinate_and_identity_mismatches(self):
        analysis, rows, scores, labels = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            for badscores, badlabels in ((scores[:2], labels), (scores, labels.astype(float)),
                                         (np.full_like(scores, np.nan), labels)):
                with self.assertRaises(ValueError):
                    write_report(Path(tmp) / "review", analysis, rows, badscores, badlabels)
            rows[1]["sample_id"] = rows[0]["sample_id"]
            with self.assertRaises(ValueError):
                write_report(Path(tmp) / "review", analysis, rows, scores, labels)

    # Purpose: Reject baseline records without matching PCA coordinates.
    # Inputs: Sibling report inputs plus baseline records without baseline_scores.
    # Outputs: ValueError prevents report publication.
    def test_missing_baseline_coordinates_rejected(self):
        analysis, rows, scores, labels = self.fixture()
        with tempfile.TemporaryDirectory() as tmp:
            with self.assertRaises(ValueError):
                write_report(Path(tmp) / "review", analysis, rows, scores, labels,
                             baseline_records=[dict(case_id="A")])


if __name__ == "__main__":
    unittest.main()
