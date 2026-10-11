"""Boundary, provenance, and refusal tests for draft cohort sidecar labels."""
import copy
import importlib.util
import json
from pathlib import Path
import tempfile
import unittest


SCRIPT = Path(__file__).resolve().parents[1] / "scripts/research/evaluation/classify_coa_cohort.py"
spec = importlib.util.spec_from_file_location("classify_coa_cohort", SCRIPT)
coa = importlib.util.module_from_spec(spec)
spec.loader.exec_module(coa)


def configuration():
    return dict(step_name="coa_narrow", positive_loss_fraction=.2,
                review_half_width=.02, sensitivity_thresholds=[.1, .2, .3],
                original_baselines_non_coa=True)


def sample(removed=20, spacing=(.4, .5, 1.)):
    after = 100 - removed
    voxel = coa.math.prod(spacing)
    guard = dict(enabled=True, accepted=True, connectivity_ok=True, status="accepted",
                 baseline_target_voxels=100, retained_target_voxels=after,
                 retained_volume_ratio=after / 100,
                 baseline_volume_mm3=100 * voxel, retained_volume_mm3=after * voxel,
                 requested_distance_mm=3., accepted_distance_mm=1.5,
                 scope_semantics="Pre-step selected target within the positive longitudinal profile support.",
                 reference_semantics="Fixed immediately before this edit step, after earlier recipe steps.",
                 attempts=[dict(distance_mm=1.5, retained_target_voxels=after,
                                retained_volume_ratio=after / 100, accepted=True, connectivity_ok=True)])
    return dict(sample_id="case:edit", case_id="case", variant_id="edit", anatomy_family="family",
                baseline=False, baseline_equivalent=False, spacing_ijk_mm=list(spacing), parameters={},
                step_outcomes=[dict(step_name="coa_narrow", iteration=1, operation="erosion", status="applied",
                                    removed_voxels=removed, added_voxels=0,
                                    requested_distance_mm=3., accepted_distance_mm=1.5,
                                    erosion_safeguard=guard)])


def baseline():
    row = sample()
    row.update(sample_id="case:baseline", variant_id="baseline", baseline=True, baseline_equivalent=True)
    step = row["step_outcomes"][0]
    step.update(operation="none", status="no_change", removed_voxels=0, added_voxels=0, erosion_safeguard=None)
    return row


class CoALabelTests(unittest.TestCase):
    def test_threshold_is_inclusive_and_uses_achieved_local_loss(self):
        below = coa.classify_sample(sample(19), configuration())
        exact = coa.classify_sample(sample(20), configuration())
        self.assertFalse(below["coa_benchmark_positive"])
        self.assertTrue(below["coa_edit_applied"])
        self.assertEqual(below["class_description"], "below_threshold_induced_coa")
        self.assertTrue(exact["coa_benchmark_positive"])
        self.assertEqual(exact[coa.SCORE_NAME], .2)
        self.assertEqual(exact["accepted_distance_mm"], 1.5)

    def test_original_baseline_is_explicit_assumption_without_local_denominator(self):
        result = coa.classify_sample(baseline(), configuration())
        self.assertFalse(result["coa_benchmark_positive"])
        self.assertFalse(result["coa_edit_applied"])
        self.assertEqual(result["class_description"], "reference_non_coa_assumed")
        self.assertIsNone(result["baseline_target_voxels"])

    def test_review_band_includes_both_endpoints(self):
        for removed in (18, 20, 22):
            self.assertTrue(coa.classify_sample(sample(removed), configuration())["review_near_threshold"])
        for removed in (17, 23):
            self.assertFalse(coa.classify_sample(sample(removed), configuration())["review_near_threshold"])

    def test_invalid_outcomes_are_unknown_never_negative(self):
        mutations = [
            lambda s: s["step_outcomes"][0].update(erosion_safeguard=None),
            lambda s: s["step_outcomes"][0]["erosion_safeguard"].update(retained_volume_ratio=.9),
            lambda s: s["step_outcomes"][0]["erosion_safeguard"].update(baseline_volume_mm3=100),
            lambda s: s["step_outcomes"][0]["erosion_safeguard"].update(accepted=False),
            lambda s: s["step_outcomes"][0]["erosion_safeguard"].update(attempts=[]),
            lambda s: s["step_outcomes"][0].update(removed_voxels=19),
            lambda s: s["step_outcomes"][0].update(iteration=2),
            lambda s: s["step_outcomes"].append(copy.deepcopy(s["step_outcomes"][0])),
            lambda s: s["step_outcomes"].append(dict(step_name="later", status="applied")),
            lambda s: s.update(baseline=True),
        ]
        for mutation in mutations:
            row = sample()
            mutation(row)
            result = coa.classify_sample(row, configuration())
            self.assertIsNone(result["coa_benchmark_positive"])
            self.assertIsNone(result[coa.SCORE_NAME])
            self.assertTrue(result["errors"])

    def test_summary_keeps_references_separate_from_below_threshold_edits(self):
        rows = [coa.classify_sample(row, configuration()) for row in [baseline(), sample(15), sample(20), sample(35)]]
        summary = coa.summarize(rows, configuration())
        self.assertEqual(summary["benchmark_positive"], 2)
        self.assertEqual(summary["benchmark_negative"], 2)
        self.assertEqual(summary["below_threshold_edited"], 1)
        self.assertEqual(summary["original_reference"], 1)
        self.assertEqual([r["benchmark_positive"] for r in summary["threshold_sensitivity"]], [3, 2, 1])

    def test_provenance_export_immutable_and_tampering_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            parents = [baseline(), sample()]
            parent_path = root / "parent.json"
            for row in parents:
                row["geometry"] = {"spacing_ijk_mm": row["spacing_ijk_mm"]}
            parent_path.write_text(json.dumps(dict(complete=True, samples=parents)))
            upstream = root / "upstream.json"
            upstream.write_text("{}")
            rows = copy.deepcopy(parents)
            for row in rows:
                row.update(prepared_manifest=str(parent_path), prepared_manifest_sha256=coa.sha256(parent_path))
            manifest = root / "manifest.json"
            manifest.write_text(json.dumps(dict(complete=True, samples=rows, sample_count=2,
                                                source_manifest=str(upstream), source_manifest_sha256=coa.sha256(upstream))))
            Path(str(manifest) + ".sha256").write_text(coa.sha256(manifest))
            ini = root / "input.ini"
            ini.write_text("fixture")
            config = {**configuration(), "dataset_manifest": str(manifest),
                      "output_directory": str(root / "labels"), "config_path": str(ini),
                      "config_sha256": coa.sha256(ini)}
            source_hash = coa.sha256(manifest)
            planned = coa.run(config, validate_only=True)
            self.assertFalse(Path(config["output_directory"]).exists())
            result = coa.run(config)
            self.assertEqual(planned["summary"], result["summary"])
            self.assertFalse(result["ready_for_training"])
            self.assertEqual(result["status"], "draft")
            self.assertEqual(coa.sha256(manifest), source_hash)
            self.assertTrue((root / "labels/report.html").is_file())
            self.assertFalse((root / "labels/INCOMPLETE").exists())
            with self.assertRaisesRegex(ValueError, "already exists"):
                coa.run(config)
            parent_path.write_text(parent_path.read_text() + " ")
            with self.assertRaisesRegex(ValueError, "SHA-256 mismatch"):
                coa.run(config, validate_only=True)


if __name__ == "__main__":
    unittest.main()
