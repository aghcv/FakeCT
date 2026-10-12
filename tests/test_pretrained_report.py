"""Native-grid overlap and lesion metrics reject clinically misleading crops."""
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import unittest
import numpy as np

from fakect_pretrained_report import evaluate_arrays, component_metrics, write_report

try:
    import nibabel as nib
except ImportError:
    nib = None


class PretrainedReportTests(unittest.TestCase):
    def masks(self):
        truth = np.zeros((25, 25, 25), bool)
        truth[5:20, 9:16, 9:16] = True
        return truth, np.ones(truth.shape, bool)

    # Purpose: Verify identical native masks have perfect overlap and surface metrics.
    # Inputs: Matching cylindrical truth/prediction and full domain at 1 mm spacing.
    # Outputs: Whole-core Dice 1, HD95 zero and surface Dice 1.
    def test_identical_native_masks_perfect_surface_metrics(self):
        truth, domain = self.masks()
        m = evaluate_arrays(truth, truth, [1, 1, 1], domain_mask=domain)
        self.assertEqual(m["whole_thoracic_core"]["dice"], 1.)
        s = m["whole_thoracic_core"]["surfaces"]
        self.assertEqual(s["hd95_mm"], 0.)
        self.assertEqual(s["surface_dice"], 1.)

    # Purpose: Verify surface distance uses physical voxel spacing.
    # Inputs: Truth shifted one i voxel with spacing 2,1,1 mm.
    # Outputs: HD95 2 mm and surface Dice below one.
    def test_physical_shift_surface_distance_and_tolerance(self):
        truth, domain = self.masks()
        shifted = np.roll(truth, 1, axis=2)
        m = evaluate_arrays(truth, shifted, [2, 1, 1], domain_mask=domain)
        s = m["whole_thoracic_core"]["surfaces"]
        self.assertAlmostEqual(s["hd95_mm"], 2.)
        self.assertLess(s["surface_dice"], 1.)

    # Purpose: Verify local metrics expose a missed small narrowing.
    # Inputs: 75-voxel interior removal predicted as unchanged with 3 mm local context.
    # Outputs: Known changed count 75, one local region and local Dice below whole-core Dice.
    def test_local_score_exposes_missed_small_narrowing(self):
        before, domain = self.masks()
        after = before.copy(); after[11:14, 10:15, 10:15] = False
        changed = before ^ after
        m = evaluate_arrays(after, before, [1, 1, 1], domain_mask=domain, changed_mask=changed,
                            local_context_mm=3, boundary_margin_mm=1)
        self.assertEqual(m["known_changed_voxels_away_from_caps"], 75)
        self.assertLess(m["lesion_local"]["dice"], m["whole_thoracic_core"]["dice"])
        self.assertEqual(m["local_region_count"], 1)

    # Purpose: Verify cropped end caps are excluded from surface distances.
    # Inputs: Cap-touching truth with first two predicted planes removed and 3 mm margin.
    # Outputs: HD95 zero after cap exclusion while volumetric Dice remains below one.
    def test_crop_caps_excluded_from_surface_distances(self):
        truth, domain = self.masks()
        truth[:5, 9:16, 9:16] = True
        prediction = truth.copy(); prediction[:2] = False
        m = evaluate_arrays(truth, prediction, [1, 1, 1], domain_mask=domain, boundary_margin_mm=3)
        self.assertEqual(m["whole_thoracic_core"]["surfaces"]["hd95_mm"], 0.)
        self.assertLess(m["whole_thoracic_core"]["dice"], 1.)

    # Purpose: Verify empty predictions do not imply zero surface distance.
    # Inputs: Nonempty native truth and entirely empty prediction.
    # Outputs: Dice zero, undefined HD95 and surface Dice zero.
    def test_empty_prediction_not_reported_as_zero_distance(self):
        truth, domain = self.masks()
        m = evaluate_arrays(truth, np.zeros_like(truth), [1, 1, 1], domain_mask=domain)
        self.assertEqual(m["whole_thoracic_core"]["dice"], 0.)
        self.assertIsNone(m["whole_thoracic_core"]["surfaces"]["hd95_mm"])
        self.assertEqual(m["whole_thoracic_core"]["surfaces"]["surface_dice"], 0.)

    # Purpose: Verify boundary-touching components are distinguished from interior islands.
    # Inputs: Cap-connected main body plus separate interior component.
    # Outputs: Two components with one boundary-touching and one interior component.
    def test_boundary_components_not_mislabeled_as_interior_islands(self):
        truth, domain = self.masks()
        truth[:5, 9:16, 9:16] = True
        truth[12:14, 20:22, 20:22] = True
        c = component_metrics(truth, domain, [1, 1, 1], boundary_margin_mm=2)
        self.assertEqual(c["components"], 2)
        self.assertEqual(c["boundary_touching_components"], 1)
        self.assertEqual(c["interior_components"], 1)

    # Purpose: Verify unchanged baselines omit lesion-local metrics.
    # Inputs: Identical masks and all-zero changed mask.
    # Outputs: No local metric and explicit unchanged-baseline/no-interior-change status.
    def test_unchanged_baseline_has_no_lesion_metric(self):
        truth, domain = self.masks()
        m = evaluate_arrays(truth, truth, [1, 1, 1], domain_mask=domain, changed_mask=np.zeros_like(truth))
        self.assertIsNone(m["lesion_local"])
        self.assertEqual(m["local_status"], "unchanged_baseline_or_no_interior_changes")

    # Purpose: Verify prediction outside the thoracic domain is excluded.
    # Inputs: Extra predicted abdominal voxels where domain is false.
    # Outputs: Whole-thoracic-core Dice remains one.
    def test_domain_excludes_abdominal_prediction(self):
        truth, domain = self.masks(); domain[:5] = False
        prediction = truth.copy(); prediction[:4, 9:16, 9:16] = True
        m = evaluate_arrays(truth, prediction, [1, 1, 1], domain_mask=domain)
        self.assertEqual(m["whole_thoracic_core"]["dice"], 1.)

    # Purpose: Reject incompatible grids, nonpositive spacing and nonbinary masks.
    # Inputs: Mismatched prediction shape, spacing zero and fractional truth values.
    # Outputs: ValueError rejects each invalid native-grid metric input.
    def test_reject_incompatible_grids_or_invalid_spacing(self):
        truth, domain = self.masks()
        with self.assertRaises(ValueError):
            evaluate_arrays(truth, truth[:4], [1, 1, 1])
        with self.assertRaises(ValueError):
            evaluate_arrays(truth, truth, [0, 1, 1])
        with self.assertRaises(ValueError):
            evaluate_arrays(truth.astype(float) * .5, truth, [1, 1, 1])


@unittest.skipUnless(nib is not None, "NIfTI integration requires the benchmark's nibabel environment")
class PretrainedNiftiReportTests(unittest.TestCase):
    masks = PretrainedReportTests.masks
    def fixture(self, root):
        truth, domain = self.masks()
        row = dict(sample_id="fixture:baseline", case_id="fixture", baseline=True)
        for name, array in (("input", truth.astype(np.int16) * 100), ("truth", truth),
                            ("prediction", truth), ("domain", domain)):
            path = root / (name + ".nii.gz")
            image = nib.Nifti1Image(array.astype(np.int16).transpose(2, 1, 0), np.diag([-1., 1., 1., 1.]))
            image.header.set_xyzt_units("mm")
            nib.save(image, path)
            row[name + "_nifti"] = str(path)
            row[name + "_nifti_sha256"] = hashlib.sha256(path.read_bytes()).hexdigest()
        return row

    # Purpose: Verify NIfTI reports publish offline viewers with checked output hashes.
    # Inputs: Signed matching native input/truth/prediction/domain NIfTI fixture.
    # Outputs: One perfect sample, one viewer scene, expected opacity presets and verified artifacts.
    def test_native_nifti_report_offline_viewer_and_checksums(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = self.fixture(root)
            result = write_report({"samples": [row]}, root / "review")
            self.assertEqual(result["completed_samples"], 1)
            payload = json.loads((root / "review/metrics.json").read_text())
            self.assertEqual(payload["samples"][0]["metrics"]["whole_thoracic_core"]["dice"], 1.)
            self.assertEqual(payload["viewer"]["viewer"], str(root / "review/report-3d/index.html"))
            viewer = json.loads((root / "review/report-3d/manifest.json").read_text())
            self.assertEqual(viewer["opacity_presets"], [.15, .45, .8])
            self.assertEqual(len(viewer["scenes"]), 1)
            provenance = json.loads((root / "review/provenance.json").read_text())
            for name, checksum in provenance["outputs_sha256"].items():
                self.assertEqual(checksum, hashlib.sha256((root / "review" / name).read_bytes()).hexdigest())

    # Purpose: Reject mismatched NIfTI affines without partial publication.
    # Inputs: Prediction affine translated 2 mm from the reference.
    # Outputs: ValueError before review directory creation.
    def test_wrong_affine_rejected_without_resampling_or_partial_output(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp); row = self.fixture(root)
            path = Path(row["prediction_nifti"])
            original = nib.load(path)
            affine = original.affine.copy(); affine[0, 3] += 2.
            nib.save(nib.Nifti1Image(np.asanyarray(original.dataobj), affine), path)
            row.pop("prediction_nifti_sha256")
            with self.assertRaises(ValueError):
                write_report({"samples": [row]}, root / "review")
            self.assertFalse((root / "review").exists())

    # Purpose: Verify streamed adapter NIfTIs interoperate with native metric reports.
    # Inputs: Adapter-exported target/domain/changed masks and exact copied target prediction.
    # Outputs: Exact masks, global/local Dice 1, two changes and preserved affine/origin slice mapping.
    def test_streamed_adapter_export_native_mask_and_report_interoperate(self):
        from fakect_pretrained_adapter import export_thorax_nifti, ThoracicCoreProvider
        from test_pretrained_adapter import PretrainedAdapterTests
        fixture = PretrainedAdapterTests()
        fixture.setUp()
        try:
            with tempfile.TemporaryDirectory() as tmp:
                root = Path(tmp)
                paths = {key: root / (key + ".nii.gz") for key in ("target", "domain", "changed")}
                image_path = root / "ct-hu.nii.gz"
                exported = export_thorax_nifti(fixture.row, fixture.selection, fixture.evidence,
                    image_path, context_mm=0, slab_slices=2, mask_output_paths=paths)
                provider = ThoracicCoreProvider(fixture.row, fixture.selection, fixture.evidence)
                expected = provider.read_block(exported["origin_ijk"], exported["shape_kji"])
                for key, path in paths.items():
                    actual = np.asanyarray(nib.load(path).dataobj).transpose(2, 1, 0)
                    np.testing.assert_array_equal(actual, expected[key])
                prediction = root / "predicted-aorta.nii.gz"
                shutil.copyfile(paths["target"], prediction)
                row = dict(sample_id=fixture.row["sample_id"], case_id="fixture", baseline=False,
                           input_nifti=str(image_path), truth_nifti=str(paths["target"]),
                           domain_nifti=str(paths["domain"]), changed_nifti=str(paths["changed"]),
                           prediction_nifti=str(prediction), export=exported)
                result = write_report({"samples": [row]}, root / "review",
                                      settings=dict(boundary_margin_mm=0, local_context_mm=3))
                self.assertEqual(result["completed_samples"], 1)
                saved = json.loads((root / "review/metrics.json").read_text())["samples"][0]
                self.assertEqual(saved["metrics"]["whole_thoracic_core"]["dice"], 1.)
                self.assertEqual(saved["metrics"]["lesion_local"]["dice"], 1.)
                self.assertEqual(saved["metrics"]["known_changed_voxels_in_domain"], 2)
                np.testing.assert_allclose(saved["native_affine_xyz"], exported["affine_ras_ijk"], atol=1e-6)
                self.assertGreater(saved["export_origin_ijk"][2], 0)
                self.assertEqual(saved["slice_native_k"],
                                 [k + exported["origin_ijk"][2] for k in saved["slice_export_k"]])
        finally:
            fixture.doCleanups()


if __name__ == "__main__":
    unittest.main()
