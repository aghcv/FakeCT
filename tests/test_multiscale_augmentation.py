from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_multiscale_augmentation import apply_transform, augment_overview, inverse_box


class MultiscaleAugmentationTests(unittest.TestCase):
    def setUp(self):
        self.shape = np.array([20, 18, 16])
        self.x = np.zeros((1, *self.shape, 2), np.float32)
        self.x[..., 1] = 1
        lo, hi = np.array([6, 5, 4]), np.array([12, 11, 10])
        self.box = np.r_[(lo + hi) / (2 * self.shape), (hi - lo) / self.shape].astype(np.float32)[None]
        self.settings = dict(translate_fraction_kji=(.08, .06, .06), scale_range=(.85, 1.15), crop_fraction_max=.12)

    # Purpose: Verify identity augmentation preserves values, geometry and inputs.
    # Inputs: 20x18x16 two-channel image with invalid planes and normalized target box.
    # Outputs: Exact independent float32 image and box copies with original crop faces.
    def test_identity_preserves_values_grid_and_inputs(self):
        rng = np.random.default_rng(7)
        self.x[..., 0] = rng.random(self.x[..., 0].shape)
        self.x[:, :2] = 0
        before, box_before = self.x.copy(), self.box.copy()
        result, box, metadata = apply_transform(self.x, self.box)
        np.testing.assert_array_equal(result, before)
        np.testing.assert_array_equal(box, box_before)
        np.testing.assert_array_equal(self.x, before)
        np.testing.assert_array_equal(self.box, box_before)
        self.assertFalse(np.shares_memory(result, self.x))
        self.assertEqual(metadata['crop_faces_k'], [0, 20])
        self.assertEqual(result.dtype, np.float32)

    # Purpose: Verify integer translation moves landmarks and boxes without wrapping.
    # Inputs: Single landmark translated by 2,-1,1 voxels.
    # Outputs: Landmark and box shifted correctly, exposed borders zero and inverse restores original box.
    def test_integer_translation_moves_landmark_and_exact_face_box_without_wrap(self):
        point = np.array([8, 7, 6])
        self.x[(0, *point, 0)] = 1
        shift = np.array([2, -1, 1])
        result, box, metadata = apply_transform(self.x, self.box, translation_kji=shift / self.shape)
        self.assertAlmostEqual(float(result[(0, *(point + shift), 0)]), 1., places=6)
        self.assertAlmostEqual(float(result[0, ..., 0].sum()), 1., places=6)
        np.testing.assert_allclose(box[0, :3], self.box[0, :3] + shift / self.shape, atol=1e-7)
        np.testing.assert_array_equal(box[0, 3:], self.box[0, 3:])
        self.assertTrue(np.all(result[:, :2] == 0))
        self.assertTrue(np.all(result[:, :, -1] == 0))
        self.assertTrue(np.all(result[:, :, :, :1] == 0))
        np.testing.assert_allclose(inverse_box(box, metadata), self.box, atol=1e-7)

    # Purpose: Verify physical ramp interpolation and box geometry share one affine.
    # Inputs: Anisotropic linear ramp, scale 1.12 and fractional translation.
    # Outputs: Analytic sampled intensities and physical box extents match; inverse recovers supervision.
    def test_anisotropic_physical_ramp_and_box_use_identical_affine(self):
        spacing = np.array([2.3, 1.1, .8])  # k,j,i mm
        coords = np.indices(tuple(self.shape), dtype=np.float64)
        physical = (coords + .5) * spacing[:, None, None, None]
        coefficients = np.array([.13, .21, .37])
        self.x[0, ..., 0] = np.einsum('a,akji->kji', coefficients, physical)
        scale, translation = 1.12, np.array([.015, -.02, .025])
        result, box, metadata = apply_transform(self.x, self.box, scale, translation)
        for point in [(6, 6, 6), (10, 9, 8), (13, 12, 10)]:
            u = (np.array(point) + .5) / self.shape
            source_u = (u - .5 - translation) / scale + .5
            expected = np.dot(coefficients, source_u * self.shape * spacing)
            self.assertAlmostEqual(float(result[(0, *point, 0)]), expected, places=5)
        extent = self.shape * spacing
        expected_center_mm = extent / 2 + scale * (self.box[0, :3] * extent - extent / 2) + translation * extent
        np.testing.assert_allclose(box[0, :3] * extent, expected_center_mm, atol=2e-6)
        np.testing.assert_allclose(box[0, 3:] * extent, self.box[0, 3:] * extent * scale, atol=2e-6)
        np.testing.assert_allclose(inverse_box(box[0], metadata), self.box[0], atol=1e-7)
        self.assertEqual(inverse_box(box[0], metadata).shape, (6,))

    # Purpose: Verify partial-edge validity weighting preserves constant tissue.
    # Inputs: Constant intensity 0.73 with zero/fractional validity and affine resampling.
    # Outputs: Positive-validity intensities remain 0.73, padding zero and validity bounded 0-1.
    def test_validity_weighting_preserves_constant_tissue_at_partial_edges(self):
        self.x[..., 0] = .73
        self.x[:, :2, ..., 1] = 0
        self.x[:, 2, ..., 1] = .35
        self.x[:, -1, ..., 1] = .4
        result, _, _ = apply_transform(self.x, self.box, .9, (.017, -.013, .012))
        valid = result[..., 1]
        self.assertTrue(np.any((valid > 0) & (valid < 1)))
        self.assertTrue(np.any(valid == 0))
        np.testing.assert_allclose(result[..., 0][valid > 0], .73, atol=1e-6)
        np.testing.assert_array_equal(result[..., 0][valid == 0], 0)
        self.assertTrue(np.all((valid >= 0) & (valid <= 1)))

    # Purpose: Verify truncation snaps inward without renumbering boxes.
    # Inputs: Crop bounds 0.051-0.949 on a 20-plane image.
    # Outputs: Crop faces 2-18, normalized bounds 0.1-0.9, unchanged boxes and interior values.
    def test_truncation_snaps_inward_and_does_not_renumber_boxes(self):
        self.x[..., 0] = .4
        result, box, metadata = apply_transform(self.x, self.box, crop_bounds_k=(.051, .949))
        self.assertEqual(metadata['crop_faces_k'], [2, 18])
        self.assertEqual(metadata['crop_bounds_k'], [.1, .9])
        np.testing.assert_array_equal(box, self.box)
        self.assertTrue(np.all(result[:, :2] == 0))
        self.assertTrue(np.all(result[:, 18:] == 0))
        np.testing.assert_array_equal(result[:, 2:18], self.x[:, 2:18])

    # Purpose: Reject probes losing target or safety context before interpolation.
    # Inputs: Excessive translation/scale/crops with interpolation mocked to fail.
    # Outputs: Ineligible ValueError before interpolation; explicit zero safety context permits smaller crop.
    def test_fixed_probe_rejects_target_or_safety_context_loss_before_interpolation(self):
        with patch('fakect_multiscale_augmentation.affine_transform', side_effect=AssertionError('interpolated')):
            for settings in [dict(translation_kji=(.6, 0, 0)), dict(scale=4),
                             dict(crop_bounds_k=(.31, 1)), dict(crop_bounds_k=(.21, 1))]:
                with self.subTest(settings=settings), self.assertRaisesRegex(ValueError, 'ineligible'):
                    apply_transform(self.x, self.box, **settings)
        # Same crop excludes context, not target; opting out of context is explicit.
        _, _, metadata = apply_transform(self.x, self.box, crop_bounds_k=(.21, 1), safety_context_voxels=0)
        self.assertEqual(metadata['crop_faces_k'], [5, 20])

    # Purpose: Verify random overview augmentation is reproducible and context-safe.
    # Inputs: Two seed-42 generators and 30 bounded scale/translation/crop draws.
    # Outputs: Identical samples, settings bounds respected, target margin retained and invertible boxes.
    def test_random_samples_are_reproducible_bounded_and_keep_target_context(self):
        rng_a, rng_b = np.random.default_rng(42), np.random.default_rng(42)
        for _ in range(30):
            a, box_a, meta_a = augment_overview(self.x, self.box, rng_a, **self.settings)
            b, box_b, meta_b = augment_overview(self.x, self.box, rng_b, **self.settings)
            np.testing.assert_array_equal(a, b)
            np.testing.assert_array_equal(box_a, box_b)
            self.assertEqual(meta_a, meta_b)
            self.assertGreaterEqual(meta_a['scale'], .85)
            self.assertLessEqual(meta_a['scale'], 1.15)
            self.assertTrue(np.all(np.abs(meta_a['translation_kji']) <= np.array(self.settings['translate_fraction_kji']) + 1e-12))
            low, high = box_a[0, :3] - box_a[0, 3:] / 2, box_a[0, :3] + box_a[0, 3:] / 2
            margin = 2 / self.shape
            self.assertTrue(np.all(low - margin >= np.array([meta_a['crop_bounds_k'][0], 0, 0]) - 1e-7))
            self.assertTrue(np.all(high + margin <= np.array([meta_a['crop_bounds_k'][1], 1, 1]) + 1e-7))
            self.assertLessEqual(meta_a['crop_bounds_k'][0], .12)
            self.assertLessEqual(1 - meta_a['crop_bounds_k'][1], .12)
            np.testing.assert_allclose(inverse_box(box_a, meta_a), self.box, atol=1e-7)

    # Purpose: Verify random scales are conditioned on feasible target support.
    # Inputs: Large 0.7-extent box with scale range 0.95-2, then infeasible 2-3.
    # Outputs: Feasible upper range reduced without underflow; impossible range raises ValueError.
    def test_random_feasible_scale_is_conditioned_without_out_of_range_fallback(self):
        large = np.array([[.5, .5, .5, .7, .7, .7]], np.float32)
        _, _, metadata = augment_overview(self.x, large, np.random.default_rng(0),
            translate_fraction_kji=(0, 0, 0), scale_range=(.95, 2.), crop_fraction_max=0)
        self.assertLess(metadata['feasible_scale_range'][1], 2.)
        self.assertGreaterEqual(metadata['scale'], .95)
        with self.assertRaisesRegex(ValueError, 'No feasible scale'):
            augment_overview(self.x, large, np.random.default_rng(0),
                translate_fraction_kji=(0, 0, 0), scale_range=(2., 3.), crop_fraction_max=0)

    # Purpose: Verify inverse boxes retain out-of-field prediction error.
    # Inputs: Scale 0.8, translation 0.02 and an off-field predicted box.
    # Outputs: Inverse center remains negative and extent becomes 0.25.
    def test_inverse_preserves_out_of_fov_prediction_error(self):
        _, _, metadata = apply_transform(self.x, self.box, .8, (.02, 0, 0))
        prediction = np.array([.01, .5, .5, .2, .2, .2], np.float32)
        inverse = inverse_box(prediction, metadata)
        self.assertLess(inverse[0], 0)
        self.assertAlmostEqual(float(inverse[3]), .25, places=6)

    # Purpose: Reject invalid augmentation contracts and excessive voxel budgets.
    # Inputs: Invalid transforms, RNG/settings, image dtype/shape/validity, huge image or unknown inverse schema.
    # Outputs: ValueError rejects each malformed contract and explicit budget excess.
    def test_invalid_contracts_and_budgets(self):
        for options in [dict(scale=0), dict(scale=float('nan')), dict(translation_kji=(0, 0)),
                        dict(crop_bounds_k=(.9, .1)), dict(crop_bounds_k=(.101, .102)),
                        dict(safety_context_voxels=-1), dict(safety_context_voxels=10)]:
            with self.subTest(options=options), self.assertRaises(ValueError):
                apply_transform(self.x, self.box, **options)
        for options in [dict(translate_fraction_kji=(-.1, 0, 0)), dict(scale_range=(1.2, .8)),
                        dict(crop_fraction_max=.5), dict(crop_fraction_max=True)]:
            config = dict(self.settings); config.update(options)
            with self.subTest(options=options), self.assertRaises(ValueError):
                augment_overview(self.x, self.box, np.random.default_rng(1), **config)
        for image in [self.x.astype(np.float64), self.x[0], self.x * np.nan]:
            with self.assertRaises(ValueError): apply_transform(image, self.box)
        bad_valid = self.x.copy(); bad_valid[..., 1] = 2
        with self.assertRaises(ValueError): apply_transform(bad_valid, self.box)
        huge_view = np.broadcast_to(np.zeros((1, 1, 1, 1, 2), np.float32), (1, 200, 200, 200, 2))
        with self.assertRaisesRegex(ValueError, 'budget'): apply_transform(huge_view, self.box)
        with self.assertRaises(ValueError): augment_overview(self.x, self.box, None, **self.settings)
        with self.assertRaises(ValueError): inverse_box(self.box, {'schema_version': 'unknown'})


if __name__ == '__main__':
    unittest.main()
