"""Physical resampling preserves native artifacts and binary target meaning."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
from fakect_model_resampling import output_shape, resample_pair, resampling_contract, spacing


class ModelResamplingTests(unittest.TestCase):
    # Purpose: Verify resampling shape computation reverses spacing axes correctly.
    # Inputs: Anisotropic shape/spacing pairs, downsampled grid, and subvoxel single-cell grid.
    # Outputs: Typed IJK spacing and expected KJI output shapes including minimum one voxel.
    def test_spacing_and_shape_use_ijk_distances_for_kji_arrays(self):
        self.assertEqual(spacing(['1', '2', '3']), (1., 2., 3.))
        self.assertEqual(output_shape((2, 3, 4), [2., 3., 4.], [1., 1., 2.]), (4, 9, 8))
        self.assertEqual(output_shape((3, 4, 5), [1., 1., 1.], [2., 2., 2.]), (2, 2, 3))
        self.assertEqual(output_shape((1, 1, 1), [.1, .2, .3], [1., 1., 1.]), (1, 1, 1))

    # Purpose: Verify resampling rejects malformed or excessive grids before allocation.
    # Inputs: Nonpositive/nonfinite/bool spacing, invalid shapes, tiny target spacing, and extreme ratio.
    # Outputs: ValueError for grid validation, crop limits, and spacing-ratio overflow.
    def test_invalid_grids_fail_before_allocation(self):
        for value in ([1., 0., 1.], [1., float('nan'), 1.], [1., float('inf'), 1.],
                      [True, 1., 1.], [None, 1., 1.], [1., 1.], '1,1,1'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                spacing(value)
        for shape in ((1, 2), (0, 2, 3), (True, 2, 3), (1., 2, 3)):
            with self.subTest(shape=shape), self.assertRaises(ValueError):
                output_shape(shape, [1., 1., 1.], [1., 1., 1.])
        for target in ([1e-300, 1e-300, 1e-300], [.01, .01, .01]):
            with self.subTest(target=target), self.assertRaisesRegex(ValueError, 'crop limit'):
                output_shape((32, 32, 32), [1., 1., 1.], target)
        with self.assertRaisesRegex(ValueError, 'spacing ratios'):
            output_shape((1, 1, 1), [1e-320, 1., 1.], [1., 1., 1.])

    # Purpose: Verify identity resampling returns original immutable arrays.
    # Inputs: Read-only float32 image/uint8 mask with equal anisotropic source/target spacing.
    # Outputs: Output objects are the original image and mask.
    def test_identity_preserves_exact_values_and_native_arrays(self):
        image = np.arange(24, dtype=np.float32).reshape(2, 3, 4)
        mask = (image % 2).astype(np.uint8)
        image.flags.writeable = mask.flags.writeable = False
        resampled_image, resampled_mask = resample_pair(image, mask, [1., 2., 3.], [1., 2., 3.])
        self.assertIs(resampled_image, image)
        self.assertIs(resampled_mask, mask)

    # Purpose: Verify upsampling aligns lower voxel faces with channel-specific interpolation.
    # Inputs: Signals 0/10/20 and binary center mask sampled from 2 mm to 1 mm.
    # Outputs: Expected linear values and nearest binary labels with original arrays preserved.
    def test_linear_image_nearest_mask_and_lower_face_alignment(self):
        image = np.array([[[0., 10., 20.]]], dtype=np.float32)
        mask = np.array([[[0, 1, 0]]], dtype=np.uint8)
        before_image, before_mask = image.copy(), mask.copy()
        image.flags.writeable = mask.flags.writeable = False
        actual_image, actual_mask = resample_pair(image, mask, [2., 1., 1.], [1., 1., 1.])
        np.testing.assert_array_equal(actual_image, [[[0., 2.5, 7.5, 12.5, 17.5, 20.]]])
        np.testing.assert_array_equal(actual_mask, [[[0, 0, 1, 1, 0, 0]]])
        self.assertEqual(actual_image.dtype, np.float32)
        self.assertEqual(actual_mask.dtype, np.uint8)
        np.testing.assert_array_equal(image, before_image)
        np.testing.assert_array_equal(mask, before_mask)

    # Purpose: Verify downsampling handles the final partial source boundary.
    # Inputs: Three 1 mm samples with signals 0/10/20 resampled at 2 mm.
    # Outputs: Signals 5/20 and mask 1/0.
    def test_downsampling_partial_boundary_uses_nearest_source(self):
        image = np.array([[[0., 10., 20.]]], dtype=np.float32)
        mask = np.array([[[0, 1, 0]]], dtype=np.uint8)
        actual_image, actual_mask = resample_pair(image, mask, [1., 1., 1.], [2., 1., 1.])
        np.testing.assert_array_equal(actual_image, [[[5., 20.]]])
        np.testing.assert_array_equal(actual_mask, [[[1, 0]]])

    # Purpose: Verify pair resampling preserves strict dtype and binary-mask contracts.
    # Inputs: Float64 image, floating/nonbinary mask, NaN image, and unaligned shapes.
    # Outputs: ValueError for each invalid pair.
    def test_invalid_pairs_are_not_silently_cast_or_relabelled(self):
        image = np.zeros((2, 3, 4), np.float32)
        mask = np.zeros(image.shape, np.uint8)
        for bad_image, bad_mask in ((image.astype(np.float64), mask), (image, mask.astype(float)),
                                    (image, mask+2), (image+np.nan, mask), (image, mask[:1])):
            with self.subTest(image=bad_image.shape, mask=bad_mask.shape), self.assertRaises(ValueError):
                resample_pair(bad_image, bad_mask, [1., 1., 1.], [1., 1., 1.])

    # Purpose: Verify resampling provenance specifies geometry, methods, and implementations.
    # Inputs: Target spacing 1/2/3 mm.
    # Outputs: Lower-face alignment, linear/nearest methods, two implementation hashes, and NumPy/SciPy versions.
    def test_contract_pins_consumer_and_interpolation_dependencies(self):
        contract = resampling_contract([1., 2., 3.])
        self.assertEqual(contract['target_spacing_mm'], [1., 2., 3.])
        self.assertEqual(contract['grid_alignment'], 'lower_voxel_face')
        self.assertEqual(contract['image_interpolation'], 'linear')
        self.assertEqual(contract['mask_interpolation'], 'nearest')
        self.assertEqual(set(contract['implementation_sha256']), {'fakect_model_resampling.py', 'fakect_segmentation.py'})
        self.assertEqual(set(contract['dependency_versions']), {'numpy', 'scipy'})
        self.assertTrue(all(len(digest) == 64 for digest in contract['implementation_sha256'].values()))


if __name__ == '__main__':
    unittest.main()
