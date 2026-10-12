"""Physical normalization and paired-data boundaries for CT descriptors."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fakect_ct_delta_features import (extract_array, extract_variant, feature_names,
    prepare_array_geometry, prepare_case_geometry)


def fixture(spacing=1., physical_scale=1., attenuation=.2):
    n = int(round(20 / spacing))
    spacing_mm = spacing * physical_scale
    geometry = {'crop_shape_kji': [n, n, n], 'crop_origin_ijk': [0, 0, 0],
        'spacing_ijk_mm': [spacing_mm] * 3,
        'roi_nodes_ijk': [[10 / spacing, 10 / spacing, 0],
                          [10 / spacing, 10 / spacing, 20 / spacing]]}
    config = {'roi': {'center_ijk': geometry['roi_nodes_ijk']},
              'centerline': {'smoothing_mm': 0}}
    selection = {'aorta_median_ring_radius_mm': 2 * physical_scale}
    image = np.full((n, n, n), attenuation, dtype=np.float32)
    target = np.ones_like(image, dtype=bool)
    return image, target, geometry, config, selection


def descriptor(*args, **kwargs):
    return prepare_array_geometry(*args, arc_bins=4, radial_bins=1,
                                  angular_bins=1, radial_extent=100, **kwargs)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class CTDeltaFeaturesTest(unittest.TestCase):
    # Purpose: Verify baseline CT differences are exactly zero and geometry is immutable.
    # Inputs: Uniform baseline attenuation and its prepared eight-feature descriptor.
    # Outputs: Features are zero, coverage is complete, near-baseline is true, and cached arrays are read-only.
    def test_same_baseline_is_exact_zero_and_arrays_are_readonly(self):
        inputs = fixture()
        case = descriptor(*inputs)
        features, summary = extract_array(case, inputs[0])
        np.testing.assert_array_equal(features, np.zeros(8))
        self.assertTrue(summary['ct_near_baseline'])
        self.assertEqual(summary['ct_delta_energy_coverage_fraction'], 1.)
        self.assertFalse(case.baseline_image_cm_inverse.flags.writeable)
        self.assertFalse(case.bin_index.flags.writeable)

    # Purpose: Verify physical CT integrals are independent of voxel resolution.
    # Inputs: 1 mm and 0.5 mm fixtures with uniform attenuation increase 0.02.
    # Outputs: Features and changed physical volume match while fine-grid voxel counts increase eightfold.
    def test_physical_voxel_volume_makes_integral_resolution_independent(self):
        coarse, fine = fixture(1.), fixture(.5)
        fc, sc = extract_array(descriptor(*coarse), coarse[0] + .02)
        ff, sf = extract_array(descriptor(*fine), fine[0] + .02)
        np.testing.assert_allclose(fc, ff, rtol=1e-6, atol=1e-10)
        self.assertEqual(sc['ct_changed_volume_mm3'], sf['ct_changed_volume_mm3'])
        self.assertEqual(sc['native_ct_changed_voxels'] * 8, sf['native_ct_changed_voxels'])

    # Purpose: Verify descriptor normalization removes phantom physical scale.
    # Inputs: Identical voxel signals with physical scale factors 1 and 2.
    # Outputs: Features and normalized changed volume match; physical changed volume increases eightfold.
    def test_normalization_removes_phantom_physical_scale(self):
        small, large = fixture(physical_scale=1), fixture(physical_scale=2)
        fs, ss = extract_array(descriptor(*small), small[0] + .02)
        fl, sl = extract_array(descriptor(*large), large[0] + .02)
        np.testing.assert_allclose(fs, fl, rtol=1e-6, atol=1e-10)
        self.assertAlmostEqual(ss['normalized_ct_changed_volume'], sl['normalized_ct_changed_volume'])
        self.assertAlmostEqual(sl['ct_changed_volume_mm3'], 8 * ss['ct_changed_volume_mm3'])

    # Purpose: Verify relative attenuation normalizes global signal amplitude.
    # Inputs: Baselines 0.2 and 0.4 each multiplied by 1.2.
    # Outputs: Extracted feature vectors match within numeric tolerance.
    def test_relative_attenuation_removes_global_signal_scale(self):
        first, second = fixture(attenuation=.2), fixture(attenuation=.4)
        f1, _ = extract_array(descriptor(*first), first[0] * 1.2)
        f2, _ = extract_array(descriptor(*second), second[0] * 1.2)
        np.testing.assert_allclose(f1, f2, rtol=1e-6, atol=1e-10)

    # Purpose: Keep positive and negative CT changes in separate channels.
    # Inputs: Half the volume increased by 0.02 and half decreased by 0.02.
    # Outputs: Both feature channels stay positive with equal normalized energy and matching feature names.
    def test_signed_channels_do_not_cancel(self):
        inputs = fixture()
        case = descriptor(*inputs)
        sibling = inputs[0].copy()
        sibling[:10] += .02
        sibling[10:] -= .02
        features, summary = extract_array(case, sibling)
        self.assertGreater(np.sum(features[:4]), 0)
        self.assertGreater(np.sum(features[4:]), 0)
        self.assertAlmostEqual(summary['normalized_positive_ct_energy'],
                               summary['normalized_negative_ct_energy'])
        self.assertEqual(len(feature_names(case)), len(features))

    # Purpose: Distinguish equal CT burdens at different longitudinal positions.
    # Inputs: Equal attenuation changes in proximal and distal five-slice slabs.
    # Outputs: Counts and total energy match but features differ in arc bins 0 and 3.
    def test_equal_extents_at_different_arc_positions_remain_distinguishable(self):
        inputs = fixture()
        case = descriptor(*inputs)
        proximal, distal = inputs[0].copy(), inputs[0].copy()
        proximal[:5] += .02
        distal[15:] += .02
        fp, sp = extract_array(case, proximal)
        fd, sd = extract_array(case, distal)
        self.assertEqual(sp['focused_ct_changed_voxels'], sd['focused_ct_changed_voxels'])
        self.assertAlmostEqual(sp['normalized_positive_ct_energy'], sd['normalized_positive_ct_energy'])
        self.assertGreater(np.linalg.norm(fp - fd), 0)
        self.assertGreater(fp[0], 0)
        self.assertGreater(fd[3], 0)

    # Purpose: Restrict descriptors to the fixed thoracic crop and report coverage.
    # Inputs: Uniform CT change with a selection limited to half the volume.
    # Outputs: Voxel and energy coverage are 0.5 and excluded arc bins are zero.
    def test_fixed_thoracic_crop_excludes_context_and_reports_coverage(self):
        inputs = list(fixture())
        inputs[4] = dict(inputs[4], geometry={'crop_low_ijk': [0, 0, 0],
                                           'crop_high_ijk_exclusive': [20, 20, 10]})
        case = descriptor(*inputs)
        features, summary = extract_array(case, inputs[0] + .02)
        self.assertEqual(summary['ct_delta_voxel_coverage_fraction'], .5)
        self.assertAlmostEqual(summary['ct_delta_energy_coverage_fraction'], .5)
        self.assertEqual(np.sum(features[2:4]), 0)

    # Purpose: Verify CT noise tolerance and finite-value validation.
    # Inputs: Baseline changed by 1e-7, then one NaN voxel.
    # Outputs: Noise produces zero features and near-baseline true; NaN raises ValueError.
    def test_noise_tolerance_and_nonfinite_validation(self):
        inputs = fixture()
        case = descriptor(*inputs)
        features, summary = extract_array(case, inputs[0] + np.float32(1e-7))
        self.assertEqual(np.sum(features), 0)
        self.assertTrue(summary['ct_near_baseline'])
        bad = inputs[0].copy()
        bad[0, 0, 0] = np.nan
        with self.assertRaisesRegex(ValueError, 'finite'):
            extract_array(case, bad)

    # Purpose: Verify native unit conversion and prevent edit metadata leakage.
    # Inputs: Hashed source/variant NPZs with per-pixel baseline and deliberately misleading edit metadata.
    # Outputs: Baseline converts correctly, metadata changes leave features equal, and bad variant hashes fail.
    def test_native_unit_conversion_checksums_and_edit_metadata_cannot_leak(self):
        image, target, geometry, config, selection = fixture()
        with tempfile.TemporaryDirectory() as folder:
            source = Path(folder) / 'source.npz'
            variant = Path(folder) / 'variant.npz'
            np.savez(source, original_attenuation_per_pixel=image / 10,
                     target_mask=target, geometry_json=json.dumps(geometry))
            np.savez(variant, image=image + .02,
                     edited_labels=np.full_like(image, 12345),
                     changed_mask=np.ones_like(image), metadata_json='untrusted ignored text')
            row = {'source_npz': str(source), 'source_npz_sha256': digest(source),
                'variant_npz': str(variant), 'variant_npz_sha256': digest(variant),
                'edit_crop_origin_ijk': [0, 0, 0], 'edit_crop_shape_kji': [20] * 3,
                'spacing_ijk_mm': [1.] * 3, 'source_atn_units': 'per_pixel',
                'image_units': 'cm^-1', 'parameters': {'coa_narrow.distance_mm': 99},
                'source_ids': [2922], 'mask_counts': {'added': 999999}}
            case = prepare_case_geometry(row, config, selection, arc_bins=4,
                radial_bins=1, angular_bins=1, radial_extent=100)
            features, summary = extract_variant(case, row)
            changed_row = deepcopy(row)
            changed_row.update(parameters={}, source_ids=[1], mask_counts={'added': 0})
            again, _ = extract_variant(case, changed_row)
            np.testing.assert_array_equal(features, again)
            np.testing.assert_allclose(case.baseline_image_cm_inverse, image, rtol=1e-6)
            self.assertEqual(summary['native_ct_changed_voxels'], image.size)
            changed_row['variant_npz_sha256'] = '0' * 64
            with self.assertRaisesRegex(ValueError, 'checksum'):
                extract_variant(case, changed_row)

    # Purpose: Reject mismatched pair geometry and invalid reference radius.
    # Inputs: Disagreeing ROI centerline and a zero median reference radius.
    # Outputs: Descriptor preparation raises ValueError for disagreement or nonpositive radius.
    def test_rejects_wrong_pair_geometry_and_missing_reference(self):
        inputs = list(fixture())
        inputs[3] = {'roi': {'center_ijk': [[1, 1, 1], [1, 1, 19]]}}
        with self.assertRaisesRegex(ValueError, 'disagree'):
            descriptor(*inputs)
        inputs = list(fixture())
        inputs[4] = {'aorta_median_ring_radius_mm': 0}
        with self.assertRaisesRegex(ValueError, 'positive'):
            descriptor(*inputs)


if __name__ == '__main__':
    unittest.main()
