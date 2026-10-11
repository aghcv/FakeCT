import copy
import hashlib
import os
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_context_volume import ContextVolume


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ContextVolumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spacing = np.array([.8, 1.1, 1.7])
        self.shape = (7, 8, 9)
        kk, jj, ii = np.indices(self.shape)
        self.native = (.1 + .02 * kk + .003 * jj + .001 * ii).astype(np.float32)
        self.source = self.root / 'source_atn.bin'
        (self.native * (self.spacing[0] / 10)).astype('<f4').tofile(self.source)
        self.variant = self.root / 'variant.npz'
        self.maskfile = self.root / 'pair.npz'
        self.edit_origin = [3, 2, 2]
        self.edit_shape = (3, 3, 3)
        self.overlay = self.native[2:5, 2:5, 3:6].copy() + .4
        self.before = np.zeros((4, 5, 6), np.uint8)
        self.before[1:3, 2:4, 2:5] = 1
        self.after = self.before.copy()
        self.after[1, 1, 2] = 1
        self.after[2, 3, 4] = 0
        self.mask_origin = [1, 1, 1]
        self.mask_data = dict(before=self.before, after=self.after,
                              added=((self.after == 1) & (self.before == 0)).astype(np.uint8),
                              removed=((self.before == 1) & (self.after == 0)).astype(np.uint8),
                              crop_origin_ijk=np.array(self.mask_origin), spacing_ijk_mm=self.spacing,
                              sample_id=np.array('fixture:edited'))
        np.savez_compressed(self.variant, image=self.overlay)
        np.savez_compressed(self.maskfile, **self.mask_data)
        stat = self.source.stat()
        self.row = dict(sample_id='fixture:edited', case_id='fixture', anatomy_family='family', baseline=False,
                        source_atn=dict(path=str(self.source), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
                        source_shape_kji=list(self.shape), spacing_ijk_mm=self.spacing.tolist(),
                        variant_npz=str(self.variant), variant_npz_sha256=sha256(self.variant),
                        edit_crop_origin_ijk=self.edit_origin, edit_crop_shape_kji=list(self.edit_shape),
                        mask_npz=str(self.maskfile), mask_npz_sha256=sha256(self.maskfile))

    def volume(self, row=None, **kw):
        volume = ContextVolume(self.row if row is None else row, **kw)
        self.addCleanup(volume.close)
        return volume

    def save_masks(self):
        np.savez_compressed(self.maskfile, **self.mask_data)
        self.row['mask_npz_sha256'] = sha256(self.maskfile)

    def expected_native(self):
        native = np.fromfile(self.source, '<f4').reshape(self.shape) * np.float32(10 / self.spacing[0])
        native[2:5, 2:5, 3:6] = self.overlay
        target = np.zeros(self.shape, np.uint8)
        target[1:5, 1:6, 1:7] = self.after
        return native, target

    def brute_block(self, origin, shape, spacing):
        native, target = self.expected_native()
        image = np.zeros(shape, np.float32)
        mask, valid = np.zeros(shape, np.uint8), np.zeros(shape, np.uint8)
        for out in np.ndindex(shape):
            p = (np.array(out) + origin + .5) * np.asarray(spacing)[::-1] / self.spacing[::-1] - .5
            if np.any(p < -.5) or np.any(p >= np.array(self.shape) - .5):
                continue
            valid[out] = 1
            near = np.floor(p + .5).astype(int)
            mask[out] = target[tuple(near)]
            p = np.clip(p, 0, np.array(self.shape) - 1)
            low = np.floor(p).astype(int)
            high = np.minimum(low + 1, np.array(self.shape) - 1)
            t = p - low
            for corners in np.ndindex((2, 2, 2)):
                q = np.where(corners, high, low)
                weight = np.prod(np.where(corners, t, 1 - t))
                image[out] += native[tuple(q)] * weight
        return dict(image=image, mask=mask, valid=valid)

    def test_native_conversion_overlay_and_mask_offsets(self):
        result = self.volume().read_block([0, 0, 0], self.shape, self.spacing)
        image, target = self.expected_native()
        np.testing.assert_allclose(result['image'], image, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(result['mask'], target)
        self.assertTrue(result['valid'].all())
        self.assertEqual(result['image'].dtype, np.float32)
        self.assertEqual(result['mask'].dtype, np.uint8)
        self.assertEqual(result['valid'].dtype, np.uint8)

    def test_anisotropic_linear_image_nearest_mask_and_padding(self):
        origin, shape, spacing = np.array([-1, -1, -2]), (12, 11, 14), [.65, .8, 1.2]
        result = self.volume().read_block(origin, shape, spacing)
        expected = self.brute_block(origin, shape, spacing)
        np.testing.assert_allclose(result['image'], expected['image'], rtol=0, atol=2e-7)
        for key in ('mask', 'valid'):
            np.testing.assert_array_equal(result[key], expected[key])

    def test_interpolation_crosses_edit_boundary_using_source_halo(self):
        # i=5 at half spacing samples native i=2.25, mixing unedited i=2
        # with edited i=3. A crop-only resample would miss this contribution.
        result = self.volume().read_block([2, 2, 5], [1, 1, 1], [.4, 1.1, 1.7])
        image, _ = self.expected_native()
        expected = .75 * image[2, 2, 2] + .25 * image[2, 2, 3]
        self.assertAlmostEqual(float(result['image'][0, 0, 0]), float(expected), places=6)

    def test_unchanged_overlay_is_same_as_whole_body_linear_interpolation(self):
        self.overlay = (np.fromfile(self.source, '<f4').reshape(self.shape)[2:5, 2:5, 3:6]
                        * np.float32(10 / self.spacing[0]))
        np.savez_compressed(self.variant, image=self.overlay)
        self.row['variant_npz_sha256'] = sha256(self.variant)
        origin, shape, spacing = [1, 1, 1], [8, 8, 10], [.5, .7, 1.1]
        result = self.volume().read_block(origin, shape, spacing)
        expected = self.brute_block(origin, tuple(shape), spacing)
        np.testing.assert_allclose(result['image'], expected['image'], rtol=0, atol=1e-7)

    def test_tiled_and_single_reads_agree_across_slab_boundaries(self):
        v = self.volume()
        origin, shape, spacing = [-1, 1, 2], [10, 7, 10], [.6, .9, 1.25]
        whole = v.read_block(origin, shape, spacing)
        first = v.read_block(origin, [4, 7, 10], spacing)
        second = v.read_block([3, 1, 2], [6, 7, 10], spacing)
        for key in whole:
            np.testing.assert_allclose(np.concatenate((first[key], second[key])), whole[key], atol=1e-7)

    def test_zero_padding_and_edges_within_physical_extent(self):
        v = self.volume()
        r = v.read_block([-50, -50, -50], [2, 2, 2], self.spacing)
        self.assertFalse(any(a.any() for a in r.values()))
        edge = v.read_block([0, 0, 0], [1, 1, 1], self.spacing / 2)
        self.assertEqual(int(edge['valid'][0, 0, 0]), 1)
        image, _ = self.expected_native()
        self.assertAlmostEqual(float(edge['image'][0, 0, 0]), float(image[0, 0, 0]))

    def test_model_shape_is_physical_ceil_without_whole_volume_budget(self):
        v = self.volume()
        self.assertEqual(v.model_shape(self.spacing), self.shape)
        self.assertEqual(v.model_shape(self.spacing / 2), tuple(2 * s for s in self.shape))
        self.assertEqual(v.model_shape([.7, 1.3, 2.1]), (6, 7, 11))
        with patch('fakect_context_volume.MAX_BLOCK_VOXELS', 1):
            self.assertEqual(v.model_shape(self.spacing), self.shape)

    def test_foreground_points_are_deterministic_and_true_on_model_grid(self):
        v = self.volume()
        spacing = [.6, .8, 1.2]
        pts = v.foreground_points(spacing, max_points=5, seed=7)
        np.testing.assert_array_equal(pts, v.foreground_points(spacing, max_points=5, seed=7))
        self.assertEqual(len(pts), 5)
        self.assertEqual(len(np.unique(pts, axis=0)), 5)
        for pt in pts:
            self.assertEqual(int(v.read_block(pt, [1, 1, 1], spacing)['mask'][0, 0, 0]), 1)

    def test_foreground_verification_matches_loader_at_sampling_ties(self):
        self.row['spacing_ijk_mm'] = [.9, .9, .9]
        self.mask_data['spacing_ijk_mm'] = np.array([.9, .9, .9])
        mask = np.zeros_like(self.after)
        mask[0, 0, 0] = 1  # Global native index (1,1,1).
        mask[2, 2, 2] = 1  # Global native index (3,3,3).
        self.mask_data.update(before=mask, after=mask,
                              added=np.zeros_like(mask), removed=np.zeros_like(mask))
        self.save_masks()
        v = self.volume()
        for spacing in ([.6, .6, .6], [1.2, 1.2, 1.2]):
            pts = v.foreground_points(spacing, max_points=100, seed=7)
            self.assertGreater(len(pts), 0)
            for pt in pts:
                block = v.read_block(pt, [1, 1, 1], spacing)
                self.assertEqual(int(block['mask'][0, 0, 0]), 1)
                self.assertEqual(int(block['valid'][0, 0, 0]), 1)
        # At .9 -> 1.2 mm, model index1 is exactly a native boundary.
        # The former multiply-then-divide check selected native1, whereas
        # the production loader selects native2, which is background here.
        pts = v.foreground_points([1.2]*3, max_points=100, seed=7)
        np.testing.assert_array_equal(pts, [[2, 2, 2]])

    def test_empty_foreground_is_explicit(self):
        for key in ('before', 'after', 'added', 'removed'):
            self.mask_data[key] = np.zeros_like(self.after)
        self.save_masks()
        self.assertEqual(self.volume().foreground_points(self.spacing).shape, (0, 3))

    def test_fails_changed_source_stat_or_geometry(self):
        row = copy.deepcopy(self.row)
        row['source_atn']['mtime_ns'] += 1
        with self.assertRaisesRegex(ValueError, 'source stat'):
            self.volume(row)
        row = copy.deepcopy(self.row)
        row['source_shape_kji'][0] += 1
        with self.assertRaisesRegex(ValueError, 'source geometry'):
            self.volume(row)
        row = copy.deepcopy(self.row)
        row['edit_crop_origin_ijk'] = [8, 0, 0]
        with self.assertRaisesRegex(ValueError, 'source bounds'):
            self.volume(row)

    def test_hashes_enforced_but_explicit_skip_keeps_geometry_checks(self):
        row = copy.deepcopy(self.row)
        row['variant_npz_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Checksum'):
            self.volume(row)
        self.volume(row, verify_hashes=False)
        row['edit_crop_shape_kji'] = [1, 1, 1]
        with self.assertRaisesRegex(ValueError, 'shape/type'):
            self.volume(row, verify_hashes=False)

    def test_rejects_bad_spacing_shifted_mask_or_binary_identity(self):
        self.mask_data['spacing_ijk_mm'] = [1, 1, 1]
        self.save_masks()
        with self.assertRaisesRegex(ValueError, 'spacing'):
            self.volume()
        self.mask_data['spacing_ijk_mm'] = self.spacing
        self.mask_data['crop_origin_ijk'] = [20, 0, 0]
        self.save_masks()
        with self.assertRaisesRegex(ValueError, 'source bounds'):
            self.volume()
        self.mask_data['crop_origin_ijk'] = self.mask_origin
        self.mask_data['added'] = np.zeros_like(self.after)
        self.save_masks()
        with self.assertRaisesRegex(ValueError, 'added/removed'):
            self.volume()
        self.mask_data['after'] = np.full_like(self.after, 2)
        self.save_masks()
        with self.assertRaisesRegex(ValueError, 'binary'):
            self.volume()

    def test_rejects_wrong_mask_sample_or_shape(self):
        self.mask_data['sample_id'] = np.array('other:edited')
        self.save_masks()
        with self.assertRaisesRegex(ValueError, 'sample ID'):
            self.volume()
        self.mask_data['sample_id'] = np.array(self.row['sample_id'])
        self.mask_data['before'] = np.zeros((1, 1, 1), np.uint8)
        self.save_masks()
        with self.assertRaisesRegex(ValueError, 'shape/type'):
            self.volume()

    def test_rejects_nonfinite_images(self):
        overlay = self.overlay.copy()
        overlay[0, 0, 0] = np.nan
        np.savez_compressed(self.variant, image=overlay)
        self.row['variant_npz_sha256'] = sha256(self.variant)
        with self.assertRaisesRegex(ValueError, 'nonfinite'):
            self.volume()

    def test_stat_changes_after_open_fail_before_reads(self):
        v = self.volume()
        stat = self.source.stat()
        os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1000000))
        with self.assertRaisesRegex(ValueError, 'stat changed'):
            v.read_block([0, 0, 0], [1, 1, 1], self.spacing)

    def test_input_files_rows_and_arrays_remain_immutable(self):
        original = copy.deepcopy(self.row)
        hashes = [sha256(p) for p in (self.source, self.variant, self.maskfile)]
        v = self.volume()
        r = v.read_block([0, 0, 0], self.shape, self.spacing)
        r['image'][:] = 0
        r['mask'][:] = 0
        self.assertTrue(v.read_block([0, 0, 0], self.shape, self.spacing)['image'].any())
        self.assertEqual(hashes, [sha256(p) for p in (self.source, self.variant, self.maskfile)])
        self.assertEqual(self.row, original)
        with self.assertRaises(ValueError):
            v.mask_after[:] = 0
        self.assertIsInstance(v._base, np.memmap)

    def test_voxel_budgets_and_invalid_requests_fail_explicitly(self):
        v = self.volume()
        with patch('fakect_context_volume.MAX_BLOCK_VOXELS', 1):
            with self.assertRaisesRegex(ValueError, 'Output block'):
                v.read_block([0, 0, 0], [2, 2, 2], self.spacing)
        with patch('fakect_context_volume.MAX_SOURCE_SLAB_VOXELS', 1):
            with self.assertRaisesRegex(ValueError, 'Native source slab'):
                v.read_block([0, 0, 0], [2, 2, 2], self.spacing)
        with patch('fakect_context_volume.MAX_CROP_VOXELS', 1):
            with self.assertRaisesRegex(ValueError, 'Edit crop'):
                self.volume()
        for bad in ([0, 1, 1], [np.nan, 1, 1], [-1, 1, 1]):
            with self.assertRaises(ValueError):
                v.model_shape(bad)
        with self.assertRaises(ValueError):
            v.read_block([.5, 0, 0], [1, 1, 1], self.spacing)


if __name__ == '__main__':
    unittest.main()
