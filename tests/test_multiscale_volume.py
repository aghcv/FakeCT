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
from fakect_multiscale_volume import build_overview


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class MultiscaleVolumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.shape = (7, 8, 9)
        self.spacing = (.8, 1.1, 1.7)
        k, j, i = np.indices(self.shape)
        raw = (.01 + k * .001 + j * .0003 + i * .0001).astype('<f4')
        self.source = self.root / 'atn.bin'
        raw.tofile(self.source)
        self.native = raw * np.float32(10 / self.spacing[0])
        # Cross all three box boundaries and include odd terminal source cells.
        self.origin = (1, 2, 2)  # k,j,i
        self.overlay = (self.native[1:7, 2:8, 2:9] + .4).copy()
        self.variant = self.root / 'variant.npz'
        np.savez_compressed(self.variant, image=self.overlay)
        stat = self.source.stat()
        self.row = dict(sample_id='fixture:sibling', case_id='fixture',
                        source_atn=dict(path=str(self.source), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
                        source_shape_kji=list(self.shape), spacing_ijk_mm=list(self.spacing),
                        variant_npz=str(self.variant), variant_npz_sha256=digest(self.variant),
                        edit_crop_origin_ijk=list(self.origin[::-1]), edit_crop_shape_kji=list(self.overlay.shape))
        self.cache = self.root / 'cache'

    def overview(self, row=None, **kwargs):
        options = dict(shape_kji=(4, 3, 4), max_chunk_voxels=19)
        options.update(kwargs)
        return build_overview(self.row if row is None else row, self.cache, **options)

    def full_reference(self, shape, row=None, overlay=None):
        row = self.row if row is None else row
        values = self.native.copy()
        start = row['edit_crop_origin_ijk'][::-1]
        crop = self.overlay if overlay is None else overlay
        values[tuple(slice(o, o + n) for o, n in zip(start, crop.shape))] = crop
        factors = tuple((n + m - 1) // m for n, m in zip(self.shape, shape))
        result = np.zeros(shape, np.float32)
        fraction = np.zeros(shape, np.float32)
        for out in np.ndindex(shape):
            part = values[tuple(slice(o * f, min((o + 1) * f, n)) for o, f, n in zip(out, factors, self.shape))]
            if part.size:
                result[out] = part.mean(dtype=np.float64)
                fraction[out] = part.size / np.prod(factors)
        return result, fraction

    def test_streamed_overlay_matches_full_composition_anisotropic_odd_grid(self):
        result = self.overview()
        expected, fraction = self.full_reference((4, 3, 4))
        np.testing.assert_allclose(result['image'], expected, rtol=0, atol=1e-7)
        np.testing.assert_array_equal(result['valid_fraction'], fraction)
        np.testing.assert_array_equal(result['valid'], fraction > 0)
        self.assertEqual(result['image'].dtype, np.float32)
        self.assertEqual(result['valid'].dtype, np.uint8)
        self.assertEqual(result['native_factors_kji'], [2, 3, 3])
        np.testing.assert_allclose(result['spacing_ijk_mm'], [2.4, 3.3, 3.4])
        np.testing.assert_allclose(result['physical_extent_ijk_mm'], [7.2, 8.8, 11.9])
        # Final k cell contains one native voxel, with its physical centroid.
        self.assertAlmostEqual(result['center_axes_ijk_mm'][2][-1], 6.5 * 1.7)
        self.assertLessEqual(result['stats']['peak_source_chunk_voxels'], 19)
        self.assertLessEqual(result['stats']['peak_overlay_chunk_voxels'], 19)
        self.assertEqual(result['stats']['source_voxels_read'], np.prod(self.shape) + self.overlay.size)
        self.assertEqual(result['stats']['overlay_voxels_read'], self.overlay.size)
        self.assertFalse(result['stats']['base_cache_hit'])

    def test_partial_edge_cells_are_not_diluted_and_padding_is_zero(self):
        np.full(self.shape, .025, '<f4').tofile(self.source)
        stat = self.source.stat()
        self.row['source_atn'].update(bytes=stat.st_size, mtime_ns=stat.st_mtime_ns)
        self.overlay[:] = .3125
        np.savez_compressed(self.variant, image=self.overlay)
        self.row['variant_npz_sha256'] = digest(self.variant)
        result = self.overview(shape_kji=(4, 3, 12))
        occupied = result['valid'].astype(bool)
        np.testing.assert_array_equal(result['image'][occupied], .3125)
        self.assertTrue(np.any((result['valid_fraction'] > 0) & (result['valid_fraction'] < 1)))
        self.assertTrue(np.all(result['image'][..., 9:] == 0))
        self.assertTrue(np.all(result['valid'][..., 9:] == 0))
        self.assertEqual(result['occupied_shape_kji'], [4, 3, 9])

    def test_crop_boundary_filter_support_includes_unedited_voxels(self):
        result = self.overview()
        # Coarse (0,0,0) crosses every crop start face: only one of its
        # 18 native voxels is edited, so crop-only averaging would be wrong.
        expected, _ = self.full_reference((4, 3, 4))
        self.assertAlmostEqual(float(result['image'][0, 0, 0]), float(expected[0, 0, 0]), places=7)
        self.assertAlmostEqual(float(result['image'][1, 1, 1]), float(expected[1, 1, 1]), places=7)
        # (3,2,2) is also simultaneously partial in k,j and crossed by the edit.
        self.assertAlmostEqual(float(result['image'][3, 2, 2]), float(expected[3, 2, 2]), places=7)

    def test_different_chunk_sizes_and_fortran_overlay_agree(self):
        first = self.overview(max_chunk_voxels=1)
        self.assertEqual(first['stats']['peak_source_chunk_voxels'], 1)
        np.savez_compressed(self.variant, image=np.asfortranarray(self.overlay))
        self.row['variant_npz_sha256'] = digest(self.variant)
        second = self.overview(max_chunk_voxels=37)
        np.testing.assert_allclose(first['image'], second['image'], rtol=0, atol=1e-7)
        self.assertEqual(first['cache_key'], second['cache_key'])
        self.assertTrue(second['stats']['base_cache_hit'])
        self.assertEqual(second['stats']['source_voxels_read'], self.overlay.size)

    def test_base_cache_is_shared_but_composition_changes_with_overlay(self):
        original = self.overview()
        second_variant = self.root / 'variant2.npz'
        second_image = self.overlay + .9
        np.savez_compressed(second_variant, image=second_image)
        second_row = copy.deepcopy(self.row)
        second_row.update(sample_id='fixture:sibling2', variant_npz=str(second_variant),
                          variant_npz_sha256=digest(second_variant))
        second = self.overview(second_row)
        expected, _ = self.full_reference((4, 3, 4), second_row, second_image)
        np.testing.assert_allclose(second['image'], expected, rtol=0, atol=1e-7)
        self.assertTrue(second['stats']['base_cache_hit'])
        self.assertEqual(second['cache_key'], original['cache_key'])
        self.assertNotEqual(second['composition_key'], original['composition_key'])
        self.assertEqual(len(list(self.cache.glob('base-*'))), 1)
        third = self.overview()
        np.testing.assert_array_equal(third['image'], original['image'])

    def test_grid_and_source_identity_invalidate_cache(self):
        first = self.overview()
        grid = self.overview(shape_kji=(3, 3, 3))
        self.assertNotEqual(first['cache_key'], grid['cache_key'])
        os.utime(self.source, ns=(self.source.stat().st_atime_ns, self.source.stat().st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(ValueError, 'stat changed'):
            self.overview()
        self.row['source_atn']['mtime_ns'] = self.source.stat().st_mtime_ns
        fresh = self.overview()
        self.assertNotEqual(first['cache_key'], fresh['cache_key'])
        self.assertFalse(fresh['stats']['base_cache_hit'])

    def test_corrupt_cache_fails_instead_of_silently_rebuilding(self):
        result = self.overview()
        path = Path(result['cache_directory']) / 'base-sums.npy'
        data = bytearray(path.read_bytes())
        data[-1] ^= 1
        path.write_bytes(data)
        with self.assertRaisesRegex(ValueError, 'checksum changed'):
            self.overview()

    def test_invalid_geometry_and_budgets_fail_before_mapping_or_cache_creation(self):
        bad_rows = []
        for key, value in [('edit_crop_origin_ijk', [-1, 0, 0]),
                           ('edit_crop_origin_ijk', [8, 7, 6]),
                           ('source_shape_kji', [7., 8, 9]),
                           ('source_atn_dtype', '>f4'), ('image_units', 'HU')]:
            row = copy.deepcopy(self.row)
            row[key] = value
            bad_rows.append(row)
        with patch('fakect_multiscale_volume.np.memmap', side_effect=AssertionError('mapped')):
            for row in bad_rows:
                with self.subTest(row=row), self.assertRaises(ValueError):
                    self.overview(row)
            for options in [dict(shape_kji=(200, 200, 200)), dict(max_chunk_voxels=0),
                            dict(max_chunk_voxels=4_000_001), dict(verify_hashes=1)]:
                with self.subTest(options=options), self.assertRaises(ValueError):
                    self.overview(**options)
        self.assertFalse(self.cache.exists())

    def test_array_header_rejected_before_loading_or_mapping(self):
        np.savez_compressed(self.variant, image=np.ones((3, 4, 5), np.float32))
        self.row['variant_npz_sha256'] = digest(self.variant)
        with patch('fakect_multiscale_volume.np.memmap', side_effect=AssertionError('mapped')):
            with self.assertRaisesRegex(ValueError, 'shape/type'):
                self.overview()
        self.assertFalse(self.cache.exists())

    def test_checksums_and_changes_during_read_are_rejected(self):
        self.row['variant_npz_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Variant checksum'):
            self.overview()
        self.row['variant_npz_sha256'] = digest(self.variant)
        self.row['source_atn']['sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'source checksum'):
            self.overview()
        del self.row['source_atn']['sha256']
        import fakect_multiscale_volume as module
        real_read = module._read_source
        changed = False

        def changing_read(*args):
            nonlocal changed
            result = real_read(*args)
            if not changed:
                stat = self.source.stat()
                os.utime(self.source, ns=(stat.st_atime_ns, stat.st_mtime_ns + 1_000_000))
                changed = True
            return result

        with patch.object(module, '_read_source', side_effect=changing_read):
            with self.assertRaisesRegex(ValueError, 'stat changed'):
                self.overview()
        self.assertFalse(self.cache.exists())

    def test_nonfinite_edit_fails_without_loading_target_masks(self):
        self.overlay[0, 0, 0] = np.nan
        np.savez_compressed(self.variant, image=self.overlay)
        self.row['variant_npz_sha256'] = digest(self.variant)
        # A mask path is deliberately absent; the overview has no label dependency.
        with self.assertRaisesRegex(ValueError, 'nonfinite attenuation'):
            self.overview()


if __name__ == '__main__':
    unittest.main()
