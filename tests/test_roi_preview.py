"""Native-coordinate and bounded-reader checks on small signed-label phantoms."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_roi import crop_bounds, prepare_crop, read_crop, resolve_preview, sphere_mask


class RoiPreviewTests(unittest.TestCase):
    def test_anisotropic_sphere_native_coordinates(self):
        mask = sphere_mask((7, 7, 7), (10, 20, 30), (13, 23, 33), (1, 2, 3), 3)
        self.assertTrue(mask[3, 3, 6])  # Three i voxels = 3 mm.
        self.assertTrue(mask[4, 3, 3])  # One k voxel = 3 mm.
        self.assertFalse(mask[3, 5, 3])  # Two j voxels = 4 mm.
        self.assertFalse(mask[4, 3, 4])
        expected = np.zeros_like(mask)
        for k, j, i in np.ndindex(mask.shape):
            expected[k, j, i] = (i-3)**2 + ((j-3)*2)**2 + ((k-3)*3)**2 <= 9
        np.testing.assert_array_equal(mask, expected)

    def test_crop_bounds_clipping_spacing_and_limits(self):
        self.assertEqual(crop_bounds((0, 5, 6), 4, (20, 20, 20), (1, 2, 4)), ((0, 3, 5), (5, 8, 8)))
        with self.assertRaises(ValueError):
            crop_bounds((20, 1, 1), 4, (20, 20, 20), (1, 1, 1))
        with self.assertRaises(ValueError):
            crop_bounds((300, 300, 300), 300, (700, 700, 700), (1, 1, 1))

    def test_crop_reads_exact_native_signed_values(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / 'labels.bin'
            original = (np.arange(7*8*9, dtype=np.float32).reshape(7, 8, 9) - 300)
            original.tofile(path)
            before = path.read_bytes()
            crop, meta = read_crop(path, original.shape, (2, 3, 1), (6, 7, 5))
            np.testing.assert_array_equal(crop, original[1:5, 3:7, 2:6])
            self.assertEqual(meta['source_bytes_read'], 4*8*9*4)
            self.assertEqual(path.read_bytes(), before)
            with self.assertRaises(ValueError):
                read_crop(path, original.shape, (0, 0, 0), (10, 8, 7))

    def fixture(self, root):
        directory = root / '001'
        directory.mkdir()
        label = np.zeros((9, 9, 9), dtype='<f4')
        label[:, 4, 4] = -7
        label.tofile(directory / '001_act_1.bin')
        (label * 0 + .018).astype('<f4').tofile(directory / '001_atn_1.bin')
        par, log = root/'001.par', directory/'001_log'
        par.write_text('test parameter metadata')
        log.write_text('test log metadata')
        catalog = {'categories': [{'id': 0, 'name': 'background'}, {'id': 5, 'name': 'artery'}, {'id': 255, 'name': 'unknown'}],
                   'records': [{'original_id': key, 'original_name': name,
                                'classification': {'tissue_id': code, 'tissue_name': group}}
                               for key, name, code, group in [(0, 'background', 0, 'background'), (-7, 'test_artery', 5, 'artery')]]}
        catalog_path = root/'catalog.json'
        catalog_path.write_text(json.dumps(catalog))
        audit = {'cases': [{'case_id': '001', 'directory': str(directory), 'shape_kji': [9, 9, 9],
                            'spacing_ijk_mm': [1, 1, 2], 'par_path': str(par), 'log_path': str(log)}],
                 'source_metadata_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [par, log]}}
        audit_path = root/'audit.json'
        audit_path.write_text(json.dumps(audit))
        config = {'input': {'root': root, 'case_id': '001', 'frame': 1, 'audit': audit_path, 'catalog': catalog_path},
                  'selection': {'tissue': 'artery', 'source_ids': (-7,)},
                  'roi': {'center_ijk': (4, 4, 4), 'radius_mm': 2, 'crop_half_width_mm': 3},
                  'preview': {'slice_ijk': None, 'volume_stride': 1, 'context_tissues': ()}}
        return config, label

    def test_resolution_selection_and_crop_mask(self):
        with tempfile.TemporaryDirectory() as directory:
            config, source = self.fixture(Path(directory))
            resolved = resolve_preview(config)
            arrays, _ = prepare_crop(resolved, config)
            self.assertEqual(resolved['source_ids'], (-7,))
            self.assertEqual(int((arrays['selected'] & arrays['roi']).sum()), 3)
            self.assertEqual(int(arrays['selected'].sum()), 5)
            self.assertTrue(np.all(arrays['tissue'][arrays['selected']] == 5))
            np.testing.assert_array_equal(arrays['act'], source[2:7, 1:8, 1:8])
            np.testing.assert_allclose(arrays['attenuation_cm_inverse'], .18, rtol=1e-6)
            config['selection']['source_ids'] = ()
            self.assertEqual(resolve_preview(config)['source_ids'], (-7,))

    def test_reject_wrong_group_and_changed_metadata(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.fixture(Path(directory))
            config['selection']['source_ids'] = (0,)
            with self.assertRaisesRegex(ValueError, 'selected tissue'):
                resolve_preview(config)
            config['selection']['source_ids'] = (-7,)
            (Path(directory)/'001.par').write_text('changed')
            with self.assertRaisesRegex(ValueError, 'changed audited metadata'):
                resolve_preview(config)

    def test_catalog_provenance_matches_loaded_snapshot(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.fixture(Path(directory))
            resolved = resolve_preview(config)
            old_bytes = config['input']['catalog'].read_bytes()
            config['input']['catalog'].write_text('{}')
            self.assertEqual(resolved['catalog_bytes'], old_bytes)
            self.assertEqual(resolved['catalog_sha256'], hashlib.sha256(old_bytes).hexdigest())
            self.assertEqual(resolved['catalog']['records'][1]['original_name'], 'test_artery')

    def test_reject_slice_outside_crop_and_too_coarse_volume(self):
        with tempfile.TemporaryDirectory() as directory:
            config, _ = self.fixture(Path(directory))
            config['preview']['slice_ijk'] = (0, 0, 0)
            with self.assertRaisesRegex(ValueError, 'outside the preview crop'):
                resolve_preview(config)
            config['preview']['slice_ijk'] = None
            config['preview']['volume_stride'] = 10
            with self.assertRaisesRegex(ValueError, 'fewer than two blocks'):
                resolve_preview(config)


if __name__ == '__main__':
    unittest.main()
