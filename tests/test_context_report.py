import sys
import json
import tempfile
from unittest.mock import patch
from pathlib import Path
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import fakect_context_report as report
from fakect_context_report import _baseline_rows, _plane_mask, _slice_specs


class ContextReportTests(unittest.TestCase):
    def test_baselines_required_once_per_case(self):
        rows = [{'case_id': 'a', 'baseline': True}, {'case_id': 'a', 'baseline': False},
                {'case_id': 'b', 'baseline': True}]
        self.assertEqual(len(_baseline_rows({'samples': rows})), 2)
        rows.append({'case_id': 'b', 'baseline': True})
        with self.assertRaisesRegex(ValueError, 'Exactly one'):
            _baseline_rows({'samples': rows})

    def test_native_slice_offsets_and_axes(self):
        mask = np.zeros((3, 4, 5), bool)
        mask[1, 2, 3] = True
        # Native location k,j,i = 11,22,33.
        for axis in range(3):
            low = np.array([8, 19, 29]); shape = np.array([8, 9, 10])
            low[axis] = [11, 22, 33][axis]; shape[axis] = 1
            plane = _plane_mask(mask, [10, 20, 30], low, shape, axis)
            expected = ([11, 22, 33] - low)[[x for x in range(3) if x != axis]]
            self.assertTrue(plane[tuple(expected)])
            self.assertEqual(plane.sum(), 1)
        self.assertFalse(_plane_mask(mask, [10, 20, 30], [0, 0, 0], [1, 5, 6], 0).any())

    def test_slice_indices_use_target_area_and_physical_context(self):
        mask = np.zeros((4, 5, 6), bool)
        mask[2, 1:4, 2:5] = True
        specs = _slice_specs(mask, [30, 20, 10], [100, 90, 80], [.5, 1, 2], 6)
        axial, coronal, sagittal = specs
        self.assertEqual(axial['native_index'], 12)
        self.assertEqual(axial['shape_kji'], [1, 90, 80])
        self.assertEqual(coronal['low_kji'], [7, 21, 0])
        self.assertEqual(coronal['shape_kji'], [10, 1, 80])
        self.assertEqual(sagittal['low_kji'], [7, 0, 32])
        self.assertEqual(sagittal['shape_kji'], [10, 90, 1])


    def test_complete_offline_report_has_native_slices_and_surface_controls(self):
        with tempfile.TemporaryDirectory() as folder:
            root = Path(folder)
            shape = [8, 9, 10]
            atn = root / 'atn.bin'
            np.full(shape, .02, np.float32).tofile(atn)
            source = {'path': str(atn), 'bytes': atn.stat().st_size, 'mtime_ns': atn.stat().st_mtime_ns}
            core = np.zeros((3, 3, 3), bool); core[1, 1, 1] = True
            branches = np.zeros_like(core); branches[1, 1, 2] = True
            combined = core | branches
            anatomy = root / 'mask.npz'
            np.savez_compressed(anatomy, mask=combined, core_mask=core, branch_mask=branches,
                                crop_origin_ijk=[2, 3, 2], spacing_ijk_mm=[1., 1., 1.])
            selection = root / 'selection.json'
            selection.write_text(json.dumps({'mask_npz': 'mask.npz', 'mask_npz_sha256': report.sha256(anatomy),
                                             'z_slab_mm': [2., 5.], 'branch_reach_mm': 3.}))
            variant = root / 'variant.npz'
            np.savez_compressed(variant, image=np.full((3,3,3), .2, np.float32))
            masks = root / 'pairs.npz'
            np.savez_compressed(masks, before=combined, after=combined, added=np.zeros_like(core),
                                removed=np.zeros_like(core), crop_origin_ijk=[2, 3, 2],
                                spacing_ijk_mm=[1., 1., 1.], sample_id=np.asarray('case:baseline'))
            row = {'sample_id': 'case:baseline', 'case_id': 'case', 'anatomy_family': 'family',
                   'baseline': True, 'source_atn': source, 'source_shape_kji': shape,
                   'spacing_ijk_mm': [1., 1., 1.], 'variant_npz': str(variant),
                   'variant_npz_sha256': report.sha256(variant), 'edit_crop_origin_ijk': [2, 3, 2],
                   'edit_crop_shape_kji': [3, 3, 3], 'mask_npz': str(masks),
                   'mask_npz_sha256': report.sha256(masks), 'selection_manifest': str(selection),
                   'selection_manifest_sha256': report.sha256(selection)}
            manifest = root / 'dataset-manifest.json'; manifest.write_text('{}')
            dataset = {'samples': [row], 'sample_count': 1, 'manifest_path': str(manifest),
                       'manifest_sha256': report.sha256(manifest)}
            with patch.object(report, 'inspect_dataset', return_value=dataset):
                result = report.render_report(manifest, root / 'review', context_mm=2.)
            self.assertEqual(result['case_count'], 1)
            self.assertEqual(result['omitted_surfaces'], 0)
            self.assertEqual(len(result['cases'][0]['slices']), 3)
            html = (root / 'review/index.html').read_text()
            self.assertIn('slice-image', html)
            self.assertIn('15%, 45%, or 80%', html)
            self.assertIn('select.addEventListener("change",update)', html)
            self.assertGreater((root / 'review/slices-case.png').stat().st_size, 1000)
            self.assertFalse((root / 'review/INCOMPLETE').exists())


if __name__ == '__main__':
    unittest.main()
