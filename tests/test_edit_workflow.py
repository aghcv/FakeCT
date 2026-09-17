"""Exercise the same INI command through geometry, reassignment and HTML export."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
from preview_roi import run


class EditWorkflowTests(unittest.TestCase):
    def test_same_command_exports_reassigned_labels_and_before_after_report(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            case = root / '001'
            case.mkdir()
            original = np.full((45, 35, 35), 10, dtype='<f4')
            original[7:38, 15:20, 15:20] = -7
            scalar = np.where(original == -7, .023, .014).astype('<f4')
            original.tofile(case / '001_act_1.bin')
            scalar.tofile(case / '001_atn_1.bin')
            source_hashes = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in case.iterdir()}
            par, log = root / '001.par', case / '001_log'
            par.write_text('test geometry'); log.write_text('test geometry')
            audit = {'cases': [{'case_id': '001', 'directory': str(case), 'shape_kji': list(original.shape),
                               'spacing_ijk_mm': [1, 1, 1], 'par_path': str(par), 'log_path': str(log)}],
                     'source_metadata_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in (par, log)}}
            (root / 'audit.json').write_text(json.dumps(audit))
            catalog = {'policy_version': 'test', 'sources': {},
                       'categories': [{'id': 0, 'name': 'background'}, {'id': 1, 'name': 'soft_tissue'},
                                      {'id': 5, 'name': 'artery'}, {'id': 255, 'name': 'unknown'}],
                       'records': [{'original_id': i, 'original_name': name,
                                    'classification': {'tissue_id': code, 'tissue_name': name}}
                                   for i, name, code in [(0, 'background', 0), (10, 'soft_tissue', 1), (-7, 'artery', 5)]]}
            (root / 'catalog.json').write_text(json.dumps(catalog))
            config = root / 'edit.ini'
            config.write_text(f'''[study]
schema_version = fakect.edit/1
name = erosion-workflow
[input]
root = {root}
case_id = 001
frame = 1
audit = {root / 'audit.json'}
catalog = {root / 'catalog.json'}
[selection]
tissue = artery
source_ids =
[roi]
shape = tube
center_ijk = 17,17,16 ; 17,17,28
radius_mm = 6,6
crop_half_width_mm = 14
coordinate_reviewed = false
[preview]
slice_ijk = roi
overlay_opacity = 0.22
volume_opacity = 0.25
context_tissues = soft_tissue
context_opacity = 0.08
volume_stride = 2
[edit]
operation = erosion
distance_mm = 1
profile = uniform
profile_axis = tube
shape_k = 10
shape_window = 0.25,0.75
[reassignment]
allowed_tissues = soft_tissue
max_distance_mm = 2
unresolved = preserve
[output]
directory = {root / 'output'}
''')
            run(config, validate_only=True)
            self.assertFalse((root / 'output').exists())
            run(config)
            out = root / 'output'
            report = json.loads((out / 'preview-report.json').read_text())
            self.assertTrue(report['geometry_edited'])
            self.assertFalse(report['source_volumes_modified'])
            self.assertFalse(report['scalar_recovery_performed'])
            self.assertEqual(report['schema_version'], 'fakect.roi-preview/5')
            counts = report['edit']['counts']
            self.assertGreater(counts['removed'], 0)
            self.assertEqual(counts['added'], 0)
            self.assertEqual(counts['unresolved'], 0)
            with np.load(out / 'edit.npz', allow_pickle=False) as edit:
                np.testing.assert_array_equal(edit['changed_mask'], edit['removed_mask'])
                self.assertFalse(edit['changed_mask'][~edit['roi_mask']].any())
                self.assertTrue((edit['edited_labels'][edit['removed_mask']] == 10).all())
                np.testing.assert_array_equal(edit['original_attenuation_per_pixel'][~edit['changed_mask']],
                                              edit['attenuation_proxy_per_pixel'][~edit['changed_mask']])
                self.assertTrue(np.allclose(edit['attenuation_proxy_per_pixel'][edit['removed_mask']], .014))
                self.assertTrue((edit['original_labels'][edit['removed_mask']] == -7).all())
            for name, expected in source_hashes.items():
                self.assertEqual(hashlib.sha256((case / name).read_bytes()).hexdigest(), expected)
            for name, expected in json.loads((out / 'artifact-manifest.json').read_text()).items():
                self.assertEqual(hashlib.sha256((out / name).read_bytes()).hexdigest(), expected, name)
            self.assertTrue((out / 'after/roi-volume.html').is_file())
            self.assertTrue((out / 'edit-comparison.png').read_bytes().startswith(b'\x89PNG'))
            html = (out / 'report.html').read_text()
            self.assertIn('Morphology trial', html)
            self.assertGreaterEqual(html.count('srcdoc='), 2)
            self.assertIn('erosion', html)


if __name__ == '__main__':
    unittest.main()
