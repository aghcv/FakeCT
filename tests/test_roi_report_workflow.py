"""End-to-end tissue-only tube selection and portable report on a tiny phantom."""
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


class RoiReportWorkflowTests(unittest.TestCase):
    def test_tissue_only_tube_isolates_one_of_two_nearby_arteries(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            case = root / '001'
            case.mkdir()
            original = np.zeros((13, 13, 13), dtype='<f4')
            original[2:11, 6, 6] = -7
            original[2:11, 6, 10] = -7  # Same original ID: geometry must discriminate.
            original.tofile(case / '001_act_1.bin')
            np.where(original != 0, .018, .014).astype('<f4').tofile(case / '001_atn_1.bin')
            source_before = (case / '001_act_1.bin').read_bytes()
            par, log = root / '001.par', case / '001_log'
            par.write_text('fixture geometry')
            log.write_text('fixture geometry log')
            audit = {'cases': [{'case_id': '001', 'directory': str(case), 'shape_kji': [13,13,13],
                                'spacing_ijk_mm': [1,1,1], 'par_path': str(par), 'log_path': str(log)}],
                     'source_metadata_sha256': {str(p): hashlib.sha256(p.read_bytes()).hexdigest() for p in [par, log]}}
            (root/'audit.json').write_text(json.dumps(audit))
            catalog = {'policy_version': 'test-policy', 'sources': {},
                       'categories': [{'id': 0, 'name': 'background'}, {'id': 5, 'name': 'artery'}, {'id': 255, 'name': 'unknown'}],
                       'records': [{'original_id': label, 'original_name': name,
                                    'classification': {'tissue_id': code, 'tissue_name': group}}
                                   for label,name,code,group in [(0,'background',0,'background'),(-7,'two parallel arteries',5,'artery')]]}
            (root/'catalog.json').write_text(json.dumps(catalog))
            config = root/'run.ini'
            config.write_text(f'''[study]
schema_version = fakect.preview/2
name = parallel-arteries
[input]
root = {root}
case_id = 001
frame = 1
audit = {root/'audit.json'}
catalog = {root/'catalog.json'}
[selection]
tissue = artery
[roi]
shape = tube
center_ijk = 6, 6, 3 ; 6, 6, 9
radius_mm = 1.2, 1.2
crop_half_width_mm = 4
coordinate_reviewed = false
[preview]
slice_ijk = roi
overlay_opacity = 0.22
volume_opacity = 0.25
context_tissues = artery
context_opacity = 0.08
volume_stride = 1
[output]
directory = {root/'output'}
''')
            run(config)
            output = root/'output'
            report = json.loads((output/'preview-report.json').read_text())
            self.assertEqual(report['selection']['candidate_voxels'], 18)
            self.assertEqual(report['selection']['selected_voxels'], 9)
            self.assertEqual(report['selection']['component_count_6'], 1)
            self.assertEqual(report['selection']['selected_original_ids'], [-7])
            self.assertEqual(report['geometry']['roi_kind'], 'tube')
            self.assertEqual(report['config']['selection']['source_ids'], [])
            self.assertIn('global_view', report)
            self.assertTrue((output/'roi-global.html').is_file())
            self.assertTrue((output/'roi-global.png').is_file())
            self.assertEqual((case/'001_act_1.bin').read_bytes(), source_before)
            with np.load(output/'crop.npz', allow_pickle=False) as crop:
                np.testing.assert_array_equal(crop['selected_mask'], crop['candidate_mask'] & crop['roi_mask'])
                self.assertEqual(int(crop['selected_mask'].sum()), 9)
                self.assertTrue(np.all(crop['original_labels'][crop['selected_mask']] == -7))
            html = (output/'report.html').read_text()
            self.assertIn('srcdoc=', html)
            self.assertIn('data:image/png;base64,', html)
            self.assertIn('parallel-arteries', html)
            self.assertIn('panel-global', html)
            self.assertIn('panel-local', html)
            copied = root/'portable.html'
            copied.write_text(html)
            self.assertEqual(copied.read_bytes(), (output/'report.html').read_bytes())
            hashes = json.loads((output/'artifact-manifest.json').read_text())
            for name, expected in hashes.items():
                self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(), expected)
            with self.assertRaisesRegex(ValueError, 'Output exists'):
                run(config)


if __name__ == '__main__':
    unittest.main()
