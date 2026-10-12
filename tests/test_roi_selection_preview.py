"""Portable, physical-coordinate contracts for the ROI workbench."""
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from fakect_config import load_preview_config
from fakect_roi import resolve_preview
from fakect_roi_selection_preview import control_frames, sample_section
from preview_roi import run


def tiny_phantom(root):
    """A redistributable phantom with signed labels and two neighboring tubes."""
    case = root/'001'
    case.mkdir()
    labels = np.zeros((13,13,13),dtype='<f4')
    labels[2:11,6,6] = -7
    labels[2:11,6,10] = -7
    labels.tofile(case/'001_act_1.bin')
    np.where(labels != 0,.018,.014).astype('<f4').tofile(case/'001_atn_1.bin')
    par, log = root/'001.par', case/'001_log'
    par.write_text('Generated test geometry')
    log.write_text('Generated test geometry log')
    audit = {'cases':[{'case_id':'001','directory':str(case),'shape_kji':[13,13,13],
                       'spacing_ijk_mm':[1,1,1],'par_path':str(par),'log_path':str(log)}],
             'source_metadata_sha256':{str(path):hashlib.sha256(path.read_bytes()).hexdigest() for path in (par,log)}}
    catalog = {'policy_version':'test-policy','sources':{},
               'categories':[{'id':0,'name':'background'},{'id':5,'name':'artery'},{'id':255,'name':'unknown'}],
               'records':[{'original_id':label,'original_name':name,
                           'classification':{'tissue_id':code,'tissue_name':group}}
                          for label,name,code,group in [(0,'background',0,'background'),(-7,'two arteries',5,'artery')]]}
    (root/'audit.json').write_text(json.dumps(audit))
    (root/'catalog.json').write_text(json.dumps(catalog))
    config = root/'run.ini'
    config.write_text(f'''[study]
schema_version = fakect.preview/2
name = roi-tube
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
center_ijk = 6.25, 6, 3 ; 6.25, 6, 9
radius_mm = 1.2, 1.8
crop_half_width_mm = 4
coordinate_reviewed = false
[preview]
mode = roi-selection
content = labels
slice_ijk = 0,0,0
overlay_opacity = 0.22
volume_opacity = 0.25
context_tissues = artery
context_opacity = 0.08
volume_stride = 1
[output]
directory = {root/'output'}
''')
    return config


class ControlGeometryTests(unittest.TestCase):
    # Purpose: Verify control frames use physical coordinates and orthonormal section bases.
    # Inputs: Three fractional controls with 2/3/4 mm anisotropic spacing.
    # Outputs: Exact physical centers/distances, expected tangent, and orthonormal tangent/u/v bases.
    def test_anisotropic_tangent_arc_length_and_exact_controls(self):
        nodes = np.array([[2.5,3,1],[3.5,3,2],[3.5,4,2]])
        frame = control_frames(nodes,[2,3,4])
        np.testing.assert_allclose(frame['centers_mm'],nodes*[2,3,4])
        np.testing.assert_allclose(frame['distance_mm'],[0,np.sqrt(20),np.sqrt(20)+3])
        np.testing.assert_allclose(frame['tangent'][0],[2/np.sqrt(20),0,4/np.sqrt(20)])
        for tangent,u,v in zip(frame['tangent'],frame['u_basis'],frame['v_basis']):
            np.testing.assert_allclose([np.linalg.norm(tangent),np.linalg.norm(u),np.linalg.norm(v)],[1,1,1])
            np.testing.assert_allclose([np.dot(tangent,u),np.dot(tangent,v),np.dot(u,v)],[0,0,0],atol=1e-12)

    # Purpose: Verify reversing paths and sphere controls have explicit frame conventions.
    # Inputs: Backtracking three-point path, one fractional sphere point, and duplicate controls.
    # Outputs: Reversal/sphere warnings, expected tangent directions, and duplicate-point rejection.
    def test_reversal_is_explicit_and_sphere_supported(self):
        frame = control_frames([[2,2,2],[3,2,2],[2,2,2]],[1,1,1])
        self.assertIn('reverses',frame['warnings'][1])
        np.testing.assert_array_equal(frame['tangent'][1],[-1,0,0])
        sphere = control_frames([[2.5,3,4]],[1,2,3])
        np.testing.assert_array_equal(sphere['tangent'],[[0,0,1]])
        self.assertIn('Sphere',sphere['warnings'][0])
        with self.assertRaises(ValueError):
            control_frames([[1,1,1],[1,1,1]],[1,1,1])

    # Purpose: Verify fractional sections interpolate signals physically and preserve discrete labels.
    # Inputs: Anisotropic offset affine CT ramp with signed labels -17/83 and 0.4 mm section pixels.
    # Outputs: Exact affine signal, all-valid section, and int32 labels limited to original IDs.
    def test_physical_linear_ct_and_nearest_labels_on_fractional_section(self):
        k,j,i = np.indices((7,8,9))
        spacing = np.array([2,3,4])
        origin = np.array([10,20,30])
        # Physical affine ramp: linear interpolation must be exact.
        ct = 2*(i+origin[0]) + 6*(j+origin[1]) + 12*(k+origin[2])
        labels = np.where(i < 4,-17,83).astype(np.int32)
        center = (origin+[3.5,3.25,2.5])*spacing
        sampled,valid,offsets = sample_section({'attenuation':ct,'label':labels},center,[1,0,0],[0,1,0],
                                              spacing,origin,half_width_mm=2,pixel_mm=.4)
        xx,yy = np.meshgrid(offsets,offsets)
        np.testing.assert_allclose(sampled['attenuation'],center[0]+xx+2*(center[1]+yy)+3*center[2])
        self.assertTrue(valid.all())
        self.assertEqual(set(np.unique(sampled['label'])),{-17,83})
        self.assertEqual(sampled['label'].dtype,np.int32)

    # Purpose: Verify section sampling bounds resolution and marks outside-volume pixels.
    # Inputs: 3-cubed Boolean ROI sampled over 4 mm half-width at requested 0.001 mm pixels.
    # Outputs: 321x321 plane cap, invalid pixels zeroed, Boolean labels, and centered offset.
    def test_invalid_pixels_are_marked_and_sampling_is_bounded(self):
        ones = np.ones((3,3,3),dtype=bool)
        sampled,valid,offsets = sample_section({'roi':ones},[0,0,0],[1,0,0],[0,1,0],[1,1,1],[0,0,0],
                                              half_width_mm=4,pixel_mm=.001)
        self.assertEqual(valid.shape,(321,321))
        self.assertFalse(valid.all())
        self.assertTrue(np.all(sampled['roi'][~valid] == 0))
        self.assertEqual(sampled['roi'].dtype,np.bool_)
        self.assertEqual(offsets[160],0)


class RoiSelectionWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.config = tiny_phantom(self.root)

    # Purpose: Verify ROI-workbench mode/content settings are strict with compatible defaults.
    # Inputs: Valid labels ROI-selection INI, typo values/key, then removed mode/content entries.
    # Outputs: Valid explicit settings, invalid entries rejected, and default full/attenuation mode.
    def test_strict_mode_content_and_compatible_defaults(self):
        text = self.config.read_text()
        config = load_preview_config(self.config)
        self.assertEqual(config['preview']['mode'],'roi-selection')
        self.assertEqual(config['preview']['content'],'labels')
        for old,new in [('roi-selection','typo'),('content = labels','content = act'),('mode = roi-selection','modee = roi-selection')]:
            self.config.write_text(text.replace(old,new))
            with self.assertRaises(ValueError):
                load_preview_config(self.config)
        self.config.write_text(text.replace('mode = roi-selection\n','').replace('content = labels\n',''))
        defaults = load_preview_config(self.config)
        self.assertEqual(defaults['preview']['mode'],'full')
        self.assertEqual(defaults['preview']['content'],'attenuation')

    # Purpose: Verify workbench-only resolution bypasses unused 3D checks but validates native files.
    # Inputs: Display cap 1, stale slice, full-mode comparison, then corrupted ACT payload.
    # Outputs: Control-derived slice, full-mode budget rejection, and workbench source-size rejection.
    def test_selection_ignores_unused_3d_budget_and_stale_slice_but_keeps_source_checks(self):
        config = load_preview_config(self.config)
        with patch('fakect_roi.MAX_DISPLAY_CELLS',1):
            resolved = resolve_preview(config)
            self.assertEqual(resolved['slice_ijk'],(6,6,9))
            config['preview']['mode'] = 'full'
            with self.assertRaisesRegex(ValueError,'3D preview exceeds'):
                resolve_preview(config)
        config['preview']['mode'] = 'roi-selection'
        (self.root/'001'/'001_act_1.bin').write_bytes(b'bad')
        with self.assertRaisesRegex(ValueError,'source size'):
            resolve_preview(config)

    # Purpose: Verify ROI-selection workflow exports every control without changing sources.
    # Inputs: 13-cubed signed-label phantom, two fractional tube controls, and forbidden 3D render mocks.
    # Outputs: Portable four-image report, nine selected voxels, preserved source/input bytes, hashes, and no overwrite.
    def test_end_to_end_simple_portable_report_source_safety_and_every_control(self):
        original_input = self.config.read_bytes()
        files = [self.root/'001'/f'001_{channel}_1.bin' for channel in ('act','atn')]
        originals = {path:path.read_bytes() for path in files}
        with patch('fakect_global_preview.render_global_preview',side_effect=AssertionError('global called')), \
             patch('fakect_volume_preview.render_volume_preview',side_effect=AssertionError('3D called')):
            report = run(self.config)
        output = self.root/'output'
        self.assertFalse(report['geometry_edited'])
        self.assertFalse(report['source_volumes_modified'])
        points = report['roi_selection']['rois'][0]['points']
        self.assertEqual(len(points),2)
        self.assertEqual(points[0]['center_ijk'],[6.25,6,3])
        self.assertEqual(points[1]['radius_mm'],1.8)
        self.assertEqual(points[1]['distance_mm'],6)
        self.assertEqual(len(list((output/'control-points').glob('*.png'))),4)
        self.assertEqual((output/'input.ini').read_bytes(),original_input)
        html = (output/'report.html').read_text()
        self.assertIn('data:image/png;base64,',html)
        self.assertIn('data-point="1"',html)
        self.assertIn('value="labels" selected',html)
        self.assertIn('content="default-src',html)
        self.assertNotIn('Plotly',html)
        self.assertNotIn('Training target',html)
        self.assertFalse((output/'roi-global.html').exists())
        self.assertFalse((output/'INCOMPLETE').exists())
        saved = json.loads((output/'preview-report.json').read_text())
        self.assertEqual(saved['preview_mode'],'roi-selection')
        for path,original in originals.items():
            self.assertEqual(path.read_bytes(),original)
        with np.load(output/'crop.npz',allow_pickle=False) as crop:
            np.testing.assert_array_equal(crop['selected_mask'],crop['candidate_mask'] & crop['roi_mask'])
            self.assertEqual(int(crop['selected_mask'].sum()),9)
        for name,expected in json.loads((output/'artifact-manifest.json').read_text()).items():
            self.assertEqual(hashlib.sha256((output/name).read_bytes()).hexdigest(),expected)
        with self.assertRaisesRegex(ValueError,'Output exists'):
            run(self.config)

    # Purpose: Verify ROI-selection review does not execute configured morphology.
    # Inputs: Edit INI with 1000 mm dilation/search and forbidden edit/halo mocks.
    # Outputs: No edited geometry or edit summary and no morphology/halo invocation.
    def test_selection_does_not_execute_or_require_edit_halo(self):
        text = self.config.read_text().replace('fakect.preview/2','fakect.edit/1')
        text += '''\n[edit]
operation = dilation
distance_mm = 1000
profile = uniform
profile_axis = tube
shape_k = 6
shape_window = 0,1
[reassignment]
allowed_tissues =
max_distance_mm = 1000
unresolved = error
'''
        self.config.write_text(text)
        with patch('fakect_morphology.apply_morphology',side_effect=AssertionError('edit called')), \
             patch('fakect_morphology.validate_edit_geometry',side_effect=AssertionError('edit halo called')):
            report = run(self.config)
        self.assertFalse(report['geometry_edited'])
        self.assertNotIn('edit',report)

    # Purpose: Verify named sphere selectors retain their exact control coordinates.
    # Inputs: Main tube plus local sphere centered at 6.5/6.25/6.5 with radius 1.5.
    # Outputs: Main/local cards, exact sphere center, zero arc distance, sphere warning, and attenuation image.
    def test_named_sphere_gets_its_own_card_and_exact_coordinates(self):
        config = load_preview_config(self.config)
        config['rois'] = {'local':{'shape':'sphere','center_ijk':(6.5,6.25,6.5),
                                  'radius_mm':1.5,'coordinate_reviewed':False}}
        report = run(self.config,config_override=config)
        rois = report['roi_selection']['rois']
        self.assertEqual([roi['name'] for roi in rois],['main','local'])
        point = rois[1]['points'][0]
        self.assertEqual(point['center_ijk'],[6.5,6.25,6.5])
        self.assertEqual(point['distance_mm'],0)
        self.assertIn('Sphere',point['warning'])
        self.assertTrue((self.root/'output'/point['images']['attenuation']['path']).exists())


if __name__ == '__main__':
    unittest.main()
