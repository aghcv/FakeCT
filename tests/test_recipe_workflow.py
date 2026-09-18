"""Exercise recipe orchestration, frozen pass states and portable HTML together."""
import base64
import configparser
from contextlib import redirect_stdout
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np
import test_study_workflow as study_fixtures

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_config import load_preview_config
from fakect_recipe import recipe_masks
from fakect_roi import prepare_crop, resolve_preview


class _Document(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.images = []
        self.frames = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'img':
            self.images.append(attrs['src'])
        if tag == 'iframe':
            self.frames.append(attrs['srcdoc'])


class RecipeWorkflowTests(unittest.TestCase):
    def setUp(self):
        # Reuse only the synthetic on-disk source builder, not its test cases.
        self.fixture = study_fixtures.StudyWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        self.path, self.output = self.fixture.path, self.fixture.output
        sections = self.fixture.sections
        sections['study']['schema_version'] = 'fakect.recipe/1'
        sections['study']['name'] = 'synthetic-recipe'
        sections['roi']['radius_mm'] = '8'
        for key in ('edit', 'train', 'model'):
            sections.pop(key)
        sections['recipe'] = {'steps': 'grow,shrink', 'overlap': 'sequential'}
        sections['roi.lower'] = {'shape': 'sphere', 'center_ijk': '15,15,12',
                                 'radius_mm': '4', 'coordinate_reviewed': 'false'}
        sections['roi.upper'] = {'shape': 'sphere', 'center_ijk': '15,15,18',
                                 'radius_mm': '4', 'coordinate_reviewed': 'false'}
        for name, region, operation in [('grow', 'lower', 'dilation'), ('shrink', 'upper', 'erosion')]:
            sections['edit.'+name] = {'roi': region, 'operation': operation, 'iterations': '1',
                                     'distance_mm': '1', 'profile': 'uniform', 'profile_axis': 'k',
                                     'shape_k': '6', 'shape_window': '0,1'}
        self.fixture.write_input()

    def run_preview(self, validate_only=False):
        spec = importlib.util.spec_from_file_location('recipe_workflow_preview', ROOT/'scripts/preview_roi.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        with redirect_stdout(io.StringIO()):
            return module.run(self.path, validate_only)

    def test_recipe_exports_chained_states_and_embeds_every_pass(self):
        original_ini = self.path.read_bytes()
        config = load_preview_config(self.path)
        resolved = resolve_preview(config)
        sources = {key: path.read_bytes() for key, path in resolved['source_files'].items()}
        report = self.run_preview()
        self.assertEqual(report['schema_version'], 'fakect.roi-preview/5')
        self.assertNotIn('edit', report)
        self.assertEqual([s['step_name'] for s in report['recipe']['steps']], ['grow', 'shrink'])
        self.assertFalse((self.output/'INCOMPLETE').exists())
        self.assertEqual((self.output/'input.ini').read_bytes(), original_ini)
        steps = report['recipe']['steps']
        with np.load(self.output/steps[0]['array_artifact'], allow_pickle=False) as first, \
                np.load(self.output/steps[1]['array_artifact'], allow_pickle=False) as second, \
                np.load(self.output/'edit.npz', allow_pickle=False) as final:
            np.testing.assert_array_equal(first['before_labels'], self.fixture.labels)
            self.assertGreater(np.count_nonzero(first['changed_mask']), 0)
            self.assertGreater(np.count_nonzero(second['changed_mask']), 0)
            np.testing.assert_array_equal(second['before_labels'], first['edited_labels'])
            np.testing.assert_array_equal(second['before_attenuation_per_pixel'], first['attenuation_proxy_per_pixel'])
            np.testing.assert_array_equal(final['edited_labels'], second['edited_labels'])
            np.testing.assert_array_equal(final['original_labels'], self.fixture.labels)
            arrays, _ = prepare_crop(resolved, config)
            masks = recipe_masks(arrays, resolved, config)
            union = np.logical_or.reduce(list(masks.values()))
            np.testing.assert_array_equal(final['edited_labels'][~union], self.fixture.labels[~union])
            np.testing.assert_array_equal(final['changed_mask'], final['edited_labels'] != final['original_labels'])
        for key, path in resolved['source_files'].items():
            self.assertEqual(path.read_bytes(), sources[key])
        html = (self.output/'report.html').read_text()
        parsed = _Document(html)
        embedded = {hashlib.sha256(base64.b64decode(value.split(',', 1)[1])).hexdigest()
                    for value in parsed.images}
        for step in steps:
            for key in ('comparison', 'profile'):
                path = self.output/step['figures'][key]
                self.assertIn(hashlib.sha256(path.read_bytes()).hexdigest(), embedded)
        self.assertEqual(len(parsed.frames), 4)
        overlay = report['recipe']['surface_overlay']
        self.assertEqual(overlay['counts']['before_voxels'], report['recipe']['full_target_before'])
        self.assertEqual(overlay['counts']['after_voxels'], report['recipe']['full_target_after'])
        self.assertEqual(overlay['counts']['added_voxels'], report['recipe']['counts']['added'])
        self.assertEqual(overlay['counts']['removed_voxels'], report['recipe']['counts']['removed'])
        self.assertTrue((self.output/'edit-overlay.html').is_file())
        inventory = json.loads((self.output/'artifact-manifest.json').read_text())
        for name, digest in inventory.items():
            self.assertEqual(hashlib.sha256((self.output/name).read_bytes()).hexdigest(), digest)
        self.assertFalse(self.fixture.dataset.exists())
        self.assertFalse(self.fixture.model.exists())

    def test_validate_only_reads_no_crop_or_outputs(self):
        with patch('fakect_roi.read_crop', side_effect=AssertionError('metadata validation cannot read crop')), \
                patch('fakect_recipe.apply_recipe', side_effect=AssertionError('validation cannot edit')):
            self.run_preview(validate_only=True)
        self.assertFalse(self.output.exists())

    def test_compact_main_percent_range_reaches_masks_artifacts_and_report(self):
        sections = self.fixture.sections
        sections.pop('roi.lower')
        sections.pop('roi.upper')
        sections.pop('edit.shrink')
        sections['recipe']['steps'] = 'grow'
        sections['roi'].update(shape='tube', center_ijk='15,15,10 ; 15,15,15 ; 15,15,20',
                               radius_mm='4,4,4')
        sections['edit.grow'].update(roi='main', path_percent='30,75', profile_axis='tube')
        self.fixture.write_input()
        report = self.run_preview()
        self.assertEqual(report['config']['rois'], {})
        planned = report['recipe']['plan']['steps'][0]
        self.assertEqual(planned['region_key'], 'edit:grow')
        self.assertEqual(planned['range_metadata']['selector_values'], [30., 75.])
        self.assertEqual(planned['geometry']['roi_nodes_ijk'], ((15., 15., 13.), (15., 15., 15.), (15., 15., 17.5)))
        with np.load(self.output/'edit.npz') as edit:
            changed = edit['changed_mask']
            self.assertGreater(changed.sum(), 0)
            # These planes lie inside round subpath endcaps, but outside the
            # requested parent-arc interval, so uniform editing must leave them.
            self.assertFalse(changed[:13].any())
            self.assertFalse(changed[18:].any())
        document = (self.output/'report.html').read_text()
        self.assertIn('30', document)
        self.assertIn('75', document)
        self.assertTrue((self.output/'recipe-roi-closeups.png').is_file())
        self.assertEqual(len(_Document(document).frames), 4)

    def test_overlap_failure_leaves_incomplete_marker_and_no_published_result(self):
        self.fixture.sections['recipe']['overlap'] = 'error'
        self.fixture.write_input()
        with patch('fakect_recipe_preview.render_recipe_rois', return_value={}), \
                self.assertRaisesRegex(ValueError, 'overlap'):
            self.run_preview()
        self.assertTrue((self.output/'INCOMPLETE').is_file())
        self.assertFalse((self.output/'report.html').exists())
        self.assertFalse((self.output/'edit.npz').exists())
        self.assertFalse((self.output/'steps').exists())

    def test_curvature_outer_range_exports_direction_evidence_and_portable_frame(self):
        sections = self.fixture.sections
        for name in ('roi.lower', 'roi.upper', 'edit.shrink'):
            sections.pop(name)
        sections['recipe']['steps'] = 'grow'
        sections['roi'].update(shape='tube',
                               center_ijk='13,15,10 ; 14.5,15,12.5 ; 15,15,15 ; 14.5,15,17.5 ; 13,15,20',
                               radius_mm='4,4,4,4,4')
        sections['centerline'] = {'smoothing_mm': '0', 'sample_step_mm': '.5',
                                  'min_curvature_per_mm': '.002'}
        sections['edit.grow'].update(roi='main', path_percent='20,80', profile_axis='tube',
                                     direction='outer', angular_width_deg='180', distance_mm='2')
        self.fixture.write_input()
        original_ini = self.path.read_bytes()
        config = load_preview_config(self.path)
        resolved = resolve_preview(config)
        sources = {name: path.read_bytes() for name, path in resolved['source_files'].items()}
        report = self.run_preview()
        steps = report['recipe']['steps']
        self.assertEqual(len(steps), 1)
        self.assertEqual(steps[0]['direction']['direction'], 'outer')
        self.assertEqual(steps[0]['range_metadata']['selector_values'], [20., 80.])
        with np.load(self.output/steps[0]['array_artifact'], allow_pickle=False) as step, \
                np.load(self.output/'edit.npz', allow_pickle=False) as final:
            changed, weight, reliable = step['changed_mask'], step['direction_weight'], step['direction_reliable_mask']
            self.assertGreater(changed.sum(), 0)
            self.assertTrue(np.all(np.isfinite(weight)))
            self.assertTrue(np.all((weight >= 0) & (weight <= 1)))
            self.assertFalse(np.any(changed & ((weight <= 0) | ~reliable)))
            self.assertFalse(np.any(changed & ~step['roi_mask']))
            self.assertTrue(np.all(step['direction_cosine'][changed] > 0))
            np.testing.assert_array_equal(step['before_labels'], self.fixture.labels)
            np.testing.assert_array_equal(final['original_labels'], self.fixture.labels)
            np.testing.assert_array_equal(final['edited_labels'], step['edited_labels'])
            np.testing.assert_array_equal(final['edited_labels'][~changed], self.fixture.labels[~changed])
            np.testing.assert_array_equal(final['attenuation_proxy_per_pixel'][~changed],
                                          step['before_attenuation_per_pixel'][~changed])
            saved_step = json.loads(str(step['step_json']))
            self.assertEqual(saved_step['direction'], steps[0]['direction'])
        self.assertEqual((self.output/'input.ini').read_bytes(), original_ini)
        for name, path in resolved['source_files'].items():
            self.assertEqual(path.read_bytes(), sources[name])
        frames = report['recipe']['centerline_frames']
        self.assertEqual(len(frames), 1)
        frame = frames[0]
        self.assertEqual(frame['parent_roi'], 'main')
        self.assertEqual(frame['metadata']['input_node_count'], 5)
        self.assertGreater(frame['metadata']['reliable_count'], 0)
        frame_record = json.loads((self.output/frame['frame_artifact']).read_text())
        self.assertEqual(frame_record['frame']['metadata'], frame['metadata'])
        self.assertEqual(frame_record['frame']['metadata'], steps[0]['direction']['frame_metadata'])
        self.assertEqual(len(frame_record['frame']['xyz_mm']), frame['sample_count'])
        self.assertEqual(sum(frame_record['frame']['reliable']), frame['reliable_samples'])
        document = (self.output/'report.html').read_text()
        parsed = _Document(document)
        self.assertIn((self.output/frame['html']).read_text(), parsed.frames)
        embedded_images = {hashlib.sha256(base64.b64decode(value.split(',', 1)[1])).hexdigest()
                           for value in parsed.images}
        for key in ('figure', 'curvature_figure'):
            self.assertIn(hashlib.sha256((self.output/frame[key]).read_bytes()).hexdigest(), embedded_images)
        self.assertEqual(len(parsed.frames), 5)
        saved_report = json.loads((self.output/'preview-report.json').read_text())
        self.assertEqual(saved_report['recipe']['centerline_frames'], frames)
        self.assertEqual(saved_report['recipe']['counts'], report['recipe']['counts'])
        inventory = json.loads((self.output/'artifact-manifest.json').read_text())
        for name, digest in inventory.items():
            self.assertEqual(hashlib.sha256((self.output/name).read_bytes()).hexdigest(), digest, name)
        self.assertFalse((self.output/'INCOMPLETE').exists())
        self.assertFalse(self.fixture.dataset.exists())
        self.assertFalse(self.fixture.model.exists())


if __name__ == '__main__':
    unittest.main()
