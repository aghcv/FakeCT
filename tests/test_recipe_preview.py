"""Recipe figures, portable HTML accounting, and per-pass display semantics."""
import base64
from html.parser import HTMLParser
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_preview_report import write_preview_report
from fakect_recipe_preview import _focus, render_recipe_rois, render_recipe_step

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jH9sAAAAASUVORK5CYII=')


class Tags(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.tags = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class RecipeReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.assets = ['recipe-rois.png', 'recipe-roi-closeups.png', 'recipe-roi-surfaces.png',
                       'steps/01-expand/edit-comparison.png', 'steps/01-expand/edit-profile.png',
                       'steps/02-shrink/edit-comparison.png', 'steps/02-shrink/edit-profile.png',
                       'edit-comparison.png', 'edit-profile.png', 'after/roi-surfaces.png']
        for name in self.assets:
            path = self.output/name
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_bytes(PNG)
        (self.output/'after/roi-volume.html').write_text('<html><body><script>window.ready=true;</script></body></html>')
        self.report = {
            'config': {'study': {'name': 'Mixed aorta'}, 'selection': {'tissue': 'artery'},
                       'recipe': {'steps': ['expand', 'shrink']}},
            'recipe': {
                'reassignment_policy': {'mode': 'stiffness',
                    'stiffness': {'default': .05, 'tissues': {'bone': 1., 'skin': .95}, 'labels': {-77: .1}},
                    'effective_input_labels': [{'original_id': -77, 'original_name': 'protected_neighbor',
                        'tissue_name': 'unknown', 'stiffness_group': 'bone', 'stiffness': .1,
                        'stiffness_basis': 'original_id_override', 'count': 2}]},
                'overlap': 'sequential', 'counts': {'before': 30, 'after': 32, 'added': 3,
                    'removed': 1, 'changed': 4, 'ever_changed': 8, 'scalar_changed': 5},
                'activity_counts': {'added': 7, 'removed': 5, 'changed': 12},
                'figures': {'overview': 'recipe-rois.png', 'closeups': 'recipe-roi-closeups.png',
                    'surfaces': 'recipe-roi-surfaces.png', 'rois': [
                        {'name': 'ascending', 'shape': 'tube', 'nodes_ijk': [[1, 2, 3], [3, 4, 5]],
                         'radii_mm': [2, 3], 'effective_voxels': 100, 'target_voxels': 30}]},
                'roi_coverage': {'ascending': {'clipped_by_outer_roi_voxels': 5}},
                'steps': [
                    {'index': 1, 'step_name': 'expand', 'roi_name': 'ascending', 'iteration': 1,
                     'blocked_input_labels': [{'original_id': -77, 'original_name': 'protected_neighbor',
                                               'tissue_name': 'artery', 'count': 2}],
                     'operation': 'dilation', 'counts': {'before': 20, 'after': 27, 'added': 7,
                         'removed': 0, 'blocked': 2, 'unresolved': 0}, 'figures': {
                             'comparison': 'steps/01-expand/edit-comparison.png',
                             'profile': 'steps/01-expand/edit-profile.png', 'focus_ijk': [2, 3, 4]},
                     'labels_before_sha256': 'first-input', 'labels_after_sha256': 'first-output'},
                    {'index': 2, 'name': 'shrink', 'roi': 'descending', 'iteration': 1,
                     'summary': {'operation': 'erosion', 'counts': {'before': 22, 'after': 17,
                         'added': 0, 'removed': 5, 'blocked': 0, 'unresolved': 1}},
                     'figures': {'comparison': 'steps/02-shrink/edit-comparison.png',
                                 'profile': 'steps/02-shrink/edit-profile.png'}}],
                'final_figures': {'comparison': 'edit-comparison.png', 'profile': 'edit-profile.png'},
                'topology_status': 'Topology can change.', 'scalar_status': 'Attenuation-copy proxy.',
                'roi_overlaps': [{'roi_a': 'ascending', 'roi_b': 'descending', 'effective_overlap_voxels': 8}],
                'transitions': [{'original_id': 10, 'new_id': 2922, 'count': 3}],
                'array_artifact': 'edit.npz'},
            'rerun_command': 'python3 scripts/preview_roi.py --config example.ini'}

    def test_all_pass_final_and_named_assets_are_portable_and_accounted_separately(self):
        result = write_preview_report(self.output, self.report, '[recipe]\nsteps = expand, shrink\n')
        document = (self.output/'report.html').read_text()
        self.assertEqual(set(result['embedded_assets']), set(self.assets+['after/roi-volume.html']))
        tags = Tags(document).tags
        self.assertEqual(sum(t == 'img' for t, _ in tags), len(self.assets))
        self.assertTrue(all(attrs['src'].startswith('data:image/png;base64,') for t, attrs in tags if t == 'img'))
        self.assertTrue(all(attrs.get('sandbox') == 'allow-scripts' for t, attrs in tags if t == 'iframe'))
        self.assertIn('<td>Net target additions</td><td>3</td>', document)
        self.assertIn('<td>Sum of additions across passes</td><td>7</td>', document)
        self.assertIn('<td>Original-to-final label differences</td><td>4</td>', document)
        self.assertIn('<td>Unique voxels changed in any pass</td><td>8</td>', document)
        self.assertLess(document.index('Pass 1: expand'), document.index('Pass 2: shrink'))
        self.assertIn('20 → 27', document)
        self.assertIn('22 → 17', document)
        self.assertIn('Selection overview — before edit', document)
        self.assertIn('Named-region recipe preview', document)
        self.assertIn('recipe sweeps are not yet connected'.capitalize(), document)
        self.assertIn('input-state attenuation proxy', document)
        self.assertIn('first-output', document)
        self.assertIn('Blocked proposals by input-state label', document)
        self.assertIn('<td>-77</td><td>protected_neighbor</td><td>artery</td><td>2</td>', document)
        self.assertIn('<td>skin</td><td>0.95</td>', document)
        self.assertIn('<td>Original ID -77</td><td>0.1</td>', document)
        self.assertIn('Effective factors for original labels present in the crop', document)

    def test_user_names_warnings_paths_and_counts_cannot_inject_markup(self):
        payload = '<script>window.BAD=true</script><img src=x onerror=bad()>'
        recipe = self.report['recipe']
        recipe['figures']['rois'][0]['name'] = payload
        recipe['steps'][0]['step_name'] = payload
        recipe['steps'][0]['warnings'] = [payload]
        recipe['steps'][0]['artifact'] = payload
        recipe['steps'][0]['blocked_input_labels'][0]['original_name'] = payload
        recipe['counts']['added'] = payload
        recipe['overlap'] = payload
        recipe['topology_status'] = payload
        recipe['reassignment_policy']['effective_input_labels'][0]['original_name'] = payload
        write_preview_report(self.output, self.report, payload)
        document = (self.output/'report.html').read_text()
        tags = Tags(document).tags
        self.assertEqual(sum(t == 'script' for t, _ in tags), 1)
        self.assertEqual(sum(t == 'img' for t, _ in tags), len(self.assets))
        self.assertFalse(any('onerror' in attrs for _, attrs in tags))
        self.assertNotIn(payload, document)
        self.assertIn('&lt;script&gt;window.BAD=true&lt;/script&gt;', document)

    def test_recipe_overlay_compares_original_with_complete_recipe_once(self):
        (self.output/'edit-overlay.html').write_text('<html><body>shared registered scene</body></html>')
        (self.output/'edit-overlay.png').write_bytes(PNG)
        self.report['recipe']['surface_overlay'] = {'counts': {'before_voxels': 30, 'after_voxels': 32}}
        result = write_preview_report(self.output, self.report, '[recipe]\nsteps = expand, shrink\n')
        document = (self.output/'report.html').read_text()
        self.assertEqual(document.count('id="surface-overlay"'), 1)
        self.assertIn('the final result of the complete recipe', document)
        self.assertIn('edit-overlay.html', result['embedded_assets'])
        self.assertIn('edit-overlay.png', result['embedded_assets'])
        self.assertLess(document.index('id="surface-overlay"'), document.index('Named ROI definitions'))


class RecipeFigureTests(unittest.TestCase):
    def test_step_focus_is_in_changed_region_and_uses_preceding_state_semantics(self):
        roi = np.ones((5, 6, 7), dtype=bool)
        changed = np.zeros_like(roi)
        changed[4, 1, 2] = True
        event = {'before_arrays': {'roi': roi, 'selected': roi},
                 'result': {'added_mask': changed, 'removed_mask': np.zeros_like(roi)},
                 'resolved': {'crop_low_ijk': (100, 200, 300), 'slice_ijk': (100, 200, 300)},
                 'config': {'edit': {'operation': 'dilation'}},
                 'index': 1, 'iteration': 2, 'step_name': 'grow', 'roi_name': 'ascending',
                 'artifact_directory': 'steps/01-grow-iteration-02'}
        with tempfile.TemporaryDirectory() as tmp, patch('fakect_edit_preview.render_edit_comparison') as render:
            render.return_value = {'comparison': 'edit-comparison.png', 'profile': 'edit-profile.png'}
            result = render_recipe_step(event, tmp)
            self.assertEqual(result['focus_ijk'], [102, 201, 304])
            self.assertEqual(event['resolved']['slice_ijk'], (100, 200, 300))
            self.assertFalse(render.call_args.kwargs['before_is_source'])
            self.assertEqual(render.call_args.args[2]['slice_ijk'], (102, 201, 304))
            self.assertEqual(result['comparison'], 'steps/01-grow-iteration-02/edit-comparison.png')
            self.assertTrue((Path(tmp)/result['artifact_directory']).is_dir())

    def test_focus_remains_on_occupied_voxel_for_curved_or_disconnected_regions(self):
        mask = np.zeros((5, 5, 5), dtype=bool)
        mask[0, 1, 1] = mask[4, 3, 3] = True
        focus = _focus(mask, mask, (10, 20, 30))
        local = np.asarray(focus)-[10, 20, 30]
        self.assertTrue(mask[tuple(local[::-1])])

    def test_named_regions_render_counts_for_full_target_and_outer_intersections(self):
        k, j, i = np.indices((12, 12, 12))
        target = ((i-5)**2+(j-5)**2 < 5)
        outer = k < 10
        named = ((i-5)**2+(j-5)**2+(k-6)**2 < 17) & outer
        arrays = {'candidates': target, 'roi': outer,
                  'attenuation_cm_inverse': np.where(target, .2, .05)}
        config = {'study': {'name': 'Small native fixture'},
                  'rois': {'middle': {'shape': 'sphere', 'center_ijk': (5, 5, 6), 'radius_mm': 4}},
                  'preview': {'overlay_opacity': .2, 'volume_stride': 2}}
        resolved = {'crop_low_ijk': (0, 0, 0), 'crop_high_ijk_exclusive': (12, 12, 12),
                    'spacing_ijk_mm': (1, 1, 2), 'slice_ijk': (5, 5, 6)}
        with tempfile.TemporaryDirectory() as tmp:
            result = render_recipe_rois(arrays, resolved, config, {'middle': named}, tmp)
            self.assertEqual(result['target_voxels_in_crop'], int(target.sum()))
            self.assertEqual(result['rois'][0]['target_voxels'], int((target & named).sum()))
            self.assertGreater(result['target_voxels_in_crop'], result['rois'][0]['target_voxels'])
            for key in ('overview', 'closeups', 'surfaces'):
                self.assertTrue((Path(tmp)/result[key]).read_bytes().startswith(b'\x89PNG\r\n'))


if __name__ == '__main__':
    unittest.main()
