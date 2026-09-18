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

    def test_compact_range_uses_human_names_and_physical_main_point_reference(self):
        self.report['geometry'] = {'roi_kind': 'tube', 'nodes_ijk': [[0, 0, 0], [3, 0, 0], [3, 4, 0]],
                                   'radii_mm': [1, 2, 3], 'spacing_ijk_mm': [2, 1, 1]}
        region = self.report['recipe']['figures']['rois'][0]
        region.update(name='edit:expand', display_name='expand', parent_roi='main',
                      range_metadata={'selector': 'path_percent', 'selector_values': [30, 75],
                                      'parent_total_length_mm': 10, 'selected_length_mm': 4.5,
                                      'start_distance_mm': 3, 'end_distance_mm': 7.5,
                                      'start_fraction': .3, 'end_fraction': .75,
                                      'original_node_numbers_retained': [2]})
        self.report['recipe']['roi_coverage'] = {'edit:expand': {'clipped_by_outer_roi_voxels': 5}}
        self.report['recipe']['steps'][0]['roi_name'] = 'edit:expand'
        write_preview_report(self.output, self.report, '[edit.expand]\nroi = main\npath_percent = 30, 75\n')
        document = (self.output/'report.html').read_text()
        self.assertIn('<td>expand</td><td>main</td><td><code>path_percent = 30, 75</code> (%)</td><td>4.5</td>', document)
        self.assertIn('ROI expand (base main)', document)
        self.assertNotIn('ROI edit:expand', document)
        self.assertIn('<td>2</td><td>3</td><td>0</td><td>0</td><td>2</td><td>6</td><td>60</td>', document)
        self.assertIn('<td>3</td><td>3</td><td>4</td><td>0</td><td>3</td><td>10</td><td>100</td>', document)
        self.assertIn('Total length: <strong>10 mm</strong>', document)
        self.assertIn('Range boundaries are strict', document)
        self.assertIn('shape_window = 0, 1', document)
        self.assertIn('within that interval, not across the full base tube', document)
        self.assertIn('<td>Distance along base path (mm)</td><td>3 → 7.5</td>', document)

    def test_range_metadata_is_escaped_and_missing_spacing_is_not_assumed(self):
        payload = '<script>window.BAD=true</script>'
        self.report['geometry'] = {'roi_kind': 'tube', 'nodes_ijk': [[0, 0, 0], [2, 3, 4]],
                                   'radii_mm': [1, 2]}
        region = self.report['recipe']['figures']['rois'][0]
        region.update(display_name=payload, parent_roi=payload,
                      range_metadata={'selector': 'point_range', 'selector_values': [1, payload],
                                      'selected_length_mm': payload})
        write_preview_report(self.output, self.report, '[recipe]\nsteps = expand, shrink\n')
        document = (self.output/'report.html').read_text()
        self.assertNotIn(payload, document)
        self.assertEqual(sum(tag == 'script' for tag, _ in Tags(document).tags), 1)
        self.assertIn('point_range = 1, &lt;script&gt;', document)
        self.assertIn('Total length: <strong>Not recorded mm</strong>', document)
        self.assertIn('<td>1</td><td>0</td><td>0</td><td>0</td><td>1</td><td>Not recorded</td><td>Not recorded</td>', document)

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

    def test_centerline_views_embed_once_with_per_pass_direction_and_provenance(self):
        (self.output/'edit-overlay.html').write_text('<html><body>registered comparison</body></html>')
        directory = self.output/'centerline-01-main'
        directory.mkdir()
        (directory/'frame.html').write_text('<html><body><script>window.frameReady=true;</script></body></html>')
        for name in ('frame.png', 'curvature.png'):
            (directory/name).write_bytes(PNG)
        payload = '<script>window.BAD=true</script>'
        self.report['recipe']['centerline_frames'] = [{
            'parent_roi': 'main', 'html': 'centerline-01-main/frame.html',
            'figure': 'centerline-01-main/frame.png', 'curvature_figure': 'centerline-01-main/curvature.png',
            'frame_artifact': 'centerline-01-main/frame.json', 'sample_count': 205,
            'reliable_samples': 203, 'unreliable_samples': 2, 'arrow_length_mm': 6.5,
            'settings': {'smoothing_mm': 1, 'sample_step_mm': 1, 'min_curvature_per_mm': .002},
            'metadata': {'fit_rms_displacement_mm': .999, 'warning_detail': payload},
            'warnings': [payload], 'artifacts': {'frame_artifact': {'sha256': 'frame-hash'}}}]
        self.report['recipe']['steps'][0]['direction'] = {
            'direction': 'outer', 'angular_width_deg': 180, 'weight_semantics': 'Cosine sector',
            'unreliable_roi_voxels': 50, 'angular_supported_roi_voxels': 100}
        self.report['recipe']['steps'][1]['summary']['direction'] = {
            'direction': 'inner', 'angular_width_deg': 180, 'weight_semantics': payload}
        result = write_preview_report(self.output, self.report, '[centerline]\nsmoothing_mm = 1\n')
        document = (self.output/'report.html').read_text()
        self.assertEqual(document.count('id="centerline-frames"'), 1)
        self.assertLess(document.index('id="surface-overlay"'), document.index('id="centerline-frames"'))
        self.assertLess(document.index('id="centerline-frames"'), document.index('Named ROI definitions'))
        for name in ('frame.html', 'frame.png', 'curvature.png'):
            self.assertIn('centerline-01-main/'+name, result['embedded_assets'])
        self.assertIn('frame-hash', document)
        self.assertIn('centerline-01-main/frame.json', document)
        self.assertNotIn('centerline-01-main/frame.json', result['embedded_assets'])
        self.assertIn('<td>Circumferential edit direction</td><td>outer</td>', document)
        self.assertIn('<td>Circumferential edit direction</td><td>inner</td>', document)
        self.assertIn('<td>Angular width (degrees)</td><td>180</td>', document)
        self.assertIn('<td>ROI voxels without a reliable inner/outer frame</td><td>50</td>', document)
        self.assertIn('original ROI points and masks remain unchanged', document)
        self.assertIn('Arrow length is a display scale', document)
        self.assertNotIn(payload, document)
        self.assertEqual(sum(tag == 'script' for tag, _ in Tags(document).tags), 1)

    def test_centerline_view_rejects_external_resources(self):
        directory = self.output/'centerline-01-main'
        directory.mkdir()
        (directory/'frame.html').write_text('<script src="https://example.com/frame.js"></script>')
        self.report['recipe']['centerline_frames'] = [{'parent_roi': 'main',
                                                      'html': 'centerline-01-main/frame.html'}]
        with self.assertRaisesRegex(ValueError, 'must embed its resources'):
            write_preview_report(self.output, self.report, '[recipe]\nsteps = expand, shrink\n')
        self.assertFalse((self.output/'report.html').exists())


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

    def test_range_region_renderer_uses_resolved_definitions_and_preserves_mask_keys(self):
        k, j, i = np.indices((8, 8, 8))
        target = ((i-3)**2+(j-3)**2 < 4)
        mask = target & (k >= 2) & (k <= 5)
        arrays = {'candidates': target, 'roi': np.ones_like(mask),
                  'attenuation_cm_inverse': np.where(target, .2, .05)}
        config = {'study': {'name': 'Range fixture'}, 'rois': {},
                  'preview': {'overlay_opacity': .2, 'volume_stride': 2}}
        resolved = {'crop_low_ijk': (0, 0, 0), 'crop_high_ijk_exclusive': (8, 8, 8),
                    'spacing_ijk_mm': (1, 1, 1), 'slice_ijk': (3, 3, 4)}
        metadata = {'selector': 'path_percent', 'selector_values': [30, 75],
                    'selected_length_mm': 3, 'start_distance_mm': 2, 'end_distance_mm': 5}
        definition = {'shape': 'tube', 'center_ijk': [[3, 3, 2], [3, 3, 5]],
                      'radius_mm': [2, 2], 'display_name': 'expand', 'parent_roi': 'main',
                      'range_metadata': metadata}
        with tempfile.TemporaryDirectory() as tmp, patch('fakect_recipe.recipe_regions') as regions:
            regions.return_value = {'edit:expand': definition}
            result = render_recipe_rois(arrays, resolved, config, {'edit:expand': mask}, tmp)
            regions.assert_called_once_with(config, resolved)
            row = result['rois'][0]
            self.assertEqual(row['name'], 'edit:expand')
            self.assertEqual(row['display_name'], 'expand')
            self.assertEqual(row['parent_roi'], 'main')
            self.assertEqual(row['range_metadata'], metadata)
            self.assertEqual(row['effective_voxels'], int(mask.sum()))
            self.assertEqual(row['target_voxels'], int((mask & target).sum()))


if __name__ == '__main__':
    unittest.main()
