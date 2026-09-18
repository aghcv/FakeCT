"""Diagnostic release markers remain distinct from assigned surrounding tissue."""
import base64
import copy
from html.parser import HTMLParser
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_edit_preview import render_edit_comparison
from fakect_preview_report import write_preview_report

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jH9sAAAAASUVORK5CYII=')


def release_metadata():
    return {'mode': 'diagnostic_label', 'label_name': 'released', 'label_id': -2147483648,
            'tissue_id': 254, 'newly_released_voxels': 35, 'current_released_voxels': 35,
            'scalar_status': 'Previous attenuation retained as an unassigned placeholder.',
            'surrounding_labels': [
                {'original_id': 10, 'original_name': 'remaining_artery', 'tissue_name': 'artery',
                 'count': 42, 'stiffness_group': 'artery', 'stiffness': .05},
                {'original_id': 11, 'original_name': 'surrounding_structure', 'tissue_name': 'soft_tissue',
                 'count': 8, 'stiffness_group': 'soft_tissue', 'stiffness': .1}]}


class ReleasedFigureTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        shape = (12, 12, 12)
        k, j, i = np.indices(shape)
        roi = ((i-6)**2+(j-6)**2+(k-6)**2 <= 25)
        before = np.zeros(shape, dtype=bool)
        before[4:8, 3:10, 3:10] = True
        released = np.zeros(shape, dtype=bool)
        released[5, 3:10, 3:8] = True
        after = before & ~released
        empty = np.zeros(shape, dtype=bool)
        self.arrays = {'roi': roi, 'selected': before, 'tissue': np.where(before, 6, 1),
                       'atn': np.where(before, .2, .05)}
        self.edit = {'target_mask_before': before, 'target_mask_after': after,
                     'edited_tissue_labels': np.where(released, 254, self.arrays['tissue']),
                     'added_mask': empty, 'removed_mask': released, 'blocked_mask': empty,
                     'unresolved_mask': empty, 'diagnostic_released_mask': released,
                     'released_mask': released, 'attenuation_unassigned_mask': released,
                     'attenuation_proxy_per_pixel': self.arrays['atn'].copy(),
                     'strength_mm': np.where(roi, 2, 0),
                     'summary': {'counts': {'added': 0, 'removed': 35, 'blocked': 0, 'unresolved': 0},
                                 'release_assignment': release_metadata()}}
        self.config = {'study': {'name': 'Released neighborhood fixture'},
                       'edit': {'operation': 'erosion', 'assign_surrounding_tissue': False}}
        self.resolved = {'crop_low_ijk': (100, 200, 300), 'crop_high_ijk_exclusive': (112, 212, 312),
                         'spacing_ijk_mm': (1, 1, 1), 'slice_ijk': (105, 206, 305), 'roi_kind': 'sphere',
                         'catalog': {'categories': [{'id': 1, 'name': 'soft_tissue'},
                                                    {'id': 6, 'name': 'artery'},
                                                    {'id': 254, 'name': 'released'}]}}

    def render_with_snapshot(self):
        from matplotlib.figure import Figure
        snapshots = {}
        save = Figure.savefig

        def record(figure, path, *args, **kwargs):
            if Path(path).name == 'edit-comparison.png':
                after = figure.axes[1]
                snapshots['after_title'] = after.get_title()
                snapshots['released_color'] = after.images[0].cmap(after.images[0].norm(254))
                snapshots['legend'] = [text.get_text() for legend in figure.legends for text in legend.get_texts()]
            return save(figure, path, *args, **kwargs)

        with patch.object(Figure, 'savefig', record):
            metadata = render_edit_comparison(self.arrays, self.edit, self.resolved, self.config,
                                             self.output, before_is_source=False)
        return metadata, snapshots

    def test_diagnostic_release_has_own_color_label_and_native_neighbor_closeup(self):
        from matplotlib.colors import to_rgba
        original_arrays = copy.deepcopy(self.arrays)
        metadata, snapshots = self.render_with_snapshot()
        np.testing.assert_allclose(snapshots['released_color'], to_rgba('#ff2ea6'))
        self.assertIn('Released: diagnostic label', snapshots['legend'])
        self.assertIn('released (diagnostic marker)', snapshots['legend'])
        self.assertIn('released markers', snapshots['after_title'])
        self.assertNotIn('After: reassigned tissue', snapshots['after_title'])
        self.assertEqual(metadata['release_display']['newly_released_voxels'], 35)
        self.assertEqual(metadata['release_display']['current_released_voxels'], 35)
        focus = np.asarray(metadata['released_focus_ijk'])-[100, 200, 300]
        self.assertTrue(self.edit['released_mask'][tuple(focus[::-1])])
        self.assertTrue((self.output/metadata['released_neighborhood']).read_bytes().startswith(b'\x89PNG\r\n'))
        self.assertEqual(metadata['released_neighborhood_mask_voxels'], 35)
        for name, array in original_arrays.items():
            np.testing.assert_array_equal(self.arrays[name], array)
        np.testing.assert_array_equal(self.edit['attenuation_proxy_per_pixel'], self.arrays['atn'])

    def test_assigned_surrounding_tissue_keeps_legacy_rendering_and_no_marker_artifact(self):
        self.config['edit']['assign_surrounding_tissue'] = True
        self.edit['summary'].pop('release_assignment')
        self.edit['edited_tissue_labels'][self.edit['removed_mask']] = 1
        for key in ('diagnostic_released_mask', 'released_mask', 'attenuation_unassigned_mask'):
            self.edit.pop(key)
        self.resolved['catalog']['categories'].pop()
        metadata, snapshots = self.render_with_snapshot()
        self.assertIn('Released + reassigned', snapshots['legend'])
        self.assertIn('After: reassigned tissue', snapshots['after_title'])
        self.assertNotIn('release_display', metadata)
        self.assertNotIn('released_neighborhood', metadata)
        self.assertFalse((self.output/'released-neighborhood.png').exists())


class ReleasedReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        for name in ('edit-comparison.png', 'edit-profile.png', 'released-neighborhood.png'):
            (self.output/name).write_bytes(PNG)
        self.report = {'config': {'study': {'name': 'Diagnostic erosion'},
                                  'selection': {'tissue': 'artery', 'source_ids': []},
                                  'edit': {'operation': 'erosion', 'assign_surrounding_tissue': False}},
                       'edit': {'operation': 'erosion', 'counts': {'before': 196, 'after': 161,
                                  'added': 0, 'removed': 35, 'blocked': 0, 'unresolved': 0},
                                'release_assignment': release_metadata(),
                                'figures': {'released_neighborhood': 'released-neighborhood.png'}}}

    def write(self):
        metadata = write_preview_report(self.output, self.report, '[edit]\nassign_surrounding_tissue = false\n')
        return metadata, (self.output/'report.html').read_text()

    def test_single_edit_reports_placeholder_attenuation_and_observed_neighboring_labels(self):
        metadata, document = self.write()
        self.assertIn('<td>Newly marked released in this edit</td><td>35</td>', document)
        self.assertIn('<td>Current released markers</td><td>35</td>', document)
        self.assertIn('released-neighborhood.png', metadata['embedded_assets'])
        self.assertIn('not an XCAT anatomical identity', document)
        self.assertIn('<code>-2147483648</code>', document)
        self.assertIn('<code>254</code>', document)
        self.assertIn('retain their previous attenuation as an unassigned placeholder', document)
        self.assertIn('attenuation_unassigned_mask', document)
        self.assertIn('existing labels in the input-state boundary neighborhood', document)
        self.assertIn('do not establish which tissue should replace the released voxels', document)
        self.assertIn('<td>10</td><td>remaining_artery</td><td>artery</td><td>42</td><td>artery</td><td>0.05</td>', document)

    def test_recipe_reports_per_pass_and_current_markers_separately_and_escapes_names(self):
        release = release_metadata()
        payload = '<script>window.BAD=true</script><img src=x onerror=bad()>'
        release['surrounding_labels'][0]['original_name'] = payload
        step_path = self.output/'steps/02-erode/released-neighborhood.png'
        step_path.parent.mkdir(parents=True)
        step_path.write_bytes(PNG)
        self.report.pop('edit')
        self.report['config'].pop('edit')
        self.report['config']['recipe'] = {'steps': ['grow', 'erode']}
        final_release = release_metadata()
        final_release.pop('surrounding_labels')
        self.report['recipe'] = {'counts': {'added': 10, 'removed': 35}, 'figures': {},
                                 'release_assignment': final_release,
                                 'final_figures': {'released_neighborhood': 'released-neighborhood.png'},
                                 'steps': [{'step_name': 'grow', 'roi_name': 'main', 'summary': {'counts': {}}},
                                           {'step_name': 'erode', 'roi_name': 'main',
                                            'summary': {'counts': {'removed': 35}, 'release_assignment': release},
                                            'figures': {'released_neighborhood': 'steps/02-erode/released-neighborhood.png'}}]}
        metadata, document = self.write()
        self.assertIn('<td>Unique voxels marked released during this recipe</td><td>35</td>', document)
        self.assertIn('<td>Newly marked released in this pass</td><td>35</td>', document)
        self.assertEqual(document.count('<td>Current released markers</td><td>35</td>'), 2)
        self.assertIn('steps/02-erode/released-neighborhood.png', metadata['embedded_assets'])
        self.assertIn('released-neighborhood.png', metadata['embedded_assets'])
        self.assertNotIn(payload, document)
        self.assertIn('&lt;script&gt;window.BAD=true&lt;/script&gt;', document)
        class Scripts(HTMLParser):
            count = 0
            def handle_starttag(self, tag, attrs):
                if tag == 'script':
                    self.count += 1
        parsed = Scripts()
        parsed.feed(document)
        self.assertEqual(parsed.count, 1)


if __name__ == '__main__':
    unittest.main()
