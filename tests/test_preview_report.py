"""Portable HTML assets, provenance and safe rendering of ROI metadata."""
import base64
import copy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_preview_report import write_preview_report

PNG = base64.b64decode('iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAQAAAC1HAwCAAAAC0lEQVR42mP8/x8AAwMCAO+jH9sAAAAASUVORK5CYII=')


class Document(HTMLParser):
    def __init__(self, document):
        super().__init__(convert_charrefs=True)
        self.tags = []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        self.tags.append((tag, dict(attrs)))


class PreviewReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name) / 'output'
        self.output.mkdir()
        self.volume = '<!doctype html><html><head></head><body><div id="plot"></div><script>window.Plotly = {embedded:true};</script></body></html>'
        for name in ('roi-closeups.png', 'roi-z-stack.png', 'roi-surfaces.png'):
            (self.output / name).write_bytes(PNG)
        (self.output / 'roi-volume.html').write_text(self.volume)
        self.report = {
            'schema_version': 'fakect.roi-preview/2',
            'generated_at_utc': '2026-09-17T01:00:00+00:00',
            'config': {'study': {'name': 'Tube artery pilot'},
                       'input': {'case_id': '260602', 'frame': 1},
                       'selection': {'tissue': 'artery', 'source_ids': []}},
            'geometry': {'roi_kind': 'tube', 'nodes_ijk': [[4, 6, 8], [5, 7, 12]],
                         'radii_mm': [2, 3], 'spacing_ijk_mm': [1, 1, 2],
                         'crop_origin_ijk': [0, 0, 0], 'crop_high_ijk_exclusive': [16, 16, 16],
                         'crop_shape_kji': [16, 16, 16]},
            'selection': {'candidate_voxels': 1250, 'roi_voxels': 800, 'selected_voxels': 150,
                          'component_count_6': 2, 'components_voxels_6': [140, 10],
                          'selected_original_ids': [-5, 1185],
                          'selected_original_id_counts': {'-5': 10, '1185': 140}, 'warnings': []},
            'source_names': {'-5': 'tiny_artery', '1185': 'internal_carotid_left'},
            'groups': {'artery': 1250, 'bone': 100, 'unknown': 17},
            'unknown_group_voxels': 17, 'unknown_group_voxels_in_roi': 5, 'missing_dictionary_ids': [-802],
            'policy_version': '1.0.0-proposed',
            'atlas_sources': {'atlas': {'path': '/pinned/anatomy_atlas.csv', 'sha256': 'abc123'},
                              'policy': {'path': '/policy.json', 'sha256': 'def456'}},
            'input_config_sha256': 'inputhash', 'catalog_sha256': 'cataloghash',
            'source_files': {'act': {'path': '/volume/act.bin', 'crop_sha256': 'crophash',
                                     'integrity_scope': 'Crop only; source volume is not hashed'}},
            'volume': {'warnings': []},
        }
        self.ini = '[selection]\ntissue = artery\nsource_ids =\n[roi]\nkind = tube\n'

    def write(self, report=None, text=None):
        metadata = write_preview_report(self.output, report or self.report, self.ini if text is None else text)
        return metadata, Path(metadata['path']).read_text()

    def test_self_contained_assets_survive_copy_without_sibling_files(self):
        metadata, document = self.write()
        moved = Path(self.temp.name) / 'copied-report.html'
        shutil.copyfile(metadata['path'], moved)
        shutil.rmtree(self.output)
        self.assertEqual(document, moved.read_text())
        parsed = Document(document)
        images = [attrs for tag, attrs in parsed.tags if tag == 'img']
        self.assertEqual(len(images), 3)
        for attrs in images:
            self.assertEqual(base64.b64decode(attrs['src'].split(',', 1)[1]), PNG)
        frames = [attrs for tag, attrs in parsed.tags if tag == 'iframe']
        self.assertEqual(len(frames), 1)
        self.assertEqual(frames[0]['srcdoc'], self.volume)
        self.assertEqual(frames[0]['sandbox'], 'allow-scripts')
        self.assertNotIn('src', frames[0])
        for tag, attrs in parsed.tags:
            self.assertFalse(attrs.get('src', '').startswith(('http:', 'https:', '/')))
            if tag == 'a':
                self.assertTrue(attrs['href'].startswith('#'))
        self.assertIn("connect-src 'none'", next(a['content'] for t, a in parsed.tags if t == 'meta' and a.get('http-equiv') == 'Content-Security-Policy'))
        self.assertEqual(metadata['bytes'], len(document.encode('utf-8')))
        self.assertEqual(metadata['sha256'], hashlib.sha256(document.encode('utf-8')).hexdigest())
        self.assertEqual(set(metadata['embedded_assets']), {'roi-closeups.png', 'roi-z-stack.png', 'roi-surfaces.png', 'roi-volume.html'})

    def test_counts_control_points_and_fine_labels(self):
        _, document = self.write()
        self.assertIn('1,250', document)
        self.assertIn('Tissue candidates in crop', document)
        self.assertIn('Selected voxels inside ROI', document)
        self.assertIn('Selected target = tissue candidates ∩ ROI', document)
        self.assertIn('<td>1</td><td>4</td><td>6</td><td>8</td><td>2</td>', document)
        self.assertIn('<td>2</td><td>5</td><td>7</td><td>12</td><td>3</td>', document)
        self.assertIn('internal_carotid_left', document)
        self.assertIn('<td>-5</td><td>tiny_artery</td><td>10</td>', document)
        self.assertIn('multiple six-connected components', document)
        self.assertIn('does not prove that only one anatomical vessel', document)
        self.assertIn('1.0.0-proposed', document)
        self.assertIn('not the DPI atlas', document)
        self.assertIn('crophash', document)
        self.assertIn('Unknown or review-required voxels inside the ROI: <strong>5</strong>', document)
        self.assertIn('captured-input', document)
        self.assertIn('URL.createObjectURL(new Blob([input.value]', document)
        self.assertIn('provide exactly one radius for each point', document)
        self.assertNotIn('shared radius', document)

    def test_user_metadata_and_ini_cannot_inject_active_markup(self):
        payload = '</textarea><script>window.BAD=true</script><img src=x onerror="evil()">&'
        report = copy.deepcopy(self.report)
        report['config']['study']['name'] = payload
        report['source_names']['1185'] = payload
        report['selection']['warnings'] = [payload]
        report['atlas_sources']['atlas']['path'] = payload
        report['generated_at_utc'] = payload
        _, document = self.write(report, payload)
        parsed = Document(document)
        self.assertEqual(sum(tag == 'script' for tag, _ in parsed.tags), 1)
        self.assertEqual(sum(tag == 'img' for tag, _ in parsed.tags), 3)
        self.assertEqual(sum(tag == 'textarea' for tag, _ in parsed.tags), 1)
        self.assertNotIn(payload, document)
        self.assertIn('&lt;/textarea&gt;&lt;script&gt;window.BAD=true&lt;/script&gt;', document)
        self.assertFalse(any('onerror' in attrs for _, attrs in parsed.tags))

    def test_legacy_sphere_does_not_misstate_per_id_roi_counts(self):
        report = copy.deepcopy(self.report)
        report.pop('selection')
        report['selected_voxels'] = 100
        report['selected_in_roi_voxels'] = 5
        report['roi_voxels'] = 25
        report['geometry'] = {'roi_center_ijk': [1, 2, 3], 'roi_radius_mm': 4}
        report['volume']['original_selected_labels'] = [{'original_id': 1185, 'voxel_count': 100,
                                                        'original_name': 'internal_carotid_left'}]
        _, document = self.write(report)
        self.assertIn('<td>1</td><td>1</td><td>2</td><td>3</td><td>4</td>', document)
        self.assertIn('Per-ID counts inside the ROI were not recorded', document)
        self.assertNotIn('<td>1185</td>', document)

    def test_empty_selection_and_missing_optional_assets_render_explicitly(self):
        report = copy.deepcopy(self.report)
        report['selection'].update(selected_voxels=0, selected_original_ids=[], selected_original_id_counts={},
                                   component_count_6=0, components_voxels_6=[])
        for path in self.output.iterdir():
            path.unlink()
        _, document = self.write(report)
        self.assertIn('tissue–ROI intersection is empty', document)
        self.assertIn('No original IDs recorded inside the ROI', document)
        self.assertIn('Interactive 3D preview was not generated', document)
        self.assertIn('Native orthogonal close-ups: preview not generated', document)

    def test_rejects_linked_3d_assets_and_does_not_leave_partial_report(self):
        for linked in ('<script src="https://cdn.example/plotly.js"></script>',
                       '<script src="plotly.js"></script>', '<link rel="stylesheet" href="theme.css">'):
            with self.subTest(linked=linked):
                (self.output / 'roi-volume.html').write_text(linked)
                with self.assertRaisesRegex(ValueError, 'must embed its resources'):
                    self.write()
                self.assertFalse((self.output / 'report.html').exists())

    def test_refuses_overwrite_and_does_not_mutate_report(self):
        original = json.dumps(self.report, sort_keys=True)
        metadata, document = self.write()
        self.assertEqual(json.dumps(self.report, sort_keys=True), original)
        with self.assertRaises(FileExistsError):
            self.write()
        self.assertEqual(Path(metadata['path']).read_text(), document)

    def test_components_and_id_counts_accept_record_form(self):
        report = copy.deepcopy(self.report)
        report['selection']['components_voxels_6'] = [{'voxel_count': 140}, {'size': 10}]
        report['selection']['selected_original_id_counts'] = [{'original_id': -5, 'voxel_count': 10},
                                                             {'original_id': 1185, 'voxel_count': 140}]
        _, document = self.write(report)
        self.assertIn('<td>-5</td><td>tiny_artery</td><td>10</td>', document)
        self.assertIn('<td>1</td><td>140</td>', document)


if __name__ == '__main__':
    unittest.main()
