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

    def test_study_target_plan_is_embedded_separate_from_roi_and_escaped(self):
        payload = '<script>window.BAD=true</script>'
        self.report['training_plan'] = {
            'variant_count': 3, 'target_source_ids': [2922],
            'variants': [{'variant_id': payload, 'operation': 'erosion', 'distance_mm': 1,
                          'shape_k': 10, 'split': 'validation'}],
            'target_preview': {'target_voxels_in_crop': 1250, 'target_voxels_in_roi': 150,
                               'target_voxels_outside_roi': 1100, 'target_components_6': 2,
                               'scope': payload},
            'model': {'architecture': 'unet2d'}, 'split_warning': 'Scenario-only holdout',
            'image_warning': 'Attenuation-copy proxy', 'dataset_directory': '/future/pairs',
            'model_directory': '/future/model'}
        self.report['rerun_command'] = 'python3 scripts/train_study.py --config aorta.ini'
        (self.output / 'training-target.png').write_bytes(PNG)
        metadata, document = self.write()
        self.assertIn('training-target.png', metadata['embedded_assets'])
        self.assertIn('No cohort or model was generated by this preview', document)
        self.assertIn('Unchanged target outside ROI</td><td>1,100', document)
        self.assertIn('python3 scripts/train_study.py --config aorta.ini', document)
        self.assertNotIn('python3 scripts/preview_roi.py --config', document)
        self.assertIn('Original', document)
        self.assertNotIn(payload, document)
        self.assertEqual(sum(tag == 'script' for tag, _ in Document(document).tags), 1)

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

    def morphology_report(self, after_assets=True):
        report = copy.deepcopy(self.report)
        report['config']['study']['schema_version'] = 'fakect.edit/1'
        report['config']['edit'] = {'operation': 'dilation', 'distance_mm': .5, 'profile': 'gaussian',
                                    'profile_axis': 'k', 'shape_k': 2.0, 'shape_window': .8}
        report['config']['reassignment'] = {'allowed_tissues': ['soft_tissue', 'adipose'],
                                            'max_distance_mm': 3.0, 'unresolved': 'retain'}
        report['edit'] = {'engine': 'native-morphology-trial', 'operation': 'dilation',
                           'strength_semantics': 'Maximum physical boundary displacement in mm.',
                           'counts': {'before': 150, 'after': 180, 'proposed_added': 40, 'proposed_removed': 0,
                                      'added': 30, 'removed': 0, 'blocked': 5, 'unresolved': 5},
                           'volume_mm3': {'before': 18.75, 'after': 22.5, 'added': 3.75, 'removed': 0},
                           'transitions': [{'original_id': 9, 'new_id': 1185, 'count': 30}],
                           'warnings': ['Five reassignment requests remain unresolved.'],
                           'scalar_status': 'Provisional local scalar proxy',
                           'components_before': 2, 'components_after': 2}
        for name in ('edit-comparison.png', 'edit-profile.png'):
            (self.output / name).write_bytes(PNG)
        if after_assets:
            (self.output / 'after').mkdir()
            (self.output / 'after/roi-volume.html').write_text(self.volume.replace('embedded:true', 'edited:true'))
            (self.output / 'after/roi-surfaces.png').write_bytes(PNG)
        return report

    def test_morphology_counts_transitions_and_original_views_are_distinct(self):
        report = self.morphology_report()
        metadata, document = self.write(report)
        self.assertIn('Morphology trial · source volumes preserved', document)
        self.assertIn('<a href="#edit">Morphology trial</a>', document)
        self.assertIn('<h2>Morphology trial: dilation</h2>', document)
        self.assertIn('Target voxels after edit', document)
        self.assertIn('target <strong>inside the ROI</strong>', document)
        self.assertIn('<td>Proposed additions</td><td>40</td>', document)
        self.assertIn('<td>Applied additions</td><td>30</td>', document)
        self.assertIn('<td>Target volume before edit</td><td>18.75</td>', document)
        self.assertIn('<td>9</td><td>1185</td><td>30</td>', document)
        self.assertIn('<td>1185</td><td>internal_carotid_left</td><td>140</td>', document)
        self.assertIn('Native 2D inspection — before edit', document)
        self.assertIn('Three-dimensional context — before edit', document)
        self.assertIn('What is inside the ROI? — before edit', document)
        self.assertIn('After-edit 3D context', document)
        self.assertIn('not AI background recovery or a reconstructed CT image', document)
        self.assertIn('Five reassignment requests remain unresolved.', document)
        self.assertIn('python3 scripts/preview_roi.py --config /path/to/xcat-roi.ini', document)
        self.assertEqual(len(metadata['embedded_assets']), 8)

    def test_morphology_assets_are_portable_including_after_frame(self):
        report = self.morphology_report()
        metadata, document = self.write(report)
        parsed = Document(document)
        frames = [attrs for tag, attrs in parsed.tags if tag == 'iframe']
        images = [attrs for tag, attrs in parsed.tags if tag == 'img']
        self.assertEqual(len(frames), 2)
        self.assertEqual(len(images), 6)
        self.assertIn(self.volume, [frame['srcdoc'] for frame in frames])
        self.assertIn(self.volume.replace('embedded:true', 'edited:true'), [frame['srcdoc'] for frame in frames])
        for frame in frames:
            self.assertEqual(frame['sandbox'], 'allow-scripts')
            self.assertEqual(frame['loading'], 'lazy')
            self.assertNotIn('src', frame)
        for image in images:
            self.assertEqual(base64.b64decode(image['src'].split(',', 1)[1]), PNG)
        copied = Path(self.temp.name) / 'standalone-edit.html'
        shutil.copyfile(metadata['path'], copied)
        shutil.rmtree(self.output)
        self.assertEqual(copied.read_text(), document)
        self.assertIn('after/roi-volume.html', metadata['embedded_assets'])

    def test_morphology_rejects_external_assets_in_after_view(self):
        report = self.morphology_report()
        (self.output / 'after/roi-volume.html').write_text('<script src="https://example.test/plotly.js"></script>')
        with self.assertRaisesRegex(ValueError, 'must embed its resources'):
            self.write(report)
        self.assertFalse((self.output / 'report.html').exists())

    def test_morphology_metadata_is_escaped_and_missing_optional_views_are_explicit(self):
        report = self.morphology_report(after_assets=False)
        payload = '</p><script>window.BAD=true</script>'
        report['edit']['warnings'] = [payload]
        report['edit']['scalar_status'] = payload
        report['edit']['engine'] = payload
        report['edit']['transitions'][0]['new_id'] = payload
        report['config']['edit']['profile'] = payload
        (self.output / 'edit-comparison.png').unlink()
        (self.output / 'edit-profile.png').unlink()
        _, document = self.write(report)
        self.assertNotIn(payload, document)
        self.assertIn('&lt;script&gt;window.BAD=true&lt;/script&gt;', document)
        self.assertEqual(sum(tag == 'script' for tag, _ in Document(document).tags), 1)
        self.assertIn('Before, after and difference: preview not generated.', document)
        self.assertIn('Achieved cross-sections and edit profile: preview not generated.', document)
        self.assertNotIn('After-edit 3D context', document)

    def test_no_edit_summary_keeps_preview_mode_even_if_extra_files_exist(self):
        self.morphology_report()
        metadata, document = self.write()
        self.assertIn('Preview only · source labels preserved', document)
        self.assertNotIn('<section id="edit">', document)
        self.assertNotIn('<a href="#edit">', document)
        self.assertNotIn('edit-comparison.png', metadata['embedded_assets'])
        self.assertNotIn('after/roi-volume.html', metadata['embedded_assets'])


if __name__ == '__main__':
    unittest.main()
