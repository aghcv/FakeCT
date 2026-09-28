"""Optional anatomy decisions remain auditable, escaped and portable."""
import base64
import copy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT / 'src'), str(ROOT / 'scripts')]
from fakect_global_preview import prepare_global
from fakect_preview_report import write_preview_report
from preview_roi import capture_anatomy_selection, selection_title


class Links(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.links = []
        self.feed(document)

    def handle_starttag(self, tag, attrs):
        if tag == 'a':
            self.links.append(dict(attrs))


class AnatomyReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.output = Path(self.temp.name)
        self.audit = {
            'schema_version': 'fakect.anatomy-selection/1',
            'anatomy': 'thoracic_aorta', 'selected_source_ids': [2922, 18],
            'connectivity_scope': 'native ROI, six-neighbor contact with accepted aorta seed',
            'counts': {'included': 2, 'excluded': 1},
            'warnings': ['Generic artery identity remains unresolved.'],
            'surface_evidence': {'method': 'bounding-box candidate screen'},
            'records': [
                {'original_id': 2922, 'original_name': 'dias_aorta',
                 'canonical_name': 'aorta', 'hierarchy_path': ['artery', 'aorta'],
                 'roi_voxels': 120, 'crop_voxels': 300, 'connected_roi_voxels': 120,
                 'decision': 'included', 'reason': 'atlas-qualified aorta inside ROI',
                 'surface_matches': [{'source_file': 'aorta.raw', 'group': 'aorta',
                                      'local_surface_index': 4, 'original_name': 'dias_aorta',
                                      'id_join_status': 'name match',
                                      'bounds_overlap_roi_envelope': True}]},
                {'original_id': 18, 'original_name': 'left_subclavian',
                 'canonical_name': 'subclavian artery', 'hierarchy_path': 'artery / systemic',
                 'roi_voxels': 20, 'crop_voxels': 80, 'connected_roi_voxels': 20,
                 'decision': 'included', 'reason': 'qualified connected branch'},
                {'original_id': 22, 'original_name': 'pulmonary_artery',
                 'canonical_name': 'pulmonary artery', 'hierarchy_path': 'artery / pulmonary',
                 'roi_voxels': 12, 'crop_voxels': 70, 'connected_roi_voxels': 0,
                 'decision': 'excluded', 'reason': 'atlas excludes pulmonary branch'},
            ],
        }
        self.config = {'study': {'name': 'Anatomy filter review'},
                       'input': {'case_id': 'test', 'frame': 1},
                       'selection': {'tissue': 'artery', 'source_ids': [], 'anatomy': 'thoracic_aorta'}}
        self.report = {'config': self.config,
                       'selection': {'candidate_voxels': 380, 'selected_voxels': 140,
                                     'roi_voxels': 500, 'anatomical_selection': self.audit},
                       'geometry': {'roi_kind': 'sphere', 'nodes_ijk': [[2, 2, 2]], 'radii_mm': [1]},
                       'volume': {}}

    def capture(self, raw=None):
        resolved = {'anatomical_selection': self.audit}
        if raw is not None:
            resolved.update(anatomy_selection_bytes=raw,
                            anatomy_selection_sha256=hashlib.sha256(raw).hexdigest())
        artifact = capture_anatomy_selection(resolved, self.output)
        self.report['anatomy_selection_artifact'] = artifact
        return artifact

    def render(self):
        metadata = write_preview_report(self.output, self.report, '[selection]\ntissue = artery\n')
        return metadata, (self.output / 'report.html').read_text()

    def test_decisions_download_preserves_exact_snapshot(self):
        raw = json.dumps(self.audit, separators=(',', ':')).encode() + b'\n\n'
        self.capture(raw)
        metadata, document = self.render()
        self.assertIn('id="anatomical-selection"', document)
        self.assertIn('<td>included</td>', document)
        self.assertIn('<td>excluded</td>', document)
        self.assertIn('atlas excludes pulmonary branch', document)
        self.assertIn('native ROI, six-neighbor contact', document)
        self.assertIn('does not remove disconnected pieces sharing an accepted ID', document)
        self.assertIn('not the final target mask', document)
        self.assertIn('never voxel label IDs', document)
        self.assertNotIn('All original IDs mapped to this tissue are candidates.', document)
        links = [link for link in Links(document).links if link.get('download') == 'anatomy-selection.json']
        self.assertEqual(len(links), 1)
        self.assertEqual(base64.b64decode(links[0]['href'].split(',', 1)[1]), raw)
        self.assertEqual((self.output / 'anatomy-selection.json').read_bytes(), raw)
        self.assertEqual(metadata['embedded_assets']['anatomy-selection.json']['sha256'], hashlib.sha256(raw).hexdigest())

    def test_untrusted_names_paths_reasons_and_warnings_are_escaped(self):
        payload = '<script>alert("atlas")</script>'
        self.audit['anatomy'] = payload
        self.audit['warnings'].append(payload)
        record = self.audit['records'][0]
        for field in ('original_name', 'canonical_name', 'hierarchy_path', 'reason', 'decision'):
            record[field] = payload
        for field in ('source_file', 'group', 'original_name', 'id_join_status'):
            record['surface_matches'][0][field] = payload
        self.capture()
        _, document = self.render()
        self.assertNotIn(payload, document)
        self.assertIn('&lt;script&gt;alert(&quot;atlas&quot;)&lt;/script&gt;', document)

    def test_absent_option_preserves_tissue_only_report_and_title(self):
        del self.report['selection']['anatomical_selection']
        del self.config['selection']['anatomy']
        _, document = self.render()
        self.assertNotIn('id="anatomical-selection"', document)
        self.assertNotIn('anatomy-selection.json', document)
        self.assertIn('All original IDs mapped to this tissue are candidates.', document)
        self.assertIn('Selected target = tissue candidates ∩ ROI.', document)
        self.assertEqual(selection_title(self.config, {}), 'artery; all source IDs in group; bounded by ROI')
        self.assertIsNone(capture_anatomy_selection({}, self.output))

    def test_snapshot_mismatch_and_overwrite_are_rejected(self):
        raw = json.dumps(self.audit).encode()
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            capture_anatomy_selection({'anatomical_selection': self.audit,
                                      'anatomy_selection_bytes': raw,
                                      'anatomy_selection_sha256': 'wrong'}, self.output)
        altered = copy.deepcopy(self.audit)
        altered['selected_source_ids'] = [22]
        with self.assertRaisesRegex(ValueError, 'snapshot changed'):
            capture_anatomy_selection({'anatomical_selection': altered,
                                      'anatomy_selection_bytes': raw}, self.output)
        self.capture(raw)
        with self.assertRaises(FileExistsError):
            self.capture(raw)
        (self.output / 'anatomy-selection.json').write_bytes(b'{}')
        with self.assertRaisesRegex(ValueError, 'does not match'):
            self.render()

    def test_filtered_title_describes_effective_ids(self):
        title = selection_title(self.config, {'source_ids': (2922, 18), 'anatomical_selection': self.audit})
        self.assertIn('thoracic_aorta; 2 accepted whole source IDs', title)
        self.assertNotIn('all source IDs', title)

    def test_manual_name_only_capture_has_no_native_connectivity_claim(self):
        self.audit.update(anatomy=None, method='explicit_original_or_canonical_names',
                          connectivity_scope='not_requested', counts={})
        self.audit.pop('surface_evidence')
        self.config['selection'].update(anatomy='', anatomical_labels=['dias_aorta'])
        for row in self.audit['records']:
            for key in ('roi_voxels', 'crop_voxels', 'connected_roi_voxels'):
                row.pop(key)
            row['surface_matches'] = []
        self.capture()
        _, document = self.render()
        self.assertIn('does not request a connectivity test', document)
        self.assertIn('Not recorded', document)
        title = selection_title(self.config, {'source_ids': (2922,), 'anatomical_selection': self.audit})
        self.assertIn('named anatomical labels', title)
        self.assertNotIn('native connectivity', title)

    def test_visible_surface_evidence_deduplicates_and_summarizes(self):
        surface = self.audit['records'][0]['surface_matches'][0]
        surface['centerline'] = {'status': 'verified_two_pole_rings', 'centerline_raw_mm': [[987654, 0, 0]]}
        surface['native_centerline_samples'] = {'roi_sample_voxels': 9, 'matching_name_id_counts': {'2922': 10}}
        self.audit['records'][1]['surface_matches'] = [copy.deepcopy(surface)]
        self.audit['surface_evidence']['records'] = [surface]
        self.capture()
        _, document = self.render()
        self.assertEqual(document.count('<td>aorta.raw</td>'), 1)
        self.assertIn('<td>2922, 18</td>', document)
        self.assertIn('<td>verified_two_pole_rings</td>', document)
        self.assertIn('surface_record_count', document)
        self.assertNotIn('987654', document)

    def test_indexed_surface_evidence_supports_compact_audit(self):
        surface = self.audit['records'][0].pop('surface_matches')[0]
        self.audit['surface_evidence']['records'] = [surface]
        self.audit['records'][0]['surface_refs'] = [0]
        self.audit['records'][1]['surface_refs'] = [0]
        # A mixed old/new record must not duplicate evidence or associated IDs.
        self.audit['records'][0]['surface_matches'] = [copy.deepcopy(surface)]
        raw = json.dumps(self.audit, separators=(',', ':')).encode()
        self.capture(raw)
        _, document = self.render()
        self.assertEqual(document.count('<td>aorta.raw</td>'), 1)
        self.assertIn('<td>2922, 18</td>', document)
        links = [link for link in Links(document).links if link.get('download') == 'anatomy-selection.json']
        self.assertEqual(base64.b64decode(links[0]['href'].split(',', 1)[1]), raw)

    def test_indexed_surface_without_group_metadata_remains_distinct(self):
        self.audit['surface_evidence']['records'] = [{'original_name': 'first'}, {'original_name': 'second'}]
        self.audit['records'][0].pop('surface_matches')
        self.audit['records'][0]['surface_refs'] = [0, 1]
        self.audit['records'][1]['surface_refs'] = [0]
        self.capture()
        _, document = self.render()
        self.assertEqual(document.count('<td>first</td>'), 1)
        self.assertEqual(document.count('<td>second</td>'), 1)
        self.assertIn('<td>2922, 18</td>', document)

    def test_invalid_surface_reference_is_rejected(self):
        for reference in (-1, 0, True, '0'):
            with self.subTest(reference=reference):
                self.audit['records'][0]['surface_refs'] = [reference]
                with self.assertRaisesRegex(ValueError, 'invalid surface reference'):
                    self.render()

    def test_global_filtered_view_is_an_id_locator_not_native_mask(self):
        labels = np.zeros((5, 5, 5), dtype='<f4')
        labels[0, 0, 0] = labels[2, 2, 2] = 2922
        labels[2, 2, 3] = 22
        labels.tofile(self.output / 'act.bin')
        np.ones_like(labels).tofile(self.output / 'atn.bin')
        catalog = {'categories': [{'id': 0, 'name': 'background'}, {'id': 5, 'name': 'artery'},
                                  {'id': 255, 'name': 'unknown'}],
                   'records': [{'original_id': key, 'classification': {'tissue_id': group}}
                               for key, group in ((0, 0), (2922, 5), (22, 5))]}
        config = copy.deepcopy(self.config)
        config['roi'] = {'shape': 'sphere', 'center_ijk': (2, 2, 2), 'radius_mm': 1}
        resolved = {'shape_kji': labels.shape, 'spacing_ijk_mm': (1, 1, 1),
                    'source_files': {key: self.output / (key + '.bin') for key in ('act', 'atn')},
                    'catalog': catalog, 'source_ids': (2922,), 'anatomical_selection': self.audit}
        arrays, metadata = prepare_global(resolved, config)
        self.assertTrue(metadata['selection']['anatomy_filter_active'])
        self.assertFalse(metadata['selection']['source_ids_explicit'])
        self.assertIn('not the final target mask', metadata['selection']['display_scope'])
        self.assertTrue(arrays['candidates'][0, 0, 0])  # Whole-volume ID locator includes voxels outside ROI.
        self.assertFalse(arrays['candidates'][2, 2, 3])  # Excluded pulmonary label is not displayed as target.


if __name__ == '__main__':
    unittest.main()
