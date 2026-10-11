"""Small disk fixtures exercise provenance and all-case atomicity without XCAT."""
import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import fakect_context_data as module


class ContextDataTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.template = self.root / 'anatomy-template.ini'
        self.template.write_text((module.ROOT / 'archive/research/configs/anatomy/thoracic-aorta-260602.ini').read_text())
        self.source_path = self.root / 'source-manifest.json'
        self.output = self.root / 'context-r1'
        self.rows, self.sources = [], {}
        self.bad_case = None
        self.before = np.zeros((3, 3, 3), bool)
        self.before[1, 1, 1] = True
        self.broad = self.before.copy()
        self.broad[0, 0, 0] = True
        self.geometry = {'source_shape_kji': [8, 9, 10], 'spacing_ijk_mm': [.8, .9, 1.],
                         'crop_origin_ijk': [2, 3, 2], 'crop_high_ijk_exclusive': [5, 6, 5],
                         'crop_shape_kji': [3, 3, 3]}
        for case in ('one', 'two'):
            folder = self.root / case
            folder.mkdir()
            files = {}
            for key in ('act', 'atn'):
                filename = folder / (key + '.bin')
                np.zeros((8, 9, 10), np.float32).tofile(filename)
                files[key] = {'path': str(filename), 'bytes': filename.stat().st_size,
                              'mtime_ns': filename.stat().st_mtime_ns}
            self.sources[case] = files
            prepared = folder / 'prepared.json'
            prepared.write_text(json.dumps({'case_id': case, 'frame': 1, 'anatomy_family': 'family-' + case,
                                            'source_files': files}))
            source_npz = folder / 'source.npz'
            np.savez_compressed(source_npz, target_mask=self.before)
            for variant in ('baseline', 'edited'):
                after = self.before.copy()
                if variant == 'edited':
                    after[1, 1, 2] = True
                lineage = np.full(after.shape, -1, np.int32)
                lineage[after] = 13
                variant_npz = folder / (variant + '.npz')
                np.savez_compressed(variant_npz, image=np.ones(after.shape, np.float32),
                                    mask=after, target_origin_index=lineage)
                self.rows.append({'sample_id': case + ':' + variant, 'case_id': case, 'variant_id': variant,
                                  'before_mask_sha256': ('1' if case == 'one' else '2') * 64,
                                  'anatomy_family': 'family-' + case, 'baseline': variant == 'baseline',
                                  'baseline_equivalent': variant == 'baseline', 'source_npz': str(source_npz),
                                  'source_npz_sha256': module.sha256(source_npz), 'variant_npz': str(variant_npz),
                                  'variant_npz_sha256': module.sha256(variant_npz), 'prepared_manifest': str(prepared),
                                  'prepared_manifest_sha256': module.sha256(prepared), 'geometry': copy.deepcopy(self.geometry),
                                  'image_units': 'cm^-1', 'image_method': 'attenuation_copy_proxy'})
        self.save_source()
        self.ini = self.root / 'context.ini'
        self.ini.write_text(f'''[cohort]
schema_version = fakect.context-dataset/1
source_manifest = {self.source_path}
anatomy_template = {self.template}
cases = all
output_directory = {self.output}
surface_inventory_pattern =
''')
        self.config = module.load_config(self.ini)
        self.mock_inspect = patch.object(module, 'inspect_source', side_effect=self.inspect_source).start()
        self.mock_validate = patch.object(module.anatomy, 'validate', side_effect=self.validate_anatomy).start()
        self.mock_run = patch.object(module.anatomy, 'run', side_effect=self.run_anatomy).start()
        self.addCleanup(patch.stopall)

    def save_source(self):
        self.source_path.write_text(json.dumps({'schema_version': 'fakect.two-stage-dataset/1', 'samples': self.rows}))
        self.source_path.with_suffix('.sha256').write_text(module.sha256(self.source_path))

    def inspect_source(self, path, verify_payloads):
        if module.sha256(path) != Path(path).with_suffix('.sha256').read_text().strip():
            raise ValueError('source checksum mismatch')
        return {**json.loads(Path(path).read_text()), 'manifest_sha256': module.sha256(path)}

    def validate_anatomy(self, config):
        case = config['input']['case_id']
        return {'schema_version': 'fakect.phantom-anatomy/1', 'case_id': case, 'frame': 1,
                'status': 'metadata_validated', 'scope': 'fixture thoracic domain',
                'act_source': self.sources[case]['act'], 'settings': config['selection'],
                'provisional': True, 'coordinate_reviewed': False,
                'config_sha256': config['config_sha256'],
                'geometry': {'shape_kji': [8, 9, 10], 'spacing_ijk_mm': [.8, .9, 1.],
                             'crop_low_ijk': [2, 3, 2], 'crop_high_ijk_exclusive': [5, 6, 5],
                             'crop_shape_kji': [3, 3, 3]}}

    def run_anatomy(self, config):
        output = config['output_directory']
        output.mkdir()
        case = config['input']['case_id']
        mask = self.broad.copy()
        if case == self.bad_case:
            mask[1, 1, 1] = False
        np.savez_compressed(output / 'mask.npz', mask=mask, crop_origin_ijk=[2, 3, 2],
                            spacing_ijk_mm=[.8, .9, 1.], source_shape_kji=[8, 9, 10],
                            case_id=np.asarray(case), frame=np.asarray(1))
        report = self.validate_anatomy(config)
        report.update(complete=True, status='complete_provisional_review', mask_npz='mask.npz',
                      mask_npz_sha256=module.sha256(output / 'mask.npz'))
        module._write(output / 'selection.json', report)
        return report

    def test_prepare_shares_images_preserves_broad_masks_and_is_inspectable(self):
        result = module.prepare(self.config)
        self.assertEqual(result['sample_count'], 4)
        self.assertEqual(result['case_count'], 2)
        self.assertFalse(result['anatomical_extent_reviewed'])
        self.assertFalse((self.output / 'INCOMPLETE').exists())
        self.assertEqual(len(list(self.output.rglob('*.bin'))), 0)
        self.assertEqual(self.mock_run.call_count, 2)
        result = module.inspect_dataset(self.output / 'dataset-manifest.json')
        self.assertTrue(result['payloads_verified'])
        for row in result['samples']:
            self.assertEqual(row['source_atn'], self.sources[row['case_id']]['atn'])
            with np.load(row['mask_npz']) as data:
                self.assertTrue(data['after'][0, 0, 0])
                self.assertEqual(int(data['added'].sum()), int(not row['baseline']))

    def test_metadata_only_does_not_extract_or_write(self):
        result = module.validate(self.config)
        self.assertEqual(result['sample_count'], 4)
        self.assertFalse(self.output.exists())
        self.mock_run.assert_not_called()

    def test_failed_scope_keeps_incomplete_and_all_case_failures(self):
        self.bad_case = 'one'
        with self.assertRaisesRegex(ValueError, 'remains INCOMPLETE'):
            module.prepare(self.config)
        status = json.loads((self.output / 'preparation-status.json').read_text())
        self.assertEqual([c['status'] for c in status['cases']], ['requires_review', 'complete_provisional_review'])
        self.assertIn('dilation ancestors', status['cases'][0]['failures'][0]['reason'])
        self.assertEqual(status['expected_samples'], 4)
        self.assertEqual(status['prepared_samples'], 2)
        self.assertTrue((self.output / 'INCOMPLETE').exists())
        self.assertFalse((self.output / 'dataset-manifest.json').exists())

    def test_resume_requires_explicit_matching_request(self):
        module.prepare(self.config)
        with self.assertRaisesRegex(ValueError, 'explicitly use --resume'):
            module.prepare(self.config)
        self.config['resume'] = True
        resumed = module.prepare(self.config)
        self.assertEqual(resumed['sample_count'], 4)
        self.assertEqual(self.mock_run.call_count, 2)
        self.template.write_text(self.template.read_text() + '\n# changed\n')
        with self.assertRaisesRegex(ValueError, 'fingerprint differs'):
            module.prepare(self.config)

    def test_completed_cases_resume_after_interruption(self):
        # Stop after both case artifacts exist, before root finalization.
        original_inputs = module._inputs
        calls = 0
        def interrupted_inputs(config):
            nonlocal calls
            calls += 1
            if calls == 3:
                raise OSError('interrupted')
            return original_inputs(config)
        with patch.object(module, '_inputs', side_effect=interrupted_inputs):
            with self.assertRaisesRegex(OSError, 'interrupted'):
                module.prepare(self.config)
        self.assertTrue((self.output / 'INCOMPLETE').exists())
        self.config['resume'] = True
        result = module.prepare(self.config)
        self.assertEqual(result['sample_count'], 4)
        self.assertEqual(self.mock_run.call_count, 2)

    def test_changed_source_file_stat_rejected_even_without_payload_verification(self):
        module.prepare(self.config)
        path = Path(self.sources['one']['atn']['path'])
        path.write_bytes(path.read_bytes() + b'x')
        with self.assertRaisesRegex(ValueError, 'Whole-body source changed'):
            module.inspect_dataset(self.output / 'dataset-manifest.json', verify_payloads=False)

    def test_duplicate_or_omitted_rows_fail_after_manifest_resigning(self):
        module.prepare(self.config)
        path = self.output / 'dataset-manifest.json'
        document = json.loads(path.read_text())
        document['samples'][1] = document['samples'][0]
        module._write(path, document)
        with self.assertRaisesRegex(ValueError, 'sample disagrees'):
            module.inspect_dataset(path, verify_payloads=False)

    def test_compact_mask_corruption_is_rejected(self):
        result = module.prepare(self.config)
        pair = Path(result['samples'][0]['mask_npz'])
        pair.write_bytes(pair.read_bytes() + b'x')
        with self.assertRaisesRegex(ValueError, 'Transferred mask changed'):
            module.inspect_dataset(self.output / 'dataset-manifest.json')

    def test_case_frame_mutation_rejected(self):
        module.prepare(self.config)
        path = self.output / 'one/anatomy/selection.json'
        document = json.loads(path.read_text())
        document['frame'] = 2
        module._write(path, document)
        preview_path = self.output / 'one/pairs/preview-manifest.json'
        preview = json.loads(preview_path.read_text())
        preview['source_selection_manifest_sha256'] = module.sha256(path)
        module._write(preview_path, preview)
        with self.assertRaisesRegex(ValueError, 'frame/geometry/source'):
            module.inspect_dataset(self.output / 'dataset-manifest.json')

    def test_unknown_input_and_invalid_case_list_rejected(self):
        self.ini.write_text(self.ini.read_text() + 'unexpected = true\n')
        with self.assertRaisesRegex(ValueError, 'documented'):
            module.load_config(self.ini)
        self.ini.write_text(self.ini.read_text().replace('unexpected = true\n', '').replace('cases = all', 'cases = one,one'))
        with self.assertRaisesRegex(ValueError, 'Duplicate'):
            module.load_config(self.ini)


if __name__ == '__main__':
    unittest.main()
