"""Review indexes report freshness without touching source or prepared payloads."""
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts'))
import cohort_status
import test_recipe_study_config


class CohortStatusTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.config_path = self.root / 'case input.ini'
        parser = test_recipe_study_config.RecipeStudyConfigTests().parser()
        parser['study']['schema_version'] = 'fakect.recipe-cohort/1'
        parser['input']['case_id'] = '260602'
        parser.remove_section('model')
        for name in ('model_directory', 'split_mode', 'validation_fraction', 'test_fraction', 'split_seed'):
            parser['train'].pop(name)
        parser['train'].update(freeze_directory='outputs/freeze', parameters_reviewed='false')
        parser['sweep.ascending_expand']['distance_mm'] = '3:9:3'
        parser['sweep.descending_narrow']['distance_mm'] = '2:10:2'
        stream = io.StringIO()
        parser.write(stream)
        self.config_path.write_text(stream.getvalue())
        self.config = cohort_status.load_study_config(self.config_path, repo_root=self.root)
        self.current_hash = cohort_status._sha(self.config_path)
        self.project = self.root / 'project.json'
        self.project.write_text(json.dumps({
            'schema_version': 'fakect.cohort-project/1', 'name': '<script>alert(1)</script>',
            'registry': 'registry', 'cases': [{
                'case_id': '260602', 'config': self.config_path.name, 'registry_name': 'coa-260602-r1',
                'template': 'vmale50.nrb', 'provisional_family': 'xcat_reference_adult'}]}))

    def write_json(self, path, value):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))

    def inventory(self, folder):
        self.write_json(folder / 'artifact-manifest.json', {
            str(path.relative_to(folder)): cohort_status._sha(path)
            for path in folder.rglob('*') if path.is_file() and path.name != 'artifact-manifest.json'})

    def preview(self):
        folder = self.config['output']['directory']
        folder.mkdir(parents=True)
        (folder / 'input.ini').write_bytes(self.config_path.read_bytes())
        (folder / 'report.html').write_text('<html>Review</html>')
        # Not a valid archive: dashboard must never attempt to decode payloads.
        (folder / 'crop.npz').write_bytes(b'payload is intentionally opaque')
        self.write_json(folder / 'preview-report.json', {
            'schema_version': 'fakect.roi-preview/5', 'input_config_sha256': self.current_hash,
            'selected_voxels': 100})
        self.inventory(folder)
        return folder

    def preflight(self):
        folder = self.config['train']['preflight_directory']
        folder.mkdir(parents=True)
        (folder / 'input.ini').write_bytes(self.config_path.read_bytes())
        (folder / 'preflight.html').write_text('<html>Preflight</html>')
        self.write_json(folder / 'preflight.json', {
            'complete': True, 'execution_valid': True, 'failed_variants': 0,
            'successful_variants': 16, 'variant_count': 16, 'input_config_sha256': self.current_hash})
        self.inventory(folder)
        return folder

    def collect(self):
        return cohort_status.collect_status(self.project, repo_root=self.root)

    def test_missing_reports_do_not_imply_review_or_preparation(self):
        row = self.collect()['cases'][0]
        self.assertTrue(row['config_valid'])
        self.assertEqual(row['planned_variants'], 16)
        self.assertFalse(row['parameters_reviewed'])
        for stage in ('preview', 'preflight', 'freeze', 'prepare'):
            self.assertEqual(row[stage]['state'], 'missing')
        self.assertEqual(row['published']['state'], 'not_published')
        self.assertNotIn('--family-verified', row['commands']['register'])

    def test_completed_preview_and_preflight_become_stale_after_ini_comment_edit(self):
        self.preview()
        self.preflight()
        first = self.collect()['cases'][0]
        self.assertEqual(first['preview']['state'], 'complete')
        self.assertEqual(first['preflight']['state'], 'complete')
        self.config_path.write_text(self.config_path.read_text() + '\n# An iterative user edit\n')
        row = self.collect()['cases'][0]
        self.assertEqual(row['preview']['state'], 'stale')
        self.assertEqual(row['preflight']['state'], 'stale')
        self.assertFalse(row['preview']['matches_current_input'])
        self.assertTrue(row['preflight']['execution_valid'])

    def test_incomplete_and_tampered_reports_are_not_complete(self):
        folder = self.preview()
        (folder / 'INCOMPLETE').write_text('still running')
        self.assertEqual(self.collect()['cases'][0]['preview']['state'], 'incomplete')
        (folder / 'INCOMPLETE').unlink()
        (folder / 'report.html').write_text('changed')
        row = self.collect()['cases'][0]
        self.assertEqual(row['preview']['state'], 'invalid')
        self.assertIn('checksum mismatch', row['preview']['error'])
        (folder / 'report.html').unlink()
        self.assertEqual(self.collect()['cases'][0]['preview']['state'], 'invalid')

    def test_status_does_not_open_voxel_payloads_or_modify_case_sources(self):
        folder = self.preview()
        source = self.root / 'phantom.raw'
        source.write_bytes(b'raw volume must not be read')
        paths = [self.config_path, source, *[path for path in folder.rglob('*') if path.is_file()]]
        before = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        original_open = Path.open

        def checked_open(path, *args, **kwargs):
            if path.suffix.lower() in ('.npz', '.npy', '.bin', '.raw'):
                raise AssertionError(f'Dashboard opened voxel payload: {path}')
            return original_open(path, *args, **kwargs)

        with patch.object(Path, 'open', checked_open):
            status = self.collect()
            output = self.root / 'index'
            cohort_status.write_status(status, output)
            cohort_status.write_status(status, output)
        after = {path: (path.read_bytes(), path.stat().st_mtime_ns) for path in paths}
        self.assertEqual(before, after)
        self.assertEqual(status['cases'][0]['preview']['payload_files_not_read'], 1)
        self.assertFalse(status['voxel_payloads_read'])
        self.assertFalse(status['generation_dependencies_checked'])
        rendered = (output / 'index.html').read_text()
        self.assertIn('#panel-global', rendered)
        self.assertIn('#panel-local', rendered)
        self.assertIn('case%20input.ini', rendered)
        self.assertNotIn('<script>alert(1)</script>', rendered)
        self.assertIn('&lt;script&gt;alert(1)&lt;/script&gt;', rendered)

    def test_saved_freeze_is_checked_and_compared_with_current_bytes(self):
        folder = self.config['train']['freeze_directory']
        folder.mkdir(parents=True)
        (folder / 'input.ini').write_bytes(self.config_path.read_bytes())
        (folder / 'preflight.json').write_text('{}')
        self.write_json(folder / 'freeze.json', {
            'schema_version': 'fakect.cohort-freeze/1', 'input_sha256': self.current_hash,
            'preflight_sha256': cohort_status._sha(folder / 'preflight.json'),
            'coordinate_reviewed': True, 'parameters_reviewed': True, 'variant_count': 16})
        (folder / 'freeze.sha256').write_text(cohort_status._sha(folder / 'freeze.json'))
        self.assertEqual(self.collect()['cases'][0]['freeze']['state'], 'complete')
        self.config_path.write_text(self.config_path.read_text() + '\n# changed\n')
        self.assertEqual(self.collect()['cases'][0]['freeze']['state'], 'stale')
        (folder / 'preflight.json').write_text('{"changed": true}')
        self.assertEqual(self.collect()['cases'][0]['freeze']['state'], 'invalid')

    def test_prepared_and_registered_snapshots_use_metadata_verification_only(self):
        folder = self.config['train']['dataset_directory']
        folder.mkdir(parents=True)
        self.write_json(folder / 'dataset-manifest.json', {})
        registry = self.root / 'registry'
        self.write_json(registry / 'entries/coa-260602-r1.json', {})
        saved = {'input_config_sha256': 'a' * 64, 'sample_count': 16, 'split_mode': 'scenario_only',
                 'dataset_fingerprint': 'b' * 64}
        entry = {'manifest': saved, 'manifest_path': str(folder / 'dataset-manifest.json'),
                 'sample_count': 16, 'family_verified': False, 'anatomy_family': 'xcat_reference_adult',
                 'original_manifest_path': 'older/cohort-pairs-v2/dataset-manifest.json'}
        with patch.object(cohort_status, 'verify_prepared_manifest', return_value=saved) as prepared:
            with patch.object(cohort_status, 'load_registry_entry', return_value=entry) as registered:
                row = self.collect()['cases'][0]
        prepared.assert_called_once_with(folder / 'dataset-manifest.json', verify_payloads=False)
        registered.assert_called_once_with(registry, 'coa-260602-r1', verify=False)
        self.assertEqual(row['prepare']['state'], 'stale')
        self.assertEqual(row['published']['state'], 'published')
        self.assertEqual(row['published']['snapshot_relation'], 'different_input_snapshot')
        self.assertFalse(row['published']['family_verified'])
        self.assertIn('--name coa-260602-r2', row['commands']['register'])
        self.assertNotIn('--family-verified', row['commands']['register'])

    def test_index_cannot_replace_immutable_case_or_registry_artifacts(self):
        status = self.collect()
        for output in [self.config['output']['directory'], self.root / 'registry',
                       self.config['train']['dataset_directory'] / 'nested']:
            with self.assertRaisesRegex(ValueError, 'outside immutable'):
                cohort_status.write_status(status, output)
            self.assertFalse((output / 'status.json').exists())


if __name__ == '__main__':
    unittest.main()
