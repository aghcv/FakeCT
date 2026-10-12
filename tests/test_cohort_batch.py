"""Conservative batch resume must never treat failed or stale output as done."""
import hashlib
import importlib.util
import json
from pathlib import Path
import tempfile
import threading
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
SPEC = importlib.util.spec_from_file_location('run_cohort_batch', ROOT / 'scripts/research/evaluation/run_cohort_batch.py')
batch = importlib.util.module_from_spec(SPEC)
SPEC.loader.exec_module(batch)


class CohortBatchTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.output = self.root / 'batch'
        self.output.mkdir()
        self.config_path = self.root / 'case.ini'
        self.config_path.write_text('example fixture input\n')
        self.config = {
            'study': {'schema_version': 'fakect.recipe-cohort/1'},
            'input': {'case_id': '260602'},
            'selection': {'anatomy': 'thoracic_aorta', 'anatomy_file': self.root / 'anatomy/selection.json'},
            'train': {'preflight_directory': self.root / 'preflight'},
        }
        self.resolved = {'source_ids': (2922, 1547)}
        self.row = {'case_id': '260602', 'config': str(self.config_path)}
        self.state = {}

    def update(self, identifier, **values):
        self.state.setdefault(identifier, {}).update(values)

    def anatomy(self):
        path = self.config['selection']['anatomy_file']
        path.parent.mkdir()
        path.write_text('{}\n')

    def preflight(self, *, fingerprint='fingerprint', ready=True):
        folder = self.config['train']['preflight_directory']
        folder.mkdir()
        document = {
            'schema_version': 'fakect.recipe-preflight/1', 'complete': True,
            'execution_complete': True, 'execution_valid': ready,
            'failed_variants': 0 if ready else 1, 'successful_variants': 4 if ready else 3,
            'variant_count': 4, 'cohort_ready': ready,
            'dataset_fingerprint': fingerprint,
            'input_config_sha256': batch.sha(self.config_path),
        }
        (folder / 'preflight.json').write_text(json.dumps(document))
        (folder / 'input.ini').write_bytes(self.config_path.read_bytes())
        (folder / 'preflight.html').write_text('<html>fixture</html>')
        for name in ('data-config.json', 'source-crop.json'):
            (folder / name).write_text('{}\n')
        (folder / 'artifact-manifest.json').write_text(json.dumps({path.name: batch.sha(path) for path in folder.iterdir()}))
        return folder

    def run_case(self, execute):
        with patch.object(batch, 'load_study_config', return_value=self.config), \
                patch.object(batch, 'resolve_preview', return_value=self.resolved), \
                patch.object(batch, '_fingerprint', return_value='fingerprint'), \
                patch.object(batch, 'execute', side_effect=execute) as call:
            batch.run_case(self.row, self.output, 'run', 'preflight', 'python', self.update,
                           threading.Event(), self.root)
        return call

    # Purpose: Verify verified complete batch outputs are reused.
    # Inputs: Existing anatomy artifact and matching complete preflight inventory.
    # Outputs: No native commands; case complete and both stages marked reused.
    def test_existing_complete_outputs_are_verified_then_reused(self):
        self.anatomy()
        self.preflight()
        call = self.run_case(AssertionError('must not re-execute native data'))
        call.assert_not_called()
        self.assertEqual(self.state['260602']['state'], 'complete')
        self.assertEqual(self.state['260602']['anatomy']['state'], 'reused')
        self.assertEqual(self.state['260602']['preflight']['state'], 'reused')

    # Purpose: Verify missing batch stages run once and resume without repetition.
    # Inputs: Missing anatomy/preflight artifacts created by mocked stage execution.
    # Outputs: Exactly anatomy then preflight commands; second run executes no commands.
    def test_missing_steps_execute_once_and_then_resume(self):
        seen = []
        def execute(command, *args):
            seen.append(Path(command[1]).name)
            if seen[-1] == 'resolve_anatomy.py':
                self.anatomy()
            else:
                self.preflight()
            return 0
        call = self.run_case(execute)
        self.assertEqual(call.call_count, 2)
        self.assertEqual(seen, ['resolve_anatomy.py', 'prepare_cohort.py'])
        self.assertEqual(self.state['260602']['state'], 'complete')
        self.run_case(AssertionError('resume repeated a command')).assert_not_called()

    # Purpose: Preserve incomplete anatomy outputs instead of retrying.
    # Inputs: Existing anatomy revision with an INCOMPLETE sentinel.
    # Outputs: Case failed, no execution and sentinel bytes preserved.
    def test_incomplete_anatomy_never_skipped_or_replaced(self):
        self.anatomy()
        sentinel = self.config['selection']['anatomy_file'].parent / 'INCOMPLETE'
        sentinel.write_text('leave untouched')
        self.run_case(AssertionError('unsafe retry')).assert_not_called()
        self.assertEqual(self.state['260602']['state'], 'failed')
        self.assertIn('INCOMPLETE', self.state['260602']['error'])
        self.assertEqual(sentinel.read_text(), 'leave untouched')

    # Purpose: Reject stale or incomplete preflight outputs without rerunning.
    # Inputs: Old fingerprint, then INCOMPLETE marker and a matching verification fingerprint.
    # Outputs: No commands for unsafe outputs; matching complete preflight remains verifiable.
    def test_stale_incomplete_and_failed_preflights_never_rerun(self):
        self.anatomy()
        folder = self.preflight(fingerprint='old')
        self.run_case(AssertionError('unsafe rerun')).assert_not_called()
        self.assertIn('stale', self.state['260602']['error'])
        (folder / 'INCOMPLETE').write_text('interrupted')
        self.run_case(AssertionError('unsafe rerun')).assert_not_called()
        self.assertIn('INCOMPLETE', self.state['260602']['error'])
        (folder / 'INCOMPLETE').unlink()
        with patch.object(batch, '_fingerprint', return_value='old'):
            info = batch.verify_preflight(folder, self.config, self.resolved)
        self.assertTrue(info['ready'])

    # Purpose: Keep matching failed preflight evidence available for review.
    # Inputs: Matching preflight with one failed variant and readiness false.
    # Outputs: No retry; case and preflight failed with failed-variant count retained.
    def test_matching_failed_preflight_stays_reviewable(self):
        self.anatomy()
        self.preflight(ready=False)
        self.run_case(AssertionError('unsafe retry')).assert_not_called()
        self.assertEqual(self.state['260602']['state'], 'failed')
        self.assertEqual(self.state['260602']['preflight']['failed_variants'], 1)
        self.assertEqual(self.state['260602']['preflight']['state'], 'failed')

    # Purpose: Reject corrupted preflight artifacts and unsafe inventory paths.
    # Inputs: Changed source-crop JSON, then corrected checksum with ../outside inventory entry.
    # Outputs: ValueError identifies changed artifact or unsafe inventory.
    def test_corrupted_artifact_and_unsafe_inventory_are_rejected(self):
        folder = self.preflight()
        (folder / 'source-crop.json').write_text('{"changed": true}')
        with self.assertRaisesRegex(ValueError, 'changed'):
            batch.verify_preflight(folder, self.config, self.resolved)
        manifest = folder / 'artifact-manifest.json'
        inventory = json.loads(manifest.read_text())
        inventory['source-crop.json'] = batch.sha(folder / 'source-crop.json')
        inventory['../outside'] = '0' * 64
        manifest.write_text(json.dumps(inventory))
        with self.assertRaisesRegex(ValueError, 'Unsafe'):
            batch.verify_preflight(folder, self.config, self.resolved)

    # Purpose: Stop before preflight when input changes during anatomy execution.
    # Inputs: Mock anatomy command creates output and rewrites the INI.
    # Outputs: One command, failed case and changed-during-anatomy error.
    def test_changed_input_during_anatomy_stops_before_preflight(self):
        def execute(command, *args):
            self.anatomy()
            self.config_path.write_text('new input')
            return 0
        self.assertEqual(self.run_case(execute).call_count, 1)
        self.assertEqual(self.state['260602']['state'], 'failed')
        self.assertIn('changed during anatomy', self.state['260602']['error'])

    def project(self, rows):
        path = self.root / 'project.json'
        path.write_text(json.dumps({'schema_version': 'fakect.cohort-project/1', 'cases': rows}))
        return path

    # Purpose: Verify project filtering, checksums and duplicate rejection.
    # Inputs: Two-case project filtered to 260611, unknown selector and duplicate case rows.
    # Outputs: Resolved selected path and project checksum; unknown and duplicate cases rejected.
    def test_project_case_selection_and_duplicate_rejection(self):
        path = self.project([self.row, {'case_id': '260611', 'config': 'other.ini'}])
        _, rows, checksum = batch.load_project(path, self.root, ['260611'])
        self.assertEqual(rows, [{'case_id': '260611', 'config': str(self.root / 'other.ini')}])
        self.assertEqual(checksum, hashlib.sha256(path.read_bytes()).hexdigest())
        with self.assertRaisesRegex(ValueError, 'Unknown'):
            batch.load_project(path, self.root, ['other'])
        self.project([self.row, self.row])
        with self.assertRaisesRegex(ValueError, 'duplicate'):
            batch.load_project(path, self.root)

    # Purpose: Verify one invalid batch case does not prevent another case finishing.
    # Inputs: Two-worker project with valid anatomy for one case and a missing INI for the other.
    # Outputs: One complete and one failed case, compact matching status JSON and no prepare/freeze.
    def test_one_invalid_case_does_not_stop_other_cases_and_status_is_small(self):
        path = self.project([self.row, {'case_id': '260611', 'config': 'missing.ini'}])
        self.anatomy()
        def load(path, **kwargs):
            Path(path).read_bytes()
            return self.config
        with patch.object(batch, 'load_study_config', side_effect=load), \
                patch.object(batch, 'resolve_preview', return_value=self.resolved), \
                patch.object(batch, 'execute', side_effect=AssertionError('no native commands expected')):
            result = batch.run(path, root=self.root, output=self.output, through='anatomy', workers=2)
        self.assertEqual(result['state'], 'failed')
        self.assertEqual(result['failed_cases'], 1)
        self.assertEqual(result['cases']['260602']['state'], 'complete')
        self.assertEqual(result['cases']['260611']['state'], 'failed')
        self.assertEqual(json.loads((self.output / 'status.json').read_text()), result)
        self.assertLess((self.output / 'status.json').stat().st_size, 10000)
        self.assertFalse(result['dataset_prepared'])
        self.assertFalse(result['input_frozen'])

    # Purpose: Reject shared output directories before native execution.
    # Inputs: Two cases loading configuration with the same output paths.
    # Outputs: ValueError reports shared output and execute is never called.
    def test_shared_output_folder_rejected_before_any_native_execution(self):
        path = self.project([self.row, {'case_id': '260611', 'config': 'other.ini'}])
        with patch.object(batch, 'load_study_config', return_value=self.config), \
                patch.object(batch, 'execute') as execute:
            with self.assertRaisesRegex(ValueError, 'share an output'):
                batch.run(path, root=self.root, output=self.output)
        execute.assert_not_called()


if __name__ == '__main__':
    unittest.main()
