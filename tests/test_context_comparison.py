import copy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import fakect_context_comparison as comparison


class ComparisonTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.text = (ROOT/'archive/research/configs/models/coa-context-comparison.ini').read_text()
        self.path = self.root/'input.ini'
        self.path.write_text(self.text)
        self.config = comparison.load_config(self.path, repo_root=self.root)
        manifest = Path(self.config['data']['manifest'])
        manifest.parent.mkdir(parents=True)
        manifest.write_text('{}')
        self.planned = {'ready': True, 'errors': [], 'samples': [], 'diagnostic': False,
            'dataset_manifest': str(manifest), 'dataset_manifest_sha256': comparison.sha256(manifest)}
        for split in ('train', 'validation', 'test'):
            for case in range(8 if split == 'train' else 2):
                for variant in range(16):
                    self.planned['samples'].append(dict(sample_id=f'{split}{case}:{variant}',
                        case_id=f'{split}{case}', split=split, anatomy_family=f'{split}{case//2}',
                        baseline=variant == 0, source_atn=f'{split}{case}',
                        source_shape_kji=[200,160,160], spacing_ijk_mm=[.6,.6,.6]))
        self.reader = patch.object(comparison.original, 'plan',
            side_effect=lambda *a, **kw: copy.deepcopy(self.planned)).start()
        self.addCleanup(patch.stopall)

    # Purpose: Verify comparison arms use equal update budgets and correct TRAIN membership.
    # Inputs: Eight TRAIN cases with sixteen siblings each plus held-out panels.
    # Outputs: 8 versus 128 training samples, 10240 updates per arm, and scheduled validation epochs.
    def test_matched_budget_and_training_membership(self):
        result = comparison.plan(self.config)
        self.assertEqual(len(result['arms']['baseline_only']['training_samples']), 8)
        self.assertEqual(len(result['arms']['baseline_plus_siblings']['training_samples']), 128)
        self.assertEqual(result['arms']['baseline_only']['total_updates'], 10240)
        self.assertEqual(result['arms']['baseline_plus_siblings']['total_updates'], 10240)
        self.assertEqual(result['validation_epochs'], [5,10,15,20])
        self.assertEqual(result['updates_per_case_per_epoch'], 64)
        self.assertEqual(self.reader.call_args.args[0]['experiment']['mode'], 'cohort')
        self.assertNotIn('mode', self.config['experiment'])

    # Purpose: Verify comparison configs reject unsupported settings and uneven case budgets.
    # Inputs: Unknown key, predictions, random/clinical modes, ignored sampling knobs, and 127 updates.
    # Outputs: ValueError for every invalid config and indivisible update budget.
    def test_invalid_or_ignored_inputs_rejected(self):
        for before, after in (
            ('updates_per_epoch = 512', 'unknown = 512'),
            ('save_predictions = false', 'save_predictions = true'),
            ('strategy = holdout', 'strategy = random'),
            ('scope = engineering', 'scope = clinical'),
            ('epochs = 20', 'epochs = 20\nmax_train_batches_per_epoch = 1'),
            ('epochs = 20', 'epochs = 20\npatches_per_volume = 7'),
            ('epochs = 20', 'epochs = 20\ndiagnostic_sample_id = 260613:baseline'),
        ):
            with self.subTest(after=after):
                self.path.write_text(self.text.replace(before, after))
                with self.assertRaises(ValueError):
                    comparison.load_config(self.path, repo_root=self.root)
        self.config['comparison']['updates_per_epoch'] = 127
        with self.assertRaisesRegex(ValueError, 'divisible'):
            comparison.plan(self.config)

    # Purpose: Verify invalid context grids stop freezing before creating output.
    # Inputs: Mock plan with ready false and bad-target-grid error.
    # Outputs: ValueError carrying grid failure and absent experiment directory.
    def test_grid_failure_prevents_output_creation(self):
        self.planned.update(ready=False, errors=['bad target grid'])
        with self.assertRaisesRegex(ValueError, 'bad target grid'):
            comparison.freeze(self.config, self.path.read_bytes())
        self.assertFalse(Path(self.config['experiment']['output_directory']).exists())

    # Purpose: Verify comparison locks pin config, implementation, and dataset bytes.
    # Inputs: Frozen tiny source/manifest then changed INI, source, or manifest.
    # Outputs: Stable valid lock, repeat-freeze refusal, and ValueError for every pinned-byte change.
    def test_lock_detects_source_data_and_input_mutation_and_refuses_overwrite(self):
        fake_source = self.root/'src'/'example.py'
        fake_source.parent.mkdir()
        fake_source.write_text('original')
        with patch.object(comparison, 'ROOT', self.root), patch.object(comparison, 'CODE_FILES', ('src/example.py',)):
            raw = self.path.read_bytes()
            lock = comparison.freeze(self.config, raw)
            self.assertEqual(comparison.inspect_lock(self.config, raw), lock)
            with self.assertRaises(FileExistsError):
                comparison.freeze(self.config, raw)
            with self.assertRaisesRegex(ValueError, 'INI changed'):
                comparison.inspect_lock(self.config, raw+b'\n')
            fake_source.write_text('changed')
            with self.assertRaisesRegex(ValueError, 'implementation changed'):
                comparison.inspect_lock(self.config, raw)
            fake_source.write_text('original')
            manifest = Path(self.config['data']['manifest'])
            manifest.write_text('{"changed":true}')
            with self.assertRaisesRegex(ValueError, 'dataset changed'):
                comparison.inspect_lock(self.config, raw)


if __name__ == '__main__':
    unittest.main()
