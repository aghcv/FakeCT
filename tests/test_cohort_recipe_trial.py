"""Native corner trials are independent, reproducible, and cannot become freezes."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/research/evaluation'))
import trial_cohort_recipe as trial
import test_recipe_training


class CohortRecipeTrialTests(unittest.TestCase):
    def setUp(self):
        fixture = test_recipe_training.RecipeTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.fixture = fixture
        config = fixture.config
        config['study']['schema_version'] = 'fakect.recipe-cohort/1'
        config.pop('model')
        config['train'].pop('model_directory')
        config['train'].update(split_mode='unassigned', validation_fraction=0., test_fraction=0.,
            freeze_directory=fixture.root / 'freeze', parameters_reviewed=False)
        self.path = fixture.root / 'trial-input.ini'
        self.path.write_bytes(fixture.input_bytes)
        self.output = fixture.root / 'trial'

    def run_trial(self):
        with patch.object(trial, 'load_study_config', return_value=self.fixture.config), redirect_stdout(io.StringIO()):
            return trial.run(self.path, self.output)

    # Purpose: Verify only extreme recipe corners run with preserved native geometry.
    # Inputs: Recipe cohort fixture sweeping growth 0.5-3 mm and native source hashes.
    # Outputs: Two trials, consistent max NPZ, unchanged sources and no full-grid freeze eligibility.
    def test_only_two_corners_execute_and_max_payload_preserves_native_geometry(self):
        before = {name: hashlib.sha256(Path(path).read_bytes()).hexdigest()
                  for name, path in self.fixture.resolved['source_files'].items()}
        with patch.object(trial, 'apply_recipe', wraps=trial.apply_recipe) as apply, \
             patch('fakect_recipe_study_config.validate_recipe_study_plan', side_effect=AssertionError('full grid forbidden')):
            result = self.run_trial()
        self.assertEqual(apply.call_count, 2)
        self.assertEqual([call.args[2]['edits']['grow']['distance_mm'] for call in apply.call_args_list], [.5, 3.])
        self.assertEqual(result['planned_variant_count'], 5)
        self.assertEqual(result['trial_count'], 2)
        self.assertTrue(result['complete'])
        self.assertFalse(result['full_parameter_space_checked'])
        self.assertFalse(result['freeze_eligible'])
        self.assertFalse((self.output / 'preflight.json').exists())
        self.assertFalse((self.output / 'dataset-manifest.json').exists())
        self.assertFalse((self.output / 'INCOMPLETE').exists())
        self.assertEqual((self.output / 'input.ini').read_bytes(), self.path.read_bytes())
        with np.load(self.output / 'max-trial.npz', allow_pickle=False) as data:
            np.testing.assert_array_equal(data['changed_mask'], data['before_labels'] != data['after_labels'])
            self.assertEqual(trial._array_hash(data['after_target'].astype(np.uint8)), result['samples'][-1]['mask_sha256'])
            self.assertEqual(trial._array_hash(data['before_labels']), result['source_crop_sha256']['act'])
            self.assertEqual(data['before_labels'].shape, data['after_target'].shape)
        self.assertTrue(all(hashlib.sha256(Path(path).read_bytes()).hexdigest() == before[name]
                            for name, path in self.fixture.resolved['source_files'].items()))
        with self.assertRaises(FileExistsError):
            self.run_trial()

    # Purpose: Verify a singleton sweep executes only one corner.
    # Inputs: Recipe cohort with only growth distance 1 mm.
    # Outputs: One minimum-and-maximum trial and a saved max-trial NPZ.
    def test_single_corner_runs_once(self):
        self.fixture.config['sweeps']['grow']['distance_mm'] = (1.,)
        with patch.object(trial, 'apply_recipe', wraps=trial.apply_recipe) as apply:
            result = self.run_trial()
        self.assertEqual(apply.call_count, 1)
        self.assertEqual(result['samples'][0]['extreme'], 'minimum_and_maximum')
        self.assertTrue((self.output / 'max-trial.npz').exists())

    # Purpose: Verify failed corners retain diagnostics while the other corner runs.
    # Inputs: Minimum-corner ValueError and successful maximum-corner execution.
    # Outputs: Overall error, failed/ok trial rows, INCOMPLETE marker and maximum NPZ retained.
    def test_failed_corner_keeps_incomplete_diagnostics_and_attempts_other_corner(self):
        original = trial.apply_recipe
        def apply(arrays, resolved, config):
            if config['edits']['grow']['distance_mm'] == .5:
                raise ValueError('diagnostic minimum failure')
            return original(arrays, resolved, config)
        with patch.object(trial, 'apply_recipe', side_effect=apply), self.assertRaisesRegex(ValueError, '1 diagnostic trial'):
            self.run_trial()
        result = json.loads((self.output / 'trial.json').read_text())
        self.assertFalse(result['complete'])
        self.assertEqual([row['status'] for row in result['samples']], ['failed', 'ok'])
        self.assertTrue((self.output / 'INCOMPLETE').exists())
        self.assertTrue((self.output / 'max-trial.npz').exists())


if __name__ == '__main__':
    unittest.main()
