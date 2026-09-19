"""Generation-only inputs, review freeze, and model-independent pair exports."""
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import test_recipe_study_config
import test_recipe_training
from fakect_cohort_config import SCHEMA, parse_cohort_sections, freeze_cohort, validate_cohort_freeze
from fakect_recipe_study_config import plan_recipe_variants
from fakect_recipe_training import preflight_recipe_dataset, prepare_recipe_dataset
from fakect_training_data import validate_dataset


class CohortPreparationTests(unittest.TestCase):
    def parser(self):
        parser = test_recipe_study_config.RecipeStudyConfigTests().parser()
        parser['study']['schema_version'] = SCHEMA
        parser.remove_section('model')
        for key in ('model_directory', 'split_mode', 'validation_fraction', 'test_fraction', 'split_seed'):
            parser['train'].pop(key)
        parser['train'].update(freeze_directory='outputs/freeze', parameters_reviewed='false')
        return parser

    def fixture(self):
        fixture = test_recipe_training.RecipeTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        config = fixture.config
        config['study']['schema_version'] = SCHEMA
        config.pop('model')
        config['train'].pop('model_directory')
        config['train'].update(split_mode='unassigned', validation_fraction=0., test_fraction=0.,
                               freeze_directory=fixture.root/'freeze', parameters_reviewed=False)
        config['roi']['coordinate_reviewed'] = True
        config['sweeps'] = {'grow': {'distance_mm': (1.,)}}
        fixture.input_bytes = b'[study]\nschema_version=fakect.recipe-cohort/1\n'
        return fixture

    def test_clean_parser_has_no_model_or_user_split_contract(self):
        parser = self.parser()
        before = {key: dict(parser[key]) for key in parser.sections()}
        config = parse_cohort_sections(parser)
        self.assertNotIn('model', config)
        self.assertNotIn('model_directory', config['train'])
        self.assertEqual(config['train']['split_mode'], 'unassigned')
        self.assertEqual({row['split'] for row in plan_recipe_variants(config)}, {'unassigned'})
        self.assertEqual(before, {key: dict(parser[key]) for key in parser.sections()})

    def test_generation_input_rejects_model_settings_and_split_options(self):
        for section, key, value in [('model', 'epochs', '3'), ('train', 'validation_fraction', '.2'),
                                    ('train', 'parameters_reviewed', 'yes'), ('train', 'stage', 'fit')]:
            parser = self.parser()
            if not parser.has_section(section):
                parser.add_section(section)
            parser[section][key] = value
            with self.assertRaises(ValueError):
                parse_cohort_sections(parser)

    def test_prepare_requires_reviewed_matching_native_freeze(self):
        fixture = self.fixture()
        config, resolved, raw = fixture.config, fixture.resolved, fixture.input_bytes
        with self.assertRaisesRegex(ValueError, 'freeze'):
            fixture.run_quiet(prepare_recipe_dataset, config, resolved, input_bytes=raw)
        self.assertFalse(config['train']['dataset_directory'].exists())
        result = fixture.run_quiet(preflight_recipe_dataset, config, resolved, input_bytes=raw)
        self.assertTrue(result['execution_valid'])
        self.assertTrue(result['cohort_ready'])
        self.assertFalse(result['fit_ready'])
        self.assertEqual(result['split_counts'], {'unassigned': 2})
        with self.assertRaisesRegex(ValueError, 'Review'):
            freeze_cohort(config, resolved, raw)
        config['train']['parameters_reviewed'] = True
        freeze_cohort(config, resolved, raw)
        with self.assertRaises(FileExistsError):
            freeze_cohort(config, resolved, raw)
        fixture.run_quiet(prepare_recipe_dataset, config, resolved, input_bytes=raw)
        manifest = validate_dataset(config, resolved)
        self.assertEqual(manifest['split_mode'], 'unassigned')
        self.assertEqual({row['split'] for row in manifest['samples']}, {'unassigned'})
        self.assertIn('cohort-freeze.json', manifest['artifacts_sha256'])
        self.assertFalse((config['train']['dataset_directory']/'INCOMPLETE').exists())

    def test_changed_ini_or_recipe_cannot_reuse_freeze(self):
        fixture = self.fixture()
        config, resolved, raw = fixture.config, fixture.resolved, fixture.input_bytes
        config['train']['parameters_reviewed'] = True
        fixture.run_quiet(preflight_recipe_dataset, config, resolved, input_bytes=raw)
        freeze_cohort(config, resolved, raw)
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_cohort_freeze(config, resolved, raw+b'# changed\n')
        changed = deepcopy(config)
        changed['edits']['grow']['distance_mm'] += .1
        with self.assertRaisesRegex(ValueError, 'changed'):
            validate_cohort_freeze(changed, resolved, raw)

    def test_tampered_preflight_cannot_be_frozen(self):
        fixture = self.fixture()
        config, resolved, raw = fixture.config, fixture.resolved, fixture.input_bytes
        config['train']['parameters_reviewed'] = True
        fixture.run_quiet(preflight_recipe_dataset, config, resolved, input_bytes=raw)
        path = config['train']['preflight_directory']/'preflight.json'
        path.write_text(path.read_text()+'\n')
        with self.assertRaisesRegex(ValueError, 'changed'):
            freeze_cohort(config, resolved, raw)
        self.assertFalse(config['train']['freeze_directory'].exists())


if __name__ == '__main__':
    unittest.main()
