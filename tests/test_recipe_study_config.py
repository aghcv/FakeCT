"""Recipe-study parsing, exact sweep expansion and metadata-only planning."""
import configparser
from copy import deepcopy
import io
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_recipe_config import parse_recipe_sections
from fakect_recipe_study_config import (SCHEMA, load_recipe_study_config, parse_recipe_study_sections,
    plan_recipe_variants, variant_recipe_config, validate_recipe_study_plan)
from fakect_study_config import load_study_config
from test_study_config import MODEL
import test_recipe


TRAIN = {'stage': 'preflight', 'dataset_directory': 'outputs/dataset', 'model_directory': 'outputs/model',
         'preflight_directory': 'outputs/preflight', 'target_scope': 'selected_lineage',
         'anatomy_family': 'xcat-260602', 'include_baseline': 'true', 'max_variants': '100',
         'image_method': 'attenuation_copy_proxy', 'split_mode': 'scenario_only',
         'validation_fraction': '.2', 'test_fraction': '.2', 'split_seed': '123'}


class RecipeStudyConfigTests(unittest.TestCase):
    def setUp(self):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        self.root = Path(temp.name)
        self.path = self.root / 'external-study.ini'

    def parser(self):
        parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                           strict=True, empty_lines_in_values=False)
        parser.optionxform = str
        parser.read_string((ROOT / 'tests/fixtures/thoracic-aorta-recipe.ini').read_text())
        parser['study']['schema_version'] = SCHEMA
        parser['recipe']['roi_role'] = 'selection'
        parser.remove_option('selection', 'source_ids')
        parser.read_dict({'train': TRAIN, 'model': MODEL,
                          'sweep.ascending_expand': {'distance_mm': '0, 1, 2'},
                          'sweep.descending_narrow': {'distance_mm': '0:1:.5'}})
        return parser

    def load(self, parser=None, loader=load_study_config):
        parser = self.parser() if parser is None else parser
        output = io.StringIO()
        parser.write(output)
        self.path.write_text(output.getvalue())
        return loader(self.path, repo_root=self.root)

    def native_config(self):
        _, resolved, config = test_recipe.RecipeTests().fixture()
        config.update(study={'schema_version': SCHEMA, 'name': 'test'},
                      input={'case_id': '001', 'frame': 1}, train=self.load()['train'])
        config['recipe'].update(steps=('grow', 'shrink'), roi_role='selection')
        config['sweeps'] = {'grow': {'distance_mm': (0., 1.)}, 'shrink': {'distance_mm': (0., 1.)}}
        return config, resolved

    def test_loader_dispatch_reuses_recipe_and_model_contract_without_ids_or_source_reads(self):
        parser = self.parser()
        snapshot = {section: dict(parser[section]) for section in parser.sections()}
        config = parse_recipe_study_sections(parser, self.root)
        self.assertEqual(config, self.load(parser, loader=load_recipe_study_config))
        self.assertEqual(config, self.load(parser))
        self.assertEqual({section: dict(parser[section]) for section in parser.sections()}, snapshot)
        recipe_parser = deepcopy(parser)
        for section in ('train', 'model', 'sweep.ascending_expand', 'sweep.descending_narrow'):
            recipe_parser.remove_section(section)
        recipe_parser['study']['schema_version'] = 'fakect.recipe/1'
        recipe = parse_recipe_sections(recipe_parser, self.root)
        actual_recipe = {key: value for key, value in config.items() if key not in ('train', 'model', 'sweeps')}
        actual_recipe['study'] = {**actual_recipe['study'], 'schema_version': 'fakect.recipe/1'}
        self.assertEqual(actual_recipe, recipe)
        self.assertEqual(config['selection']['source_ids'], ())
        self.assertNotIn('target_source_ids', config['train'])
        self.assertEqual(config['train']['target_scope'], 'selected_lineage')
        for key in ('preflight_directory', 'dataset_directory', 'model_directory'):
            self.assertEqual(config['train'][key], self.root / TRAIN[key])
            self.assertFalse(config['train'][key].exists())
        self.assertEqual(config['model']['patch_size'], (64, 64))
        self.assertEqual(config['model']['learning_rate'], .001)
        self.assertEqual(config['sweeps'], {'ascending_expand': {'distance_mm': (0., 1., 2.)},
                                           'descending_narrow': {'distance_mm': (0., .5, 1.)}})

    def test_exact_inclusive_decimal_and_integer_ranges_preserve_endpoints(self):
        for parameter, value, expected in (
                ('distance_mm', '.1:.3:.1', (.1, .2, .3)),
                ('distance_mm', '1:0:-.25', (1., .75, .5, .25, 0.)),
                ('distance_mm', '0, 50', (0., 50.)),
                ('shape_k', '2.5:2.5:.1', (2.5,)),
                ('shape_k', '1e1, 2.5e1', (10., 25.)),
                ('iterations', '1:5:2', (1, 3, 5)),
                ('iterations', '10, 1', (10, 1))):
            with self.subTest(parameter=parameter, value=value):
                parser = self.parser()
                parser['sweep.ascending_expand'] = {parameter: value}
                actual = self.load(parser)['sweeps']['ascending_expand'][parameter]
                self.assertEqual(actual, expected)
                self.assertTrue(all(type(item) is (int if parameter == 'iterations' else float) for item in actual))

    def test_sweeps_reject_malformed_nonfinite_duplicate_and_unbounded_values(self):
        invalid = {'distance_mm': ('', '1,', '1,,2', '1,1.0', '-.1', '50.1', 'nan', 'inf', 'true',
                                    '0:1:.3', '0:1:0', '0:1:-.5', '0:1', '0:1:.5:2', '0:1:.5,2',
                                    ':1:.5', '0:1:', '0:1:1e-9', '1e-999', '1e999', '1\n2'),
                   'shape_k': ('0', '-1', 'nan', '1,1.0', '1:1001:1'),
                   'iterations': ('0', '11', '1.0', '1e0', 'true', '1:4:2', '1:3:.5', '1,1')}
        for parameter, values in invalid.items():
            for value in values:
                with self.subTest(parameter=parameter, value=value), self.assertRaises(ValueError):
                    parser = self.parser()
                    parser['sweep.ascending_expand'] = {parameter: value}
                    self.load(parser)

    def test_required_sections_references_typos_and_legacy_keys_are_strict(self):
        for section, key, value in (('train', 'target_source_ids', '2922'), ('train', 'operations', 'erosion'),
                                   ('train', 'distances_mm', '1,2'), ('train', 'shape_ks', '6'),
                                   ('sweep.ascending_expand', 'distance_m', '1,2'),
                                   ('model', 'Architecture', 'unet2d')):
            with self.subTest(section=section, key=key), self.assertRaisesRegex(ValueError, 'unknown'):
                parser = self.parser()
                parser[section][key] = value
                self.load(parser)
        for section in ('train', 'model'):
            for key in (TRAIN if section == 'train' else MODEL):
                with self.subTest(section=section, missing=key), self.assertRaisesRegex(ValueError, 'missing'):
                    parser = self.parser()
                    parser.remove_option(section, key)
                    self.load(parser)
        for name in ('sweep.missing', 'sweep.', 'sweep.bad name'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                parser = self.parser()
                parser[name] = {'distance_mm': '1,2'}
                self.load(parser)
        parser = self.parser()
        parser['sweep.ascending_expand'] = {}
        with self.assertRaisesRegex(ValueError, 'one or more'):
            self.load(parser)
        parser.remove_section('sweep.ascending_expand')
        parser.remove_section('sweep.descending_narrow')
        with self.assertRaisesRegex(ValueError, 'at least one'):
            self.load(parser)
        parser = self.parser()
        parser['DEFAULT']['iterations'] = '1'
        with self.assertRaisesRegex(ValueError, 'DEFAULT'):
            self.load(parser)

    def test_selection_scope_training_bounds_and_shared_model_rules(self):
        invalid = [('train', 'stage', 'train'), ('train', 'target_scope', 'all_arteries'),
                   ('train', 'include_baseline', 'True'), ('train', 'max_variants', '0'),
                   ('train', 'max_variants', '1001'), ('train', 'split_mode', 'patient'),
                   ('train', 'image_method', 'HU'), ('train', 'validation_fraction', '0'),
                   ('train', 'test_fraction', 'nan'), ('train', 'test_fraction', '.9'),
                   ('train', 'preflight_directory', '$HOME/preflight'), ('train', 'preflight_directory', ''),
                   ('train', 'split_seed', '-1'), ('train', 'anatomy_family', '../bad'),
                   ('recipe', 'roi_role', 'boundary'), ('model', 'patch_size', '18,16'),
                   ('model', 'normalization', 'per_image'), ('model', 'clip_max', '1e999'),
                   ('model', 'learning_rate', 'nan'), ('model', 'epochs', '0')]
        for section, key, value in invalid:
            with self.subTest(section=section, key=key, value=value), self.assertRaises(ValueError):
                parser = self.parser()
                parser[section][key] = value
                self.load(parser)
        parser = self.parser()
        parser.remove_option('recipe', 'roi_role')
        with self.assertRaisesRegex(ValueError, 'roi_role=selection'):
            self.load(parser)
        parser = self.parser()
        parser['edit.ascending_expand']['profile'] = 'uniform'
        parser['sweep.ascending_expand']['shape_k'] = '1,2'
        with self.assertRaisesRegex(ValueError, 'uniform profile ignores shape_k'):
            self.load(parser)

    def test_cartesian_recipes_are_complete_independent_and_zero_distance_disables_guards(self):
        parser = self.parser()
        parser['edit.descending_narrow'].update(min_volume_ratio='.75', preserve_connectivity='true',
                                                backoff_factor='.5', max_backoff_steps='8',
                                                assign_surrounding_tissue='false')
        config = self.load(parser)
        snapshot = deepcopy(config)
        variants = plan_recipe_variants(config)
        self.assertEqual(len(variants), 10)
        self.assertEqual(variants[0]['variant_id'], 'baseline')
        self.assertEqual(variants[0]['operation'], 'none')
        self.assertEqual(variants[0]['split'], 'train')
        self.assertEqual(variants[0]['parameters'], {})
        for edit in variants[0]['edits'].values():
            self.assertEqual((edit['operation'], edit['distance_mm']), ('none', 0.))
            self.assertNotIn('assign_surrounding_tissue', edit)
            self.assertNotIn('min_volume_ratio', edit)
        combinations = set()
        for variant in variants[1:]:
            self.assertEqual(variant['operation'], 'recipe')
            self.assertEqual(set(variant['edits']), set(config['edits']))
            self.assertEqual(variant['edits']['arch_refine'], config['edits']['arch_refine'])
            values = variant['parameters']
            combinations.add((values['ascending_expand']['distance_mm'], values['descending_narrow']['distance_mm']))
            shrink = variant['edits']['descending_narrow']
            if shrink['distance_mm'] == 0:
                self.assertEqual(shrink['operation'], 'none')
                self.assertNotIn('min_volume_ratio', shrink)
                self.assertNotIn('assign_surrounding_tissue', shrink)
            else:
                self.assertEqual(shrink['operation'], 'erosion')
                self.assertEqual(shrink['min_volume_ratio'], .75)
                self.assertIs(shrink['assign_surrounding_tissue'], False)
            run_config = variant_recipe_config(config, variant)
            self.assertEqual(run_config['edits'], variant['edits'])
            self.assertEqual(run_config['recipe']['roi_role'], 'selection')
            run_config['edits']['arch_refine']['distance_mm'] = 45.
            self.assertEqual(variant['edits']['arch_refine']['distance_mm'], 1.)
        self.assertEqual(combinations, {(a, b) for a in (0., 1., 2.) for b in (0., .5, 1.)})
        self.assertEqual(config, snapshot)

    def test_ids_and_splits_ignore_sweep_order_stages_outputs_and_model_but_bind_recipe(self):
        config = self.load()
        planned = plan_recipe_variants(config)
        self.assertEqual({row['split'] for row in planned}, {'train', 'validation', 'test'})
        changed = deepcopy(config)
        changed['sweeps'] = {name: {key: tuple(reversed(values)) for key, values in reversed(list(axes.items()))}
                             for name, axes in reversed(list(config['sweeps'].items()))}
        changed['model']['epochs'] = 50
        changed['output']['directory'] = self.root / 'new-display'
        changed['preview']['volume_stride'] = 7
        for stage in ('preview', 'plan', 'preflight', 'prepare', 'fit'):
            changed['train'].update(stage=stage, dataset_directory=self.root / stage,
                                    model_directory=self.root / ('model-' + stage), preflight_directory=self.root / 'other')
            self.assertEqual(plan_recipe_variants(changed), planned)
        for section, key, value in (('selection', 'source_ids', (2922,)), ('roi', 'radius_mm', (11.,) * 14),
                                   ('reassignment', 'max_distance_mm', 4.)):
            with self.subTest(section=section, key=key):
                changed = deepcopy(config)
                changed[section][key] = value
                updated = plan_recipe_variants(changed)
                self.assertNotEqual([row['variant_id'] for row in updated[1:]],
                                    [row['variant_id'] for row in planned[1:]])
        changed = deepcopy(config)
        changed['edits']['descending_narrow']['min_volume_ratio'] = .75
        guarded = plan_recipe_variants(changed)
        self.assertNotEqual(guarded[-1]['variant_id'], planned[-1]['variant_id'])

    def test_bound_product_before_materialization_and_reject_insufficient_holdouts(self):
        config = self.load()
        config['train']['max_variants'] = 8
        with patch('fakect_recipe_study_config.product', side_effect=AssertionError('must reject before expansion')):
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                plan_recipe_variants(config)

        config['train']['max_variants'] = 1000
        config['sweeps'] = {'ascending_expand': {'distance_mm': tuple(index / 20 for index in range(1000))}}
        with patch('fakect_recipe_study_config.product', side_effect=AssertionError('must reject before expansion')):
            with self.assertRaisesRegex(ValueError, 'exceeds'):
                plan_recipe_variants(config)
        config['sweeps'] = {'ascending_expand': {'distance_mm': (1.,)}}
        with self.assertRaisesRegex(ValueError, 'Too few variants'):
            plan_recipe_variants(config)
        config['train'].update(include_baseline=False, validation_fraction=.2, test_fraction=0.)
        with self.assertRaisesRegex(ValueError, 'Too few variants'):
            plan_recipe_variants(config)
        for key, values in (('distance_mm', (True,)), ('iterations', (1.,)),
                            ('shape_k', (float('nan'),)), ('distance_mm', (1., 1.))):
            config['sweeps'] = {'ascending_expand': {key: values}}
            with self.subTest(key=key, values=values), self.assertRaises(ValueError):
                plan_recipe_variants(config)

    def test_multiple_axes_on_one_edit_multiply_with_other_edit_axes(self):
        parser = self.parser()
        parser['sweep.ascending_expand'] = {'distance_mm': '0,1', 'shape_k': '2,4', 'iterations': '1,2'}
        parser['sweep.descending_narrow'] = {'distance_mm': '.5,1'}
        config = self.load(parser)
        variants = plan_recipe_variants(config)
        self.assertEqual(len(variants), 17)
        combinations = set()
        for variant in variants[1:]:
            grow = variant['parameters']['ascending_expand']
            shrink = variant['parameters']['descending_narrow']
            combinations.add((grow['distance_mm'], grow['shape_k'], grow['iterations'], shrink['distance_mm']))
            self.assertEqual(variant['edits']['ascending_expand']['shape_k'], grow['shape_k'])
            self.assertEqual(variant['edits']['ascending_expand']['iterations'], grow['iterations'])
        self.assertEqual(combinations, {(distance, shape, passes, shrink)
                         for distance in (0., 1.) for shape in (2., 4.)
                         for passes in (1, 2) for shrink in (.5, 1.)})

    def test_metadata_validation_checks_all_variants_and_names_failed_parameters(self):
        config, resolved = self.native_config()
        snapshot = deepcopy(config)
        with patch('numpy.memmap', side_effect=AssertionError('must not read voxel sources')):
            self.assertEqual(validate_recipe_study_plan(config, resolved), plan_recipe_variants(config))
            bad = deepcopy(config)
            bad['sweeps']['grow']['distance_mm'] = (0., 20.)
            with self.assertRaisesRegex(ValueError, 'Variant recipe-.*parameters=.*20.0.*halo'):
                validate_recipe_study_plan(bad, resolved)
        self.assertEqual(config, snapshot)
        config['edits']['idle'] = {**config['edits']['shrink'], 'operation': 'none'}
        config['recipe']['steps'] += ('idle',)
        config['sweeps']['grow']['iterations'] = (1, 10)
        config['sweeps']['shrink']['iterations'] = (1, 10)
        with self.assertRaisesRegex(ValueError, 'Variant recipe-.*parameters=.*total passes'):
            validate_recipe_study_plan(config, resolved)

    def test_training_plan_rejects_active_diagnostic_release_but_allows_disabled_step(self):
        config, resolved = self.native_config()
        config['edits']['shrink']['assign_surrounding_tissue'] = False
        with self.assertRaisesRegex(ValueError, 'Variant recipe-.*edit.shrink.*assign_surrounding_tissue=true'):
            validate_recipe_study_plan(config, resolved)
        config['sweeps']['shrink']['distance_mm'] = (0.,)
        variants = validate_recipe_study_plan(config, resolved)
        for variant in variants:
            self.assertEqual(variant['edits']['shrink']['operation'], 'none')
            self.assertNotIn('assign_surrounding_tissue', variant['edits']['shrink'])


if __name__ == '__main__':
    unittest.main()
