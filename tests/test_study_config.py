"""Strict staged-study parsing and the inherited preview/edit input contract."""
import configparser
import io
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_config import load_preview_config
from fakect_segmentation import model_settings
from fakect_study_config import load_study_config
from fakect_training_data import plan_variants


TRAIN = {
    'stage': 'preview', 'dataset_directory': 'outputs/study/dataset-v1',
    'model_directory': 'outputs/study/model-v1', 'target_source_ids': '-1185, 1185',
    'anatomy_family': 'xcat-260602', 'operations': 'erosion, dilation',
    'distances_mm': '1, 2', 'shape_ks': '5, 10', 'include_baseline': 'true',
    'max_variants': '20', 'image_method': 'attenuation_copy_proxy', 'split_mode': 'scenario_only',
    'validation_fraction': '.2', 'test_fraction': '.2', 'split_seed': '123',
}
MODEL = {
    'architecture': 'unet2d', 'patch_size': '64, 64', 'slice_axis': 'k',
    'normalization': 'fixed_clip', 'clip_min': '0', 'clip_max': '.5',
    'epochs': '2', 'batch_size': '4', 'learning_rate': '.001', 'seed': '123',
}


def study_text():
    # Retain the real edit example's ordered tube syntax and inline comments.
    base = (ROOT / 'configs/examples/xcat-edit.ini').read_text()
    base = base.replace('schema_version = fakect.edit/1', 'schema_version = fakect.study/1', 1)
    for section, fields in (('train', TRAIN), ('model', MODEL)):
        base += '\n[' + section + ']\n' + ''.join(f'{key} = {value}\n' for key, value in fields.items())
    return base


def change(text, section, key, value):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',), strict=True)
    parser.optionxform = str
    parser.read_string(text)
    if value is None:
        parser.remove_option(section, key)
    else:
        parser[section][key] = value
    output = io.StringIO()
    parser.write(output)
    return output.getvalue()


class StudyConfigTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name) / 'checkout'
        self.root.mkdir()
        self.input = Path(self.temp.name) / 'outside-checkout.ini'
        self.text = study_text()

    def load(self, text=None):
        self.input.write_text(self.text if text is None else text)
        return load_study_config(self.input, repo_root=self.root)

    def test_typed_study_keeps_shared_roi_and_model_contract(self):
        config = self.load()
        self.assertEqual(config['study']['schema_version'], 'fakect.study/1')
        self.assertEqual(config['train']['target_source_ids'], (-1185, 1185))
        self.assertEqual(config['train']['distances_mm'], (1., 2.))
        self.assertEqual(config['train']['shape_ks'], (5., 10.))
        self.assertTrue(config['train']['include_baseline'])
        self.assertEqual(config['roi']['shape'], 'tube')
        self.assertEqual(len(config['roi']['center_ijk']), 3)
        self.assertFalse(config['roi']['coordinate_reviewed'])
        self.assertEqual(config['preview']['slice_ijk'], None)
        self.assertEqual(config['train']['dataset_directory'], self.root / TRAIN['dataset_directory'])
        self.assertEqual(config['train']['model_directory'], self.root / TRAIN['model_directory'])
        self.assertEqual(model_settings(config)['patch_size'], [64, 64])
        self.assertEqual(model_settings(config)['clip_max'], .5)
        # Parsing never creates or reads the referenced crop/dataset/model outputs.
        self.assertEqual(list(self.root.iterdir()), [])

    def test_all_explicit_stages_parse_and_do_not_change_variant_plan(self):
        reference = None
        for stage in ('preview', 'plan', 'prepare', 'fit'):
            with self.subTest(stage=stage):
                config = self.load(change(self.text, 'train', 'stage', stage))
                self.assertEqual(config['train']['stage'], stage)
                plan = plan_variants(config)
                self.assertEqual(len(plan), 9)
                self.assertEqual(plan[0]['operation'], 'none')
                self.assertEqual(plan[0]['split'], 'train')
                if reference is None:
                    reference = plan
                else:
                    self.assertEqual(plan, reference)
        self.assertEqual(list(self.root.iterdir()), [])

    def test_stage_target_and_list_validation(self):
        for key, value in (
            ('stage', 'train'), ('stage', 'FIT'), ('target_source_ids', ''),
            ('target_source_ids', '1185,1185'), ('target_source_ids', '1185.0'),
            ('target_source_ids', '2147483648'), ('target_source_ids', '-2147483649'),
            ('operations', ''), ('operations', 'none'), ('operations', 'erode'),
            ('operations', 'erosion,erosion'), ('operations', 'dilation,'),
            ('distances_mm', ''), ('distances_mm', '0'), ('distances_mm', '-1'),
            ('distances_mm', '50.1'), ('distances_mm', 'nan'), ('distances_mm', 'inf'),
            ('distances_mm', '1,1.0'), ('shape_ks', ''), ('shape_ks', '0'),
            ('shape_ks', '1,1'), ('include_baseline', 'True'), ('max_variants', '0'),
            ('max_variants', '1001'), ('split_seed', '-1'), ('anatomy_family', '../case'),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load(change(self.text, 'train', key, value))

    def test_split_fractions_and_image_contract_are_explicit(self):
        for key, value in (
            ('split_mode', 'patient_holdout'), ('image_method', 'HU'),
            ('validation_fraction', '0'), ('test_fraction', '0'),
            ('validation_fraction', '.8'), ('test_fraction', '1'),
            ('test_fraction', '-.1'), ('validation_fraction', 'nan'),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load(change(self.text, 'train', key, value))

    def test_uniform_profile_rejects_ignored_shape_parameter_sweep(self):
        text = change(self.text, 'edit', 'profile', 'uniform')
        with self.assertRaisesRegex(ValueError, 'uniform profile ignores shape_k'):
            self.load(text)
        config = self.load(change(text, 'train', 'shape_ks', '10'))
        self.assertEqual(config['train']['shape_ks'], (10.,))
        self.assertEqual(len(plan_variants(config)), 5)

    def test_model_bounds_and_unsupported_architecture_fail_before_any_training(self):
        for key, value in (
            ('architecture', 'unet3d'), ('slice_axis', 'i'), ('normalization', 'per_image'),
            ('patch_size', '16'), ('patch_size', '16,16,16'), ('patch_size', '15,16'),
            ('patch_size', '18,16'), ('patch_size', '516,64'), ('patch_size', '16.0,16'),
            ('clip_min', '.5'), ('clip_max', '-1'), ('clip_max', 'inf'),
            ('epochs', '0'), ('epochs', '10001'), ('batch_size', '0'), ('batch_size', '257'),
            ('learning_rate', '0'), ('learning_rate', 'nan'), ('seed', '-1'),
            ('seed', '2147483648'),
        ):
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                self.load(change(self.text, 'model', key, value))

    def test_clipping_bounds_must_be_representable_in_float32(self):
        for lower, upper in (('1e308', '1.1e308'), ('-1e308', '1e308'), ('0', '1e-300'),
                             ('1', '1.00000001'), ('-3e38', '3e38')):
            with self.subTest(lower=lower, upper=upper):
                text = change(change(self.text, 'model', 'clip_min', lower), 'model', 'clip_max', upper)
                with self.assertRaisesRegex(ValueError, 'float32'):
                    self.load(text)

    def test_unknown_missing_capitalized_and_duplicate_fields_are_rejected(self):
        invalid = (
            change(self.text, 'train', 'unknown_setting', 'x'),
            change(self.text, 'model', 'Architecture', 'unet2d'),
            change(self.text, 'train', 'stage', None),
            change(self.text, 'model', 'normalization', None),
            change(self.text, 'roi', 'radius', '5'),
            self.text + '\n[extra]\nvalue=1\n',
            self.text + '\n[train]\nstage=preview\n',
            self.text + '\nseed=1\n',
        )
        for text in invalid:
            with self.subTest(tail=text[-150:]), self.assertRaises(ValueError):
                self.load(text)

    def test_defaults_multiline_and_environment_path_expansion_are_rejected(self):
        invalid = (
            '[DEFAULT]\nepochs=1\n' + self.text,
            self.text.replace('stage = preview', 'stage = preview\n  fit'),
            self.text.replace('patch_size = 64, 64', 'patch_size = 64,\n  64'),
            change(self.text, 'train', 'dataset_directory', '$HOME/data'),
            change(self.text, 'train', 'model_directory', ''),
        )
        for text in invalid:
            with self.subTest(tail=text[-150:]), self.assertRaises(ValueError):
                self.load(text)

    def test_study_schema_does_not_leak_into_preview_or_edit_parser(self):
        self.load()
        with self.assertRaisesRegex(ValueError, 'schema_version'):
            load_preview_config(self.input, repo_root=self.root)
        old = (ROOT / 'configs/examples/xcat-edit.ini').read_text()
        with self.assertRaisesRegex(ValueError, 'schema_version'):
            self.load(old)

    def test_existing_preview_and_edit_inputs_remain_backward_compatible(self):
        for name in ('xcat-roi.ini', 'xcat-roi-tube.ini', 'xcat-edit.ini'):
            with self.subTest(name=name):
                config = load_preview_config(ROOT / 'configs/examples' / name, repo_root=self.root)
                self.assertIn(config['study']['schema_version'], ('fakect.preview/1', 'fakect.preview/2', 'fakect.edit/1'))
                self.assertNotIn('train', config)
                self.assertNotIn('model', config)
                self.assertEqual('edit' in config, name == 'xcat-edit.ini')
        sphere = (ROOT / 'configs/examples/xcat-roi.ini').read_text()
        parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
        parser.optionxform = str
        parser.read_string(sphere)
        parser['study']['schema_version'] = 'fakect.preview/1'
        parser['roi']['center_ijk'] = '404, 363, 1584'
        parser['roi']['radius_mm'] = '12'
        parser.remove_option('roi', 'shape')
        serialized = io.StringIO()
        parser.write(serialized)
        self.input.write_text(serialized.getvalue())
        legacy = load_preview_config(self.input, repo_root=self.root)
        self.assertEqual(legacy['roi']['shape'], 'sphere')
        self.assertEqual(legacy['roi']['center_ijk'], (404, 363, 1584))

    def test_plan_enforces_variant_limit_and_available_split_sizes(self):
        config = self.load(change(self.text, 'train', 'max_variants', '2'))
        with self.assertRaisesRegex(ValueError, 'exceeds'):
            plan_variants(config)
        text = self.text
        for key, value in (('operations', 'erosion'), ('distances_mm', '1'), ('shape_ks', '10'),
                           ('include_baseline', 'false')):
            text = change(text, 'train', key, value)
        config = self.load(text)
        with self.assertRaisesRegex(ValueError, 'Too few variants'):
            plan_variants(config)


if __name__ == '__main__':
    unittest.main()
