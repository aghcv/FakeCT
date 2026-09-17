"""Exercise staged CLI orchestration on synthetic files, without model training."""
import base64
import configparser
from contextlib import redirect_stdout
import hashlib
from html.parser import HTMLParser
import importlib.util
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_roi import resolve_preview
from fakect_study_config import load_study_config
from fakect_training_data import validate_dataset


class _Images(HTMLParser):
    def __init__(self, document):
        super().__init__()
        self.sources = []
        self.feed(document)

    def handle_starttag(self, tag, attributes):
        if tag == 'img':
            self.sources.append(dict(attributes).get('src', ''))


class StudyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        case = self.root / 'case001'
        case.mkdir()
        shape = (31, 31, 31)
        k, j, i = np.indices(shape)
        self.labels = np.full(shape, 20, dtype='<f4')
        self.labels[(i-15)**2 + (j-15)**2 <= 4] = -7
        self.labels[:, 2:4, 2:4] = 8
        scalar = np.where(self.labels == -7, .026, .019).astype('<f4')
        self.labels.tofile(case / 'case001_act_1.bin')
        scalar.tofile(case / 'case001_atn_1.bin')
        par, log = case / 'case001.par', case / 'case001.log'
        par.write_text('synthetic source geometry')
        log.write_text('synthetic source scalar metadata')
        catalog = {'policy_version': 'workflow-test', 'sources': {},
                   'categories': [{'id': code, 'name': name} for code, name in
                                  ((0, 'background'), (1, 'soft_tissue'), (5, 'artery'), (255, 'unknown'))],
                   'records': [{'original_id': value, 'original_name': name,
                                'classification': {'tissue_id': code, 'tissue_name': tissue}}
                               for value, name, code, tissue in
                               ((0, 'background', 0, 'background'), (20, 'soft', 1, 'soft_tissue'),
                                (-7, 'synthetic_aorta', 5, 'artery'), (8, 'other_artery', 5, 'artery'))]}
        self.catalog = self.root / 'catalog.json'
        self.catalog.write_text(json.dumps(catalog))
        audit = {'cases': [{'case_id': 'case001', 'directory': str(case), 'shape_kji': shape,
                           'spacing_ijk_mm': [1., 1., 1.], 'par_path': str(par), 'log_path': str(log)}],
                 'source_metadata_sha256': {str(path): hashlib.sha256(path.read_bytes()).hexdigest()
                                            for path in (par, log)}}
        self.audit = self.root / 'audit.json'
        self.audit.write_text(json.dumps(audit))
        self.output = self.root / 'report'
        self.dataset = self.root / 'dataset'
        self.model = self.root / 'model'
        self.path = self.root / 'study.ini'
        self.sections = {
            'study': {'schema_version': 'fakect.study/1', 'name': 'synthetic-aorta-workflow'},
            'input': {'root': str(self.root), 'case_id': 'case001', 'frame': '1',
                      'audit': str(self.audit), 'catalog': str(self.catalog)},
            'selection': {'tissue': 'artery', 'source_ids': '-7'},
            'roi': {'shape': 'sphere', 'center_ijk': '15,15,15', 'radius_mm': '5',
                    'crop_half_width_mm': '15', 'coordinate_reviewed': 'false'},
            'preview': {'slice_ijk': 'roi', 'overlay_opacity': '.22', 'volume_opacity': '.25',
                        'context_tissues': 'artery,soft_tissue', 'context_opacity': '.08', 'volume_stride': '2'},
            'edit': {'operation': 'erosion', 'distance_mm': '1', 'profile': 'uniform',
                     'profile_axis': 'k', 'shape_k': '10', 'shape_window': '0,1'},
            'reassignment': {'allowed_tissues': 'soft_tissue', 'max_distance_mm': '3', 'unresolved': 'preserve'},
            'output': {'directory': str(self.output)},
            'train': {'stage': 'preview', 'dataset_directory': str(self.dataset),
                      'model_directory': str(self.model), 'target_source_ids': '-7',
                      'anatomy_family': 'synthetic-case001', 'operations': 'erosion,dilation',
                      'distances_mm': '.5,1,2', 'shape_ks': '10', 'include_baseline': 'true',
                      'max_variants': '10', 'image_method': 'attenuation_copy_proxy',
                      'split_mode': 'scenario_only', 'validation_fraction': '.2',
                      'test_fraction': '.2', 'split_seed': '42'},
            'model': {'architecture': 'unet2d', 'patch_size': '16,16', 'slice_axis': 'k',
                      'normalization': 'fixed_clip', 'clip_min': '0', 'clip_max': '.3',
                      'epochs': '1', 'batch_size': '2', 'learning_rate': '.001', 'seed': '42'}}
        self.write_input()

    def write_input(self):
        parser = configparser.ConfigParser(interpolation=None)
        parser.optionxform = str
        parser.read_dict(self.sections)
        text = io.StringIO()
        text.write('# Synthetic fixture, including comments for the exact saved snapshot.\n')
        parser.write(text)
        self.path.write_bytes(text.getvalue().encode())

    def run_stage(self, stage=None, validate_only=False):
        # Import beneath each test's mocks so local and module-level routing
        # both exercise the actual CLI without permitting an accidental fit.
        spec = importlib.util.spec_from_file_location('fakect_test_study_cli', ROOT / 'scripts/train_study.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        captured = io.StringIO()
        with redirect_stdout(captured):
            result = module.run(self.path, stage=stage, validate_only=validate_only)
        return result, captured.getvalue()

    def test_preview_embeds_full_target_and_correct_study_command_without_preparing_or_fitting(self):
        # An explicit preview override must remain a preview when rerun from
        # the report, even though the preserved source INI requests fitting.
        self.path.write_text(self.path.read_text().replace('stage = preview', 'stage = fit'))
        original = self.path.read_bytes()
        with patch('fakect_training_data.prepare_training_dataset', side_effect=AssertionError('preview cannot prepare')), \
                patch('fakect_segmentation.train_segmentation', side_effect=AssertionError('preview cannot fit')):
            self.run_stage(stage='preview')
        html = (self.output / 'report.html').read_text()
        self.assertIn('scripts/train_study.py', html)
        self.assertNotIn('python3 scripts/preview_roi.py --config', html)
        report = json.loads((self.output / 'preview-report.json').read_text())
        self.assertTrue(report['rerun_command'].endswith('--stage preview'))
        self.assertEqual(report['training_plan']['stage_override'], 'preview')
        target_report = report['training_plan']['target_preview']
        self.assertEqual(target_report['target_voxels_in_crop'], int(np.count_nonzero(self.labels == -7)))
        self.assertGreater(target_report['target_voxels_outside_roi'], 0)
        target_png = (self.output / 'training-target.png').read_bytes()
        embedded = [base64.b64decode(source.split(',', 1)[1]) for source in _Images(html).sources
                    if source.startswith('data:image/png;base64,')]
        self.assertIn(target_png, embedded)
        self.assertEqual((self.output / 'input.ini').read_bytes(), original)
        self.assertFalse(self.dataset.exists())
        self.assertFalse(self.model.exists())
        # The complete original aorta passes through the crop faces, far beyond
        # the editing sphere. Saved paired-design evidence must retain it.
        with np.load(self.output / 'crop.npz', allow_pickle=False) as data:
            target = np.isin(data['original_labels'], (-7,))
            self.assertGreater(np.count_nonzero(target & ~data['roi_mask']), 0)
            np.testing.assert_array_equal(target, self.labels == -7)

    def test_plan_reads_metadata_only_and_preserves_input_snapshot(self):
        original = self.path.read_bytes()
        with patch('fakect_roi.read_crop', side_effect=AssertionError('plan cannot read voxels')), \
                patch('fakect_training_data.prepare_crop', side_effect=AssertionError('plan cannot read crops')), \
                patch('fakect_segmentation.train_segmentation', side_effect=AssertionError('plan cannot fit')):
            self.run_stage('plan')
        plan = json.loads((self.output / 'plan.json').read_text())
        self.assertEqual(plan['split_mode'], 'scenario_only')
        self.assertEqual(plan['variant_count'], 7)
        self.assertEqual(plan['target_source_ids'], [-7])
        self.assertEqual(plan['crop_shape_kji'], [31, 31, 31])
        self.assertFalse(plan['dataset_prepared'])
        self.assertFalse(plan['model_fitted'])
        self.assertEqual((self.output / 'input.ini').read_bytes(), original)
        self.assertTrue((self.output / 'resolved-config.json').is_file())
        self.assertFalse((self.output / 'crop.npz').exists())
        self.assertFalse(self.dataset.exists())
        self.assertFalse(self.model.exists())

    def test_prepare_is_explicit_and_freezes_exact_input_and_complete_binary_pairs(self):
        original = self.path.read_bytes()
        with patch('fakect_segmentation.train_segmentation', side_effect=AssertionError('prepare cannot fit')):
            self.run_stage('prepare')
        config = load_study_config(self.path)
        manifest = validate_dataset(config, resolve_preview(config))
        self.assertEqual((self.dataset / 'input.ini').read_bytes(), original)
        self.assertEqual(manifest['sample_count'], 7)
        self.assertEqual(manifest['input_config_sha256'], hashlib.sha256(original).hexdigest())
        self.assertFalse(self.model.exists())
        baseline = next(sample for sample in manifest['samples'] if sample['variant_id'] == 'baseline')
        with np.load(self.dataset / baseline['path'], allow_pickle=False) as data:
            np.testing.assert_array_equal(data['mask'], self.labels == -7)
            self.assertTrue(data['mask'][0, 15, 15])
            self.assertFalse(data['mask'][0, 2, 2])

    def test_fit_requires_dataset_validation_first_and_never_prepares_implicitly(self):
        events = []
        def validate(*args, **kwargs):
            events.append('validate')
            return {'schema_version': 'fakect.training-dataset/1', 'samples': []}
        def fit(*args, **kwargs):
            events.append('fit')
            self.model.mkdir()
            return {'best_epoch': 1, 'best_validation_loss': .25}
        with patch('fakect_training_data.validate_dataset', side_effect=validate), \
                patch('fakect_training_data.prepare_training_dataset', side_effect=AssertionError('fit cannot prepare')), \
                patch('fakect_segmentation.train_segmentation', side_effect=fit):
            self.run_stage('fit')
        self.assertEqual(events, ['validate', 'fit'])
        self.assertFalse(self.dataset.exists())
        self.assertEqual((self.model / 'input.ini').read_bytes(), self.path.read_bytes())
        self.assertTrue((self.model / 'artifact-manifest.json').is_file())
        events.clear()
        with patch('fakect_training_data.validate_dataset', side_effect=ValueError('dataset not prepared')), \
                patch('fakect_training_data.prepare_training_dataset', side_effect=AssertionError('fit cannot prepare')), \
                patch('fakect_segmentation.train_segmentation') as fit_mock:
            with self.assertRaisesRegex(ValueError, 'dataset not prepared'):
                self.run_stage('fit')
            fit_mock.assert_not_called()

    def test_validate_only_creates_no_outputs_or_crop_reads_at_any_stage(self):
        with patch('fakect_roi.read_crop', side_effect=AssertionError('validate-only cannot read voxels')), \
                patch('fakect_training_data.prepare_crop', side_effect=AssertionError('validate-only cannot read crops')), \
                patch('fakect_training_data.prepare_training_dataset', side_effect=AssertionError('validate-only cannot prepare')), \
                patch('fakect_segmentation.train_segmentation', side_effect=AssertionError('validate-only cannot fit')):
            for stage in ('preview', 'plan', 'prepare', 'fit'):
                with self.subTest(stage=stage):
                    result, _ = self.run_stage(stage, validate_only=True)
                    self.assertEqual(result['stage'], stage)
                    self.assertFalse(result['outputs_written'])
                    self.assertFalse(result['voxel_payloads_read'])
        self.assertFalse(self.output.exists())
        self.assertFalse(self.dataset.exists())
        self.assertFalse(self.model.exists())


if __name__ == '__main__':
    unittest.main()
