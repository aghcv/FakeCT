"""CLI routes recipe grid validation, native preflight and paired export."""
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import test_study_workflow
from fakect_segmentation import PatchDataset, inspect_manifest
from fakect_study_config import load_study_config
from fakect_roi import resolve_preview
from fakect_training_data import validate_dataset


class RecipeStudyWorkflowTests(unittest.TestCase):
    def setUp(self):
        self.fixture = test_study_workflow.StudyWorkflowTests()
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)
        f = self.fixture
        s = f.sections
        s['study']['schema_version'] = 'fakect.recipe-study/1'
        s['selection']['source_ids'] = ''
        s['recipe'] = {'steps': 'grow,shrink', 'roi_role': 'selection', 'overlap': 'sequential'}
        edit = s.pop('edit')
        s['edit.grow'] = {**edit, 'roi': 'main', 'iterations': '1', 'operation': 'dilation'}
        s['edit.shrink'] = {**edit, 'roi': 'main', 'iterations': '1', 'operation': 'erosion',
                            'assign_surrounding_tissue': 'true'}
        s['sweep.grow'] = {'distance_mm': '0:2:1'}
        s['sweep.shrink'] = {'distance_mm': '0,1'}
        for key in ('target_source_ids', 'operations', 'distances_mm', 'shape_ks'):
            s['train'].pop(key)
        s['train'].update(target_scope='selected_lineage',
                           preflight_directory=str(f.root/'preflight'), stage='plan')
        f.write_input()

    def test_validate_only_checks_all_combinations_without_native_execution(self):
        with patch('fakect_recipe.apply_recipe', side_effect=AssertionError('metadata only')), \
                patch('fakect_roi.prepare_crop', side_effect=AssertionError('metadata only')), \
                patch('fakect_segmentation.train_segmentation', side_effect=AssertionError('no fit')):
            result, _ = self.fixture.run_stage('preflight', validate_only=True)
        self.assertEqual(result['variant_count'], 7)
        self.assertFalse(result['native_simulations_checked'])
        self.assertFalse(result['voxel_payloads_read'])
        self.assertFalse(result['outputs_written'])
        self.assertFalse((self.fixture.root/'preflight').exists())

    def test_plan_freezes_named_sweeps_and_roi_target_without_requiring_ids(self):
        self.fixture.run_stage('plan')
        plan = json.loads((self.fixture.output/'plan.json').read_text())
        self.assertEqual(plan['target_scope'], 'selected_lineage')
        self.assertEqual(plan['sweeps']['grow']['distance_mm'], [0., 1., 2.])
        self.assertEqual(plan['recipe_steps'], ['grow', 'shrink'])
        self.assertEqual(plan['variant_count'], 7)
        self.assertNotIn('target_source_ids', plan)
        self.assertEqual((self.fixture.output/'input.ini').read_bytes(), self.fixture.path.read_bytes())

    def test_native_preflight_and_prepare_export_tf_readable_pairs_without_fitting(self):
        original = self.fixture.path.read_bytes()
        with patch('fakect_segmentation.train_segmentation', side_effect=AssertionError('no fit')):
            self.fixture.run_stage('preflight')
            output = self.fixture.root/'preflight'
            self.assertTrue((output/'preflight.json').is_file())
            self.assertFalse(list(output.rglob('*.npz')))
            result, _ = self.fixture.run_stage('prepare')
        self.assertEqual(result['sample_count'], 7)
        config = load_study_config(self.fixture.path)
        manifest = validate_dataset(config, resolve_preview(config))
        manifest_path = self.fixture.dataset/'dataset-manifest.json'
        inspected = inspect_manifest(manifest_path)
        self.assertTrue(all(inspected['split_sizes'].values()))
        group_splits = {}
        for sample in manifest['samples']:
            prior = group_splits.setdefault(sample['mask_sha256'], sample['split'])
            self.assertEqual(prior, sample['split'])
        baseline = next(s for s in manifest['samples'] if s['variant_id'] == 'baseline')
        with np.load(self.fixture.dataset/baseline['path'], allow_pickle=False) as data:
            # Category selection is constrained by the reviewed main ROI.
            self.assertLess(int(data['mask'].sum()), int(np.count_nonzero(self.fixture.labels == -7)))
            self.assertEqual(data['image'].dtype, np.float32)
            self.assertEqual(data['mask'].dtype, np.uint8)
        dataset = PatchDataset(manifest_path, config['model'])
        images, masks, valid = next(dataset.batches('train', epoch=0))
        self.assertEqual(images.shape, masks.shape)
        self.assertEqual(valid.shape, masks.shape)
        self.assertEqual(self.fixture.path.read_bytes(), original)
        self.assertFalse(self.fixture.model.exists())

    def test_recipe_preview_passes_plan_to_existing_renderer(self):
        with patch('preview_roi.run', return_value={'preview': True}) as renderer:
            result, _ = self.fixture.run_stage('preview')
        self.assertEqual(result, {'preview': True})
        kwargs = renderer.call_args.kwargs
        self.assertEqual(kwargs['training_plan']['target_scope'], 'selected_lineage')
        self.assertEqual(kwargs['training_plan']['variant_count'], 7)
        self.assertIn(ROOT/'src/fakect_recipe_training.py', kwargs['extra_code_files'])


if __name__ == '__main__':
    unittest.main()
