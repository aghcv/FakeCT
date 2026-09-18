"""Independent ordered-recipe cohorts retain lineage and actual native outcomes."""
from contextlib import redirect_stdout
from copy import deepcopy
import hashlib
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import test_training_data
from fakect_recipe import apply_recipe
from fakect_recipe_study_config import plan_recipe_variants, variant_recipe_config
from fakect_recipe_training import preflight_recipe_dataset, prepare_recipe_dataset
from fakect_roi import prepare_crop, resolve_preview
from fakect_segmentation import PatchDataset
from fakect_training_data import (dataset_fingerprint, plan_variants, prepare_training_dataset,
                                  validate_dataset, validate_study_plan)


class RecipeTrainingTests(unittest.TestCase):
    def setUp(self):
        fixture = test_training_data.TrainingDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.config = deepcopy(fixture.config)
        config = self.config
        config['study']['schema_version'] = 'fakect.recipe-study/1'
        config['selection']['source_ids'] = ()
        config['roi']['radius_mm'] = 2.5
        edit = config.pop('edit')
        config['rois'] = {}
        config['recipe'] = {'steps': ('grow', 'shrink'), 'overlap': 'sequential', 'roi_role': 'selection'}
        config['edits'] = {
            'grow': {**edit, 'roi': 'main', 'iterations': 1, 'operation': 'dilation', 'distance_mm': 1.},
            'shrink': {**edit, 'roi': 'main', 'iterations': 1, 'operation': 'erosion', 'distance_mm': .5,
                       'min_volume_ratio': .8, 'preserve_connectivity': True}}
        config['sweeps'] = {'grow': {'distance_mm': (.5, 1., 2., 3.)}}
        for key in ('target_source_ids', 'operations', 'distances_mm', 'shape_ks'):
            config['train'].pop(key)
        config['train'].update(target_scope='selected_lineage', preflight_directory=self.root/'preflight',
                               validation_fraction=.25, test_fraction=.25)
        self.resolved = resolve_preview(config)
        self.input_bytes = b'# Exact user input snapshot\n[study]\nschema_version = fakect.recipe-study/1\n'

    def run_quiet(self, function, *args, **kwargs):
        with redirect_stdout(io.StringIO()):
            return function(*args, **kwargs)

    def test_preflight_runs_all_variants_without_exporting_pairs_or_mutating_sources(self):
        before = {key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                  for key, value in self.resolved['source_files'].items()}
        self.assertEqual(plan_variants(self.config), plan_recipe_variants(self.config))
        self.assertEqual(validate_study_plan(self.config, self.resolved), plan_variants(self.config))
        result = self.run_quiet(preflight_recipe_dataset, self.config, self.resolved, input_bytes=self.input_bytes)
        output = self.config['train']['preflight_directory']
        self.assertTrue(result['complete'])
        self.assertTrue(result['execution_valid'])
        self.assertEqual(result['variant_count'], 5)
        self.assertEqual(result['failed_variants'], 0)
        self.assertTrue(result['fit_ready'])
        self.assertFalse(list(output.rglob('*.npz')))
        self.assertFalse((output/'dataset-manifest.json').exists())
        self.assertFalse((output/'INCOMPLETE').exists())
        self.assertFalse(self.config['train']['dataset_directory'].exists())
        self.assertEqual((output/'input.ini').read_bytes(), self.input_bytes)
        self.assertEqual((output/'source-label-catalog.json').read_bytes(), self.resolved['catalog_bytes'])
        self.assertTrue((output/'preflight.html').is_file())
        self.assertIn('accepted_distance_mm', (output/'preflight.csv').read_text())
        self.assertGreater(result['duplicate_variant_count'], 0)
        for row in result['samples']:
            if row['operation'] == 'recipe':
                self.assertEqual([step['step_name'] for step in row['step_outcomes']], ['grow', 'shrink'])
                self.assertIn('erosion_safeguard', row['step_outcomes'][1])
        after = {key: hashlib.sha256(Path(value).read_bytes()).hexdigest()
                 for key, value in self.resolved['source_files'].items()}
        self.assertEqual(before, after)

    def test_prepare_uses_selected_lineage_and_is_compatible_with_patch_dataset(self):
        result = self.run_quiet(prepare_training_dataset, self.config, self.resolved, input_bytes=self.input_bytes)
        manifest = validate_dataset(self.config, self.resolved)
        self.assertEqual(manifest['sample_count'], 5)
        self.assertEqual(manifest['target_scope'], 'selected_lineage')
        self.assertTrue(manifest['variants_start_from_original'])
        self.assertTrue(all(manifest['geometry_group_counts'][split] > 0 for split in ('train','validation','test')))
        arrays, _ = prepare_crop(self.resolved, self.config)
        original = {key: value.copy() for key, value in arrays.items()}
        variants = {row['variant_id']: row for row in plan_recipe_variants(self.config)}
        saw_growth_outside_roi = False
        for sample in manifest['samples']:
            with np.load(self.config['train']['dataset_directory']/sample['path'], allow_pickle=False) as saved:
                direct = apply_recipe(arrays, self.resolved, variant_recipe_config(self.config, variants[sample['variant_id']]))
                np.testing.assert_array_equal(saved['mask'], direct['target_mask_after'].astype(np.uint8))
                np.testing.assert_array_equal(saved['edited_labels'], direct['edited_labels'])
                np.testing.assert_array_equal(saved['target_origin_index'], direct['target_origin_index'])
                np.testing.assert_array_equal(saved['image'], np.asarray(direct['attenuation_proxy_per_pixel']/.1,dtype=np.float32))
                self.assertFalse(saved['mask'][arrays['candidates'] & ~arrays['selected']].any())
                saw_growth_outside_roi |= bool(saved['mask'][~arrays['roi']].any())
                metadata = json.loads(str(saved['metadata_json']))
                self.assertEqual(metadata['split'], sample['split'])
                self.assertEqual(metadata['step_outcomes'], sample['step_outcomes'])
                self.assertEqual(metadata['mask_sha256'], sample['mask_sha256'])
        self.assertTrue(saw_growth_outside_roi)
        for key, value in arrays.items():
            np.testing.assert_array_equal(value, original[key])
        dataset = PatchDataset(result['manifest_path'], self.config['model'])
        batch = dataset.batch(dataset.patches('validation')[:2])
        self.assertEqual(batch[0].dtype, np.float32)
        self.assertFalse(set(row['variant_id'] for row in manifest['samples'] if row['split']=='test') & dataset.loaded_variants)

    def test_actual_geometry_grouping_is_independent_of_provisional_variant_splits(self):
        first = self.run_quiet(preflight_recipe_dataset, self.config, self.resolved)
        config = deepcopy(self.config)
        config['train']['preflight_directory'] = self.root/'preflight-reordered'
        config['sweeps']['grow']['distance_mm'] = tuple(reversed(config['sweeps']['grow']['distance_mm']))
        planned = [{**row, 'split': 'train'} for row in reversed(plan_recipe_variants(config))]
        with patch('fakect_recipe_training.validate_recipe_study_plan', return_value=planned):
            second = self.run_quiet(preflight_recipe_dataset, config, self.resolved)
        self.assertEqual({row['mask_sha256']: row['split'] for row in first['samples']},
                         {row['mask_sha256']: row['split'] for row in second['samples']})
        self.assertEqual(first['geometry_group_counts'], second['geometry_group_counts'])
        for result in (first, second):
            groups = {}
            for row in result['samples']:
                self.assertEqual(groups.setdefault(row['mask_sha256'],row['split']),row['split'])
                if row['baseline_equivalent']:
                    self.assertEqual(row['split'],'train')

    def test_preflight_records_failure_and_continues_to_later_variants(self):
        calls = []
        def trial(arrays, resolved, config):
            calls.append(config['edits']['grow']['distance_mm'])
            if config['edits']['grow']['distance_mm'] == 1.:
                raise ValueError('Synthetic native failure <must be escaped>')
            return apply_recipe(arrays,resolved,config)
        with patch('fakect_recipe_training.apply_recipe',side_effect=trial):
            result = self.run_quiet(preflight_recipe_dataset,self.config,self.resolved)
        self.assertEqual(len(calls),5)
        self.assertIn(3.,calls)
        self.assertEqual(result['failed_variants'],1)
        self.assertFalse(result['execution_valid'])
        self.assertFalse(result['fit_ready'])
        failures=[row for row in result['samples'] if row['status']=='failed']
        self.assertEqual(failures[0]['error_type'],'ValueError')
        self.assertIn('&lt;must be escaped&gt;',Path(result['report_path']).read_text())

    def test_collapsed_guarded_geometry_reports_empty_holdout_and_preparation_stays_incomplete(self):
        config=deepcopy(self.config)
        config['recipe']['steps']=('shrink',)
        config['edits'].pop('grow')
        config['edits']['shrink'].update(min_volume_ratio=1.,max_backoff_steps=0,preserve_connectivity=True)
        config['sweeps']={'shrink':{'distance_mm':(.5,1.,2.)}}
        preflight=self.run_quiet(preflight_recipe_dataset,config,self.resolved)
        self.assertTrue(preflight['execution_valid'])
        self.assertFalse(preflight['fit_ready'])
        self.assertEqual(preflight['unique_geometry_count'],1)
        self.assertEqual(preflight['duplicate_variant_count'],3)
        self.assertTrue(preflight['split_errors'])
        self.assertEqual(preflight['split_counts']['validation'],0)
        with self.assertRaisesRegex(ValueError,'remains INCOMPLETE'):
            self.run_quiet(prepare_recipe_dataset,config,self.resolved)
        directory=config['train']['dataset_directory']
        self.assertTrue((directory/'INCOMPLETE').exists())
        self.assertTrue((directory/'preflight.json').exists())
        self.assertFalse((directory/'dataset-manifest.json').exists())

    def test_diagnostic_edits_rejected_before_native_io_and_unassigned_sources_before_output(self):
        config=deepcopy(self.config)
        config['edits']['shrink']['assign_surrounding_tissue']=False
        with patch('fakect_recipe_training.prepare_crop',side_effect=AssertionError('No native I/O allowed')):
            with self.assertRaises(ValueError):
                preflight_recipe_dataset(config,self.resolved)
        from fakect_released import RELEASED_LABEL_ID
        source=self.resolved['source_files']['act']
        labels=np.fromfile(source,dtype='<f4').reshape((31,31,31))
        labels[15,15,18]=RELEASED_LABEL_ID
        labels.tofile(source)
        with self.assertRaisesRegex(ValueError,'cannot form training pairs'):
            preflight_recipe_dataset(self.config,resolve_preview(self.config))
        self.assertFalse(self.config['train']['preflight_directory'].exists())

    def test_fingerprint_binds_recipe_sweeps_and_guards_but_not_training_stage_or_display(self):
        expected=dataset_fingerprint(self.config,self.resolved)
        for key in ('iterations','distance_mm','min_volume_ratio'):
            config=deepcopy(self.config)
            config['edits']['shrink'][key]=config['edits']['shrink'].get(key,0)+.1
            self.assertNotEqual(dataset_fingerprint(config,self.resolved),expected)
        config=deepcopy(self.config)
        config['sweeps']['grow']['distance_mm']=(.5,1.,2.,4.)
        self.assertNotEqual(dataset_fingerprint(config,self.resolved),expected)
        config=deepcopy(self.config)
        config['train'].update(stage='fit',dataset_directory=self.root/'other',preflight_directory=self.root/'other-preflight')
        config['model']['epochs']=99
        config['preview']['volume_stride']=3
        self.assertEqual(dataset_fingerprint(config,self.resolved),expected)


if __name__=='__main__':
    unittest.main()
