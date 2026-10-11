import copy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_context_experiment import load_config, plan, freeze, inspect_lock
from fakect_context_volume import ContextVolume, MAX_SOURCE_SLAB_VOXELS


class ExperimentTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.path = self.root/'model.ini'
        text = (ROOT/'archive/research/configs/models/coa-context-diagnostic.ini').read_text()
        text = text.replace('260613:baseline', '')
        self.path.write_text(text)
        self.config = load_config(self.path, repo_root=self.root)
        self.manifest = Path(self.config['data']['manifest'])
        self.manifest.parent.mkdir(parents=True)
        self.manifest.write_text('{}')
        rows = []
        for family in range(6):
            for member in range(2):
                case = f'{family}-{member}'
                rows.append({'sample_id':case+':baseline', 'case_id':case,
                             'anatomy_family':f'family{family}', 'baseline':True,
                             'source_shape_kji':[257,193,193], 'spacing_ijk_mm':[1.,1.,1.],
                             'original_reference_mask_sha256':hashlib.sha256(str(family).encode()).hexdigest()})
        self.dataset = {'samples':rows,'anatomical_extent_reviewed':False}
        self.reader = patch('fakect_context_data.inspect_dataset', side_effect=lambda *a,**kw:copy.deepcopy(self.dataset)).start()
        self.addCleanup(patch.stopall)

    def test_plan_preserves_families_and_counts_full_volume_tiles(self):
        result = plan(self.config)
        self.assertEqual(result['split_counts'], {'train':8,'validation':2,'test':2})
        self.assertEqual(result['patch_extent_kji_mm'], [128.,96.,96.])
        self.assertEqual(result['evaluation_tiles']['test'], 54)
        grid = result['model_grids']['0-0']
        self.assertEqual(grid['native_slab_upper_bound_shape_kji'], [129,97,97])
        self.assertEqual(grid['native_slab_upper_bound_bytes'], 4*129*97*97)
        self.assertEqual(result['memory_bounds']['maximum_native_slab_bytes'], 4*129*97*97)
        self.assertTrue(result['diagnostic'])
        self.assertEqual(len({r['anatomy_family'] for r in result['samples'] if r['split']=='test'}),1)
        self.assertFalse(Path(self.config['experiment']['output_directory']).exists())

    def test_diagnostic_cannot_use_held_out_case(self):
        result = plan(self.config)
        self.config['model']['diagnostic_sample_id'] = next(r['sample_id'] for r in result['samples'] if r['split']=='test')
        with self.assertRaisesRegex(ValueError,'training partition'):
            plan(self.config)

    def test_provisional_extent_requires_explicit_opt_in(self):
        self.config['data']['allow_provisional_anatomy'] = False
        self.assertFalse(plan(self.config)['ready'])

    def test_identical_anatomy_cannot_cross_families_and_splits(self):
        result = plan(self.config)
        a = next(r for r in result['samples'] if r['split']=='train')
        b = next(r for r in self.dataset['samples'] if r['sample_id']==next(r['sample_id'] for r in result['samples'] if r['split']=='test'))
        b['original_reference_mask_sha256'] = a['original_reference_mask_sha256']
        with self.assertRaisesRegex(ValueError,'crosses splits'):
            plan(self.config)

    def test_reference_hash_is_required_and_well_formed(self):
        row = self.dataset['samples'][0]
        for bad in (None, '', 'not-a-hash', 'a'*63, 'A'*64):
            row['original_reference_mask_sha256'] = bad
            with self.assertRaisesRegex(ValueError, 'original_reference_mask_sha256'):
                plan(self.config)
        del row['original_reference_mask_sha256']
        with self.assertRaisesRegex(ValueError, 'original_reference_mask_sha256'):
            plan(self.config)

    def test_native_slab_memory_checked_before_execution(self):
        for row in self.dataset['samples']:
            row['source_shape_kji'] = [2000, 1000, 1000]
            row['spacing_ijk_mm'] = [.2, .2, .2]
        result = plan(self.config)
        self.assertFalse(result['ready'])
        self.assertEqual(len(result['errors']), 12)
        self.assertTrue(all('native interpolation slab bound' in e for e in result['errors']))
        grid = result['model_grids']['0-0']
        self.assertEqual(grid['native_slab_upper_bound_shape_kji'], [637,477,477])
        self.assertGreater(grid['native_slab_upper_bound_voxels'], MAX_SOURCE_SLAB_VOXELS)
        self.assertEqual(result['memory_bounds']['native_slab_budget_bytes'], 4*MAX_SOURCE_SLAB_VOXELS)
        with self.assertRaisesRegex(ValueError, 'native interpolation slab'):
            freeze(self.config, self.path.read_bytes())
        self.assertFalse(Path(self.config['experiment']['output_directory']).exists())

    def test_native_slab_bound_clips_to_small_source(self):
        for row in self.dataset['samples']:
            row['source_shape_kji'] = [20, 15, 10]
            row['spacing_ijk_mm'] = [.2, .3, .4]
        result = plan(self.config)
        self.assertTrue(result['ready'])
        self.assertEqual(result['model_grids']['0-0']['native_slab_upper_bound_shape_kji'], [20,15,10])

    def test_grid_rounding_matches_loader_at_float_boundary(self):
        for row in self.dataset['samples']:
            row['source_shape_kji'] = [7, 13, 19]
            row['spacing_ijk_mm'] = [.3, .2, .1]
        self.config['model']['target_spacing_mm'] = [.7, .3, .7]
        # No payload reads are needed to exercise the loader's geometry method.
        volume = object.__new__(ContextVolume)
        volume.source_shape_kji = (7,13,19)
        volume._native_spacing_kji = np.array([.1,.2,.3])
        expected = volume.model_shape(self.config['model']['target_spacing_mm'])
        self.assertEqual(expected[0], 2)  # ceil(7 * .1 / .7) retains roundoff.
        result = plan(self.config)
        self.assertEqual(tuple(result['model_grids']['0-0']['model_shape_kji']), expected)

    def test_lock_detects_dataset_and_configuration_changes(self):
        raw = self.path.read_bytes()
        freeze(self.config,raw)
        inspect_lock(self.config,raw)
        with self.assertRaisesRegex(ValueError,'INI changed'):
            inspect_lock(self.config,raw+b'\n')
        self.manifest.write_text('{"changed":true}')
        with self.assertRaisesRegex(ValueError,'dataset changed'):
            inspect_lock(self.config,raw)

    def test_cohort_grid_failures_block_freeze(self):
        self.config['experiment']['mode'] = 'cohort'
        failed = {'ready': False, 'errors': ['Material target fragmentation on model grid'],
                  'summary': {'flagged_samples': 1}, 'samples': []}
        with patch('fakect_context_grid.audit_target_grid', return_value=failed) as audit:
            result = plan(self.config)
            self.assertFalse(result['ready'])
            self.assertEqual(result['target_grid_audit'], failed)
            self.assertIn('Material target fragmentation on model grid', result['errors'])
            self.assertEqual(len(audit.call_args.args[0]), 12)
            with self.assertRaisesRegex(ValueError, 'fragmentation'):
                freeze(self.config, self.path.read_bytes())
        self.assertFalse(Path(self.config['experiment']['output_directory']).exists())

    def test_missing_dataset_not_ready(self):
        self.manifest.unlink()
        self.assertFalse(plan(self.config)['ready'])


if __name__ == '__main__':
    unittest.main()
