"""Portable source/INI generation and separation of ROI review from cohort work."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
sys.path.insert(0,str(ROOT/'scripts'))
from generate_demo import initialize
from fakect_study_config import load_study_config
from fakect_roi import prepare_crop, resolve_preview
from fakect_recipe_study_config import plan_recipe_variants
from train_study import run


class DemoWorkflowTests(unittest.TestCase):
    def setUp(self):
        temporary=tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.output=Path(temporary.name)/'demo'
        self.project=initialize(self.output)

    def test_demo_inputs_match_native_contract_and_keep_review_manual(self):
        self.assertEqual(len(self.project['cases']),3)
        geometries=[]
        for case in self.project['cases']:
            config=load_study_config(case['cohort_ini'])
            self.assertFalse(config['roi']['coordinate_reviewed'])
            self.assertFalse(config['train']['parameters_reviewed'])
            self.assertEqual(config['selection']['source_ids'],())
            variants=plan_recipe_variants(config)
            self.assertEqual(len(variants),9)
            arrays,_=prepare_crop(resolve_preview(config),config)
            self.assertTrue(arrays['selected'].any())
            self.assertFalse(np.any(arrays['selected'] & (arrays['act']==9)))
            np.testing.assert_array_equal(arrays['selected'],arrays['candidates'] & arrays['roi'])
            self.assertAlmostEqual(float(arrays['attenuation_cm_inverse'][arrays['selected']].mean()),.26,places=5)
            geometries.append(hashlib.sha256(arrays['selected'].tobytes()).hexdigest())
        self.assertEqual(len(set(geometries)),3)
        receipt=json.loads((self.output/'source-manifest.json').read_text())
        for relative,digest in receipt['artifacts_sha256'].items():
            self.assertEqual(hashlib.sha256((self.output/relative).read_bytes()).hexdigest(),digest)
        self.assertFalse((self.output/'registry').exists())

    def test_existing_revision_is_never_overwritten(self):
        saved=(self.output/'project.json').read_bytes()
        with self.assertRaises(FileExistsError):initialize(self.output)
        self.assertEqual((self.output/'project.json').read_bytes(),saved)

    def test_metadata_validation_does_not_read_voxels_or_mark_review(self):
        ini=self.project['cases'][0]['cohort_ini']
        with patch('fakect_roi.read_crop',side_effect=AssertionError('No voxel reads')):
            with redirect_stdout(io.StringIO()): result=run(ini,stage='preflight',validate_only=True)
        self.assertFalse(result['native_simulations_checked'])
        self.assertFalse(result['outputs_written'])
        self.assertEqual(result['variant_count'],9)
        self.assertFalse((Path(ini).parent/'preflight-r1').exists())

    def test_preview_override_cannot_leak_to_generation_stage(self):
        for stage in ('preflight','freeze','prepare','plan'):
            with self.subTest(stage=stage), self.assertRaisesRegex(ValueError,'only for the preview'):
                run(self.project['cases'][0]['cohort_ini'],stage=stage,preview_mode='roi-selection')
        self.assertFalse((Path(self.project['cases'][0]['cohort_ini']).parent/'pairs-r1').exists())


if __name__=='__main__':unittest.main()
