"""Optional anatomy selection preserves native ownership and frozen evidence."""
from copy import deepcopy
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
import test_morphology
import test_recipe_training
import test_training_data
from fakect_config import load_preview_config
from fakect_morphology import apply_morphology
from fakect_roi import prepare_crop, resolve_preview, selection_diagnostics
from fakect_tissues import coarse_labels
from fakect_training_data import dataset_fingerprint, _write_anatomical_selection_snapshot


class AnatomyIntegrationTests(unittest.TestCase):
    # Purpose: Verify optional anatomical selection parsing preserves the legacy contract.
    # Inputs: ROI INI with human anatomy names, label lists and invalid selector options.
    # Outputs: Normalized names, defaults and paths; invalid options raise ValueError without outputs.
    def test_optional_human_anatomy_inputs_and_unchanged_legacy_contract(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            path = root/'input.ini'
            template = (ROOT/'configs/examples/xcat-roi.ini').read_text()
            path.write_text(template)
            before = load_preview_config(path, repo_root=root)
            self.assertEqual(set(before['selection']), {'tissue', 'source_ids'})
            extra = ('anatomy = Thoracic Aorta\nanatomical_labels = dias_aorta, left_subclavian\n'
                     'anatomy_file = outputs/anatomy.json\n')
            path.write_text(template.replace('[selection]', '[selection]\n'+extra))
            selection = load_preview_config(path, repo_root=root)['selection']
            self.assertEqual(selection['anatomy'], 'thoracic_aorta')
            self.assertEqual(selection['anatomical_labels'], ('dias_aorta', 'left_subclavian'))
            self.assertEqual(selection['anatomy_file'], root/'outputs/anatomy.json')
            self.assertTrue(selection['protect_other_arteries'])
            self.assertTrue(selection['include_branches'])
            self.assertEqual(selection['connectivity'], 6)
            self.assertFalse((root/'outputs').exists())
            for line in ('connectivity = 18', 'protect_other_arteries = yes',
                         'include_branches = 1', 'anatomical_labels = aorta,'):
                with self.subTest(line=line):
                    path.write_text(template.replace('[selection]', '[selection]\n'+line))
                    with self.assertRaises(ValueError):
                        load_preview_config(path, repo_root=root)

    def _morphology(self, operation):
        fixture = test_morphology.MorphologyTests()
        arrays, resolved, config = fixture.fixture(operation=operation)
        config['reassignment'].update(mode='permissive_except_bone_skin', allowed_tissues=())
        arrays['act'][15, 15, 16] = 8  # A nearby artery excluded from the target.
        if operation == 'erosion':
            arrays['act'][14:17, 14:17, 14:17] = 30
            arrays['act'][15, 15, 15] = -7
            arrays['act'][15, 15, 16] = 8
        resolved.update(source_ids=(-7,), anatomical_protected_ids=(8,))
        arrays['tissue'] = coarse_labels(arrays['act'], resolved['catalog'])
        arrays['candidates'] = arrays['act'] == -7
        arrays['selected'] = arrays['candidates'] & arrays['roi']
        return arrays, resolved, config

    # Purpose: Verify dilation protects an excluded neighboring artery.
    # Inputs: Morphology fixture with target -7 and protected artery 8.
    # Outputs: Neighbor label and attenuation preserved and marked blocked; removing protection permits growth.
    def test_filtered_neighbor_artery_cannot_be_borrowed_by_dilation(self):
        arrays, resolved, config = self._morphology('dilation')
        protected = apply_morphology(arrays, resolved, config)
        self.assertEqual(protected['edited_labels'][15, 15, 16], 8)
        self.assertEqual(protected['attenuation_proxy_per_pixel'][15, 15, 16], arrays['atn'][15, 15, 16])
        self.assertTrue(protected['blocked_mask'][15, 15, 16])
        self.assertEqual(protected['summary']['reassignment_policy']['anatomical_protection']['source_ids'], [8])
        resolved.pop('anatomical_protected_ids')
        unprotected = apply_morphology(arrays, resolved, config)
        self.assertEqual(unprotected['edited_labels'][15, 15, 16], -7)

    # Purpose: Verify erosion cannot use an excluded artery as replacement tissue.
    # Inputs: Erosion fixture with target -7 beside protected artery 8.
    # Outputs: Protected removal stays unresolved; unprotected replacement takes label 8.
    def test_filtered_neighbor_artery_cannot_seed_erosion_reassignment(self):
        arrays, resolved, config = self._morphology('erosion')
        protected = apply_morphology(arrays, resolved, config)
        self.assertTrue(protected['unresolved_mask'][15, 15, 15])
        self.assertEqual(protected['edited_labels'][15, 15, 15], -7)
        resolved.pop('anatomical_protected_ids')
        unprotected = apply_morphology(arrays, resolved, config)
        self.assertEqual(unprotected['edited_labels'][15, 15, 15], 8)

    # Purpose: Reject anatomical protection of targets or nonarterial labels.
    # Inputs: Dilation fixture with protected IDs -7, 20 or unknown 12345.
    # Outputs: ValueError reports protection must stay outside the selected target.
    def test_anatomical_protection_cannot_overwrite_target_or_nonarterial_policy(self):
        arrays, resolved, config = self._morphology('dilation')
        for ids in ((-7,), (20,), (12345,)):
            with self.subTest(ids=ids):
                resolved['anatomical_protected_ids'] = ids
                with self.assertRaisesRegex(ValueError, 'outside the selected target'):
                    apply_morphology(arrays, resolved, config)

    def _training_fixture(self):
        fixture = test_training_data.TrainingDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        return fixture

    # Purpose: Verify manual anatomy resolution filters labels without reading voxels.
    # Inputs: Training fixture selecting aorta by name and crop reads mocked to fail.
    # Outputs: Metadata selects -7, protects 8 and yields the expected ROI mask.
    def test_manual_anatomical_name_filters_roi_without_reading_voxels_in_resolution(self):
        fixture = self._training_fixture()
        config = deepcopy(fixture.config)
        config['selection'].update(source_ids=(), anatomical_labels=('aorta',))
        with patch('fakect_roi.read_crop', side_effect=AssertionError('metadata resolver read voxels')):
            resolved = resolve_preview(config)
        self.assertEqual(tuple(resolved['source_ids']), (-7,))
        self.assertIn(8, resolved['anatomical_protected_ids'])
        arrays, _ = prepare_crop(resolved, config)
        np.testing.assert_array_equal(arrays['selected'], (arrays['act'] == -7) & arrays['roi'])
        self.assertIn('anatomical_selection', selection_diagnostics(arrays, resolved))
        base = resolve_preview(config, apply_anatomy=False)
        self.assertEqual(set(base['source_ids']), {-7, 8})
        self.assertNotIn('anatomical_selection', base)

    # Purpose: Verify anatomical audit and protection affect dataset identity.
    # Inputs: Resolved aorta fixture with protection or audit changed and a snapshot directory.
    # Outputs: Both changes alter fingerprint; snapshot JSON equals anatomical selection.
    def test_selection_audit_and_protection_change_dataset_identity(self):
        fixture = self._training_fixture()
        config = deepcopy(fixture.config)
        config['selection'].update(source_ids=(), anatomical_labels=('aorta',))
        resolved = resolve_preview(config)
        before = dataset_fingerprint(config, resolved)
        changed = deepcopy(resolved)
        changed['anatomical_protected_ids'] = ()
        self.assertNotEqual(dataset_fingerprint(config, changed), before)
        changed = deepcopy(resolved)
        changed['anatomical_selection']['integration_test_audit'] = 'changed'
        self.assertNotEqual(dataset_fingerprint(config, changed), before)
        output = fixture.root/'snapshot'
        output.mkdir()
        _write_anatomical_selection_snapshot(output, resolved)
        self.assertEqual(json.loads((output/'anatomical-selection.json').read_bytes()),
                         resolved['anatomical_selection'])

    # Purpose: Verify recipe preflight publishes anatomical selection provenance.
    # Inputs: Recipe training fixture with aorta selection and frozen input bytes.
    # Outputs: Valid preflight, checksum inventoried selection snapshot and matching data provenance.
    def test_recipe_preflight_saves_anatomical_selection_snapshot(self):
        fixture = test_recipe_training.RecipeTrainingTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        fixture.config['selection'].update(anatomical_labels=('aorta',))
        fixture.resolved = resolve_preview(fixture.config)
        from fakect_recipe_training import preflight_recipe_dataset
        result = fixture.run_quiet(preflight_recipe_dataset, fixture.config, fixture.resolved,
                                   input_bytes=fixture.input_bytes)
        self.assertTrue(result['execution_valid'])
        output = fixture.config['train']['preflight_directory']
        data = (output/'anatomical-selection.json').read_bytes()
        inventory = json.loads((output/'artifact-manifest.json').read_text())
        self.assertEqual(inventory['anatomical-selection.json'], hashlib.sha256(data).hexdigest())
        provenance = json.loads((output/'data-config.json').read_text())
        self.assertIn('anatomical_selection', provenance)
        self.assertIn('fakect_anatomy.py', provenance['code_sha256'])
        self.assertIn('fakect_anatomy_surfaces.py', provenance['code_sha256'])


if __name__ == '__main__':
    unittest.main()
