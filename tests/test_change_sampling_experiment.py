"""Guard the TRAIN-only panel, its actual masks and immutable provenance."""
from copy import deepcopy
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

import fakect_change_sampling_experiment as exp


def edited_rows(case):
    rows = []
    for index in range(12):
        for category in ('added', 'removed', 'mixed'):
            rows.append({'sample_id': case+':'+category+'-'+str(index), 'case_id': case,
                         'baseline': False, 'mask_counts': {
                             'added': index+1 if category in ('added', 'mixed') else 0,
                             'removed': index+1 if category in ('removed', 'mixed') else 0}})
    return rows


class ConfigurationTests(unittest.TestCase):
    def load(self, old='', new=''):
        source = (exp.ROOT/'configs/models/aorta-change-sampling-pilot.ini').read_text()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'input.ini'
            path.write_text(source.replace(old, new) if old else source)
            return exp.load_config(path)

    def test_initial_diagnostic_is_small_and_finite(self):
        config = self.load()
        self.assertEqual(config['diagnostic']['case_ids'], ['260602', '260613'])
        self.assertEqual(config['training']['epochs']*config['training']['steps_per_epoch'], 1792)
        self.assertEqual(config['training']['patch_shape_kji'], [96]*3)

    def test_bounds_and_unsupported_inputs_fail(self):
        changes = [('epochs = 64', 'epochs = 0'), ('epochs = 64', 'epochs = 201'),
                   ('steps_per_epoch = 28', 'steps_per_epoch = 27'),
                   ('case_ids = 260602,260613', 'case_ids = 260602,260602'),
                   ('burden_quantiles = 3', 'burden_quantiles = 4'),
                   ('change_weight = 8', 'change_weight = 65'),
                   ('inference_overlap = 24', 'inference_overlap = 64')]
        for old, new in changes:
            with self.subTest(new=new), self.assertRaises(ValueError):
                self.load(old, new)

    def test_output_cannot_modify_reference(self):
        with self.assertRaises(ValueError):
            self.load('outputs/experiments/aorta/change-sampling-pilot-r1',
                      'outputs/experiments/aorta/diverse-change-r2/diagnostic')


class PanelTests(unittest.TestCase):
    def test_balances_case_class_burden_and_is_order_invariant(self):
        rows = edited_rows('a')+edited_rows('b')
        selected, slots = exp.select_panel(rows, ['b', 'a'])
        repeated, other = exp.select_panel(rows[::-1], ['a', 'b'])
        self.assertEqual(selected, repeated)
        self.assertEqual(slots, other)
        self.assertEqual(len(selected), 12)
        self.assertEqual(len({row['sample_id'] for row in selected}), 12)
        self.assertEqual({(row['case_id'], row['mode'], row['burden_bin']) for row in slots},
                         {(case, mode, tier) for case in ('a', 'b')
                          for mode in ('added', 'removed') for tier in range(3)})
        for slot in slots:
            self.assertGreater(slot['actual_mask_counts'][slot['mode']], 0)
            self.assertEqual(slot['mixed_changes'], slot['burden_bin'] == 1)

    def test_zero_burden_is_never_positive(self):
        rows = edited_rows('a')
        rows += [{'sample_id': 'a:noop', 'case_id': 'a', 'baseline': False,
                  'mask_counts': {'added': 0, 'removed': 0}}]
        selected, slots = exp.select_panel(rows, ['a'])
        self.assertNotIn('a:noop', {row['sample_id'] for row in selected})
        self.assertTrue(all(row['target_voxels'] > 0 for row in slots))

    def test_invalid_or_missing_actual_counts_rejected(self):
        for value in (-1, True, 1.5, None):
            rows = edited_rows('a')
            rows[0]['mask_counts']['added'] = value
            with self.subTest(value=value), self.assertRaises(ValueError):
                exp.select_panel(rows, ['a'])

    def test_missing_class_and_duplicate_id_rejected(self):
        rows = [row for row in edited_rows('a') if row['mask_counts']['added'] == 0]
        with self.assertRaises(ValueError):
            exp.select_panel(rows, ['a'])
        with self.assertRaises(ValueError):
            exp.select_panel(edited_rows('a')+[edited_rows('a')[0]], ['a'])


class SourceFixture:
    def __init__(self, folder):
        self.folder = Path(folder)
        self.parent = self.folder/'parent'
        self.parent.mkdir()
        template = exp.load_config(exp.ROOT/'configs/models/aorta-change-sampling-pilot.ini')
        self.config = deepcopy(template)
        self.config['experiment']['output_directory'] = str(self.folder/'diagnostic')
        self.config['reference']['experiment_directory'] = str(self.parent)
        self.config['diagnostic']['case_ids'] = ['a', 'b']
        rows = edited_rows('a')+edited_rows('b')
        rows += [{'sample_id': case+':baseline', 'case_id': case, 'baseline': True,
                  'mask_counts': {'added': 0, 'removed': 0}} for case in ('a', 'b')]
        for index, row in enumerate(rows):
            before, after = np.zeros((8, 8, 8), np.uint8), np.zeros((8, 8, 8), np.uint8)
            removed, added = row['mask_counts']['removed'], row['mask_counts']['added']
            before.reshape(-1)[:removed] = 1
            after.reshape(-1)[32:32+added] = 1
            payload = self.folder/('mask-'+str(index)+'.npz')
            np.savez_compressed(payload, before=before, after=after,
                                added=(after == 1) & (before == 0), removed=(before == 1) & (after == 0))
            row.update(mask_npz=str(payload), mask_npz_sha256=exp.base.sha256(payload), split='unassigned')
        self.rows = rows
        self.stage = {'id': 'family_holdout/xcat_reference_10yr/seed-42/outer_refit',
                      'outer_fold_id': 'xcat_reference_10yr', 'train_case_ids': ['a', 'b'],
                      'train_sample_ids': [row['sample_id'] for row in rows],
                      'validation_case_ids': [], 'final_test_case_ids': ['heldout'],
                      'final_test_sample_ids': ['heldout:edited']}
        self.old = {'schema_version': exp.reference.SCHEMA, 'config': {'training': template['training'],
                     'validation': template['validation']}, 'stages': [self.stage], 'samples': rows,
                    'reserved_case_ids': ['reserved'], 'reserved_test_evaluated': False,
                    'runtime': {'example': 'unchanged'}, 'inputs': 'CT and validity only',
                    'target_semantics': 'unchanged0 added1 removed2'}
        self.stage_folder = self.parent/'stages'/self.stage['id']
        self.stage_folder.mkdir(parents=True)
        self.audit = {'selection_allowed': False, 'samples': [
            {'sample_id': row['sample_id'], 'split': 'train', 'coverage_passed': True,
             'origins_kji': [[10, 20, 30], [40, 50, 60]]} for row in rows]}
        self.save()

    def save(self):
        exp.base.write_json(self.parent/'experiment-lock.json', self.old)
        (self.parent/'input.ini').write_text('immutable source input')
        exp.base.write_json(self.stage_folder/'results.json', {'status': 'completed',
                            'reserved_test_evaluated': False})
        exp.base.write_json(self.stage_folder/'all-training-proposals.json', self.audit)


class PlanTests(unittest.TestCase):
    def test_only_training_panel_and_frozen_crop_origins_enter_plan(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SourceFixture(directory)
            with patch.object(exp.reference, 'inspect_lock'):
                lock = exp.plan(source.config)
            self.assertEqual(len(lock['samples']), 14)
            stage = lock['stages'][0]
            self.assertEqual(stage['validation_sample_ids'], [])
            self.assertEqual(stage['final_test_sample_ids'], [])
            self.assertFalse(stage['final_test_evaluation_permitted'])
            self.assertEqual(len(stage['diagnostic_strata']), 12)
            self.assertTrue(all(row['split'] == 'train' for row in lock['samples']))
            self.assertEqual({tuple(row['origins_kji'][0]) for row in lock['proposals'][stage['id']].values()},
                             {(10, 20, 30)})
            self.assertFalse(lock['classifier_started'])

    def test_heldout_and_reserved_cases_rejected_before_mask_loading(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SourceFixture(directory)
            for prohibited in ('heldout', 'reserved'):
                config = deepcopy(source.config)
                config['diagnostic']['case_ids'] = ['a', prohibited]
                with patch.object(exp.reference, 'inspect_lock'), patch.object(exp, '_verify_masks') as verify:
                    with self.assertRaises(ValueError):
                        exp.plan(config)
                    verify.assert_not_called()

    def test_non_training_proposal_role_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SourceFixture(directory)
            source.audit['samples'][0]['split'] = 'final_test'
            source.save()
            with patch.object(exp.reference, 'inspect_lock'), self.assertRaises(ValueError):
                exp.plan(source.config)

    def test_changed_model_loss_and_gate_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SourceFixture(directory)
            for section, key in [('training', 'learning_rate'), ('validation', 'minimum_dice')]:
                config = deepcopy(source.config)
                config[section][key] *= .5
                with patch.object(exp.reference, 'inspect_lock'), self.assertRaises(ValueError):
                    exp.plan(config)

    def test_actual_payload_burden_mismatch_rejected(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SourceFixture(directory)
            selected, _ = exp.select_panel(source.rows, ['a', 'b'])
            sid = selected[0]['sample_id']
            row = next(row for row in source.rows if row['sample_id'] == sid)
            payload = Path(row['mask_npz'])
            with np.load(payload) as arrays:
                values = {key: arrays[key] for key in arrays.files}
            values['added'].reshape(-1)[-1] = 1
            np.savez_compressed(payload, **values)
            row['mask_npz_sha256'] = exp.base.sha256(payload)
            source.save()
            with patch.object(exp.reference, 'inspect_lock'), self.assertRaises(ValueError):
                exp.plan(source.config)


class ProvenanceTests(unittest.TestCase):
    def test_unsigned_json_still_requires_frozen_hash(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'old.json'
            path.write_text('{"old": true}')
            digest = exp.base.sha256(path)
            self.assertEqual(exp._read_json(path, digest), {'old': True})
            path.write_text('{"old": false}')
            with self.assertRaises(ValueError):
                exp._read_json(path, digest)

    def test_signed_json_requires_sidecar_and_expected_digest(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory)/'new.json'
            exp.base.write_json(path, {'new': True})
            with self.assertRaises(ValueError):
                exp._read_json(path, '0'*64)
            Path(str(path)+'.sha256').write_text('0'*64)
            with self.assertRaises(ValueError):
                exp._read_json(path)

    def test_freeze_and_inspection_detect_source_and_dependency_changes(self):
        with tempfile.TemporaryDirectory() as directory:
            source = SourceFixture(directory)
            code = Path(directory)/'source.py'
            code.write_text('pinned implementation')
            with patch.object(exp.reference, 'inspect_lock'), patch.object(exp, 'ROOT', Path(directory)), \
                    patch.object(exp, 'CODE_FILES', ('source.py',)):
                lock = exp.freeze(source.config, b'new frozen input')
                self.assertEqual(exp.inspect_lock(source.config, b'new frozen input'), lock)
                code.write_text('changed implementation')
                with self.assertRaises(ValueError):
                    exp.inspect_lock(source.config, b'new frozen input')
                code.write_text('pinned implementation')
                path = Path(lock['samples'][0]['mask_npz'])
                path.write_bytes(b'changed payload')
                with self.assertRaises(ValueError):
                    exp.inspect_lock(source.config, b'new frozen input')


if __name__ == '__main__':
    unittest.main()
