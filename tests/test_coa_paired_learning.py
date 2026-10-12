"""Balanced sibling updates, using metadata and tiny synthetic CT fixtures.

The real TensorFlow test runs two epochs, four paired optimizer updates, and
never opens a cohort or reserved test payload. Earlier frozen tests/code are
left unchanged.
"""
from contextlib import ExitStack
from copy import deepcopy
import importlib.util
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'tests'))

import fakect_coa_paired_experiment as exp
import fakect_coa_paired_learning as learning
from test_coa_focused_experiment import fixture_config, fixture_contract
from test_multiscale_experiment_review import MiniVolume, mini_overview


def paired_config(root):
    c = fixture_config(root)
    c['experiment']['schema_version'] = 'fakect.coa-paired/1'
    c['locator']['focused_experiment_directory'] = str(Path(root) / 'focused-reference')
    return c


class PairedMetadataTests(unittest.TestCase):
    def setUp(self):
        directory = tempfile.TemporaryDirectory()
        self.addCleanup(directory.cleanup)
        self.root = Path(directory.name)
        self.c = paired_config(self.root)
        self.lock = fixture_contract(self.root)
        self.rows = exp.select_diagnostic(self.c, self.lock)

    # Purpose: Verify sibling pairing preserves case/split identity and class order.
    # Inputs: Reversed four-sibling diagnostic list and complete 40-row development panel.
    # Outputs: Two ordered diagnostic pairs and twenty same-case/split panel pairs.
    def test_pairs_keep_case_identity_and_negative_positive_order(self):
        pairs = learning.pair_rows(list(reversed(self.rows)))
        self.assertEqual(len(pairs), 2)
        self.assertEqual({negative['case_id'] for negative, _ in pairs}, {'260613', '260616'})
        for negative, positive in pairs:
            self.assertEqual(negative['case_id'], positive['case_id'])
            self.assertEqual(negative['split'], positive['split'])
            self.assertIs(negative['annotation']['coa_benchmark_positive'], False)
            self.assertIs(positive['annotation']['coa_benchmark_positive'], True)
            self.assertEqual((negative['pilot_role'], positive['pilot_role']),
                             ('edited_negative', 'higher_positive'))
        # Development evaluation also pairs originals/lower positives, without
        # ever pairing records across anatomy families or partitions.
        panel_pairs = learning.pair_rows(self.lock['samples'])
        self.assertEqual(len(panel_pairs), 20)
        self.assertTrue(all(a['case_id'] == b['case_id'] and a['split'] == b['split']
                            for a, b in panel_pairs))

    # Purpose: Verify pairs cannot mix partitions, roles, classes, or companion edits.
    # Inputs: Test/mixed splits, missing/duplicate records, wrong class, and altered companion distance.
    # Outputs: ValueError for every invalid or confounded sibling panel.
    def test_invalid_or_confounded_pairs_are_rejected(self):
        for mutation in ('test', 'mixed_split', 'missing', 'duplicate', 'class', 'companions'):
            with self.subTest(mutation=mutation):
                rows = deepcopy(self.rows)
                if mutation == 'test':
                    rows[0]['split'] = 'test'
                elif mutation == 'mixed_split':
                    rows[0]['split'] = 'validation'
                elif mutation == 'missing':
                    rows.pop()
                elif mutation == 'duplicate':
                    rows.append(deepcopy(rows[0]))
                elif mutation == 'class':
                    rows[1]['annotation']['coa_benchmark_positive'] = False
                else:
                    rows[1]['parameters']['arch_outer_expand']['distance_mm'] = 4.
                with self.assertRaises(ValueError):
                    learning.pair_rows(rows)

    # Purpose: Verify paired predictors contain only ordered negative and positive images.
    # Inputs: Two MiniVolume siblings with 4/16 removals and identical two-origin proposals.
    # Outputs: Float32 2x32x32x32x2 channels matching each single input and only two samples accessed.
    def test_paired_inputs_use_only_two_image_samples_in_order(self):
        runtime = deepcopy(self.lock['parent_config'])
        runtime['input.detail']['spacing_ijk_mm'] = [1., 1., 1.]
        negative, positive = self.rows[:2]
        negative, positive = deepcopy(negative), deepcopy(positive)
        negative['test_removed'], positive['test_removed'] = 4, 16
        proposal = {'origins_kji': [[-8, -8, -8], [-4, -8, -8]]}
        proposals = {negative['sample_id']: proposal, positive['sample_id']: proposal}
        MiniVolume.accessed = []
        with patch.object(learning, 'ContextVolume', MiniVolume):
            reader = learning.FineInputs(runtime, 2)
            values = learning.pair_inputs(reader, negative, positive, proposals)
            self.assertEqual(set(values), {'detail_0', 'detail_1'})
            for key, value in values.items():
                self.assertEqual(value.shape, (2, 32, 32, 32, 2))
                self.assertEqual(value.dtype, np.float32)
                np.testing.assert_array_equal(value[:1], reader.classifier(negative, proposal)[key])
                np.testing.assert_array_equal(value[1:], reader.classifier(positive, proposal)[key])
            self.assertTrue(any(np.any(value[0, ..., 0] != value[1, ..., 0]) for value in values.values()))
        self.assertEqual(set(MiniVolume.accessed), {negative['sample_id'], positive['sample_id']})

    # Purpose: Verify paired-fit smoke records require batch size two.
    # Inputs: Valid pinned smoke then missing batch_size or batch_size 1.
    # Outputs: Valid batch-two smoke accepted and ValueError for absent/wrong batch size.
    def test_batch_one_smoke_cannot_authorize_paired_training(self):
        out = Path(self.c['experiment']['output_directory'])
        out.mkdir()
        (out / 'experiment-lock.json').write_text('paired synthetic lock\n')
        detail = self.lock['parent_config']['input.detail']
        smoke = {'status': 'passed', 'checkpoint_parity': True,
                 'source_lock_sha256': exp.sha256(out / 'experiment-lock.json'),
                 'detail_shape_kji': detail['shape_kji'], 'proposals': detail['proposals'],
                 'traces': {'train': 1, 'predict': 1}, 'gpu_visible': True, 'batch_size': 2}
        exp.write_json(out / 'runtime-smoke.json', smoke)
        self.assertEqual(exp.require_smoke(self.c, self.lock), smoke)
        for batch in (None, 1):
            with self.subTest(batch=batch):
                invalid = deepcopy(smoke)
                if batch is None:
                    invalid.pop('batch_size')
                else:
                    invalid['batch_size'] = batch
                exp.write_json(out / 'runtime-smoke.json', invalid)
                with self.assertRaises(ValueError):
                    exp.require_smoke(self.c, self.lock)


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class PairedTensorFlowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_coa_paired_steps as network
        import fakect_multiscale_network_v3 as locator_network
        cls.network, cls.locator_network = network, locator_network
        cls.tf = locator_network._tensorflow()

    def setUp(self):
        self.tf.keras.backend.clear_session()
        MiniVolume.accessed = []
        MiniVolume.returned_mask_value = None

    # Purpose: Verify real TensorFlow paired diagnostics train balanced batches with independent inference.
    # Inputs: Two epochs on two sibling pairs using saved locator and observed optimizer steps.
    # Outputs: Four [0,1] updates, unchanged locator, checkpoint parity, and predictions independent of the mate.
    def test_two_epoch_diagnostic_uses_four_balanced_updates_and_frozen_locator(self):
        from test_multiscale_v3_integration import config as parent_config
        import fakect_multiscale_experiment_v3 as geometry
        from fakect_tensorflow_steps import StableSteps

        def stable_overview(row, c):
            return mini_overview({**row, 'test_removed': 4}, c)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            c, lock = paired_config(root), fixture_contract(root)
            c['training']['require_gpu'] = False
            c['model']['base_filters'] = 2
            c['diagnostic']['epochs'] = 2
            lock['parent_config'] = parent_config(root / 'parent')
            for row in lock['samples']:
                row['test_removed'] = {'original': 0, 'edited_negative': 4,
                                       'lower_positive': 12, 'higher_positive': 16}[row['pilot_role']]
            selected = exp.select_diagnostic(c, lock)
            lock['diagnostic_sample_ids'] = [row['sample_id'] for row in selected]
            out = Path(c['experiment']['output_directory'])
            out.mkdir()
            (out / 'input.ini').write_text('synthetic paired diagnostic config\n')
            locator = self.locator_network.build_locator((16, 16, 16), base_filters=2, seed=42)
            checkpoint = Path(lock['locator_checkpoint']['path'])
            locator.save(checkpoint)
            locator_hash = exp.sha256(checkpoint)
            lock['locator_checkpoint']['sha256'] = locator_hash
            with patch.object(learning.legacy, 'overview', side_effect=stable_overview):
                reader = learning.legacy.Inputs(lock['parent_config'])
                for row in selected:
                    ov, image = reader.coarse(row)
                    predicted = self.locator_network.predict_boxes(locator, image)[0]
                    origins, boxes = geometry.proposal_origins(predicted, ov, lock['parent_config'])
                    lock['proposals'][row['sample_id']] = {'predicted_box': predicted.tolist(),
                        'origins_kji': origins, 'proposal_boxes': boxes.tolist(), 'coverage_passed': True}
            self.assertEqual(len(exp.select_diagnostic(c, lock)), 4)
            exp.write_json(out / 'experiment-lock.json', lock)
            MiniVolume.accessed = []
            updates = []
            real_make = self.network.make_steps

            def observe_steps(model, *args, **kwargs):
                actual = real_make(model, *args, **kwargs)
                def observe_train(values, targets):
                    np.testing.assert_array_equal(np.asarray(targets['coa_logit']).ravel(), [0., 1.])
                    for value in values.values():
                        self.assertEqual(int(value.shape[0]), 2)
                    before = int(np.asarray(actual.optimizer.iterations))
                    result = actual.train(values, targets)
                    after = int(np.asarray(actual.optimizer.iterations))
                    self.assertEqual(after, before + 1)
                    updates.append({'labels': np.asarray(targets['coa_logit']).ravel().tolist(),
                                    'before': before, 'after': after})
                    return result
                observe_train.experimental_get_tracing_count = actual.train.experimental_get_tracing_count
                return StableSteps(observe_train, actual.predict, actual.optimizer)

            with ExitStack() as stack:
                stack.enter_context(patch.object(exp, 'inspect_lock', return_value=lock))
                stack.enter_context(patch.object(exp, 'require_smoke', return_value={'status': 'passed', 'batch_size': 2}))
                stack.enter_context(patch.object(learning.legacy, 'overview', side_effect=stable_overview))
                stack.enter_context(patch.object(learning, 'ContextVolume', MiniVolume))
                stack.enter_context(patch.object(learning.legacy, 'configure_tensorflow',
                                                return_value=(self.tf, self.locator_network)))
                stack.enter_context(patch.object(self.network, 'make_steps', side_effect=observe_steps))
                stack.enter_context(patch.object(learning, '_render'))
                forbidden = stack.enter_context(patch.object(self.locator_network, 'build_locator',
                    side_effect=AssertionError('Paired classifier must reuse the frozen locator')))
                result = learning.fit(c, lock, diagnostic=True)
                forbidden.assert_not_called()
            self.assertEqual(len(updates), 4)
            self.assertTrue(all(update['labels'] == [0., 1.] for update in updates))
            self.assertEqual(result['optimizer_steps'], 4)
            self.assertEqual(result['epochs_completed'], 2)
            self.assertEqual(set(MiniVolume.accessed), set(lock['diagnostic_sample_ids']))
            self.assertTrue(result['locator_unchanged'])
            self.assertFalse(result['test_evaluated'])
            self.assertIsNone(result['validation'])
            self.assertTrue(result['selected_checkpoint_prediction_parity']['passed'])
            self.assertEqual(result['training_traces']['classifier'], {'train': 1, 'predict': 1})
            self.assertEqual(exp.sha256(checkpoint), locator_hash)
            folder = out / 'diagnostic'
            self.assertTrue(exp.checked_json(folder / 'proposal-replay.json')['passed'])
            history = exp.checked_json(folder / 'classifier-history.json')
            self.assertEqual(result['selected_epoch'],
                             min(history['epochs'], key=lambda row: row['checkpoint_selection_bce'])['epoch'])
            for epoch in history['epochs']:
                self.assertEqual(set(epoch['training_sample_order']), set(lock['diagnostic_sample_ids']))
            self.assertEqual(len(exp.checked_json(folder / 'sibling-sensitivity-selected.json')['pairs']), 2)
            # Balanced training batches do not require a comparison sibling at
            # inference: each image's prediction is independent of its mate.
            restored = self.network.load_model(folder / 'model/classifier.keras')
            negative, positive = selected[:2]
            with patch.object(learning, 'ContextVolume', MiniVolume):
                reader = learning.FineInputs(exp.runtime_config(c, lock), 2)
                images = learning.pair_inputs(reader, negative, positive, lock['proposals'])
            paired_predictions = restored(images, training=False)
            single_predictions = restored({key: value[:1] for key, value in images.items()}, training=False)
            altered_mate = {key: value.copy() for key, value in images.items()}
            for value in altered_mate.values():
                value[1, ..., 0] += .25
            altered_predictions = restored(altered_mate, training=False)
            for key in paired_predictions:
                np.testing.assert_allclose(paired_predictions[key][:1], single_predictions[key], atol=1e-6)
                np.testing.assert_allclose(paired_predictions[key][:1], altered_predictions[key][:1], atol=1e-6)


if __name__ == '__main__':
    unittest.main()
