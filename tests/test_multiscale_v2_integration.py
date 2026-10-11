"""Revision-two staged integration on tiny synthetic arrays, never real cohorts."""
from contextlib import ExitStack
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
import fakect_multiscale_experiment_v2 as exp
import fakect_multiscale_learning_v2 as learning
from test_multiscale_experiment_review import (
    MiniVolume, fixture_config as original_config, fixture_lock, fixture_rows,
)


def config(directory):
    value = original_config(directory)
    value['training'] = dict(locator_learning_rate=.0003, minimum_target_coverage=.9,
                             host_memory_limit_mb=12288, maximum_traces=1)
    return value


class ShiftedValidationVolume(MiniVolume):
    """Validation anatomy intentionally differs from every training baseline."""
    def __init__(self, row, verify_hashes=True):
        super().__init__(row, verify_hashes=verify_hashes)
        if row['split'] == 'validation':
            self.mask_before = np.roll(self.mask_before, 3, axis=0)
            self.mask_after = np.roll(self.mask_after, 3, axis=0)
            self.image = np.roll(self.image, 3, axis=0)


def overview(row, c):
    volume = ShiftedValidationVolume(row)
    return dict(image=volume.image, valid_fraction=np.ones_like(volume.image),
                valid=np.ones_like(volume.image, np.uint8), spacing_ijk_mm=[1., 1., 1.],
                composition_key=row['sample_id'], cache_key=row['case_id'], stats={})


class V2GateAndInputTests(unittest.TestCase):
    def setUp(self):
        MiniVolume.accessed = []
        ShiftedValidationVolume.returned_mask_value = None

    def tearDown(self):
        ShiftedValidationVolume.returned_mask_value = None

    def test_original_requires_before_and_after_anatomy_despite_no_edits(self):
        c = config('/tmp/unused-fakect-test')
        empty_changes = dict(target_coverage=0., before_coverage=0.,
                             changed_coverage=None, removed_coverage=None)
        self.assertFalse(exp.coverage_gate(empty_changes, c))
        self.assertFalse(exp.coverage_gate({**empty_changes, 'target_coverage': 1.}, c))
        self.assertFalse(exp.coverage_gate({**empty_changes, 'before_coverage': 1.}, c))
        self.assertTrue(exp.coverage_gate({**empty_changes, 'before_coverage': 1.,
                                          'target_coverage': 1.}, c))

    def test_resource_guard_persists_failure_before_raising(self):
        with tempfile.TemporaryDirectory() as directory:
            c = config(directory)
            guard = learning.RuntimeGuard(c, None)
            with patch.object(learning, 'memory_snapshot', return_value={
                    'rss_mib': 13000., 'host_peak_rss_mib': 13000.}):
                with self.assertRaisesRegex(RuntimeError, 'Host memory'):
                    guard.check('simulated-budget-excess')
            result = exp.checked_json(Path(directory) / 'training-runtime.json')
            self.assertEqual(result['status'], 'failed')
            self.assertEqual(result['records'][-1]['stage'], 'simulated-budget-excess')
            self.assertEqual(result['records'][-1]['rss_mib'], 13000.)

    def test_resource_guard_rejects_second_graph_trace(self):
        class RetracedSteps:
            def tracing_counts(self):
                return {'train': 2, 'predict': 1}
        with tempfile.TemporaryDirectory() as directory:
            guard = learning.RuntimeGuard(config(directory), None)
            with patch.object(learning, 'memory_snapshot', return_value={
                    'rss_mib': 128., 'host_peak_rss_mib': 256.}):
                with self.assertRaisesRegex(RuntimeError, 'retraced'):
                    guard.check('unexpected-graph', steps=RetracedSteps())
            result = exp.checked_json(Path(directory) / 'training-runtime.json')
            self.assertEqual(result['records'][-1]['traces'], {'train': 2, 'predict': 1})

    def test_supervision_masks_do_not_change_predictor_inputs(self):
        row = fixture_rows()[1]
        proposal = dict(origins_kji=[[-8, -8, -8], [-4, -8, -8]],
                        proposal_boxes=[[.5, .5, .5, 2, 2, 2], [.75, .5, .5, 2, 2, 2]])
        with patch.object(learning, 'overview', side_effect=overview), \
             patch.object(learning, 'ContextVolume', ShiftedValidationVolume):
            inputs = learning.Inputs(config('/tmp/unused-fakect-test'))
            first = inputs.classifier(row, proposal)
            ShiftedValidationVolume.returned_mask_value = 1
            second = inputs.classifier(row, proposal)
        self.assertEqual(set(first), {'overview', 'proposal_boxes', 'detail_0', 'detail_1'})
        for key in first:
            np.testing.assert_array_equal(first[key], second[key])


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class V2RealTensorFlowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_multiscale_network_v2 as network
        cls.network = network
        cls.tf = network._tensorflow()

    def setUp(self):
        self.tf.keras.backend.clear_session()
        MiniVolume.accessed = []
        ShiftedValidationVolume.returned_mask_value = None

    def patches(self, lock):
        stack = ExitStack()
        stack.enter_context(patch.object(learning, 'overview', side_effect=overview))
        stack.enter_context(patch.object(learning, 'ContextVolume', ShiftedValidationVolume))
        stack.enter_context(patch.object(learning, '_preview'))
        # Temporary fixture has no actual cohort registry. Runtime image/model,
        # preparation, proposal, resource and checkpoint paths remain exercised.
        stack.enter_context(patch.object(exp, 'inspect_lock', return_value=lock))
        stack.enter_context(patch.object(learning, 'configure_tensorflow',
                                        return_value=(self.tf, self.network)))
        return stack

    def test_prepare_stable_fit_prior_isolation_and_checkpoint_reload(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            c, lock = config(directory), fixture_lock(directory)
            with self.patches(lock):
                prepared = learning.prepare(c, lock)
                result = learning.fit(c, lock)
                self.assertTrue(result['classifier_trained'])
                self.assertFalse(result['test_evaluated'])
                self.assertEqual(result['training_traces']['locator'], {'train': 1, 'predict': 1})
                self.assertEqual(result['training_traces']['classifier'], {'train': 1, 'predict': 1})
                self.assertEqual(result['training_traces']['selected_classifier'], {'train': 0, 'predict': 1})
                self.assertEqual(set(result['checkpoint_sha256']), {'locator.keras', 'classifier.keras'})
                diagnoses = exp.checked_json(out / 'locator-diagnostics.json')
                rows = {r['sample_id']: r for r in prepared['samples']}
                expected = rows['train:original']['target_box_kji']
                np.testing.assert_allclose(diagnoses['prior_box'], expected, atol=1e-7)
                self.assertEqual(diagnoses['prior_source_sample_ids'], ['train:original'])
                self.assertFalse(np.allclose(expected, rows['validation:original']['target_box_kji']))
                self.assertFalse(np.allclose(expected, rows['train:lower_positive']['target_box_kji']))
                loaded = self.network.load_model(out / 'model/classifier.keras')
                audit = exp.checked_json(out / 'proposal-audit.json')
                self.assertTrue(audit['passed'])
                proposals = {r['sample_id']: r for r in audit['samples']}
                actual = learning._score_classifier(loaded, lock['samples'], learning.Inputs(c), proposals)
                np.testing.assert_allclose([r['probability'] for r in actual],
                                           [r['probability'] for r in result['predictions']], atol=1e-6)
                runtime = exp.checked_json(out / 'training-runtime.json')
                self.assertEqual(runtime['status'], 'within_budget')
                self.assertTrue(runtime['records'])
            self.assertNotIn('test:never_open', MiniVolume.accessed)
            self.assertTrue((out / 'report.html').is_file())

    def test_original_anatomy_miss_stops_classifier_even_with_no_changed_voxels(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            c, lock = config(directory), fixture_lock(directory)
            actual_coverage = exp.coverage
            def original_miss(volume, origins, settings):
                cov = actual_coverage(volume, origins, settings)
                if cov['changed_voxels'] == 0:
                    cov.update(target_coverage=0., before_coverage=0.)
                return cov
            with self.patches(lock):
                learning.prepare(c, lock)
                with patch.object(exp, 'coverage', side_effect=original_miss), \
                     patch.object(self.network, 'build_classifier') as build:
                    result = learning.fit(c, lock)
                build.assert_not_called()
            self.assertEqual(result['status'], 'stopped_at_locator_coverage_gate')
            self.assertEqual(set(result['failed_samples']), {'train:original', 'validation:original'})
            self.assertFalse(result['classifier_trained'])
            self.assertFalse((out / 'model/classifier.keras').exists())


if __name__ == '__main__':
    unittest.main()
