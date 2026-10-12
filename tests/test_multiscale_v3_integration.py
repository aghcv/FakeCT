"""Revision-three geometry, inference gates and tiny real-TF staged integration.

The passing end-to-end test mocks robustness approval explicitly to exercise
classifier/save/reload plumbing, not to assert one-epoch localization quality.
Separate deterministic image-only predictors exercise the real robustness gate.
No actual cohort payload or reserved test family is opened.
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
import fakect_multiscale_experiment_v3 as exp
import fakect_multiscale_learning_v3 as learning
from test_multiscale_experiment_review import MiniVolume, fixture_lock, fixture_rows
from test_multiscale_v2_integration import config as v2_config, ShiftedValidationVolume, overview


def config(directory):
    value = v2_config(directory)
    value['augmentation'] = dict(translate_fraction_kji=[.02, .02, .02],
                                 scale_range=[.98, 1.02], crop_fraction_max=0., samples_per_case=1)
    value['robustness'] = dict(translation_fraction=.04, maximum_relative_error=.5)
    return value


class V3GeometryAndPartitionTests(unittest.TestCase):
    # Purpose: Verify physical detail margins round within the voxel budget.
    # Inputs: Production v3 INI with 9.6 mm margins, then 20 mm margins.
    # Outputs: Base [216,160,160] expands to [216,192,192]; oversized margins raise ValueError.
    def test_physical_margin_rounding_and_effective_budget(self):
        source = ROOT / 'archive/research/configs/models/coa-multiscale-pilot-v3.ini'
        c = exp.load_config(source)
        self.assertEqual(c['input.detail']['base_shape_kji'], [216, 160, 160])
        self.assertEqual(c['input.detail']['shape_kji'], [216, 192, 192])
        np.testing.assert_allclose(c['input.detail']['effective_margin_mm_ijk'], [9.6, 9.6, 0.])
        with tempfile.TemporaryDirectory() as directory:
            modified = Path(directory) / 'too-large.ini'
            original = source.read_text()
            self.assertIn('margin_mm_ijk = 9.6,9.6,0', original)
            modified.write_text(original.replace('margin_mm_ijk = 9.6,9.6,0',
                                                 'margin_mm_ijk = 20,20,20'))
            with self.assertRaisesRegex(ValueError, '[Bb]udget|8,388,608|8388608'):
                exp.load_config(modified)

    # Purpose: Verify larger detail crops preserve predicted physical centers.
    # Inputs: Fractional predicted box and anisotropic overview with old/new detail shapes.
    # Outputs: Native and normalized centers stay equal while proposal extents increase.
    def test_larger_detail_retains_each_predicted_physical_center(self):
        c = exp.load_config(ROOT / 'archive/research/configs/models/coa-multiscale-pilot-v3.ini')
        old = deepcopy(c)
        old['input.detail']['shape_kji'] = c['input.detail']['base_shape_kji']
        # Deliberately anisotropic overview and off-grid prediction detect axis
        # swaps, unit mistakes and origin rounding changes.
        ov = dict(image=np.zeros((160, 96, 96), np.uint8), spacing_ijk_mm=[3.1, 2.7, 8.3])
        box = np.array([.5731, .4735, .5127, .2263, .2341, .1769])
        old_origins, old_boxes = exp.proposal_origins(box, ov, old)
        new_origins, new_boxes = exp.proposal_origins(box, ov, c)
        old_centers = np.asarray(old_origins) + np.asarray(old['input.detail']['shape_kji']) / 2
        new_centers = np.asarray(new_origins) + np.asarray(c['input.detail']['shape_kji']) / 2
        np.testing.assert_array_equal(old_centers, new_centers)
        np.testing.assert_array_equal(old_boxes[:, :3], new_boxes[:, :3])
        self.assertTrue(np.all(new_boxes[:, 4:] > old_boxes[:, 4:]))

    # Purpose: Reject test partitions before preparation, fitting, or readers.
    # Inputs: Fixture lock containing a test row with overview/volume/TensorFlow spies.
    # Outputs: Both stages raise ValueError and all readers/configuration remain uncalled.
    def test_test_split_rejected_before_readers_in_prepare_and_fit(self):
        with tempfile.TemporaryDirectory() as directory:
            c, lock = config(directory), fixture_lock(directory)
            lock['samples'] = [{**lock['samples'][0], 'split': 'test'}]
            with patch.object(learning, 'overview') as reader, \
                 patch.object(learning, 'ContextVolume') as volume, \
                 patch.object(learning, 'configure_tensorflow') as configure:
                with self.assertRaisesRegex(ValueError, '[Tt]est'):
                    learning.prepare(c, lock)
                exp.write_json(Path(directory) / 'preparation.json', dict(
                    complete=True, lock_sha256=exp.sha256(Path(directory) / 'experiment-lock.json'), samples=[]))
                with self.assertRaisesRegex(ValueError, '[Tt]est'):
                    learning.fit(c, lock)
                reader.assert_not_called()
                volume.assert_not_called()
                configure.assert_not_called()


class ImageCentroidPredictor:
    """Uses attenuation only; no anatomy box or mask is accepted by this API."""
    channel = 0

    @classmethod
    def predict_boxes(cls, locator, x):
        weights = x[0, ..., cls.channel].astype(np.float64)
        if weights.sum() == 0:
            center = np.full(3, .25)
        else:
            shape = np.asarray(weights.shape)
            coordinates = (np.indices(weights.shape) + .5) / shape[:, None, None, None]
            center = (coordinates * weights).sum(axis=(1, 2, 3)) / weights.sum()
            if cls.channel == 1:
                # Acquisition edges alone approximately encode translations.
                # A response test alone must not approve this shortcut.
                center = .5 + 2 * (center - .5)
        return np.r_[center, [.25, .25, .25]][None].astype(np.float32)


class ValidityShortcutPredictor(ImageCentroidPredictor):
    channel = 1


class V3RobustnessTests(unittest.TestCase):
    def audit(self, network):
        row = next(r for r in fixture_rows() if r['sample_id'] == 'validation:original')
        volume = MiniVolume(row)
        ov = dict(image=volume.mask_before.astype(np.float32), spacing_ijk_mm=[1., 1., 1.])
        x = np.stack((ov['image'], np.ones_like(ov['image'])), axis=-1)[None]
        target = exp.bbox_target(volume, ov)[None]
        class Inputs:
            def coarse(self, supplied):
                if supplied['sample_id'] != row['sample_id']:
                    raise AssertionError('Unexpected sample read')
                return ov, x
        predictions = {row['sample_id']: network.predict_boxes(None, x)[0]}
        with patch.object(learning, 'ContextVolume', MiniVolume):
            result = learning.robustness_audit(None, network, [row], Inputs(),
                {row['sample_id']: target}, {row['sample_id']: np.array([16., 16., 16.])},
                predictions, config('/tmp/unused-fakect-v3-test'))
        return result, target[0]

    # Purpose: Verify translated image predictions recover original native coverage.
    # Inputs: Image-centroid predictor audited on six fixed signed axis translations.
    # Outputs: All response/dependence/coverage gates pass and recovered native boxes equal the target.
    def test_fixed_image_translations_inverse_map_to_native_coverage(self):
        result, target = self.audit(ImageCentroidPredictor)
        self.assertTrue(result['passed'])
        self.assertTrue(result['response_gate_passed'])
        self.assertTrue(result['image_dependence_gate_passed'])
        self.assertTrue(result['native_coverage_gate_passed'])
        self.assertEqual(len(result['probes']), 6)
        self.assertFalse(result['test_evaluated'])
        self.assertEqual({r['probe'] for r in result['probes']}, {'k-', 'k+', 'j-', 'j+', 'i-', 'i+'})
        for probe in result['probes']:
            self.assertTrue(probe['eligible'])
            np.testing.assert_allclose(probe['recovered_native_box'], target, atol=1e-6)
            self.assertGreater(np.linalg.norm(np.asarray(probe['predicted_box'])[:3] - target[:3]), .039)
            self.assertLess(probe['relative_response_error'], 1e-5)
            self.assertEqual(probe['native_coverage']['before_coverage'], 1.)
            self.assertEqual(probe['native_coverage']['target_coverage'], 1.)
            self.assertTrue(probe['coverage_passed'])

    # Purpose: Reject native coverage failure despite valid image response.
    # Inputs: Image-centroid predictor with coverage mocked to 0.8.
    # Outputs: Response/dependence pass but native coverage and combined approval fail.
    def test_native_coverage_failure_rejects_otherwise_valid_image_response(self):
        with patch.object(exp, 'coverage', return_value=dict(
                target_coverage=.8, before_coverage=.8, changed_coverage=None,
                added_coverage=None, removed_coverage=None, changed_voxels=0)):
            result, _ = self.audit(ImageCentroidPredictor)
        self.assertTrue(result['response_gate_passed'])
        self.assertTrue(result['image_dependence_gate_passed'])
        self.assertFalse(result['native_coverage_gate_passed'])
        self.assertFalse(result['passed'])

    # Purpose: Reject validity-only shortcuts through blank-image dependence.
    # Inputs: ValidityShortcutPredictor on real and blank translated inputs.
    # Outputs: Translation response passes but equal real/blank errors fail image dependence.
    def test_validity_shortcut_fails_blank_dependence_despite_translation_response(self):
        result, _ = self.audit(ValidityShortcutPredictor)
        self.assertTrue(result['response_gate_passed'])
        self.assertFalse(result['image_dependence_gate_passed'])
        self.assertFalse(result['passed'])
        self.assertAlmostEqual(result['validation_real_mean_error_mm'],
                               result['validation_blank_mean_error_mm'])

    # Purpose: Keep ineligible robustness probes at their predetermined displacement.
    # Inputs: 16-cube input and target near the lower K boundary.
    # Outputs: First probe remains K- at -0.04 and reports ineligible supervision.
    def test_fixed_probe_is_not_moved_to_rescue_ineligible_supervision(self):
        x = np.ones((1, 16, 16, 16, 2), np.float32)
        target = np.array([[.24, .5, .5, .2, .25, .25]], np.float32)
        probes = list(learning.fixed_probes(x, target, config('/tmp/unused-fakect-v3-test')))
        first = probes[0]
        self.assertEqual(first[0], 'k-')
        np.testing.assert_array_equal(first[2], [-.04, 0., 0.])
        self.assertIsNone(first[3])
        self.assertIn('ineligible', first[-1])


@unittest.skipUnless(importlib.util.find_spec('tensorflow'), 'TensorFlow unavailable')
class V3RealTensorFlowIntegrationTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        import fakect_multiscale_network_v3 as network
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
        stack.enter_context(patch.object(exp, 'inspect_lock', return_value=lock))
        stack.enter_context(patch.object(learning, 'configure_tensorflow', return_value=(self.tf, self.network)))
        return stack

    # Purpose: Verify v3 stage plumbing with explicit mocked robustness approval.
    # Inputs: Small real TensorFlow fit, shifted validation fixture, and an approving audit mock.
    # Outputs: Single traces, training-only priors, both checkpoints, reload parity, and no reserved reads are verified.
    def test_real_prepare_fit_reload_with_explicit_mocked_robustness_approval(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            c, lock = config(directory), fixture_lock(directory)
            with self.patches(lock), patch.object(learning, 'robustness_audit', return_value={
                    'passed': True, 'test_evaluated': False,
                    'test_fixture_note': 'Mocked only to exercise classifier stage plumbing'}) as robust:
                prepared = learning.prepare(c, lock)
                result = learning.fit(c, lock)
                robust.assert_called_once()
                self.assertTrue(result['classifier_trained'])
                self.assertFalse(result['test_evaluated'])
                self.assertEqual(result['training_traces']['locator'], {'train': 1, 'predict': 1})
                self.assertEqual(result['training_traces']['classifier'], {'train': 1, 'predict': 1})
                self.assertEqual(set(result['checkpoint_sha256']), {'locator.keras', 'classifier.keras'})
                diagnostics = exp.checked_json(out / 'locator-diagnostics.json')
                targets = {r['sample_id']: r['target_box_kji'] for r in prepared['samples']}
                np.testing.assert_allclose(diagnostics['prior_box'], targets['train:original'])
                self.assertEqual(diagnostics['prior_source_sample_ids'], ['train:original'])
                self.assertFalse(np.allclose(targets['train:original'], targets['validation:original']))
                augmentations = exp.checked_json(out / 'augmentation-audit.json')
                self.assertEqual({r['sample_id'] for r in augmentations['examples']}, {'train:original'})
                locator = self.network.load_model(out / 'model/locator.keras')
                _, x = learning.Inputs(c).coarse(lock['samples'][0])
                self.assertEqual(self.network.predict_boxes(locator, x).shape, (1, 6))
                loaded = self.network.load_model(out / 'model/classifier.keras')
                audit = exp.checked_json(out / 'proposal-audit.json')
                self.assertTrue(audit['passed'])
                proposals = {r['sample_id']: r for r in audit['samples']}
                scores = learning._score_classifier(loaded, lock['samples'], learning.Inputs(c), proposals)
                np.testing.assert_allclose([r['probability'] for r in scores],
                                           [r['probability'] for r in result['predictions']], atol=1e-6)
            self.assertNotIn('test:never_open', MiniVolume.accessed)
            self.assertTrue((out / 'report.html').is_file())

    # Purpose: Stop classifier construction when robustness fails despite coverage.
    # Inputs: Small real locator fit with failed robustness audit and classifier spy.
    # Outputs: Coverage passes; fit stops at robustness gate with no classifier construction or artifact.
    def test_failed_robustness_blocks_classifier_after_successful_coverage(self):
        with tempfile.TemporaryDirectory() as directory:
            out = Path(directory)
            c, lock = config(directory), fixture_lock(directory)
            with self.patches(lock), patch.object(learning, 'robustness_audit', return_value={
                    'passed': False, 'response_gate_passed': False,
                    'image_dependence_gate_passed': False, 'test_evaluated': False}), \
                 patch.object(self.network, 'build_classifier') as classifier:
                learning.prepare(c, lock)
                result = learning.fit(c, lock)
                classifier.assert_not_called()
            self.assertTrue(exp.checked_json(out / 'proposal-audit.json')['passed'])
            self.assertEqual(result['status'], 'stopped_at_locator_robustness_gate')
            self.assertFalse(result['classifier_trained'])
            self.assertFalse((out / 'model/classifier.keras').exists())


if __name__ == '__main__':
    unittest.main()
