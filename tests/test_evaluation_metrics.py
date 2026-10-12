"""Held-out payload isolation, provenance, and family-balanced evaluation."""
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
sys.path.insert(0, str(ROOT / 'src'))
from fakect_evaluation_metrics import evaluate_segmentation_model, summarize_evaluations
from fakect_segmentation import inspect_manifest, model_settings


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def array_digest(array):
    return hashlib.sha256(array.tobytes(order='C')).hexdigest()


class IdentityPredictor:
    def predict_on_batch(self, batch):
        return batch.copy()


class HeldOutEvaluationTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.root = Path(self.temp.name)
        self.path = self.root / 'manifest.json'
        self.model_dir = self.root / 'model'
        self.model_dir.mkdir()
        (self.model_dir / 'model.keras').write_bytes(b'fake weights; no TensorFlow import')
        self.shape = (2, 3, 5)
        samples = []
        for index, split in enumerate(('train', 'validation', 'test')):
            mask = np.zeros(self.shape, np.uint8)
            mask.ravel()[index:index+3] = 1
            image = mask.astype(np.float32)
            path = self.root / f'{split}.npz'
            np.savez_compressed(path, image=image, mask=mask)
            samples.append({'variant_id': split, 'split': split, 'path': path.name,
                            'sha256': digest(path), 'image_sha256': array_digest(image),
                            'mask_sha256': array_digest(mask), 'geometry_group': split})
        self.manifest = {'schema_version': 'fakect.training-dataset/1', 'split_mode': 'scenario_only',
                         'anatomy_family': 'family-1', 'case_id': 'case-1', 'image_units': 'cm^-1',
                         'geometry': {'array_order': 'kji', 'crop_shape_kji': list(self.shape)},
                         'samples': samples}
        self.metadata = {'source_manifest': str(self.path), 'settings': model_settings(
            {'model': {'patch_size': [4, 4], 'clip_min': 0, 'clip_max': 1}}),
                         'model_file': 'model.keras', 'model_sha256': digest(self.model_dir / 'model.keras')}
        self.refresh()

    def refresh(self):
        self.path.write_text(json.dumps(self.manifest))
        self.metadata['source_manifest_sha256'] = digest(self.path)
        (self.model_dir / 'model-metadata.json').write_text(json.dumps(self.metadata))

    def run_evaluation(self, **kwargs):
        with patch('fakect_evaluation_metrics.load_segmentation', return_value=(IdentityPredictor(), self.metadata)):
            return evaluate_segmentation_model(self.model_dir, self.path, **kwargs)

    # Purpose: Verify evaluation loads only held-out payloads and excludes padding.
    # Inputs: Identity predictor, 2x3x5 test data, and deleted train/validation payloads.
    # Outputs: Only test.npz loads; metrics count 3 positives and 27 negatives with Dice 1 and provenance hashes.
    def test_only_test_payload_loaded_and_padding_excluded(self):
        for split in ('train', 'validation'):
            (self.root / f'{split}.npz').unlink()
        with patch('fakect_evaluation_metrics.np.load', wraps=np.load) as load:
            result = self.run_evaluation()
        self.assertEqual(load.call_count, 1)
        self.assertEqual(load.call_args.args[0], self.root / 'test.npz')
        self.assertEqual(result['test_samples'], 1)
        row = result['samples'][0]
        self.assertEqual((row['case_id'], row['anatomy_family']), ('case-1', 'family-1'))
        self.assertEqual((row['true_positive'], row['true_negative']), (3, 27))
        self.assertEqual(row['valid_voxels'], 30)
        self.assertEqual(row['dice'], 1)
        self.assertEqual(result['source_manifest_sha256'], digest(self.path))
        self.assertEqual(result['model_metadata_sha256'], digest(self.model_dir / 'model-metadata.json'))

    # Purpose: Require the model to match the exact source manifest.
    # Inputs: Model metadata with wrong manifest digest or path.
    # Outputs: Evaluation raises ValueError for source manifest mismatch.
    def test_model_must_match_exact_source_manifest(self):
        for name, value in (('source_manifest_sha256', '0' * 64), ('source_manifest', str(self.root / 'other.json'))):
            with self.subTest(name=name):
                metadata = dict(self.metadata, **{name: value})
                with patch('fakect_evaluation_metrics.load_segmentation', return_value=(IdentityPredictor(), metadata)):
                    with self.assertRaisesRegex(ValueError, 'source manifest'):
                        evaluate_segmentation_model(self.model_dir, self.path)

    # Purpose: Verify payload-file and decoded-array hashes independently.
    # Inputs: Test image modified, then manifest file digest refreshed.
    # Outputs: Evaluation first rejects file hash, then rejects image_sha256 mismatch.
    def test_file_hash_and_array_hash_are_independently_verified(self):
        path = self.root / 'test.npz'
        with np.load(path) as archive:
            image, mask = archive['image'], archive['mask']
        image[0, 0, 0] = .3
        np.savez_compressed(path, image=image, mask=mask)
        with self.assertRaisesRegex(ValueError, 'file hash changed'):
            self.run_evaluation()
        self.manifest['samples'][-1]['sha256'] = digest(path)
        self.refresh()
        with self.assertRaisesRegex(ValueError, 'image_sha256 disagrees'):
            self.run_evaluation()

    # Purpose: Reject invalid evaluation array dtype, values, and shape.
    # Inputs: Float64/NaN/truncated images or float/nonbinary masks with refreshed file hashes.
    # Outputs: Evaluation raises ValueError naming required float32 or uint8 data.
    def test_invalid_dtype_binary_mask_and_shape_are_rejected(self):
        path = self.root / 'test.npz'
        original = path.read_bytes()
        with np.load(path) as archive:
            image, mask = archive['image'], archive['mask']
        fixtures = [(image.astype(np.float64), mask, 'float32'),
                    (np.full_like(image, np.nan), mask, 'float32'),
                    (image[:, :, :-1], mask, 'float32'),
                    (image, mask.astype(np.float32), 'uint8'),
                    (image, np.full_like(mask, 2), 'uint8')]
        for invalid_image, invalid_mask, message in fixtures:
            with self.subTest(message=message):
                np.savez_compressed(path, image=invalid_image, mask=invalid_mask)
                self.manifest['samples'][-1]['sha256'] = digest(path)
                self.refresh()
                with self.assertRaisesRegex(ValueError, message):
                    self.run_evaluation()
        path.write_bytes(original)

    # Purpose: Verify evaluation applies frozen model-grid resampling.
    # Inputs: Native 2x3x5 data mapped to 4x6x10, then a mismatched frozen shape.
    # Outputs: One resample produces 240 valid voxels; inconsistent frozen geometry raises ValueError.
    def test_model_time_resampling_uses_frozen_geometry(self):
        inspected = inspect_manifest(self.path)
        sample = inspected['samples'][-1]
        sample.update(source_shape_kji=list(self.shape), shape_kji=[4, 6, 10],
                      source_spacing_ijk_mm=[1, 1, 1], target_spacing_ijk_mm=[.5, .5, .5])
        from fakect_model_resampling import resample_pair
        with patch('fakect_evaluation_metrics.inspect_manifest', return_value=inspected), \
                patch('fakect_model_resampling.resample_pair', wraps=resample_pair) as resample:
            result = self.run_evaluation()
        self.assertEqual(resample.call_count, 1)
        self.assertEqual(result['samples'][0]['valid_voxels'], 240)
        sample['shape_kji'] = [4, 6, 9]
        with patch('fakect_evaluation_metrics.inspect_manifest', return_value=inspected):
            with self.assertRaisesRegex(ValueError, 'frozen experiment geometry'):
                self.run_evaluation()

    # Purpose: Reject invalid probabilities and artifacts changed during evaluation.
    # Inputs: Wrong-shaped, NaN, or out-of-range predictions and a predictor mutating the manifest.
    # Outputs: Evaluation rejects invalid probabilities and concurrent manifest changes.
    def test_invalid_predictions_and_changed_artifacts_fail(self):
        for values in (np.zeros((3, 4)), np.full((3, 5), np.nan), np.full((3, 5), 1.1)):
            with self.subTest(shape=values.shape), \
                    patch('fakect_evaluation_metrics.predict_axial_slice', return_value=values):
                with self.assertRaisesRegex(ValueError, 'invalid test probabilities'):
                    self.run_evaluation()
        def change_manifest(model, image, settings):
            self.path.write_text(self.path.read_text() + ' ')
            return image.copy()
        with patch('fakect_evaluation_metrics.predict_axial_slice', side_effect=change_manifest):
            with self.assertRaisesRegex(ValueError, 'changed during'):
                self.run_evaluation()

    # Purpose: Validate thresholds and source-volume limits before payload loading.
    # Inputs: Boolean/invalid/nonfinite thresholds and source shape [256,256,257].
    # Outputs: Evaluation raises ValueError and oversized volumes never reach np.load.
    def test_threshold_and_volume_limit_checked_before_loading(self):
        for value in (True, 0, 1, -1, float('nan'), float('inf'), 'invalid'):
            with self.subTest(threshold=value), self.assertRaisesRegex(ValueError, 'threshold'):
                self.run_evaluation(threshold=value)
        inspected = inspect_manifest(self.path)
        inspected['samples'][-1]['source_shape_kji'] = [256, 256, 257]
        with patch('fakect_evaluation_metrics.inspect_manifest', return_value=inspected), \
                patch('fakect_evaluation_metrics.np.load') as load:
            with self.assertRaisesRegex(ValueError, 'bounded'):
                self.run_evaluation()
        load.assert_not_called()


def result_row(variant, case, family, perfect=True):
    tp, fp, fn, tn = (2, 0, 0, 8) if perfect else (0, 2, 2, 6)
    return {'variant_id': variant, 'case_id': case, 'anatomy_family': family,
            'true_positive': tp, 'false_positive': fp, 'false_negative': fn, 'true_negative': tn,
            'valid_voxels': 10, **{name: float(perfect) for name in ('dice', 'iou', 'precision', 'recall')}}


class FamilySummaryTests(unittest.TestCase):
    def evaluations(self):
        result = []
        for seed in (11, 29):
            first = seed == 11
            result.append({'fold_id': 'A', 'seed': seed, 'threshold': .5, 'samples': [
                *[result_row(f'a1-{index}', 'A1', 'A', first) for index in range(3)],
                result_row('a2-0', 'A2', 'A', not first)]})
            result.append({'fold_id': 'B', 'seed': seed, 'threshold': .5,
                           'samples': [result_row('b1-0', 'B1', 'B', first)]})
        return result

    # Purpose: Verify summary weighting gives equal weight to cases and families.
    # Inputs: Unequal sibling counts in two families across seeds 11 and 29.
    # Outputs: Macro Dice is 0.5, seed means are 0.75/0.25, and reversed input gives the same summary.
    def test_equal_case_and_family_weights_not_sibling_or_voxel_pooling(self):
        summary = summarize_evaluations(self.evaluations())
        self.assertEqual(summary['macro_average']['dice'], .5)
        self.assertEqual([row['macro_average']['dice'] for row in summary['per_seed']], [.75, .25])
        self.assertEqual([row['dice'] for row in summary['family_reports']], [.5, .5])
        self.assertEqual(summary['family_reports'][0]['case_ids'], ['A1', 'A2'])
        self.assertEqual(summary['counts'], {'folds': 2, 'seeds': 2, 'evaluations': 4,
                                             'families': 2, 'cases': 3, 'unique_variants': 5,
                                             'sample_evaluations': 10})
        reversed_input = list(reversed(self.evaluations()))
        self.assertEqual(summarize_evaluations(reversed_input), summary)

    # Purpose: Reject duplicate folds and repeated out-of-fold predictions.
    # Inputs: Repeated fold/seed entry or a sample copied into another fold.
    # Outputs: Summary raises ValueError for either duplicate identity.
    def test_duplicate_fold_or_out_of_fold_predictions_rejected(self):
        evaluations = self.evaluations()
        with self.assertRaisesRegex(ValueError, 'Duplicate fold_id/seed'):
            summarize_evaluations(evaluations + [copy.deepcopy(evaluations[0])])
        evaluations[1]['samples'].append(copy.deepcopy(evaluations[0]['samples'][0]))
        with self.assertRaisesRegex(ValueError, 'Duplicate out-of-fold'):
            summarize_evaluations(evaluations)

    # Purpose: Require matching seeds, memberships, and case/family identities.
    # Inputs: Missing seed evaluation, missing variant, changed case, or reassigned family.
    # Outputs: Summary rejects inconsistent repeat membership and identity.
    def test_incomplete_seeds_or_changed_membership_cannot_be_averaged(self):
        with self.assertRaisesRegex(ValueError, 'same seeds'):
            summarize_evaluations(self.evaluations()[:-1])
        evaluations = self.evaluations()
        evaluations[2]['samples'].pop()
        with self.assertRaisesRegex(ValueError, 'membership'):
            summarize_evaluations(evaluations)
        evaluations = self.evaluations()
        evaluations[2]['samples'][0]['case_id'] = 'other'
        with self.assertRaisesRegex(ValueError, 'identity changes'):
            summarize_evaluations(evaluations)
        evaluations = self.evaluations()
        evaluations[0]['samples'][0]['anatomy_family'] = 'different'
        with self.assertRaisesRegex(ValueError, 'case cannot belong'):
            summarize_evaluations(evaluations)

    # Purpose: Reject malformed metrics, confusion counts, and mixed thresholds.
    # Inputs: Nonfinite/out-of-range/bool metrics, invalid counts, inconsistent Dice, or threshold 0.6.
    # Outputs: Summary raises ValueError for every inconsistent evaluation.
    def test_invalid_metrics_and_counts_cannot_be_summarized(self):
        for key, value in (('dice', float('nan')), ('iou', float('inf')), ('recall', -.1),
                           ('precision', True), ('true_positive', 1.5), ('false_positive', -1),
                           ('valid_voxels', 9), ('dice', .9)):
            with self.subTest(key=key, value=value):
                evaluations = self.evaluations()
                evaluations[0]['samples'][0][key] = value
                with self.assertRaises(ValueError):
                    summarize_evaluations(evaluations)
        evaluations = self.evaluations()
        evaluations[0]['threshold'] = .6
        with self.assertRaisesRegex(ValueError, 'different prediction thresholds'):
            summarize_evaluations(evaluations)

    # Purpose: Verify explicit finite metrics for empty targets.
    # Inputs: Empty perfect target/prediction, then one false-positive voxel.
    # Outputs: All perfect empty metrics are 1; false-positive empty recall is 0.
    def test_empty_masks_have_explicit_finite_policy(self):
        row = result_row('empty', 'case', 'family')
        row.update(true_positive=0, true_negative=10)
        summary = summarize_evaluations([{'fold_id': 'held-out', 'seed': 1, 'samples': [row]}])
        self.assertEqual(summary['macro_average'], dict.fromkeys(('dice', 'iou', 'precision', 'recall'), 1.))
        row.update(false_positive=1, true_negative=9, dice=0., iou=0., precision=0., recall=0.)
        summary = summarize_evaluations([{'fold_id': 'held-out', 'seed': 1, 'samples': [row]}])
        self.assertEqual(summary['macro_average']['recall'], 0.)


if __name__ == '__main__':
    unittest.main()
