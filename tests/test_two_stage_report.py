"""A portable review reads saved measurements without changing experiment data."""
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
import fakect_two_stage_report as report


def _write(path, value):
    path.write_text(json.dumps(value))


def _hash(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


class TwoStageReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.root = Path(temporary.name)
        for name in ('model1', 'model2', 'test-model1', 'test-model2'):
            (self.root/name).mkdir()
        metric = dict(dice=.7123, precision=.8, recall=.9, false_positive_mm3=2., false_negative_mm3=3.)
        first = dict(stage=1, threshold=.5, records=[], target_spacing_ijk_mm=[.5, .75, 1.], summary={}, controls_summary={})
        second = deepcopy(first); second['stage'] = 2
        for stage, channels in ((first, ['aorta']), (second, ['oracle_reference', 'oracle_added', 'oracle_removed',
                'identity_reference', 'identity_added', 'identity_removed', 'chained_reference', 'chained_added',
                'chained_removed', 'chained_residual_added', 'chained_residual_removed'])):
            stage['summary'] = {key: {'family_macro': deepcopy(metric)} for key in channels}
            stage['controls_summary'] = deepcopy(stage['summary'])
        rows = []
        for index, control in enumerate((True, False)):
            before = np.zeros((4, 8, 8), np.uint8); before[:, 2:6, 2:6] = 1
            after = before.copy()
            if not control:
                after[2, 2, 2] = 0; after[2, 4, 6] = 1
            added, removed = after.astype(bool) & ~before.astype(bool), before.astype(bool) & ~after.astype(bool)
            sample_id = f'case:sample-{index}'
            row = dict(sample_id=sample_id, case_id='case', anatomy_family='test-family', is_control=control,
                       native_added_voxels=0 if control else 4, native_removed_voxels=0 if control else 8,
                       added_truth_voxels=int(added.sum()), removed_truth_voxels=int(removed.sum()),
                       segmentation_error_voxels=0, chained_added_on_segmentation_error_voxels=0,
                       chained_removed_on_segmentation_error_voxels=0, prediction_file=f'{index:04d}-prediction.npz')
            first['records'].append(deepcopy(row)); second['records'].append(deepcopy(row))
            rows.append(dict(sample_id=sample_id, geometry={'spacing_ijk_mm': [1., 1., 1.]}))
            base = dict(before=before, after=after, added=added, removed=removed,
                        sample_id=np.asarray(sample_id), spacing_ijk_mm=np.array([.5, .75, 1.]))
            np.savez_compressed(self.root/'test-model1'/row['prediction_file'], **base, aorta_probability=after.astype(np.float32))
            np.savez_compressed(self.root/'test-model2'/row['prediction_file'], **base,
                oracle_reference_probability=before.astype(np.float32), chained_reference_probability=before.astype(np.float32),
                stage1_aorta_probability=after.astype(np.float32), oracle_added_mask=added, oracle_removed_mask=removed,
                chained_added_mask=added, chained_removed_mask=removed)
        self.results = dict(schema_version='fakect.two-stage-results/1', dataset_manifest_sha256='a'*64,
                            split_counts={'train': 128, 'validation': 32, 'test': 32}, stage1=first, stage2=second,
                            design={'families': {'train-family': ['train-case'], 'validation-family': ['validation-case'], 'test-family': ['case']},
                                    'folds': [{'family_assignment': {'train-family': 'train', 'validation-family': 'validation', 'test-family': 'test'}}]},
                            warnings=['Provisional anatomy groups; selected crop only.'])
        _write(self.root/'results.json', self.results)
        _write(self.root/'experiment-lock.json', {'samples': rows})
        for number in (1, 2):
            (self.root/f'model{number}'/'model.keras').write_bytes(b'unchanged weights')
            _write(self.root/f'model{number}'/'history.json', [dict(epoch=1, train_loss=1., validation_loss=.8, validation_dice=.6),
                                                            dict(epoch=2, train_loss=.8, validation_loss=.7, validation_dice=.7)])
            _write(self.root/f'model{number}'/'model-metadata.json', {'settings': {'architecture': 'unet3d', 'threshold': .5},
                   'best_epoch': 2, 'epochs_completed': 2, 'tensorflow_version': '2.17.0', 'model_sha256': 'b'*64})

    def render_fast(self):
        with patch.object(report, '_sample_figure', return_value='<figure>fixture preview</figure>'), \
                patch.object(report, '_history', return_value='<p>fixture history</p>'):
            return report.render_report(self.root)

    def test_portable_figures_preserve_saved_weights_and_measurements(self):
        files = [p for p in self.root.rglob('*') if p.is_file()]
        original = {p: _hash(p) for p in files}
        result = report.render_report(self.root)
        document = Path(result['report']).read_text()
        self.assertEqual(result['embedded_prediction_figures'], 6)
        self.assertEqual(document.count('src="data:image/png;base64,'), 8)
        self.assertNotIn('src="http', document)
        self.assertIn('Model grid k=2', document)  # Embedded figure alt caption remains separate; see helper below.
        self.assertIn('0.712', document)
        self.assertIn('128', document)
        self.assertIn('Model 2 · exact observed-mask input', document)
        self.assertIn('End-to-end · Model 1 prediction into Model 2', document)
        self.assertEqual(result['interactive_3d']['scenarios'], 2)
        self.assertIn('report-3d/index.html', document)
        self.assertIn('entire saved', (self.root/'report-3d'/'index.html').read_text())
        self.assertEqual(original, {p: _hash(p) for p in files})

    def test_mask_only_results_skip_model1_and_chained_sections(self):
        self.results['schema_version'] = 'fakect.mask-model-results/1'
        self.results.pop('stage1')
        _write(self.root/'results.json', self.results)
        result = self.render_fast()
        document = Path(result['report']).read_text()
        self.assertNotIn('id="model1"', document)
        self.assertNotIn('id="pipeline"', document)
        self.assertIn('Estimate the reference shape from a segmented aorta', document)
        self.assertEqual(result['interactive_3d']['scenarios'], 2)

    def test_predictions_disabled_still_reports_metrics_and_empty_controls(self):
        for stage in ('stage1', 'stage2'):
            self.results[stage]['controls_summary'] = {}
            for row in self.results[stage]['records']:
                row.pop('prediction_file')
        _write(self.root/'results.json', self.results)
        result = self.render_fast()
        document = Path(result['report']).read_text()
        self.assertEqual(result['embedded_prediction_figures'], 0)
        self.assertIn('predictions were not saved', document)
        self.assertIn('No measurements are available', document)
        self.assertIn('0.712', document)

    def test_user_text_is_escaped(self):
        self.results['warnings'] = ['<script>alert("bad")</script>']
        self.results['design']['families']['test-family'] = ['<img onerror="bad">']
        _write(self.root/'results.json', self.results)
        document = Path(self.render_fast()['report']).read_text()
        self.assertNotIn('<script>', document)
        self.assertNotIn('<img onerror=', document)
        self.assertIn('&lt;script&gt;', document)
        self.assertIn('&lt;img onerror=', document)

    def test_slice_choice_uses_true_changes_and_physical_grid(self):
        path = self.root/'test-model2'/'0001-prediction.npz'
        fields, spacing, k = report._slice_prediction(path, 'case:sample-1')
        self.assertEqual(k, 2)
        self.assertEqual(fields['before'].shape, (8, 8))
        self.assertEqual(spacing.tolist(), [.5, .75, 1.])
        with self.assertRaisesRegex(ValueError, 'another sample'):
            report._slice_prediction(path, 'wrong sample')

    def test_representatives_ignore_prediction_scores(self):
        rows = deepcopy(self.results['stage2']['records'])
        extra = dict(rows[1], sample_id='case:strongest', native_added_voxels=999,
                     metrics={'dice': .0001})
        rows[1]['metrics'] = {'dice': 1.0}; rows.append(extra)
        self.assertEqual(report._representatives(rows), ['case:sample-0', 'case:strongest'])

    def test_native_and_model_counts_and_volumes_remain_distinct(self):
        document = Path(self.render_fast()['report']).read_text()
        self.assertIn('Native added voxels', document)
        self.assertIn('Model added voxels', document)
        self.assertIn('12.000', document)
        self.assertIn('0.750', document)
        self.assertIn('Voxel counts on different grids are not directly comparable', document)

    def test_prediction_paths_cannot_escape_evaluation_folder(self):
        for name in ('../model1/model.keras', '/tmp/prediction.npz'):
            with self.subTest(name=name), self.assertRaises(ValueError):
                report._prediction_path(self.root/'test-model1', name)

    def test_nonfinite_metrics_have_no_nan_or_infinity_text(self):
        for value in (float('nan'), float('inf'), -float('inf')):
            self.assertEqual(report._number(value), '—')
        self.results['stage1']['summary']['aorta']['family_macro']['dice'] = float('nan')
        _write(self.root/'results.json', self.results)
        document = Path(self.render_fast()['report']).read_text()
        self.assertNotIn('>nan<', document)
        self.assertNotIn('>inf<', document)


if __name__ == '__main__':
    unittest.main()
