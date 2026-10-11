"""Classifier metrics and portable report tests without TensorFlow or cohort data."""
import json
import math
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_coa_classifier_report as report


def record(label, probability, *, score=0.2, prediction=0.25, **extra):
    row = dict(sample_id='case:variant', case_id='case', family='family', split='train',
               role='lower_positive' if label else 'edited_negative', label=label,
               probability=probability, score_label=score, score_prediction=prediction)
    row.update(extra)
    return row


class MetricTests(unittest.TestCase):
    def test_tied_auroc_and_confusion_counts(self):
        rows = [record(1, .7), record(0, .7), record(1, .1), record(0, .2)]
        result = report.summarize(rows)
        self.assertEqual(result['auroc'], .375)
        self.assertEqual(result['roc_auc'], .375)
        self.assertEqual(result['confusion_counts'], dict(TP=1, TN=1, FP=1, FN=1))
        for key in ('accuracy', 'sensitivity', 'specificity', 'balanced_accuracy', 'precision'):
            self.assertEqual(result[key], .5)
        self.assertAlmostEqual(result['score_mae'], .05)
        self.assertAlmostEqual(result['severity_mae_percentage_points'], 5)
        self.assertAlmostEqual(result['regression_rmse'], .05)
        self.assertAlmostEqual(result['positive_probability_mean'], .4)
        self.assertAlmostEqual(result['negative_probability_mean'], .45)

    def test_identical_scores_are_chance_even_when_unbalanced(self):
        rows = [record(1, .5)] + [record(0, .5) for _ in range(7)]
        result = report.summarize(rows)
        self.assertEqual(result['auroc'], .5)
        self.assertEqual(result['confusion_counts'], dict(TP=1, TN=0, FP=7, FN=0))
        self.assertEqual(result['precision'], 1 / 8)

    def test_ranking_is_order_independent(self):
        rows = [record(1, .99), record(0, .01), record(1, .55), record(0, .4)]
        self.assertEqual(report.summarize(rows)['auroc'], 1)
        self.assertEqual(report.summarize(list(reversed(rows)))['auroc'], 1)
        for row in rows:
            row['probability'] = 1 - row['probability']
        self.assertEqual(report.summarize(rows)['auroc'], 0)

    def test_empty_and_one_class_are_json_safe(self):
        self.assertIsNone(report.summarize([]))
        negatives = report.summarize([record(0, .2), record(0, .1)])
        for key in ('sensitivity', 'auroc', 'balanced_accuracy', 'precision', 'positive_probability_mean'):
            self.assertIsNone(negatives[key])
        self.assertEqual(negatives['specificity'], 1)
        positives = report.summarize([record(1, .2)])
        self.assertIsNone(positives['specificity'])
        self.assertIsNone(positives['negative_probability_mean'])
        self.assertEqual(positives['sensitivity'], 0)
        json.dumps([negatives, positives], allow_nan=False)

    def test_float32_epsilon_bce_compatibility_and_rmse_units(self):
        rows = [record(1, 0, score=.1, prediction=.4), record(0, 1, score=.2, prediction=.3)]
        result = report.summarize(rows)
        self.assertAlmostEqual(result['bce'], -math.log(2 ** -23))
        self.assertAlmostEqual(result['score_mae'], .2)
        self.assertAlmostEqual(result['severity_mae_percentage_points'], 20)
        self.assertAlmostEqual(result['regression_rmse'], math.sqrt(.05))

    def test_invalid_values_are_rejected_instead_of_silent_nan(self):
        for row in [record(2, .5), record(0, -.1), record(1, 1.1),
                    record(1, float('nan')), record(1, .3, prediction=float('inf'))]:
            with self.assertRaises(ValueError):
                report.summarize([row])


class ReportTests(unittest.TestCase):
    def write(self, directory, name, value):
        (Path(directory) / name).write_text(json.dumps(value))

    def test_plan_only_report_needs_no_images_or_models(self):
        with tempfile.TemporaryDirectory() as directory:
            lock = dict(pilot_counts={'train': {'samples': 32}, 'validation': {'samples': 8}},
                        config={'source_atn': '/does/not/exist.bin'},
                        locator_checkpoint={'path': '/does/not/exist.keras', 'sha256': 'abc', 'selected_epoch': 99})
            self.write(directory, 'experiment-lock.json', lock)
            path = Path(report.write_report(directory))
            text = path.read_text()
            self.assertIn('Planned; classifier results not available', text)
            self.assertIn('32 training and 8 validation scenarios', text)
            self.assertIn('20% achieved local target volume loss', text)
            self.assertIn('not held-out test results', text)
            self.assertIn('Frozen locator epoch: 99', text)
            self.assertNotIn('data:image/png', text)
            self.assertEqual(json.loads((Path(directory) / 'experiment-lock.json').read_text()), lock)

    def test_completed_report_plots_roles_threshold_errors_and_escapes_text(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [record(0, .6, score=0, prediction=.1, role='original', sample_id='case:<script>'),
                    record(0, .3, score=.19, prediction=.18, sample_id='case:edited-negative'),
                    record(1, .4, score=.21, prediction=.19, split='validation', sample_id='case:positive')]
            training = report.summarize(rows[:2])
            validation = report.summarize(rows[2:])
            self.write(directory, 'results.json', dict(status='completed', selected_epoch=1, predictions=rows,
                       training=training, validation=validation, test_evaluated=False))
            self.write(directory, 'classifier-history.json', dict(initial_training=training, initial_validation=validation,
                       epochs=[dict(epoch=1, train_loss=.5, training=training, validation=validation, predictions=rows)]))
            output = Path(report.write_report(directory))
            text = output.read_text()
            self.assertIn('Selected classifier epoch: <strong>1', text)
            self.assertIn('Training, originals', text)
            self.assertIn('Training, edited negatives', text)
            self.assertIn('False positive', text)
            self.assertIn('False negative', text)
            self.assertIn('class="near"', text)
            self.assertIn('case:&lt;script&gt;', text)
            self.assertNotIn('case:<script>', text)
            self.assertEqual(text.count('src="data:image/png;base64,'), 3)
            self.assertNotIn('src="figures/', text)
            figures = list((Path(directory) / 'figures').glob('*.png'))
            self.assertEqual(len(figures), 3)
            for figure in figures:
                self.assertEqual(figure.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
            self.assertEqual(json.loads((Path(directory) / 'results.json').read_text())['predictions'], rows)

    def test_prediction_file_fallback_and_live_history_are_marked(self):
        with tempfile.TemporaryDirectory() as directory:
            rows = [record(0, .1, role='original', score=0)]
            self.write(directory, 'classifier-predictions.json', dict(selected_epoch=2, predictions=rows))
            text = Path(report.write_report(directory)).read_text()
            self.assertIn('Saved prediction checkpoint epoch (run not finalized): <strong>2', text)
            self.assertNotIn('Selected classifier epoch', text)
            self.assertIn('No misclassifications', text)
            (Path(directory) / 'classifier-predictions.json').unlink()
            self.write(directory, 'classifier-history.json', dict(epochs=[dict(epoch=3, training=report.summarize(rows),
                       validation=None, train_predictions=rows, validation_predictions=[])]))
            text = Path(report.write_report(directory)).read_text()
            self.assertIn('latest completed epoch, not a finalized selected checkpoint', text)
            self.assertIn('Training in progress', text)
            self.assertNotIn('Performance at the selected checkpoint', text)


if __name__ == '__main__':
    unittest.main()
