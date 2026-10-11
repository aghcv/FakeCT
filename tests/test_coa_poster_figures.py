"""Check empirical plot data, including ties and invalid prediction records."""
import math
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fakect_coa_poster_figures import (checked_predictions, empirical_roc,
                                     source_bundle, verify_sources)


class PosterMetricTests(unittest.TestCase):
    def test_perfect_and_reversed_ranking(self):
        for probabilities, expected in [([.1, .2, .8, .9], 1.),
                                         ([.9, .8, .2, .1], 0.)]:
            with self.subTest(probabilities=probabilities):
                result = empirical_roc([0, 0, 1, 1], probabilities)
                self.assertAlmostEqual(result['auroc'], expected)
                self.assertEqual((result['positive'], result['negative']), (2, 2))
                self.assertEqual((result['fpr'][0], result['tpr'][0]), (0., 0.))
                self.assertEqual((result['fpr'][-1], result['tpr'][-1]), (1., 1.))
                self.assertTrue(np.all(np.diff(result['fpr']) >= 0))
                self.assertTrue(np.all(np.diff(result['tpr']) >= 0))

    def test_ties_enter_roc_together_and_have_half_credit(self):
        result = empirical_roc([0, 1, 0, 1], [.1, .5, .5, .9])
        self.assertAlmostEqual(result['auroc'], .875)
        np.testing.assert_allclose(result['fpr'], [0., 0., .5, 1.])
        np.testing.assert_allclose(result['tpr'], [0., .5, 1., 1.])
        tied = empirical_roc([0, 1, 0, 1], [.5, .5, .5, .5])
        self.assertEqual(tied['auroc'], .5)
        self.assertEqual(len(tied['fpr']), 2)

    def test_roc_rejects_undefined_or_invalid_inputs(self):
        invalid = [([], []), ([1, 1], [.2, .8]), ([0, 0], [.2, .8]),
                   ([0, 1], [.2]), ([0, 2], [.2, .8]),
                   ([0, 1], [.2, math.nan]), ([0, 1], [.2, math.inf]),
                   ([0, 1], [-.1, .8]), ([0, 1], [.2, 1.1])]
        for labels, scores in invalid:
            with self.subTest(labels=labels, scores=scores), self.assertRaises(ValueError):
                empirical_roc(labels, scores)


class PosterPredictionTests(unittest.TestCase):
    @staticmethod
    def records():
        return [dict(sample_id='case-01:baseline', case_id='case-01',
                     family='family-A', split='validation', role='original',
                     label=0, probability=.2, logit=math.log(.2/.8),
                     score_label=0., score_prediction=.1, near_threshold=False),
                dict(sample_id='case-01:edited', case_id='case-01',
                     family='family-A', split='validation', role='higher_positive',
                     label=1, probability=.8, logit=math.log(.8/.2),
                     score_label=.4, score_prediction=.3, near_threshold=False)]

    def test_checked_records_preserve_observed_values(self):
        rows = self.records()
        self.assertEqual(checked_predictions(rows), rows)

    def test_duplicates_reserved_predictions_and_invalid_values_fail(self):
        rows = self.records()
        with self.assertRaises(ValueError):
            checked_predictions(rows + rows[:1])
        mutations = [('split', 'test'), ('probability', math.nan),
                     ('probability', 1.1), ('label', 2),
                     ('score_prediction', math.inf), ('score_label', math.nan),
                     ('logit', 100.)]
        for key, value in mutations:
            rows = self.records()
            rows[0][key] = value
            with self.subTest(key=key, value=value), self.assertRaises(ValueError):
                checked_predictions(rows)


class PosterSourceTests(unittest.TestCase):
    @staticmethod
    def fixture(root, *, annotation_mismatch=False, reserved_overlap=False):
        """Hand-calculated four-scan metadata; no imaging or weight files."""
        def digest(path):
            return hashlib.sha256(path.read_bytes()).hexdigest()

        def write(relative, value):
            path = root / relative
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(json.dumps(value, indent=2) + '\n')
            Path(str(path)+'.sha256').write_text(digest(path)+'\n')
            return digest(path)

        model_source = root/'source/src/fakect_coa_focused_network.py'
        model_source.parent.mkdir(parents=True)
        model_source.write_text('# synthetic architecture metadata fixture\n')
        (root/'input.ini').write_text('[training]\nepochs = 2\n')
        records, samples = [], []
        for split, family in [('train', 'family-A'), ('validation', 'family-B')]:
            case = split+'-case'
            for label, role in [(0, 'original'), (1, 'higher_positive')]:
                probability = .8 if label else .2
                identity = case+':'+role
                records.append(dict(sample_id=identity, case_id=case, family=family,
                                    split=split, role=role, label=label,
                                    probability=probability,
                                    logit=math.log(probability/(1-probability)),
                                    score_label=.4 if label else 0.,
                                    score_prediction=.3 if label else .1))
                samples.append(dict(sample_id=identity, case_id=case,
                                    anatomy_family=family, split=split, pilot_role=role,
                                    annotation={'coa_benchmark_positive':bool(label),
                                                'coa_local_target_volume_loss_fraction':
                                                (.3 if annotation_mismatch else .4) if label else 0.}))
        lock = {'samples':samples,
                'reserved_test_sample_ids':[samples[-1]['sample_id']] if reserved_overlap else [],
                'annotation_acceptance':{'threshold':.2},
                'pilot_counts':{'train':{'samples':2},'validation':{'samples':2}},
                'code_sha256':{'src/fakect_coa_focused_network.py':digest(model_source)},
                'input_sha256':digest(root/'input.ini')}
        source_hash = write('experiment-lock.json', lock)
        write('pilot/experiment-lock.json', lock)
        metrics = {'samples':2, 'bce':-math.log(.8), 'accuracy':1.,
                   'sensitivity':1.,'specificity':1.,'auroc':1.,'score_mae':.1,
                   'confusion_counts':{'TP':1,'TN':1,'FP':0,'FN':0}}
        initial_metrics = dict(metrics, bce=math.log(2), accuracy=.5,
                               specificity=0., auroc=.5,
                               confusion_counts={'TP':1,'TN':0,'FP':1,'FN':0})
        initial_records = [dict(row, probability=.5, logit=0.) for row in records]
        history = {'selected_epoch':2, 'epochs':[
            {'epoch':1,'predictions':initial_records,
             'training':initial_metrics,'validation':initial_metrics},
            {'epoch':2,'predictions':records,'training':metrics,'validation':metrics}]}
        results = {'classifier_trained':True,'test_evaluated':False,
                   'source_lock_sha256':source_hash,'selected_epoch':2,'epochs_completed':2,
                   'checkpoint_sha256':{'classifier.keras':'a'*64},
                   'predictions':records, 'training':metrics,'validation':metrics,
                   'classification_decision_threshold':.5}
        artifact = {'source_lock_sha256':source_hash,'test_evaluated':False,
                    'selected_epoch':2,'checkpoint_sha256':'a'*64,'predictions':records}
        write('pilot/results.json',results)
        write('pilot/classifier-history.json',history)
        write('pilot/classifier-predictions.json',artifact)

    def test_verified_bundle_needs_no_imaging_or_checkpoint_payloads(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            self.fixture(root)
            bundle = source_bundle(root)
            self.assertEqual(len(bundle['predictions']),4)
            self.assertEqual(bundle['results']['selected_epoch'],2)
            self.assertFalse((root/'pilot/model/classifier.keras').exists())
            verify_sources(bundle)
            (root/'input.ini').write_text('changed input\n')
            with self.assertRaises(ValueError):
                verify_sources(bundle)

    def test_checksum_annotation_and_reserved_identity_failures_are_rejected(self):
        for failure in ('checksum','annotation','reserved','incomplete'):
            with self.subTest(failure=failure), tempfile.TemporaryDirectory() as directory:
                root = Path(directory)
                self.fixture(root, annotation_mismatch=failure=='annotation',
                             reserved_overlap=failure=='reserved')
                if failure=='checksum':
                    (root/'pilot/results.json.sha256').write_text('b'*64+'\n')
                if failure=='incomplete':
                    (root/'pilot/INCOMPLETE').touch()
                with self.assertRaises(ValueError):
                    source_bundle(root)


if __name__ == '__main__':
    unittest.main()
