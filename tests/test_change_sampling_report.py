"""Fixed TRAIN learning scores remain distinct from whole-focus validation."""
import hashlib
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

import fakect_change_report as reference
import fakect_change_sampling_report as report


def record(name, cls=1, *, perfect=False, control=False):
    labels = np.zeros((3, 3, 3), np.uint8)
    if not control:
        labels[1, 1, 1] = cls
    predicted = labels.copy() if perfect else np.zeros_like(labels)
    return dict(sample_id=name, base_sample_id=name.split(':diagnostic')[0], case_id='case-' + str(cls),
        anatomy_family='family-' + str(cls), baseline=control, mode='uniform' if control else ('added' if cls == 1 else 'removed'),
        burden_bin='control' if control else 'medium', proposal_index=0,
        metrics=reference.change_metrics(labels, predicted, np.ones_like(labels), [.6, .6, .6]))


class SamplingSummaryTests(unittest.TestCase):
    def document(self):
        initial = [record('a:diagnostic-added-medium', 1), record('b:diagnostic-removed-medium', 2), record('c:control', control=True)]
        final = [record('a:diagnostic-added-medium', 1, perfect=True), record('b:diagnostic-removed-medium', 2, perfect=True), record('c:control', control=True)]
        return dict(initial_patch_records=initial, final_patch_records=final,
            whole_focus_training_records=[record('whole-a', 1), record('whole-b', 2)],
            gate=dict(passed=True), classifier_started=False, held_out_evaluated=False)

    # Purpose: Verify fixed-patch learning metrics remain separate from complete-focus scores.
    # Inputs: Missed initial targets, perfect final targets, and unchanged baseline controls.
    # Outputs: Patch Dice rises 0 to 1, whole-focus Dice stays 0, and supervised TRAIN scope recorded.
    def test_initial_and_final_same_targets_control_does_not_inflate_overlap(self):
        summary = report.summarize(self.document())
        for cls in ('added', 'removed'):
            self.assertEqual(summary['initial_fixed_train_patches']['classes'][cls]['case_macro']['dice'], 0.)
            self.assertEqual(summary['final_fixed_train_patches']['classes'][cls]['case_macro']['dice'], 1.)
            self.assertEqual(summary['complete_focus_train']['classes'][cls]['case_macro']['dice'], 0.)
        self.assertFalse(summary['held_out_evaluated'])
        self.assertIn('supervised', summary['scope'])

    # Purpose: Verify initial/final diagnostic comparisons require stable supervision.
    # Inputs: Changed case, mode, tile geometry, or final target count.
    # Outputs: ValueError for identity, geometry, and target-count drift.
    def test_changed_patch_identity_and_targets_rejected(self):
        for field, value in [('case_id', 'new-case'), ('mode', 'removed'), ('tile', {'offset_kji': [1, 0, 0]})]:
            document = self.document()
            document['final_patch_records'][0][field] = value
            with self.assertRaisesRegex(ValueError, 'identity|geometry'):
                report.summarize(document)
        document = self.document()
        document['final_patch_records'][0]['metrics']['classes']['added']['target_voxels'] = 2
        with self.assertRaisesRegex(ValueError, 'target counts'):
            report.summarize(document)

    # Purpose: Verify fixed diagnostics require unique matched measurements.
    # Inputs: Duplicated final row and removed initial row.
    # Outputs: ValueError for nonunique IDs and unmatched fixed-patch panels.
    def test_duplicate_and_missing_patch_measurements_rejected(self):
        document = self.document()
        document['final_patch_records'].append(document['final_patch_records'][0])
        with self.assertRaisesRegex(ValueError, 'unique'):
            report.summarize(document)
        document = self.document()
        document['initial_patch_records'].pop()
        with self.assertRaisesRegex(ValueError, 'same fixed'):
            report.summarize(document)


class SamplingReportTests(unittest.TestCase):
    def setUp(self):
        temporary = tempfile.TemporaryDirectory()
        self.addCleanup(temporary.cleanup)
        self.directory = Path(temporary.name)
        self.config = {'experiment': {'output_directory': str(self.directory)}}
        self.document = SamplingSummaryTests().document()
        self.document['initial_patch_records'][0]['sample_id'] = '<a>:diagnostic-added-medium'
        self.document['final_patch_records'][0]['sample_id'] = '<a>:diagnostic-added-medium'
        self._write()
        (self.directory / 'history.json').write_text(json.dumps({'epochs': [dict(epoch=1, loss=1., ce=.5, dice_loss=.5, updates=18)]}))

    def _write(self):
        (self.directory / 'results.json').write_text(json.dumps(self.document))

    def _archives(self):
        for section in ('initial_patch_records', 'final_patch_records'):
            for i, row in enumerate(self.document[section]):
                labels = np.zeros((3, 3, 3), np.uint8)
                if not row['baseline']:
                    labels[1, 1, 1] = 1 if row['mode'] == 'added' else 2
                before, after = labels == 2, labels == 1
                predicted = labels.copy() if section.startswith('final') else np.zeros_like(labels)
                path = self.directory / (section + '-' + str(i) + '.npz')
                np.savez_compressed(path, labels=labels, predicted=predicted, valid=np.ones_like(labels),
                    before=before.astype(np.uint8), after=after.astype(np.uint8), spacing_ijk_mm=[.6] * 3,
                    origin_kji=[1, 2, 3], sample_id=np.asarray(row['sample_id']))
                row.update(prediction_npz=path.name, prediction_sha256=hashlib.sha256(path.read_bytes()).hexdigest())
        self._write()

    # Purpose: Verify diagnostic reports escape labels and preserve source artifacts.
    # Inputs: Saved TRAIN results/history with sample ID containing angle brackets.
    # Outputs: Escaped HTML, unchanged inputs, PNG/PDF/SVG figure, and verified output hashes.
    def test_saved_report_preserves_inputs_escapes_text_and_exports_figure(self):
        original = (self.directory / 'results.json').read_bytes()
        history = (self.directory / 'history.json').read_bytes()
        output = report.write_report(self.config, {})
        html = Path(output['report']).read_text()
        self.assertIn('&lt;a&gt;', html)
        self.assertIn('Complete-focus TRAIN', html)
        self.assertIn('not held-out validation', html)
        self.assertIn('original-aorta volume', html)
        self.assertEqual(original, (self.directory / 'results.json').read_bytes())
        self.assertEqual(history, (self.directory / 'history.json').read_bytes())
        for suffix in ('png', 'pdf', 'svg'):
            self.assertTrue((self.directory / ('change-sampling-diagnostic.' + suffix)).is_file())
        provenance = json.loads((self.directory / 'change-sampling-report-provenance.json').read_text())
        for relative, expected in provenance['outputs_sha256'].items():
            self.assertEqual(expected, hashlib.sha256((self.directory / relative).read_bytes()).hexdigest())

    # Purpose: Verify missing results produce an explicit pending learning report.
    # Inputs: Deleted results JSON and TRAIN case roster a/b.
    # Outputs: Pending status, absent gate, classifier false, and visible unevaluated-gate explanation.
    def test_pending_report_never_implies_passing_or_starts_classifier(self):
        (self.directory / 'results.json').unlink()
        result = report.write_report(self.config, {'stages': [{'train_case_ids': ['a', 'b']}]})
        self.assertEqual(result['status'], 'pending')
        self.assertIsNone(result['gate'])
        self.assertFalse(result['classifier_started'])
        text = (self.directory / 'report.html').read_text()
        self.assertIn('Results pending', text)
        self.assertIn('No learning gate has been evaluated', text)
        self.assertIn('a, b', text)

    # Purpose: Verify saved patch predictions populate the interactive TRAIN viewer.
    # Inputs: Initial/final prediction archives for added, removed, and control patches.
    # Outputs: Three scenarios, standard opacity presets, and viewer link in the report.
    def test_true_and_final_predicted_changes_use_existing_interactive_viewer(self):
        self._archives()
        result = report.write_report(self.config, {})
        self.assertEqual(result['interactive_3d']['scenarios'], 3)
        manifest = json.loads((self.directory / 'report-3d' / 'manifest.json').read_text())
        self.assertEqual(manifest['opacity_presets'], [.15, .45, .8])
        self.assertIn('TRAIN patch viewer', (self.directory / 'report.html').read_text())

    # Purpose: Verify saved supervision is compared spatially rather than by counts alone.
    # Inputs: Final added target shifted one voxel with its checksum updated.
    # Outputs: ValueError reporting changed supervision.
    def test_exact_saved_truth_change_rejected_even_if_class_counts_match(self):
        self._archives()
        row = self.document['final_patch_records'][0]
        path = self.directory / row['prediction_npz']
        with np.load(path, allow_pickle=False) as data:
            arrays = {key: data[key] for key in data.files}
        arrays['labels'][1, 1, 1] = 0
        arrays['labels'][1, 1, 2] = 1
        np.savez_compressed(path, **arrays)
        row['prediction_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
        self._write()
        with self.assertRaisesRegex(ValueError, 'supervision changed'):
            report.write_report(self.config, {})

    # Purpose: Verify report prediction archives stay inside the run and match digests.
    # Inputs: Parent/absolute prediction paths and an incorrect archive SHA256.
    # Outputs: ValueError for path escape and checksum mismatch.
    def test_prediction_cannot_escape_run_or_change_checksum(self):
        for value in ('../outside.npz', '/tmp/outside.npz'):
            self.document['final_patch_records'][0]['prediction_npz'] = value
            self._write()
            with self.assertRaisesRegex(ValueError, 'inside the run'):
                report.write_report(self.config, {})
        self._archives()
        self.document['final_patch_records'][0]['prediction_sha256'] = '0' * 64
        self._write()
        with self.assertRaisesRegex(ValueError, 'checksum'):
            report.write_report(self.config, {})

    # Purpose: Verify reports require the pinned TRAIN-only experiment scope.
    # Inputs: Changed lock digest and held_out_evaluated/classifier_started flags set true.
    # Outputs: ValueError for lock drift and forbidden TRAIN-only scope changes.
    def test_source_lock_and_train_only_scope_are_enforced(self):
        (self.directory / 'experiment-lock.json').write_text('{}')
        self.document['source_lock_sha256'] = '0' * 64
        self._write()
        with self.assertRaisesRegex(ValueError, 'lock changed'):
            report.write_report(self.config, {})
        self.document.pop('source_lock_sha256')
        for key in ('held_out_evaluated', 'classifier_started'):
            self.document[key] = True
            self._write()
            with self.assertRaisesRegex(ValueError, 'TRAIN-only'):
                report.write_report(self.config, {})
            self.document[key] = False


if __name__ == '__main__':
    unittest.main()
