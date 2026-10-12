"""Native result review distinguishes failed edits from expected resistance."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/research/evaluation'))
import review_cohort_outcomes as review


class CohortOutcomeReviewTests(unittest.TestCase):
    def step(self, operation='erosion', **changes):
        return {'step_name': 'arch', 'iteration': 1, 'operation': operation,
            'status': 'applied', 'requested_distance_mm': 4., 'accepted_distance_mm': 4.,
            'added_voxels': 0 if operation == 'erosion' else 20,
            'removed_voxels': 20 if operation == 'erosion' else 0,
            'proposed_added_voxels': 0 if operation == 'erosion' else 100,
            'proposed_removed_voxels': 20 if operation == 'erosion' else 0,
            'blocked_voxels': 0 if operation == 'erosion' else 80,
            'unresolved_voxels': 0, **changes}

    def sample(self, **changes):
        return {'variant_id': 'variant1', 'operation': 'recipe', 'status': 'ok',
            'foreground_voxels': 100, 'baseline_equivalent': False, 'zero_change': False,
            'duplicate_of': None, 'mask_sha256': 'mask', 'image_sha256': 'image',
            'geometry': {'crop_shape_kji': [10, 10, 10], 'spacing_ijk_mm': [1, 1, 1]},
            'step_outcomes': [self.step()], **changes}

    def document(self):
        return {'schema_version': 'fakect.recipe-preflight/1', 'complete': True,
            'execution_valid': True, 'variant_count': 1, 'samples': [self.sample()],
            'source_crop_sha256': {'act': 'act', 'atn': 'atn'}}

    def reasons(self, row):
        return {issue['reason'] for issue in row['issues']}

    # Purpose: Verify resistant obstacles are ordinary dilation outcomes.
    # Inputs: Dilation with 20 accepted and 80 blocked proposed voxels.
    # Outputs: No review flags and acceptance fraction 0.2.
    def test_blocked_dilation_is_reported_without_failure(self):
        row = review.summarize_step(self.step('dilation'))
        self.assertEqual(row['review_state'], 'no_flags')
        self.assertEqual(row['dilation_acceptance_fraction'], .2)

    # Purpose: Verify erosion review distinguishes partial release from ineffective edits.
    # Inputs: 20 released of 100 proposed voxels and then zero released with unresolved recipients.
    # Outputs: Needs-review partial release at 0.2 and rejection for no active effect.
    def test_partial_reassignment_requires_review_but_noop_rejects(self):
        row = review.summarize_step(self.step(unresolved_voxels=80, proposed_removed_voxels=100))
        self.assertEqual(row['review_state'], 'needs_review')
        self.assertEqual(row['release_assignment_fraction'], .2)
        row = review.summarize_step(self.step(removed_voxels=0, unresolved_voxels=20, status='no_change'))
        self.assertEqual(row['review_state'], 'reject')
        self.assertIn('active_edit_has_no_effect', self.reasons(row))

    # Purpose: Verify erosion guard outcomes affect review severity.
    # Inputs: Reduced connected guard, disconnected guard, and skipped edit.
    # Outputs: Review for reduced distance, rejection for disconnection, and skipped-edit issue.
    def test_guard_reduction_review_and_guard_violation_reject(self):
        guard = {'status': 'reduced', 'connectivity_ok': True, 'baseline_target_voxels': 50,
                 'retained_volume_ratio': .7, 'attempts': [{}, {}]}
        step = self.step(accepted_distance_mm=2., erosion_safeguard=guard)
        self.assertEqual(review.summarize_step(step)['review_state'], 'needs_review')
        guard['connectivity_ok'] = False
        self.assertEqual(review.summarize_step(step)['review_state'], 'reject')
        guard.update(status='skipped', connectivity_ok=True)
        step['accepted_distance_mm'] = 0
        self.assertIn('erosion_guard_skipped_edit', self.reasons(review.summarize_step(step)))

    # Purpose: Verify connectivity review considers the full target crop.
    # Inputs: Full components increase 2 to 3 while ROI components stay 1.
    # Outputs: Needs-review state and full-target-component-count issue.
    def test_component_count_warning_uses_full_crop_not_clipped_roi(self):
        native = {'full_target_components_before': 2, 'full_target_components_after': 3,
                  'components_before': 1, 'components_after': 1}
        row = review.summarize_step(self.step(), native)
        self.assertEqual(row['review_state'], 'needs_review')
        self.assertIn('full_target_component_count_changed', self.reasons(row))

    # Purpose: Verify only declared baseline operations may be baseline-equivalent noops.
    # Inputs: Same zero-change sample marked operation none then recipe.
    # Outputs: No flags for baseline and reject for ineffective recipe.
    def test_baseline_noop_is_expected_but_edited_baseline_is_rejected(self):
        sample = self.sample(operation='none', baseline_equivalent=True, zero_change=True,
            step_outcomes=[self.step(operation='none', status='no_change', requested_distance_mm=0,
                                     accepted_distance_mm=0, removed_voxels=0)])
        self.assertEqual(review.summarize_sample(sample)['review_state'], 'no_flags')
        sample['operation'] = 'recipe'
        self.assertEqual(review.summarize_sample(sample)['review_state'], 'reject')

    # Purpose: Verify failed variants retain their diagnostic error.
    # Inputs: Failed sample with no steps and error no recipient.
    # Outputs: Reject state with the original no-recipient error.
    def test_failed_variant_preserves_error(self):
        row = review.summarize_sample(self.sample(status='failed', error='no recipient', step_outcomes=[]))
        self.assertEqual(row['review_state'], 'reject')
        self.assertEqual(row['error'], 'no recipient')

    # Purpose: Verify cohort review detects duplicates/input drift and protects artifacts.
    # Inputs: Two matching preflights, unsafe project title, changed input INI, and repeated review output.
    # Outputs: Duplicate findings, escaped HTML, drift issue, preserved preflights, and overwrite protection.
    def test_project_reports_duplicate_crops_and_changed_inputs_without_touching_artifacts(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp)
            project = root / 'project.json'
            cases = []
            configs = {}
            before = {}
            for name in ('a', 'b'):
                config = root / (name + '.ini')
                config.write_text('preserve this input')
                folder = root / name
                folder.mkdir()
                document = self.document()
                document['input_config_sha256'] = hashlib.sha256(config.read_bytes()).hexdigest()
                path = folder / 'preflight.json'
                path.write_text(json.dumps(document))
                before[path] = path.read_bytes()
                cases.append({'case_id': name, 'config': config.name})
                configs[config] = {'train': {'preflight_directory': folder}}
            project.write_text(json.dumps({'schema_version': 'fakect.cohort-project/1',
                                          'name': '<script>unsafe</script>', 'cases': cases}))
            with patch.object(review, 'load_study_config', side_effect=lambda path, **kw: configs[path]):
                result = review.collect_project(project, root)
                self.assertEqual(result['review_state'], 'needs_review')
                self.assertEqual(len(result['duplicate_source_crops']), 1)
                self.assertEqual(len(result['duplicate_target_masks']), 1)
                self.assertFalse(result['freeze_approved'])
                self.assertNotIn('<script>', review.render_html(result))
                (root / 'a.ini').write_text('changed')
                changed = review.collect_project(project, root)
                self.assertIn('input_bytes_differ_from_preflight', self.reasons(changed['cases'][0]))
            self.assertTrue(all(path.read_bytes() == raw for path, raw in before.items()))
            output = root / 'review.json'
            review.write_review(result, output)
            with self.assertRaises(FileExistsError):
                review.write_review(result, output)
            review.write_review(result, output, overwrite=True)

    # Purpose: Verify outcome review rejects incomplete or inconsistent preflight evidence.
    # Inputs: Empty document, sample without steps, and variant-count mismatch.
    # Outputs: ValueError for invalid schema plus rejection/count-mismatch issues.
    def test_invalid_document_and_missing_steps_reject(self):
        with self.assertRaises(ValueError):
            review.summarize_preflight({}, 'a')
        self.assertEqual(review.summarize_sample(self.sample(step_outcomes=[]))['review_state'], 'reject')
        document = self.document()
        document['variant_count'] = 2
        self.assertIn('preflight_variant_count_mismatch_or_empty',
                      self.reasons(review.summarize_preflight(document, 'a')))

    # Purpose: Verify duplicate comparison keys require matching known geometry.
    # Inputs: Missing voxel geometry and otherwise identical summaries with different spacing.
    # Outputs: Unknown geometry yields no key and different spacing yields distinct keys.
    def test_hash_comparisons_require_known_voxel_geometry(self):
        self.assertIsNone(review._geometry_key({'voxel_geometry': {}}))
        a = review.summarize_sample(self.sample())
        b = deepcopy(a)
        b['voxel_geometry']['spacing_ijk_mm'] = [.8, .8, .8]
        self.assertNotEqual(review._geometry_key(a), review._geometry_key(b))

    # Purpose: Verify distance-response audits compare each axis at fixed companion settings.
    # Inputs: 3x5 growth/erosion sweep then repeated removal count and target-mask hash.
    # Outputs: Eight increasing clean curves and explicit equal-count/duplicate-mask warnings.
    def test_distance_audit_holds_other_axes_fixed_and_detects_quantization(self):
        samples = []
        for grow in (1., 2., 3.):
            for erosion in (1., 2., 3., 4., 5.):
                steps = [self.step('dilation', step_name='grow', added_voxels=int(10*grow),
                                   requested_distance_mm=grow, accepted_distance_mm=grow),
                         self.step('erosion', step_name='shrink', removed_voxels=int(20*erosion),
                                   requested_distance_mm=erosion, accepted_distance_mm=erosion)]
                sample = self.sample(variant_id=f'{grow}-{erosion}', mask_sha256=f'{grow}-{erosion}',
                    parameters={'grow': {'distance_mm': grow}, 'shrink': {'distance_mm': erosion}},
                    step_outcomes=steps)
                samples.append(review.summarize_sample(sample))
        responses = review.distance_responses(samples)
        self.assertEqual(len(responses), 8)  # Five dilation curves and three erosion curves.
        self.assertTrue(all(row['strictly_increasing_change_voxels'] for row in responses))
        self.assertTrue(all(row['review_state'] == 'no_flags' for row in responses))
        samples[1]['steps'][1]['removed_voxels'] = samples[0]['steps'][1]['removed_voxels']
        samples[1]['mask_sha256'] = samples[0]['mask_sha256']
        responses = review.distance_responses(samples)
        affected = next(row for row in responses if row['edit_name'] == 'shrink' and row['held_sweep_parameters'] == {'grow.distance_mm': 1.})
        self.assertIn('equal_change_counts_at_distinct_distances', self.reasons(affected))
        self.assertIn('distance_axis_contains_duplicate_target_masks', self.reasons(affected))

    # Purpose: Verify guard backoff can trigger nonmonotonic severity warnings.
    # Inputs: Requested erosions 1/2/3 with accepted distances 1/2/1.5 and removals 20/40/30.
    # Outputs: Nonmonotonic change and retention flags in the distance response.
    def test_backoff_can_reverse_achieved_severity_and_is_flagged(self):
        samples = []
        for request, accepted, removed, ratio in ((1., 1., 20, .8), (2., 2., 40, .6), (3., 1.5, 30, .7)):
            step = self.step('erosion', requested_distance_mm=request, accepted_distance_mm=accepted,
                removed_voxels=removed, erosion_safeguard={'status': 'accepted' if request == accepted else 'reduced',
                    'connectivity_ok': True, 'retained_volume_ratio': ratio, 'baseline_target_voxels': 100})
            samples.append(review.summarize_sample(self.sample(variant_id=str(request), mask_sha256=str(request),
                parameters={'arch': {'distance_mm': request}}, step_outcomes=[step])))
        row = review.distance_responses(samples)[0]
        self.assertFalse(row['nondecreasing_change_voxels'])
        self.assertFalse(row['nonincreasing_retained_volume_ratio'])
        self.assertIn('nonmonotonic_achieved_distance_response', self.reasons(row))
        self.assertIn('nonmonotonic_retained_volume_ratio', self.reasons(row))


if __name__ == '__main__':
    unittest.main()
