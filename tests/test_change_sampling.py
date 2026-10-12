"""TRAIN isolation, actual positive patches and balanced burden exposure."""
from copy import deepcopy
from collections import Counter
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

sys.path[:0] = [str(Path(__file__).resolve().parents[1] / 'src'), str(Path(__file__).resolve().parent)]

from fakect_change_sampling import (BalancedChangeSchedule, EmptyTargetClassError,
    InvalidTargetPatchError, available_target_classes, strict_training_patch)
import test_change_data as data_fixture


def row(case, name, *, added=0, removed=0, baseline=False):
    return {'sample_id': f'{case}:{name}', 'case_id': case, 'split': 'train',
        'baseline': baseline, 'mask_counts': {'before': 100, 'after': 100 + added - removed,
            'added': added, 'removed': removed}}


def roster(cases=('a', 'b')):
    rows = []
    for case in cases:
        rows.append(row(case, 'baseline', baseline=True))
        for code, mode in enumerate(('added', 'removed'), start=1):
            for index in range(9):
                rows.append(row(case, f'{mode}-{index}', **{mode: index + code}))
    return rows


class ScheduleTests(unittest.TestCase):
    # Purpose: Verify balanced training schedules exclude non-TRAIN data.
    # Inputs: Rows marked final_test, validation, unassigned, or absent and an outside case.
    # Outputs: ValueError for split leakage and cases outside the TRAIN roster.
    def test_train_roster_is_required_and_heldout_rows_cannot_enter(self):
        rows = roster()
        for role in ('final_test', 'validation', 'unassigned', None):
            bad = deepcopy(rows)
            bad[-1]['split'] = role
            with self.assertRaisesRegex(ValueError, 'TRAIN'):
                BalancedChangeSchedule(bad, ['a', 'b'], 42)
        bad = rows + [row('heldout', 'baseline', baseline=True)]
        with self.assertRaisesRegex(ValueError, 'outside the TRAIN roster'):
            BalancedChangeSchedule(bad, ['a', 'b'], 42)

    # Purpose: Verify update budgets balance cases, classes, burden strata, and controls.
    # Inputs: Two-case roster with nine siblings per change class across three blocks.
    # Outputs: Equal strata exposure, four sibling visits, twelve baseline visits, and one-seventh controls.
    def test_case_class_burden_and_original_controls_are_balanced(self):
        schedule = BalancedChangeSchedule(roster(), ['a', 'b'], 42)
        draws = sum((schedule.block() for _ in range(3)), [])
        self.assertEqual(len(draws), 3 * 28 * 2)
        self.assertEqual(Counter(d['case_id'] for d in draws), {'a': 84, 'b': 84})
        self.assertEqual(sum(d['control'] for d in draws), len(draws) // 7)
        self.assertEqual(Counter(d['mode'] for d in draws if not d['control']), {'added': 72, 'removed': 72})
        strata = Counter((d['case_id'], d['mode'], d['burden_bin']) for d in draws if not d['control'])
        self.assertEqual(set(strata.values()), {12})
        visits = schedule.receipt()['sample_patch_visits']
        self.assertEqual({v for sid, v in visits.items() if 'baseline' not in sid}, {4})
        self.assertEqual({v for sid, v in visits.items() if 'baseline' in sid}, {12})
        self.assertFalse(schedule.receipt()['reporting_block_is_complete_population_epoch'])

    # Purpose: Verify single-class rosters schedule only available positives.
    # Inputs: One baseline and two added-only siblings.
    # Outputs: Edited draws request added class 1 exclusively.
    def test_single_type_never_requests_absent_class(self):
        rows = [row('a', 'baseline', baseline=True), row('a', 'one', added=10), row('a', 'two', added=30)]
        schedule = BalancedChangeSchedule(rows, ['a'], 7)
        draws = schedule.block()
        self.assertEqual({d['mode'] for d in draws if not d['control']}, {'added'})
        self.assertEqual({d['class_id'] for d in draws if not d['control']}, {1})

    # Purpose: Verify equal burdens form one auditable stratum.
    # Inputs: Two added-only siblings with equal burden 10.
    # Outputs: All edited draws use medium bin and receipt marks equal cutoffs.
    def test_equal_burdens_use_one_explicit_populated_stratum(self):
        rows = [row('a', 'baseline', baseline=True), row('a', 'one', added=10), row('a', 'two', added=10)]
        schedule = BalancedChangeSchedule(rows, ['a'], 7)
        self.assertEqual({d['burden_bin'] for d in schedule.block() if not d['control']}, {'medium'})
        self.assertTrue(schedule.receipt()['strata'][0]['equal_cutoffs'])

    # Purpose: Verify edited noops remain declared outside positive/control draws.
    # Inputs: One-case roster plus an edited sibling with zero net change.
    # Outputs: No noop draws, zero recorded visits, and four baseline control updates.
    def test_noops_are_explicit_and_never_implicit_positive_or_background_fallback(self):
        rows = roster(('a',)) + [row('a', 'edited-noop')]
        schedule = BalancedChangeSchedule(rows, ['a'], 42)
        draws = schedule.block()
        self.assertNotIn('a:edited-noop', [d['sample_id'] for d in draws])
        receipt = schedule.receipt()
        self.assertEqual(receipt['edited_noop_ids'], ['a:edited-noop'])
        self.assertEqual(receipt['sample_patch_visits']['a:edited-noop'], 0)
        self.assertEqual(receipt['control_updates'], 4)

    # Purpose: Verify schedule reproducibility ignores input roster ordering.
    # Inputs: Reversed rows/case order with seed 42 and alternate seed 43.
    # Outputs: Five identical blocks and receipts for seed 42, differing draws for seed 43.
    def test_queues_and_receipts_reproduce_with_same_seed(self):
        a = BalancedChangeSchedule(roster(), ['b', 'a'], 42)
        b = BalancedChangeSchedule(list(reversed(roster())), ['a', 'b'], 42)
        for _ in range(5):
            self.assertEqual(a.block(), b.block())
        self.assertEqual(a.receipt(), b.receipt())
        other = BalancedChangeSchedule(roster(), ['a', 'b'], 43)
        self.assertNotEqual(a.block(), other.block())

    # Purpose: Verify schedule invariants reject malformed burdens and update budgets.
    # Inputs: Negative/fractional/impossible counts, missing controls/positives, and invalid update counts.
    # Outputs: ValueError for every invalid roster and noninteger/nonmultiple-of-seven budget.
    def test_bad_counts_no_control_no_positive_and_update_fraction_rejected(self):
        for field, value in [('added', -1), ('removed', 101), ('before', 0), ('added', 1.5)]:
            rows = roster(('a',))
            rows[-1]['mask_counts'][field] = value
            with self.assertRaises(ValueError):
                BalancedChangeSchedule(rows, ['a'], 42)
        with self.assertRaisesRegex(ValueError, 'exactly one'):
            BalancedChangeSchedule(roster(('a',))[1:], ['a'], 42)
        with self.assertRaisesRegex(ValueError, 'positive change'):
            BalancedChangeSchedule([row('a', 'baseline', baseline=True), row('a', 'noop')], ['a'], 42)
        schedule = BalancedChangeSchedule(roster(('a',)), ['a'], 42)
        for updates in (0, 6, 8, 28.0, True):
            with self.assertRaisesRegex(ValueError, 'multiple of seven'):
                schedule.block(updates)


class StrictPatchTests(unittest.TestCase):
    def setUp(self):
        self.fixture = data_fixture.FocusedChangeVolumeTests(methodName='test_training_class_sampling_contains_requested_class')
        self.fixture.setUp()
        self.addCleanup(self.fixture.doCleanups)

    # Purpose: Verify strict training checks positives after anisotropic resampling.
    # Inputs: Focused-change fixture and seeded added/removed 4-cubed patches.
    # Outputs: Every patch contains its requested class and records verified positivity without fallback.
    def test_actual_anisotropic_resampling_preserves_requested_positive(self):
        reader = self.fixture.open()
        self.assertEqual(available_target_classes(reader), ('added', 'removed'))
        rng = np.random.default_rng(4)
        for mode, code in [('added', 1), ('removed', 2)]:
            for _ in range(5):
                result = strict_training_patch(reader, [4, 4, 4], rng, mode, changed_fraction=1.)
                self.assertTrue(np.any(result['data']['labels'] == code))
                self.assertTrue(result['audit']['positive_class_verified_after_resampling'])
                self.assertFalse(result['audit']['missing_class_fallback'])
                self.assertIsNone(result['audit']['uniform_reason'])

    # Purpose: Verify centered diagnostics use actual changed voxels without altering focus.
    # Inputs: Added-class centered patch and saved origins/inference tiles.
    # Outputs: Valid clipped centering offset, class 1 present, and unchanged focus/schedule.
    def test_centered_diagnostic_contains_real_class_and_keeps_focus_fixed(self):
        reader = self.fixture.open()
        origins = reader.origins_kji.copy()
        inference = list(reader.inference_tiles([4, 4, 4]))
        result = strict_training_patch(reader, [4, 4, 4], np.random.default_rng(4),
            'added', changed_fraction=1., centered=True)
        self.assertTrue(np.any(result['data']['labels'] == 1))
        points = reader._change_points(1) - reader.origins_kji[0]
        expected = {tuple(np.clip(p - 2, 0, reader.shape_kji - 4)) for p in points
            if np.all((p >= 0) & (p < reader.shape_kji))}
        self.assertIn(result['tile']['offset_kji'], expected)
        self.assertEqual(result['audit']['centering_policy'], 'verified_changed_voxel_at_center')
        np.testing.assert_array_equal(reader.origins_kji, origins)
        self.assertEqual(inference, list(reader.inference_tiles([4, 4, 4])))

    # Purpose: Verify declared uniform sampling avoids target-based crop selection.
    # Inputs: Uniform mode with available_classes mocked to raise.
    # Outputs: Uniform audit reason and no target-class lookup.
    def test_uniform_is_declared_and_never_consults_targets_to_select_crop(self):
        reader = self.fixture.open()
        rng = np.random.default_rng(4)
        with patch.object(reader, 'available_classes', side_effect=AssertionError('uniform target leakage')):
            result = strict_training_patch(reader, [4, 4, 4], rng, 'uniform')
        self.assertEqual(result['audit']['realized_mode'], 'uniform')
        self.assertEqual(result['audit']['uniform_reason'], 'declared_uniform')
        self.assertIsNone(result['audit']['available_target_classes'])

    # Purpose: Verify requested background sampling is audited explicitly.
    # Inputs: Added mode with changed_fraction 0 on a positive fixture.
    # Outputs: Uniform realized mode with declared-background reason and no missing-class fallback.
    def test_declared_background_fraction_is_distinct_from_missing_class_fallback(self):
        reader = self.fixture.open()
        result = strict_training_patch(reader, [4, 4, 4], np.random.default_rng(4), 'added', changed_fraction=0.)
        self.assertEqual(result['audit']['requested_mode'], 'added')
        self.assertEqual(result['audit']['realized_mode'], 'uniform')
        self.assertEqual(result['audit']['uniform_reason'], 'declared_background_fraction')
        self.assertFalse(result['audit']['missing_class_fallback'])

    # Purpose: Verify empty requested classes never silently fall back to background.
    # Inputs: Noop native targets with changed_fraction 0 and 1.
    # Outputs: EmptyTargetClassError specifying no uniform fallback.
    def test_absent_native_target_raises_even_when_background_fraction_is_one(self):
        self.fixture.mask_data.update(after=self.fixture.before, added=np.zeros_like(self.fixture.before),
            removed=np.zeros_like(self.fixture.before))
        self.fixture.save_masks()
        reader = self.fixture.open()
        for fraction in (0., 1.):
            with self.assertRaisesRegex(EmptyTargetClassError, 'no uniform fallback'):
                strict_training_patch(reader, [4, 4, 4], np.random.default_rng(4), 'added', changed_fraction=fraction)

    # Purpose: Verify positives lost during model-grid resampling are rejected.
    # Inputs: Native changes resampled to 2-cubed grid with 20 mm spacing.
    # Outputs: No available model-grid classes and EmptyTargetClassError for removed sampling.
    def test_native_positive_erased_by_model_grid_raises(self):
        runtime = deepcopy(self.fixture.runtime)
        runtime['input.detail'].update(shape_kji=[2, 2, 2], spacing_ijk_mm=[20., 20., 20.])
        reader = self.fixture.open(runtime=runtime, origins=[[0, 0, 0]])
        self.assertTrue(np.any(reader.volume.mask_before != reader.volume.mask_after))
        self.assertEqual(available_target_classes(reader), ())
        with self.assertRaisesRegex(EmptyTargetClassError, 'model-grid'):
            strict_training_patch(reader, [1, 1, 1], np.random.default_rng(4), 'removed', changed_fraction=1.)

    # Purpose: Verify strict sampling checks returned labels rather than reader claims.
    # Inputs: Reader whose read_patch erases all labels after target-centered selection.
    # Outputs: InvalidTargetPatchError for the actual resampled patch.
    def test_reader_claim_cannot_replace_verification_of_actual_patch(self):
        reader = self.fixture.open()
        original = reader.read_patch
        def erased(**tile):
            data = original(**tile)
            data['labels'][:] = 0
            return data
        with patch.object(reader, 'read_patch', side_effect=erased):
            with self.assertRaisesRegex(InvalidTargetPatchError, 'actual resampled patch'):
                strict_training_patch(reader, [4, 4, 4], np.random.default_rng(4), 'added', changed_fraction=1.)


if __name__ == '__main__':
    unittest.main()
