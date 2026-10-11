"""Leakage-free grouped experiment designs and strict high-level INI inputs."""
import copy
import itertools
from pathlib import Path
import sys
import unittest
from unittest.mock import patch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_evaluation_design import (build_evaluation_design, parse_evaluation,
                                       validate_assignment)


def datasets(sizes):
    return [{'case_id': f'case-{family}-{index}', 'anatomy_family': f'family-{family}'}
            for family, size in enumerate(sizes) for index in range(size)]


class EvaluationParserTests(unittest.TestCase):
    def test_explicit_holdout_and_defaults(self):
        parsed = parse_evaluation({'strategy': 'holdout', 'case_counts': '8, 2, 2'})
        self.assertEqual(parsed, {'strategy': 'holdout', 'case_counts': [8, 2, 2],
                                  'seeds': [42], 'threshold': .5})

    def test_leave_one_out_defaults_and_multiple_training_seeds(self):
        parsed = parse_evaluation({'strategy': 'leave_one_out', 'seeds': '42,1729,2026'})
        self.assertEqual(parsed['validation_families'], 1)
        self.assertEqual(parsed['seeds'], [42, 1729, 2026])
        nested = parse_evaluation({'strategy': 'nested_leave_one_out'})
        self.assertNotIn('case_counts', nested)
        self.assertNotIn('validation_families', nested)

    def test_planning_project_is_explicit_nonblank_path(self):
        result = parse_evaluation({'strategy': 'leave_one_out', 'planning_project': 'configs/cohort.ini'})
        self.assertEqual(result['planning_project'], 'configs/cohort.ini')
        for value in ('', '  ', 'a\nb', 'a\rb', 'a\0b'):
            with self.subTest(value=value), self.assertRaises(ValueError):
                parse_evaluation({'strategy': 'leave_one_out', 'planning_project': value})

    def test_reject_misspellings_implicit_modes_and_conflicting_fields(self):
        cases = [{}, {'strategy': 'typo'}, {'strategy': 'holdout'},
                 {'strategy': 'leave_one_out', 'case_counts': '8,2,2'},
                 {'strategy': 'nested_leave_one_out', 'case_counts': '8,2,2'},
                 {'strategy': 'nested_leave_one_out', 'validation_families': '1'},
                 {'strategy': 'holdout', 'case_counts': '8,2,2', 'validation_families': '1'},
                 {'strategy': 'leave_one_out', 'seed': '42'}]
        for case in cases:
            with self.subTest(case=case), self.assertRaises(ValueError):
                parse_evaluation(case)

    def test_reject_bad_counts_seeds_and_nonfinite_thresholds(self):
        for counts in ('8,2', '8,2,2,1', '8,0,2', '8,-2,2', '8,2.0,2', '8,2,', '2147483648,2,2'):
            with self.subTest(counts=counts), self.assertRaises(ValueError):
                parse_evaluation({'strategy': 'holdout', 'case_counts': counts})
        for seeds in ('', '1,1', '-1', '1.0', 'nan', '42,', '2147483648',
                      ','.join(map(str, range(101)))):
            with self.subTest(seeds=seeds), self.assertRaises(ValueError):
                parse_evaluation({'strategy': 'leave_one_out', 'seeds': seeds})
        for threshold in ('nan', 'inf', '-inf', '1e1000', '0', '1', '-.01', '1.01', 'text'):
            with self.subTest(threshold=threshold), self.assertRaises(ValueError):
                parse_evaluation({'strategy': 'leave_one_out', 'threshold': threshold})
        for value in ('0', '-1', '1.5', ''):
            with self.subTest(validation_families=value), self.assertRaises(ValueError):
                parse_evaluation({'strategy': 'leave_one_out', 'validation_families': value})

    def test_reject_multiline_or_nonstring_values(self):
        for values in ({'strategy': 'leave_one_out\n'}, {'strategy': 'leave_one_out', 'seeds': '1\r2'},
                       {'strategy': 'leave_one_out', 'threshold': .5}, None):
            with self.subTest(values=values), self.assertRaises(ValueError):
                parse_evaluation(values)


class EvaluationDesignTests(unittest.TestCase):
    def design(self, strategy, sizes=(2, 2, 2, 2, 2, 2), **values):
        if strategy == 'holdout':
            values.setdefault('case_counts', '8,2,2')
        return build_evaluation_design(datasets(sizes), parse_evaluation({'strategy': strategy, **values}))

    def assert_complete_assignment(self, design, fold, *, validation=True, train_min=1):
        assignment = fold['family_assignment']
        counts = validate_assignment(design['families'], assignment,
                                     require_validation=validation, min_training_families=train_min)
        self.assertEqual(counts['case_counts'], fold['case_counts'])
        self.assertEqual(counts['family_counts'], fold['family_counts'])
        self.assertEqual(sum(fold['case_counts'].values()), design['counts']['cases'])
        self.assertEqual(sum(fold['family_counts'].values()), design['counts']['families'])
        self.assertFalse(set(fold['test_families']) & set(fold['development_families']))

    def test_holdout_assigns_exact_case_counts_and_complete_families(self):
        design = self.design('holdout')
        self.assertEqual(design['counts']['outer_folds'], 1)
        fold = design['folds'][0]
        self.assert_complete_assignment(design, fold)
        self.assertEqual(fold['case_counts'], {'train': 8, 'validation': 2, 'test': 2})
        self.assertEqual(fold['family_counts'], {'train': 4, 'validation': 1, 'test': 1})
        self.assertEqual(fold['inner_folds'], [])

    def test_unequal_families_can_have_exact_case_partition(self):
        for sizes, counts in (((3, 2, 1, 1, 1), '4,2,2'), ((4, 3, 2, 1), '3,5,2'),
                              ((5, 2, 2, 1), '2,2,6'), ((1, 1, 1), '1,1,1')):
            with self.subTest(sizes=sizes, counts=counts):
                design = self.design('holdout', sizes, case_counts=counts)
                fold = design['folds'][0]
                self.assert_complete_assignment(design, fold)
                self.assertEqual(list(fold['case_counts'].values()), list(map(int, counts.split(','))))

    def test_impossible_holdout_reports_family_sizes_instead_of_leaking(self):
        with self.assertRaisesRegex(ValueError, 'without splitting anatomy families.*Family sizes'):
            self.design('holdout', (2, 2, 2), case_counts='4,1,1')
        with self.assertRaisesRegex(ValueError, 'distinct cases'):
            self.design('holdout', (2, 2, 2), case_counts='8,2,2')

    def test_exact_partition_matches_exhaustive_small_reference(self):
        for sizes in itertools.product((1, 2), repeat=4):
            feasible = set()
            for allocation in itertools.product(range(3), repeat=4):
                counts = tuple(sum(size for size, group in zip(sizes, allocation) if group == split)
                               for split in range(3))
                if min(counts) > 0:
                    feasible.add(counts)
            total = sum(sizes)
            for train in range(1, total-1):
                for validation in range(1, total-train):
                    counts = (train, validation, total-train-validation)
                    with self.subTest(sizes=sizes, counts=counts):
                        if counts in feasible:
                            design = self.design('holdout', sizes, case_counts=','.join(map(str, counts)))
                            self.assertEqual(tuple(design['folds'][0]['case_counts'].values()), counts)
                        else:
                            with self.assertRaisesRegex(ValueError, 'without splitting anatomy families'):
                                self.design('holdout', sizes, case_counts=','.join(map(str, counts)))

    def test_design_stable_under_input_order_or_training_seed_changes(self):
        rows = datasets([2]*6)
        settings = parse_evaluation({'strategy': 'holdout', 'case_counts': '8,2,2'})
        baseline = build_evaluation_design(rows, settings, split_seed=42)
        reordered = build_evaluation_design(list(reversed(rows)), settings, split_seed=42)
        self.assertEqual(baseline, reordered)
        repeated = build_evaluation_design(rows, {**settings, 'seeds': [42, 1729, 2026]}, split_seed=42)
        self.assertEqual(baseline['folds'], repeated['folds'])
        self.assertEqual(repeated['counts']['training_seeds'], 3)
        different = {tuple(build_evaluation_design(rows, settings, split_seed=seed)['folds'][0]
                           ['family_assignment'].items()) for seed in range(10)}
        self.assertGreater(len(different), 1)

    def test_leave_one_out_tests_each_family_once(self):
        design = self.design('leave_one_out')
        held_out = []
        for fold in design['folds']:
            self.assert_complete_assignment(design, fold)
            self.assertEqual(fold['case_counts'], {'train': 8, 'validation': 2, 'test': 2})
            self.assertEqual(len(fold['test_families']), 1)
            held_out.extend(fold['test_families'])
        self.assertEqual(sorted(held_out), list(design['families']))
        self.assertEqual(design['counts']['inner_folds'], 0)
        alternate = self.design('leave_one_out', seeds='42,2026')
        self.assertEqual(design['folds'], alternate['folds'])

    def test_multiple_validation_families_leave_training_anatomy(self):
        design = self.design('leave_one_out', validation_families='2')
        for fold in design['folds']:
            self.assert_complete_assignment(design, fold)
            self.assertEqual(fold['family_counts'], {'train': 3, 'validation': 2, 'test': 1})
        with self.assertRaisesRegex(ValueError, 'no training family'):
            self.design('leave_one_out', validation_families='5')

    def test_nested_design_has_six_outer_thirty_inner_folds(self):
        design = self.design('nested_leave_one_out')
        self.assertEqual(design['counts'], {'cases': 12, 'families': 6, 'outer_folds': 6,
                                           'inner_folds': 30, 'training_seeds': 1})
        for outer in design['folds']:
            self.assert_complete_assignment(design, outer, validation=False)
            self.assertEqual(outer['case_counts'], {'train': 10, 'validation': 0, 'test': 2})
            validated = []
            for inner in outer['inner_folds']:
                self.assert_complete_assignment(design, inner, train_min=2)
                self.assertEqual(inner['test_families'], outer['test_families'])
                self.assertEqual(inner['case_counts'], {'train': 8, 'validation': 2, 'test': 2})
                validated.extend(inner['validation_families'])
            self.assertEqual(sorted(validated), outer['development_families'])
        repeated = self.design('nested_leave_one_out', seeds='10,11')
        self.assertEqual(design['folds'], repeated['folds'])

    def test_repeated_receipts_count_a_case_once_and_do_not_mutate_inputs(self):
        rows = datasets([2]*6)
        rows += [{**rows[0], 'name': 'another-receipt', 'frame': 2}, dict(rows[1])]
        settings = parse_evaluation({'strategy': 'holdout', 'case_counts': '8,2,2'})
        original_rows, original_settings = copy.deepcopy(rows), copy.deepcopy(settings)
        design = build_evaluation_design(rows, settings)
        self.assertEqual(design, build_evaluation_design(datasets([2]*6), settings))
        self.assertEqual(rows, original_rows)
        self.assertEqual(settings, original_settings)

    def test_case_cannot_cross_families_including_legacy_integer_ids(self):
        rows = datasets([2]*6)
        rows += [{'case_id': rows[0]['case_id'], 'anatomy_family': 'new-family'}]
        settings = parse_evaluation({'strategy': 'leave_one_out'})
        with self.assertRaisesRegex(ValueError, 'multiple anatomy families'):
            build_evaluation_design(rows, settings)
        rows = datasets([1, 1, 1]) + [{'case_id': 7, 'anatomy_family': 'family-0'},
                                     {'case_id': '7', 'anatomy_family': 'family-1'}]
        with self.assertRaisesRegex(ValueError, 'multiple anatomy families'):
            build_evaluation_design(rows, settings)

    def test_insufficient_families_and_unidentified_data(self):
        for strategy, size in (('holdout', 2), ('leave_one_out', 2), ('nested_leave_one_out', 3)):
            with self.subTest(strategy=strategy), self.assertRaisesRegex(ValueError, 'independent anatomy families'):
                self.design(strategy, [1]*size)
        for row in ({'case_id': 'a'}, {'case_id': 'a', 'anatomy_family': ''},
                    {'case_id': 'a', 'anatomy_family': 'family\nother'},
                    {'case_id': True, 'anatomy_family': 'family'},
                    {'case_id': ' ', 'anatomy_family': 'family'}):
            with self.subTest(row=row), self.assertRaises(ValueError):
                build_evaluation_design([row], parse_evaluation({'strategy': 'leave_one_out'}))

    def test_planning_size_is_bounded(self):
        with self.assertRaisesRegex(ValueError, 'at most 128'):
            self.design('leave_one_out', [1]*129)
        with patch('fakect_evaluation_design.MAX_DP_STATES', 3):
            with self.assertRaisesRegex(ValueError, 'planning budget'):
                self.design('holdout')

    def test_typed_inputs_are_revalidated(self):
        settings = parse_evaluation({'strategy': 'leave_one_out'})
        for update in ({'seeds': [True]}, {'seeds': [-1]}, {'seeds': [1, 1]}, {'seeds': '42'},
                       {'threshold': float('nan')}, {'threshold': True}, {'validation_families': True}):
            with self.subTest(update=update), self.assertRaises(ValueError):
                build_evaluation_design(datasets([2]*6), {**settings, **update})
        for seed in (-1, True, 1.5, '42'):
            with self.subTest(seed=seed), self.assertRaises(ValueError):
                build_evaluation_design(datasets([2]*6), settings, split_seed=seed)

    def test_assignment_rejects_missing_families_unknown_splits_or_shared_cases(self):
        families = {'a': ['case-a'], 'b': ['case-b'], 'c': ['case-c']}
        assignment = {'a': 'train', 'b': 'validation', 'c': 'test'}
        self.assertEqual(validate_assignment(families, assignment)['case_counts'],
                         {'train': 1, 'validation': 1, 'test': 1})
        for invalid in ({'a': 'train', 'b': 'test'}, {**assignment, 'a': 'testing'},
                        {**assignment, 'a': 'validation'}, {**assignment, 'c': 'train'}):
            with self.subTest(assignment=invalid), self.assertRaises(ValueError):
                validate_assignment(families, invalid)
        with self.assertRaisesRegex(ValueError, 'exactly once'):
            validate_assignment({**families, 'b': ['case-a']}, assignment)
        with self.assertRaisesRegex(ValueError, 'Final refit'):
            validate_assignment(families, assignment, require_validation=False)


if __name__ == '__main__':
    unittest.main()
