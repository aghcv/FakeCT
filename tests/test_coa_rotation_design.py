"""Origin isolation and declared age/sex comparisons, without image payloads."""
from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import fakect_coa_rotation_design as design


def fixture():
    project = {'schema_version': 'fakect.cohort-project/1', 'cases': []}
    rows, labels = [], []
    for age, first in (('adult', 10), ('infant', 20), ('1yr', 30), ('5yr', 40), ('10yr', 50), ('15yr', 60)):
        for offset, sex in enumerate(('male', 'female')):
            case = str(first+offset)
            family = 'family_'+age
            template = ('v'+sex+'50.nrb') if age == 'adult' else sex+'_'+age+'_ref.nrb'
            project['cases'].append({'case_id': case, 'template': template, 'provisional_family': family})
            sid = case+':baseline'
            rows.append({'sample_id': sid, 'case_id': case, 'anatomy_family': family, 'baseline': True, 'parameters': {}})
            labels.append({'sample_id': sid, 'case_id': case, 'anatomy_family': family,
                           'original_baseline': True, 'coa_edit_applied': False,
                           'coa_benchmark_positive': False, design.SCORE: 0., 'errors': []})
            for companion in range(3):
                for severity in (.05, .15, .21, .35, .55):
                    sid = f'{case}:edit-{companion}-{severity}'
                    rows.append({'sample_id': sid, 'case_id': case, 'anatomy_family': family,
                                 'baseline': False, 'parameters': {
                                     'arch_outer_expand': {'distance_mm': companion+1},
                                     'arch_inner_narrow': {'distance_mm': 2},
                                     'coa_narrow': {'distance_mm': severity*10}}})
                    labels.append({'sample_id': sid, 'case_id': case, 'anatomy_family': family,
                                   'original_baseline': False, 'coa_edit_applied': True,
                                   'coa_benchmark_positive': severity >= .2,
                                   design.SCORE: severity, 'errors': []})
    inventory = design.build_inventory(project)
    panel = design.select_matched_panel(rows, labels, inventory)
    return project, rows, labels, inventory, panel


class RotationDesignTests(unittest.TestCase):
    def test_inventory_does_not_invent_infant_age_or_confirm_ancestry(self):
        _, _, _, inventory, _ = fixture()
        self.assertEqual({r['age_years'] for r in inventory if r['age_bin'] == 'infant'}, {None})
        self.assertTrue(all(r['family_verified'] is False for r in inventory))
        self.assertEqual({r['age_years'] for r in inventory if r['age_bin'] == 'adult'}, {50})

    def test_all_cases_and_siblings_are_partitioned_not_only_selected_images(self):
        _, _, _, inventory, panel = fixture()
        value = design.build_rotation_design(inventory, panel)
        self.assertEqual(len(value['stages']), 45)
        self.assertEqual(len(value['samples']), 60)
        self.assertEqual(len(value['baseline_samples']), 10)
        self.assertEqual(len(value['reserved_sample_ids']), 32)
        for stage in value['stages']:
            self.assertTrue(design.assert_isolation(stage, inventory, panel))
            self.assertEqual(len(stage['all_final_test_sample_ids']), 16*len(stage['final_test_case_ids']))
            self.assertEqual({stage['case_assignment'][case] for case in ('20', '21')}, {'excluded'})
            self.assertNotIn('20', stage['train_case_ids']+stage['validation_case_ids']+stage['final_test_case_ids'])

    def test_primary_inner_selection_and_fresh_refit_have_distinct_roles(self):
        _, _, _, inventory, panel = fixture()
        stages = design.build_rotation_design(inventory, panel)['stages']
        inner = [s for s in stages if s['stage'] == 'inner_selection']
        refits = [s for s in stages if s['stage'] == 'outer_refit']
        self.assertEqual((len(inner), len(refits)), (20, 5))
        for stage in inner:
            self.assertEqual(tuple(len(stage[k+'_case_ids']) for k in ('train','validation','final_test')), (6,2,2))
            self.assertFalse(stage['final_test_evaluation_permitted'])
        for stage in refits:
            self.assertEqual(tuple(len(stage[k+'_case_ids']) for k in ('train','validation','final_test')), (8,0,2))
            self.assertEqual(len(stage['parent_inner_stage_ids']), 4)
            self.assertTrue(stage['final_test_evaluation_permitted'])
            self.assertEqual(stage['locator_policy'], 'fresh_training_on_train_case_ids_only')

    def test_secondary_protocols_predict_identical_target_scenarios(self):
        _, _, _, inventory, panel = fixture()
        stages = design.build_rotation_design(inventory, panel)['stages']
        a = {s['target_case_id']: s for s in stages if s['protocol'] == 'same_age_other_sex'}
        b = {s['target_case_id']: s for s in stages if s['protocol'] == 'nearest_age_same_sex'}
        self.assertEqual(set(a), set(b))
        self.assertEqual(len(a), 10)
        for case in a:
            self.assertEqual(a[case]['final_test_sample_ids'], b[case]['final_test_sample_ids'])
            self.assertEqual(tuple(len(a[case][k+'_case_ids']) for k in ('train','validation','final_test')), (8,1,1))
            self.assertEqual(a[case]['roles']['validation'], 'model_selection_validation')

    def test_nearest_known_age_declares_younger_tie_and_avoids_reserved_infants(self):
        _, _, _, inventory, panel = fixture()
        stages = design.build_rotation_design(inventory, panel)['stages']
        nearest = {s['target_case_id']:s for s in stages if s['protocol']=='nearest_age_same_sex'}
        self.assertEqual(nearest['50']['validation_case_ids'], ['40']) # 5 versus 15 tie at age 10
        self.assertEqual(nearest['30']['validation_case_ids'], ['40']) # infant age unknown/reserved
        self.assertEqual(nearest['10']['validation_case_ids'], ['60'])
        self.assertEqual(nearest['10']['validation_age_difference_years'], 35)

    def test_pairs_are_distinct_companion_matched_balanced_and_span_positive_strength(self):
        _, _, _, _, panel = fixture()
        for case, ids in panel['sample_ids_by_case'].items():
            rows = [r for r in panel['samples'] if r['case_id'] == case]
            self.assertEqual(len(set(ids)), 6)
            self.assertEqual(sum(r['annotation']['coa_benchmark_positive'] for r in rows), 3)
            self.assertEqual(sorted(r['annotation'][design.SCORE] for r in rows if r['pair_role']=='positive'), [.21,.35,.55])
            self.assertEqual({r['annotation'][design.SCORE] for r in rows if r['pair_role']=='edited_negative'}, {.15})
        for pair in panel['pairs']:
            a, b = [next(r for r in panel['samples'] if r['sample_id']==sid) for sid in pair['sample_ids']]
            self.assertEqual(a['companion_parameters'], b['companion_parameters'])
            self.assertEqual(a['rotation_pair_id'], b['rotation_pair_id'])
            self.assertFalse(a['baseline']); self.assertFalse(b['baseline'])

    def test_shared_age_counterpart_training_is_declared_secondary_only(self):
        _, _, _, inventory, panel = fixture()
        stages = design.build_rotation_design(inventory, panel)['stages']
        target = next(s for s in stages if s['protocol']=='nearest_age_same_sex' and s['target_case_id']=='50')
        self.assertIn('51', target['train_case_ids'])
        self.assertIn('unverified_ancestry', target['endpoint'])
        bad = deepcopy(target); bad['protocol'] = 'family_holdout'
        with self.assertRaisesRegex(ValueError, 'split a provisional'):
            design.assert_isolation(bad, inventory, panel)

    def test_isolation_rejects_omitted_sibling_in_metadata_partition(self):
        _, _, _, inventory, panel = fixture()
        stage = design.build_rotation_design(inventory, panel)['stages'][0]
        changed = deepcopy(stage)
        changed['all_final_test_sample_ids'].pop()
        with self.assertRaisesRegex(ValueError, 'sibling partition'):
            design.assert_isolation(changed, inventory, panel)

    def test_isolation_rejects_case_role_lookup_tampering(self):
        _, _, _, inventory, panel = fixture()
        stage = deepcopy(design.build_rotation_design(inventory, panel)['stages'][0])
        stage['case_assignment'][stage['final_test_case_ids'][0]] = 'train'
        with self.assertRaisesRegex(ValueError, 'role lookup'):
            design.assert_isolation(stage, inventory, panel)

    def test_annotations_and_source_identity_fail_closed(self):
        _, rows, labels, inventory, _ = fixture()
        altered = deepcopy(labels)
        altered[1]['coa_benchmark_positive'] = True
        with self.assertRaisesRegex(ValueError, 'Invalid achieved-loss'):
            design.select_matched_panel(rows, altered, inventory)
        altered = deepcopy(rows)
        altered[1]['anatomy_family'] = 'unrelated'
        with self.assertRaisesRegex(ValueError, 'family mismatch'):
            design.select_matched_panel(altered, labels, inventory)

    def test_reserved_policy_rejects_release_and_design_is_deterministic(self):
        _, _, _, inventory, panel = fixture()
        with self.assertRaisesRegex(ValueError, 'entire infant pair'):
            design.build_rotation_design(inventory, panel, reserved_case_ids=['20'])
        expected = design.build_rotation_design(inventory, panel, seeds=[42,43])
        actual = design.build_rotation_design(list(reversed(inventory)), panel, seeds=[43,42])
        self.assertEqual(expected['stages'], actual['stages'])
        self.assertEqual(len(expected['stages']), 90)

    def test_config_is_clean_typed_and_rejects_unknown_fields(self):
        path = ROOT/'archive/research/configs/models/coa-rotating-evaluation.ini'
        c = design.load_config(path)
        self.assertEqual(c['evaluation']['seeds'], [42])
        self.assertEqual(c['training']['locator_epochs'], 120)
        self.assertEqual(c['training']['classifier_learning_rate'], .001)
        self.assertEqual(c['training']['require_gpu'], True)
        with tempfile.TemporaryDirectory() as tmp:
            changed = Path(tmp)/'input.ini'
            changed.write_text(path.read_text().replace('seeds = 42', 'unknown_input = 42'))
            with self.assertRaisesRegex(ValueError, 'Unsupported'):
                design.load_config(changed)

    @unittest.skipUnless((ROOT/'outputs/datasets/coa/context-aorta-r1/dataset-manifest.json').exists(),
                         'Production frozen cohort metadata is not installed')
    def test_production_metadata_freeze_and_known_reference_isolation(self):
        import fakect_coa_rotation_experiment as experiment
        path = ROOT/'archive/research/configs/models/coa-rotating-evaluation.ini'
        with tempfile.TemporaryDirectory() as tmp:
            output = Path(tmp)/'frozen-rotation'
            input_path = Path(tmp)/'rotation.ini'
            input_path.write_text(path.read_text().replace(
                'outputs/experiments/coa/rotating-evaluation-r1', str(output)))
            c = design.load_config(input_path)
            locked = experiment.freeze(c, input_path.read_bytes())
            self.assertEqual(len(locked['stages']), 45)
            self.assertEqual(sum(s['execution_eligible'] for s in locked['stages']), 31)
            self.assertTrue(all(s['execution_eligible'] for s in locked['stages'] if s['protocol']=='family_holdout'))
            self.assertEqual(experiment.inspect_lock(c, input_path.read_bytes())['samples'], locked['samples'])
            (output/'input.ini').write_text(input_path.read_text()+'\n# Changed frozen bytes\n')
            with self.assertRaisesRegex(ValueError, 'Frozen inputs changed'):
                experiment.inspect_lock(c, input_path.read_bytes())


if __name__ == '__main__':
    unittest.main()
