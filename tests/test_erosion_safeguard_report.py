"""Report accepted erosion distances without hiding rejected geometric trials."""
import copy
from html.parser import HTMLParser
from pathlib import Path
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
from fakect_preview_report import write_preview_report


def safeguard():
    return {'enabled': True, 'status': 'reduced', 'requested_distance_mm': 4,
            'accepted_distance_mm': 2, 'min_volume_ratio': .6, 'retained_volume_ratio': .75,
            'baseline_target_voxels': 200, 'retained_target_voxels': 150,
            'preserve_connectivity': True, 'connectivity_ok': True, 'affected_components': 1,
            'scope_semantics': 'Target in the configured local erosion region.',
            'reference_semantics': 'The target before iteration 1 of this edit step.',
            'attempts': [
                {'distance_mm': 4, 'retained_volume_ratio': .4, 'retained_target_voxels': 80,
                 'connectivity_ok': False, 'accepted': False,
                 'reasons': ['Local volume below configured minimum', 'Target component split']},
                {'distance_mm': 2, 'retained_volume_ratio': .75, 'retained_target_voxels': 150,
                 'connectivity_ok': True, 'accepted': True, 'reasons': []}]}


class ErosionSafeguardReportTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.report = {'config': {'study': {'name': 'Erosion safeguard fixture'},
                                  'selection': {'tissue': 'artery', 'source_ids': []},
                                  'edit': {'operation': 'erosion', 'distance_mm': 4}},
                       'edit': {'operation': 'erosion', 'distance_mm': 2,
                                'counts': {'before': 200, 'after': 150, 'removed': 50},
                                'erosion_safeguard': safeguard()}}
        self.number = 0

    def write(self, report=None):
        self.number += 1
        output = Path(self.temp.name)/str(self.number)
        output.mkdir()
        write_preview_report(output, self.report if report is None else report,
                             '[edit]\noperation = erosion\ndistance_mm = 4\n')
        return (output/'report.html').read_text()

    def test_reduced_single_edit_distances_ratios_and_rejected_trials_are_visible(self):
        document = self.write()
        self.assertIn('<td>Requested erosion distance (mm)</td><td>4</td>', document)
        self.assertIn('<td>Accepted erosion distance (mm)</td><td>2</td>', document)
        self.assertIn('<td>Local volume retained</td><td>75%</td>', document)
        self.assertIn('<td>Minimum local retention</td><td>60%</td>', document)
        self.assertIn('<td>Target voxels in the fixed baseline</td><td>200</td>', document)
        self.assertIn('A smaller erosion distance passed', document)
        self.assertIn('<td>1</td><td>4</td><td>40%</td><td>80</td><td>Failed</td><td>Rejected</td>', document)
        self.assertIn('Local volume below configured minimum, Target component split', document)
        self.assertIn('<td>2</td><td>2</td><td>75%</td><td>150</td><td>Passed</td><td>Accepted</td><td>Checks passed</td>', document)
        self.assertIn('remains constant across its iterations and retry attempts', document)
        self.assertIn('rejected trials do not accumulate changes', document)
        self.assertIn('does not bound minimum cross-sectional area', document)

    def test_skipped_pass_reports_zero_accepted_distance_and_kept_input(self):
        guard = self.report['edit']['erosion_safeguard']
        guard.update(status='skipped', accepted_distance_mm=0, retained_volume_ratio=1,
                     retained_target_voxels=200)
        guard['attempts'] = guard['attempts'][:1]
        document = self.write()
        self.assertIn('<td>Accepted erosion distance (mm)</td><td>0</td>', document)
        self.assertIn('The original input state of this pass was kept', document)
        self.assertIn('earlier accepted passes remain in place', document)
        self.assertNotIn('Checks passed</td>', document)

    def test_physical_volumes_reason_codes_and_fragment_counts_are_readable(self):
        guard = self.report['edit']['erosion_safeguard']
        guard.update(baseline_volume_mm3=1600, retained_volume_mm3=1200)
        guard['attempts'][0]['reasons'] = ['local_volume_below_floor', 'affected_component_split_or_lost',
                                          '<unknown&reason>']
        guard['attempts'][0]['component_details'] = [{'component_id': 2, 'surviving_components': 12}]
        guard['attempts'][1]['component_details'] = [{'component_id': 2, 'surviving_components': 1}]
        document = self.write()
        self.assertIn('<td>Local target volume in the fixed baseline (mm³)</td><td>1,600</td>', document)
        self.assertIn('<td>Retained local target volume (mm³)</td><td>1,200</td>', document)
        self.assertIn('Local volume below configured minimum, An affected target component split or disappeared', document)
        self.assertIn('&lt;unknown&amp;reason&gt;', document)
        self.assertNotIn('<unknown&reason>', document)
        self.assertIn('<th scope="col">Baseline → surviving components</th>', document)
        self.assertIn('<td>Component 2: 1 → 12</td>', document)
        self.assertIn('<td>Component 2: 1 → 1</td>', document)

    def test_accepted_edit_with_connectivity_disabled_does_not_claim_a_passed_check(self):
        guard = self.report['edit']['erosion_safeguard']
        guard.update(status='accepted', accepted_distance_mm=4,
                     preserve_connectivity=False, connectivity_ok=None)
        guard['attempts'] = []
        document = self.write()
        self.assertIn('The requested distance passed the configured erosion checks', document)
        self.assertIn('<td>Connectivity check</td><td>Not requested</td>', document)
        self.assertNotIn('Erosion attempts and retry reasons', document)

    def test_recipe_table_exposes_reduction_and_skip_across_fixed_baseline_iterations(self):
        first = copy.deepcopy(self.report['edit'])
        second = copy.deepcopy(first)
        second['erosion_safeguard'].update(status='skipped', accepted_distance_mm=0,
                                          retained_target_voxels=150)
        second['counts'] = {'before': 150, 'after': 150, 'removed': 0}
        self.report.pop('edit')
        self.report['config'].pop('edit')
        self.report['config']['recipe'] = {'steps': ['grow', 'narrow']}
        self.report['recipe'] = {'steps': [
            {'step_name': 'grow', 'roi_name': 'main', 'iteration': 1,
             'summary': {'operation': 'dilation', 'counts': {}}},
            {'step_name': 'narrow', 'roi_name': 'main', 'iteration': 1, 'summary': first},
            {'step_name': 'narrow', 'roi_name': 'main', 'iteration': 2, 'summary': second}]}
        document = self.write()
        self.assertIn('<th scope="col">Requested distance (mm)</th>', document)
        self.assertIn('<th scope="col">Accepted distance (mm)</th>', document)
        self.assertIn('<td>Not enabled</td><td>—</td><td>—</td>', document)
        self.assertIn('<td>reduced</td><td>4</td><td>2</td>', document)
        self.assertIn('<td>skipped</td><td>4</td><td>0</td>', document)
        self.assertEqual(document.count('<div class="erosion-safeguard">'), 2)
        self.assertEqual(document.count('<td>Target voxels in the fixed baseline</td><td>200</td>'), 2)

    def test_optional_metadata_preserves_legacy_single_and_recipe_sections(self):
        for metadata in (None, {'enabled': False, 'status': 'skipped'}):
            with self.subTest(metadata=metadata):
                report = copy.deepcopy(self.report)
                if metadata is None:
                    report['edit'].pop('erosion_safeguard')
                else:
                    report['edit']['erosion_safeguard'] = metadata
                document = self.write(report)
                self.assertNotIn('class="erosion-safeguard"', document)
                report['recipe'] = {'steps': [{'step_name': 'narrow', 'summary': report.pop('edit')}]}
                document = self.write(report)
                self.assertNotIn('<th scope="col">Erosion safeguard</th>', document)
                self.assertNotIn('class="erosion-safeguard"', document)

    def test_metadata_and_attempt_reasons_are_escaped(self):
        payload = '<script>window.BAD=true</script><img src=x onerror=bad()>'
        guard = self.report['edit']['erosion_safeguard']
        for key in ('scope_semantics', 'reference_semantics', 'status'):
            guard[key] = payload
        guard['attempts'][0]['reasons'] = [payload]
        document = self.write()
        self.assertNotIn(payload, document)
        self.assertIn('&lt;script&gt;window.BAD=true&lt;/script&gt;', document)

        class Scripts(HTMLParser):
            count = 0

            def handle_starttag(self, tag, attrs):
                if tag == 'script':
                    self.count += 1

        parsed = Scripts()
        parsed.feed(document)
        self.assertEqual(parsed.count, 1)


if __name__ == '__main__':
    unittest.main()
