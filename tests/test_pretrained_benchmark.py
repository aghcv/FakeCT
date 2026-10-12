"""Panel selection protects original/sibling provenance before model inference."""
import copy
import importlib.util
from pathlib import Path
import unittest

path = Path(__file__).resolve().parents[1] / 'scripts/research/benchmark_pretrained_aorta.py'
spec = importlib.util.spec_from_file_location('pretrained_benchmark', path)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)


class PanelTests(unittest.TestCase):
    def panel(self):
        rows = [dict(case_id='case', sample_id='case:baseline', baseline=True,
                     mask_counts=dict(added=0, removed=0))]
        for i in range(1, 9):
            rows.append(dict(case_id='case', sample_id='case:n' + str(i), baseline=False,
                             mask_counts=dict(added=0, removed=i * 10)))
        rows += [dict(case_id='case', sample_id='case:enlarged', baseline=False,
                      mask_counts=dict(added=100, removed=0)),
                 dict(case_id='case', sample_id='case:mixed', baseline=False,
                      mask_counts=dict(added=2000, removed=2000))]
        return dict(complete=True, samples=rows)

    # Purpose: Verify benchmark panels select achieved pure edits deterministically.
    # Inputs: Baseline, eight narrowing siblings, one enlargement, and one mixed edit.
    # Outputs: Baseline/n2/n8/enlarged panel with unique IDs, stable ordering, and unchanged input.
    def test_original_and_achieved_pure_edits_selected_before_inference(self):
        source = self.panel()
        original = copy.deepcopy(source)
        result = module.select_panel(source, ['case'])
        self.assertEqual([r['sample_id'] for r in result],
                         ['case:baseline', 'case:n2', 'case:n8', 'case:enlarged'])
        self.assertEqual(len({r['sample_id'] for r in result}), 4)
        self.assertEqual(source, original)
        source['samples'].reverse()
        self.assertEqual([r['sample_id'] for r in module.select_panel(source, ['case'])],
                         [r['sample_id'] for r in result])

    # Purpose: Verify benchmark development selection excludes reserved infant anatomy.
    # Inputs: Panel requested for case 260612.
    # Outputs: ValueError identifying reserved infant data.
    def test_reserved_infant_cannot_enter_development_panel(self):
        with self.assertRaisesRegex(ValueError, 'reserved infant'):
            module.select_panel(self.panel(), ['260612'])

    # Purpose: Verify benchmark panels require a complete frozen population.
    # Inputs: Fixture population with complete flag false.
    # Outputs: ValueError requiring complete frozen data.
    def test_incomplete_population_cannot_enter(self):
        source = self.panel(); source['complete'] = False
        with self.assertRaisesRegex(ValueError, 'complete frozen'):
            module.select_panel(source, ['case'])

    # Purpose: Verify panel selection requires the prescribed pure-edit classes.
    # Inputs: Population missing enlargement but retaining a large mixed edit.
    # Outputs: ValueError for insufficient pure-edit scenarios.
    def test_missing_pure_edit_does_not_substitute_mixed_edit(self):
        source = self.panel()
        source['samples'] = [r for r in source['samples'] if r['sample_id'] != 'case:enlarged']
        with self.assertRaisesRegex(ValueError, 'Insufficient'):
            module.select_panel(source, ['case'])


if __name__ == '__main__': unittest.main()
