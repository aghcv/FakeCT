"""Behavioral checks for trust boundaries and reversible XCAT grouping."""
import csv
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_tissues import AtlasClassifier, build_catalog, coarse_labels, read_organ_table, subtype_mask


class TissueTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp.cleanup)
        self.directory = Path(self.temp.name)
        self.policy = ROOT / 'configs/tissues/tissue-policy.v1.json'
        self.header = ['original_name', 'original_block_index', 'source_file',
                       'canonical_name', 'normalized_name', 'system', 'subsystem',
                       'structure_type', 'hierarchy_path', 'laterality',
                       'review_action', 'review_reason', 'example_source_file']
        self.hierarchy = '''aliases:
  lkidney:
    canonical_name: kidney
    normalized_name: kidney
    laterality: left
rules:
  - id: kidney
    priority: 90
    pattern: ".*kidney.*"
    structure_type: organ
    system: urinary
    hierarchy_path: 05_Urinary/Kidneys
  - id: muscle
    priority: 80
    pattern: ".*muscle.*"
    structure_type: muscle
    system: musculoskeletal
    hierarchy_path: 09_Musculoskeletal/Skeletal_Muscle
  - id: heart
    priority: 90
    pattern: ".*(heart|myocardium|papillary_muscle).*"
    structure_type: heart
    system: cardiovascular
    hierarchy_path: 01_Cardiovascular/Heart
  - id: skull
    priority: 90
    pattern: ".*skull.*"
    structure_type: bone
    system: musculoskeletal
    hierarchy_path: 09_Musculoskeletal/Skeleton/Skull
'''

    def classifier(self, rows=(), hierarchy=None, extra=None):
        atlas = self.directory / 'atlas.csv'
        with atlas.open('w', newline='') as stream:
            writer = csv.writer(stream)
            writer.writerow(self.header)
            for row in rows:
                writer.writerow([row.get(field, '') for field in self.header])
            if extra:
                writer.writerow([extra.get(field, '') for field in self.header] + ['unexpected'])
        hierarchy_path = self.directory / 'hierarchy.yml'
        hierarchy_path.write_text(self.hierarchy if hierarchy is None else hierarchy)
        return AtlasClassifier(atlas, hierarchy_path, self.policy)

    def row(self, name='kidney', **updates):
        return dict(original_name=name, canonical_name=name, structure_type='artery',
                    system='cardiovascular', hierarchy_path='01_Cardiovascular/Arteries',
                    review_action='corrected', **updates)

    def test_active_override_preserves_full_metadata(self):
        row = self.row(review_reason='Reviewed branch anatomy', example_source_file='example.raw')
        result = self.classifier([row]).classify('kidney')
        self.assertEqual(result['tissue_name'], 'artery')
        self.assertEqual(result['atlas_rows'][0]['review_reason'], 'Reviewed branch anatomy')
        self.assertTrue(any(item['kind'] == 'atlas_override' for item in result['evidence']))

    def test_pending_and_unknown_actions_do_not_override(self):
        for action in ['needs_review', 'pending', 'arbitrary']:
            row = self.row()
            row['review_action'] = action
            result = self.classifier([row]).classify('kidney')
            self.assertEqual(result['tissue_name'], 'soft_tissue')
            self.assertFalse(any(item['kind'] == 'atlas_override' for item in result['evidence']))

    def test_native_blank_action_and_malformed_quarantine(self):
        row = self.row()
        row['review_action'] = ''
        self.assertEqual(self.classifier([row]).classify('kidney')['tissue_name'], 'artery')
        classifier = self.classifier(extra=row)
        self.assertEqual(classifier.classify('kidney')['tissue_name'], 'soft_tissue')
        self.assertEqual(classifier.warnings[0]['kind'], 'atlas_row_width')
        self.assertEqual(classifier.atlas_rows[0]['_extra_columns'], ['unexpected'])

    def test_scoped_consensus_cannot_claim_resolved_hierarchy(self):
        a = self.row(name='vessel', original_block_index='41', source_file='/a.raw', laterality='left')
        b = self.row(name='vessel', original_block_index='42', source_file='/a.raw', laterality='right')
        c = self.classifier([a, b])
        result = c.classify('vessel')
        self.assertEqual((result['status'], result['tissue_name']), ('tissue_consensus', 'artery'))
        self.assertEqual(result['hierarchy_path'], '99_Unclassified')
        self.assertEqual(len(result['atlas_rows']), 2)
        exact = c.classify('vessel', source_file='/a.raw', block_index=41)
        self.assertEqual((exact['status'], exact['laterality']), ('classified', 'left'))
        self.assertEqual(c.classify('vessel', source_file='/other/a.raw', block_index=41)['status'], 'tissue_consensus')

    def test_scoped_disagreement_requires_source_identity(self):
        a = self.row(name='vessel', original_block_index='41', source_file='/a.raw')
        b = self.row(name='vessel', original_block_index='41', source_file='/b.raw')
        b['structure_type'] = 'vein'
        result = self.classifier([a, b]).classify('vessel', source_file='/a.raw')
        self.assertEqual((result['status'], result['tissue_id']), ('scope_required', 255))

    def test_example_fields_do_not_create_scope(self):
        row = self.row(example_source_file='/original/example.raw')
        result = self.classifier([row]).classify('kidney', source_file='/elsewhere.raw')
        self.assertEqual(result['tissue_name'], 'artery')

    def test_conflicting_equal_overrides_fail_closed(self):
        a, b = self.row(), self.row()
        b['structure_type'] = 'vein'
        result = self.classifier([a, b]).classify('kidney')
        self.assertEqual((result['status'], result['tissue_id']), ('ambiguous_atlas', 255))

    def test_conflicting_top_hierarchy_rules_fail_closed(self):
        hierarchy = self.hierarchy + '''  - id: kidney_artery
    priority: 90
    pattern: ".*kidney.*"
    structure_type: artery
    system: cardiovascular
    hierarchy_path: 01_Cardiovascular/Arteries
'''
        result = self.classifier(hierarchy=hierarchy).classify('kidney')
        self.assertEqual((result['status'], result['tissue_id']), ('ambiguous_hierarchy', 255))
        self.assertEqual(len(result['matched_rules']), 2)
        # An explicit reviewed override resolves a rule collision.
        self.assertEqual(self.classifier([self.row()], hierarchy=hierarchy).classify('kidney')['tissue_id'], 5)

    def test_normalization_keeps_phase_side_piece_and_alias(self):
        result = self.classifier().classify('dias_lkidney_002')
        self.assertEqual(result['canonical_name'], 'kidney')
        self.assertEqual(result['laterality'], 'left')
        self.assertEqual(result['piece_number'], '002')
        self.assertEqual(result['temporal_phase'], 'diastole')
        self.assertEqual(result['tissue_name'], 'soft_tissue')

    def test_cardiac_material_ambiguity(self):
        classifier = self.classifier()
        self.assertEqual(classifier.classify('heart')['tissue_id'], 255)
        self.assertEqual(classifier.classify('left_myocardium')['tissue_name'], 'muscle')
        self.assertEqual(classifier.classify('papillary_muscle')['tissue_name'], 'muscle')
        self.assertIsNone(classifier.classify('kidney')['material_type'])

    def test_bone_inner_is_not_assumed_bone_material(self):
        result = self.classifier().classify('skull_inner')
        self.assertEqual((result['status'], result['tissue_id']), ('material_review_required', 255))
        self.assertEqual(result['structure_type'], 'bone')
        self.assertIn('Skull', result['hierarchy_path'])

    def test_signed_dictionary_and_duplicate_names_are_preserved(self):
        path = self.directory / 'organ_ids.txt'
        path.write_text('kidney = 8\nkidney = -8\nother = 9\n')
        table = read_organ_table(path)
        self.assertEqual(table, {8: 'kidney', -8: 'kidney', 9: 'other'})
        catalog = build_catalog(self.classifier(), table)
        self.assertEqual([r['original_id'] for r in catalog['records']], [-8, 0, 8, 9])
        self.assertEqual(len(catalog['sources']['atlas']['sha256']), 64)
        path.write_text('kidney = 8\nother = 8\n')
        with self.assertRaisesRegex(ValueError, 'conflicting names'):
            read_organ_table(path)

    def test_coarsening_and_subdivision_keep_signed_source_exactly(self):
        catalog = build_catalog(self.classifier(), {-8: 'kidney', 8: 'muscle', 9: 'kidney'})
        original = np.array([[-8, 8], [9, 0]], dtype=np.float32)
        saved = original.copy()
        coarse = coarse_labels(original, catalog, strict=True)
        np.testing.assert_array_equal(coarse, [[1, 4], [1, 0]])
        self.assertEqual(coarse.dtype, np.uint8)
        np.testing.assert_array_equal(original, saved)
        self.assertFalse(np.shares_memory(original, coarse))
        selected = subtype_mask(original, catalog, 1, [-8])
        np.testing.assert_array_equal(selected, [[True, False], [False, False]])
        with self.assertRaisesRegex(ValueError, 'do not belong'):
            subtype_mask(original, catalog, 1, [8])

    def test_absent_and_unresolved_are_unknown(self):
        catalog = build_catalog(self.classifier(), {4: 'unfamiliar'})
        np.testing.assert_array_equal(coarse_labels(np.array([4, 17]), catalog), [255, 255])
        with self.assertRaisesRegex(ValueError, 'Unresolved original IDs'):
            coarse_labels(np.array([4, 17]), catalog, strict=True)

    def test_invalid_labels_are_rejected(self):
        catalog = build_catalog(self.classifier(), {1: 'kidney'})
        invalid = [np.array([np.nan]), np.array([np.inf]), np.array([1.5]),
                   np.array([2**31], dtype=np.float32), np.array([2**31], dtype=np.int64), np.array([2**32], dtype=np.uint64),
                   np.array([True]), np.array([1j])]
        for values in invalid:
            with self.subTest(values=values), self.assertRaises(ValueError):
                coarse_labels(values, catalog)
        self.assertEqual(coarse_labels(np.empty((0, 3)), catalog).shape, (0, 3))
        with self.assertRaises(ValueError):
            build_catalog(self.classifier(), {2**31: 'kidney'})

    def test_int32_lower_bound_is_valid(self):
        catalog = build_catalog(self.classifier(), {-(2**31): 'kidney'})
        np.testing.assert_array_equal(coarse_labels(np.array([-(2**31)], dtype=np.float32), catalog), [1])

    def test_edited_catalog_cannot_wrap_uint8_codes(self):
        base = build_catalog(self.classifier(), {1: 'kidney'})
        mutations = [lambda c: c['categories'].append(c['categories'][0]),
                     lambda c: c['categories'][1].update(id=257),
                     lambda c: c['records'][1]['classification'].update(tissue_id=257),
                     lambda c: c['records'][1]['classification'].update(tissue_id=1.5),
                     lambda c: c['records'][1]['classification'].update(tissue_name='bone'),
                     lambda c: c['records'].append(c['records'][0]),
                     lambda c: c['categories'].pop()]
        for mutate in mutations:
            catalog = json.loads(json.dumps(base))
            mutate(catalog)
            with self.assertRaises(ValueError):
                coarse_labels(np.array([1]), catalog)

    def test_numeric_muscle_pattern_collision_requires_review(self):
        # The source YAML's c1 vertebra pattern also matches musc104.
        hierarchy = self.hierarchy + '  - id: vertebra\n    priority: 95\n    pattern: ".*c1.*"\n    structure_type: bone\n    system: musculoskeletal\n    hierarchy_path: 09_Musculoskeletal/Skeleton\n'
        result = self.classifier(hierarchy=hierarchy).classify('musc104')
        self.assertEqual((result['status'], result['tissue_id']), ('hierarchy_review_required', 255))
        row = self.row(name='musc104')
        row['structure_type'] = 'muscle'
        self.assertEqual(self.classifier([row], hierarchy=hierarchy).classify('musc104')['tissue_name'], 'muscle')

    def test_bladder_coronary_substring_collision_requires_review(self):
        hierarchy = self.hierarchy + '  - id: coronary_artery\n    priority: 95\n    pattern: ".*(coronary|lad).*"\n    structure_type: artery\n    system: cardiovascular\n    hierarchy_path: 01_Cardiovascular/Coronary_Circulation/Coronary_Arteries\n'
        classifier = self.classifier(hierarchy=hierarchy)
        for name in ['bladder_inner', 'bladder_outer', 'urinary_bladder']:
            result = classifier.classify(name)
            self.assertEqual((result['status'], result['tissue_id']), ('anatomy_review_required', 255))
            self.assertEqual(result['structure_type'], 'artery')
            self.assertIn('Coronary', result['hierarchy_path'])
            self.assertTrue(any(e.get('rule') == 'bladder_coronary_substring_collision' for e in result['evidence']))
        # A genuine artery name is not the organ-boundary collision.
        self.assertEqual(classifier.classify('bladder_artery')['tissue_id'], 5)
        # A reviewed organ override resolves the false source-hierarchy result.
        row = self.row(name='bladder_inner')
        row.update(structure_type='organ', system='urinary', hierarchy_path='05_Urinary/Urinary_Bladder')
        self.assertEqual(self.classifier([row], hierarchy=hierarchy).classify('bladder_inner')['tissue_name'], 'soft_tissue')

    def test_catalog_serializes_without_losing_provenance(self):
        catalog = build_catalog(self.classifier([self.row()]), {3: 'kidney'})
        copy = json.loads(json.dumps(catalog))
        self.assertEqual(copy['records'][1]['classification']['atlas_rows'][0]['original_name'], 'kidney')
        self.assertEqual(copy['taxonomy_status'], 'proposed')
        self.assertIn('original per-voxel', copy['reversibility'])


if __name__ == '__main__':
    unittest.main()
