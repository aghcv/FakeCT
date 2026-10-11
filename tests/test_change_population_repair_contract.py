"""Metadata-only future repair contract against the immutable native population.

These checks never regenerate anatomy, open GPU models, or modify production
receipts. Stand-in metadata for failed recipes tests whether accepted parameter
adjustments and new revision paths can enter the existing guarded Model 1 plan.
It does not claim that these stand-ins are actual repaired voxel payloads.
"""
from collections import Counter
from copy import deepcopy
from pathlib import Path
import json
import hashlib
import sys
import tempfile
import unittest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import fakect_change_population_experiment as exp
from fakect_multiscale_experiment import write_json, sha256

REFERENCE = ROOT/'outputs/experiments/aorta/focused-change-r1'
NATIVE = ROOT/'outputs/cohorts/aorta/diversity-r1/cases'
CONFIG = ROOT/'configs/models/aorta-diverse-change.ini'


@unittest.skipUnless((REFERENCE/'experiment-lock.json').exists()
                     and len(list(NATIVE.glob('*/dataset-manifest.json'))) == 10,
                     'Completed native case metadata and immutable focused reference unavailable')
class PopulationRepairMetadataTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.temp = tempfile.TemporaryDirectory(prefix='fakect-population-repair-contract-')
        cls.addClassCleanup(cls.temp.cleanup)
        cls.directory = Path(cls.temp.name)
        cls.manifest = cls.directory/'diversity-r2/dataset-manifest.json'
        cls.manifest.parent.mkdir()
        cls.old = json.loads((REFERENCE/'experiment-lock.json').read_text())
        cls.rows, cls.repair_ids = [], []
        for path in sorted(NATIVE.glob('*/dataset-manifest.json')):
            # Raw reading deliberately preserves and accepts an INCOMPLETE
            # source receipt as evidence, not as a completed release manifest.
            case = json.loads(path.read_text())
            cls.rows.extend(deepcopy(case['samples']))
            template = next(row for row in case['samples'] if not row['baseline'])
            for failed in (row for row in case['records'] if row['status'] != 'ok'):
                stand_in = deepcopy(template)
                requested = deepcopy(failed['parameters'])
                accepted = deepcopy(requested)
                for values in accepted.values():
                    if 'distance_mm' in values:
                        values['distance_mm'] *= .5
                new_id = 'repair-'+hashlib.sha256(json.dumps(accepted, sort_keys=True).encode()).hexdigest()[:16]
                sample_id = failed['case_id']+':'+new_id
                stand_in.update(sample_id=sample_id, variant_id=new_id,
                                requested_sample_id=failed['sample_id'],
                                parameters=accepted, requested_parameters=requested,
                                population_group=failed['group'],
                                metadata_contract_only=True)
                cls.rows.append(stand_in)
                cls.repair_ids.append(sample_id)
        # Newly completed per-case receipts retain original source provenance;
        # the aggregate refers to their hashes without creating self references.
        for case in sorted({row['case_id'] for row in cls.rows}):
            case_rows = [row for row in cls.rows if row['case_id'] == case]
            receipt = cls.manifest.parent/'cases'/case/'dataset-manifest.json'
            receipt.parent.mkdir(parents=True)
            write_json(receipt, {'schema_version': 'fakect.sibling-case/1',
                                'complete': True, 'case_id': case, 'samples': case_rows,
                                'metadata_contract_only': True})
            for row in case_rows:
                if not row['baseline']:
                    row['prepared_manifest'] = str(receipt)
                    row['prepared_manifest_sha256'] = sha256(receipt)
        write_json(cls.manifest, {'schema_version': 'fakect.sibling-population/1',
                                 'complete': True, 'samples': cls.rows})
        content = CONFIG.read_text()
        lines = []
        for line in content.splitlines():
            if line.startswith('output_directory ='):
                line = 'output_directory = '+str(cls.directory/'diverse-change-r2')
            elif line.startswith('manifest ='):
                line = 'manifest = '+str(cls.manifest)
            lines.append(line)
        cls.content = ('\n'.join(lines)+'\n').encode()
        cls.config_path = cls.directory/'model-r2.ini'
        cls.config_path.write_bytes(cls.content)
        cls.config = exp.load_config(cls.config_path)

    def test_future_metadata_keeps_all_ids_groups_and_fixed_holdouts(self):
        document = exp.plan(self.config)
        self.assertEqual(len(document['stages']), 5)
        self.assertEqual(len(document['samples']), 2630)
        for case in sorted({row['case_id'] for row in self.rows}):
            edited = [row for row in self.rows if row['case_id'] == case and not row['baseline']]
            self.assertEqual(len(edited), 256)
            self.assertEqual(Counter(row['population_group'] for row in edited),
                             {'enlargement_only': 85, 'narrowing_only': 85, 'mixed': 86})
        old = {stage['id']: stage for stage in self.old['stages']}
        for stage in document['stages']:
            self.assertEqual(len(stage['train_sample_ids']), 2056)
            self.assertEqual(stage['final_test_sample_ids'], old[stage['id']]['final_test_sample_ids'])
            self.assertEqual(stage['training_schedule']['minimum_edited_visits'], 6)
            self.assertEqual(stage['training_schedule']['maximum_edited_visits'], 6)
            self.assertEqual(stage['training_schedule']['total_optimizer_updates'], 14336)
        self.assertFalse(document['reserved_test_evaluated'])
        self.assertFalse(document['classifier_started'])

    def test_new_revision_metadata_can_freeze_and_inspect_without_source_edits(self):
        document = exp.freeze(self.config, self.content)
        inspected = exp.inspect_lock(self.config, self.content)
        self.assertEqual(document, inspected)
        self.assertEqual(inspected['population_manifest'], str(self.manifest))
        # Native incomplete receipts remain preserved and are not misrepresented
        # as complete prepared inputs in the repaired aggregate.
        incomplete = {str(path.resolve()) for path in NATIVE.glob('*/dataset-manifest.json')
                      if (path.parent/'INCOMPLETE').exists()}
        self.assertFalse(incomplete & set(inspected['dependency_sha256']))

    def test_different_accepted_distances_preserve_partitions_and_original_source(self):
        other = deepcopy(self.rows)
        for row in other:
            if row['sample_id'] in self.repair_ids:
                for values in row['parameters'].values():
                    if 'distance_mm' in values:
                        values['distance_mm'] *= .5
        alternate = self.manifest.parent/'alternate-accepted-manifest.json'
        write_json(alternate, {'schema_version': 'fakect.sibling-population/1',
                              'complete': True, 'samples': other})
        c = deepcopy(self.config)
        c['population']['manifest'] = str(alternate)
        first, second = exp.plan(self.config), exp.plan(c)
        for a, b in zip(first['stages'], second['stages']):
            self.assertEqual(a['train_sample_ids'], b['train_sample_ids'])
            self.assertEqual(a['final_test_sample_ids'], b['final_test_sample_ids'])
            self.assertEqual(a['locator_checkpoint_sha256'], b['locator_checkpoint_sha256'])
        original = {row['sample_id']: row for row in first['samples']}
        for row in second['samples']:
            for field in exp.SOURCE_FIELDS:
                self.assertEqual(row[field], original[row['sample_id']][field])

    def test_repair_cannot_change_the_original_reference_source(self):
        rows = deepcopy(self.rows)
        next(row for row in rows if not row['baseline'])['original_reference_mask_sha256'] = '0'*64
        bad = self.manifest.parent/'bad-reference-manifest.json'
        write_json(bad, {'schema_version': 'fakect.sibling-population/1',
                         'complete': True, 'samples': rows})
        c = deepcopy(self.config)
        c['population']['manifest'] = str(bad)
        with self.assertRaisesRegex(ValueError, 'original source or reference mask'):
            exp.plan(c)


if __name__ == '__main__':
    unittest.main()
