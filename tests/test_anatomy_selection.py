"""Atlas eligibility, native contact, and immutable optional selector artifacts."""
from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
import test_training_data
from fakect_anatomy import (CORE, BRANCHES, PULMONARY, create_anatomy_selection,
                           select_anatomy_labels, verify_selection_sources)
from fakect_roi import prepare_crop, resolve_preview


def _record(key, name, hierarchy):
    return {'original_id': key, 'original_name': name,
            'classification': {'status': 'classified', 'tissue_id': 5,
                               'tissue_name': 'artery', 'hierarchy_path': hierarchy,
                               'canonical_name': name}}


class AnatomicalSelectionTests(unittest.TestCase):
    def fixture(self):
        labels = np.zeros((9, 9, 9), dtype=np.float32)
        labels[4, 4, 4] = 10
        labels[4, 4, 5] = 20
        labels[4, 5, 4] = 30
        labels[1, 1, 1] = 40
        labels[4, 6, 4] = 50  # Only pulmonary tissue connects this systemic label.
        records = [_record(10, 'aorta', CORE), _record(20, 'carotid', BRANCHES[0]),
                   _record(30, 'pulmonary', PULMONARY[0]),
                   _record(40, 'distant_artery', BRANCHES[0]),
                   _record(50, 'pulmonary_neighbor', BRANCHES[0])]
        settings = {'anatomy': 'thoracic_aorta', 'anatomical_labels': [],
                    'include_branches': True, 'connectivity': 6,
                    'protect_other_arteries': True}
        surfaces = {'records': [{'original_name': row['original_name'],
                                  'candidate_original_ids': [row['original_id']],
                                  'source_file': 'fixture.raw', 'id_join_status': 'unique_name_match'}
                                 for row in records]}
        return labels, np.ones(labels.shape, dtype=bool), {'records': records}, settings, surfaces

    def select(self, labels, roi, catalog, settings, surfaces):
        return select_anatomy_labels(labels, roi, catalog,
                                     tuple(row['original_id'] for row in catalog['records']),
                                     settings, surfaces)

    # Purpose: Verify systemic connectivity excludes paths through pulmonary tissue.
    # Inputs: 9-cube labeled aorta, carotid, pulmonary bridge and disconnected arteries.
    # Outputs: Only IDs 10 and 20 selected; disabling branches leaves ID 10.
    def test_connected_systemic_branches_retained_but_pulmonary_contact_never_carries_paths(self):
        values = self.fixture()
        result = self.select(*values)
        self.assertEqual(result['selected_source_ids'], [10, 20])
        rows = {row['original_id']: row for row in result['records']}
        self.assertEqual(rows[20]['decision'], 'included')
        self.assertIn('Pulmonary circulation', rows[30]['reason'])
        self.assertEqual(rows[40]['decision'], 'excluded')
        self.assertEqual(rows[50]['decision'], 'excluded')
        self.assertEqual(result['counts']['selected_roi_voxels'], 2)
        values[3]['include_branches'] = False
        self.assertEqual(self.select(*values)['selected_source_ids'], [10])

    # Purpose: Verify admitted IDs retain disconnected voxels with an explicit warning.
    # Inputs: Connected carotid ID 20 plus a second disconnected voxel of the same ID.
    # Outputs: Two connected voxels, three selected voxels and one warned retained voxel.
    def test_disconnected_same_id_is_explicit_whole_label_limitation(self):
        labels, roi, catalog, settings, surfaces = self.fixture()
        labels[1, 1, 2] = 20
        result = self.select(labels, roi, catalog, settings, surfaces)
        self.assertEqual(result['selected_source_ids'], [10, 20])
        self.assertEqual(result['counts']['connected_roi_voxels'], 2)
        self.assertEqual(result['counts']['selected_roi_voxels'], 3)
        self.assertEqual(result['counts']['unconnected_retained_roi_voxels'], 1)
        self.assertTrue(any('share admitted IDs' in warning for warning in result['warnings']))

    # Purpose: Reject missing ROI anchors and fractional ACT labels.
    # Inputs: Fixture ROI excluding aorta, then native label value 1.25.
    # Outputs: ValueError identifies missing anchor or nonintegral original IDs.
    def test_native_connection_requires_anchor_in_roi_and_integral_original_labels(self):
        labels, roi, catalog, settings, surfaces = self.fixture()
        roi[:] = False
        roi[4, 4, 5] = True
        with self.assertRaisesRegex(ValueError, 'anchor intersects this ROI'):
            self.select(labels, roi, catalog, settings, surfaces)
        roi[:] = True
        labels[0, 0, 0] = 1.25
        with self.assertRaisesRegex(ValueError, 'integral ACT IDs'):
            self.select(labels, roi, catalog, settings, surfaces)

    # Purpose: Reject an aortic anchor without exact RAW name evidence.
    # Inputs: Anatomy fixture with the aorta surface record removed.
    # Outputs: ValueError requires a RAW-supported aortic anchor.
    def test_missing_exact_raw_name_cannot_supply_an_aortic_anchor(self):
        labels, roi, catalog, settings, surfaces = self.fixture()
        surfaces['records'] = [row for row in surfaces['records'] if row['original_name'] != 'aorta']
        with self.assertRaisesRegex(ValueError, 'RAW-supported aortic anchor'):
            self.select(labels, roi, catalog, settings, surfaces)

    # Purpose: Verify duplicate anatomical names reference shared geometry only once.
    # Inputs: Three pulmonary IDs sharing two 500-node surface records.
    # Outputs: Each ID references both surfaces; geometry appears once per surface and selection stays 10, 20.
    def test_duplicate_names_reference_shared_surface_geometry_once(self):
        labels, roi, catalog, settings, surfaces = self.fixture()
        for key, position in ((31, (4, 5, 5)), (32, (4, 5, 6))):
            labels[position] = key
            catalog['records'].append(_record(key, 'pulmonary', PULMONARY[0]))
        first = surfaces['records'][2]
        first['candidate_original_ids'] = [30, 31, 32]
        first['centerline'] = {'status': 'verified_two_pole_rings',
                               'centerline_raw_mm': [[12345.125, float(n), 8.5] for n in range(500)]}
        second = deepcopy(first)
        second['centerline']['centerline_raw_mm'] = [[67890.875, float(n), 9.5] for n in range(500)]
        surfaces['records'].append(second)
        result = self.select(labels, roi, catalog, settings, surfaces)
        rows = {row['original_id']: row for row in result['records']}
        expected_refs = [2, len(surfaces['records']) - 1]
        for key in (30, 31, 32):
            self.assertEqual(rows[key]['surface_refs'], expected_refs)
            self.assertNotIn('surface_matches', rows[key])
            self.assertNotIn('centerline', rows[key])
        serialized = json.dumps(result, separators=(',', ':'))
        self.assertEqual(serialized.count('"centerline_raw_mm"'), 2)
        for surface in (first, second):
            coordinates = json.dumps(surface['centerline']['centerline_raw_mm'], separators=(',', ':'))
            self.assertEqual(serialized.count(coordinates), 1)
        self.assertEqual(result['selected_source_ids'], [10, 20])


class AnatomicalArtifactTests(unittest.TestCase):
    def setUp(self):
        fixture = test_training_data.TrainingDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.config = deepcopy(fixture.config)
        catalog_path = self.config['input']['catalog']
        catalog = json.loads(catalog_path.read_text())
        for row in catalog['records']:
            if row['classification']['tissue_name'] == 'artery':
                row['classification'].update(status='classified', canonical_name=row['original_name'],
                    hierarchy_path=CORE if row['original_name'] == 'aorta' else BRANCHES[0])
        catalog_path.write_text(json.dumps(catalog))
        self.path = self.root/'anatomy-r1'/'selection.json'
        self.config['selection'].update(source_ids=(), anatomy='thoracic_aorta', anatomy_file=self.path,
                                        anatomical_labels=(), include_branches=True, connectivity=6,
                                        protect_other_arteries=True)
        surfaces = {'files': [], 'coordinate_contract': {},
                    'records': [{'original_name': row['original_name'],
                                 'candidate_original_ids': [row['original_id']],
                                 'source_file': 'fixture.raw'} for row in catalog['records']
                                if row['classification']['tissue_name'] == 'artery']}
        with patch('fakect_anatomy_surfaces.inventory_surfaces', return_value=surfaces):
            self.audit = create_anatomy_selection(self.config, b'# frozen fixture input\n')

    # Purpose: Verify saved anatomy resolves without voxels while legacy selection remains available.
    # Inputs: Published aorta artifact and legacy tissue-only selection.
    # Outputs: Artifact selects -7 and protects 8; legacy selects both arteries without dataset publication.
    def test_automatic_artifact_loads_without_voxel_reads_and_keeps_default_legacy_selection(self):
        with patch('fakect_roi.read_crop', side_effect=AssertionError('metadata resolution read voxels')):
            resolved = resolve_preview(self.config)
        self.assertEqual(resolved['source_ids'], (-7,))
        self.assertEqual(resolved['anatomical_protected_ids'], (8,))
        self.assertEqual(resolved['anatomy_selection_bytes'], self.path.read_bytes())
        self.assertEqual(resolved['anatomy_selection_sha256'], hashlib.sha256(self.path.read_bytes()).hexdigest())
        arrays, _ = prepare_crop(resolved, self.config)
        np.testing.assert_array_equal(arrays['selected'], (arrays['act'] == -7) & arrays['roi'])
        legacy = deepcopy(self.config)
        legacy['selection'] = {'tissue': 'artery', 'source_ids': ()}
        old = resolve_preview(legacy)
        self.assertEqual(set(old['source_ids']), {-7, 8})
        self.assertNotIn('anatomical_selection', old)
        self.assertFalse((self.path.parent/'INCOMPLETE').exists())
        self.assertFalse(self.config['train']['dataset_directory'].exists())

    # Purpose: Verify display edits reuse selection but ROI edits invalidate it.
    # Inputs: Saved anatomy with changed opacity and slice, then radius 4.5 mm.
    # Outputs: Display configuration resolves; changed ROI raises mismatch ValueError.
    def test_display_changes_reuse_artifact_but_roi_changes_require_new_resolution(self):
        changed = deepcopy(self.config)
        changed['preview'].update(volume_opacity=.6, overlay_opacity=.3, slice_ijk=(14, 15, 15))
        self.assertEqual(resolve_preview(changed)['source_ids'], (-7,))
        changed['roi']['radius_mm'] = 4.5
        with self.assertRaisesRegex(ValueError, 'no longer matches ROI'):
            resolve_preview(changed)

    # Purpose: Reject anatomy resolution after native source metadata changes.
    # Inputs: ACT source mtime advanced by one millisecond.
    # Outputs: ValueError reports the saved artifact no longer matches the ROI.
    def test_modified_native_source_stat_invalidates_resolution(self):
        source = self.root/'001'/'001_act_1.bin'
        before = source.stat()
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns + 1_000_000))
        with self.assertRaisesRegex(ValueError, 'no longer matches ROI'):
            resolve_preview(self.config)

    # Purpose: Verify selection bytes are rechecked after metadata resolution.
    # Inputs: Resolved artifact with a newline appended afterward.
    # Outputs: Source verification rejects changed selection; new resolution rejects checksum mismatch.
    def test_saved_selection_bytes_are_rechecked_after_metadata_resolution(self):
        resolved = resolve_preview(self.config)
        original = self.path.read_bytes()
        self.path.write_bytes(original+b'\n')
        with self.assertRaisesRegex(ValueError, 'selection changed'):
            verify_selection_sources(resolved)
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            resolve_preview(self.config)

    # Purpose: Reject changed crop content despite unchanged file timestamps.
    # Inputs: ACT voxel changed to label 20 with original stat restored.
    # Outputs: Metadata resolves but crop preparation rejects native content mismatch.
    def test_crop_hash_rejects_changed_native_content_even_with_preserved_stat(self):
        source = self.root/'001'/'001_act_1.bin'
        before = source.stat()
        data = np.fromfile(source, dtype='<f4').reshape((31, 31, 31))
        data[15, 15, 15] = 20
        data.tofile(source)
        os.utime(source, ns=(before.st_atime_ns, before.st_mtime_ns))
        resolved = resolve_preview(self.config)
        with self.assertRaisesRegex(ValueError, 'Native source crop differs'):
            prepare_crop(resolved, self.config)

    # Purpose: Prevent overwriting an existing anatomy revision.
    # Inputs: Previously published anatomy_file and repeat publication request.
    # Outputs: FileExistsError requires a fresh revision and existing bytes remain identical.
    def test_artifact_publication_refuses_to_overwrite_existing_revision(self):
        before = self.path.read_bytes()
        with self.assertRaisesRegex(FileExistsError, 'fresh anatomy_file revision'):
            create_anatomy_selection(self.config)
        self.assertEqual(self.path.read_bytes(), before)

    # Purpose: Preserve a competing job's anatomy revision during publication races.
    # Inputs: Mock surface inventory creates another job's marker, INI and selection files.
    # Outputs: Publication raises FileExistsError and every competitor file remains unchanged.
    def test_competing_publication_cannot_change_another_jobs_files(self):
        config = deepcopy(self.config)
        path = self.root/'competing-revision'/'selection.json'
        config['selection']['anatomy_file'] = path
        competitor = {'INCOMPLETE': b'other job owns this revision\n',
                      'input.ini': b'# other job input\n',
                      'selection.json': b'{"owner": "other job"}\n'}

        def competing_inventory(*args):
            path.parent.mkdir()
            for name, content in competitor.items():
                (path.parent/name).write_bytes(content)
            return deepcopy(self.audit['surface_evidence'])

        with patch('fakect_anatomy_surfaces.inventory_surfaces', side_effect=competing_inventory):
            with self.assertRaises(FileExistsError):
                create_anatomy_selection(config, b'# losing job input\n')
        self.assertEqual({item.name: item.read_bytes() for item in path.parent.iterdir()}, competitor)


if __name__ == '__main__':
    unittest.main()
