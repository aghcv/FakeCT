import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_phantom_anatomy import load_config, landmark_domain, select_masks, _surfaces


def catalog():
    names = {1: ('aorta', '01_Cardiovascular/Heart/Aorta'),
             2: ('arteries', '01_Cardiovascular/Systemic_Arteries'),
             3: ('pulmonary', '01_Cardiovascular/Pulmonary_Circulation'),
             4: ('gall_bladder_inner', '01_Cardiovascular/Coronary_Circulation/Coronary_Arteries')}
    return {'records': [{'original_id': i, 'original_name': name,
                         'classification': {'tissue_name': 'artery', 'status': 'classified',
                                            'hierarchy_path': hierarchy}}
                        for i, (name, hierarchy) in names.items()]}


def select(labels, **kwargs):
    options = dict(z_slab_mm=[2, 7], origin_raw_mm=[0, 0, 0], spacing_ijk_mm=[1, 1, 1],
                   branch_reach_mm=5, connectivity=6)
    options.update(kwargs)
    return select_masks(labels, catalog(), **options)


class PhantomAnatomyTests(unittest.TestCase):
    def test_no_roi_needed_and_unknown_roi_rejected(self):
        template = Path(__file__).resolve().parents[1] / 'archive/research/configs/anatomy/thoracic-aorta-260602.ini'
        config = load_config(template)
        self.assertNotIn('roi', config)
        self.assertEqual(config['selection']['branch_reach_radii'], 3)
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'bad.ini'
            p.write_text(template.read_text() + '\n[roi]\ncenter_ijk=1,2,3\n')
            with self.assertRaisesRegex(ValueError, 'no supplied ROI'):
                load_config(p)

    def test_pulmonary_excluded_before_connectivity(self):
        labels = np.zeros((10, 10, 10), np.int32)
        labels[1:9, 5, 5] = 1
        labels[4, 5, 6] = 3
        labels[4, 5, 7] = 2
        arrays, counts = select(labels)
        self.assertEqual(arrays['mask'].sum(), 6)
        self.assertFalse(arrays['mask'][4, 5, 6])
        self.assertFalse(arrays['mask'][4, 5, 7])
        self.assertEqual(counts['excluded_arterial_voxels_in_slab'], 1)

    def test_connected_component_not_whole_original_id(self):
        labels = np.zeros((10, 10, 10), np.int32)
        labels[1:9, 5, 5] = 1
        labels[4, 5, 6:8] = 2
        labels[4, 2, 5] = 2
        labels[4, 1, 5] = 1
        arrays, counts = select(labels)
        self.assertEqual(arrays['branch_mask'].sum(), 2)
        self.assertFalse(arrays['mask'][4, 2, 5])
        self.assertFalse(arrays['mask'][4, 1, 5])
        self.assertEqual(counts['disconnected_aorta_voxels'], 1)
        self.assertEqual(counts['disconnected_branch_voxels_in_reach'], 1)

    def test_slab_excludes_abdominal_same_aorta_id(self):
        labels = np.zeros((10, 5, 5), np.int32)
        labels[:, 2, 2] = 1
        arrays, counts = select(labels)
        np.testing.assert_array_equal(np.flatnonzero(arrays['mask'][:, 2, 2]), np.arange(2, 8))
        self.assertEqual(counts['aortic_voxels_outside_slab_in_crop'], 4)

    def test_anisotropic_physical_distance_and_voxel_center_origin(self):
        labels = np.zeros((5, 8, 8), np.int32)
        labels[:, 3, 3] = 1
        labels[2, 3, 4:7] = 2
        labels[2, 4:7, 3] = 2
        arrays, counts = select(labels, z_slab_mm=[104, 108], origin_raw_mm=[10, 20, 100],
                                spacing_ijk_mm=[2, 1, 2], branch_reach_mm=3)
        self.assertTrue(arrays['mask'][2, 3, 4])
        self.assertFalse(arrays['mask'][2, 3, 5])
        self.assertTrue(arrays['mask'][2, 6, 3])
        self.assertFalse(arrays['mask'][1, 3, 3])
        self.assertEqual(counts['branch_voxels'], 4)

    def test_optional_branches_and_lexical_misclassification_guard(self):
        labels = np.zeros((10, 8, 8), np.int32)
        labels[:, 3, 3] = 1
        labels[4, 3, 4] = 4
        labels[5, 3, 4] = 2
        arrays, _ = select(labels)
        self.assertFalse(arrays['mask'][4, 3, 4])
        arrays, _ = select(labels, include_branches=False)
        self.assertEqual(arrays['branch_mask'].sum(), 0)

    def test_missing_aorta_and_malformed_inputs_fail(self):
        labels = np.zeros((10, 8, 8), np.int32)
        with self.assertRaisesRegex(ValueError, 'No native aortic'):
            select(labels)
        with self.assertRaises(ValueError):
            select(labels.astype(float) + .5)
        with self.assertRaises(ValueError):
            select(labels, spacing_ijk_mm=[1, -1, 1])

    def test_landmarks_union_all_surface_parts_and_relative_reach(self):
        def row(name, lo, hi, ids):
            return {'original_name': name, 'source_file': 'fixture.raw', 'local_surface_index': 0,
                    'candidate_original_ids': ids, 'id_join_status': 'unique_name_match',
                    'bounds_raw_mm': {'min_xyz': [0, 0, lo], 'max_xyz': [10, 10, hi]}}
        aorta = row('aorta', 0, 20, [1])
        aorta['centerline'] = {'status': 'verified_two_pole_rings', 'ring_mean_radii_mm': [2, 3, 4]}
        records = [aorta, row('T12_vertebrae', 3, 5, [10]), row('T12_vertebrae', 2, 6, [10]),
                   row('T1_vertebrae', 15, 16, [11]), row('T1_vertebrae', 15, 17, [11])]
        settings = {'include_branches': True, 'branch_reach_mm': None, 'branch_reach_radii': 3, 'padding_mm': 2}
        domain = landmark_domain(records, catalog(), settings)
        self.assertEqual(domain['z_slab_mm'], [4, 17])
        self.assertEqual(domain['branch_reach_mm'], 9)
        self.assertEqual(domain['crop_bounds_raw_mm']['min_xyz'], [-11, -11, 2])
        with self.assertRaisesRegex(ValueError, 'T1_vertebrae'):
            landmark_domain(records[:3], catalog(), settings)

    def test_changed_surface_cache_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'selection.json'
            p.write_text('{}')
            p.with_suffix('.json.sha256').write_text('wrong')
            with self.assertRaisesRegex(ValueError, 'checksum'):
                _surfaces({'input': {'surface_inventory': p}}, {}, {})

    def test_cache_registration_and_case_mismatch_fail(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'inventory.json'
            inventory = {'schema_version': 'fakect.anatomy-surface-inventory/1',
                         'case_id': 'fixture', 'frame': 1, 'files': [], 'coordinate_dependencies': [],
                         'coordinate_contract': {'status': 'bounded_contract_validated',
                                                 'grid_boundary_origin_raw_mm': [0, 0, 0],
                                                 'spacing_ijk_mm': [1, 1, 1]}}
            def save():
                p.write_text(json.dumps(inventory))
                p.with_suffix('.json.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
            save()
            config = {'input': {'surface_inventory': p, 'case_id': 'fixture', 'frame': 1}}
            contract = {'grid_boundary_origin_raw_mm': [0, 0, .5], 'spacing_ijk_mm': [1, 1, 1]}
            with self.assertRaisesRegex(ValueError, 'geometry disagrees'):
                _surfaces(config, {}, contract)
            inventory['case_id'] = 'another'
            save()
            with self.assertRaisesRegex(ValueError, 'case/frame/schema mismatch'):
                _surfaces(config, {}, contract)

    def test_changed_cached_raw_file_rejected(self):
        with tempfile.TemporaryDirectory() as d:
            p = Path(d) / 'inventory.json'
            raw = Path(d) / 'fixture.raw'
            raw.write_text('aorta')
            inventory = {'schema_version': 'fakect.anatomy-surface-inventory/1',
                         'case_id': 'fixture', 'frame': 1,
                         'files': [{'path': str(raw), 'bytes': 1, 'mtime_ns': raw.stat().st_mtime_ns}],
                         'coordinate_contract': {'status': 'bounded_contract_validated'}}
            p.write_text(json.dumps(inventory))
            p.with_suffix('.json.sha256').write_text(hashlib.sha256(p.read_bytes()).hexdigest())
            with self.assertRaisesRegex(ValueError, 'Cached surface source changed'):
                _surfaces({'input': {'surface_inventory': p, 'case_id': 'fixture', 'frame': 1}}, {}, {})


if __name__ == '__main__':
    unittest.main()
