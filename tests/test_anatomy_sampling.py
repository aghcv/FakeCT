"""Native ACT corroboration of RAW centerlines without ordinal-to-ID guesses."""
import importlib.util
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_anatomy import sample_surface_centerlines


class NativeCenterlineSamplingTests(unittest.TestCase):
    def setUp(self):
        self.origin = np.array([-100., -200., 0.])
        self.spacing = np.array([2., 3., 4.])
        self.low = np.array([10, 20, 30])
        labels = np.full((2, 2, 4), 9876., dtype=np.float32)
        labels[0, 0] = [-7, 1547, 1550, 999]
        roi = np.zeros(labels.shape, dtype=bool)
        roi[0, 0, 1:3] = True
        labels.flags.writeable = False
        roi.flags.writeable = False
        self.arrays = {'act': labels, 'roi': roi}
        self.resolved = {'crop_low_ijk': self.low.tolist(), 'spacing_ijk_mm': self.spacing.tolist()}
        self.evidence = {'coordinate_contract': {
            'status': 'bounded_contract_validated', 'grid_boundary_origin_mm': self.origin.tolist(),
            'spacing_ijk_mm': self.spacing.tolist(), 'voxel_center_offset_ijk': [.5, .5, .5]},
            'records': [self.surface([[10.1, 20.25, 30.25], [13.9, 20.25, 30.25]])]}

    def surface(self, boundary_indices, ordinal=2):
        world = self.origin + np.asarray(boundary_indices) * self.spacing
        return {'original_name': 'arteries', 'local_surface_index': ordinal,
                'candidate_original_ids': [-7, 1547, 1550], 'id_join_status': 'ambiguous_name',
                'centerline': {'status': 'verified_two_pole_rings', 'centerline_raw_mm': world.tolist()}}

    def sample(self):
        sample_surface_centerlines(self.evidence, self.arrays, self.resolved)
        return self.evidence['records'][0]['native_centerline_samples']

    # Purpose: Verify RAW world centerlines sample native ACT cells.
    # Inputs: Anisotropic spacing [2,3,4], crop offset [10,20,30], and signed ACT labels.
    # Outputs: Four distinct cells and two ROI cells are counted without ordinal or background IDs.
    def test_anisotropic_world_coordinates_crop_offset_and_signed_act_values(self):
        actual = self.sample()
        self.assertEqual(actual['sample_voxels'], 4)
        self.assertEqual(actual['roi_sample_voxels'], 2)
        self.assertEqual(actual['sampled_id_counts'], {-7: 1, 999: 1, 1547: 1, 1550: 1})
        self.assertEqual(actual['matching_name_id_counts'], {-7: 1, 1547: 1, 1550: 1})
        self.assertNotIn(2, actual['sampled_id_counts'])  # RAW ordinal is not an ACT label.
        self.assertNotIn(9876, actual['sampled_id_counts'])  # k,j,i addressing must be correct.

    # Purpose: Verify containing-cell lookup applies the boundary origin once.
    # Inputs: Two nearby centerline points in native cell [10,20,30].
    # Outputs: Only signed label -7 is sampled once.
    def test_containing_cell_uses_boundary_origin_without_second_half_offset(self):
        self.evidence['records'] = [self.surface([[10.1, 20.1, 30.1], [10.2, 20.1, 30.1]])]
        self.assertEqual(self.sample()['sampled_id_counts'], {-7: 1})

    # Purpose: Verify repeated surface names retain separate native evidence.
    # Inputs: Two same-name surfaces with candidate IDs -7,1547,1550 and different paths.
    # Outputs: Candidate sets agree; the second path samples only ID 1547 rather than surface ordinal 3.
    def test_repeated_names_keep_candidate_set_but_sample_distinct_native_voxels(self):
        self.evidence['records'].append(self.surface([[11.1, 20.25, 30.25], [11.8, 20.25, 30.25]], ordinal=3))
        self.sample()
        first, second = self.evidence['records']
        self.assertEqual(first['candidate_original_ids'], second['candidate_original_ids'])
        self.assertEqual(second['native_centerline_samples']['matching_name_id_counts'], {1547: 1})
        self.assertNotIn(3, second['native_centerline_samples']['sampled_id_counts'])

    # Purpose: Verify sampling deduplicates cells independently of path direction.
    # Inputs: The same centerline sampled forwards and backwards.
    # Outputs: Both directions produce identical sample counts and metadata.
    def test_resampling_and_reverse_direction_do_not_double_count_voxels(self):
        first = self.sample()
        self.evidence['records'][0]['centerline']['centerline_raw_mm'].reverse()
        self.assertEqual(self.sample(), first)

    # Purpose: Skip native evidence when coordinate or centerline verification is unavailable.
    # Inputs: Unavailable coordinate contract, then unavailable centerline status.
    # Outputs: Neither case adds native_centerline_samples.
    def test_unavailable_contract_or_unverified_centerline_has_no_samples(self):
        self.evidence['coordinate_contract']['status'] = 'unavailable'
        sample_surface_centerlines(self.evidence, self.arrays, self.resolved)
        self.assertNotIn('native_centerline_samples', self.evidence['records'][0])
        self.evidence['coordinate_contract']['status'] = 'bounded_contract_validated'
        self.evidence['records'][0]['centerline']['status'] = 'unavailable'
        sample_surface_centerlines(self.evidence, self.arrays, self.resolved)
        self.assertNotIn('native_centerline_samples', self.evidence['records'][0])

    # Purpose: Reject native sampling with inconsistent physical spacing.
    # Inputs: Contract spacing [1,1,1] versus resolved spacing [2,3,4].
    # Outputs: ValueError identifies the spacing disagreement.
    def test_contract_spacing_mismatch_is_rejected(self):
        self.evidence['coordinate_contract']['spacing_ijk_mm'] = [1, 1, 1]
        with self.assertRaisesRegex(ValueError, 'spacing disagrees'):
            self.sample()

    # Purpose: Verify path clipping limits sampling to the native crop.
    # Inputs: A million-cell crossing path and a wholly outside segment.
    # Outputs: Crossing samples four crop IDs; the outside path adds no native samples.
    def test_clip_distant_path_before_sampling_and_discard_outside_segments(self):
        self.evidence['records'] = [self.surface([[-1e6, 20.25, 30.25], [1e6, 20.25, 30.25]])]
        self.assertEqual(self.sample()['sampled_id_counts'], {-7: 1, 999: 1, 1547: 1, 1550: 1})
        self.evidence['records'] = [self.surface([[10, 50, 30], [14, 50, 30]])]
        sample_surface_centerlines(self.evidence, self.arrays, self.resolved)
        self.assertNotIn('native_centerline_samples', self.evidence['records'][0])

    # Purpose: Enforce the aggregate sampling budget before allocating combined segments.
    # Inputs: Eight alternating points spanning 20000 cells and a concatenate failure mock.
    # Outputs: ValueError reports the aggregate budget before concatenation.
    def test_aggregate_budget_stops_many_segments_before_concatenation(self):
        labels = np.zeros((1, 1, 20000), dtype=np.float32)
        arrays = {'act': labels, 'roi': np.ones(labels.shape, dtype=bool)}
        points = [[.1 if i % 2 == 0 else 19999.9, .5, .5] for i in range(8)]
        evidence = {'coordinate_contract': {'status': 'bounded_contract_validated',
                    'grid_boundary_origin_mm': [0, 0, 0], 'spacing_ijk_mm': [1, 1, 1]},
                    'records': [{'candidate_original_ids': [0], 'centerline': {
                        'status': 'verified_two_pole_rings', 'centerline_raw_mm': points}}]}
        with patch('fakect_anatomy.np.concatenate', side_effect=AssertionError('must stop before concatenating')):
            with self.assertRaisesRegex(ValueError, 'aggregate sampling budget'):
                sample_surface_centerlines(evidence, arrays, {'spacing_ijk_mm': [1, 1, 1], 'crop_low_ijk': [0, 0, 0]})


class AnatomyCliConfigurationTests(unittest.TestCase):
    # Purpose: Verify the anatomy CLI loads cohort configuration without native data.
    # Inputs: Archived 260602 anatomy INI and a creation mock that must not run.
    # Outputs: Recipe-cohort schema, thoracic_aorta selection, empty source IDs, and freeze settings load.
    def test_cohort_example_uses_generation_loader_without_reading_native_data(self):
        spec = importlib.util.spec_from_file_location('fakect_resolve_anatomy_cli_test', ROOT / 'scripts/resolve_anatomy.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
        path = ROOT / 'archive/research/configs/cohorts/coa/260602-anatomy.ini'
        with patch.object(module, 'create_anatomy_selection', side_effect=AssertionError('configuration only')):
            config = module.load_config(path)
        self.assertEqual(config['study']['schema_version'], 'fakect.recipe-cohort/1')
        self.assertEqual(config['input']['case_id'], '260602')
        self.assertEqual(config['selection']['anatomy'], 'thoracic_aorta')
        self.assertEqual(config['selection']['source_ids'], ())
        self.assertIn('freeze_directory', config['train'])


if __name__ == '__main__':
    unittest.main()
