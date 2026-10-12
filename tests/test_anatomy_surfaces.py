"""RAW identity, scoped coordinates and structured centerline evidence."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
import fakect_anatomy_surfaces as surfaces
from fakect_anatomy_centerlines import extract_centerline, centerline_envelope_overlap


def tube_triangles():
    angles = np.arange(8) * 2 * np.pi / 8
    rings = np.asarray([np.column_stack((np.cos(angles), np.sin(angles), np.full(8, z)))
                        for z in (1., 2., 3., 4.)])
    faces = []
    for n in range(8):
        nxt = (n + 1) % 8
        faces.append([[0, 0, 0], rings[0, n], rings[0, nxt]])
        faces.append([[0, 0, 5], rings[-1, nxt], rings[-1, n]])
        for a, b in zip(rings[:-1], rings[1:]):
            faces.extend([[a[n], b[n], a[nxt]], [a[nxt], b[n], b[nxt]]])
    return np.asarray(faces)


class SurfaceInventoryTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = self.root / '260602'
        self.case.mkdir()
        self.config = {'input': {'root': self.root, 'case_id': '260602', 'frame': 1}}
        self.par = self.root / '260602.par'
        self.par.write_text('startslice = 1\nendslice = 10\nx_array_size = 10\ny_array_size = 10\n'
                            'pixel_width = 0.1\nslice_width = 0.1\nactiv_output_format = 0\n'
                            'color_code = 1\nactivity_unit = 0\n')
        self.resolved = {'case': {'par_path': self.par}, 'shape_kji': [10, 10, 10],
                         'spacing_ijk_mm': [1., 1., 1.], 'roi_nodes_ijk': [[5, 5, 5]], 'roi_radii_mm': [1.],
                         'catalog': {'records': [
                             {'original_id': 5, 'original_name': 'arteries', 'classification': {'tissue_name': 'artery'}},
                             {'original_id': 6, 'original_name': 'arteries', 'classification': {'tissue_name': 'artery'}},
                             {'original_id': -8, 'original_name': 'signed_label', 'classification': {'tissue_name': 'artery'}}]}}
        for group in surfaces.RAW_GROUPS:
            (self.case / f'260602_1_{group}.raw').write_text('\narteries\n0 0 5 1 0 5 0 1 5\n\n')
        self.evidence, self.provenance = surfaces._load_evidence()
        self.generator = self.root / 'dxcat2_linux_64bit'
        self.generator.write_bytes(b'fixture generator')
        self.fixture_evidence = deepcopy(self.evidence)
        self.fixture_evidence['external_generator'].update(
            sha256=hashlib.sha256(self.generator.read_bytes()).hexdigest(), size_bytes=self.generator.stat().st_size)

    def contract(self, evidence=None):
        return surfaces._coordinate_contract(self.config, self.resolved, evidence or self.fixture_evidence, self.provenance)

    # Purpose: Verify RAW names join catalog IDs without guessing identity by order.
    # Inputs: Repeated arteries blocks and signed_label block with catalog IDs 5, 6, and -8.
    # Outputs: Ambiguous IDs retained, signed ID joined, dependencies hashed, and no unsupported spatial claim.
    def test_exact_name_join_keeps_all_signed_candidate_ids_without_ordinal_guess(self):
        path = self.case / '260602_1_heart.raw'
        path.write_text('arteries\n0 0 5 1 0 5 0 1 5\narteries\n0 0 6 1 0 6 0 1 6\n'
                        'signed_label\n0 0 7 1 0 7 0 1 7\n')
        result = surfaces.inventory_surfaces(self.config, self.resolved)
        records = [r for r in result['records'] if r['group'] == 'heart']
        self.assertEqual([r['local_surface_index'] for r in records], [0, 1, 2])
        self.assertEqual([r['candidate_original_ids'] for r in records], [[5, 6], [5, 6], [-8]])
        self.assertEqual(records[0]['id_join_status'], 'ambiguous_name')
        self.assertEqual(result['coordinate_contract']['status'], 'unavailable')
        self.assertNotIn('bounds_overlap_roi_envelope', records[0])
        self.assertEqual(result['files'][2]['sha256'], hashlib.sha256(path.read_bytes()).hexdigest())
        dependencies = {Path(row['path']).name: row for row in result['coordinate_dependencies']}
        self.assertEqual(set(dependencies), {'260602.par', 'dxcat2_linux_64bit', surfaces.LAYOUT_EVIDENCE.name})
        self.assertEqual(dependencies['260602.par']['bytes'], self.par.stat().st_size)
        self.assertEqual(dependencies['dxcat2_linux_64bit']['mtime_ns'], self.generator.stat().st_mtime_ns)
        json.dumps(result, allow_nan=False)

    # Purpose: Verify coordinate evidence requires the pinned generator content.
    # Inputs: Matching fixture generator, original evidence mismatch, then deleted generator.
    # Outputs: Validated matching contract and explicit mismatch/unavailable reasons.
    def test_contract_requires_content_matching_generator(self):
        self.assertEqual(self.contract()['status'], 'bounded_contract_validated')
        self.assertIn('generator_content_mismatch', self.contract(self.evidence)['reasons'])
        self.generator.unlink()
        self.assertIn('generator_content_unavailable', self.contract()['reasons'])

    # Purpose: Verify spatial contracts stay within their pinned scope.
    # Inputs: Frame 2, unknown case, startslice 0, and inconsistent voxel spacing.
    # Outputs: Explicit scope/geometry reasons and absent grid origin for unsupported startslice.
    def test_unsupported_frame_case_startslice_and_geometry_have_no_spatial_claim(self):
        self.config['input']['frame'] = 2
        self.assertIn('frame_outside_pinned_scope', self.contract()['reasons'])
        self.config['input']['case_id'] = 'unknown'
        self.assertIn('case_outside_pinned_scope', self.contract()['reasons'])
        self.par.write_text(self.par.read_text().replace('startslice = 1', 'startslice = 0'))
        self.assertIn('startslice_other_than_1', self.contract()['reasons'])
        self.assertNotIn('grid_boundary_origin_mm', self.contract())
        self.par.write_text(self.par.read_text().replace('startslice = 0', 'startslice = 1'))
        self.resolved['spacing_ijk_mm'] = [2, 1, 1]
        self.assertIn('parameter_geometry_disagrees_with_resolved_volume', self.contract()['reasons'])

    # Purpose: Verify surface screening uses voxel-center bounds as broad-phase evidence.
    # Inputs: Validated 10-cubed grid and triangles inside then outside the ROI envelope.
    # Outputs: Correct origin/envelope, overlap-only status, and no independent registration claim.
    def test_voxel_center_offset_and_broad_phase_only(self):
        contract = self.contract()
        self.assertEqual(contract['grid_boundary_origin_mm'], [-5, -5, 0])
        self.assertEqual(contract['roi_envelope_raw_mm'], {'min_xyz': [-.5, -.5, 4.5], 'max_xyz': [1.5, 1.5, 6.5]})
        path = self.case / '260602_1_heart.raw'
        file, records = surfaces._scan_file(path, 'heart', {'arteries': [5, 6]}, contract)
        self.assertTrue(records[0]['bounds_overlap_roi_envelope'])
        self.assertEqual(records[0]['spatial_status'], 'bounds_overlap_only')
        self.assertFalse(contract['independent_registration'])
        path.write_text('arteries\n100 0 5 101 0 5 100 1 5\n')
        _, records = surfaces._scan_file(path, 'heart', {}, contract)
        self.assertFalse(records[0]['bounds_overlap_roi_envelope'])

    # Purpose: Verify malformed RAW geometry cannot be mistaken for anatomy names.
    # Inputs: Short, nonnumeric, nonfinite, extra-token, and unnamed triangle rows.
    # Outputs: ValueError for every malformed or unlabelled geometry case.
    def test_invalid_geometry_is_rejected_instead_of_creating_names(self):
        path = self.case / '260602_1_heart.raw'
        for bad in ('0 0 5 1 0 5 0 1', '0 0 5 1 invalid 5 0 1 5',
                    'nan 0 5 1 0 5 0 1 5', 'inf 0 5 1 0 5 0 1 5',
                    '0 0 5 1 0 5 0 1 1e999', '0 0 5 1 0 5 0 1 5 extra'):
            with self.subTest(row=bad):
                path.write_text('arteries\n' + bad + '\n')
                with self.assertRaises(ValueError):
                    surfaces._scan_file(path, 'heart', {}, {'status': 'unavailable'})
        path.write_text('0 0 5 1 0 5 0 1 5\n')
        with self.assertRaises(ValueError):
            surfaces._scan_file(path, 'heart', {}, {'status': 'unavailable'})

    # Purpose: Verify centerline extraction is restricted by artery identity and budget.
    # Inputs: Tube triangles in arterial/nonvascular blocks and a three-triangle cap.
    # Outputs: Verified arterial centerline and explicit nonvascular/budget rejection reasons.
    def test_arterial_blocks_only_retain_bounded_centerline_geometry(self):
        path = self.case / '260602_1_heart.raw'
        rows = '\n'.join(' '.join(map(str, row)) for row in tube_triangles().reshape(-1, 9))
        path.write_text('arteries\n' + rows + '\nnonvascular\n' + rows + '\n')
        _, records = surfaces._scan_file(path, 'heart', {'arteries': [5, 6]}, self.contract(), {'arteries'})
        self.assertEqual(records[0]['centerline']['status'], 'verified_two_pole_rings')
        self.assertEqual(records[1]['centerline']['reason'], 'not_a_catalog_artery_candidate')
        with patch.object(surfaces, 'MAX_TRIANGLES', 3):
            _, records = surfaces._scan_file(path, 'heart', {}, self.contract(), {'arteries'})
        self.assertEqual(records[0]['centerline']['reason'], 'surface_exceeds_bounded_centerline_budget')

    # Purpose: Verify inventories require every expected RAW group.
    # Inputs: Fixture with vessels RAW file deleted.
    # Outputs: FileNotFoundError instead of an incomplete inventory.
    def test_missing_raw_file_fails_instead_of_claiming_complete_inventory(self):
        (self.case / '260602_1_vessels.raw').unlink()
        with self.assertRaises(FileNotFoundError):
            surfaces.inventory_surfaces(self.config, self.resolved)


class StructuredCenterlineTests(unittest.TestCase):
    # Purpose: Verify structured tube centerlines survive shuffled triangle ordering.
    # Inputs: Four unit-radius rings and two poles shuffled with seed 42.
    # Outputs: Verified two-pole topology, four rings, two boundary cycles, and z centers 0 through 5.
    def test_closed_two_pole_rings_recover_centroids_independent_of_row_order(self):
        triangles = tube_triangles()
        np.random.default_rng(42).shuffle(triangles)
        result = extract_centerline(triangles)
        self.assertEqual(result['status'], 'verified_two_pole_rings')
        self.assertEqual(result['proof']['ring_count'], 4)
        self.assertEqual(result['proof']['boundary_cycles'], 2)
        self.assertTrue(np.allclose(result['ring_mean_radii_mm'], 1))
        self.assertTrue(np.allclose(result['ring_max_radii_mm'], 1))
        self.assertTrue(np.allclose(np.array(result['centerline_raw_mm'])[:, 2], np.arange(6)))

    # Purpose: Verify ambiguous triangle topology yields no centerline.
    # Inputs: Tube with missing face, duplicate face, or disconnected translated copy.
    # Outputs: Unavailable centerline status for all malformed soups.
    def test_open_duplicate_and_disconnected_soups_never_invent_centerlines(self):
        triangles = tube_triangles()
        for altered in (triangles[:-1], np.concatenate((triangles, triangles[:1])),
                        np.concatenate((triangles, triangles + [10, 0, 0]))):
            with self.subTest(triangles=len(altered)):
                self.assertEqual(extract_centerline(altered)['status'], 'unavailable')

    # Purpose: Verify radial extent contributes to centerline-envelope screening.
    # Inputs: Unit-radius tube and ROI envelope near x=1, then x=2 to 3, then invalid contract.
    # Outputs: Overlap true, false, and unknown respectively.
    def test_radius_screen_retains_surface_when_centerline_is_outside_roi(self):
        centerline = extract_centerline(tube_triangles())
        contract = {'status': 'bounded_contract_validated',
                    'roi_envelope_raw_mm': {'min_xyz': [.8, -.1, 2.2], 'max_xyz': [1.2, .1, 2.8]}}
        self.assertTrue(centerline_envelope_overlap(centerline, contract))
        contract['roi_envelope_raw_mm']['min_xyz'][0] = 2
        contract['roi_envelope_raw_mm']['max_xyz'][0] = 3
        self.assertFalse(centerline_envelope_overlap(centerline, contract))
        contract['status'] = 'unavailable'
        self.assertIsNone(centerline_envelope_overlap(centerline, contract))


if __name__ == '__main__':
    unittest.main()
