"""Calibration, anatomical axis signs and frozen input evidence boundaries."""
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
import fakect_pretrained_evidence as evidence

PARAMETERS = '''pixel_width = 0.1
slice_width = 0.2
energy = 120
startslice = 1
endslice = 100
x_array_size = 100
y_array_size = 100
activ_output_format = 0
color_code = 1
activity_unit = 0
phan_rotx = 0
phan_roty = 0
phan_rotz = 0
# The x-axis runs from the right side of the body to the left.
# The y-axis runs from the front of the body to the back.
# The z-axis runs from the feet to the head.
'''
LOG = '''radionuclide energy (keV)= 120.0000
Linear Attenuation Coefficients (1/cm):
 Body (water) = 0.1614
 Blood = 0.1697
Linear Attenuation Coefficients (1/pixel):
 Body (water) = 0.0161
 Blood = 0.0170
'''


def landmark_records():
    out = []
    for name, group, midpoint in (
            ('llung', 'organs', [20, 0, 100]), ('rlung', 'organs', [-20, 0, 100]),
            ('lkidney', 'organs', [15, 0, 70]), ('rkidney', 'organs', [-15, 0, 70]),
            ('sternum', 'bones', [0, -20, 110]), ('T6_vertebrae', 'bones', [0, 25, 110]),
            ('head', 'body', [0, 0, 180]), ('foot_left', 'body', [10, 0, 10]),
            ('foot_right', 'body', [-10, 0, 10])):
        center = np.array(midpoint, dtype=float)
        out.append({'original_name': name, 'group': group, 'source_file': group + '.raw',
                    'local_surface_index': len(out), 'bounds_raw_mm': {
                        'min_xyz': (center - 1).tolist(), 'max_xyz': (center + 1).tolist()}})
    return out


class CalibrationAndOrientationTests(unittest.TestCase):
    # Purpose: Verify native attenuation calibration records generator energy and precision.
    # Inputs: 120 keV parameter/log fixture with 1/1/2 mm spacing.
    # Outputs: Water coefficient 0.1614, rounding 0.00005, scale 10, and zero attenuation at -1000 HU.
    def test_water_scaling_energy_and_declared_rounding(self):
        result = evidence.calibration_evidence(PARAMETERS, LOG, [1, 1, 2])
        self.assertEqual(result['mu_water_cm_inverse'], .1614)
        self.assertAlmostEqual(result['mu_water_rounding_half_unit_cm_inverse'], .00005)
        self.assertEqual(result['source_atn_to_cm_inverse_scale'], 10.)
        self.assertEqual(result['photon_energy_keV'], 120.)
        self.assertFalse(result['clinical_reconstruction'])
        self.assertEqual(result['zero_attenuation_HU'], -1000.)

    # Purpose: Verify calibration rejects contradictory frozen metadata.
    # Inputs: Mismatched energy, inconsistent water per-pixel coefficient, wrong spacing, or duplicate energy.
    # Outputs: ValueError for every contradictory calibration case.
    def test_energy_spacing_and_perpixel_contradictions_fail(self):
        for par, log, spacing in (
                (PARAMETERS, LOG.replace('120.0000', '80.0000'), [1, 1, 2]),
                (PARAMETERS, LOG.replace('0.0161', '0.0400'), [1, 1, 2]),
                (PARAMETERS, LOG, [1, 1, 1]),
                (PARAMETERS + 'energy = 100\n', LOG, [1, 1, 2])):
            with self.subTest(log=log, spacing=spacing):
                with self.assertRaises(ValueError):
                    evidence.calibration_evidence(par, log, spacing)

    # Purpose: Verify anatomical landmarks establish signed voxel-center orientation.
    # Inputs: Left/right lungs/kidneys, sternum/vertebra/head/feet on 100-cubed anisotropic grid.
    # Outputs: L/P/S axes, expected nonidentity RAS affine, five checks, and no registration claim.
    def test_landmarks_prove_nonidentity_ras_affine_at_voxel_centers(self):
        result = evidence.landmark_orientation(landmark_records(), PARAMETERS, [100, 100, 100], [1, 1, 2], [-50, -50, 0])
        self.assertEqual(result['native_axis_directions'], ['L', 'P', 'S'])
        self.assertEqual(result['affine_ras_ijk'], [[-1, 0, 0, 49.5], [0, -1, 0, 49.5], [0, 0, 2, 1], [0, 0, 0, 1]])
        self.assertEqual(len(result['landmark_checks']), 5)
        self.assertFalse(result['independent_surface_to_volume_registration'])
        affine = np.asarray(result['affine_ras_ijk'])
        self.assertTrue(np.allclose(affine @ [10, 20, 30, 1], [39.5, 29.5, 61, 1]))

    # Purpose: Verify contradictory or incomplete landmarks cannot be repaired silently.
    # Inputs: Mirrored x landmark bounds and missing foot landmark.
    # Outputs: ValueError for generator-axis contradiction and missing independent landmarks.
    def test_mirrored_or_missing_landmarks_cannot_be_silently_reoriented(self):
        records = landmark_records()
        for row in records:
            low, high = row['bounds_raw_mm']['min_xyz'], row['bounds_raw_mm']['max_xyz']
            low[0], high[0] = -high[0], -low[0]
        with self.assertRaisesRegex(ValueError, 'contradicts generator axis'):
            evidence.landmark_orientation(records, PARAMETERS, [100]*3, [1, 1, 2], [-50, -50, 0])
        with self.assertRaisesRegex(ValueError, 'Missing independent'):
            evidence.landmark_orientation(landmark_records()[:-1], PARAMETERS, [100]*3, [1, 1, 2], [-50, -50, 0])

    # Purpose: Verify orientation requires unrotated documented native geometry.
    # Inputs: 15-degree rotation, reversed axis note, or incorrect grid origin.
    # Outputs: ValueError for each unsupported orientation contract.
    def test_rotation_missing_axis_notes_or_wrong_origin_are_rejected(self):
        for par, origin in ((PARAMETERS.replace('phan_rotx = 0', 'phan_rotx = 15'), [-50, -50, 0]),
                            (PARAMETERS.replace('feet to the head', 'head to the feet'), [-50, -50, 0]),
                            (PARAMETERS, [0, 0, 0])):
            with self.subTest(origin=origin):
                with self.assertRaises(ValueError):
                    evidence.landmark_orientation(landmark_records(), par, [100]*3, [1, 1, 2], origin)


class FrozenCaseEvidenceTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.case = self.root / '260602'
        self.case.mkdir()
        self.par = self.root / '260602.par'
        self.par.write_text(PARAMETERS)
        self.log = self.case / '260602_log'
        self.log.write_text(LOG)
        self.generator = self.root / 'dxcat2_linux_64bit'
        self.generator.write_bytes(b'fixture generator')
        self.row = {'case_id': '260602', 'frame': 1, 'source_atn_units': 'per_pixel', 'image_units': 'cm^-1',
                    'source_atn_dtype': '<f4', 'source_shape_kji': [100]*3, 'spacing_ijk_mm': [1, 1, 2]}
        for channel in ('atn', 'act'):
            path = self.case / f'260602_{channel}_1.bin'
            with path.open('wb') as stream:
                stream.truncate(4 * 100**3)
            self.row['source_' + channel] = self.record(path)
        raw_files = []
        for group in ('body', 'bones', 'heart', 'muscle', 'organs', 'vessels'):
            path = self.case / f'260602_1_{group}.raw'
            path.write_text('cached geometry source fixture\n')
            raw_files.append(self.record(path))
        self.surface_path = self.root / 'surface-cache.json'
        self.surface_path.write_text(json.dumps({'surface_evidence': {'case_id': '260602', 'frame': 1,
                                               'files': raw_files, 'records': landmark_records()}}))
        self.audit_path = self.root / 'source-audit.json'
        self.audit_path.write_text(json.dumps({'source_metadata_sha256': {
            str(self.par): self.record(self.par)['sha256'], str(self.log): self.record(self.log)['sha256']}}))
        self.selection_path = self.root / 'selection.json'
        self.selection_path.write_text(json.dumps({'case_id': '260602', 'frame': 1, 'complete': True,
            'audit_path': str(self.audit_path), 'audit_sha256': self.record(self.audit_path)['sha256'],
            'surface_inventory': {'path': str(self.surface_path), 'sha256': self.record(self.surface_path)['sha256']},
            'coordinate_contract': {'status': 'bounded_contract_validated', 'parameter_source': self.record(self.par)},
            'geometry': {'shape_kji': [100]*3, 'spacing_ijk_mm': [1, 1, 2]}}))
        self.row['selection_manifest'] = str(self.selection_path)
        self.row['selection_manifest_sha256'] = self.record(self.selection_path)['sha256']
        contract, provenance = evidence._load_evidence()
        contract = deepcopy(contract)
        contract['external_generator'].update(sha256=self.record(self.generator)['sha256'], size_bytes=self.generator.stat().st_size)
        patcher = patch.object(evidence, '_load_evidence', return_value=(contract, provenance))
        patcher.start()
        self.addCleanup(patcher.stop)

    @staticmethod
    def record(path):
        stat = path.stat()
        return {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns,
                'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}

    # Purpose: Verify frozen case evidence binds calibration, orientation, and source metadata.
    # Inputs: Checksummed parameter/log/generator/audit/surface fixtures for case 260602 frame 1.
    # Outputs: Expected water calibration and axes, required dependencies, and finite JSON.
    def test_case_evidence_binds_calibration_landmarks_and_external_dependencies(self):
        result = evidence.case_evidence(self.row)
        self.assertEqual(result['calibration']['mu_water_cm_inverse'], .1614)
        self.assertEqual(result['orientation']['native_axis_directions'], ['L', 'P', 'S'])
        paths = {row['path'] for row in result['dependencies']}
        self.assertTrue({str(self.par), str(self.log), str(self.generator), str(self.surface_path), str(self.audit_path)} <= paths)
        json.dumps(result, allow_nan=False)

    # Purpose: Verify changed calibration or surface bytes invalidate frozen evidence.
    # Inputs: Modified water log then overwritten surface-cache JSON.
    # Outputs: ValueError for both checksum mismatches.
    def test_modified_log_and_surface_cache_fail_frozen_binding(self):
        self.log.write_text(LOG.replace('0.1614', '0.1600'))
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            evidence.case_evidence(self.row)
        self.log.write_text(LOG)
        self.surface_path.write_text('{}')
        with self.assertRaisesRegex(ValueError, 'checksum mismatch'):
            evidence.case_evidence(self.row)

    # Purpose: Verify case evidence enforces frame scope and native payload byte lengths.
    # Inputs: Case changed to frame 2 or attenuation byte count changed to 12.
    # Outputs: ValueError for unsupported frame and inconsistent byte lengths.
    def test_unsupported_frame_and_wrong_source_size_fail(self):
        changed = deepcopy(self.row)
        changed['frame'] = 2
        with self.assertRaises(ValueError):
            evidence.case_evidence(changed)
        changed = deepcopy(self.row)
        changed['source_atn']['bytes'] = 12
        with self.assertRaisesRegex(ValueError, 'byte lengths'):
            evidence.case_evidence(changed)


if __name__ == '__main__':
    unittest.main()
