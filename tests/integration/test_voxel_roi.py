"""Review larger generated crops on the native grid and physical normal planes."""
from contextlib import redirect_stdout
import hashlib
import io
import json
from pathlib import Path
import tempfile
import unittest

import numpy as np

from fakect_roi import prepare_crop, resolve_preview
from fakect_roi_selection_preview import control_frames, sample_section
from fakect_study_config import load_study_config
from train_study import run
from voxel_fixture import write_voxel_fixture


class VoxelRoiIntegrationTests(unittest.TestCase):
    # Purpose: Verify physical section sampling preserves continuous signals and signed labels.
    # Inputs: 64- and 96-voxel fixtures, anisotropic spacing, fractional controls, and a physical affine ramp.
    # Outputs: Sections match the ramp within 2e-12 and nearest labels retain ID -7.
    def test_physical_sections_preserve_affine_signal_and_signed_ids_at_both_resolutions(self):
        for size in (64, 96):
            with self.subTest(size=size), tempfile.TemporaryDirectory() as temporary:
                fixture = write_voxel_fixture(Path(temporary)/'source', size=size)
                config = load_study_config(fixture['cohort_ini'])
                resolved = resolve_preview(config)
                arrays, _ = prepare_crop(resolved, config)
                spacing = np.asarray(fixture['spacing_ijk_mm'])
                low = np.asarray(resolved['crop_low_ijk'])
                original_origin = np.asarray(fixture['crop_origin_ijk'])
                k, j, i = np.indices(arrays['act'].shape)
                xyz = (np.stack((i, j, k)).astype(float)
                       + low[:, None, None, None]
                       + original_origin[:, None, None, None]) * spacing[:, None, None, None]
                coefficients = np.array([1e-4, -2e-5, 3e-5])
                ramp = .025 + np.sum(xyz * coefficients[:, None, None, None], axis=0)
                frames = control_frames(resolved['roi_nodes_ijk'], spacing)
                np.testing.assert_allclose(frames['centers_mm'],
                                           np.asarray(resolved['roi_nodes_ijk'])*spacing)
                for index in (0, 2, 4):
                    section, valid, offsets = sample_section(
                        {'attenuation': ramp, 'activity': arrays['act']},
                        frames['centers_mm'][index], frames['u_basis'][index],
                        frames['v_basis'][index], spacing, low,
                        half_width_mm=4., pixel_mm=float(spacing.min()/2))
                    self.assertTrue(valid.all())
                    u, v = np.meshgrid(offsets, offsets)
                    physical = (frames['centers_mm'][index, :, None, None]
                                + frames['u_basis'][index, :, None, None]*u
                                + frames['v_basis'][index, :, None, None]*v)
                    original_xyz = physical + (original_origin*spacing)[:, None, None]
                    expected = .025 + np.sum(original_xyz*coefficients[:, None, None], axis=0)
                    np.testing.assert_allclose(section['attenuation'], expected, rtol=0, atol=2e-12)
                    nearest = np.floor(physical/spacing[:, None, None] - low[:, None, None] + .5).astype(int)
                    expected_ids = arrays['act'][nearest[2], nearest[1], nearest[0]]
                    np.testing.assert_array_equal(section['activity'], expected_ids)
                    self.assertIn(-7, np.unique(section['activity']))
                    self.assertTrue(set(np.unique(section['activity'])) <= set(np.unique(fixture['labels'])))
                self.assertTrue(np.any(np.asarray(resolved['roi_nodes_ijk']) % 1))
                self.assertTrue(np.any(low > 0))

    # Purpose: Distinguish padded section pixels from sampled tissue.
    # Inputs: Voxel fixture sampled near the crop edge with a 4 mm half-width.
    # Outputs: Validity marks both regions; padded attenuation and labels are zero while tissue remains positive.
    def test_section_padding_is_distinguished_from_observed_tissue(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_voxel_fixture(Path(temporary)/'source')
            resolved = resolve_preview(load_study_config(fixture['cohort_ini']))
            arrays, _ = prepare_crop(resolved, load_study_config(fixture['cohort_ini']))
            low = np.asarray(resolved['crop_low_ijk'])
            spacing = np.asarray(fixture['spacing_ijk_mm'])
            center = (low + np.array([.2, 8., 8.]))*spacing
            section, valid, _ = sample_section(
                {'attenuation': arrays['attenuation_cm_inverse'], 'activity': arrays['act']},
                center, np.array([1., 0., 0.]), np.array([0., 1., 0.]),
                spacing, low, half_width_mm=4., pixel_mm=float(spacing.min()))
            self.assertTrue(valid.any())
            self.assertFalse(valid.all())
            self.assertTrue(np.all(section['attenuation'][~valid] == 0))
            self.assertTrue(np.all(section['activity'][~valid] == 0))
            self.assertGreater(float(section['attenuation'][valid].max()), 0)

    # Purpose: Verify complete ROI report artifacts preserve native source data.
    # Inputs: Generated voxel fixture processed by the ROI selection preview.
    # Outputs: Every control has hashed PNGs; HTML embeds images and source bytes remain unchanged.
    def test_real_report_covers_every_control_and_preserves_source_bytes(self):
        with tempfile.TemporaryDirectory() as temporary:
            fixture = write_voxel_fixture(Path(temporary)/'source')
            with redirect_stdout(io.StringIO()):
                report = run(fixture['cohort_ini'], stage='preview', preview_mode='roi-selection')
            output = fixture['cohort_ini'].parent/'review'
            points = report['roi_selection']['rois'][0]['points']
            self.assertEqual(len(points), len(fixture['nodes_ijk']))
            self.assertFalse(report['geometry_edited'])
            self.assertEqual(report['geometry']['array_order'], 'kji')
            np.testing.assert_allclose(report['geometry']['spacing_ijk_mm'], fixture['spacing_ijk_mm'])
            for index, point in enumerate(points):
                self.assertEqual(point['point'], index+1)
                np.testing.assert_allclose(point['center_ijk'], fixture['nodes_ijk'][index], atol=5e-7)
                self.assertEqual(set(point['images']), {'attenuation', 'labels'})
                self.assertGreater(point['section_valid_fraction'], 0)
                for image in point['images'].values():
                    payload = (output/image['path']).read_bytes()
                    self.assertTrue(payload.startswith(b'\x89PNG\r\n\x1a\n'))
                    self.assertEqual(hashlib.sha256(payload).hexdigest(), image['sha256'])
            document = (output/'report.html').read_text()
            self.assertIn('data:image/png;base64,', document)
            self.assertIn('id="point"', document)
            self.assertNotIn('<iframe', document)
            self.assertFalse((output/'edit.npz').exists())
            self.assertFalse((fixture['cohort_ini'].parent/'pairs').exists())
            self.assertFalse((output/'INCOMPLETE').exists())
            self.assertEqual((output/'input.ini').read_bytes(), fixture['cohort_ini'].read_bytes())
            for path, expected in fixture['input_hashes'].items():
                self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected)
            manifest = json.loads((output/'artifact-manifest.json').read_text())
            for relative, expected in manifest.items():
                self.assertEqual(hashlib.sha256((output/relative).read_bytes()).hexdigest(), expected)


if __name__ == '__main__':
    unittest.main()
