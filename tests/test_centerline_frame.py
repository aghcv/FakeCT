"""Analytic geometry and ambiguity handling for fitted centerline frames."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_centerline_frame import build_centerline_frame


class CenterlineFrameTests(unittest.TestCase):
    def circle(self, radius=20., count=41):
        angle = np.linspace(.2, 2.4, count)
        return np.column_stack((50 + radius * np.cos(angle),
                                50 + radius * np.sin(angle), np.full(count, 30.)))

    def frame(self, nodes=None, spacing=(1, 1, 1), **kwargs):
        defaults = dict(smoothing_mm=0., sample_step_mm=.5, min_curvature_per_mm=.002)
        defaults.update(kwargs)
        return build_centerline_frame(self.circle() if nodes is None else nodes, spacing, **defaults)

    def assert_orthonormal(self, result):
        for value in result.values():
            if isinstance(value, np.ndarray) and value.dtype.kind in 'fiu':
                self.assertTrue(np.isfinite(value).all())
        tangent, normal, binormal = (result[key] for key in ('tangent', 'normal', 'binormal'))
        for direction in (tangent, normal, binormal, result['reference_normal'], result['reference_binormal']):
            np.testing.assert_allclose(np.linalg.norm(direction, axis=1), 1., atol=1e-10)
        for first, last in ((tangent, normal), (tangent, binormal), (normal, binormal),
                            (tangent, result['reference_normal'])):
            np.testing.assert_allclose(np.sum(first * last, axis=1), 0., atol=1e-10)
        np.testing.assert_allclose(np.cross(tangent, normal), binormal, atol=1e-10)
        np.testing.assert_allclose(np.cross(tangent, result['reference_normal']),
                                   result['reference_binormal'], atol=1e-10)

    def test_circle_inner_normal_curvature_handedness_and_arc_spacing(self):
        result = self.frame()
        self.assert_orthonormal(result)
        keep = result['reliable']
        self.assertGreater(keep.mean(), .9)
        radial = result['xyz_mm'] - (50, 50, 30)
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        np.testing.assert_allclose(result['normal'][keep], -radial[keep], atol=.001)
        np.testing.assert_allclose(result['curvature_per_mm'][keep], .05, rtol=.005)
        np.testing.assert_allclose(result['binormal'][keep], np.tile((0, 0, 1), (keep.sum(), 1)), atol=1e-8)
        self.assertFalse(result['reliable'][0])
        self.assertFalse(result['reliable'][-1])
        self.assertEqual(result['endpoint'].sum(), 2)
        self.assertFalse(result['inflection_candidate'].any())
        differences = np.diff(result['spline_distance_mm'])
        self.assertLessEqual(float(differences.max()), .5 + 1e-12)
        np.testing.assert_allclose(differences, differences[0], atol=1e-12)
        self.assertAlmostEqual(result['metadata']['sampled_spline_length_mm'], 44., places=3)

    def test_helix_principal_normal_and_curvature(self):
        radius, pitch = 12., 3.
        angle = np.linspace(.1, 4 * np.pi, 121)
        nodes = np.column_stack((50 + radius * np.cos(angle), 50 + radius * np.sin(angle), 10 + pitch * angle))
        result = self.frame(nodes)
        self.assert_orthonormal(result)
        keep = result['reliable']
        radial = result['xyz_mm'] - (50, 50, 0)
        radial[:, 2] = 0
        radial /= np.linalg.norm(radial, axis=1)[:, None]
        np.testing.assert_allclose(result['normal'][keep], -radial[keep], atol=.003)
        np.testing.assert_allclose(result['curvature_per_mm'][keep], radius / (radius**2 + pitch**2), rtol=.015)
        self.assertGreater(keep.mean(), .95)

    def test_reversed_path_preserves_inner_direction_and_flips_tangent_binormal(self):
        nodes = self.circle()
        first, reverse = self.frame(nodes), self.frame(nodes[::-1])
        self.assertEqual(first['xyz_mm'].shape, reverse['xyz_mm'].shape)
        np.testing.assert_allclose(first['xyz_mm'], reverse['xyz_mm'][::-1], atol=1e-8)
        np.testing.assert_allclose(first['normal'], reverse['normal'][::-1], atol=1e-8)
        np.testing.assert_allclose(first['tangent'], -reverse['tangent'][::-1], atol=1e-8)
        np.testing.assert_allclose(first['binormal'], -reverse['binormal'][::-1], atol=1e-8)
        np.testing.assert_allclose(first['parent_fraction'], 1 - reverse['parent_fraction'][::-1], atol=1e-10)

    def test_straight_and_two_node_paths_have_display_frames_without_inner_semantics(self):
        for nodes in (np.array(((1, 2, 3), (3, 4, 5), (9, 10, 11), (15, 16, 17))),
                      np.array(((1, 2, 3), (15, 16, 17)))):
            with self.subTest(nodes=len(nodes)):
                result = self.frame(nodes, smoothing_mm=1.)
                self.assert_orthonormal(result)
                self.assertFalse(result['reliable'].any())
                self.assertTrue(np.all(result['normal_status'] == 'transported_low_curvature'))
                np.testing.assert_allclose(result['curvature_per_mm'], 0, atol=1e-12)
                np.testing.assert_allclose(result['normal'], result['reference_normal'], atol=1e-12)
                np.testing.assert_allclose(result['normal'], np.broadcast_to(result['normal'][0], result['normal'].shape), atol=1e-12)

    def test_anisotropic_conversion_preserves_original_polyline_parameter_mapping(self):
        physical = self.circle()
        spacing = np.array((.7, 1.3, 2.5))
        anisotropic = self.frame(physical / spacing, spacing)
        isotropic = self.frame(physical)
        for name in ('xyz_mm', 'tangent', 'normal', 'binormal', 'curvature_per_mm', 'parent_fraction'):
            np.testing.assert_allclose(anisotropic[name], isotropic[name], atol=1e-9)
        np.testing.assert_allclose(anisotropic['parent_distance_mm'],
                                   anisotropic['parent_fraction'] * anisotropic['metadata']['parent_total_length_mm'])
        # Uneven control spacing is retained in u, rather than node-number fractions.
        uneven = physical[[0, 2, 11, 20, 40]]
        result = self.frame(uneven)
        distances = np.r_[0., np.cumsum(np.linalg.norm(np.diff(uneven, axis=0), axis=1))]
        np.testing.assert_allclose(result['metadata']['parent_node_fraction'], distances / distances[-1])
        self.assertNotAlmostEqual(result['metadata']['parent_node_fraction'][1], .25)

    def test_inflection_is_flagged_without_flipping_true_normals(self):
        x = np.linspace(-2., 2., 41)
        nodes = np.column_stack((10 + x, 10 + x**3, np.full(len(x), 10.)))
        result = self.frame(nodes, sample_step_mm=.05, min_curvature_per_mm=.01)
        self.assert_orthonormal(result)
        self.assertTrue(result['inflection_candidate'].any())
        self.assertFalse(np.any(result['reliable'] & result['inflection_candidate']))
        keep = result['reliable']
        # Every valid normal keeps the curvature vector's sign, including both
        # sides of the inflection. Continuity may not turn inner into outer.
        dot = np.sum(result['normal'][keep] * result['curvature_vector_per_mm'][keep], axis=1)
        self.assertTrue(np.all(dot > 0))
        left = keep & (result['xyz_mm'][:, 0] < 9.9)
        right = keep & (result['xyz_mm'][:, 0] > 10.1)
        self.assertTrue(left.any() and right.any())
        self.assertTrue(np.all(result['normal'][left, 1] < 0))
        self.assertTrue(np.all(result['normal'][right, 1] > 0))

    def test_rms_smoothing_budget_is_measured_and_inputs_are_preserved(self):
        nodes = self.circle(count=25)
        rng = np.random.default_rng(817)
        nodes += rng.normal(0, .2, nodes.shape)
        original = nodes.copy()
        nodes.setflags(write=False)
        result = self.frame(nodes, smoothing_mm=.25)
        repeated = self.frame(nodes, smoothing_mm=.25)
        self.assert_orthonormal(result)
        np.testing.assert_array_equal(nodes, original)
        fitted = np.asarray(result['metadata']['fitted_controls_xyz_mm'])
        residual = np.linalg.norm(fitted - original, axis=1)
        self.assertAlmostEqual(result['metadata']['fit_rms_displacement_mm'], float(np.sqrt(np.mean(residual**2))))
        self.assertAlmostEqual(result['metadata']['fit_max_displacement_mm'], float(residual.max()))
        self.assertLessEqual(result['metadata']['fit_rms_displacement_mm'], .25 * 1.001)
        for key, value in result.items():
            if isinstance(value, np.ndarray):
                np.testing.assert_array_equal(value, repeated[key])
        self.assertEqual(result['metadata'], repeated['metadata'])

    def test_stationary_turn_has_a_finite_display_frame_and_no_inner_semantics(self):
        nodes = ((1, 1, 1), (2, 1, 1), (3, 1, 1), (2, 1, 1), (1, 1, 1))
        result = self.frame(nodes, sample_step_mm=100)
        self.assert_orthonormal(result)
        self.assertTrue(result['degenerate_tangent'][1])
        self.assertFalse(result['reliable'].any())
        self.assertEqual(result['normal_status'][1], 'transported_degenerate_tangent')

    def test_invalid_and_excessive_inputs_are_rejected(self):
        invalid = [dict(nodes=np.ones((1, 3))), dict(nodes=np.ones((513, 3))),
                   dict(nodes=np.ones((4, 2))), dict(nodes=np.ones((4, 3))),
                   dict(nodes=np.array(((-1, 0, 0), (1, 0, 0)))),
                   dict(nodes=np.array(((float('nan'), 0, 0), (1, 0, 0)))),
                   dict(spacing=(1, 0, 1)), dict(spacing=(1, float('inf'), 1)),
                   dict(smoothing_mm=-1), dict(smoothing_mm=float('nan')), dict(smoothing_mm=1e308),
                   dict(sample_step_mm=0), dict(sample_step_mm=1e-20),
                   dict(min_curvature_per_mm=-1), dict(max_samples=2), dict(max_samples=True),
                   dict(max_samples=4097), dict(max_samples=3)]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.frame(**kwargs)


if __name__ == '__main__':
    unittest.main()
