"""Physical subpath resolution independent of the voxel ROI implementation."""
from pathlib import Path
import sys
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_tube_range import resolve_tube_range


class TubeRangeTests(unittest.TestCase):
    def setUp(self):
        # Physical segment lengths are 6, 12, and 16 mm, in the given order.
        self.nodes = np.array(((1, 2, 3), (4, 2, 3), (4, 6, 3), (4, 6, 7)), dtype=float)
        self.radii = np.array((1, 3, 9, 5), dtype=float)
        self.spacing = np.array((2, 3, 4), dtype=float)

    def resolve(self, **kwargs):
        return resolve_tube_range(kwargs.pop('nodes_ijk', self.nodes),
                                  kwargs.pop('radii_mm', self.radii),
                                  kwargs.pop('spacing_ijk_mm', self.spacing), **kwargs)

    def test_full_path_is_identity_with_immutable_serializable_output(self):
        originals = [item.copy() for item in (self.nodes, self.radii, self.spacing)]
        for item in (self.nodes, self.radii, self.spacing):
            item.setflags(write=False)
        result = self.resolve()
        self.assertEqual(result['nodes_ijk'], tuple(map(tuple, self.nodes)))
        self.assertEqual(result['radii_mm'], tuple(self.radii))
        self.assertIsInstance(result['nodes_ijk'], tuple)
        self.assertIsInstance(result['nodes_ijk'][0], tuple)
        self.assertEqual(result['metadata']['selector'], 'full')
        self.assertEqual(result['metadata']['parent_total_length_mm'], 34)
        self.assertEqual(result['metadata']['selected_length_mm'], 34)
        self.assertEqual(result['metadata']['original_node_numbers_retained'], [1, 2, 3, 4])
        self.assertEqual((result['metadata']['start_fraction'], result['metadata']['end_fraction']), (0, 1))
        import json
        json.dumps(result, allow_nan=False)
        for source, original in zip((self.nodes, self.radii, self.spacing), originals):
            np.testing.assert_array_equal(source, original)

    def test_point_range_is_one_based_inclusive_without_reordering(self):
        result = self.resolve(point_range=(2, 3))
        self.assertEqual(result['nodes_ijk'], ((4., 2., 3.), (4., 6., 3.)))
        self.assertEqual(result['radii_mm'], (3., 9.))
        metadata = result['metadata']
        self.assertEqual(metadata['selector'], 'point_range')
        self.assertEqual(metadata['selector_values'], [2, 3])
        self.assertEqual(metadata['start_distance_mm'], 6)
        self.assertEqual(metadata['end_distance_mm'], 18)
        self.assertEqual(metadata['selected_length_mm'], 12)
        self.assertEqual(metadata['original_node_numbers_retained'], [2, 3])
        self.assertAlmostEqual(metadata['start_fraction'], 6 / 34)
        self.assertAlmostEqual(metadata['end_fraction'], 18 / 34)

    def test_percentage_interpolates_physical_arc_length_and_radius(self):
        result = self.resolve(path_percent=(25, 75))
        np.testing.assert_allclose(result['nodes_ijk'],
                                   ((4, 2 + 4 * (2.5 / 12), 3), (4, 6, 3), (4, 6, 4.875)))
        np.testing.assert_allclose(result['radii_mm'], (4.25, 9, 7.125))
        metadata = result['metadata']
        self.assertEqual(metadata['original_node_numbers_retained'], [3])
        self.assertEqual(metadata['start_distance_mm'], 8.5)
        self.assertEqual(metadata['end_distance_mm'], 25.5)
        self.assertEqual(metadata['selected_length_mm'], 17)
        self.assertEqual((metadata['start_fraction'], metadata['end_fraction']), (.25, .75))
        # A same-segment interval keeps just its two interpolated endpoints.
        short = self.resolve(path_percent=(30, 40))
        self.assertEqual(len(short['nodes_ijk']), 2)
        self.assertEqual(short['metadata']['original_node_numbers_retained'], [])
        self.assertAlmostEqual(short['metadata']['selected_length_mm'], 3.4)

    def test_node_coincidences_and_full_percent_do_not_duplicate_endpoints(self):
        result = self.resolve(path_percent=(6 / 34 * 100, 18 / 34 * 100))
        self.assertEqual(result['nodes_ijk'], ((4., 2., 3.), (4., 6., 3.)))
        self.assertEqual(result['radii_mm'], (3., 9.))
        self.assertEqual(result['metadata']['original_node_numbers_retained'], [2, 3])
        complete = self.resolve(path_percent=(0, 100))
        self.assertEqual(complete['nodes_ijk'], self.resolve()['nodes_ijk'])
        self.assertEqual(complete['radii_mm'], self.resolve()['radii_mm'])
        self.assertEqual(complete['metadata']['original_node_numbers_retained'], [1, 2, 3, 4])

    def test_reversing_parent_reverses_percentage_direction(self):
        forward = self.resolve(path_percent=(20, 65))
        reversed_result = self.resolve(nodes_ijk=self.nodes[::-1], radii_mm=self.radii[::-1],
                                       path_percent=(35, 80))
        np.testing.assert_allclose(reversed_result['nodes_ijk'], forward['nodes_ijk'][::-1])
        np.testing.assert_allclose(reversed_result['radii_mm'], forward['radii_mm'][::-1])
        reverse_points = self.resolve(nodes_ijk=self.nodes[::-1], radii_mm=self.radii[::-1], point_range=(1, 2))
        self.assertEqual(reverse_points['nodes_ijk'], ((4., 6., 7.), (4., 6., 3.)))
        self.assertEqual(reverse_points['radii_mm'], (5., 9.))

    def test_nonadjacent_repeated_node_keeps_order_and_arc_length(self):
        nodes = ((0, 0, 0), (1, 0, 0), (0, 0, 0), (0, 2, 0))
        result = self.resolve(nodes_ijk=nodes, radii_mm=(1, 1, 1, 1), spacing_ijk_mm=(1, 1, 1))
        self.assertEqual(result['nodes_ijk'], nodes)
        self.assertEqual(result['metadata']['parent_total_length_mm'], 4)

    def test_rejects_invalid_selectors(self):
        invalid = [dict(point_range=(1, 3), path_percent=(0, 50)),
                   *[dict(point_range=value) for value in
                     ((0, 2), (1, 5), (2, 2), (3, 2), (1.5, 3), (1., 3.),
                      (True, False), (1,), (1, 2, 3), (1, float('nan')), ('1', '3'))],
                   *[dict(path_percent=value) for value in
                     ((-1, 50), (1, 101), (30, 30), (80, 20), (0, float('inf')), (0,), (0, 20, 100))]]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.resolve(**kwargs)

    def test_rejects_invalid_parent_geometry(self):
        invalid = [dict(nodes_ijk=((0, 0, 0),)),
                   dict(nodes_ijk=((0, 0), (1, 1))),
                   dict(nodes_ijk=((0, 0, 0), (0, 0, 0), (1, 0, 0), (2, 0, 0))),
                   dict(nodes_ijk=((-1, 0, 0), (1, 0, 0))),
                   dict(nodes_ijk=((float('nan'), 0, 0), (1, 0, 0))),
                   dict(radii_mm=(1, 2)), dict(radii_mm=(1, 0, 1, 1)),
                   dict(radii_mm=(1, 1, 1, float('inf'))),
                   dict(spacing_ijk_mm=(1, 0, 1)), dict(spacing_ijk_mm=(1, 1)),
                   dict(spacing_ijk_mm=(1, 1, float('nan'))),
                   dict(nodes_ijk=((0, 0, 0), (1e308, 0, 0)), radii_mm=(1, 1), spacing_ijk_mm=(100, 1, 1))]
        for kwargs in invalid:
            with self.subTest(kwargs=kwargs), self.assertRaises(ValueError):
                self.resolve(**kwargs)


if __name__ == '__main__':
    unittest.main()
