"""Independent native visual-audit metrics catch hidden splits and protected edits."""
from pathlib import Path
import sys
import unittest

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/research/evaluation'))
import visualize_cohort_trials as review


class TrialVisualReviewTests(unittest.TestCase):
    def test_correspondence_detects_split_and_loss_even_when_total_count_is_equal(self):
        before = np.zeros((3, 3, 9), dtype=bool)
        before[1, 1, 1:6] = True
        before[1, 1, 8] = True
        after = before.copy()
        after[1, 1, 3] = False
        after[1, 1, 8] = False
        result = review.component_correspondence(before, after)
        self.assertEqual(result['before_components'], result['after_components'])
        self.assertEqual(result['split_or_lost_original_components'], 2)

    def test_correspondence_detects_merge(self):
        before = np.zeros((3, 3, 7), dtype=bool)
        before[1, 1, [1, 3]] = True
        after = before.copy(); after[1, 1, 2] = True
        result = review.component_correspondence(before, after)
        self.assertEqual(result['split_or_lost_original_components'], 0)
        self.assertEqual(len(result['merged_original_components']), 1)

    def test_arc_coordinates_use_physical_spacing_and_order(self):
        points = np.asarray([[1., 0., 0.], [2., 1., 0.]])
        distance, length = review.parent_arc(points, [[0, 0, 0], [2, 0, 0], [2, 2, 0]], [2., 3., 1.])
        np.testing.assert_allclose(distance, [2., 7.])
        self.assertEqual(length, 10.)
        reversed_distance, _ = review.parent_arc(points, [[2, 2, 0], [2, 0, 0], [0, 0, 0]], [2., 3., 1.])
        np.testing.assert_allclose(reversed_distance, [8., 3.])


if __name__ == '__main__':
    unittest.main()
