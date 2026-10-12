"""Conservative clearance includes missing crop context and all growth sweeps."""
from pathlib import Path
import io
import json
import sys
import tempfile
import unittest
from contextlib import redirect_stdout
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'scripts/research/evaluation'))
import review_internal_fill as review


class InternalFillReviewTests(unittest.TestCase):
    # Purpose: Verify crop padding conservatively bounds interior clearance.
    # Inputs: 21-cube full tissue mask, then a nearby background voxel.
    # Outputs: Center clearance is 11 and lower bound 11-sqrt(3); nearby background reduces distance to 2.
    def test_crop_padding_conservatively_limits_interior_clearance(self):
        labels = np.ones((21, 21, 21), dtype=np.int32)
        centers, lower = review.conservative_clearance(labels, (1., 1., 1.))
        self.assertEqual(centers[10, 10, 10], 11.)
        self.assertAlmostEqual(lower[10, 10, 10], 11 - np.sqrt(3))
        self.assertLess(lower[0, 10, 10], 1.)
        labels[10, 10, 12] = 0
        nearer, _ = review.conservative_clearance(labels, (1., 1., 1.))
        self.assertEqual(nearer[10, 10, 10], 2.)

    # Purpose: Verify clearance respects KJI array order.
    # Inputs: 9-cube mask with an I-axis background neighbor and spacing [2,3,4].
    # Outputs: Physical clearance at the center is 2 mm.
    def test_anisotropic_axis_order_is_kji(self):
        labels = np.ones((9, 9, 9), dtype=np.int32)
        labels[4, 4, 5] = 0
        centers, _ = review.conservative_clearance(labels, (2., 3., 4.))
        self.assertEqual(centers[4, 4, 4], 2.)

    # Purpose: Verify growth envelopes use physical Euclidean distance.
    # Inputs: Single-cell ROI, spacing [1,1,2], radius 1.5, zero radius, and empty ROI.
    # Outputs: In-plane diagonal is included, K neighbor excluded, zero radius is identity, and empty ROI fails.
    def test_growth_envelope_expands_native_roi_with_euclidean_metric(self):
        roi = np.zeros((7, 7, 7), dtype=bool)
        roi[3, 3, 3] = True
        expanded = review.growth_envelope(roi, [1., 1., 2.], 1.5)
        self.assertTrue(expanded[3, 4, 4])
        self.assertFalse(expanded[4, 3, 3])
        np.testing.assert_array_equal(review.growth_envelope(roi, [1., 1., 2.], 0.), roi)
        with self.assertRaises(ValueError):
            review.growth_envelope(np.zeros_like(roi), [1., 1., 2.], 1.)

    # Purpose: Verify cumulative dilation budget includes sweep maxima and iterations.
    # Inputs: Two dilation steps, one erosion, and distance/iteration sweeps.
    # Outputs: Budget is 15 mm and includes only dilation steps a and c.
    def test_budget_combines_max_distance_and_pass_count_for_every_dilation(self):
        config = {'recipe': {'steps': ['a', 'b', 'c']}, 'edits': {
            'a': {'operation': 'dilation', 'distance_mm': 2., 'iterations': 1},
            'b': {'operation': 'erosion', 'distance_mm': 99., 'iterations': 8},
            'c': {'operation': 'dilation', 'distance_mm': 1.5, 'iterations': 2}},
            'sweeps': {'a': {'distance_mm': (1., 4.), 'iterations': (1, 3)}}}
        value, steps = review.cumulative_dilation_budget(config)
        self.assertEqual(value, 15.)
        self.assertEqual([row['step_name'] for row in steps], ['a', 'c'])

    # Purpose: Verify safety margins use conservative bounds and preserve empty uncertainty.
    # Inputs: Clearance centers [4,8], lower bounds [3,7], and a 3 mm margin.
    # Outputs: One voxel is within margin; empty selection returns no fabricated lower-bound distance.
    def test_margin_uses_lower_bound_and_empty_mask_never_fabricates_distance(self):
        centers = np.asarray([[[4., 8.]]])
        lower = np.asarray([[[3., 7.]]])
        mask = np.ones_like(centers, dtype=bool)
        result = review._measure(mask, centers, lower, np.ones_like(centers), 3.)
        self.assertEqual(result['voxels_within_margin'], 1)
        empty = review._measure(~mask, centers, lower, np.ones_like(centers), 3.)
        self.assertIsNone(empty['conservative_surface_clearance_lower_bound_mm'])

    # Purpose: Verify a failed case does not cancel remaining project evidence.
    # Inputs: Two-case project with one inspect_case failure and one valid result.
    # Outputs: Review completes with one failure, saves the second case, and refuses output replacement.
    def test_failed_case_is_recorded_and_does_not_cancel_remaining_evidence(self):
        with tempfile.TemporaryDirectory() as temporary:
            root = Path(temporary)
            project = root / 'project.json'
            project.write_text(json.dumps({'schema_version': 'fakect.cohort-project/1',
                'cases': [{'case_id': 'a', 'config': 'a.ini'}, {'case_id': 'b', 'config': 'b.ini'}]}))
            output = root / 'review.json'
            passed = {'case_id': 'b', 'status': 'ok', 'interior_margin_supported': True}
            with patch.object(review, 'inspect_case', side_effect=[ValueError('crop too small'), passed]), redirect_stdout(io.StringIO()):
                result = review.collect_project(project, output, repo_root=root)
            self.assertTrue(result['complete'])
            self.assertEqual(result['failed_case_count'], 1)
            self.assertEqual(result['cases'][0]['status'], 'failed')
            self.assertFalse(result['cases'][0]['interior_margin_supported'])
            self.assertEqual(result['cases'][1]['case_id'], 'b')
            self.assertEqual(json.loads(output.read_text()), result)
            with self.assertRaises(FileExistsError):
                review.collect_project(project, output, repo_root=root)


if __name__ == '__main__':
    unittest.main()
