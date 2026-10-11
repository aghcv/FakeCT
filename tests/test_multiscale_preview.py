from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
from fakect_multiscale_preview import preview


class MultiscalePreviewTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.path = Path(self.tmp.name) / 'preview.png'
        self.row = {'sample_id': 'fixture:baseline', 'split': 'validation'}
        k, j, i = np.indices((20, 18, 16))
        self.ov = {'image': (.1 + .01*k + .002*j + .001*i).astype(np.float32),
                   'spacing_ijk_mm': [1.3, 2.1, 3.7]}
        self.target = [.5, .5, .5, .4, .3, .2]
        self.predicted = [.55, .45, .52, .35, .35, .25]
        # First production-style list proposal intersects every displayed plane.
        # Second lies outside the target axial slice and partly outside the FOV.
        self.proposals = [[.5, .5, .5, .3, .4, .4], [1.05, .6, .5, .2, .4, .4]]

    def check_png(self):
        self.assertGreater(self.path.stat().st_size, 1000)
        self.assertEqual(self.path.read_bytes()[:8], b'\x89PNG\r\n\x1a\n')
        import matplotlib.pyplot as plt
        self.assertEqual(plt.get_fignums(), [])

    def test_renders_production_list_proposals_at_unchanged_absolute_coordinates(self):
        result = preview(self.row, self.ov, self.target, self.path,
                         predicted=self.predicted, proposal_boxes=self.proposals)
        self.check_png()
        self.assertEqual(result['slice_indices_kji'], [10, 9, 8])
        self.assertEqual(len(result['overlays']), 4)
        prediction = result['overlays'][1]
        np.testing.assert_allclose(prediction['box_kji'], self.predicted)
        extent = np.array([20, 18, 16]) * np.array([3.7, 2.1, 1.3])
        np.testing.assert_allclose(prediction['lower_kji_mm'],
                                   (np.array(self.predicted[:3]) - np.array(self.predicted[3:])/2) * extent)
        outside = result['overlays'][-1]
        self.assertFalse(outside['intersects_slice_kji'][0])
        self.assertGreater(outside['box_kji'][0], 1)
        np.testing.assert_allclose(outside['box_kji'], self.proposals[-1])

    def test_renders_numpy_arrays_without_ambiguous_truth_value(self):
        result = preview(self.row, self.ov, np.array(self.target)[None], self.path,
                         predicted=np.array(self.predicted)[None], proposal_boxes=np.array(self.proposals))
        self.check_png()
        self.assertEqual([r['label'] for r in result['overlays']],
                         ['Supervision', 'Prediction', 'Detail block 1', 'Detail block 2'])

    def test_optional_and_empty_proposals_render(self):
        for proposals in [None, [], np.empty((0, 6))]:
            with self.subTest(proposals=proposals):
                result = preview(self.row, self.ov, self.target, self.path, proposal_boxes=proposals)
                self.check_png()
                self.assertEqual(len(result['overlays']), 1)

    def test_single_proposal_and_off_fov_prediction_are_allowed(self):
        prediction = [-.1, .5, .5, .1, .2, .2]
        result = preview(self.row, self.ov, self.target, self.path,
                         predicted=prediction, proposal_boxes=self.proposals[0])
        self.check_png()
        self.assertFalse(result['overlays'][1]['intersects_slice_kji'][0])
        self.assertEqual(result['overlays'][1]['box_kji'], prediction)
        self.assertEqual(len(result['overlays']), 3)

    def test_invalid_geometry_fails_before_writing_a_figure(self):
        bad = [[.5, .5, .5, -.1, .2, .2], [.5, .5, .5, .1, 0, .2],
               [.5, .5, float('nan'), .1, .2, .2], [.5, .5, .5, .1, .2],
               [.5, .5, .5, .1, .2, float('inf')]]
        for box in bad:
            for field in ('box', 'predicted', 'proposal_boxes'):
                options = {'box': self.target, 'predicted': self.predicted, 'proposal_boxes': self.proposals}
                options[field] = [box] if field == 'proposal_boxes' else box
                with self.subTest(field=field, box=box), self.assertRaises(ValueError):
                    preview(self.row, self.ov, output=self.path, **options)
                self.assertFalse(self.path.exists())
        for spacing in [[1, 0, 2], [1, float('nan'), 2], [1, 2]]:
            with self.assertRaises(ValueError):
                preview(self.row, {**self.ov, 'spacing_ijk_mm': spacing}, self.target, self.path)
            self.assertFalse(self.path.exists())
        with self.assertRaises(ValueError):
            preview(self.row, {**self.ov, 'image': np.zeros((2, 3))}, self.target, self.path)
        self.assertFalse(self.path.exists())


if __name__ == '__main__':
    unittest.main()
