import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))
import fakect_context_cached_evaluation as cached
import fakect_context_learning as learning
from fakect_context_volume import ContextVolume


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class NumpyNetwork:
    """A deterministic image model with patch-wide context and distant FPs."""
    def __init__(self, bias=-2.):
        self.calls = 0
        self.bias = bias

    def __call__(self, image, training=False):
        self.calls += 1
        logits = (image * 6 + image.mean() + self.bias).astype(np.float32)
        probability = (1 / (1 + np.exp(-logits))).astype(np.float32)
        return probability, logits


class CachedEvaluationTests(unittest.TestCase):
    def setUp(self):
        tmp = tempfile.TemporaryDirectory()
        self.addCleanup(tmp.cleanup)
        self.root = Path(tmp.name)
        self.shape = (13, 15, 17)
        self.spacing = [1., 1., 1.]
        rng = np.random.default_rng(45)
        native = rng.uniform(.01, .04, self.shape).astype(np.float32)
        native[2:8, 2:8, 2:8] += .21
        native[11:, 13:, 15:] = .4  # Foreground-like material far from target.
        self.source = self.root / 'base.bin'
        (native / 10).astype('<f4').tofile(self.source)
        self.native = np.fromfile(self.source, '<f4').reshape(self.shape) * np.float32(10.)
        self.before = np.zeros(self.shape, np.uint8)
        self.before[2:8, 2:8, 2:8] = 1
        self.settings = {**learning.DEFAULTS, 'patch_size_kji': [4, 4, 4],
                         'target_spacing_mm': self.spacing}
        self.mock_inference = patch.object(learning, '_inference_model', side_effect=lambda model: model)
        self.mock_inference.start()
        self.addCleanup(self.mock_inference.stop)

    def row(self, name, *, after=None, before=None, image=None,
            edit_box=((2, 2, 2), (8, 8, 8)), mask_box=((1, 1, 1), (9, 9, 9)),
            source=None, case='fixture'):
        before = self.before if before is None else before
        after = before if after is None else after
        image = self.native if image is None else image
        sid = f'{case}:{name}'
        variant, maskfile = self.root / f'{sid}-image.npz', self.root / f'{sid}-mask.npz'
        edit_slice = tuple(slice(a, b) for a, b in zip(*edit_box))
        mask_slice = tuple(slice(a, b) for a, b in zip(*mask_box))
        b, a = before[mask_slice], after[mask_slice]
        np.savez_compressed(variant, image=image[edit_slice])
        np.savez_compressed(maskfile, before=b, after=a, added=(a & ~b), removed=(b & ~a),
                            crop_origin_ijk=np.array(mask_box[0][::-1]),
                            spacing_ijk_mm=self.spacing, sample_id=np.array(sid))
        source = self.source if source is None else source
        stat = source.stat()
        return dict(sample_id=sid, case_id=case, anatomy_family=f'{case}-family', split='validation',
                    baseline=name == 'baseline', source_atn=dict(path=str(source), bytes=stat.st_size,
                    mtime_ns=stat.st_mtime_ns), source_shape_kji=list(self.shape), spacing_ijk_mm=self.spacing,
                    variant_npz=str(variant), variant_npz_sha256=digest(variant),
                    edit_crop_origin_ijk=list(edit_box[0][::-1]),
                    edit_crop_shape_kji=(np.array(edit_box[1])-edit_box[0]).tolist(),
                    mask_npz=str(maskfile), mask_npz_sha256=digest(maskfile))

    def assert_exact(self, rows, settings=None):
        settings = self.settings if settings is None else settings
        model = NumpyNetwork()
        loaded, events = set(), []
        results = cached.score_rows(model, rows, settings, loaded, events.append)
        cached_calls = model.calls
        self.assertEqual(loaded, {row['sample_id'] for row in rows})
        for row, result in zip(rows, results):
            with ContextVolume(row) as volume:
                expected = learning.score_volume(NumpyNetwork(), volume, settings)
            for key, value in expected.items():
                self.assertEqual(result[key], value, key)
        self.assertEqual(cached_calls, sum(r['evaluation_reuse']['inferred_tiles'] for r in results))
        self.assertTrue(any(e['event'] == 'sample_complete' for e in events))
        self.assertEqual(learning.summarize(results)['sample_macro']['dice'],
                         np.mean([r['metrics']['dice'] for r in results]))
        return results

    def test_unchanged_variant_reuses_every_tile_and_full_fov_false_positives(self):
        rows = [self.row('baseline'), self.row('unchanged')]
        records = self.assert_exact(rows)
        self.assertEqual(records[1]['evaluation_reuse']['inferred_tiles'], 0)
        self.assertGreater(records[0]['metrics']['false_positive_mm3'], 0)
        self.assertEqual(records[0]['metrics']['valid_voxels'], np.prod(self.shape))
        self.assertEqual(records[0]['tiles'], 80)

    def test_dilation_and_different_crop_origins_are_physically_compared(self):
        after = self.before.copy()
        after[7:10, 4:6, 4:6] = 1  # Crosses baseline compact mask boundary.
        image = self.native.copy()
        image[7:10, 4:6, 4:6] = .3
        baseline = self.row('baseline')
        changed = self.row('dilation', after=after, image=image,
                           edit_box=((0, 1, 1), (11, 9, 9)), mask_box=((0, 0, 0), (11, 10, 10)))
        records = self.assert_exact([changed, baseline])  # Preserve caller order.
        reuse = records[0]['evaluation_reuse']
        self.assertEqual(reuse['proof']['status'], 'proven')
        self.assertGreater(reuse['reused_tiles'], 0)
        self.assertGreater(reuse['inferred_tiles'], 0)
        self.assertEqual(reuse['proof']['native_target_difference_box_kji'], [[8, 4, 4], [10, 6, 6]])

    def test_erosion_and_image_differences_both_invalidate_tiles(self):
        after = self.before.copy()
        after[3, 3:5, 3:5] = 0
        image = self.native.copy()
        image[9, 9, 9] = .4  # Separate image change with unchanged target.
        records = self.assert_exact([self.row('baseline'), self.row('two_regions', after=after,
            image=image, edit_box=((2, 2, 2), (10, 10, 10)))])
        proof = records[1]['evaluation_reuse']['proof']
        self.assertEqual(proof['native_image_difference_box_kji'], [[9, 9, 9], [10, 10, 10]])
        self.assertEqual(proof['native_target_difference_box_kji'], [[3, 3, 3], [4, 5, 5]])

    def test_interpolation_halo_recomputes_adjacent_tile(self):
        image = self.native.copy()
        image[6, 6, 2] = .4
        rows = [self.row('baseline'), self.row('halo', image=image)]
        settings = {**self.settings, 'target_spacing_mm': [.5, .5, .5]}
        self.assert_exact(rows, settings)
        with ContextVolume(rows[0]) as base, ContextVolume(rows[1]) as sibling:
            _, support = cached.prove_difference(base, sibling)
        self.assertTrue(cached._affected((12, 12, 0), [4, 4, 4], [.5, .5, .5], self.spacing, support))
        self.assertFalse(cached._affected((0, 0, 0), [4, 4, 4], [.5, .5, .5], self.spacing, support))

    def test_shifted_equivalent_crop_reuses_all_tiles(self):
        rows = [self.row('baseline'), self.row('shifted',
                edit_box=((0, 0, 0), (10, 10, 10)), mask_box=((0, 0, 0), (10, 10, 10)))]
        result = self.assert_exact(rows)[1]
        self.assertEqual(result['evaluation_reuse']['inferred_tiles'], 0)

    def test_affine_arithmetic_boundary_is_conservatively_recomputed(self):
        # For an aligned tile the two arrangements differ at native 122:
        # global endpoint=121.99999999999999; affine endpoint=122.0.
        origin, width, scale = 84, 4, 1.4
        global_end = (origin + width - .5) * scale - .5
        native_lo = np.floor((origin + .5) * scale - .5)
        affine_end = ((origin + .5) * scale - .5 - native_lo
                      + (width - 1) * scale + native_lo)
        self.assertLess(global_end, affine_end)
        support = (np.array([122., 122., 122.]), np.array([124., 124., 124.]))
        self.assertTrue(cached._affected([84, 84, 84], [4, 4, 4], [1.4]*3, [1.]*3, support))
        # The margin does not swallow a real geometric separation.
        support = (np.array([122.001]*3), np.array([124.]*3))
        self.assertFalse(cached._affected([84, 84, 84], [4, 4, 4], [1.4]*3, [1.]*3, support))

    def test_source_identity_mismatch_falls_back_without_assuming_same_content(self):
        copied = self.root / 'different-source.bin'
        copied.write_bytes(self.source.read_bytes())
        rows = [self.row('baseline'), self.row('different-source', source=copied)]
        reuse = self.assert_exact(rows)[1]['evaluation_reuse']
        self.assertEqual(reuse['proof']['status'], 'recompute_all')
        self.assertEqual(reuse['reused_tiles'], 0)

    def test_source_geometry_mismatch_falls_back(self):
        rows = [self.row('baseline'), self.row('geometry')]
        # Same byte count but a different interpretation of the source axes.
        rows[1]['source_shape_kji'] = [15, 13, 17]
        reuse = self.assert_exact(rows)[1]['evaluation_reuse']
        self.assertEqual(reuse['proof']['status'], 'recompute_all')
        self.assertEqual(reuse['reused_tiles'], 0)

    def test_changed_physical_before_ref_falls_back(self):
        before = self.before.copy()
        before[2, 2, 2] = 0
        rows = [self.row('baseline'), self.row('other-reference', before=before)]
        reuse = self.assert_exact(rows)[1]['evaluation_reuse']
        self.assertEqual(reuse['proof']['reason'], 'Physical before masks differ')
        self.assertEqual(reuse['reused_tiles'], 0)

    def test_budget_failure_falls_back(self):
        rows = [self.row('baseline'), self.row('same')]
        with patch.object(cached, 'MAX_COMPARISON_VOXELS', 1):
            result = self.assert_exact(rows)[1]
        self.assertEqual(result['evaluation_reuse']['reused_tiles'], 0)

    def test_corrupt_payload_raises_instead_of_falling_back(self):
        rows = [self.row('baseline'), self.row('bad')]
        rows[1]['mask_npz_sha256'] = '0' * 64
        with self.assertRaisesRegex(ValueError, 'Checksum changed'):
            cached.score_rows(NumpyNetwork(), rows, self.settings)

    def test_cache_is_fresh_for_each_model_and_call(self):
        rows = [self.row('baseline'), self.row('same')]
        first, second = NumpyNetwork(-2), NumpyNetwork(2)
        a = cached.score_rows(first, rows, self.settings)
        b = cached.score_rows(second, rows, self.settings)
        self.assertEqual(first.calls, second.calls)
        self.assertGreater(first.calls, 0)
        self.assertNotEqual(a[0]['metrics'], b[0]['metrics'])

    def test_cases_never_share_cache(self):
        rows = [self.row('baseline'), self.row('same'), self.row('baseline', case='other'),
                self.row('same', case='other')]
        results = self.assert_exact(rows)
        self.assertEqual(results[0]['evaluation_reuse']['inferred_tiles'], results[2]['tiles'])
        self.assertEqual(results[2]['evaluation_reuse']['baseline_sample_id'], 'other:baseline')

    def test_baseline_is_required_and_partition_consistent(self):
        row = self.row('no_baseline')
        with self.assertRaisesRegex(ValueError, 'exactly one baseline'):
            cached.plan_rows([row], self.settings)
        baseline = self.row('baseline')
        row['split'] = 'test'
        with self.assertRaisesRegex(ValueError, 'partitions'):
            cached.plan_rows([baseline, row], self.settings)

    def test_loader_plan_matches_actual_inference_count(self):
        image = self.native.copy()
        image[4, 4, 4] = .4
        rows = [self.row('baseline'), self.row('same'), self.row('changed', image=image)]
        plan = cached.plan_rows(rows, self.settings)
        records = self.assert_exact(rows)
        self.assertEqual(plan['inferred_tiles'], sum(r['evaluation_reuse']['inferred_tiles'] for r in records))
        self.assertEqual(plan['total_tiles'], sum(r['tiles'] for r in records))
        self.assertGreater(plan['forward_pass_reduction_factor'], 1)

    def test_anisotropic_sampling_and_padding_match_uncached(self):
        image = self.native.copy()
        image[6, 5, 4] = .4
        rows = [self.row('baseline'), self.row('changed', image=image)]
        self.assert_exact(rows, {**self.settings, 'target_spacing_mm': [.7, 1.3, 2.1]})


if __name__ == '__main__':
    unittest.main()
