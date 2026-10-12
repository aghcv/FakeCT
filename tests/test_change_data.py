"""Physical alignment and leakage boundaries of focused net-change targets."""
from copy import deepcopy
import hashlib
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / 'src'))

from fakect_change_data import FocusedChangeVolume, net_change_labels


def checksum(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


class ChangeLabelsTests(unittest.TestCase):
    # Purpose: Verify change labels describe final additions and removals.
    # Inputs: Binary masks with one added voxel, one removed voxel, and cancelled edits.
    # Outputs: Class 1 addition, class 2 removal, exact difference mask, and zero cancelled targets.
    def test_net_not_edit_history_includes_removed_outside_current_aorta(self):
        before = np.zeros((2, 3, 4), np.uint8)
        before[0, 1, 1:3] = 1
        after = before.copy()
        after[0, 1, 1] = 0
        after[0, 2, 1] = 1
        labels = net_change_labels(before, after)
        self.assertEqual(labels[0, 1, 1], 2)
        self.assertEqual(after[0, 1, 1], 0)
        self.assertEqual(labels[0, 2, 1], 1)
        self.assertEqual(labels[0, 1, 2], 0)
        self.assertEqual(np.count_nonzero(labels), 2)
        np.testing.assert_array_equal(labels > 0, before != after)
        # An expansion followed by a restoring erosion has no final difference.
        np.testing.assert_array_equal(net_change_labels(before, before), np.zeros_like(before))

    # Purpose: Verify invalid acquisition padding and malformed masks are handled explicitly.
    # Inputs: Binary masks with invalid slab, nonbinary/NaN values, and mismatched shapes.
    # Outputs: Padding suppressed and ValueError for nonbinary or unaligned masks.
    def test_invalid_and_padding_are_explicit(self):
        b = np.zeros((2, 2, 2), np.uint8)
        a = np.ones_like(b)
        valid = np.ones_like(b)
        valid[0] = 0
        np.testing.assert_array_equal(net_change_labels(b, a, valid), valid)
        for invalid in (np.full_like(a, 2), np.full(a.shape, np.nan)):
            with self.assertRaisesRegex(ValueError, 'binary'):
                net_change_labels(b, invalid)
        with self.assertRaisesRegex(ValueError, 'aligned'):
            net_change_labels(b, a[:1])


class FocusedChangeVolumeTests(unittest.TestCase):
    def setUp(self):
        self.tmp = tempfile.TemporaryDirectory()
        self.addCleanup(self.tmp.cleanup)
        self.root = Path(self.tmp.name)
        self.spacing = np.array([.8, 1.1, 1.7])
        self.native_shape = (7, 8, 9)
        kk, jj, ii = np.indices(self.native_shape)
        self.native = (.1 + .02 * kk + .003 * jj + .001 * ii).astype(np.float32)
        source = self.root / 'source.bin'
        (self.native * (self.spacing[0] / 10)).astype('<f4').tofile(source)
        self.overlay = self.native[2:5, 2:5, 3:6] + .4
        self.variant = self.root / 'variant.npz'
        self.maskfile = self.root / 'mask.npz'
        self.before = np.zeros((4, 5, 6), np.uint8)
        self.before[1:3, 2:4, 2:5] = 1
        self.after = self.before.copy()
        self.after[1, 1, 2] = 1
        self.after[2, 3, 4] = 0
        np.savez_compressed(self.variant, image=self.overlay)
        self.mask_data = dict(before=self.before, after=self.after,
            added=((self.after == 1) & (self.before == 0)).astype(np.uint8),
            removed=((self.before == 1) & (self.after == 0)).astype(np.uint8),
            crop_origin_ijk=np.array([1, 1, 1]), spacing_ijk_mm=self.spacing,
            sample_id=np.array('fixture:edited'))
        np.savez_compressed(self.maskfile, **self.mask_data)
        stat = source.stat()
        self.row = dict(sample_id='fixture:edited', case_id='fixture', anatomy_family='family', baseline=False,
            source_atn=dict(path=str(source), bytes=stat.st_size, mtime_ns=stat.st_mtime_ns),
            source_shape_kji=list(self.native_shape), spacing_ijk_mm=self.spacing.tolist(),
            variant_npz=str(self.variant), variant_npz_sha256=checksum(self.variant),
            edit_crop_origin_ijk=[3, 2, 2], edit_crop_shape_kji=[3, 3, 3],
            mask_npz=str(self.maskfile), mask_npz_sha256=checksum(self.maskfile))
        self.runtime = {'input.detail': {'shape_kji': [12, 11, 14],
            'spacing_ijk_mm': [.65, .8, 1.2], 'proposals': 1},
            'model': {'clip_min': 0., 'clip_max': 1.}}
        self.origins = [[-1, -1, -2]]

    def open(self, row=None, runtime=None, origins=None, **kw):
        reader = FocusedChangeVolume(self.row if row is None else row,
            self.runtime if runtime is None else runtime,
            self.origins if origins is None else origins, **kw)
        self.addCleanup(reader.close)
        return reader

    def save_masks(self):
        np.savez_compressed(self.maskfile, **self.mask_data)
        self.row['mask_npz_sha256'] = checksum(self.maskfile)

    def brute_labels(self, origin, shape, spacing, before=None, after=None):
        before = self.before if before is None else before
        after = self.after if after is None else after
        expected = np.zeros(shape, np.uint8)
        valid = np.zeros(shape, np.uint8)
        for q in np.ndindex(shape):
            native = (np.asarray(q) + origin + .5) * np.asarray(spacing)[::-1] / self.spacing[::-1] - .5
            if np.any(native < -.5) or np.any(native >= np.asarray(self.native_shape) - .5):
                continue
            valid[q] = 1
            local = np.floor(native + .5).astype(int) - 1
            if np.any(local < 0) or np.any(local >= before.shape):
                continue
            b, a = before[tuple(local)], after[tuple(local)]
            expected[q] = 1 if a and not b else 2 if b and not a else 0
        return expected, valid

    # Purpose: Verify change targets align with physically resampled predictor channels.
    # Inputs: Anisotropic native/variant fixture, negative origin, and 12x11x14 detail grid.
    # Outputs: Brute-force labels/validity agreement, zero padding, and removed targets outside current mask.
    def test_anisotropic_same_grid_native_origin_and_padding(self):
        reader = self.open()
        out = reader.whole_focus(0)
        labels, valid = self.brute_labels(np.asarray(self.origins[0]),
            tuple(self.runtime['input.detail']['shape_kji']), self.runtime['input.detail']['spacing_ijk_mm'])
        np.testing.assert_array_equal(out['labels'], labels)
        np.testing.assert_array_equal(out['valid'], valid)
        np.testing.assert_array_equal(out['x'][..., 1], valid)
        self.assertTrue(np.any(labels == 1))
        self.assertTrue(np.any(labels == 2))
        self.assertFalse(np.any(out['x'][valid == 0]))
        self.assertEqual(out['x'].dtype, np.float32)
        self.assertEqual(out['labels'].dtype, np.uint8)
        self.assertEqual(out['metadata']['input_channels'], ['normalized_attenuation', 'acquisition_validity'])
        removed = out['labels'] == 2
        block = reader.volume.read_block(self.origins[0], self.runtime['input.detail']['shape_kji'],
                                       self.runtime['input.detail']['spacing_ijk_mm'])
        self.assertFalse(np.any(block['mask'][removed]))

    # Purpose: Verify target annotations cannot leak into input channels.
    # Inputs: Same volume with original change masks and then zeroed target masks.
    # Outputs: Identical predictors while change labels become zero.
    def test_targets_cannot_change_predictor_channels(self):
        a = self.open().whole_focus(0)
        self.mask_data.update(before=np.zeros_like(self.before), after=np.zeros_like(self.after),
            added=np.zeros_like(self.before), removed=np.zeros_like(self.before))
        self.save_masks()
        b = self.open().whole_focus(0)
        np.testing.assert_array_equal(a['x'], b['x'])
        self.assertTrue(np.any(a['labels']))
        self.assertFalse(np.any(b['labels']))

    # Purpose: Verify inference scheduling covers the focus without consulting targets.
    # Inputs: 5x6x7 tiles with overlap and a truth-query mock that raises.
    # Outputs: Complete coverage, reconstructed labels, and unchanged inference schedule.
    def test_inference_tiles_complete_and_target_independent(self):
        reader = self.open()
        schedule = list(reader.inference_tiles([5, 6, 7], [1, 2, 3]))
        counts = np.zeros(tuple(reader.shape_kji), np.uint16)
        out = np.zeros(tuple(reader.shape_kji), np.uint8)
        for record in schedule:
            value = reader.read_patch(**record)
            o, n = np.asarray(record['offset_kji']), np.asarray(record['shape_kji'])
            sl = tuple(slice(int(a), int(a+b)) for a,b in zip(o,n))
            counts[sl] += 1
            out[sl] = value['labels']
        self.assertTrue(np.all(counts > 0))
        np.testing.assert_array_equal(out, reader.whole_focus(0)['labels'])
        # Inference must neither query truth-centered candidates nor labels.
        with patch.object(reader, '_change_points', side_effect=AssertionError('target leakage')):
            self.assertEqual(schedule, list(reader.inference_tiles([5, 6, 7], [1, 2, 3])))

    # Purpose: Verify class-centered training sampling stays bounded.
    # Inputs: Seeded 4-cubed added/removed sampling and truth-query guard for uniform mode.
    # Outputs: Requested class in every patch, bounded reused reservoir, and target-free uniform sampling.
    def test_training_class_sampling_contains_requested_class(self):
        reader = self.open()
        rng = np.random.default_rng(8)
        for mode, code in [('added', 1), ('removed', 2)]:
            for _ in range(12):
                offset = reader.training_offset(0, [4, 4, 4], rng, mode=mode)
                self.assertTrue(np.any(reader.read_patch(0, offset, [4, 4, 4])['labels'] == code))
        # Reservoir remains bounded and is reused rather than decompressed again.
        self.assertLessEqual(reader.stats()['cached_change_point_bytes'], 2 * 4096 * 3 * 8)
        with patch.object(reader, '_change_points', side_effect=AssertionError('uniform consulted truth')):
            reader.training_offset(0, [4, 4, 4], rng, mode='uniform')
            reader.training_patch([4, 4, 4], rng, mode='uniform')

    # Purpose: Verify training chooses proposals containing the requested class.
    # Inputs: Two origins, one outside the acquisition and one containing additions/removals.
    # Outputs: Changed-class training uses proposal 1 while inference includes both proposals.
    def test_changed_training_selects_eligible_proposal_without_changing_inference(self):
        runtime = deepcopy(self.runtime)
        runtime['input.detail']['proposals'] = 2
        reader = self.open(runtime=runtime, origins=[[-30, -30, -30], self.origins[0]])
        self.assertEqual(reader.available_classes(0), ())
        self.assertEqual(reader.available_classes(1), ('added', 'removed'))
        rng = np.random.default_rng(2)
        for mode, code in [('added', 1), ('removed', 2)]:
            record = reader.training_patch([4, 4, 4], rng, mode=mode)
            self.assertEqual(record['proposal_index'], 1)
            self.assertTrue(np.any(reader.read_patch(**record)['labels'] == code))
        schedule = list(reader.inference_tiles([4, 4, 4]))
        self.assertEqual({r['proposal_index'] for r in schedule}, {0, 1})

    # Purpose: Verify native baseline controls bypass edited signals and labels.
    # Inputs: Native-grid baseline_control reader over the edited fixture.
    # Outputs: Original attenuation, zero targets, baseline metadata, and usable training offsets.
    def test_native_control_restores_original_attenuation_and_has_zero_targets(self):
        control = self.open(baseline_control=True)
        runtime = deepcopy(self.runtime)
        runtime['input.detail'].update(shape_kji=list(self.native_shape), spacing_ijk_mm=self.spacing.tolist())
        control = self.open(runtime=runtime, origins=[[0, 0, 0]], baseline_control=True)
        out = control.whole_focus(0)
        expected = (np.fromfile(self.row['source_atn']['path'], '<f4').reshape(self.native_shape)
                    * np.float32(10 / self.spacing[0]))
        np.testing.assert_allclose(out['x'][..., 0], expected, atol=1e-7)
        self.assertFalse(np.any(out['labels']))
        self.assertTrue(out['metadata']['baseline_control'])
        rng = np.random.default_rng(5)
        self.assertEqual(len(control.training_offset(0, [3, 3, 3], rng, mode='removed')), 3)

    # Purpose: Verify unchanged target masks remain valid for baseline and edited rows.
    # Inputs: Restored before/after masks with zero net changes and baseline flag toggled.
    # Outputs: Zero change labels in both real baseline and cancelled-edit views.
    def test_real_baseline_and_cancelled_edits_remain_unchanged(self):
        self.mask_data.update(after=self.before, added=np.zeros_like(self.before), removed=np.zeros_like(self.before))
        self.save_masks()
        row = deepcopy(self.row)
        row['baseline'] = True
        self.assertFalse(np.any(self.open(row=row).whole_focus(0)['labels']))
        self.assertFalse(np.any(self.open().whole_focus(0)['labels']))

    # Purpose: Verify focused readers reject invalid geometry and inconsistent baselines.
    # Inputs: Fractional origins, oversized grid, edited baseline, invalid patches, and excessive overlap.
    # Outputs: ValueError for each invalid origin, budget, baseline, patch, and tile schedule.
    def test_rejects_invalid_geometry_budget_and_nonnoop_baseline(self):
        with self.assertRaisesRegex(ValueError, 'integer'):
            self.open(origins=[[-.1, -1, -2]])
        runtime = deepcopy(self.runtime)
        runtime['input.detail']['shape_kji'] = [300, 300, 300]
        with self.assertRaisesRegex(ValueError, 'budget'):
            self.open(runtime=runtime)
        row = deepcopy(self.row)
        row['baseline'] = True
        with self.assertRaisesRegex(ValueError, 'unchanged'):
            self.open(row=row)
        reader = self.open()
        for args in [(0, [-1,0,0], [2,2,2]), (0,[0,0,0],[100,2,2]), (1,[0,0,0],[2,2,2])]:
            with self.assertRaises(ValueError):
                reader.read_patch(*args)
        with self.assertRaises(ValueError):
            list(reader.inference_tiles([3, 3, 3], [3, 0, 0]))


if __name__ == '__main__':
    unittest.main()
