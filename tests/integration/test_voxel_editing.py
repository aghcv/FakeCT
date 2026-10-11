"""Native file → crop → serial edits on portable CT-like voxel configurations."""
from copy import deepcopy
import hashlib
import json
from pathlib import Path
import sys
import tempfile
import unittest

import numpy as np
from scipy import ndimage
from scipy.spatial import cKDTree

ROOT = Path(__file__).resolve().parents[2]
sys.path[:0] = [str(ROOT/'src'), str(ROOT/'tests')]
from voxel_fixture import write_voxel_fixture
from fakect_study_config import load_study_config
from fakect_roi import prepare_crop, resolve_preview
from fakect_recipe import apply_recipe


def source_crop(value, resolved):
    low, high = resolved['crop_low_ijk'], resolved['crop_high_ijk_exclusive']
    return value[low[2]:high[2], low[1]:high[1], low[0]:high[0]]


class VoxelEditingIntegrationTests(unittest.TestCase):
    def fixture(self, size=64):
        temp = tempfile.TemporaryDirectory()
        self.addCleanup(temp.cleanup)
        fixture = write_voxel_fixture(temp.name, size=size)
        config = load_study_config(fixture['cohort_ini'])
        resolved = resolve_preview(config)
        arrays, sources = prepare_crop(resolved, config)
        return fixture, config, resolved, arrays, sources

    def assert_inputs_unchanged(self, fixture):
        for path, expected in fixture['input_hashes'].items():
            self.assertEqual(hashlib.sha256(path.read_bytes()).hexdigest(), expected, str(path))

    def test_two_resolutions_preserve_signed_native_channels_and_crop_geometry(self):
        volumes = []
        families = set()
        for size, payload_bytes in ((64, 2*1024**2), (96, 7077888)):
            with self.subTest(size=size):
                fixture, config, resolved, arrays, sources = self.fixture(size)
                families.add(config['train']['anatomy_family'])
                self.assertEqual(sum(p.stat().st_size for p in fixture['source_paths'].values()), payload_bytes)
                self.assertEqual(fixture['labels'].shape, (size,)*3)
                self.assertEqual(fixture['labels'].dtype, np.dtype('<f4'))
                self.assertEqual(len(set(fixture['spacing_ijk_mm'])), 3)
                self.assertTrue(any(c != round(c) for node in fixture['nodes_ijk'] for c in node))
                self.assertTrue(all(c > 0 for c in resolved['crop_low_ijk']))
                np.testing.assert_array_equal(arrays['act'], source_crop(fixture['labels'], resolved))
                np.testing.assert_array_equal(arrays['atn'], source_crop(fixture['attenuation_per_pixel'], resolved))
                np.testing.assert_allclose(arrays['attenuation_cm_inverse'],
                                           source_crop(fixture['attenuation_cm_inverse'], resolved), rtol=2e-7)
                for code, category in ((-7,5),(8,5),(9,5),(-10,6),(20,1),(21,1),(30,2),(40,10),(99,255)):
                    selected = arrays['act'] == code
                    if selected.any():
                        self.assertTrue(np.all(arrays['tissue'][selected] == category))
                self.assertTrue(np.all(arrays['selected'] == (arrays['candidates'] & arrays['roi'])))
                self.assertFalse((source_crop(fixture['masks']['neighbor'],resolved) & arrays['selected']).any())
                audit_case = json.loads(fixture['audit_path'].read_text())['cases'][0]
                self.assertEqual(audit_case['crop_origin_ijk'], list(fixture['crop_origin_ijk']))
                self.assertEqual(audit_case['original_shape_kji'], list(fixture['original_shape_kji']))
                self.assertEqual(set(sources), {'act','atn'})
                volumes.append(fixture['masks']['main'].sum()*np.prod(fixture['spacing_ijk_mm']))
                self.assert_inputs_unchanged(fixture)
        # Independent native samplings describe one physical configuration.
        self.assertEqual(families, {'analytic-voxel-configuration'})
        self.assertLess(abs(volumes[1]-volumes[0])/volumes[0], .08)

    def test_dilation_exits_selector_and_preserves_neighbor_and_resistant_obstacles(self):
        fixture, config, resolved, arrays, _ = self.fixture()
        events = []
        apply_recipe(arrays, resolved, config, on_step=events.append)
        event = events[0]
        result, before = event['result'], event['before_arrays']
        added = result['added_mask']
        self.assertGreater(added.sum(), 100)
        self.assertGreater((added & ~arrays['roi']).sum(), 0)
        self.assertFalse((before['selected'] & source_crop(fixture['masks']['neighbor'],resolved)).any())
        for name in ('neighbor','bone','skin'):
            mask = source_crop(fixture['masks'][name], resolved)
            np.testing.assert_array_equal(result['edited_labels'][mask], before['act'][mask], err_msg=name)
            np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][mask], before['atn'][mask], err_msg=name)
        for name in ('bone','skin'):
            mask = source_crop(fixture['masks'][name], resolved)
            self.assertGreater((result['proposed_added_mask'] & mask).sum(), 0, name)
            self.assertGreater((result['blocked_mask'] & mask).sum(), 0, name)
        owners = result['target_seed_index']
        self.assertTrue(np.all(owners[~added] == -1))
        self.assertTrue(before['selected'].ravel()[owners[added]].all())
        np.testing.assert_array_equal(result['edited_labels'][added], before['act'].ravel()[owners[added]])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][added], before['atn'].ravel()[owners[added]])
        self.assert_inputs_unchanged(fixture)

    def test_erosion_reassigns_compatible_labels_and_their_native_attenuation(self):
        fixture, config, resolved, arrays, _ = self.fixture()
        events = []
        apply_recipe(arrays, resolved, config, on_step=events.append)
        before, result = events[1]['before_arrays'], events[1]['result']
        removed = result['removed_mask']
        self.assertGreater(removed.sum(), 0)
        self.assertFalse(result['unresolved_mask'].any())
        assigned = set(map(int, np.unique(result['edited_labels'][removed])))
        self.assertTrue(assigned <= {20,21,22,-10,99})
        self.assertTrue({20,21} <= assigned)
        # Verify assignment independently using known physical signals and
        # nearby INPUT-STATE recipient coordinates, not the engine's owners.
        spacing_kji = np.asarray(resolved['spacing_ijk_mm'])[::-1]
        for code in assigned:
            mask = removed & (result['edited_labels'] == code)
            expected_signal = np.float32(fixture['label_signals_cm_inverse'][code]) * np.float32(resolved['spacing_ijk_mm'][0]/10)
            np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][mask], np.full(mask.sum(), expected_signal))
            tree = cKDTree(np.argwhere(before['act'] == code) * spacing_kji)
            nearest = tree.query(np.argwhere(mask) * spacing_kji)[0]
            self.assertTrue(np.all(nearest <= config['reassignment']['max_distance_mm'] + 1e-8))
        untouched = ~result['changed_mask']
        np.testing.assert_array_equal(result['edited_labels'][untouched], before['act'][untouched])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][untouched], before['atn'][untouched])
        self.assert_inputs_unchanged(fixture)

    def test_serial_recipe_keeps_state_chain_net_masks_and_source_bytes(self):
        fixture, config, resolved, arrays, _ = self.fixture()
        snapshot = {k:v.copy() for k,v in arrays.items()}
        events = []
        result = apply_recipe(arrays, resolved, config, on_step=events.append)
        self.assertEqual([e['step_name'] for e in events], ['expansion','local_narrowing'])
        np.testing.assert_array_equal(events[1]['before_arrays']['act'], events[0]['result']['edited_labels'])
        np.testing.assert_array_equal(events[1]['before_arrays']['atn'], events[0]['result']['attenuation_proxy_per_pixel'])
        changed = result['edited_labels'] != arrays['act']
        np.testing.assert_array_equal(result['changed_mask'], changed)
        np.testing.assert_array_equal(result['added_mask'], result['target_mask_after'] & ~result['target_mask_before'])
        np.testing.assert_array_equal(result['removed_mask'], result['target_mask_before'] & ~result['target_mask_after'])
        np.testing.assert_array_equal(result['edited_labels'][~changed], arrays['act'][~changed])
        np.testing.assert_array_equal(result['attenuation_proxy_per_pixel'][~changed], arrays['atn'][~changed])
        for key, original in snapshot.items():
            np.testing.assert_array_equal(arrays[key], original, err_msg=key)
        self.assert_inputs_unchanged(fixture)

    def test_local_safeguard_backs_off_without_untouched_parent_diluting_the_floor(self):
        fixture, config, resolved, arrays, _ = self.fixture()
        config = deepcopy(config)
        config['recipe']['steps'] = ('local_narrowing',)
        edit = config['edits']['local_narrowing']
        edit.update(distance_mm=2.5, min_volume_ratio=.4, preserve_connectivity=True)
        guarded_events = []
        guarded = apply_recipe(arrays, resolved, config, on_step=guarded_events.append)
        ordinary = deepcopy(config)
        ordinary['edits']['local_narrowing'].update(min_volume_ratio=0., preserve_connectivity=False)
        ordinary_events = []
        unrestricted = apply_recipe(arrays, resolved, ordinary, on_step=ordinary_events.append)
        event = guarded_events[0]
        scope = event['result']['erosion_guard_scope_mask']
        selected = event['before_arrays']['selected']
        np.testing.assert_array_equal(scope, selected)
        self.assertLess(scope.sum(), arrays['selected'].sum()/3)
        unrestricted_local = (unrestricted['target_mask_after'] & scope).sum()/scope.sum()
        unrestricted_whole = unrestricted['target_mask_after'].sum()/arrays['selected'].sum()
        self.assertLess(unrestricted_local, .4)
        self.assertGreater(unrestricted_whole, .4)
        retained = (guarded['target_mask_after'] & scope).sum()/scope.sum()
        self.assertGreaterEqual(retained, .4)
        self.assertGreater(guarded['removed_mask'].sum(), 0)
        self.assertLess(guarded['removed_mask'].sum(), unrestricted['removed_mask'].sum())
        guard = event['result']['summary']['erosion_safeguard']
        self.assertLess(guard['accepted_distance_mm'], guard['requested_distance_mm'])
        self.assertTrue(guard['connectivity_ok'])
        self.assertEqual(ndimage.label(guarded['target_mask_after'])[1], ndimage.label(arrays['selected'])[1])
        self.assert_inputs_unchanged(fixture)


if __name__ == '__main__':
    unittest.main()
