"""Native crop limits stay consistent across editing, ancestry and review."""
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_limits import MAX_NATIVE_EDIT_VOXELS
from fakect_growth import initial_origins, lineage_selection, expanded_region
from fakect_direction import direction_weight_field
from fakect_arc_profile import build_arc_profile, SERIES
from fakect_frame_preview import _target_context


class NativeEditLimitTests(unittest.TestCase):
    def fixture(self, shape):
        nodes = ((10., 5., 10.), (14., 8., 10.), (14., 12., 10.), (10., 15., 10.))
        resolved = {'roi_kind': 'tube', 'roi_nodes_ijk': nodes,
                    'spacing_ijk_mm': (1., 1., 1.), 'crop_low_ijk': (0, 0, 0)}
        config = {'edit': {'direction': 'outer', 'angular_width_deg': 180.},
                  'centerline': {'smoothing_mm': 0., 'sample_step_mm': 1., 'min_curvature_per_mm': .002}}
        mask = np.zeros(shape, dtype=bool)
        mask[10, 10, 16] = True
        return mask, resolved, config

    # Purpose: Verify native consumers share the expanded edit-crop allowance.
    # Inputs: 129x128x128 Boolean crop with one target and curved tube direction reference.
    # Outputs: Lineage/region identity, one reliable direction weight, unit profile volume, and target mesh.
    def test_crop_above_old_two_million_limit_keeps_lineage_direction_and_profile(self):
        mask, resolved, config = self.fixture((129, 128, 128))
        self.assertGreater(mask.size, 2_000_000)
        self.assertLess(mask.size, MAX_NATIVE_EDIT_VOXELS)
        origins = initial_origins(mask)
        np.testing.assert_array_equal(lineage_selection(origins, mask), mask)
        np.testing.assert_array_equal(expanded_region(mask, 0., (1., 1., 1.)), mask)
        del origins
        field = direction_weight_field(mask.shape, resolved, config, mask)
        self.assertEqual(field['summary']['roi_voxels'], 1)
        self.assertGreater(field['direction_weight'][10, 10, 16], .9)
        self.assertEqual(np.count_nonzero(field['direction_weight']), 1)
        del field
        empty = np.zeros(mask.shape, dtype=bool)
        edit = {key: empty for key in SERIES.values()}
        edit['target_mask_before'] = edit['target_mask_after'] = mask
        profile = build_arc_profile(edit, resolved, config, mask)['arc_profile']
        for name in ('before', 'after'):
            self.assertEqual(sum(profile['volume_mm3'][name]) + sum(profile['endpoint_volume_mm3'][name]), 1.)
        vertices, faces, _, _ = _target_context({'candidates': mask}, resolved, {'preview': {'volume_stride': 4}})
        self.assertGreater(len(vertices), 0)
        self.assertGreater(len(faces), 0)

    # Purpose: Verify oversized crops fail consistently before geometry processing.
    # Inputs: 184x128x128 target crop above the shared native budget.
    # Outputs: ValueError from lineage, expansion, direction, profile, and frame-context consumers.
    def test_crop_over_shared_limit_rejected_before_native_geometry_work(self):
        mask, resolved, config = self.fixture((184, 128, 128))
        self.assertGreater(mask.size, MAX_NATIVE_EDIT_VOXELS)
        with self.assertRaisesRegex(ValueError, 'bounded native Boolean crop'):
            initial_origins(mask)
        with self.assertRaisesRegex(ValueError, 'bounded native Boolean crop'):
            expanded_region(mask, 0., (1., 1., 1.))
        with self.assertRaisesRegex(ValueError, 'bounded native Boolean crop'):
            direction_weight_field(mask.shape, resolved, config, mask)
        with self.assertRaisesRegex(ValueError, '3,000,000'):
            build_arc_profile({}, resolved, config, mask)
        with self.assertRaisesRegex(ValueError, '3,000,000'):
            _target_context({'candidates': mask}, resolved, {})

    # Purpose: Verify native editing and visualization limits use one shared budget.
    # Inputs: Exported morphology/direction/growth/frame/overlay limit constants.
    # Outputs: All aliases equal MAX_NATIVE_EDIT_VOXELS at 3000000.
    def test_public_limit_aliases_agree_for_all_native_consumers(self):
        from fakect_morphology import MAX_MORPHOLOGY_VOXELS
        from fakect_direction import MAX_DIRECTION_VOXELS
        from fakect_growth import MAX_VOXELS
        from fakect_frame_preview import MAX_SOURCE_VOXELS as FRAME_LIMIT
        from fakect_surface_overlay import MAX_SOURCE_VOXELS as OVERLAY_LIMIT
        self.assertEqual(MAX_NATIVE_EDIT_VOXELS, 3_000_000)
        self.assertEqual({MAX_MORPHOLOGY_VOXELS, MAX_DIRECTION_VOXELS, MAX_VOXELS,
                          FRAME_LIMIT, OVERLAY_LIMIT}, {MAX_NATIVE_EDIT_VOXELS})

    # Purpose: Verify training provenance pins the shared native-budget implementation.
    # Inputs: Legacy/recipe fixtures with fakect_limits.py hash mocked to change.
    # Outputs: Both code provenance maps contain the limits file and change with its digest.
    def test_legacy_and_recipe_fingerprints_bind_shared_budget_source(self):
        import fakect_training_data
        import fakect_recipe_training
        import test_training_data
        import test_recipe_training
        for kind, fixture_class, module, function in (
                ('legacy', test_training_data.TrainingDataTests, fakect_training_data, fakect_training_data._provenance),
                ('recipe', test_recipe_training.RecipeTrainingTests, fakect_recipe_training, fakect_recipe_training.recipe_provenance)):
            with self.subTest(kind=kind):
                fixture = fixture_class()
                fixture.setUp()
                self.addCleanup(fixture.doCleanups)
                before = function(fixture.config, fixture.resolved)
                self.assertIn('fakect_limits.py', before['code_sha256'])
                original_hash = module._file_hash
                def changed_hash(path):
                    return 'changed-native-budget' if Path(path).name == 'fakect_limits.py' else original_hash(path)
                with patch.object(module, '_file_hash', side_effect=changed_hash):
                    after = function(fixture.config, fixture.resolved)
                self.assertNotEqual(before['code_sha256'], after['code_sha256'])


if __name__ == '__main__':
    unittest.main()
