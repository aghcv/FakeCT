"""Whole-volume sampling, geometry, and portable navigator contracts."""
import base64
from copy import deepcopy
import hashlib
from html.parser import HTMLParser
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import patch

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]/'src'))
import fakect_global_preview as global_preview
from fakect_global_preview import (prepare_global, read_sampled, render_global_preview,
                                   roi_definitions, sampled_roi_mask, sampling_axes)
from fakect_roi import sphere_mask, tube_mask


class _Payload(HTMLParser):
    def __init__(self, text):
        super().__init__()
        self.capture = False
        self.payload = ''
        self.external = []
        self.feed(text)

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        if tag == 'script':
            self.capture = attrs.get('id') == 'global-data'
        for key in ('src', 'href'):
            if key in attrs:
                self.external.append(attrs[key])

    def handle_endtag(self, tag):
        if tag == 'script':
            self.capture = False

    def handle_data(self, value):
        if self.capture:
            self.payload += value


class GlobalPreviewTests(unittest.TestCase):
    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.addCleanup(self.temporary.cleanup)
        self.root = Path(self.temporary.name)
        self.labels = np.zeros((13, 9, 11), dtype='<f4')
        self.labels[:, 3, 3] = 7
        self.labels[:, 5, 7] = -8
        self.labels[2:11, 2:7, 1] = 12
        self.attenuation = (np.arange(self.labels.size).reshape(self.labels.shape)/1000).astype('<f4')
        self.labels.tofile(self.root/'act.bin')
        self.attenuation.tofile(self.root/'atn.bin')
        categories = [{'id': 0, 'name': 'background'}, {'id': 5, 'name': 'artery'},
                      {'id': 3, 'name': 'muscle'}, {'id': 255, 'name': 'unknown'}]
        catalog = {'categories': categories, 'records': [
            {'original_id': key, 'original_name': name,
             'classification': {'tissue_id': group, 'tissue_name': tissue}}
            for key, name, group, tissue in [(0, 'background', 0, 'background'),
                                            (7, 'first_artery', 5, 'artery'),
                                            (-8, 'second_artery', 5, 'artery'),
                                            (12, 'muscle', 3, 'muscle')]]}
        self.resolved = {'shape_kji': self.labels.shape, 'spacing_ijk_mm': (1, 2, 3),
                         'source_files': {'act': self.root/'act.bin', 'atn': self.root/'atn.bin'},
                         'source_ids': (7, -8), 'catalog': catalog, 'catalog_sha256': 'fixture',
                         'slice_ijk': (3, 3, 6), 'crop_low_ijk': (2, 2, 4),
                         'crop_high_ijk_exclusive': (5, 5, 8)}
        self.config = {'selection': {'tissue': 'artery', 'source_ids': ()},
                       'roi': {'shape': 'tube', 'center_ijk': ((3, 3, 4), (3, 3, 8)),
                               'radius_mm': (2, 3)},
                       'rois': {'upper': {'shape': 'sphere', 'center_ijk': (3, 3, 8), 'radius_mm': 2}}}

    def test_anisotropic_sampling_is_bounded_and_retains_exact_final_edges(self):
        shape, spacing = (39, 23, 31), (0.5, 1, 2.5)
        axes, strides = sampling_axes(shape, spacing, max_voxels=700)
        self.assertLessEqual(np.prod([len(a) for a in axes]), 700)
        self.assertGreater(strides[0], strides[2])
        for axis, dimension in zip(axes, shape[::-1]):
            self.assertEqual(axis[0], 0)
            self.assertEqual(axis[-1], dimension-1)
            self.assertTrue(np.all(np.diff(axis) > 0))
        source = np.arange(np.prod(shape), dtype='<f4').reshape(shape)-500
        path = self.root/'noncubic.bin'
        source.tofile(path)
        before = path.read_bytes()
        actual, metadata = read_sampled(path, shape, axes)
        expected = source[np.ix_(axes[2], axes[1], axes[0])]
        np.testing.assert_array_equal(actual, expected)
        self.assertEqual(actual[-1, -1, -1], source[-1, -1, -1])
        self.assertEqual(metadata['sample_sha256'], hashlib.sha256(expected.tobytes()).hexdigest())
        self.assertEqual(metadata['source_bytes_read'], len(axes[2])*shape[1]*shape[2]*4)
        self.assertEqual(path.read_bytes(), before)

    def test_no_id_filter_selects_both_arteries_outside_local_crop(self):
        arrays, metadata = prepare_global(self.resolved, self.config)
        np.testing.assert_array_equal(arrays['candidates'], np.isin(self.labels, (7, -8)))
        self.assertEqual(arrays['act'].shape, self.labels.shape)
        self.assertTrue(arrays['candidates'][0, 5, 7])
        self.assertFalse(metadata['selection']['source_ids_explicit'])
        self.assertEqual(metadata['selection']['resolved_source_id_count'], 2)
        self.assertEqual(metadata['initial_crosshair_ijk'], [3, 3, 6])
        np.testing.assert_allclose(arrays['attenuation_cm_inverse'], self.attenuation/.1)
        limited = deepcopy(self.resolved)
        limited['source_ids'] = (7,)
        config = deepcopy(self.config)
        config['selection']['source_ids'] = (7,)
        arrays, metadata = prepare_global(limited, config)
        np.testing.assert_array_equal(arrays['candidates'], self.labels == 7)
        self.assertTrue(metadata['selection']['source_ids_explicit'])

    def test_sampled_geometries_match_native_masks_at_irregular_coordinates(self):
        axes = [np.array([0, 2, 3, 7, 10]), np.array([0, 3, 5, 8]), np.array([0, 4, 6, 8, 12])]
        spacing = self.resolved['spacing_ijk_mm']
        tube, sphere = roi_definitions(self.config)
        expected_tube = tube_mask(self.labels.shape, (0, 0, 0), tube['nodes_ijk'], spacing, tube['radii_mm'])
        expected_sphere = sphere_mask(self.labels.shape, (0, 0, 0), sphere['nodes_ijk'][0], spacing, sphere['radii_mm'][0])
        for definition, expected in [(tube, expected_tube), (sphere, expected_sphere)]:
            actual = sampled_roi_mask(axes, spacing, definition)
            np.testing.assert_array_equal(actual, expected[np.ix_(axes[2], axes[1], axes[0])])
        clipped = global_preview._plane_masks(axes, spacing, [tube, sphere], 2, 3)
        self.assertFalse(np.any(clipped[1] & ~clipped[0]))

    def test_changed_source_between_channel_reads_is_rejected(self):
        original = global_preview.read_sampled

        def changed(path, *args):
            result = original(path, *args)
            if path.name == 'atn.bin':
                values = self.labels.copy()
                values[0, 0, 0] = 7
                values.tofile(self.root/'act.bin')
            return result

        with patch.object(global_preview, 'read_sampled', side_effect=changed), \
                self.assertRaisesRegex(ValueError, 'changed while preparing'):
            prepare_global(self.resolved, self.config)

    def test_invalid_axes_size_and_sample_values_are_rejected(self):
        with self.assertRaisesRegex(ValueError, 'retain both edges'):
            sampling_axes((3, 4, 5), (1, 1, 1), max_voxels=7)
        with self.assertRaisesRegex(ValueError, 'sorted distinct'):
            read_sampled(self.root/'act.bin', self.labels.shape, [[0, 2, 2], [0, 1], [0, 1]])
        values = self.attenuation.copy()
        values[-1, -1, -1] = np.nan
        values.tofile(self.root/'atn.bin')
        with self.assertRaisesRegex(ValueError, 'Nonfinite'):
            prepare_global(self.resolved, self.config)
        with patch.object(global_preview, 'MAX_PLANE_BYTES', 10), \
                self.assertRaisesRegex(ValueError, 'axial plane'):
            prepare_global(self.resolved, self.config)

    def test_portable_html_embeds_exact_axes_samples_and_safe_names(self):
        config = deepcopy(self.config)
        config['rois']['upper</script><script>alert(1)</script>'] = config['rois'].pop('upper')
        before = {channel: path.read_bytes() for channel, path in self.resolved['source_files'].items()}
        output = self.root/'report'
        metadata = render_global_preview(self.resolved, config, output)
        document = (output/metadata['html']).read_text()
        parsed = _Payload(document)
        payload = json.loads(parsed.payload)
        self.assertEqual(parsed.external, [])
        self.assertNotIn('upper</script><script>', document)
        self.assertEqual(payload['axes_ijk'][0], list(range(11)))
        target = np.frombuffer(base64.b64decode(payload['target']), dtype=np.uint8).reshape(self.labels.shape)
        np.testing.assert_array_equal(target, np.isin(self.labels, (7, -8)))
        self.assertEqual(payload['rois'][1]['name'], 'upper</script><script>alert(1)</script>')
        self.assertIn('Moving this crosshair or sphere does not edit the INI', document)
        self.assertIn('Copy coordinates', document)
        self.assertNotIn('postMessage', document)
        self.assertTrue((output/metadata['figure']).read_bytes().startswith(b'\x89PNG'))
        for channel, path in self.resolved['source_files'].items():
            self.assertEqual(path.read_bytes(), before[channel])


if __name__ == '__main__':
    unittest.main()
