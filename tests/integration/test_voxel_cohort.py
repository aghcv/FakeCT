"""Portable native-volume integration checks for reviewed cohort publication.

These tests exercise production generation and registry code using temporary
64³/96³ mathematical volumes. They need neither XCAT data nor a neural-network
backend. The fixture's recorded parent-grid geometry is provenance, not a claim
that its generated pixels came from a patient or licensed XCAT phantom.
"""
from contextlib import redirect_stdout
import configparser
import hashlib
import io
import json
from pathlib import Path
import shutil
from unittest.mock import patch

import numpy as np
import pytest

from fakect_cohort_registry import (load_registry_entry, register_dataset,
                                    verify_prepared_manifest)
from fakect_study_config import load_study_config
from train_study import run
from voxel_fixture import write_voxel_fixture


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _run(path, stage, *, validate_only=False):
    with redirect_stdout(io.StringIO()):
        return run(path, stage=stage, validate_only=validate_only)


def _update_ini(path, changes):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    parser.read(path)
    for section, values in changes.items():
        parser[section].update({key: str(value) for key, value in values.items()})
    with Path(path).open('w') as stream:
        parser.write(stream)


def _file_inventory(root):
    return {path.relative_to(root): (path.stat().st_size, path.stat().st_mtime_ns)
            for path in root.rglob('*') if path.is_file()}


@pytest.fixture(scope='module')
def prepared64(tmp_path_factory):
    """One actual preview-approved → preflight → freeze → prepare publication."""
    root = tmp_path_factory.mktemp('native-cohort-64')
    fixture = write_voxel_fixture(root / 'fixture', size=64, reviewed=True)
    path = fixture['cohort_ini']
    # ROI placement is already reviewed; achieved sweep outcomes are approved
    # only after native preflight. This approval does not alter generation.
    _update_ini(path, {'train': {'parameters_reviewed': 'false'}})
    config = load_study_config(path)
    assert not config['train']['dataset_directory'].exists()
    with pytest.raises(ValueError, match='freeze'):
        _run(path, 'prepare')
    assert not config['train']['dataset_directory'].exists()
    metadata = _run(path, 'prepare', validate_only=True)
    preflight = _run(path, 'preflight')
    assert preflight['execution_valid'] and preflight['cohort_ready']
    with pytest.raises(ValueError, match='Review'):
        _run(path, 'freeze')
    assert not config['train']['freeze_directory'].exists()
    _update_ini(path, {'train': {'parameters_reviewed': 'true'}})
    frozen = _run(path, 'freeze')
    prepared = _run(path, 'prepare')
    manifest_path = Path(prepared['manifest_path'])
    manifest = verify_prepared_manifest(manifest_path)
    registry = root / 'registry'
    registered = register_dataset(registry, manifest_path, 'voxel-64-r1', 'mathematical-family')
    return {**fixture, 'root': root, 'metadata': metadata, 'preflight': preflight,
            'frozen': frozen, 'prepared': prepared, 'manifest': manifest,
            'manifest_path': manifest_path, 'registry': registry, 'registered': registered}


def test_metadata_only_sweep_reads_no_voxels_and_creates_no_outputs(tmp_path):
    fixture = write_voxel_fixture(tmp_path / 'metadata64', size=64)
    config = load_study_config(fixture['cohort_ini'])
    before = _file_inventory(tmp_path)
    with patch('fakect_roi.read_crop', side_effect=AssertionError('Metadata may not read voxels')), \
            patch('numpy.fromfile', side_effect=AssertionError('Metadata may not open binary arrays')):
        result = _run(fixture['cohort_ini'], 'prepare', validate_only=True)
    assert result['validated'] and result['variant_count'] > 1
    assert result['planned_split_counts'] == {'unassigned': result['variant_count']}
    assert result['outputs_written'] is False
    assert result['voxel_payloads_read'] is False
    assert result['native_simulations_checked'] is False
    assert _file_inventory(tmp_path) == before
    for key in ('dataset_directory', 'preflight_directory', 'freeze_directory'):
        assert not config['train'][key].exists()


def test_96_volume_local_recipe_records_both_steps_and_refuses_insufficient_halo(tmp_path):
    fixture = write_voxel_fixture(tmp_path / 'native96', size=96, reviewed=True)
    path = fixture['cohort_ini']
    config = load_study_config(path)
    result = _run(path, 'preflight')
    assert result['execution_valid'] and result['failed_variants'] == 0
    document = json.loads(Path(result['preflight_path']).read_text())
    edits = [sample for sample in document['samples'] if sample['operation'] == 'recipe']
    assert edits
    assert all([step['step_name'] for step in sample['step_outcomes']] == list(config['recipe']['steps'])
               for sample in edits)
    assert any(step['added_voxels'] > 0 for sample in edits for step in sample['step_outcomes'])
    assert any(step['removed_voxels'] > 0 for sample in edits for step in sample['step_outcomes'])
    assert not list(config['train']['preflight_directory'].rglob('*.npz'))
    assert not config['train']['dataset_directory'].exists()
    max_radius = max(config['roi']['radius_mm'])
    _update_ini(path, {'roi': {'crop_half_width_mm': max_radius}})
    before = _file_inventory(tmp_path)
    with patch('fakect_roi.read_crop', side_effect=AssertionError('Invalid halo must fail before arrays')):
        with pytest.raises(ValueError, match='halo|context'):
            _run(path, 'prepare', validate_only=True)
    assert _file_inventory(tmp_path) == before


def test_reviewed_native_pipeline_publishes_decoupled_verified_cohort(prepared64):
    data = prepared64
    preflight, manifest = data['preflight'], data['manifest']
    assert preflight['variant_count'] == data['metadata']['variant_count'] == manifest['sample_count']
    assert preflight['dataset_fingerprint'] == data['frozen']['dataset_fingerprint'] == manifest['dataset_fingerprint']
    assert manifest['split_mode'] == 'unassigned'
    assert manifest['target_scope'] == 'selected_lineage'
    assert manifest['variants_start_from_original'] is True
    assert manifest['original_sources_modified'] is False
    assert manifest['split_counts'] == {'unassigned': manifest['sample_count']}
    assert manifest['image_units'] == 'cm^-1'
    assert manifest['image_method'] == 'attenuation_copy_proxy'
    assert not (data['manifest_path'].parent / 'INCOMPLETE').exists()
    assert 'cohort-freeze.json' in manifest['artifacts_sha256']
    published = load_registry_entry(data['registry'], 'voxel-64-r1')
    assert published['sample_count'] == manifest['sample_count']
    assert published['manifest']['dataset_fingerprint'] == manifest['dataset_fingerprint']
    assert published['family_verified'] is False
    repeat = register_dataset(data['registry'], data['manifest_path'], 'voxel-64-r1', 'mathematical-family')
    assert repeat['already_registered'] is True


def test_exported_masks_attenuation_lineage_and_crop_provenance(prepared64):
    data = prepared64
    folder, manifest = data['manifest_path'].parent, data['manifest']
    with np.load(folder / 'source.npz', allow_pickle=False) as source:
        before = source['target_mask'].astype(bool)
        original_labels = source['original_labels']
        attenuation = source['original_attenuation_per_pixel']
        roi = source['roi_mask'].astype(bool)
        geometry = json.loads(str(source['geometry_json']))
        assert geometry == manifest['geometry']
        assert json.loads(str(source['catalog_json'])) == json.loads(data['catalog_path'].read_text())
    spacing = np.asarray(data['spacing_ijk_mm'])
    low = geometry['crop_origin_ijk']
    high = geometry['crop_high_ijk_exclusive']
    index = tuple(slice(lo, hi) for lo, hi in zip(low[::-1], high[::-1]))
    np.testing.assert_array_equal(original_labels, data['labels'][index])
    np.testing.assert_array_equal(attenuation,
        np.asarray(data['attenuation_cm_inverse'] * (spacing[0] / 10), dtype='<f4')[index])
    baseline_image = np.asarray(attenuation / (spacing[0] / 10), dtype=np.float32)
    assert original_labels.shape == tuple(geometry['crop_shape_kji'])
    assert geometry['source_shape_kji'] == list(data['shape_kji'])
    np.testing.assert_array_equal(geometry['spacing_ijk_mm'], spacing)
    assert (folder / 'source-audit.json').read_bytes() == data['audit_path'].read_bytes()
    assert (folder / 'source-label-catalog.json').read_bytes() == data['catalog_path'].read_bytes()
    audit = json.loads((folder / 'source-audit.json').read_text())
    # Original-grid origin/shape survives export even though the native reader
    # processes only the local 64³ fixture grid.
    assert audit['cases'][0]['crop_origin_ijk'] == list(data['crop_origin_ijk'])
    assert audit['cases'][0]['original_shape_kji'] == list(data['original_shape_kji'])
    assert json.loads((folder / 'source-label-catalog.json').read_text())['crop_provenance'] == {
        key: audit['cases'][0][key] for key in
        ('crop_origin_ijk', 'original_shape_kji', 'coordinate_semantics', 'fixture_kind')}
    nearby_artery = data['masks']['neighbor'][index]
    protected = (data['masks']['bone'] | data['masks']['skin'])[index]
    assert nearby_artery.any() and protected.any()
    saw_added = saw_removed = saw_growth_outside_selector = False
    for sample in manifest['samples']:
        with np.load(folder / sample['path'], allow_pickle=False) as arrays:
            after = arrays['mask'].astype(bool)
            added, removed = after & ~before, before & ~after
            assert not np.any(added & removed)
            assert int(after.sum()) == int(before.sum()) + int(added.sum()) - int(removed.sum())
            saw_added |= bool(added.any())
            saw_removed |= bool(removed.any())
            saw_growth_outside_selector |= bool(after[~roi].any())
            origins = arrays['target_origin_index']
            expected_target = (origins >= 0) & before.ravel()[np.maximum(origins, 0)]
            np.testing.assert_array_equal(after, expected_target)
            assert not after[nearby_artery].any()
            np.testing.assert_array_equal(arrays['edited_labels'][nearby_artery], original_labels[nearby_artery])
            np.testing.assert_array_equal(arrays['edited_labels'][protected], original_labels[protected])
            np.testing.assert_array_equal(arrays['changed_mask'], arrays['edited_labels'] != original_labels)
            scalar_changes = arrays['scalar_changed_mask'].astype(bool)
            assert not np.any(scalar_changes & ~arrays['edit_region_mask'])
            np.testing.assert_array_equal(arrays['image'][~scalar_changes], baseline_image[~scalar_changes])
            # Target descendants inherit their original ancestor's attenuation.
            np.testing.assert_array_equal(arrays['image'][after], baseline_image.ravel()[origins[after]])
            # Independent label/signal table also checks reassigned releases,
            # where target ancestry alone does not describe the recipient.
            expected_mu = np.empty(original_labels.shape, dtype=np.float32)
            for label in np.unique(arrays['edited_labels']):
                expected_mu[arrays['edited_labels'] == label] = data['label_signals_cm_inverse'][int(label)]
            np.testing.assert_allclose(arrays['image'], expected_mu, rtol=0, atol=6e-8)
            assert arrays['mask'].dtype == np.uint8
            assert arrays['image'].dtype == np.float32 and np.isfinite(arrays['image']).all()
            metadata = json.loads(str(arrays['metadata_json']))
            assert metadata['step_outcomes'] == sample['step_outcomes']
    assert saw_added and saw_removed and saw_growth_outside_selector
    for path, checksum in data['input_hashes'].items():
        if Path(path) != Path(data['cohort_ini']):
            assert _sha(path) == checksum
    # The generated source pair is the proposed 2 MiB 64³ float32 pack. No full
    # scanner/phantom volume or checkpoint is needed by this integration check.
    assert sum(Path(path).stat().st_size for path in data['source_paths'].values()) == 2 * 64**3 * 4
    assert max(path.stat().st_size for path in folder.rglob('*.npz')) < 4 * 1024**2


def test_published_data_tamper_and_frozen_input_rewrites_are_rejected(prepared64, tmp_path):
    data = prepared64
    original = data['manifest_path'].parent
    copied = tmp_path / 'copied-dataset'
    shutil.copytree(original, copied)
    manifest = json.loads((copied / 'dataset-manifest.json').read_text())
    payload = copied / manifest['samples'][-1]['path']
    payload.write_bytes(payload.read_bytes() + b'tampered')
    with pytest.raises(ValueError, match='artifact checksum'):
        verify_prepared_manifest(copied / 'dataset-manifest.json')
    snapshot = _file_inventory(original)
    with pytest.raises(FileExistsError):
        _run(data['cohort_ini'], 'prepare')
    with pytest.raises(FileExistsError):
        _run(data['cohort_ini'], 'freeze')
    assert _file_inventory(original) == snapshot
    changed_ini = tmp_path / 'changed.ini'
    changed_ini.write_bytes(Path(data['cohort_ini']).read_bytes() + b'\n# A new reviewed input revision\n')
    with patch('fakect_roi.read_crop', side_effect=AssertionError('Freeze rejection precedes native reads')):
        with pytest.raises(ValueError, match='changed'):
            _run(changed_ini, 'prepare')
    assert _file_inventory(original) == snapshot
    assert load_registry_entry(data['registry'], 'voxel-64-r1')['sample_count'] == manifest['sample_count']
