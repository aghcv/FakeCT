"""Matched aorta-plus-connected-branches supervision on a frozen native CT box.

The original atlas core seeds native connectivity. Pulmonary arteries, other
excluded classes and disconnected components cannot carry a connection. The
default includes eligible branch continuations to the shared CT-box/slab caps,
rather than introducing a radius-dependent branch cut inside that box. Original
component membership and edit ancestry, never whole anatomical IDs, transfer
the annotation to digital siblings. This module does not change CT input crops.
"""
from collections import OrderedDict
from copy import copy, deepcopy
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

from fakect_phantom_anatomy import anatomical_roles
from fakect_pretrained_adapter import (ThoracicCoreProvider, _checked, _paste,
                                      _sha, _triple)
from fakect_context_volume import MAX_BLOCK_VOXELS
from fakect_roi import read_crop, MAX_CROP_VOXELS


SCHEMA = 'fakect.pretrained-target/1'
MAX_ANNOTATION_VOXELS = 128_000_000


def _canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':'),
                                    allow_nan=False).encode()).hexdigest()


def _readonly(value):
    value = np.asarray(value)
    value.flags.writeable = False
    return value


def _array_sha(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _binary(value, shape, name):
    value = np.asarray(value)
    if value.shape != tuple(shape) or value.dtype.kind not in 'bui' or not np.all((value == 0) | (value == 1)):
        raise ValueError(name + ' must be a binary native-grid mask')
    return value.astype(np.uint8)


class BranchTargetProvider(ThoracicCoreProvider):
    """Compatible streaming interface; legacy core aliases mean selected target.

``baseline_core`` and ``current_edit_core`` are retained for the existing data
adapter. Their meaning is recorded explicitly in metadata. ``core_provider``
on the combined provider supplies the independent original core annotation.
    """
    def _clamp_domain(self, mask, origin_ijk):
        origin = np.asarray(origin_ijk, int)[::-1]
        low = np.asarray(self.box_origin_ijk, int)[::-1]
        high = low + np.asarray(self.box_shape_kji, int)
        axes = [np.arange(n) + o for n, o in zip(mask.shape, origin)]
        z = (axes[0] >= max(low[0], self.domain_k[0])) & (axes[0] < min(high[0], self.domain_k[1]))
        mask *= z[:, None, None]
        mask *= ((axes[1] >= low[1]) & (axes[1] < high[1]))[None, :, None]
        mask *= ((axes[2] >= low[2]) & (axes[2] < high[2]))[None, None, :]

    def read_block(self, origin_ijk, shape_kji):
        origin = _triple(origin_ijk, 'Target block origin', integer=True)
        shape = tuple(_triple(shape_kji, 'Target block shape', integer=True, positive=True))
        if math.prod(shape) > MAX_BLOCK_VOXELS:
            raise ValueError('Target block exceeds bounded read budget')
        before = np.zeros(shape, np.uint8)
        _paste(before, self.baseline_core, self.core_origin, origin)
        self._clamp_domain(before, origin)
        after = before.copy()
        _paste(after, self.current_edit_core, self.edit_origin, origin)
        self._clamp_domain(after, origin)
        domain = np.ones(shape, np.uint8)
        self._clamp_domain(domain, origin)
        return dict(target=after, baseline=before, domain=domain,
                    changed=np.logical_xor(after, before).astype(np.uint8))


class BranchInclusiveTemplate:
    """Freeze one original component annotation and transfer it to siblings.

``annotation_directory`` optionally builds or loads a signed immutable cache.
The shared input grid, source ACT statistics, original labels and atlas evidence
must match exactly when that cache is reused. All mask arrays are native KJI.
    """
    def __init__(self, baseline_row, selection, evidence, grid, *, verify_hashes=True,
                 branch_extent='common_box', annotation_directory=None):
        if baseline_row.get('baseline') is not True:
            raise ValueError('Branch template requires an unchanged baseline row')
        if branch_extent not in ('common_box', 'proximal'):
            raise ValueError('branch_extent must be common_box or proximal')
        if str(grid['case_id']) != str(baseline_row['case_id']):
            raise ValueError('Frozen branch box belongs to a different case')
        self.baseline_row = deepcopy(baseline_row)
        self.selection, self.evidence, self.grid = deepcopy(selection), deepcopy(evidence), deepcopy(grid)
        self.branch_extent = branch_extent
        self._cache = OrderedDict()
        original_core = ThoracicCoreProvider(baseline_row, selection, evidence, verify_hashes=verify_hashes)
        self.spacing = original_core.spacing.copy()
        if not np.allclose(grid['native_spacing_ijk_mm'], self.spacing, atol=1e-9, rtol=0):
            raise ValueError('Frozen branch box spacing disagrees with native source')
        self.box_origin_ijk = _triple(grid['native_origin_ijk'], 'Branch box origin', integer=True)
        self.box_shape_kji = _triple(grid['native_shape_kji'], 'Branch box shape', integer=True, positive=True)
        high = self.box_origin_ijk + self.box_shape_kji[::-1]
        if (np.any(self.box_origin_ijk < 0) or np.any(high > original_core.source_shape[::-1]) or
                math.prod(self.box_shape_kji) > MAX_ANNOTATION_VOXELS):
            raise ValueError('Frozen branch annotation box is invalid or exceeds native budget')
        selection_path = Path(baseline_row['selection_manifest']).resolve()
        catalog_path = Path(selection['catalog_path'])
        if not catalog_path.is_absolute():
            catalog_path = selection_path.parent / catalog_path
        self.catalog_path = _checked(catalog_path, selection['catalog_sha256'], verify_hashes)
        catalog = json.loads(self.catalog_path.read_text())
        roles = anatomical_roles(catalog, include_branches=True)
        self.anchor_ids = np.asarray(sorted(i for i, role in roles.items() if role == 'anchor'), np.int64)
        self.branch_ids = np.asarray(sorted(i for i, role in roles.items() if role == 'branch'), np.int64)
        self.eligible_ids = np.concatenate((self.anchor_ids, self.branch_ids))
        if not set(map(int, selection['anchor_source_ids'])) <= set(map(int, self.anchor_ids)):
            raise ValueError('Frozen atlas aortic anchors disagree with anatomical roles')
        connectivity = selection.get('settings', {}).get('connectivity', 6)
        if connectivity not in (6, 26) or isinstance(connectivity, bool):
            raise ValueError('Branch connectivity must be 6 or 26')
        self.connectivity = connectivity
        target_path = Path(selection['mask_npz'])
        if not target_path.is_absolute():
            target_path = selection_path.parent / target_path
        target_path = _checked(target_path, selection['mask_npz_sha256'], verify_hashes)
        with np.load(target_path, allow_pickle=False) as masks:
            proximal = _binary(masks['mask'], original_core.baseline_core.shape, 'Frozen proximal union')
            branches = _binary(masks['branch_mask'], proximal.shape, 'Frozen proximal branches')
        if not np.array_equal(proximal, original_core.baseline_core | branches) or np.any(original_core.baseline_core & branches):
            raise ValueError('Frozen proximal union/core/branch masks disagree')
        reference_core = np.zeros(tuple(self.box_shape_kji), np.uint8)
        _paste(reference_core, original_core.baseline_core, original_core.core_origin, self.box_origin_ijk)
        proximal_full = np.zeros_like(reference_core)
        _paste(proximal_full, proximal, original_core.core_origin, self.box_origin_ijk)
        original_labels_path = _checked(baseline_row['source_npz'], baseline_row['source_npz_sha256'], verify_hashes)
        with np.load(original_labels_path, allow_pickle=False) as source:
            self.original_labels = _readonly(np.asarray(source['original_labels']))
        self.source_act = self._act_source()
        dependencies = dict(case_id=str(baseline_row['case_id']),
            source_npz_sha256=baseline_row['source_npz_sha256'],
            selection_manifest_sha256=baseline_row['selection_manifest_sha256'],
            selection_mask_sha256=selection['mask_npz_sha256'], catalog_sha256=selection['catalog_sha256'],
            source_act=self.source_act, grid_sha256=_canonical_sha(grid), branch_extent=branch_extent,
            connectivity=connectivity, z_slab_mm=selection['z_slab_mm'])
        self.annotation_manifest_path = None
        directory = Path(annotation_directory).resolve() if annotation_directory is not None else None
        if directory is not None and directory.exists():
            combined, saved = self._load_annotation(directory, dependencies, verify_hashes)
            self.native_read = saved['native_read']
            if saved['array_sha256']['core'] != _array_sha(reference_core):
                raise ValueError('Cached core disagrees with frozen original atlas component')
        elif branch_extent == 'proximal':
            combined = proximal_full.copy()
            self.native_read = {'method': 'frozen_proximal_component_mask', 'path': str(target_path)}
        else:
            combined, self.native_read = self._build_connected(reference_core, original_core.domain_k)
        if np.any(proximal_full & ~combined):
            raise ValueError('Frozen proximal target is not a subset of the branch-inclusive annotation')
        if np.any(reference_core & ~combined):
            raise ValueError('Branch annotation omitted original aortic core')
        self.baseline_target = _readonly(combined)
        self.baseline_branch_mask = _readonly((combined & ~reference_core).astype(np.uint8))
        self.reference_aorta_core = _readonly(reference_core)
        self.original_aorta_core_voxels = int(reference_core.sum())
        # Existing adapter/export APIs expect these names; they now mean union.
        self.original_core_voxels = int(combined.sum())
        self.original_target_voxels = self.original_core_voxels
        self.voxel_volume_mm3 = float(np.prod(self.spacing))
        self.reference_core = np.zeros(self.original_labels.shape, np.uint8)
        _paste(self.reference_core, combined, self.box_origin_ijk, original_core.edit_origin)
        self.reference_core = _readonly(self.reference_core)
        self.reference_edit_aorta_core = np.zeros(self.original_labels.shape, np.uint8)
        _paste(self.reference_edit_aorta_core, reference_core, self.box_origin_ijk, original_core.edit_origin)
        self.reference_edit_aorta_core = _readonly(self.reference_edit_aorta_core)
        if not np.all(np.isin(self.original_labels[self.reference_core.astype(bool)], self.eligible_ids)):
            raise ValueError('Original source crop disagrees with selected arterial component labels')
        self.counts = dict(core_voxels=self.original_aorta_core_voxels,
                          branch_voxels=int(self.baseline_branch_mask.sum()),
                          target_voxels=self.original_target_voxels,
                          proximal_target_voxels=int(proximal_full.sum()),
                          branch_continuation_voxels=int((combined & ~proximal_full).sum()))
        self.metadata = dict(schema_version=SCHEMA,
            target_semantics='thoracic aorta plus atlas-qualified connected systemic/cerebrovascular/coronary branch components',
            branch_extent=branch_extent, branch_terminal_policy='common native CT box and strict T12--T1 slab caps' if branch_extent == 'common_box' else 'frozen proximal physical reach',
            legacy_core_alias='baseline_core/current_edit_core/reference_core/original_core_voxels denote combined selected target',
            growth_policy='original component ancestry; no ROI or original target bounding-box clipping',
            excluded_policy='pulmonary/unclassified/outside-hierarchy labels cannot carry connectivity; disconnected whole-ID islands excluded',
            dependencies=dependencies, counts=self.counts, native_read=self.native_read,
            eligible_source_ids=list(map(int, self.eligible_ids)),
            selected_source_ids=self._selected_ids(),
            array_sha256=dict(target=_array_sha(combined), core=_array_sha(reference_core), branch=_array_sha(self.baseline_branch_mask)))
        self.baseline_provider = self._make_provider(original_core, combined, self.reference_core, self.metadata)
        core_metadata = dict(self.metadata, target_semantics='thoracic aorta core only; separate component audit', legacy_core_alias='native aortic core')
        self.baseline_provider.core_provider = self._make_provider(original_core, reference_core, self.reference_edit_aorta_core, core_metadata)
        self.baseline_provider.core_provider.core_provider = None
        # Baseline transfer must be unchanged even if its file has bad lineage.
        self.provider(baseline_row, verify_hashes=verify_hashes)
        if directory is not None and not directory.exists():
            self._save_annotation(directory)

    def _act_source(self):
        record = deepcopy(self.baseline_row['source_act'])
        record['path'] = str(Path(record['path']).resolve())
        stat = Path(record['path']).stat()
        expected = math.prod(self.baseline_row['source_shape_kji']) * 4
        if stat.st_size != expected or stat.st_size != record['bytes'] or stat.st_mtime_ns != record['mtime_ns']:
            raise ValueError('Frozen native ACT source size/stat changed')
        selected = self.selection.get('act_source')
        if selected and (Path(selected['path']).resolve() != Path(record['path']) or
                         any(selected[k] != record[k] for k in ('bytes', 'mtime_ns'))):
            raise ValueError('Frozen ACT source disagrees with atlas selection')
        return record

    def _build_connected(self, seed, domain_k):
        shape = tuple(self.box_shape_kji)
        branch_candidates = np.zeros(shape, bool)
        native_labels = np.empty(shape, np.int32)
        plane = shape[1] * shape[2]
        slices = max(1, MAX_CROP_VOXELS // plane)
        if plane > MAX_CROP_VOXELS:
            raise ValueError('One native branch annotation plane exceeds bounded read budget')
        reads = []
        code_counts = dict(negative_activity_voxels=0)
        high_ijk = self.box_origin_ijk + self.box_shape_kji[::-1]
        for start in range(0, shape[0], slices):
            stop = min(shape[0], start + slices)
            low, high = self.box_origin_ijk.copy(), high_ijk.copy()
            low[2] += start; high[2] = self.box_origin_ijk[2] + stop
            labels, receipt = read_crop(self.source_act['path'], self.baseline_row['source_shape_kji'], low.tolist(), high.tolist())
            # ACT contains signed IDs, retained unchanged by the original
            # source crop. Only EXACT atlas-role IDs can become candidates:
            # never abs(), round(), or truncate a value into another label.
            # Fractional mixtures are unsupported by the frozen label decoder.
            if not np.all(labels == np.rint(labels)):
                raise ValueError('Native ACT codes must be integral signed original IDs')
            # The exclusive upper bound is exactly representable in float32;
            # comparing with int32.max would round that bound up to 2**31.
            if np.any(labels < -(2. ** 31)) or np.any(labels >= 2. ** 31):
                raise ValueError('Native ACT codes exceed the signed int32 ID range')
            code_counts['negative_activity_voxels'] += int((labels < 0).sum())
            z = np.arange(start, stop) + self.box_origin_ijk[2]
            slab = ((z >= domain_k[0]) & (z < domain_k[1]))[:, None, None]
            branch_candidates[start:stop] = np.isin(labels, self.branch_ids) & slab
            native_labels[start:stop] = labels.astype(np.int32)
            edit = self.baseline_row
            overlap = np.zeros(labels.shape, dtype=self.original_labels.dtype)
            _paste(overlap, self.original_labels, edit['edit_crop_origin_ijk'], low)
            inside = np.zeros(labels.shape, np.uint8)
            _paste(inside, np.ones(self.original_labels.shape, np.uint8), edit['edit_crop_origin_ijk'], low)
            if not np.array_equal(labels[inside.astype(bool)], overlap[inside.astype(bool)]):
                raise ValueError('Native ACT and frozen original edit labels disagree')
            reads.append(receipt)
        if not np.all(np.isin(native_labels[seed.astype(bool)], self.anchor_ids)):
            raise ValueError('Original aortic seed disagrees with native ACT labels')
        structure = ndimage.generate_binary_structure(3, 1 if self.connectivity == 6 else 3)
        selected = ndimage.binary_propagation(seed.astype(bool), structure=structure,
                                            mask=seed.astype(bool) | branch_candidates)
        return selected.astype(np.uint8), dict(method='bounded native ACT reads plus atlas-eligible component propagation',
            source_stat=self.source_act, slab_reads=reads,
            candidate_id_policy='exact signed atlas-role IDs; no absolute-value or rounding conversion; fractional codes unsupported',
            activity_code_counts=code_counts,
            selected_source_ids=list(map(int, np.unique(native_labels[selected]))),
            eligible_branch_voxels=int(branch_candidates.sum()),
            disconnected_branch_voxels=int((branch_candidates & ~selected).sum()))

    def _selected_ids(self):
        # The common-box build records exact selected IDs before releasing its
        # original-label volume; a frozen proximal cache supplies its own IDs.
        proximal = self.native_read.get('selected_source_ids', self.selection.get('selected_source_ids', []))
        inside = np.unique(self.original_labels[self.reference_core.astype(bool)])
        return sorted(set(map(int, proximal)) | set(map(int, inside)))

    def _make_provider(self, original, baseline, reference, metadata):
        provider = BranchTargetProvider.__new__(BranchTargetProvider)
        provider.row, provider.selection, provider.evidence = deepcopy(self.baseline_row), self.selection, self.evidence
        provider.spacing, provider.source_shape, provider.domain_k = original.spacing, original.source_shape, original.domain_k
        provider.box_origin_ijk, provider.box_shape_kji = self.box_origin_ijk, self.box_shape_kji
        provider.core_origin, provider.edit_origin = self.box_origin_ijk, original.edit_origin
        provider.baseline_core, provider.current_edit_core = baseline, reference
        provider.metadata = deepcopy(metadata)
        return provider

    def provider(self, row, *, verify_hashes=True):
        fields = ('case_id', 'source_npz', 'source_npz_sha256', 'edit_crop_origin_ijk',
                  'edit_crop_shape_kji', 'spacing_ijk_mm', 'selection_manifest',
                  'selection_manifest_sha256', 'source_shape_kji', 'source_act')
        if any(row[f] != self.baseline_row[f] for f in fields):
            raise ValueError('Sibling transfer changed its original case geometry, labels or anatomy')
        key = (row['sample_id'], row['variant_npz_sha256'])
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = _checked(row['variant_npz'], row['variant_npz_sha256'], verify_hashes)
        with np.load(path, allow_pickle=False) as variant:
            labels, origins = variant['edited_labels'], variant['target_origin_index']
        if (labels.shape != self.reference_core.shape or origins.shape != labels.shape or
                origins.dtype.kind not in 'iu'):
            raise ValueError('Sibling labels/ancestry do not match native transfer geometry')
        valid = (origins >= 0) & (origins < origins.size)
        inherited = np.zeros(labels.shape, bool)
        inherited[valid] = self.reference_core.ravel()[origins[valid]].astype(bool)
        eligible = np.isin(labels, self.eligible_ids)
        introduced = eligible & ~np.isin(self.original_labels, self.eligible_ids)
        indices = origins[introduced].astype(np.int64)
        if (np.any(indices < 0) or np.any(indices >= origins.size) or
                not np.all(np.isin(self.original_labels.ravel()[indices], self.eligible_ids))):
            raise ValueError('Introduced arterial target lacks verified original arterial ancestry')
        current = (eligible & (self.reference_core.astype(bool) | inherited)).astype(np.uint8)
        result = copy(self.baseline_provider)
        result.row = deepcopy(row)
        result.current_edit_core = current
        result._clamp_domain(current, result.edit_origin)
        if row['baseline'] and not np.array_equal(current, self.reference_core):
            raise ValueError('Unchanged baseline branch-inclusive target unexpectedly changed')
        result.current_edit_core = _readonly(current)
        result.metadata = deepcopy(self.metadata)
        result.metadata['new_target_ancestry_checked'] = int(((current == 1) & (self.reference_core == 0)).sum())
        inherited_core = np.zeros(labels.shape, bool)
        inherited_core[valid] = self.reference_edit_aorta_core.ravel()[origins[valid]].astype(bool)
        core_current = (np.isin(labels, self.anchor_ids) &
                        (self.reference_edit_aorta_core.astype(bool) | inherited_core)).astype(np.uint8)
        core_result = copy(self.baseline_provider.core_provider)
        core_result.row = deepcopy(row); core_result._clamp_domain(core_current, core_result.edit_origin)
        core_result.current_edit_core = _readonly(core_current)
        result.core_provider = core_result
        self._cache[key] = result
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return result

    def extent(self, row, *, verify_hashes=True):
        provider = self.provider(row, verify_hashes=verify_hashes)
        before, after = self.reference_core, provider.current_edit_core
        added, removed = int(((after == 1) & (before == 0)).sum()), int(((before == 1) & (after == 0)).sum())
        core_before, core_after = self.reference_edit_aorta_core, provider.core_provider.current_edit_core
        core_added = int(((core_after == 1) & (core_before == 0)).sum())
        core_removed = int(((core_before == 1) & (core_after == 0)).sum())
        changed = added + removed
        kind = 'mixed' if added and removed else 'addition_only' if added else 'removal_only' if removed else 'none'
        bucket = '0' if changed == 0 else '1-100' if changed <= 100 else '101-500' if changed <= 500 else '501-1000' if changed <= 1000 else '1001-5000' if changed <= 5000 else '>5000'
        return dict(sample_id=row['sample_id'], case_id=str(row['case_id']), anatomy_family=row['anatomy_family'], baseline=row['baseline'],
            added_voxels=added, removed_voxels=removed, changed_voxels=changed,
            added_target_voxels=added, removed_target_voxels=removed, changed_target_voxels=changed,
            baseline_target_voxels=self.original_target_voxels, current_target_voxels=self.original_target_voxels + added - removed,
            baseline_core_voxels=self.original_target_voxels, current_core_voxels=self.original_target_voxels + added - removed,
            legacy_core_count_alias='combined selected target; use aorta_core fields for core-only counts',
            baseline_aorta_core_voxels=self.original_aorta_core_voxels,
            current_aorta_core_voxels=self.original_aorta_core_voxels + core_added - core_removed,
            added_aorta_core_voxels=core_added, removed_aorta_core_voxels=core_removed,
            voxel_volume_mm3=self.voxel_volume_mm3, added_mm3=added*self.voxel_volume_mm3,
            removed_mm3=removed*self.voxel_volume_mm3, changed_mm3=changed*self.voxel_volume_mm3,
            change_kind=kind, burden_bin=bucket, target_semantics=self.metadata['target_semantics'], branch_extent=self.branch_extent,
            severity_scope='synthetic net native selected-target edit burden; not clinical disease severity')

    def _save_annotation(self, directory):
        directory.mkdir(parents=True, exist_ok=False)
        marker = directory / 'INCOMPLETE'; marker.touch()
        path = directory / 'annotation.npz'
        with path.open('xb') as stream:
            np.savez_compressed(stream, target=self.baseline_target, core=self.reference_aorta_core,
                branch=self.baseline_branch_mask, crop_origin_ijk=self.box_origin_ijk,
                spacing_ijk_mm=self.spacing, source_shape_kji=self.baseline_row['source_shape_kji'])
        record = dict(self.metadata, complete=True, mask_npz='annotation.npz', mask_npz_sha256=_sha(path))
        manifest = directory / 'annotation.json'
        manifest.write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
        manifest.with_suffix('.json.sha256').write_text(_sha(manifest) + '\n')
        marker.unlink(); self.annotation_manifest_path = manifest

    def _load_annotation(self, directory, dependencies, verify_hashes):
        manifest = directory / 'annotation.json'
        if (directory / 'INCOMPLETE').exists() or not manifest.exists():
            raise ValueError('Branch annotation cache is incomplete')
        if verify_hashes and manifest.with_suffix('.json.sha256').read_text().strip() != _sha(manifest):
            raise ValueError('Branch annotation manifest checksum changed')
        record = json.loads(manifest.read_text())
        if record.get('schema_version') != SCHEMA or record.get('complete') is not True or record.get('dependencies') != dependencies:
            raise ValueError('Frozen branch annotation dependencies/box/policy changed')
        path = _checked(directory / record['mask_npz'], record['mask_npz_sha256'], verify_hashes)
        with np.load(path, allow_pickle=False) as masks:
            target = _binary(masks['target'], self.box_shape_kji, 'Cached target')
            core = _binary(masks['core'], self.box_shape_kji, 'Cached core')
            branch = _binary(masks['branch'], self.box_shape_kji, 'Cached branches')
            if (not np.array_equal(masks['crop_origin_ijk'], self.box_origin_ijk) or
                    not np.allclose(masks['spacing_ijk_mm'], self.spacing, atol=1e-9, rtol=0) or
                    not np.array_equal(masks['source_shape_kji'], self.baseline_row['source_shape_kji'])):
                raise ValueError('Cached branch annotation native geometry changed')
        if not np.array_equal(target, core | branch) or np.any(core & branch):
            raise ValueError('Cached branch annotation component masks disagree')
        arrays = dict(target=_array_sha(target), core=_array_sha(core), branch=_array_sha(branch))
        if arrays != record['array_sha256']:
            raise ValueError('Cached branch annotation array checksums changed')
        self.annotation_manifest_path = manifest
        return target, record
