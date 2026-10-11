"""Fixed-region CT-only fine-tuning data with native aorta-core supervision.

This is an explicitly region-assisted development study.  A frozen atlas crop
supplies X/Y bounds plus 40 mm context; T12--T1 plus 50 mm supplies Z bounds.
Neither a sibling target bounding box nor a predicted mask recenters the input.
The same case grid applies to the untouched checkpoint and both fine-tuning
arms.  Native paired core differences describe synthetic edit burden only.
"""
from collections import OrderedDict
from copy import copy, deepcopy
from itertools import product
import math
import os
from pathlib import Path
from contextlib import ExitStack

import numpy as np
from scipy.ndimage import affine_transform, distance_transform_edt, generate_binary_structure, label

from fakect_context_volume import ContextVolume, MAX_BLOCK_VOXELS
from fakect_pretrained_adapter import (ThoracicCoreProvider, attenuation_to_hu,
    crop_ras_affine, plan_thorax_crop, _checked, _paste, _triple)
from fakect_pretrained_adapter import _open_writer, _sha


SCHEMA = 'fakect.pretrained-fine-data/1'
MAX_MODEL_VOXELS = 16_000_000
MAX_NATIVE_AUDIT_VOXELS = 32_000_000
CT_CLIP = (-101., 813.)
CT_MEAN = 142.1977119916391
CT_STD = 154.74032651622053


def _readonly(value):
    value = np.asarray(value)
    value.flags.writeable = False
    return value


def normalize_hu(resampled_hu):
    """Dataset293 checkpoint normalization after the post-resample int32 cast."""
    hu = np.asarray(resampled_hu)
    if not np.all(np.isfinite(hu)):
        raise ValueError('Fine-tuning HU must be finite')
    data = hu.astype(np.int32).astype(np.float32)
    np.clip(data, *CT_CLIP, out=data)
    data -= np.float32(CT_MEAN)
    data /= np.float32(CT_STD)
    return data


def fixed_case_grid(row, selection, evidence, *, spacing_mm=1.5,
                    padding_xy_mm=40., context_z_mm=50.):
    """Freeze one atlas-defined native crop and model-center grid per case."""
    spacing_mm, padding_xy_mm = float(spacing_mm), float(padding_xy_mm)
    if (not np.isfinite(spacing_mm) or not .5 <= spacing_mm <= 3. or
            not np.isfinite(padding_xy_mm) or not 0 <= padding_xy_mm <= 100.):
        raise ValueError('Invalid fixed model spacing or anatomical X/Y padding')
    thorax = plan_thorax_crop(row, selection, evidence, context_mm=context_z_mm)
    native_spacing = np.asarray(row['spacing_ijk_mm'], float)
    full = np.asarray(row['source_shape_kji'], int)[::-1]
    atlas = selection['geometry']
    low = _triple(atlas['crop_low_ijk'], 'Atlas crop low', integer=True)
    high = _triple(atlas['crop_high_ijk_exclusive'], 'Atlas crop high', integer=True)
    if np.any(low < 0) or np.any(high <= low) or np.any(high > full):
        raise ValueError('Frozen anatomical crop has invalid native bounds')
    padding = np.ceil(padding_xy_mm / native_spacing[:2]).astype(int)
    low[:2] = np.maximum(0, low[:2] - padding)
    high[:2] = np.minimum(full[:2], high[:2] + padding)
    low[2] = thorax['origin_ijk'][2]
    high[2] = low[2] + thorax['shape_kji'][0]
    native_shape = (high - low)[::-1]
    model_spacing = np.full(3, spacing_mm)
    shape = np.ceil((high - low) * native_spacing / model_spacing).astype(int)[::-1]
    if math.prod(shape) > MAX_MODEL_VOXELS:
        raise ValueError('Fixed model crop exceeds the 16-million-voxel budget')
    native_affine = crop_ras_affine(evidence, low)
    # Shared LOWER voxel faces: model center0 is half a model voxel from that face.
    # Official NibabelIOWithReorient canonicalizes to RAS before the clinical
    # checkpoint sees KJI. Native XCAT I/J point L/P, so both must be reversed
    # for CT and every supervision mask, rather than only changing the affine.
    direction = native_affine[:3, :3] / native_spacing
    if not np.allclose(direction,np.diag([-1.,-1.,1.]),rtol=0,atol=1e-6):
        raise ValueError('This declared adapter requires verified native L/P/S axes')
    signs = np.asarray([-1.,-1.,1.])
    model_direction = direction * signs
    boundary = native_affine[:3,3] - direction @ (.5*native_spacing)
    canonical_boundary = boundary + direction @ ((high-low)*native_spacing*(signs<0))
    affine = np.eye(4)
    affine[:3,:3] = model_direction * model_spacing
    affine[:3,3] = canonical_boundary + model_direction @ (.5*model_spacing)
    return {'schema_version': SCHEMA, 'case_id': str(row['case_id']),
        'source_shape_kji': list(map(int, row['source_shape_kji'])),
        'native_origin_ijk': low.tolist(), 'native_shape_kji': native_shape.tolist(),
        'native_spacing_ijk_mm': native_spacing.tolist(),
        'shape_kji': shape.tolist(), 'spacing_ijk_mm': model_spacing.tolist(),
        'affine_ras_ijk': affine.tolist(), 'native_affine_ras_ijk': native_affine.tolist(),
        'padding_xy_mm': padding_xy_mm, 'context_z_mm': float(context_z_mm),
        'scope': 'known atlas thoracic region plus fixed physical context; region-assisted',
        'grid_alignment': 'shared native crop lower voxel face; model centers at index+0.5',
        'native_axis_directions_ijk': ['L','P','S'],
        'model_axis_directions_ijk': ['R','A','S'],
        'model_array_order': 'k,j,i = S,A,R; canonicalized before clinical checkpoint',
        'native_to_model_index_sign_ijk': signs.astype(int).tolist(),
        'arm_matching': 'identical region, spacing, centers, normalization and tiling across all arms',
        'normalization': {'clip_hu': list(CT_CLIP), 'mean': CT_MEAN, 'std': CT_STD,
            'dtype_chain': 'native HU int32 -> linear resampling -> int32 -> float32 normalization'}}


class NativeCoreTemplate:
    """Verify shared source once; validate each sibling's core ancestry once."""
    def __init__(self, baseline_row, selection, evidence, *, verify_hashes=True):
        if baseline_row['baseline'] is not True:
            raise ValueError('Native core template requires an unchanged baseline row')
        self.baseline_row = deepcopy(baseline_row)
        self.selection, self.evidence = deepcopy(selection), deepcopy(evidence)
        self.baseline_provider = ThoracicCoreProvider(baseline_row, selection, evidence,
                                                     verify_hashes=verify_hashes)
        with np.load(baseline_row['source_npz'], allow_pickle=False) as source:
            original = source['original_labels']
        self.original_anchor = _readonly(np.isin(original, selection['anchor_source_ids']))
        self.reference_core = self.baseline_provider.current_edit_core
        self.original_core_voxels = int(self.baseline_provider.baseline_core.sum())
        self.voxel_volume_mm3 = float(np.prod(baseline_row['spacing_ijk_mm']))
        self._cache = OrderedDict()

    def provider(self, row, *, verify_hashes=True):
        fields = ('case_id','source_npz','source_npz_sha256','edit_crop_origin_ijk',
                  'edit_crop_shape_kji','spacing_ijk_mm','selection_manifest',
                  'selection_manifest_sha256','source_shape_kji')
        if any(row[f] != self.baseline_row[f] for f in fields):
            raise ValueError('Sibling core transfer changed its original source geometry or anatomy')
        key = (row['sample_id'], row['variant_npz_sha256'])
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        path = _checked(row['variant_npz'], row['variant_npz_sha256'], verify_hashes)
        with np.load(path, allow_pickle=False) as variant:
            labels, origins = variant['edited_labels'], variant['target_origin_index']
        if (labels.shape != self.reference_core.shape or origins.shape != labels.shape or
                origins.dtype.kind not in 'iu'):
            raise ValueError('Sibling labels/ancestry do not match the native source crop')
        anchors = np.isin(labels, self.selection['anchor_source_ids'])
        introduced = anchors & ~self.original_anchor
        indices = origins[introduced].astype(np.int64)
        if (np.any(indices < 0) or np.any(indices >= origins.size) or
                not np.all(self.original_anchor.ravel()[indices])):
            raise ValueError('Introduced aorta labels do not inherit original aorta ancestry')
        valid = (origins >= 0) & (origins < origins.size)
        inherited = np.zeros(labels.shape, bool)
        inherited[valid] = self.reference_core.ravel()[origins[valid]].astype(bool)
        current = (anchors & (self.reference_core.astype(bool) | inherited)).astype(np.uint8)
        result = copy(self.baseline_provider)
        result.row = deepcopy(row)
        result.current_edit_core = current
        result._clamp_domain(result.current_edit_core, result.edit_origin)
        if row['baseline'] and not np.array_equal(current, self.reference_core):
            raise ValueError('An unchanged baseline core unexpectedly changed')
        result.current_edit_core.flags.writeable = False
        result.metadata = deepcopy(self.baseline_provider.metadata)
        result.metadata['new_core_ancestry_checked'] = int(np.sum((current == 1) & (self.reference_core == 0)))
        self._cache[key] = result
        while len(self._cache) > 2:
            self._cache.popitem(last=False)
        return result

    def extent(self, row, *, verify_hashes=True):
        provider = self.provider(row, verify_hashes=verify_hashes)
        before, after = self.reference_core, provider.current_edit_core
        added = int(np.sum((after == 1) & (before == 0)))
        removed = int(np.sum((before == 1) & (after == 0)))
        changed = added + removed
        kind = ('mixed' if added and removed else 'addition_only' if added else
                'removal_only' if removed else 'none')
        bucket = ('0' if changed == 0 else '1-100' if changed <= 100 else
                  '101-500' if changed <= 500 else '501-1000' if changed <= 1000 else
                  '1001-5000' if changed <= 5000 else '>5000')
        return {'sample_id':row['sample_id'], 'case_id':str(row['case_id']),
            'anatomy_family':row['anatomy_family'], 'baseline':row['baseline'],
            'added_voxels':added, 'removed_voxels':removed, 'changed_voxels':changed,
            'baseline_core_voxels':self.original_core_voxels,
            'current_core_voxels':self.original_core_voxels + added - removed,
            'voxel_volume_mm3':self.voxel_volume_mm3,
            'added_mm3':added*self.voxel_volume_mm3, 'removed_mm3':removed*self.voxel_volume_mm3,
            'changed_mm3':changed*self.voxel_volume_mm3, 'change_kind':kind,
            'burden_bin':bucket, 'severity_scope':'synthetic native core edit burden; not clinical disease severity'}


def native_core_extent(row, selection, evidence, *, template=None, verify_hashes=True):
    """Exact net native added/removed core counts, independent of model sampling."""
    if template is None:
        provider = ThoracicCoreProvider(row, selection, evidence, verify_hashes=verify_hashes)
        before = np.zeros(provider.current_edit_core.shape, np.uint8)
        _paste(before, provider.baseline_core, provider.core_origin, provider.edit_origin)
        provider._clamp_domain(before, provider.edit_origin)
        # Minimal immutable template sharing the already-verified native reference.
        template = object.__new__(NativeCoreTemplate)
        template.reference_core = before
        template.baseline_provider = provider
        template.original_core_voxels = int(provider.baseline_core.sum())
        template.voxel_volume_mm3 = float(np.prod(row['spacing_ijk_mm']))
        template.provider = lambda unused, **kwargs: provider
    return template.extent(row, verify_hashes=verify_hashes)


def _even_select(candidates, count, extents):
    ordered = sorted(candidates, key=lambda r:(extents[r['sample_id']]['changed_voxels'],r['sample_id']))
    if len(ordered) < count:
        raise ValueError('Insufficient distinct positive siblings for a frozen balanced roster')
    positions = np.linspace(0,len(ordered)-1,count).round().astype(int)
    return [ordered[int(i)] for i in positions]


def choose_nested_rosters(rows, extent_by_sample_id, *, train_siblings=16,
                          evaluation_siblings=12, seed=42):
    """Fixed equal-case panels; extent strata never alter held-out predictions."""
    if train_siblings != 16 or evaluation_siblings != 12 or not isinstance(seed,int):
        raise ValueError('This matched pilot fixes 16 training and 12 evaluation siblings per case')
    result = {'schema_version':SCHEMA, 'seed':seed, 'training_baseline_only':{},
        'training_siblings':{}, 'training_nested':{}, 'evaluation':{},
        'selection_policy':'test-first even extent quantiles within native core addition/removal/mixed; training disjoint and nested',
        'severity_scope':'native voxel burden and mm3; no clinical intermediate/severe assignment'}
    for case in sorted({r['case_id'] for r in rows}):
        case_rows = [r for r in rows if r['case_id']==case]
        controls = [r for r in case_rows if r['baseline']]
        if len(controls) != 1:
            raise ValueError('Require exactly one original baseline per phantom')
        control = controls[0]['sample_id']
        groups = {kind:[] for kind in ('addition_only','removal_only','mixed')}
        for row in case_rows:
            kind = extent_by_sample_id[row['sample_id']]['change_kind']
            if not row['baseline'] and row.get('duplicate_of') is None and kind in groups:
                groups[kind].append(row)
        eval_rows, training = [], {}
        for kind, group in groups.items():
            picked = _even_select(group,4,extent_by_sample_id)
            eval_rows.extend(picked)
            picked_ids = {r['sample_id'] for r in picked}
            remaining = [r for r in group if r['sample_id'] not in picked_ids]
            selected = _even_select(remaining,8 if kind=='mixed' else 4,extent_by_sample_id)
            # Prefixes see both burden extremes before intermediate siblings.
            interleaved = []
            while selected:
                interleaved.append(selected.pop(-1))
                if selected: interleaved.append(selected.pop(0))
            training[kind] = interleaved
        train_rows = []
        for _ in range(4):
            for kind in ('removal_only','addition_only','mixed','mixed'):
                train_rows.append(training[kind].pop(0))
        train_ids = [r['sample_id'] for r in train_rows]
        result['training_baseline_only'][case] = [control]
        result['training_siblings'][case] = [control]+train_ids
        result['training_nested'][case] = {str(n):[control]+train_ids[:n] for n in (4,8,16)}
        result['evaluation'][case] = [control]+[r['sample_id'] for r in eval_rows]
    return result


class PretrainedCaseVolume:
    """Read CT-only model patches and separate supervision on a frozen case grid."""
    def __init__(self, row, grid, selection, evidence, *, template=None,
                 verify_hashes=True, patch_cache_size=2):
        if str(row['case_id']) != str(grid['case_id']):
            raise ValueError('Fine-tuning row and fixed case grid disagree')
        if type(patch_cache_size) is not int or not 0 <= patch_cache_size <= 4:
            raise ValueError('Patch cache must contain at most four bounded patches')
        self.row, self.grid = deepcopy(row), deepcopy(grid)
        self.shape_kji = np.asarray(grid['shape_kji'],int)
        self.spacing_ijk_mm = np.asarray(grid['spacing_ijk_mm'],float)
        self.native_spacing_kji = np.asarray(row['spacing_ijk_mm'],float)[::-1]
        self.native_origin_kji = np.asarray(grid['native_origin_ijk'],int)[::-1]
        self.native_high_kji = self.native_origin_kji + np.asarray(grid['native_shape_kji'],int)
        self.axis_sign_kji = np.asarray(grid['native_to_model_index_sign_ijk'],int)[::-1]
        self.scale_kji = self.spacing_ijk_mm[::-1] / self.native_spacing_kji
        self.evidence = evidence
        self.provider = (template.provider(row,verify_hashes=verify_hashes) if template else
                         ThoracicCoreProvider(row,selection,evidence,verify_hashes=verify_hashes))
        self.volume = ContextVolume(row,verify_hashes=verify_hashes)
        self._cache, self._point_cache = OrderedDict(), {}
        self.patch_cache_size = patch_cache_size

    def _coordinates(self, offset, shape):
        offset = _triple(offset,'Model patch offset',integer=True)
        shape = _triple(shape,'Model patch shape',integer=True,positive=True)
        if math.prod(shape)>MAX_BLOCK_VOXELS:
            raise ValueError('Model patch exceeds the bounded reader budget')
        coords = [(int(lo)+(int(o)+np.arange(int(n))+.5)*s-.5 if sign>0 else
                   int(hi)-(int(o)+np.arange(int(n))+.5)*s-.5)
            for lo,hi,o,n,s,sign in zip(self.native_origin_kji,self.native_high_kji,
                                       offset,shape,self.scale_kji,self.axis_sign_kji)]
        model_ok = [(int(o)+np.arange(int(n))>=0)&(int(o)+np.arange(int(n))<full)
                    for o,n,full in zip(offset,shape,self.shape_kji)]
        native_ok = [(c>=-.5)&(c<full-.5) for c,full in zip(coords,self.volume.source_shape_kji)]
        valid = (model_ok[0][:,None,None]&model_ok[1][None,:,None]&model_ok[2][None,None,:]
                 &native_ok[0][:,None,None]&native_ok[1][None,:,None]&native_ok[2][None,None,:])
        return offset,shape,coords,valid

    def read_masks(self,offset_kji,shape_kji):
        _,shape,coords,valid = self._coordinates(offset_kji,shape_kji)
        nearest = [np.floor(c+.5).astype(int) for c in coords]
        lo = np.asarray([max(0,int(c.min())) for c in nearest])
        hi = np.asarray([min(full,int(c.max())+1) for c,full in zip(nearest,self.volume.source_shape_kji)])
        result = {key:np.zeros(tuple(shape),np.uint8) for key in ('target','baseline','domain')}
        if np.all(hi>lo):
            # X/Y can exceed the native block budget for very fine sources;
            # slice along model Z and let each bounded slab provide its masks.
            if math.prod(hi-lo)>MAX_BLOCK_VOXELS:
                if shape[0]==1: raise ValueError('One nearest-mask source slice exceeds budget')
                mid = int(shape[0]//2)
                first = self.read_masks(offset_kji,[mid,int(shape[1]),int(shape[2])])
                second_offset = np.asarray(offset_kji,int)+[mid,0,0]
                second = self.read_masks(second_offset,[int(shape[0])-mid,int(shape[1]),int(shape[2])])
                return {key:np.concatenate((first[key],second[key])) for key in first}
            source = self.provider.read_block(lo[::-1],hi-lo)
            positions = [np.clip(c-l,0,int(h-l)-1) for c,l,h in zip(nearest,lo,hi)]
            for key in result:
                result[key] = source[key][np.ix_(*positions)].astype(np.uint8)
                result[key] *= valid
        result['valid'] = valid.astype(np.uint8)
        result['added'] = ((result['target']==1)&(result['baseline']==0)).astype(np.uint8)
        result['removed'] = ((result['baseline']==1)&(result['target']==0)).astype(np.uint8)
        result['changed'] = (result['added']|result['removed']).astype(np.uint8)
        return result

    def _read_hu(self,offset_kji,shape_kji):
        offset,shape,coords,valid = self._coordinates(offset_kji,shape_kji)
        result = np.full(tuple(shape),-1000.,np.float32)
        # Bounded target-Z slabs prevent a 128^3 patch from allocating a 384^3
        # native intermediate at pediatric 0.5 mm resolution.
        for z in range(0,int(shape[0]),4):
            count = min(4,int(shape[0])-z)
            cz = coords[0][z:z+count]
            axes = [cz,coords[1],coords[2]]
            lo = np.asarray([max(0,int(np.floor(c.min()))) for c in axes])
            hi = np.asarray([min(full,int(np.ceil(c.max()))+1)
                            for c,full in zip(axes,self.volume.source_shape_kji)])
            if np.any(hi<=lo): continue
            if math.prod(hi-lo)>MAX_BLOCK_VOXELS:
                raise ValueError('Fine-tuning HU native slab exceeds the bounded source budget')
            native = self.volume.read_block(lo,hi-lo,self.row['spacing_ijk_mm'])['image']
            # Official crop_to_mask casts native CT to int32 before its linear
            # resample; preserve both truncation stages rather than resampling μ.
            native_hu = attenuation_to_hu(native,self.evidence['calibration']['mu_water_cm_inverse']).astype(np.int32)
            sampled = affine_transform(native_hu.astype(np.float32),np.diag(self.scale_kji*self.axis_sign_kji),
                offset=np.asarray([c[0] for c in axes])-lo,
                output_shape=(count,int(shape[1]),int(shape[2])),order=1,mode='nearest',
                prefilter=False,output=np.float32)
            result[z:z+count] = sampled.astype(np.int32).astype(np.float32)
        result[~valid] = -1000.
        return result,valid

    def read_patch(self,offset_kji,shape_kji=(128,128,128)):
        key = (tuple(map(int,offset_kji)),tuple(map(int,shape_kji)))
        if key in self._cache:
            self._cache.move_to_end(key)
            return self._cache[key]
        hu,valid = self._read_hu(offset_kji,shape_kji)
        masks = self.read_masks(offset_kji,shape_kji)
        image = normalize_hu(hu)
        image[~valid] = 0 # nnUNet-style normalized padding, excluded from loss.
        result = {'image':_readonly(image[None]),**{k:_readonly(v) for k,v in masks.items()},
            'metadata':{'sample_id':self.row['sample_id'],'offset_kji':list(offset_kji),
                'shape_kji':list(shape_kji),'input_channels':'normalized CT only',
                'native_hu_int32_before_linear':True,'normalized_padding':0.}}
        if self.patch_cache_size:
            self._cache[key] = result
            while len(self._cache)>self.patch_cache_size: self._cache.popitem(last=False)
        return result

    def masks_full(self):
        result = {k:np.zeros(tuple(self.shape_kji),np.uint8)
                  for k in ('target','baseline','domain','valid','added','removed','changed')}
        for z in range(0,int(self.shape_kji[0]),8):
            count = min(8,int(self.shape_kji[0])-z)
            block = self.read_masks([z,0,0],[count,int(self.shape_kji[1]),int(self.shape_kji[2])])
            for key in result: result[key][z:z+count] = block[key]
        return result

    def full_image(self):
        result = np.zeros(tuple(self.shape_kji),np.float32)
        for z in range(0,int(self.shape_kji[0]),8):
            count = min(8,int(self.shape_kji[0])-z)
            hu,valid = self._read_hu([z,0,0],[count,int(self.shape_kji[1]),int(self.shape_kji[2])])
            data = normalize_hu(hu);data[~valid]=0;result[z:z+count]=data
        return result[None]

    def sample_patch_offset(self,rng,shape_kji=(128,128,128),*,positive_fraction=.75):
        """Training-only target/change sampling with absent-class avoidance."""
        shape = np.asarray(shape_kji,int)
        if shape.shape!=(3,) or np.any(shape<=0) or not 0<=positive_fraction<=1:
            raise ValueError('Invalid training patch sampling settings')
        if not self._point_cache:
            masks = self.masks_full()
            for key in ('target','added','removed'):
                points = np.argwhere(masks[key]>0)
                if len(points)>8192:
                    points = points[np.linspace(0,len(points)-1,8192).astype(int)]
                self._point_cache[key] = points
        choices = [key for key,points in self._point_cache.items() if len(points)]
        mode = 'uniform'
        if choices and rng.random()<positive_fraction:
            mode = str(rng.choice(choices))
            points = self._point_cache[mode]
            point = points[int(rng.integers(len(points)))]
            offset = point - np.asarray([rng.integers(int(n)) for n in shape])
        else:
            offset = np.asarray([rng.integers(max(1,int(full-n+1)))
                                 for full,n in zip(self.shape_kji,shape)])
        for axis,(full,n) in enumerate(zip(self.shape_kji,shape)):
            offset[axis] = (-(int(n-full)//2) if full<n else np.clip(offset[axis],0,full-n))
        return {'offset_kji':offset.astype(int).tolist(),'shape_kji':shape.tolist(),
                'sampling_mode':mode,'sampling_scope':'training supervision only; inference tiles fixed'}

    def close(self):
        self.volume.close();self._cache.clear();self._point_cache.clear()

    def __enter__(self): return self
    def __exit__(self,*unused): self.close()


def inference_tiles(grid,shape_kji=(128,128,128),overlap=64):
    """Identical target-independent padded tile schedule for all study arms."""
    shape = _triple(shape_kji,'Inference tile shape',integer=True,positive=True)
    if type(overlap) is not int or not 0<=overlap<min(shape):
        raise ValueError('Invalid fixed tile overlap')
    axes = []
    for full,n in zip(grid['shape_kji'],shape):
        if full<n: axes.append([-(int(n-full)//2)]);continue
        last = int(full-n);starts=list(range(0,last+1,int(n-overlap)))
        if starts[-1]!=last: starts.append(last)
        axes.append(starts)
    return [{'offset_kji':list(offset),'shape_kji':shape.tolist()} for offset in product(*axes)]


def restore_native_mask(model_mask,grid,*,native_origin_ijk=None,native_shape_kji=None):
    """Undo canonical R/A flips and sample a model mask on the native crop grid."""
    model=np.asarray(model_mask)
    if model.shape!=tuple(grid['shape_kji']) or not np.all((model==0)|(model==1)):
        raise ValueError('Native restoration requires a binary mask on the frozen canonical grid')
    origin=np.asarray(grid['native_origin_ijk'] if native_origin_ijk is None else native_origin_ijk,int)
    shape=np.asarray(grid['native_shape_kji'] if native_shape_kji is None else native_shape_kji,int)
    if origin.shape!=(3,) or shape.shape!=(3,) or np.any(shape<=0) or math.prod(shape)>128_000_000:
        raise ValueError('Native restoration requires a bounded native crop')
    low=np.asarray(grid['native_origin_ijk'])[::-1]
    high=low+np.asarray(grid['native_shape_kji'])
    signs=np.asarray(grid['native_to_model_index_sign_ijk'])[::-1]
    native_spacing=np.asarray(grid['native_spacing_ijk_mm'])[::-1]
    model_spacing=np.asarray(grid['spacing_ijk_mm'])[::-1]
    coords=[np.floor(((np.arange(int(n))+int(o)+.5-int(l)) if sign>0 else
                      (int(h)-np.arange(int(n))-int(o)-.5))*ns/ms).astype(int)
            for n,o,l,h,sign,ns,ms in zip(shape,origin[::-1],low,high,signs,native_spacing,model_spacing)]
    valid=[(c>=0)&(c<n) for c,n in zip(coords,model.shape)]
    clipped=[np.clip(c,0,n-1) for c,n in zip(coords,model.shape)]
    restored=np.zeros(tuple(shape),np.uint8)
    for z in range(0,int(shape[0]),8):
        n=min(8,int(shape[0])-z)
        data=model[np.ix_(clipped[0][z:z+n],clipped[1],clipped[2])].astype(np.uint8)
        data*=valid[0][z:z+n,None,None];data*=valid[1][None,:,None];data*=valid[2][None,None,:]
        restored[z:z+n]=data
    return restored


def export_case_native_inputs(row,grid,selection,evidence,output_directory,*,
                              template=None,verify_hashes=True,slab_slices=8,
                              export_baseline=False):
    """Compact native ROI report artifacts shared by all three model arms."""
    output=Path(output_directory)
    paths={key:output/(name+'.nii.gz') for key,name in
           [('image','input'),('target','truth'),('domain','domain'),('changed','changed')]}
    if export_baseline:paths['baseline']=output/'baseline.nii.gz'
    if any(path.exists() for path in paths.values()):
        raise ValueError('Refusing to overwrite frozen common native evaluation inputs')
    if type(slab_slices) is not int or not 1<=slab_slices<=32:
        raise ValueError('Invalid native evaluation export slab size')
    provider=(template.provider(row,verify_hashes=verify_hashes) if template else
              ThoracicCoreProvider(row,selection,evidence,verify_hashes=verify_hashes))
    shape=tuple(grid['native_shape_kji']);origin=np.asarray(grid['native_origin_ijk'],int)
    affine=np.asarray(grid['native_affine_ras_ijk'],float)
    count_slices=min(slab_slices,MAX_BLOCK_VOXELS//(shape[1]*shape[2]))
    if count_slices<1:raise ValueError('Native report crop exceeds bounded slice reads')
    staged=[];counts={k:0 for k in paths if k!='image'}
    try:
        with ExitStack() as stack:
            writers={}
            for key,path in paths.items():
                writer,temp,final=_open_writer(stack,path,shape,affine,
                    np.float32 if key=='image' else np.uint8,
                    'FakeCT fixed native atlas thoracic ROI; common matched-arm '+key)
                writers[key]=writer;staged.append((temp,final))
            volume=stack.enter_context(ContextVolume(row,verify_hashes=verify_hashes))
            for z in range(0,shape[0],count_slices):
                n=min(count_slices,shape[0]-z);point=origin+[0,0,z]
                block_shape=(n,shape[1],shape[2])
                mu=volume.read_block(point[::-1],block_shape,row['spacing_ijk_mm'])['image']
                hu=attenuation_to_hu(mu,evidence['calibration']['mu_water_cm_inverse'])
                writers['image'].write(np.asarray(hu,'<f4').tobytes(order='C'))
                masks=provider.read_block(point,block_shape)
                for key in counts:
                    data=masks[key].astype(np.uint8)
                    writers[key].write(data.tobytes(order='C'));counts[key]+=int(data.sum())
        for temp,final in staged:os.link(temp,final);temp.unlink()
    except Exception:
        for temp,_ in staged:temp.unlink(missing_ok=True)
        raise
    result={'sample_id':row['sample_id'],'case_id':row['case_id'],
        'input_nifti':str(paths['image']),'truth_nifti':str(paths['target']),
        'domain_nifti':str(paths['domain']),'changed_nifti':str(paths['changed']),
        'export':{'origin_ijk':origin.tolist(),'shape_kji':list(shape),'affine_ras_ijk':affine.tolist(),
                  'voxel_counts':counts,'crop_scope':grid['scope']},
        'artifacts':{str(path):{'sha256':_sha(path),'bytes':path.stat().st_size} for path in paths.values()}}
    extent=native_core_extent(row,selection,evidence,template=template,verify_hashes=verify_hashes)
    if (counts['target']!=extent['current_core_voxels'] or
            counts['changed']!=extent['changed_voxels']):
        raise ValueError('Frozen region misses native core/changes; increase the predefined region before freezing')
    result.update(extent)
    for key in ('input','truth','domain','changed'):
        result[key+'_nifti_sha256']=result['artifacts'][result[key+'_nifti']]['sha256']
    if export_baseline:
        result['baseline_nifti']=str(paths['baseline'])
        result['baseline_nifti_sha256']=result['artifacts'][str(paths['baseline'])]['sha256']
    return result


def export_native_reference(*args,**kwargs):
    """Named API for common native CT/truth/domain/change report inputs."""
    return export_case_native_inputs(*args,**kwargs)


def model_to_native_mask(model_mask,grid):
    """Restore the complete fixed native ROI from a canonical model-grid mask."""
    return restore_native_mask(model_mask,grid)


def _dice(truth,prediction,region=None):
    truth,prediction=np.asarray(truth,bool),np.asarray(prediction,bool)
    if region is not None: truth=truth&region;prediction=prediction&region
    denominator=int(truth.sum()+prediction.sum())
    return 2*int((truth&prediction).sum())/denominator if denominator else 1.


def _connectivity(mask):
    components,count=label(mask,generate_binary_structure(3,1))
    values=np.bincount(components.ravel())[1:]
    return {'component_count_6':int(count),
            'largest_component_fraction':float(values.max()/values.sum()) if len(values) else 0.}


def audit_resampling_ceiling(volume,*,local_radius_mm=5.,max_native_voxels=MAX_NATIVE_AUDIT_VOXELS):
    """Native GT -> model nearest mask -> native; quantify geometry lost by grid."""
    provider=volume.provider
    low=np.minimum(provider.core_origin,provider.edit_origin)
    high=np.maximum(provider.core_origin+np.asarray(provider.baseline_core.shape)[::-1],
                    provider.edit_origin+np.asarray(provider.current_edit_core.shape)[::-1])
    shape=(high-low)[::-1]
    if type(max_native_voxels) is not int or not 0<max_native_voxels<=128_000_000:
        raise ValueError('Native audit budget must be a positive integer at most 128 million')
    if math.prod(shape)>max_native_voxels:
        raise ValueError('Native resampling audit exceeds its declared voxel budget')
    native={k:np.zeros(tuple(shape),np.uint8) for k in ('target','changed','domain')}
    for z in range(0,int(shape[0]),8):
        n=min(8,int(shape[0])-z);origin=low+[0,0,z]
        block=provider.read_block(origin,[n,int(shape[1]),int(shape[2])])
        for k in native:native[k][z:z+n]=block[k]
    model=volume.masks_full()['target']
    roundtrip=np.zeros(tuple(shape),np.uint8)
    native_spacing=np.asarray(volume.row['spacing_ijk_mm'])[::-1]
    model_spacing=np.asarray(volume.grid['spacing_ijk_mm'])[::-1]
    origin_model=np.asarray(volume.grid['native_origin_ijk'])[::-1]
    high_model=origin_model+np.asarray(volume.grid['native_shape_kji'])
    signs=np.asarray(volume.grid['native_to_model_index_sign_ijk'])[::-1]
    coordinates=[np.floor(((np.arange(int(n))+int(l)+.5-int(o)) if sign>0 else
                            (int(h)-np.arange(int(n))-int(l)-.5))*ns/ms).astype(int)
                 for n,l,o,h,ns,ms,sign in zip(shape,low[::-1],origin_model,high_model,
                                              native_spacing,model_spacing,signs)]
    valid=[(c>=0)&(c<n) for c,n in zip(coordinates,model.shape)]
    clipped=[np.clip(c,0,n-1) for c,n in zip(coordinates,model.shape)]
    for z in range(0,int(shape[0]),8):
        n=min(8,int(shape[0])-z)
        values=model[np.ix_(clipped[0][z:z+n],clipped[1],clipped[2])].copy()
        values*=valid[0][z:z+n,None,None];values*=valid[1][None,:,None];values*=valid[2][None,None,:]
        roundtrip[z:z+n]=values
    roundtrip*=native['domain']
    truth=native['target'].astype(bool)
    lost=int(np.sum(truth&~roundtrip.astype(bool)))
    outside_model=int(np.sum(truth & ~(valid[0][:,None,None]&valid[1][None,:,None]&valid[2][None,None,:])))
    local=None
    if native['changed'].any():
        local=(distance_transform_edt(native['changed']==0,sampling=native_spacing)<=local_radius_mm)&native['domain'].astype(bool)
    return {'sample_id':volume.row['sample_id'],'model_spacing_ijk_mm':volume.grid['spacing_ijk_mm'],
        'native_target_voxels':int(truth.sum()),'roundtrip_target_voxels':int(roundtrip.sum()),
        'native_gt_outside_fixed_model_crop_voxels':outside_model,
        'native_gt_lost_voxels':lost,'global_dice_ceiling':_dice(truth,roundtrip),
        'local_change_dice_ceiling':_dice(truth,roundtrip,local) if local is not None else None,
        'local_radius_mm':float(local_radius_mm),'native_connectivity':_connectivity(truth),
        'roundtrip_connectivity':_connectivity(roundtrip),
        'max_native_audit_voxels':max_native_voxels,
        'scope':'nearest native GT -> declared model grid -> native; geometry representability, not learned accuracy'}
