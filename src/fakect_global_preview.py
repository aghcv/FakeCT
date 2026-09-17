"""Bounded whole-phantom navigation with exact native coordinate references.

The global navigator samples original volumes; it is an anatomical locator,
not an edit result or a replacement for the native-grid local previews. Binary
sources are opened read-only and only one axial plane is retained at a time.
"""
import base64
import hashlib
import json
import os
from pathlib import Path

import numpy as np

from fakect_roi import COLORS
from fakect_tissues import coarse_labels


MAX_GLOBAL_VOXELS = 1_500_000
MAX_PLANE_BYTES = 256 * 1024 ** 2
ROI_COLORS = ('#ed7c21', '#7060d8', '#16a37b', '#de508c', '#c2a300', '#238ec7')
PLANES = ((2, 0, 1, 'Axial'), (1, 0, 2, 'Coronal'), (0, 1, 2, 'Sagittal'))


def sampling_axes(shape_kji, spacing_ijk_mm, max_voxels=MAX_GLOBAL_VOXELS):
    """Approximately isotropic sampling, always retaining both source edges.

    Returned axes are in i,j,k order, contain exact native voxel indices, and
    need not have a uniform last interval. They are the authoritative mapping.
    """
    shape = np.asarray(shape_kji)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if (shape.shape != (3,) or shape.dtype.kind not in 'iu' or np.any(shape <= 0)
            or spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0)):
        raise ValueError('Global sampling requires positive k,j,i dimensions and finite i,j,k spacing')
    if (isinstance(max_voxels, bool) or not isinstance(max_voxels, (int, np.integer))
            or max_voxels < int(np.prod(np.minimum(shape, 2)))):
        raise ValueError('Global sample budget must retain both edges of every axis')
    shape_ijk = shape[::-1]
    step_mm = float(max(spacing.min(), (float(np.prod(shape)) * float(np.prod(spacing))
                                       / max_voxels) ** (1 / 3)))
    while True:
        strides = np.maximum(1, np.floor(step_mm / spacing).astype(np.int64))
        axes = [np.unique(np.append(np.arange(0, int(n), int(s), dtype=np.int64), int(n)-1))
                for n, s in zip(shape_ijk, strides)]
        if int(np.prod([len(a) for a in axes])) <= max_voxels:
            return axes, tuple(map(int, strides))
        step_mm *= 1.04


def _stat(path):
    value = Path(path).stat()
    return {'bytes': value.st_size, 'mtime_ns': value.st_mtime_ns,
            'ctime_ns': value.st_ctime_ns, 'device': value.st_dev, 'inode': value.st_ino}


def read_sampled(path, shape_kji, axes_ijk):
    """Read selected whole axial planes, copying only the bounded sample grid."""
    path = Path(path)
    before = _stat(path)
    plane_size = int(shape_kji[1]) * int(shape_kji[2])
    if before['bytes'] != int(np.prod(shape_kji)) * 4:
        raise ValueError(f'Unexpected binary source size: {path}')
    if plane_size * 4 > MAX_PLANE_BYTES:
        raise ValueError('Source axial plane exceeds the bounded global reader limit')
    if len(axes_ijk) != 3:
        raise ValueError('Global sample requires three coordinate axes')
    for axis, n in zip(axes_ijk, shape_kji[::-1]):
        axis = np.asarray(axis)
        if (axis.ndim != 1 or not len(axis) or axis.dtype.kind not in 'iu'
                or axis[0] < 0 or axis[-1] >= n or np.any(np.diff(axis) <= 0)):
            raise ValueError('Global sample axes must be sorted distinct in-range native indices')
    sample_shape = tuple(len(a) for a in axes_ijk[::-1])
    if int(np.prod(sample_shape)) > MAX_GLOBAL_VOXELS:
        raise ValueError('Global sample exceeds the voxel budget')
    result = np.empty(sample_shape, dtype='<f4')
    ii, jj, kk = axes_ijk
    with path.open('rb', buffering=0) as stream:
        for index, k in enumerate(kk):
            stream.seek(int(k) * plane_size * 4)
            plane = np.fromfile(stream, dtype='<f4', count=plane_size)
            if plane.size != plane_size:
                raise ValueError(f'Short source read: {path}')
            result[index] = plane.reshape(shape_kji[1:])[np.ix_(jj, ii)]
    if before != _stat(path):
        raise ValueError(f'Source changed during global preview: {path}')
    if not np.all(np.isfinite(result)):
        raise ValueError(f'Nonfinite values in global source sample: {path}')
    return result, {'path': str(path), **before,
                    'source_bytes_read': len(kk) * plane_size * 4,
                    'sample_sha256': hashlib.sha256(result.tobytes()).hexdigest(),
                    'sample_dtype': '<f4',
                    'integrity_scope': 'Sample content hash and source stat; full source volume not hashed'}


def roi_definitions(config):
    """Serializable original geometries; named regions remain clipped to main."""
    rows = []
    for index, (name, definition) in enumerate([('main', config['roi']), *config.get('rois', {}).items()]):
        nodes = np.asarray(definition['center_ijk'], dtype=float)
        if nodes.ndim == 1:
            nodes = nodes[None, :]
        rows.append({'name': name, 'shape': definition.get('shape', 'sphere'),
                     'nodes_ijk': nodes.tolist(),
                     'radii_mm': np.atleast_1d(definition['radius_mm']).astype(float).tolist(),
                     'color': '#ffffff' if index == 0 else ROI_COLORS[(index-1) % len(ROI_COLORS)],
                     'clipped_to_main': index > 0})
    return rows


def sampled_roi_mask(axes_ijk, spacing_ijk_mm, roi):
    """Evaluate sphere/tube geometry exactly at supplied native sample centers."""
    ii, jj, kk = [np.asarray(a, dtype=float) for a in axes_ijk]
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    positions = [ii[None, None, :] * spacing[0], jj[None, :, None] * spacing[1],
                 kk[:, None, None] * spacing[2]]
    nodes = np.asarray(roi['nodes_ijk'], dtype=float) * spacing
    radii = np.asarray(roi['radii_mm'], dtype=float)
    if roi['shape'] == 'sphere':
        return sum((v - nodes[0, axis]) ** 2 for axis, v in enumerate(positions)) <= radii[0] ** 2
    result = np.zeros((len(kk), len(jj), len(ii)), dtype=bool)
    for start, end, r0, r1 in zip(nodes[:-1], nodes[1:], radii[:-1], radii[1:]):
        delta, dr = end-start, r1-r0
        displacement = [v-start[axis] for axis, v in enumerate(positions)]
        squared = sum(v*v for v in displacement)
        projection = sum(v*delta[axis] for axis, v in enumerate(displacement)) + r0*dr
        constant = squared-r0*r0
        quadratic = float(np.dot(delta, delta)-dr*dr)
        minimum = np.minimum(constant, quadratic-2*projection+constant)
        if quadratic > 0:
            t = np.clip(projection/quadratic, 0, 1)
            minimum = np.minimum(minimum, (quadratic*t-2*projection)*t+constant)
        tolerance = 64*np.finfo(float).eps*np.maximum(1, squared+r0*r0+r1*r1)
        result |= minimum <= tolerance
    return result


def prepare_global(resolved, config, max_voxels=MAX_GLOBAL_VOXELS):
    """Return bounded sampled source arrays and auditable coordinate metadata."""
    axes, strides = sampling_axes(resolved['shape_kji'], resolved['spacing_ijk_mm'], max_voxels)
    arrays, sources = {}, {}
    initial = {key: _stat(path) for key, path in resolved['source_files'].items()}
    for channel in ('act', 'atn'):
        arrays[channel], sources[channel] = read_sampled(resolved['source_files'][channel],
                                                        resolved['shape_kji'], axes)
    for channel, before in initial.items():
        if before != _stat(resolved['source_files'][channel]):
            raise ValueError(f'Source changed while preparing global channels: {channel}')
    arrays['tissue'] = coarse_labels(arrays['act'], resolved['catalog'])
    arrays['candidates'] = np.isin(arrays['act'], resolved['source_ids'])
    arrays['attenuation_cm_inverse'] = arrays['atn'] / (resolved['spacing_ijk_mm'][0]/10)
    sample = arrays['attenuation_cm_inverse']
    foreground = sample[sample > 0]
    values = foreground if foreground.size else sample.ravel()
    lo, hi = float(min(0, values.min())), float(np.percentile(values, 99.5))
    if hi <= lo:
        hi = lo+1
    arrays['display'] = np.rint(np.clip((sample-lo)/(hi-lo), 0, 1)*255).astype(np.uint8)
    focus = resolved.get('slice_ijk', resolved.get('focus_ijk', tuple(n//2 for n in resolved['shape_kji'][::-1])))
    crosshair = [int(np.argmin(np.abs(axis-float(value)))) for axis, value in zip(axes, focus)]
    metadata = {'schema_version': 'fakect.global-preview/1',
                'shape_kji': list(resolved['shape_kji']),
                'spacing_ijk_mm': list(resolved['spacing_ijk_mm']),
                'sample_shape_kji': list(arrays['act'].shape),
                'sample_voxels': int(arrays['act'].size), 'max_sample_voxels': int(max_voxels),
                'strides_ijk': list(strides), 'axes_ijk': [a.tolist() for a in axes],
                'sampling': 'Nearest native voxel samples, approximately isotropic; first and last voxel retained on every axis. Final intervals may differ from the nominal stride.',
                'sources': sources, 'catalog_sha256': resolved.get('catalog_sha256'),
                'selection': {'tissue': config['selection']['tissue'],
                              'source_ids_explicit': bool(config['selection'].get('source_ids')),
                              'resolved_source_id_count': len(resolved['source_ids']),
                              'sampled_candidate_voxels': int(arrays['candidates'].sum())},
                'rois': roi_definitions(config),
                'initial_crosshair_ijk': [int(axis[index]) for axis, index in zip(axes, crosshair)],
                'initial_crosshair_sample_ijk': crosshair,
                'display_window_cm_inverse': [lo, hi],
                'warnings': ['Global views are sampled anatomical locators. Thin structures may be missed; review the regenerated native-grid local views before editing.',
                             'Moving the crosshair or temporary sphere changes this navigator only. Copy coordinates into the INI and rerun to regenerate the local views and edits.']}
    return arrays, metadata


def _plane_masks(axes, spacing, rois, fixed, level):
    plane_axes = [np.asarray(a) for a in axes]
    plane_axes[fixed] = plane_axes[fixed][level:level+1]
    masks = [np.squeeze(sampled_roi_mask(plane_axes, spacing, roi), axis=2-fixed) for roi in rois]
    return [masks[0], *(mask & masks[0] for mask in masks[1:])]


def _render_static(arrays, metadata, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    from matplotlib.lines import Line2D

    axes = [np.asarray(axis) for axis in metadata['axes_ijk']]
    spacing = metadata['spacing_ijk_mm']
    focus = metadata['initial_crosshair_sample_ijk']
    fig, panels = plt.subplots(1, 3, figsize=(18, 9), constrained_layout=True)
    for ax, (fixed, x, y, name) in zip(panels, PLANES):
        image = np.take(arrays['display'], focus[fixed], axis=2-fixed)
        ax.pcolormesh(axes[x], axes[y], image, cmap='gray', vmin=0, vmax=255,
                      shading='nearest', rasterized=True)
        candidates = np.take(arrays['candidates'], focus[fixed], axis=2-fixed)
        if candidates.any():
            ax.pcolormesh(axes[x], axes[y], np.ma.masked_where(~candidates, candidates),
                          cmap=ListedColormap(['#00efff']), alpha=.55, shading='nearest', rasterized=True)
        masks = _plane_masks(axes, spacing, metadata['rois'], fixed, focus[fixed])
        for roi, mask in zip(metadata['rois'], masks):
            if mask.any():
                ax.pcolormesh(axes[x], axes[y], np.ma.masked_where(~mask, mask),
                              cmap=ListedColormap([roi['color']]), alpha=.25,
                              shading='nearest', rasterized=True)
                if not mask.all() and min(mask.shape) > 1:
                    ax.contour(axes[x], axes[y], mask, levels=[.5], colors=[roi['color']], linewidths=1)
            nodes = np.asarray(roi['nodes_ijk'])
            ax.plot(nodes[:, x], nodes[:, y], color=roi['color'], marker='.', linewidth=.7, alpha=.85)
        native = metadata['initial_crosshair_ijk']
        ax.axvline(native[x], color='#ffe06a', linewidth=.65, linestyle='--')
        ax.axhline(native[y], color='#ffe06a', linewidth=.65, linestyle='--')
        ax.set_xlim(-.5, metadata['shape_kji'][2-x]-.5)
        ax.set_ylim(-.5, metadata['shape_kji'][2-y]-.5)
        ax.set_aspect(spacing[y]/spacing[x])
        ax.set_xlabel('ijk'[x]+' (native voxel index)')
        ax.set_ylabel('ijk'[y]+' (native voxel index)')
        ax.set_title(f'{name}: {"ijk"[fixed]} = {native[fixed]}')
    handles = [Line2D([], [], color='#00efff', label='Selected tissue / optional IDs')]
    handles.extend(Line2D([], [], color=r['color'] if r['color'] != '#ffffff' else '#666666',
                          label=r['name']+' ROI (centerline projected)') for r in metadata['rois'])
    fig.legend(handles=handles, loc='outside lower center' if tuple(map(int, matplotlib.__version__.split('.')[:2])) >= (3, 7)
               else 'lower center', ncol=min(len(handles), 4), fontsize=8)
    fig.suptitle('Whole-phantom context · sampled source anatomy · ROI outlines at sampled planes\n'
                 'Use Global navigator to move slices; use Local views for native-resolution review', fontsize=13)
    fig.savefig(output/'roi-global.png', dpi=125, facecolor='white')
    plt.close(fig)


def _encoded(array):
    return base64.b64encode(np.ascontiguousarray(array, dtype=np.uint8).tobytes()).decode('ascii')


def render_global_preview(resolved, config, output):
    """Write a portable full-volume navigator and PNG; return report metadata."""
    arrays, metadata = prepare_global(resolved, config)
    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    _render_static(arrays, metadata, output)
    payload = {**metadata, 'gray': _encoded(arrays['display']),
               'tissue': _encoded(arrays['tissue']), 'target': _encoded(arrays['candidates']),
               'categories': [{'id': c['id'], 'name': c['name'], 'color': COLORS.get(c['name'], '#ff00cc')}
                              for c in resolved['catalog']['categories']]}
    # Escape HTML delimiters even when metadata includes user-supplied names.
    packed = json.dumps(payload, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')
    (output/'roi-global.html').write_text(_HTML.replace('__GLOBAL_DATA__', packed), encoding='utf-8')
    for channel, source in metadata['sources'].items():
        if {key: source[key] for key in _stat(source['path'])} != _stat(source['path']):
            raise ValueError(f'Source changed while rendering global preview: {channel}')
    metadata.update(html='roi-global.html', figure='roi-global.png')
    return metadata


_HTML = r'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>FakeCT whole-phantom navigator</title>
<style>
:root{color-scheme:dark;font-family:system-ui,sans-serif;background:#111a23;color:#e8f1f7}
body{margin:0;padding:16px}h1{font-size:20px;margin:0 0 8px}p{line-height:1.5;margin:8px 0}
.muted{color:#bbcedc;font-size:13px}.controls{display:flex;gap:15px;align-items:center;flex-wrap:wrap;background:#1d2b38;padding:12px;border-radius:8px;margin:12px 0}
label{font-size:13px}input,select,button,textarea{font:inherit;color:inherit;background:#102230;border:1px solid #648097;border-radius:4px;padding:5px}
button{cursor:pointer;background:#1d4860}input[type=number]{width:65px}input[type=range]{padding:0;vertical-align:middle;width:150px}
.views{display:grid;grid-template-columns:repeat(3,minmax(0,1fr));gap:12px}.view{background:#070b10;border:1px solid #3c5366;border-radius:8px;padding:10px;min-width:0}
h2{font-size:15px;margin:0 0 8px}.frame{height:420px;display:flex;align-items:center;justify-content:center;overflow:hidden}
canvas{max-width:100%;max-height:100%;image-rendering:pixelated;cursor:crosshair;touch-action:none;background:#000}
.slider{display:flex;align-items:center;gap:5px;margin:8px 0}.slider input{width:100%}.slider output{min-width:60px}
.legend{display:flex;flex-wrap:wrap;gap:12px;margin:10px 0;font-size:12px}.swatch{display:inline-block;width:11px;height:11px;margin-right:4px;border:1px solid #777}
textarea{box-sizing:border-box;width:100%;min-height:92px;font-family:monospace}.copy{display:flex;align-items:center;gap:12px;flex-wrap:wrap;margin-top:8px}
#readout{font-family:monospace;font-size:16px;color:#ffe06a}details{margin:12px 0}summary{cursor:pointer}a{color:#8ed5ff}
@media(max-width:760px){.views{grid-template-columns:1fr}.frame{height:370px}}
</style></head><body>
<h1>Global · whole-phantom navigator</h1>
<p class="muted">Start with a tissue type; surface IDs are optional. Move the slices or click an image to locate anatomy in native <b>i, j, k</b> coordinates. Cyan marks the selected tissue, optionally narrowed by IDs.</p>
<div class="controls"><label>View <select id="mode"><option value="attenuation">Attenuation</option><option value="tissue">Tissue categories</option></select></label>
<label><input type="checkbox" id="target" checked> Selected tissue</label><label><input type="checkbox" id="rois" checked> Saved ROIs</label>
<label><input type="checkbox" id="guide" checked> Temporary sphere guide</label>
<label>Guide radius <input type="number" id="radius" min="0.1" step="0.5" value="10"> mm</label>
<button id="reset" type="button">Return to saved ROI</button></div>
<p id="readout" aria-live="polite"></p>
<div class="views" id="views"></div>
<div class="legend" id="roiLegend"></div>
<p class="muted" id="hover">Click an image to position the crosshair and temporary sphere. Dashed crosshair = proposed center; saved ROI geometry remains fixed.</p>
<div class="controls"><span>Go to approximate native index:</span>
<label>i <input type="number" id="input-i" min="0" step="1"></label><label>j <input type="number" id="input-j" min="0" step="1"></label><label>k <input type="number" id="input-k" min="0" step="1"></label>
<button type="button" id="go">Go to nearest sampled voxel</button></div>
<label for="coordinateText">Copy this point to the relevant sphere center or tube node in your INI:</label>
<textarea id="coordinateText" readonly></textarea><div class="copy"><button type="button" id="copy">Copy coordinates</button><span id="copyStatus" class="muted" aria-live="polite"></span></div>
<p class="muted"><b>Exploratory guide only.</b> Moving this crosshair or sphere does not edit the INI, saved ROIs, local images, or anatomy. For a tube, update the intended node in its ordered center_ijk list and its matching radius; preserve the other nodes. Save the INI, rerun the preview, then use Local views for detailed review.</p>
<details><summary>Sampling and tissue legend</summary><p class="muted" id="sampling"></p><div id="tissueLegend" class="legend"></div>
<p class="muted">Global images use bounded, approximately isotropic samples from the whole source volume. Small vessels can fall between samples. The native local ROI views determine the actual selection and editing. Named ROI overlays are clipped to the main ROI.</p></details>
<script id="global-data" type="application/json">__GLOBAL_DATA__</script>
<script>
'use strict';
const D=JSON.parse(document.getElementById('global-data').textContent), axes=D.axes_ijk, spacing=D.spacing_ijk_mm;
const decode=s=>Uint8Array.from(atob(s),c=>c.charCodeAt(0));
const gray=decode(D.gray), tissue=decode(D.tissue), target=decode(D.target), dims=D.sample_shape_kji.slice().reverse();
delete D.gray;delete D.tissue;delete D.target;
const hex=s=>[parseInt(s.slice(1,3),16),parseInt(s.slice(3,5),16),parseInt(s.slice(5,7),16)];
const palette=new Map(D.categories.map(c=>[c.id,hex(c.color)])), names=new Map(D.categories.map(c=>[c.id,c.name]));
const roiColors=D.rois.map(r=>hex(r.color)), cursor=D.initial_crosshair_sample_ijk.slice();
D.rois.forEach(r=>{const radius=Math.max(...r.radii_mm);r.bounds=[0,1,2].map(a=>[Math.min(...r.nodes_ijk.map(n=>n[a]))-radius/spacing[a],Math.max(...r.nodes_ijk.map(n=>n[a]))+radius/spacing[a]]);});
const spec=[[2,0,1,'Axial'],[1,0,2,'Coronal'],[0,1,2,'Sagittal']], views=[];
const el=id=>document.getElementById(id), native=()=>cursor.map((v,a)=>axes[a][v]);
const index=p=>(p[2]*dims[1]+p[1])*dims[0]+p[0];
function nearest(axis,value){let lo=0,hi=axis.length-1;while(lo<hi){let mid=Math.floor((lo+hi)/2);if(axis[mid]<value)lo=mid+1;else hi=mid;}return lo>0&&Math.abs(axis[lo-1]-value)<=Math.abs(axis[lo]-value)?lo-1:lo;}
function inside(p,roi){
 if(p.some((value,a)=>value<roi.bounds[a][0]||value>roi.bounds[a][1]))return false;
 const points=roi.nodes_ijk,rs=roi.radii_mm;
 const dist=(q)=>p.reduce((s,v,a)=>s+((v-q[a])*spacing[a])**2,0);
 if(roi.shape==='sphere')return dist(points[0])<=rs[0]*rs[0];
 for(let n=0;n<points.length-1;n++){
  const start=points[n],end=points[n+1],delta=end.map((v,a)=>(v-start[a])*spacing[a]);
  const disp=p.map((v,a)=>(v-start[a])*spacing[a]),r0=rs[n],r1=rs[n+1],dr=r1-r0;
  const squared=disp.reduce((s,v)=>s+v*v,0),q=delta.reduce((s,v)=>s+v*v,0)-dr*dr;
  const projection=disp.reduce((s,v,a)=>s+v*delta[a],0)+r0*dr,c=squared-r0*r0;
  let minimum=Math.min(c,q-2*projection+c);
  if(q>0){const t=Math.max(0,Math.min(1,projection/q));minimum=Math.min(minimum,(q*t-2*projection)*t+c);}
  if(minimum<=64*Number.EPSILON*Math.max(1,squared+r0*r0+r1*r1))return true;
 }return false;
}
function legend(container,color,label){const span=document.createElement('span'),swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=color;span.append(swatch,document.createTextNode(label));container.append(span);}
legend(el('roiLegend'),'#00efff',D.selection.source_ids_explicit?'Selected '+D.selection.tissue+' (IDs restricted)':'All '+D.selection.tissue+' labels');
D.rois.forEach(r=>legend(el('roiLegend'),r.color,r.name+' ROI'));
legend(el('roiLegend'),'#ffe06a','Temporary sphere / crosshair');
D.categories.forEach(c=>legend(el('tissueLegend'),c.color,c.name));
spec.forEach(([fixed,x,y,name])=>{
 const box=document.createElement('section');box.className='view';const title=document.createElement('h2');title.textContent=name+' · '+ 'ijk'[x]+' / '+'ijk'[y];
 const frame=document.createElement('div');frame.className='frame';const canvas=document.createElement('canvas');canvas.width=dims[x];canvas.height=dims[y];
 const physicalWidth=(D.shape_kji[2-x]-1||1)*spacing[x],physicalHeight=(D.shape_kji[2-y]-1||1)*spacing[y];
 canvas.style.aspectRatio=String(physicalWidth/physicalHeight);canvas.style.width=physicalWidth>=physicalHeight?'100%':'auto';canvas.style.height=physicalHeight>physicalWidth?'100%':'auto';
 canvas.setAttribute('aria-label',name+' sampled whole-volume slice; click to position crosshair');frame.append(canvas);
 const sliderBox=document.createElement('label');sliderBox.className='slider';const slider=document.createElement('input');slider.type='range';slider.min=0;slider.max=dims[fixed]-1;slider.step=1;slider.value=cursor[fixed];slider.setAttribute('aria-label',name+' native '+'ijk'[fixed]+' slice');const label=document.createElement('output');
 sliderBox.append(document.createTextNode('ijk'[fixed]),slider,label);box.append(title,frame,sliderBox);el('views').append(box);
 const view={fixed,x,y,name,canvas,slider,label};views.push(view);slider.oninput=()=>{cursor[fixed]=Number(slider.value);draw();};
 function position(event){const r=canvas.getBoundingClientRect(),p=cursor.slice();p[x]=Math.max(0,Math.min(dims[x]-1,Math.floor((event.clientX-r.left)/r.width*dims[x])));p[y]=Math.max(0,Math.min(dims[y]-1,dims[y]-1-Math.floor((event.clientY-r.top)/r.height*dims[y])));return p;}
 canvas.addEventListener('click',event=>{const p=position(event);cursor[x]=p[x];cursor[y]=p[y];draw();});
 canvas.addEventListener('mousemove',event=>{const p=position(event),n=p.map((v,a)=>axes[a][v]);el('hover').textContent='Under pointer: i,j,k = '+n.join(', ')+' · '+(names.get(tissue[index(p)])||'unknown');});
});
function draw(){
 const center=native(),radius=Number(el('radius').value),showRois=el('rois').checked,showTarget=el('target').checked,mode=el('mode').value;
 views.forEach(v=>{
  const ctx=v.canvas.getContext('2d'),image=ctx.createImageData(dims[v.x],dims[v.y]),data=image.data,p=cursor.slice();
  for(let row=0;row<dims[v.y];row++)for(let col=0;col<dims[v.x];col++){
   p[v.x]=col;p[v.y]=row;const n=index(p),offset=((dims[v.y]-1-row)*dims[v.x]+col)*4;
   let color=mode==='tissue'?(palette.get(tissue[n])||[255,0,204]):[gray[n],gray[n],gray[n]];
   function blend(rgb,alpha){color=color.map((value,a)=>value*(1-alpha)+rgb[a]*alpha);}
   if(showTarget&&target[n])blend([0,239,255],.6);
   if(showRois){const point=p.map((s,a)=>axes[a][s]);if(inside(point,D.rois[0])){blend(roiColors[0],.16);for(let r=1;r<D.rois.length;r++)if(inside(point,D.rois[r]))blend(roiColors[r],.34);}}
   data[offset]=color[0];data[offset+1]=color[1];data[offset+2]=color[2];data[offset+3]=255;
  }
  ctx.putImageData(image,0,0);ctx.save();ctx.strokeStyle='#ffe06a';ctx.lineWidth=.8;ctx.setLineDash([3,3]);
  const cx=cursor[v.x]+.5,cy=dims[v.y]-cursor[v.y]-.5;ctx.beginPath();ctx.moveTo(cx,0);ctx.lineTo(cx,dims[v.y]);ctx.moveTo(0,cy);ctx.lineTo(dims[v.x],cy);ctx.stroke();
  if(el('guide').checked&&Number.isFinite(radius)&&radius>0){
   const width=(D.shape_kji[2-v.x]-1||1)*spacing[v.x],height=(D.shape_kji[2-v.y]-1||1)*spacing[v.y];
   ctx.setLineDash([]);ctx.beginPath();ctx.ellipse(cx,cy,radius/width*Math.max(1,dims[v.x]-1),radius/height*Math.max(1,dims[v.y]-1),0,0,Math.PI*2);ctx.fillStyle='rgba(255,224,106,.12)';ctx.fill();ctx.stroke();
  }ctx.restore();v.slider.value=cursor[v.fixed];v.label.textContent='= '+center[v.fixed];
 });
 el('readout').textContent='Native center i, j, k = '+center.join(', ')+'  |  physical index × spacing = '+center.map((v,a)=>(v*spacing[a]).toFixed(2)).join(', ')+' mm (source grid origin)';
 ['i','j','k'].forEach((name,a)=>{el('input-'+name).value=center[a];el('input-'+name).max=D.shape_kji[2-a]-1;});
 el('coordinateText').value='# Proposed native point; update the intended sphere center or tube node.\ncenter_ijk = '+center.join(', ')+'\n# Temporary sphere radius; update the matching tube radius separately if needed.\nradius_mm = '+(Number.isFinite(radius)&&radius>0?radius:10);
 el('copyStatus').textContent='';
}
['mode','target','rois','guide','radius'].forEach(id=>el(id).addEventListener('input',draw));
el('go').onclick=()=>{['i','j','k'].forEach((name,a)=>{const value=Number(el('input-'+name).value);if(Number.isFinite(value))cursor[a]=nearest(axes[a],value);});draw();};
el('reset').onclick=()=>{D.initial_crosshair_sample_ijk.forEach((v,a)=>cursor[a]=v);draw();};
el('copy').onclick=async()=>{const field=el('coordinateText');field.select();try{await navigator.clipboard.writeText(field.value);el('copyStatus').textContent='Copied. Paste into your INI, save, and rerun for Local views.';}catch(error){el('copyStatus').textContent='Text selected. Use Ctrl+C / Command+C, then paste into your INI.';}};
el('sampling').textContent='Full shape k,j,i: '+D.shape_kji.join(' × ')+'; sample: '+D.sample_shape_kji.join(' × ')+' ('+D.sample_voxels.toLocaleString()+' voxels). Nominal native strides i,j,k: '+D.strides_ijk.join(', ')+'. Both edges are retained; the exact axes are embedded in this HTML. Attenuation display window: '+D.display_window_cm_inverse.map(v=>v.toFixed(4)).join('–')+' cm⁻¹. '+D.sampling;
draw();
</script></body></html>'''
