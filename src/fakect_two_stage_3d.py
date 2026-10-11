"""Offline, lazy-loaded 3D views of saved masks on their shared physical grid.

Surfaces use the existing voxel-boundary renderer at exact model resolution.
Packed meshes contain no CT volumes. One scene is loaded at a time through a
local script, so the viewer works with file:// without a web server or CDN.
"""
import base64
import hashlib
from html import escape
import json
from pathlib import Path

import numpy as np

from fakect_surface_overlay import OPACITY_PRESETS, _surface_edge_count
from fakect_volume_preview import occupancy_grid, _surface

MAX_VOXELS = 16_777_216
MAX_SURFACE_TRIANGLES = 180_000
MAX_SCENE_TRIANGLES = 600_000
MAX_SCENES = 512
MAX_SCENE_BYTES = 32 * 1024 * 1024
DEFAULT_COORDINATES = 'Model-grid millimeters from the lower crop face; patient orientation is unverified.'
DEFAULT_FOOTER = ('Surfaces use exact Boolean occupancy on the saved model grid, with crop-edge closures. '
                  'They are not attenuation volume renderings or native XCAT coordinates. No surface smoothing or pooling is applied. '
                  'Transparency can hide overlapping surfaces: isolate labels with the controls to inspect them. Empty masks are listed with zero voxels. '
                  'Counts and saved scores use the full model crop. The ROI-selected training crop is not a verified whole-thoracic-aorta segmentation.')


def _json(value):
    return json.dumps(value, separators=(',', ':'), allow_nan=False).replace('<', '\\u003c').replace('>', '\\u003e').replace('&', '\\u0026')


def _packed(array, dtype):
    return base64.b64encode(np.ascontiguousarray(array, dtype=dtype).tobytes()).decode('ascii')


def build_scene(masks, spacing_ijk_mm, *, sample_id, modes=None, metadata=None,
                coordinate_description=None, footer_note=None, show_metrics=None):
    """Build exact surfaces for named Boolean masks; no resampling or smoothing.

    ``masks`` maps stable field names to arrays. ``modes`` is an optional list
    of dictionaries with key, label, explanation and layers. Each layer has a
    mask key, label, color, visible flag, and an opacity from OPACITY_PRESETS.
    All coordinates are millimeters from the lower model-crop face, not patient
    anatomical orientation. Over-budget surfaces are explicitly omitted.
    Optional display arguments override the viewer-wide descriptions for this
    scene. ``show_metrics=False`` hides model scores for native anatomy reviews.
    Descriptions do not transform the underlying coordinates.
    """
    if not masks:
        raise ValueError('At least one mask is required for a 3D scene')
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('3D spacing requires three finite positive values')
    first = np.asarray(next(iter(masks.values())))
    shape = first.shape
    if first.ndim != 3 or min(shape) < 2 or first.size > MAX_VOXELS:
        raise ValueError('3D scene requires a bounded grid with at least two voxels per axis')
    bounds = np.column_stack((np.zeros(3), np.asarray(shape[::-1])*spacing))
    surfaces, total, warnings = {}, 0, []
    for key, source in masks.items():
        mask = np.asarray(source)
        if mask.shape != shape or not np.isfinite(mask).all() or not np.isin(mask, (0, 1)).all():
            raise ValueError('All 3D masks must be binary and use the same finite grid')
        mask = mask.astype(bool, copy=False)
        entry = {'voxels': int(mask.sum()), 'vertices': 0, 'triangles': 0, 'omitted': False}
        # Count edges before allocating a marching-cubes mesh. No decimation or
        # pooling is allowed to change the visual membership of diagnostic masks.
        if _surface_edge_count(mask) > MAX_SURFACE_TRIANGLES:
            entry['omitted'] = True
        else:
            grid, axes = occupancy_grid(mask, 1, [.5]*3, spacing)
            vertices, faces = _surface(grid, axes, bounds)
            if len(faces) > MAX_SURFACE_TRIANGLES or total+len(faces) > MAX_SCENE_TRIANGLES:
                entry['omitted'] = True
            else:
                total += len(faces)
                entry.update(vertices=len(vertices), triangles=len(faces),
                             xyz=_packed(vertices, '<f4'), ijk=_packed(faces, '<u4'))
        if entry['omitted']:
            warnings.append(f'{key}: exact surface exceeds the display budget; its {entry["voxels"]:,} voxels remain in saved metrics.')
        surfaces[key] = entry
    if modes is None:
        modes = [{'key': 'masks', 'label': 'Masks', 'explanation': 'Registered categorical surfaces.',
                  'layers': [{'key': key, 'label': key, 'color': '#127bb7', 'opacity': .45, 'visible': True} for key in masks]}]
    for mode in modes:
        for layer in mode['layers']:
            if layer['key'] not in surfaces or layer['opacity'] not in OPACITY_PRESETS:
                raise ValueError('Viewer layers require a known mask and a supported opacity')
    display = _display_options(coordinate_description, footer_note, show_metrics)
    return {'schema_version': 'fakect.mask-scene/1', 'sample_id': str(sample_id),
            'shape_kji': list(shape), 'spacing_ijk_mm': spacing.tolist(), 'bounds_ijk_mm': bounds.tolist(),
            'surfaces': surfaces, 'triangles': total, 'warnings': warnings, 'modes': modes,
            'metadata': metadata or {}, 'display': display}


def _display_options(coordinate_description, footer_note, show_metrics):
    values = {'coordinate_description': coordinate_description, 'footer_note': footer_note,
              'show_metrics': show_metrics}
    for key in ('coordinate_description', 'footer_note'):
        if values[key] is not None and not isinstance(values[key], str):
            raise ValueError(key+' must be text or None')
    if show_metrics is not None and not isinstance(show_metrics, bool):
        raise ValueError('show_metrics must be Boolean or None')
    return {key: value for key, value in values.items() if value is not None}


def _layer(key, label, color, *, visible=False, opacity=.8):
    return dict(key=key, label=label, color=color, visible=visible, opacity=opacity)


def _prediction_scene(path, row, threshold, first_threshold):
    """Read one saved NPZ and derive displays from the saved binary thresholds."""
    if row.get('prediction_sha256'):
        actual = hashlib.sha256(path.read_bytes()).hexdigest()
        if actual != row['prediction_sha256']:
            raise ValueError('Saved prediction checksum mismatch')
    with np.load(path, allow_pickle=False) as data:
        if 'sample_id' not in data or str(data['sample_id'].item()) != row['sample_id']:
            raise ValueError('Saved prediction belongs to another sample')
        after = np.asarray(data['after'])
        if after.ndim != 3 or after.size > MAX_VOXELS:
            raise ValueError('Prediction requires a bounded three-dimensional model grid')
        def binary(key, probability=False, cutoff=threshold):
            a = np.asarray(data[key])
            if a.shape != after.shape or not np.isfinite(a).all():
                raise ValueError('Prediction fields use inconsistent or nonfinite grids')
            if probability:
                if np.any((a < 0) | (a > 1)):
                    raise ValueError('Saved probabilities must lie in [0, 1]')
                return a >= cutoff
            if not np.isin(a, (0, 1)).all():
                raise ValueError('Saved masks must be binary')
            return a.astype(bool)
        masks = {key: binary(key) for key in ('before', 'after', 'added', 'removed')}
        if not np.array_equal(masks['added'], masks['after'] & ~masks['before']) or not np.array_equal(masks['removed'], masks['before'] & ~masks['after']):
            raise ValueError('Known added/removed masks disagree with before/after')
        masks['oracle_reference'] = binary('oracle_reference_probability', True)
        masks['oracle_added'] = masks['after'] & ~masks['oracle_reference']
        masks['oracle_removed'] = masks['oracle_reference'] & ~masks['after']
        chained = all(key in data for key in ('stage1_aorta_probability', 'chained_reference_probability'))
        if chained:
            masks['segmentation'] = binary('stage1_aorta_probability', True, first_threshold)
            masks['chained_reference'] = binary('chained_reference_probability', True)
            masks['chained_added'] = masks['segmentation'] & ~masks['chained_reference']
            masks['chained_removed'] = masks['chained_reference'] & ~masks['segmentation']
            masks['segmentation_fp'] = masks['segmentation'] & ~masks['after']
            masks['segmentation_fn'] = masks['after'] & ~masks['segmentation']
        for prefix in ('oracle', 'chained') if chained else ('oracle',):
            for name in ('added', 'removed'):
                key = prefix+'_'+name
                if key+'_mask' in data and not np.array_equal(binary(key+'_mask'), masks[key]):
                    raise ValueError('Saved binary changes disagree with thresholded reference')
        spacing = data['spacing_ijk_mm'].copy()
    truth = [_layer('before', 'Original reference · truth', '#7187a3', opacity=.15),
             _layer('after', 'Observed aorta · truth', '#6f7785', visible=True, opacity=.15),
             _layer('added', 'Expansion · true added voxels', '#e41a80', visible=True),
             _layer('removed', 'Narrowing · true removed voxels', '#147df5', visible=True)]
    def reconstruction(prefix):
        return [_layer(prefix+'_reference', 'Estimated reference · Model 2', '#7c3aed', opacity=.45),
                _layer(prefix+'_added', 'Expansion · predicted added voxels', '#f4b400', visible=True),
                _layer(prefix+'_removed', 'Narrowing · predicted removed voxels', '#18a572', visible=True)]
    modes = [dict(key='oracle', label='Model 2 · exact observed-mask input',
                  explanation='Exact saved observed mask → Model 2. True additions/removals are A−B and B−A; predictions compare input A with the estimated reference. This isolates Model 2 from segmentation errors.',
                  layers=truth+reconstruction('oracle'))]
    errors = [_layer('segmentation_fp', 'Segmentation false positives', '#f97316'),
              _layer('segmentation_fn', 'Segmentation false negatives', '#3fc6dd')]
    if chained:
        modes.append(dict(key='chained', label='Both models · Model 1 input into Model 2',
                          explanation='Model 1 segmentation S → Model 2. Predicted differences are S−estimated reference and its inverse; they include segmentation errors. The true change layers still show known synthetic A−B and B−A, not segmentation residuals.',
                          layers=truth+[_layer('segmentation', 'Observed aorta · Model 1 prediction', '#d16b22', opacity=.15)]+reconstruction('chained')+errors))
        modes.append(dict(key='segmentation', label='Model 1 · segmentation errors',
                          explanation='Compare the observed ground-truth aorta A with Model 1 segmentation S. Orange false positives are S−A; cyan false negatives are A−S. The entire model crop remains visible.',
                          layers=[truth[1], _layer('segmentation', 'Observed aorta · Model 1 prediction', '#d16b22', opacity=.15)]+
                                 [dict(layer, visible=True) for layer in errors]))
    metadata = {key: row.get(key) for key in ('case_id', 'is_control', 'native_added_voxels', 'native_removed_voxels', 'metrics')}
    metadata.update(threshold=threshold, first_threshold=first_threshold, prediction_sha256=row.get('prediction_sha256'))
    return build_scene(masks, spacing, sample_id=row['sample_id'], modes=modes, metadata=metadata)


_STYLE = """
body{font:15px/1.4 system-ui,sans-serif;background:#fff;color:#172d3c;margin:0;padding:16px}h1{font-size:22px;margin:0 0 8px}
p{margin:9px 0}.toolbar{display:flex;flex-wrap:wrap;gap:12px;align-items:center}select,button{font:inherit;padding:7px;border:1px solid #9aacba;border-radius:5px;background:white}select{max-width:100%}button{cursor:pointer}label{cursor:pointer}.layout{display:grid;grid-template-columns:minmax(260px,330px) minmax(300px,1fr);gap:12px;margin-top:12px}#controls{max-height:675px;overflow:auto}fieldset{border:1px solid #c5d1dc;border-radius:7px;margin:0 0 9px;padding:8px}legend{font-weight:600;padding:0 4px}.swatch{display:inline-block;width:13px;height:13px;border-radius:2px;margin-right:5px}.opacity{display:inline-block;margin:6px 6px 0 0;font-size:13px}.count,.help{font-size:13px;color:#465e71}.notice{padding:8px 12px;background:#fff4d6;border-left:3px solid #b88913}#plot{height:680px;min-width:0}#metrics{font-variant-numeric:tabular-nums}.busy{opacity:.55;pointer-events:none}#error{color:#9c1723}#details{overflow-wrap:anywhere}.note{border-top:1px solid #d8e2e9;padding-top:12px}@media(max-width:850px){.layout{grid-template-columns:1fr}#controls{max-height:250px;display:grid;grid-template-columns:repeat(auto-fit,minmax(220px,1fr));gap:8px}#plot{height:550px}}.empty{opacity:.65}
"""

_VIEWER_SCRIPT = r"""
(function(){
'use strict';
const manifest=__MANIFEST__, plot=document.getElementById('plot'), sample=document.getElementById('sample'), mode=document.getElementById('mode'), controls=document.getElementById('controls');
let scene=null, currentMode=null, loadedScript=null, pending=null, request=0, active=[];
const camera={eye:{x:1.6,y:-1.9,z:1.3}}, nf=new Intl.NumberFormat();
function text(id,value){document.getElementById(id).textContent=value;}
function packed(base64,Type){const b=atob(base64), bytes=new Uint8Array(b.length);for(let n=0;n<b.length;n++)bytes[n]=b.charCodeAt(n);return new Type(bytes.buffer);}
function mesh(source){if(source.decoded)return source.decoded;const xyz=packed(source.xyz,Float32Array),ijk=packed(source.ijk,Uint32Array);const x=new Float32Array(source.vertices),y=new Float32Array(source.vertices),z=new Float32Array(source.vertices),i=new Uint32Array(source.triangles),j=new Uint32Array(source.triangles),k=new Uint32Array(source.triangles);for(let n=0;n<x.length;n++){x[n]=xyz[3*n];y[n]=xyz[3*n+1];z[n]=xyz[3*n+2];}for(let n=0;n<i.length;n++){i[n]=ijk[3*n];j[n]=ijk[3*n+1];k[n]=ijk[3*n+2];}source.decoded={x,y,z,i,j,k};return source.decoded;}
function fail(message){text('error',message);document.getElementById('view').classList.remove('busy');sample.disabled=false;mode.disabled=false;}
function displayOptions(){return {...manifest.display,...(scene.display||{})};}
function metricText(){const shown=displayOptions().show_metrics!==false;document.getElementById('metrics').hidden=!shown;if(!shown){text('metrics','');return;}const m=scene.metadata.metrics||{};let keys=currentMode.key==='oracle'?['oracle_reference','oracle_added','oracle_removed']:currentMode.key==='chained'?['chained_reference','chained_added','chained_removed']:['stage1_aorta'];const labels=['Reference Dice','Expansion Dice','Narrowing Dice'];text('metrics',keys.map((key,n)=>{const value=(m[key]||{}).dice;return (keys.length===1?'Segmentation Dice':labels[n])+': '+(typeof value==='number'?value.toFixed(3):'not recorded');}).join(' · '));}
function scenarioSuffix(control){return control===true?' · unchanged control':control===false?' · edited scenario':'';}
function setVisibility(index,value){active[index].visible=value;const input=document.getElementById('visible-'+index);if(input)input.checked=value;Plotly.restyle(plot,{visible:value},[index]);}
function preset(kind){active.forEach((layer,index)=>{const key=layer.key;let value=false;if(kind==='changes')value=key==='after'||key==='added'||key==='removed'||key.endsWith('_added')||key.endsWith('_removed');if(kind==='reference')value=['before','after','oracle_reference','chained_reference'].includes(key);if(kind==='errors')value=key==='after'||key.startsWith('segmentation_');if(kind==='all')value=true;if(scene.surfaces[key].omitted||!scene.surfaces[key].voxels)value=false;setVisibility(index,value);});}
function reset(){if(scene)Plotly.relayout(plot,{'scene.camera':camera,'scene.xaxis.range':scene.bounds_ijk_mm[0],'scene.yaxis.range':scene.bounds_ijk_mm[1],'scene.zaxis.range':scene.bounds_ijk_mm[2]});}
async function draw(resetCamera){
 currentMode=scene.modes.find(row=>row.key===mode.value)||scene.modes[0];active=currentMode.layers.map(row=>({...row}));controls.replaceChildren();
 const traces=active.map((layer,index)=>{
  const source=scene.surfaces[layer.key],disabled=source.omitted||!source.voxels;layer.visible=layer.visible&&!disabled;
  const field=document.createElement('fieldset'),legend=document.createElement('legend'),swatch=document.createElement('span');swatch.className='swatch';swatch.style.background=layer.color;legend.append(swatch,document.createTextNode(layer.label));field.append(legend);if(disabled)field.className='empty';
  const checkbox=document.createElement('input');checkbox.type='checkbox';checkbox.id='visible-'+index;checkbox.checked=layer.visible;checkbox.disabled=disabled;checkbox.addEventListener('change',()=>setVisibility(index,checkbox.checked));const label=document.createElement('label');label.append(checkbox,document.createTextNode(' Show'));field.append(label);
  const count=document.createElement('span');count.className='count';count.textContent=' · '+nf.format(source.voxels)+' voxels'+(source.omitted?' · omitted: display budget':!source.voxels?' · empty':'');field.append(count,document.createElement('br'));
  manifest.opacity_presets.forEach(value=>{const radio=document.createElement('input');radio.type='radio';radio.name='opacity-'+index;radio.value=value;radio.checked=layer.opacity===value;radio.disabled=disabled;radio.addEventListener('change',()=>{if(radio.checked){active[index].opacity=value;Plotly.restyle(plot,{opacity:value},[index]);}});const choice=document.createElement('label');choice.className='opacity';choice.append(radio,document.createTextNode(Math.round(value*100)+'%'));field.append(choice);});controls.append(field);
  return {type:'mesh3d',...(source.omitted?{x:[],y:[],z:[],i:[],j:[],k:[]}:mesh(source)),name:layer.label,color:layer.color,opacity:layer.opacity,visible:layer.visible,flatshading:false,showlegend:false,lighting:{ambient:.65,diffuse:.8,specular:.12,roughness:.85},hovertemplate:'i: %{x:.2f} mm<br>j: %{y:.2f} mm<br>k: %{z:.2f} mm<extra>%{fullData.name}</extra>'};
 });
 const oldCamera=!resetCamera&&plot.layout&&plot.layout.scene?plot.layout.scene.camera:camera;
 const axis=(n,title)=>({title:{text:title},range:scene.bounds_ijk_mm[n],backgroundcolor:'#f4f7fa',gridcolor:'#dae3eb',zeroline:false});
 await Plotly.react(plot,traces,{scene:{xaxis:axis(0,'i · mm'),yaxis:axis(1,'j · mm'),zaxis:axis(2,'k · mm'),aspectmode:'data',camera:oldCamera,bgcolor:'#ffffff'},uirevision:scene.sample_id,margin:{l:0,r:0,t:8,b:15},paper_bgcolor:'#ffffff'},{responsive:true,displaylogo:false,toImageButtonOptions:{filename:'fakect-'+scene.sample_id.replace(/[^a-zA-Z0-9_-]/g,'-'),format:'png',scale:2}});
 text('explanation',currentMode.explanation);metricText();text('coordinates',displayOptions().coordinate_description||'');text('scope',displayOptions().footer_note||'');text('details','Grid (k,j,i): '+scene.shape_kji.join(' × ')+' · spacing (i,j,k): '+scene.spacing_ijk_mm.join(', ')+' mm · '+nf.format(scene.triangles)+' exact surface triangles.');text('warnings',scene.warnings.join(' '));document.getElementById('warnings').hidden=!scene.warnings.length;
 document.querySelector('[data-preset="changes"]').hidden=!active.some(row=>row.key==='added'||row.key==='removed'||row.key.endsWith('_added')||row.key.endsWith('_removed'));
 document.querySelector('[data-preset="reference"]').hidden=active.filter(row=>['before','after','oracle_reference','chained_reference'].includes(row.key)).length<2;
 document.getElementById('errors').hidden=!active.some(row=>row.key.startsWith('segmentation_'));document.getElementById('view').classList.remove('busy');sample.disabled=false;mode.disabled=false;text('status',scene.sample_id+scenarioSuffix(scene.metadata.is_control));
}
window.fakeCTReceiveScene=async function(value){if(!pending||value.sample_id!==pending.sample_id)return;scene=value;const previous=mode.value;mode.replaceChildren();scene.modes.forEach(item=>mode.add(new Option(item.label,item.key)));if(scene.modes.some(row=>row.key===previous))mode.value=previous;try{await draw(true);}catch(error){fail('3D rendering failed: '+error.message);}};
function load(){request++;const row=manifest.scenes[Number(sample.value)];pending=row;text('status','Loading '+row.sample_id+'…');text('error','');document.getElementById('view').classList.add('busy');sample.disabled=true;mode.disabled=true;if(loadedScript)loadedScript.remove();Plotly.purge(plot);scene=null;active=[];loadedScript=document.createElement('script');loadedScript.src=row.file;const number=request;loadedScript.onerror=()=>{if(number===request)fail('Cannot open this scene. Keep this entire viewer directory together; when embedded, also copy its parent report.');};document.body.append(loadedScript);}
manifest.scenes.forEach((row,index)=>sample.add(new Option(row.sample_id+scenarioSuffix(row.is_control),String(index))));sample.addEventListener('change',load);mode.addEventListener('change',()=>draw(false).catch(error=>fail(error.message)));
document.querySelectorAll('[data-preset]').forEach(button=>button.addEventListener('click',()=>preset(button.dataset.preset)));document.getElementById('reset').addEventListener('click',reset);if(typeof Plotly==='undefined'){fail('Local Plotly renderer is missing. Copy this entire viewer directory; when embedded, also copy its parent report.');}else if(manifest.scenes.length){load();}else{fail('No saved masks are available.');}
}());
"""


def render_viewer(output_directory, scenes, *, title='Aorta predictions · interactive 3D review',
                  coordinate_description=DEFAULT_COORDINATES, footer_note=DEFAULT_FOOTER,
                  show_metrics=True):
    """Write a viewer with local shared JS and one independently loaded scene.

    Scenes may be an iterator to keep Python memory bounded. No input files are
    changed. Copy this whole directory when sharing the viewer offline.
    Coordinate and footer text are plain text; individual scenes can override
    these defaults. Set ``show_metrics=False`` for non-model anatomy reviews.
    """
    from plotly.offline import get_plotlyjs
    display = _display_options(coordinate_description, footer_note, show_metrics)
    output = Path(output_directory)
    output.mkdir(parents=True, exist_ok=True)
    records, total_bytes = [], 0
    for index, scene in enumerate(scenes):
        if index >= MAX_SCENES:
            raise ValueError('3D viewer exceeds its bounded scenario count')
        content = 'window.fakeCTReceiveScene('+_json(scene)+');\n'
        if len(content.encode('utf-8')) > MAX_SCENE_BYTES:
            raise ValueError('3D scene exceeds its packed display budget')
        name = f'scene-{index:04d}.js'
        (output/name).write_text(content, encoding='utf-8')
        size = (output/name).stat().st_size
        total_bytes += size
        records.append({'sample_id': scene['sample_id'], 'file': name,
                        'is_control': scene.get('metadata', {}).get('is_control'),
                        'triangles': scene['triangles'], 'bytes': size,
                        'sha256': hashlib.sha256(content.encode()).hexdigest(), 'warnings': scene['warnings']})
    manifest = {'schema_version': 'fakect.mask-viewer/1', 'scenes': records,
                'opacity_presets': list(OPACITY_PRESETS), 'coordinates': coordinate_description, 'display': display,
                'sampling': 'exact Boolean voxel-boundary surfaces; no pooling, smoothing, or mesh decimation',
                'limits': {'voxels': MAX_VOXELS, 'surface_triangles': MAX_SURFACE_TRIANGLES,
                           'scene_triangles': MAX_SCENE_TRIANGLES, 'scene_bytes': MAX_SCENE_BYTES, 'scenes': MAX_SCENES},
                'scene_bytes': total_bytes}
    (output/'manifest.json').write_text(json.dumps(manifest, indent=2)+'\n')
    (output/'plotly.min.js').write_text(get_plotlyjs(), encoding='utf-8')
    (output/'viewer.js').write_text(_VIEWER_SCRIPT.replace('__MANIFEST__', _json(manifest)), encoding='utf-8')
    document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
                '<title>'+escape(title)+'</title><style>'+_STYLE+'</style><script src="plotly.min.js"></script></head><body>'
                '<h1>'+escape(title)+'</h1><div class="toolbar"><label>Scenario <select id="sample"></select></label>'
                '<label>Comparison <select id="mode"></select></label></div><p id="status" role="status"></p><p id="explanation"></p><p id="metrics"></p>'
                '<div class="toolbar"><button data-preset="changes">Compare changes</button><button data-preset="reference">Compare reference shapes</button>'
                '<button id="errors" data-preset="errors">Segmentation errors</button><button data-preset="none">Hide all</button><button id="reset">Reset full-crop view</button></div>'
                '<p class="help">Drag to orbit · wheel to zoom · right-drag to pan. Each label has independent visibility and 15%, 45%, or 80% opacity. '
                'Changing controls preserves the camera; Reset shows the entire saved crop, including distant false positives.</p>'
                '<p id="error" role="alert"></p><p id="warnings" class="notice" hidden></p>'
                '<div id="view" class="layout"><div id="controls" aria-label="Independent label visibility and opacity"></div><div id="plot"></div></div>'
                '<p id="details" class="help"></p><p id="coordinates" class="help">'+escape(coordinate_description or '')+'</p>'
                '<p id="scope" class="help note">'+escape(footer_note or '')+'</p>'
                '<p class="help">Offline: copy this entire viewer directory and open index.html. When embedded, also copy its parent report and preserve the relative folder layout. One scenario is loaded at a time; no web server or internet connection is required.</p>'
                '<script src="viewer.js"></script></body></html>')
    (output/'index.html').write_text(document, encoding='utf-8')
    return {'viewer': str((output/'index.html').resolve()), 'scenarios': len(records),
            'scene_bytes': total_bytes, 'omitted_surfaces': sum(len(row['warnings']) for row in records)}


def render_prediction_viewer(experiment_directory, results):
    """Create a complete view of every saved held-out Model 2 prediction."""
    from fakect_two_stage_report import _prediction_path
    directory = Path(experiment_directory)
    second = results.get('stage2', {})
    rows = [row for row in second.get('records', []) if row.get('prediction_file')]
    if not rows:
        return None
    def scenes():
        for row in rows:
            yield _prediction_scene(_prediction_path(directory/'test-model2', row['prediction_file']), row,
                                    second.get('threshold', .5), results.get('stage1', {}).get('threshold', .5))
    return render_viewer(directory/'report-3d', scenes())
