"""Small, offline ROI workbench: three native slices and a normal section per control.

This is a display consumer. It never changes source geometry, interpolates label
identities, or moves the configured controls to a fitted spline.
"""
import base64
import hashlib
import html
import json
import os
from pathlib import Path

import numpy as np
from scipy.ndimage import map_coordinates

from fakect_roi import COLORS, sphere_mask, tube_mask

MAX_CONTROL_POINTS = 256
MAX_SECTION_SIDE = 321
ROI_COLOR = '#ff00b8'
TARGET_COLOR = '#00eaff'
RADIUS_COLOR = '#ffd34e'


def control_frames(nodes_ijk, spacing_ijk_mm):
    """Physical polyline frames at exact controls; reversal fallback is explicit."""
    nodes = np.asarray(nodes_ijk, dtype=float)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if (nodes.ndim != 2 or nodes.shape[1:] != (3,) or not 1 <= len(nodes) <= MAX_CONTROL_POINTS
            or not np.all(np.isfinite(nodes)) or np.any(nodes < 0)):
        raise ValueError(f'ROI selection requires 1..{MAX_CONTROL_POINTS} finite nonnegative controls')
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('ROI selection spacing must contain three positive finite millimetre values')
    xyz = nodes * spacing
    delta = np.diff(xyz, axis=0)
    lengths = np.linalg.norm(delta, axis=1)
    if not np.all(np.isfinite(xyz)) or np.any(lengths <= 0):
        raise ValueError('Consecutive physical ROI controls must be distinct and finite')
    warnings = [''] * len(nodes)
    if len(nodes) == 1:
        tangents = np.array([[0., 0., 1.]])
        warnings[0] = 'Sphere: the fourth panel uses the i/j plane; there is no tube tangent.'
    else:
        directions = delta / lengths[:, None]
        tangents = np.empty_like(nodes)
        tangents[0], tangents[-1] = directions[0], directions[-1]
        for index in range(1, len(nodes)-1):
            tangent = directions[index-1] + directions[index]
            norm = np.linalg.norm(tangent)
            if norm <= 1e-10:
                tangent = directions[index]
                warnings[index] = 'Path reverses at this control; the outgoing segment defines the display plane.'
            else:
                tangent /= norm
            tangents[index] = tangent
    normals = []
    for tangent in tangents:
        axis = np.eye(3)[int(np.argmin(np.abs(tangent)))]
        normal = axis - np.dot(axis, tangent) * tangent
        normals.append(normal / np.linalg.norm(normal))
    normals = np.asarray(normals)
    return {'centers_mm': xyz, 'tangent': tangents, 'u_basis': normals,
            'v_basis': np.cross(tangents, normals),
            'distance_mm': np.r_[0., np.cumsum(lengths)], 'warnings': warnings}


def sample_section(fields, center_mm, u_basis, v_basis, spacing_ijk_mm, origin_ijk,
                   *, half_width_mm, pixel_mm):
    """Resample a display plane; CT is linear, all other fields nearest neighbor.

    Validity uses native voxel-center bounds. Outside pixels are masked in the
    report, so zero padding is never shown as observed tissue or attenuation.
    """
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    origin = np.asarray(origin_ijk, dtype=float)
    center = np.asarray(center_mm, dtype=float)
    u, v = np.asarray(u_basis, dtype=float), np.asarray(v_basis, dtype=float)
    if (not np.isfinite(half_width_mm) or half_width_mm <= 0
            or not np.isfinite(pixel_mm) or pixel_mm <= 0):
        raise ValueError('Section width and display sampling must be positive and finite')
    if (spacing.shape != (3,) or origin.shape != (3,) or center.shape != (3,)
            or u.shape != (3,) or v.shape != (3,) or np.any(spacing <= 0)
            or not all(np.all(np.isfinite(value)) for value in (spacing, origin, center, u, v))
            or not np.allclose([np.linalg.norm(u), np.linalg.norm(v), np.dot(u,v)], [1,1,0], atol=1e-8)):
        raise ValueError('Section geometry requires finite coordinates and orthonormal physical axes')
    with np.errstate(over='ignore',divide='ignore'):
        requested = float(np.float64(half_width_mm) / np.float64(pixel_mm)) * 2
    side = MAX_SECTION_SIDE if not np.isfinite(requested) or requested >= MAX_SECTION_SIDE-1 else max(3,int(np.ceil(requested))+1)
    if side % 2 == 0:
        side += 1
    offsets = np.linspace(-half_width_mm, half_width_mm, side)
    x, y = np.meshgrid(offsets, offsets)
    points = center[:, None, None] + u[:, None, None]*x + v[:, None, None]*y
    coordinates = (points / spacing[:, None, None] - origin[:, None, None])[::-1]
    shapes = {np.asarray(value).shape for value in fields.values()}
    if len(shapes) != 1 or len(next(iter(shapes), ())) != 3 or min(next(iter(shapes), (0,))) < 1:
        raise ValueError('Section fields must share one three-dimensional native crop')
    shape = next(iter(shapes))
    valid = np.ones(x.shape, dtype=bool)
    for axis, size in enumerate(shape):
        valid &= (coordinates[axis] >= -1e-9) & (coordinates[axis] <= size - 1 + 1e-9)
        coordinates[axis] = np.clip(coordinates[axis], 0, size-1)
    sampled = {}
    for name, value in fields.items():
        array = np.asarray(value)
        order = 1 if name == 'attenuation' else 0
        result = map_coordinates(array.astype(float) if order else array, coordinates,
                                 order=order, mode='nearest', prefilter=False)
        result[~valid] = 0
        sampled[name] = result
    return sampled, valid, offsets


def _definitions(config):
    yield 'main', config['roi']
    yield from config.get('rois', {}).items()


def _contour(ax, mask, x, y, color, *, width=1.6):
    if mask.any() and not mask.all():
        ax.contour(x, y, mask.astype(float), levels=[.5], colors=[color], linewidths=width)


def render_control_point_panels(arrays, resolved, config, output_dir):
    """Render CT and tissue versions of every configured ROI control card."""
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba
    from matplotlib.patches import Circle

    output = Path(output_dir)
    destination = output/'control-points'
    destination.mkdir(parents=True, exist_ok=True)
    spacing = np.asarray(resolved['spacing_ijk_mm'], dtype=float)
    low = np.asarray(resolved['crop_low_ijk'], dtype=int)
    high = np.asarray(resolved['crop_high_ijk_exclusive'], dtype=int)
    shape = np.asarray(arrays['act'].shape)
    window = [float(arrays['attenuation_cm_inverse'].min()), float(arrays['attenuation_cm_inverse'].max())]
    if window[0] == window[1]:
        window[1] += .001
    lut = np.tile(to_rgba(COLORS['unknown']), (256, 1))
    for category in resolved['catalog']['categories']:
        lut[category['id']] = to_rgba(COLORS.get(category['name'], '#999999'))
    tissue_cmap = ListedColormap(lut)
    rois = []
    definitions = list(_definitions(config))
    count = sum(len(roi['center_ijk']) if roi['shape'] == 'tube' else 1 for _, roi in definitions)
    if count > MAX_CONTROL_POINTS:
        raise ValueError(f'ROI selection has {count} controls; display limit is {MAX_CONTROL_POINTS}')
    for roi_index, (name, definition) in enumerate(definitions):
        tube = definition.get('shape', 'sphere') == 'tube'
        nodes = np.asarray(definition['center_ijk'] if tube else [definition['center_ijk']], dtype=float)
        radii = np.asarray(definition['radius_mm'] if tube else [definition['radius_mm']], dtype=float)
        if radii.shape != (len(nodes),) or not np.all(np.isfinite(radii)) or np.any(radii <= 0):
            raise ValueError(f'ROI {name}: each control needs one finite positive radius')
        frames = control_frames(nodes, spacing)
        if np.any(nodes < low) or np.any(nodes > high - 1):
            raise ValueError(f'ROI {name} controls fall outside the main preview crop; increase main crop_half_width_mm')
        roi_mask = arrays['roi'] if name == 'main' else (
            tube_mask(tuple(shape), low, nodes, spacing, radii) if tube else
            sphere_mask(tuple(shape), low, nodes[0], spacing, radii[0]))
        fields = {'attenuation': arrays['attenuation_cm_inverse'], 'tissue': arrays['tissue'],
                  'roi': roi_mask, 'selected': arrays['candidates'] & roi_mask}
        records = []
        for index, (center, radius) in enumerate(zip(nodes, radii)):
            fixed = np.rint(center).astype(int)
            half = max(radius*1.6, radius + 5.)
            section, valid, offsets = sample_section(
                fields, frames['centers_mm'][index], frames['u_basis'][index], frames['v_basis'][index],
                spacing, low, half_width_mm=half, pixel_mm=float(spacing.min()))
            images = {}
            for content in ('attenuation', 'labels'):
                fig, axes = plt.subplots(2, 2, figsize=(11, 9.5))
                fig.subplots_adjust(left=.09, right=.98, bottom=.13, top=.84, wspace=.28, hspace=.40)
                for panel_index, fixed_axis in enumerate((0,1,2)):
                    ax = axes.flat[panel_index]
                    x_axis, y_axis = [axis for axis in range(3) if axis != fixed_axis]
                    xmin, xmax = max(low[x_axis], int(np.floor(center[x_axis]-half/spacing[x_axis]))), min(high[x_axis], int(np.ceil(center[x_axis]+half/spacing[x_axis]))+1)
                    ymin, ymax = max(low[y_axis], int(np.floor(center[y_axis]-half/spacing[y_axis]))), min(high[y_axis], int(np.ceil(center[y_axis]+half/spacing[y_axis]))+1)
                    xs, ys = np.arange(xmin,xmax), np.arange(ymin,ymax)
                    n = int(fixed[fixed_axis]-low[fixed_axis])
                    plane = {key: np.take(value,n,axis=2-fixed_axis)[ymin-low[y_axis]:ymax-low[y_axis], xmin-low[x_axis]:xmax-low[x_axis]] for key,value in fields.items()}
                    image = plane['attenuation' if content == 'attenuation' else 'tissue']
                    extent = (xmin-.5,xmax-.5,ymin-.5,ymax-.5)
                    ax.imshow(image, origin='lower', extent=extent, interpolation='nearest',
                              aspect=spacing[y_axis]/spacing[x_axis], cmap='gray' if content == 'attenuation' else tissue_cmap,
                              vmin=window[0] if content == 'attenuation' else 0,
                              vmax=window[1] if content == 'attenuation' else 255)
                    fill = np.zeros((*plane['roi'].shape,4))
                    fill[...,:3] = to_rgba(ROI_COLOR)[:3]
                    fill[...,3] = plane['roi']*config['preview']['overlay_opacity']
                    ax.imshow(fill, origin='lower', extent=extent, aspect=spacing[y_axis]/spacing[x_axis], interpolation='nearest')
                    _contour(ax,plane['roi'],xs,ys,ROI_COLOR)
                    _contour(ax,plane['selected'],xs,ys,TARGET_COLOR)
                    ax.axvline(center[x_axis],color=RADIUS_COLOR,lw=1,ls='--')
                    ax.axhline(center[y_axis],color=RADIUS_COLOR,lw=1,ls='--')
                    ax.plot(center[x_axis],center[y_axis],'+',color=RADIUS_COLOR,ms=10,mew=1.5)
                    ax.set(xlabel=f'{"ijk"[x_axis]} (native index)',ylabel=f'{"ijk"[y_axis]} (native index)',
                           title=f'Fixed {"ijk"[fixed_axis]} = {fixed[fixed_axis]} | point {index+1}', xlim=extent[:2],ylim=extent[2:])
                ax = axes.flat[3]
                pixel = float(offsets[1]-offsets[0])
                extent = (offsets[0]-pixel/2,offsets[-1]+pixel/2)*2
                image = section['attenuation' if content == 'attenuation' else 'tissue']
                ax.set_facecolor('#555555')
                ax.imshow(np.ma.array(image,mask=~valid),origin='lower',extent=extent,interpolation='nearest',
                          cmap='gray' if content == 'attenuation' else tissue_cmap,
                          vmin=window[0] if content == 'attenuation' else 0,
                          vmax=window[1] if content == 'attenuation' else 255)
                fill = np.zeros((*valid.shape,4))
                fill[...,:3] = to_rgba(ROI_COLOR)[:3]
                fill[...,3] = section['roi']*valid*config['preview']['overlay_opacity']
                ax.imshow(fill,origin='lower',extent=extent,interpolation='nearest')
                _contour(ax,section['roi'] & valid,offsets,offsets,ROI_COLOR)
                _contour(ax,section['selected'] & valid,offsets,offsets,TARGET_COLOR)
                ax.add_patch(Circle((0,0),radius,fill=False,edgecolor=RADIUS_COLOR,lw=1.8,ls='--'))
                ax.axvline(0,color=RADIUS_COLOR,lw=.7,alpha=.6)
                ax.axhline(0,color=RADIUS_COLOR,lw=.7,alpha=.6)
                ax.set(xlabel='u (mm from exact control)',ylabel='v (mm from exact control)',
                       title=f'Tube-normal section | radius {radius:g} mm' if tube else f'Center section | radius {radius:g} mm',
                       xlim=extent[:2],ylim=extent[2:],aspect='equal')
                distance = frames['distance_mm'][index]
                content_title = 'attenuation (1/cm)' if content == 'attenuation' else 'tissue labels'
                fig.suptitle(f'{name} · control {index+1}/{len(nodes)} · i,j,k = {", ".join(f"{v:g}" for v in center)}\n'
                             f'Radius {radius:g} mm · distance {distance:.2f} mm along ordered path · {content_title}',fontsize=15,y=.97)
                note = frames['warnings'][index]
                if not valid.all():
                    note += ' Gray padding has no source/crop data.'
                fig.text(.5,.055,'Cyan: selected tissue   Magenta: actual ROI   Gold crosshair: exact control\n'
                         'Gold dashed circle: requested radius; the actual tube may extend farther at bends/tapers.',ha='center',fontsize=10)
                if note:
                    fig.text(.5,.015,note.strip(),ha='center',fontsize=9,color='#9b1c46')
                relative = f'control-points/{roi_index:02d}-{index+1:03d}-{content}.png'
                path = output/relative
                if path.exists():
                    plt.close(fig)
                    raise FileExistsError(f'ROI control card already exists: {path}')
                fig.savefig(path,dpi=130)
                plt.close(fig)
                images[content] = {'path':relative,'sha256':hashlib.sha256(path.read_bytes()).hexdigest()}
            records.append({'point':index+1,'center_ijk':center.tolist(),'native_slice_ijk':fixed.tolist(),
                            'radius_mm':float(radius),'distance_mm':float(frames['distance_mm'][index]),
                            'tangent_ijk_unit_physical':frames['tangent'][index].tolist(),
                            'u_basis_ijk_unit_physical':frames['u_basis'][index].tolist(),
                            'v_basis_ijk_unit_physical':frames['v_basis'][index].tolist(),
                            'section_shape':list(valid.shape),'section_pixel_mm':float(offsets[1]-offsets[0]),
                            'section_valid_fraction':float(valid.mean()),'warning':frames['warnings'][index],
                            'images':images})
        rois.append({'name':name,'shape':'tube' if tube else 'sphere','points':records})
    return {'schema_version':'fakect.roi-selection-preview/1','rois':rois,
            'attenuation_window_cm_inverse':window,'ct_interpolation':'linear; display only',
            'label_interpolation':'nearest neighbor; display only',
            'frame_semantics':'Exact configured controls; physical ordered-polyline tangent, averaged adjacent unit directions at interior controls; no spline fitting',
            'native_plane_semantics':'Nearest native plane with exact fractional control crosshairs',
            'invalid_pixel_semantics':'Gray padding is outside available voxel-center bounds, never observed tissue'}


def write_roi_selection_report(output_dir, report, input_ini_text):
    """Publish a self-contained navigation page with no network or 3D scripts."""
    output = Path(output_dir)
    destination = output/'report.html'
    if destination.exists():
        raise FileExistsError('report.html already exists; choose a new output directory')
    escape = lambda value: html.escape(str(value),quote=True)
    config = report['config']
    default = config['preview'].get('content','attenuation')
    cards, options, rows = [], [], []
    for roi in report['roi_selection']['rois']:
        for point in roi['points']:
            key = len(cards)
            title = f'{roi["name"]} · point {point["point"]}'
            options.append(f'<option value="{key}">{escape(title)}</option>')
            images = []
            for content, artifact in point['images'].items():
                encoded = base64.b64encode((output/artifact['path']).read_bytes()).decode('ascii')
                images.append(f'<img id="control-image-{key}-{content}" data-content="{content}" alt="{escape(title)}: native i, j, k planes and normal section in {content}" src="data:image/png;base64,{encoded}">')
            metadata = f'i,j,k = {", ".join(f"{value:g}" for value in point["center_ijk"])} · radius {point["radius_mm"]:g} mm · path {point["distance_mm"]:.2f} mm'
            cards.append(f'<section id="control-card-{key}" class="card" data-point="{key}" data-title="{escape(title)}"><h2>{escape(title)}</h2><p id="control-metadata-{key}" class="control-metadata">{escape(metadata)}</p>'+''.join(images)+'</section>')
            rows.append('<tr><td>'+f'<a href="#control-view" data-jump="{key}">{escape(title)}</a></td>'+''.join(f'<td>{escape(f"{value:g}")}</td>' for value in point['center_ijk'])+f'<td>{point["radius_mm"]:g}</td><td>{point["distance_mm"]:.2f}</td></tr>')
    selection = report.get('selection',{})
    spacing = report['geometry']['spacing_ijk_mm']
    content = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<meta http-equiv="Content-Security-Policy" content="default-src 'none'; img-src data:; style-src 'unsafe-inline'; script-src 'unsafe-inline'; connect-src 'none'; base-uri 'none'; form-action 'none'">
<title>FakeCT · ROI selection</title><style>
*{box-sizing:border-box}body{font-family:system-ui,sans-serif;max-width:1240px;margin:0 auto;padding:24px;color:#172c42;background:#f7f9fc}h1{margin-bottom:8px}.toolbar{display:flex;gap:14px;align-items:center;flex-wrap:wrap;position:sticky;top:0;background:#fff;padding:16px;border:1px solid #ccd4df;border-radius:10px;z-index:2}select,button{font:inherit;padding:8px}img{display:block;width:100%;height:auto}.card{background:#fff;border:1px solid #ccd4df;border-radius:10px;margin-top:18px;overflow:hidden}.card h2{padding:0 20px;font-size:20px}.control-metadata{padding:0 20px;color:#4b5d70}[hidden]{display:none!important}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:9px;border-bottom:1px solid #ccd4df}details{margin-top:20px}pre{white-space:pre-wrap;background:white;padding:16px;border:1px solid #ccd4df;overflow-wrap:anywhere}.subtle{color:#4b5d70}.notice{border-left:4px solid #ff00b8;padding:10px;background:#fff}a{color:#005d99}noscript{display:block;padding:12px;background:#fff4d0}
</style></head><body>'''
    content += f'<h1>ROI selection · {escape(config["study"]["name"])}</h1><p class="subtle">Case {escape(config["input"]["case_id"])} · {escape(config["selection"]["tissue"])} · native spacing i,j,k = {escape(spacing)} mm</p>'
    content += '<p>Use the three native planes to move each center. Use the fourth section to adjust its radius. Point numbers follow your INI path order.</p>'
    if selection.get('warnings'):
        content += '<div class="notice">'+''.join('<p>'+escape(warning)+'</p>' for warning in selection['warnings'])+'</div>'
    content += '<div id="control-view" class="toolbar"><label>Control <select id="point">'+''.join(options)+'</select></label><button type="button" id="previous">Previous</button><button type="button" id="next">Next</button><label>Content <select id="content"><option value="attenuation"'+(' selected' if default=='attenuation' else '')+'>Attenuation</option><option value="labels"'+(' selected' if default=='labels' else '')+'>Tissue labels</option></select></label><output id="current-point" aria-live="polite"></output></div><noscript>JavaScript is disabled: all controls and both contents are shown below.</noscript>'
    content += ''.join(cards)
    content += '<details open><summary><strong>Coordinates to edit</strong></summary><table><thead><tr><th>Control</th><th>i</th><th>j</th><th>k</th><th>Radius (mm)</th><th>Path (mm)</th></tr></thead><tbody>'+''.join(rows)+'</tbody></table></details>'
    content += '<details open><summary><strong>Change the INI and rerun</strong></summary><ol><li>Edit <code>center_ijk</code> and the matching <code>radius_mm</code> value. Keep the same point order.</li><li>Choose a fresh <code>[output] directory</code> to keep previous attempts.</li><li>Save the INI, rerun the command, and open the new <code>report.html</code>. Set <code>coordinate_reviewed = true</code> when satisfied.</li></ol><pre>'+escape(report.get('rerun_command','python3 scripts/preview_roi.py --config /path/to/roi.ini'))+'</pre><p class="subtle">These are captured views, not a live editor. ROI selection applies no edit, generates no cohort and trains no model. CT sections use linear display interpolation; labels and masks use nearest neighbor. Gray padding has no source/crop data. Anatomical orientation is not inferred from native indices.</p></details>'
    content += '<details><summary>Captured INI</summary><pre>'+escape(input_ini_text)+'</pre></details><details><summary>Traceability</summary><p>Source volumes remain unchanged. Full source, sampling geometry and artifact hashes are recorded in <code>preview-report.json</code> and <code>artifact-manifest.json</code>.</p><pre>'+escape(json.dumps({'input_config_sha256':report['input_config_sha256'],'catalog_sha256':report['catalog_sha256'],'source_files':report['source_files']},indent=2))+'</pre></details>'
    content += '''<script>
const point=document.getElementById('point'), content=document.getElementById('content');
function show(){document.querySelectorAll('.card').forEach(card=>{card.hidden=card.dataset.point!==point.value;card.querySelectorAll('img').forEach(img=>img.hidden=img.dataset.content!==content.value);if(!card.hidden){document.getElementById('current-point').textContent=card.dataset.title;}});document.getElementById('previous').disabled=point.selectedIndex===0;document.getElementById('next').disabled=point.selectedIndex===point.options.length-1;}
point.addEventListener('change',show);content.addEventListener('change',show);
document.getElementById('previous').addEventListener('click',()=>{point.selectedIndex=Math.max(0,point.selectedIndex-1);show();});document.getElementById('next').addEventListener('click',()=>{point.selectedIndex=Math.min(point.options.length-1,point.selectedIndex+1);show();});
document.querySelectorAll('[data-jump]').forEach(link=>link.addEventListener('click',()=>{point.value=link.dataset.jump;show();}));show();
</script></body></html>'''
    with destination.open('xb') as handle:
        handle.write(content.encode('utf-8'))
    return {'path':'report.html','self_contained':True,'bytes':destination.stat().st_size}
