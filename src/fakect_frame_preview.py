"""Portable visual review of smoothed centerline directions and reliability.

The fitted frame is a direction reference. This renderer preserves the input
ROI control points, source target, and supplied frame arrays.
"""
import hashlib
import html
import json
import os
from pathlib import Path
import re

import numpy as np

from fakect_volume_preview import occupancy_grid, _surface


COLORS = {'target': '#8fcbd3', 'original': '#64748b', 'spline': '#5b35ad',
          'inner': '#d84a26', 'outer': '#147daa', 'tangent': '#2e8b48',
          'binormal': '#b28a08', 'unreliable': '#8b8b8b'}
MAX_SOURCE_VOXELS = 2_000_000
MAX_CONTEXT_CELLS = 120_000
MAX_CONTEXT_TRIANGLES = 100_000
MAX_ARROWS_PER_DIRECTION = 14
SEMANTICS = (
    'The original ROI points and masks are unchanged; the smoothed spline is a direction reference. '
    'T follows the configured point order. Where reliable, N points toward the local curvature center '
    '(inner), -N points outward, and B = T × N. B changes sign when path order reverses. '
    'These directions describe the fitted ROI path, not a verified anatomical wall. '
    'Gray samples have undefined or unreliable inner/outer directions; no N, -N or B arrows are shown there.')


def _json_value(value):
    if isinstance(value, np.ndarray):
        return _json_value(value.tolist())
    if isinstance(value, np.generic):
        return _json_value(value.item())
    if isinstance(value, dict):
        return {str(key): _json_value(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_value(item) for item in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    return value


def _sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _validate_frame(frame):
    xyz = np.asarray(frame['xyz_mm'], dtype=float)
    if xyz.ndim != 2 or xyz.shape[1] != 3 or not 2 <= len(xyz) <= 4096 or not np.isfinite(xyz).all():
        raise ValueError('Frame preview requires 2–4096 finite three-dimensional samples')
    result = {'xyz_mm': xyz}
    for name in ('tangent', 'normal', 'binormal'):
        value = np.asarray(frame[name], dtype=float)
        if value.shape != xyz.shape or not np.isfinite(value).all():
            raise ValueError(f'Frame {name} must have one finite vector per sample')
        result[name] = value
    for name in ('parent_fraction', 'curvature_per_mm', 'reliable'):
        value = np.asarray(frame[name])
        if value.shape != (len(xyz),):
            raise ValueError(f'Frame {name} must have one value per sample')
        if name != 'reliable' and not np.isfinite(value).all():
            raise ValueError(f'Frame {name} must be finite')
        if name == 'reliable' and value.dtype != np.bool_:
            raise ValueError('Frame reliable must have Boolean dtype')
        result[name] = value
    fraction = result['parent_fraction']
    if np.any(np.diff(fraction) < 0) or np.any((fraction < 0) | (fraction > 1)):
        raise ValueError('Frame parent fractions must be ordered in [0, 1]')
    if np.any(result['curvature_per_mm'] < 0):
        raise ValueError('Frame curvature must be nonnegative')
    return result


def _indices(mask, limit=MAX_ARROWS_PER_DIRECTION):
    candidates = np.flatnonzero(mask)
    if not len(candidates):
        return candidates
    return candidates[np.unique(np.linspace(0, len(candidates)-1, min(limit, len(candidates))).round().astype(int))]


def _arrow_lines(points, vectors, length):
    """Draw direction-preserving shafts and open arrowheads in physical mm."""
    segments = []
    for point, vector in zip(points, vectors):
        norm = np.linalg.norm(vector)
        if norm <= 1e-12:
            continue
        direction = vector / norm
        axis = np.eye(3)[np.argmin(np.abs(direction))]
        side = np.cross(direction, axis)
        side /= np.linalg.norm(side)
        tip = point + length * direction
        base = tip - length * .23 * direction
        segments += [[point, tip], [base + length * .10 * side, tip],
                     [base - length * .10 * side, tip]]
    coordinates = []
    for segment in segments:
        coordinates.extend([*np.asarray(segment).tolist(), [None, None, None]])
    return np.asarray(coordinates, dtype=object).reshape(-1, 3)


def _target_context(arrays, resolved, config):
    target = np.asarray(arrays['candidates'])
    if target.dtype != np.bool_ or target.ndim != 3 or target.size > MAX_SOURCE_VOXELS:
        raise ValueError('Frame target context requires a Boolean crop of at most 2,000,000 voxels')
    origin = np.asarray(resolved['crop_low_ijk'], dtype=float)
    spacing = np.asarray(resolved['spacing_ijk_mm'], dtype=float)
    if origin.shape != (3,) or spacing.shape != (3,) or not np.isfinite(origin).all() or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Frame preview needs finite crop origin and positive physical spacing')
    bounds = np.column_stack(((origin-.5)*spacing, (origin+np.asarray(target.shape[::-1])-.5)*spacing))
    requested = config.get('preview', {}).get('volume_stride', 2)
    stride = max(1, int(requested))
    warnings = []
    if min(target.shape) < 2 or not target.any():
        warnings.append('No original target surface is available in this crop.')
        return np.empty((0, 3)), np.empty((0, 3), dtype=int), stride, warnings
    while True:
        sampled, axes = occupancy_grid(target, stride, origin, spacing)
        if any(size < 2 for size in sampled.shape):
            warnings.append('Target context omitted because the bounded display would have fewer than two blocks on an axis.')
            return np.empty((0, 3)), np.empty((0, 3), dtype=int), stride, warnings
        if sampled.size <= MAX_CONTEXT_CELLS:
            vertices, faces = _surface(sampled, axes, bounds)
            if len(faces) <= MAX_CONTEXT_TRIANGLES:
                break
        stride += 1
    if stride != requested:
        warnings.append(f'Target context display stride increased from {requested} to {stride} to bound rendering size.')
    return vertices, faces, stride, warnings


def _static_views(data, original_mm, arrows, vertices, faces, bounds, parent, path, curve_path, threshold):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection

    xyz, reliable = data['xyz_mm'], data['reliable']
    figure, axes = plt.subplots(1, 2, figsize=(16, 8), subplot_kw={'projection': '3d'})
    try:
        for axis, azimuth in zip(axes, (-60, 40)):
            if len(faces):
                axis.add_collection3d(Poly3DCollection(vertices[faces], facecolor=COLORS['target'],
                                     alpha=.10, edgecolor='none', linewidth=0, rasterized=True))
            axis.plot(*original_mm.T, color=COLORS['original'], marker='o', markersize=3,
                      linestyle='--', linewidth=1.2)
            axis.plot(*xyz.T, color=COLORS['spline'], linewidth=2)
            axis.scatter(*xyz[~reliable].T, color=COLORS['unreliable'], marker='x', s=17, depthshade=False)
            for index, point in enumerate(original_mm):
                axis.text(*point, str(index+1), color=COLORS['original'], fontsize=7)
            for arrow in arrows:
                lines = arrow['lines']
                if len(lines):
                    axis.plot(*np.asarray(lines, dtype=float).T, color=arrow['color'], linewidth=1.4)
            axis.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2],
                     xlabel='i × spacing (mm)', ylabel='j × spacing (mm)', zlabel='k × spacing (mm)')
            axis.set_box_aspect(bounds[:, 1]-bounds[:, 0])
            axis.view_init(elev=22, azim=azimuth)
        handles = [Line2D([0], [0], color=COLORS['original'], linestyle='--', marker='o', label='Original ROI points'),
                   Line2D([0], [0], color=COLORS['spline'], label='Smoothed direction reference')]
        if len(faces):
            handles.append(Line2D([0], [0], color=COLORS['target'], linewidth=6,
                                  label='Original target context (sampled)'))
        handles += [Line2D([0], [0], color=arrow['color'], label=arrow['label']) for arrow in arrows if len(arrow['indices'])]
        handles.append(Line2D([0], [0], color=COLORS['unreliable'], marker='x', linestyle='none',
                              label='Inner/outer undefined or unreliable'))
        figure.suptitle(f'{parent} | Centerline directions\nOriginal ROI geometry is unchanged', fontsize=15)
        figure.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .015), ncol=3, fontsize=9)
        figure.tight_layout(rect=(0, .13, 1, .92))
        figure.savefig(path, dpi=145)
    finally:
        plt.close(figure)
    figure, axis = plt.subplots(figsize=(12, 3.7))
    try:
        percent = 100 * data['parent_fraction']
        curvature = data['curvature_per_mm']
        axis.plot(percent, curvature, color=COLORS['spline'], label='Spline curvature')
        axis.scatter(percent[~reliable], curvature[~reliable], color=COLORS['unreliable'], marker='x',
                     s=25, label='Inner/outer undefined or unreliable', zorder=3)
        if threshold is not None:
            axis.axhline(threshold, color='#bb6d00', linestyle='--', linewidth=1, label='Minimum trusted curvature')
        axis.set(xlabel='Parent ROI path (%)', ylabel='Curvature (1/mm)', xlim=(0, 100),
                 title='Curvature and direction reliability along the original ROI path')
        axis.grid(alpha=.2)
        axis.legend(fontsize=8)
        figure.tight_layout()
        figure.savefig(curve_path, dpi=145)
    finally:
        plt.close(figure)


def render_centerline_frames(frames, arrays, resolved, config, output):
    """Write one portable frame view and full frame JSON per parent ROI.

    Coordinates are native indices times spacing, without an anatomical origin
    transform. Arrows are display references and do not represent edit distance.
    Only trusted Frenet normals/binormals are drawn; transported fallback normals
    are preserved in JSON but are not shown as anatomical inner/outer directions.
    """
    import plotly.graph_objects as go
    from plotly.subplots import make_subplots

    if not frames:
        return []
    output = Path(output)
    spacing = np.asarray(resolved['spacing_ijk_mm'], dtype=float)
    prepared = []
    for index, (parent, frame) in enumerate(frames.items(), 1):
        data = _validate_frame(frame)
        definition = config['roi'] if parent == 'main' else config.get('rois', {}).get(parent)
        if not definition or definition.get('shape') != 'tube':
            raise ValueError(f'Frame parent {parent!r} must identify a configured tube ROI')
        original = np.asarray(definition['center_ijk'], dtype=float)
        if original.ndim != 2 or original.shape[1] != 3 or not np.isfinite(original).all():
            raise ValueError('Original tube points must be finite three-dimensional coordinates')
        slug = re.sub(r'[^a-zA-Z0-9_-]+', '-', str(parent)).strip('-')[:48] or 'roi'
        directory = Path(f'centerline-{index:02d}-{slug}')
        paths = {name: directory/filename for name, filename in
                 (('html', 'frame.html'), ('figure', 'frame.png'), ('curvature_figure', 'curvature.png'),
                  ('frame_artifact', 'frame.json'))}
        if any((output/path).exists() for path in paths.values()):
            raise FileExistsError('Centerline frame preview already exists; choose a new output directory')
        prepared.append((parent, frame, data, definition, original*spacing, paths))
    vertices, faces, stride, context_warnings = _target_context(arrays, resolved, config)
    results = []
    for parent, frame, data, definition, original_mm, paths in prepared:
        destination = output/paths['html'].parent
        destination.mkdir(parents=True, exist_ok=True)
        xyz, reliable = data['xyz_mm'], data['reliable']
        metadata = _json_value(frame.get('metadata', {}))
        warnings = list(metadata.get('warnings', [])) + list(context_warnings)
        if not reliable.all():
            warnings.append(f'{int((~reliable).sum())} of {len(xyz)} samples have undefined or unreliable inner/outer directions; N, -N and B arrows are omitted there.')
        arrow_length = float(np.clip(np.median(np.atleast_1d(definition['radius_mm']))*.65, 2, 10))
        arrows = []
        for key, label, vectors, valid in (
                ('inner', 'Inner N (trusted curvature)', data['normal'], reliable),
                ('outer', 'Outer -N (trusted curvature)', -data['normal'], reliable),
                ('binormal', 'B = T × N (trusted curvature)', data['binormal'], reliable),
                ('tangent', 'T: configured path order', data['tangent'], np.ones(len(xyz), dtype=bool))):
            indices = _indices(valid & (np.linalg.norm(vectors, axis=1) > 1e-12))
            arrows.append({'name': key, 'label': label, 'color': COLORS[key], 'indices': indices,
                           'lines': _arrow_lines(xyz[indices], vectors[indices], arrow_length)})
        combined = np.vstack([xyz, original_mm])
        margin = max(arrow_length*1.4, float(np.max(np.atleast_1d(definition['radius_mm']))))
        bounds = np.column_stack((combined.min(axis=0)-margin, combined.max(axis=0)+margin))
        figure = make_subplots(rows=2, cols=1, specs=[[{'type': 'scene'}], [{'type': 'xy'}]],
                               row_heights=[.79, .21], vertical_spacing=.10)
        if len(faces):
            figure.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                             i=faces[:, 0], j=faces[:, 1], k=faces[:, 2], color=COLORS['target'],
                             opacity=.10, name='Original target context (sampled)', showlegend=True,
                             hoverinfo='skip'), row=1, col=1)
        figure.add_trace(go.Scatter3d(x=original_mm[:, 0], y=original_mm[:, 1], z=original_mm[:, 2],
                         mode='lines+markers', line={'color': COLORS['original'], 'dash': 'dash', 'width': 3},
                         marker={'size': 4, 'color': COLORS['original']}, name='Original ROI control points',
                         text=[f'Point {index+1}: i,j,k={tuple(point)}' for index, point in
                               enumerate(np.asarray(definition['center_ijk']))],
                         hovertemplate='%{text}<extra></extra>'), row=1, col=1)
        status = frame.get('normal_status', ['Reliable' if value else 'Unreliable' for value in reliable])
        hover = [f'Parent path {fraction*100:.2f}%<br>Curvature {curvature:.5g} /mm<br>{html.escape(str(state))}'
                 for fraction, curvature, state in zip(data['parent_fraction'], data['curvature_per_mm'], status)]
        figure.add_trace(go.Scatter3d(x=xyz[:, 0], y=xyz[:, 1], z=xyz[:, 2], mode='lines',
                         line={'color': COLORS['spline'], 'width': 5}, name='Smoothed direction reference',
                         text=hover, hovertemplate='%{text}<extra></extra>'), row=1, col=1)
        if (~reliable).any():
            figure.add_trace(go.Scatter3d(x=xyz[~reliable, 0], y=xyz[~reliable, 1], z=xyz[~reliable, 2],
                             mode='markers', marker={'color': COLORS['unreliable'], 'symbol': 'x', 'size': 4},
                             name='Inner/outer undefined or unreliable',
                             text=np.asarray(hover)[~reliable], hovertemplate='%{text}<extra></extra>'), row=1, col=1)
        for arrow in arrows:
            lines = arrow['lines']
            if len(lines):
                figure.add_trace(go.Scatter3d(x=lines[:, 0], y=lines[:, 1], z=lines[:, 2], mode='lines',
                                 line={'color': arrow['color'], 'width': 4}, name=arrow['label'],
                                 hoverinfo='name'), row=1, col=1)
        percent, curvature = data['parent_fraction']*100, data['curvature_per_mm']
        figure.add_trace(go.Scatter(x=percent, y=curvature, mode='lines', line={'color': COLORS['spline']},
                                   name='Spline curvature', showlegend=False), row=2, col=1)
        figure.add_trace(go.Scatter(x=percent[~reliable], y=curvature[~reliable], mode='markers',
                         marker={'color': COLORS['unreliable'], 'symbol': 'x'}, showlegend=False,
                         name='Unreliable direction'), row=2, col=1)
        threshold = metadata.get('min_curvature_per_mm', config.get('centerline', {}).get('min_curvature_per_mm'))
        if threshold is not None:
            figure.add_trace(go.Scatter(x=[0, 100], y=[threshold, threshold], mode='lines',
                             line={'color': '#bb6d00', 'dash': 'dash'}, name='Minimum trusted curvature'), row=2, col=1)
        figure.update_xaxes(title_text='Parent ROI path (%)', range=[0, 100], row=2, col=1)
        figure.update_yaxes(title_text='Curvature (1/mm)', rangemode='tozero', row=2, col=1)
        figure.update_layout(height=970, template='plotly_white', margin={'l': 65, 'r': 20, 't': 45, 'b': 45},
                             scene={'xaxis': {'title': 'i × spacing (mm)', 'range': bounds[0]},
                                    'yaxis': {'title': 'j × spacing (mm)', 'range': bounds[1]},
                                    'zaxis': {'title': 'k × spacing (mm)', 'range': bounds[2]},
                                    'aspectmode': 'data', 'camera': {'eye': {'x': 1.6, 'y': -1.8, 'z': 1.2}}},
                             legend={'orientation': 'h', 'x': 0, 'y': 1.05, 'font': {'size': 10}})
        plot = figure.to_html(include_plotlyjs=True, full_html=False, div_id='centerline-frame',
                              config={'responsive': True, 'displaylogo': False})
        document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" '
                    'content="width=device-width, initial-scale=1"><title>Centerline direction reference</title>'
                    '<style>body{margin:0;padding:15px;font:14px/1.5 system-ui,sans-serif;color:#203040}'
                    'h1{font-size:21px;margin:0 0 8px}p{margin:8px 0}.notice{background:#f1f5f8;padding:10px}</style>'
                    '</head><body><h1>' + html.escape(str(parent)) + ' — Centerline directions</h1><p>' +
                    html.escape(SEMANTICS) + '</p><p class="notice">Drag to orbit; wheel to zoom. '
                    'Click legend entries to toggle curves, arrows or target context. '
                    f'Arrows have a display length of {arrow_length:g} mm; this is not the edit displacement. '
                    f'Target context uses occupancy stride {stride}; thin structures may appear wider.</p>' + plot + '</body></html>')
        (output/paths['html']).write_text(document, encoding='utf-8')
        _static_views(data, original_mm, arrows, vertices, faces, bounds, str(parent),
                      output/paths['figure'], output/paths['curvature_figure'], threshold)
        frame_record = {'schema_version': 'fakect.centerline-frame-preview/1', 'parent_roi': parent,
                        'coordinate_semantics': 'Native i,j,k indices times spacing in mm; orientation and origin unverified',
                        'spacing_ijk_mm': spacing.tolist(), 'original_roi': _json_value(definition),
                        'frame': _json_value(frame), 'display_arrow_length_mm': arrow_length,
                        'display_arrow_sample_indices': {arrow['name']: arrow['indices'].tolist() for arrow in arrows},
                        'target_context_stride': stride, 'target_context_surface_triangles': int(len(faces)),
                        'direction_semantics': SEMANTICS, 'warnings': warnings, 'source_arrays_modified': False}
        (output/paths['frame_artifact']).write_text(json.dumps(frame_record, indent=2, ensure_ascii=False,
                                                             allow_nan=False)+'\n', encoding='utf-8')
        results.append({'parent_roi': parent, **{key: str(path) for key, path in paths.items()},
                        'metadata': metadata, 'settings': _json_value(config.get('centerline', {})),
                        'sample_count': len(xyz), 'reliable_samples': int(reliable.sum()),
                        'unreliable_samples': int((~reliable).sum()), 'arrow_length_mm': arrow_length,
                        'arrow_sample_indices': frame_record['display_arrow_sample_indices'],
                        'target_context_stride': stride, 'target_context_surface_triangles': int(len(faces)),
                        'direction_semantics': SEMANTICS, 'warnings': warnings,
                        'artifacts': {key: {'path': str(path), 'sha256': _sha256(output/path),
                                           'bytes': (output/path).stat().st_size} for key, path in paths.items()}})
    return results
