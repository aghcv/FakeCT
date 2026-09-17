"""Registered, native-voxel before/after surfaces for bounded edit previews.

This renderer intentionally does not pool anatomy using ``volume_stride``:
pooling before/after separately can conceal the very voxel changes being
reviewed. Both surfaces use the same voxel-boundary marching-cubes convention,
crop origin, spacing, and camera. They are categorical surfaces, not attenuation
volume rendering or an anatomical coordinate transform.
"""
from pathlib import Path
import hashlib
import json
import os

import numpy as np

from fakect_volume_preview import occupancy_grid, _surface


COLORS = {'before': '#0077b6', 'after': '#e54b1b'}
OPACITY_PRESETS = (.15, .45, .8)
INITIAL_OPACITY = {'before': .15, 'after': .45}
MAX_SOURCE_VOXELS = 2_000_000
MAX_SURFACE_TRIANGLES = 600_000


def _mask_sha256(mask):
    digest = hashlib.sha256()
    digest.update(json.dumps({'dtype': str(mask.dtype), 'shape': list(mask.shape)},
                             sort_keys=True, separators=(',', ':')).encode())
    digest.update(np.ascontiguousarray(mask).tobytes())
    return digest.hexdigest()


def _surface_edge_count(mask):
    """Count crossings including the zero border, before allocating meshes."""
    edges = 0
    for axis in range(3):
        lower = [slice(None)] * 3
        upper = [slice(None)] * 3
        lower[axis], upper[axis] = slice(None, -1), slice(1, None)
        edges += int(np.count_nonzero(mask[tuple(lower)] != mask[tuple(upper)]))
        edges += int(np.count_nonzero(np.take(mask, 0, axis=axis)))
        edges += int(np.count_nonzero(np.take(mask, -1, axis=axis)))
    return edges


def _controls_html():
    controls = []
    for key in ('before', 'after'):
        choices = []
        for opacity in OPACITY_PRESETS:
            percentage = round(opacity * 100)
            checked = ' checked' if opacity == INITIAL_OPACITY[key] else ''
            choices.append(
                f'<label class="preset"><input type="radio" name="{key}-opacity" '
                f'id="overlay-{key}-opacity-{percentage}" value="{opacity}"{checked}>'
                f'{percentage}%</label>')
        controls.append(
            f'<fieldset><legend><span class="swatch" style="background:{COLORS[key]}"></span>'
            f'{key.capitalize()} surface</legend><label class="visibility">'
            f'<input id="overlay-{key}-visible" type="checkbox" checked> Show {key}</label>'
            f'<span class="opacity-label">Opacity:</span>{"".join(choices)}</fieldset>')
    return '<div class="controls" aria-label="Independent surface controls">' + ''.join(controls) + '</div>'


_CONTROL_SCRIPT = """
(function () {
  const plot = document.getElementById('{plot_id}');
  const keys = ['before', 'after'];
  keys.forEach(function (key, index) {
    const visible = document.getElementById('overlay-' + key + '-visible');
    visible.addEventListener('change', function () {
      Plotly.restyle(plot, {visible: visible.checked}, [index]);
    });
    document.querySelectorAll('input[name="' + key + '-opacity"]').forEach(function (radio) {
      radio.addEventListener('change', function () {
        if (radio.checked) Plotly.restyle(plot, {opacity: Number(radio.value)}, [index]);
      });
    });
  });
  function syncControls() {
    keys.forEach(function (key, index) {
      const trace = plot.data[index];
      document.getElementById('overlay-' + key + '-visible').checked =
        trace.visible !== false && trace.visible !== 'legendonly';
      document.querySelectorAll('input[name="' + key + '-opacity"]').forEach(function (radio) {
        radio.checked = Math.abs(Number(radio.value) - trace.opacity) < 1e-9;
      });
    });
  }
  plot.on('plotly_restyle', syncControls);
  plot.on('plotly_legendclick', function (event) {
    const index = event.curveNumber;
    const trace = plot.data[index];
    const wasVisible = trace.visible !== false && trace.visible !== 'legendonly';
    Plotly.restyle(plot, {visible: !wasVisible}, [index]);
    return false;
  });
  plot.on('plotly_legenddoubleclick', function () { return false; });
  syncControls();
}());
"""


def _write_html(figure, path, counts, warnings):
    plot = figure.to_html(include_plotlyjs=True, full_html=False,
                          div_id='surface-overlay-plot', post_script=_CONTROL_SCRIPT,
                          config={'responsive': True, 'displaylogo': False})
    count_line = (f"Before: {counts['before_voxels']:,} voxels · After: {counts['after_voxels']:,} voxels · "
                  f"Added: {counts['added_voxels']:,} · Removed: {counts['removed_voxels']:,}")
    # All strings interpolated below are fixed renderer text or validated numbers.
    warning_html = ''.join(f'<p class="notice">{warning}</p>' for warning in warnings)
    path.write_text('''<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>Before and after surface overlay</title><style>
body{margin:0;padding:16px;font:15px system-ui,sans-serif;color:#162536;background:#fff}
h1{font-size:22px;margin:0 0 8px}p{margin:8px 0;line-height:1.5}
.controls{display:flex;flex-wrap:wrap;gap:14px;margin:14px 0}
fieldset{border:1px solid #bbc7d1;border-radius:8px;padding:12px;display:flex;align-items:center;gap:12px;flex-wrap:wrap}
legend{font-weight:650;padding:0 6px}.swatch{display:inline-block;width:15px;height:15px;margin-right:7px;border-radius:3px}
.preset{padding:6px 8px;background:#edf2f6;border-radius:5px;cursor:pointer}.preset:has(input:checked){outline:2px solid #46596c}
input{margin-right:5px;accent-color:#354f69}.visibility{font-weight:600}.opacity-label{margin-left:6px}
.notice{padding:10px;background:#fff3cf;border-left:3px solid #9f7400}.help{font-size:13px;color:#435466}
#surface-overlay-plot{min-height:570px}
</style></head><body><h1>Before and after in one 3D view</h1>'''
                    + '<p>' + count_line + '</p>' + warning_html + _controls_html()
                    + '<p class="help">Drag to orbit; wheel to zoom. Toggle either surface or adjust its opacity '
                      'independently. Changing opacity keeps hidden surfaces hidden and preserves the camera. '
                      'Blue is before; orange is after. Coincident surfaces overlap; toggle one to inspect it.</p>'
                    + plot
                    + '<p class="help">Native voxel-boundary surfaces of the selected anatomy throughout the crop. '
                      'Coordinates are source indices × spacing in mm; patient orientation is unverified. '
                      'No max pooling or mesh decimation is applied. Crop-edge closures are display boundaries. '
                      'Use the slice comparisons to inspect exact label changes; translucent overlap colors '
                      'are not a quantitative change map.</p></body></html>', encoding='utf-8')


def _write_png(meshes, bounds, counts, warnings, path):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    figure = plt.figure(figsize=(12, 10))
    try:
        ax = figure.add_subplot(111, projection='3d')
        for key, vertices, faces in meshes:
            if len(faces):
                ax.add_collection3d(Poly3DCollection(vertices[faces], facecolor=COLORS[key],
                                    edgecolor='none', alpha=INITIAL_OPACITY[key],
                                    linewidth=0, rasterized=True))
        ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2],
               xlabel='i × spacing (relative mm)', ylabel='j × spacing (relative mm)',
               zlabel='k × spacing (relative mm)')
        ax.set_box_aspect(bounds[:, 1] - bounds[:, 0])
        ax.view_init(elev=22, azim=-55)
        ax.legend(handles=[Patch(facecolor=COLORS[key], alpha=INITIAL_OPACITY[key],
                                  label=f'{key.capitalize()} — {INITIAL_OPACITY[key]:.0%} opacity')
                            for key in ('before', 'after')], loc='upper left')
        figure.suptitle('Before and after surface overlay\n'
                        f"Added: {counts['added_voxels']:,} voxels · Removed: {counts['removed_voxels']:,} voxels",
                        fontsize=16, y=.98)
        if warnings:
            ax.text2D(.03, .03, warnings[0], transform=ax.transAxes, fontsize=10,
                      bbox=dict(facecolor='#fff3cf', edgecolor='none', alpha=.9))
        figure.text(.04, .025, 'Native voxel-boundary surfaces in the same coordinate frame; orientation unverified.\n'
                    'Open the interactive overlay to toggle surfaces and choose independent opacity presets.', fontsize=10)
        figure.subplots_adjust(left=.035, right=.96, top=.9, bottom=.12)
        figure.savefig(path, dpi=160)
    finally:
        plt.close(figure)


def render_surface_overlay(before_mask, after_mask, *, crop_origin_ijk, spacing_ijk_mm,
                           volume_stride, output_dir):
    """Write standalone HTML and PNG overlays without modifying either mask.

    ``volume_stride`` records the ordinary volume preview setting for provenance;
    this comparison always uses native stride 1 to preserve small edits. The
    bounded crop must contain at most two million voxels, and the pair of meshes
    at most 600,000 triangles. A smaller crop is required if either limit is hit.
    """
    import plotly.graph_objects as go
    before, after = np.asarray(before_mask), np.asarray(after_mask)
    if before.ndim != 3 or before.shape != after.shape or any(n < 2 for n in before.shape):
        raise ValueError('Before and after masks must share a 3D shape with at least two voxels per axis')
    if before.dtype != np.bool_ or after.dtype != np.bool_:
        raise ValueError('Before and after masks must have Boolean dtype')
    if before.size > MAX_SOURCE_VOXELS:
        raise ValueError('Surface overlay is limited to 2,000,000 native voxels; use a smaller crop')
    if isinstance(volume_stride, bool) or not isinstance(volume_stride, (int, np.integer)) or volume_stride < 1:
        raise ValueError('volume_stride must be a positive integer')
    origin, spacing = np.asarray(crop_origin_ijk, dtype=float), np.asarray(spacing_ijk_mm, dtype=float)
    if origin.shape != (3,) or not np.all(np.isfinite(origin)) or np.any(origin < 0) or np.any(origin != np.trunc(origin)):
        raise ValueError('Crop origin must contain three finite nonnegative integer indices')
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Spacing must contain three finite positive values')
    bounds = np.column_stack(((origin - .5) * spacing,
                              (origin + np.asarray(before.shape[::-1]) - .5) * spacing))
    if not np.all(np.isfinite(bounds)):
        raise ValueError('Crop coordinates overflow physical bounds')
    # Every crossing can be a surface vertex. This preflight avoids constructing
    # huge checkerboard meshes only to reject them after allocation.
    if sum(_surface_edge_count(mask) for mask in (before, after)) > MAX_SURFACE_TRIANGLES:
        raise ValueError('Surface overlay is too complex at native resolution; use a smaller crop')
    output = Path(output_dir)
    paths = {key: output / name for key, name in (
        ('html', 'edit-overlay.html'), ('png', 'edit-overlay.png'), ('metadata', 'edit-overlay.json'))}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('Surface overlay output already exists; choose a new output directory')
    counts = {'before_voxels': int(before.sum()), 'after_voxels': int(after.sum()),
              'added_voxels': int(np.count_nonzero(after & ~before)),
              'removed_voxels': int(np.count_nonzero(before & ~after)),
              'unchanged_target_voxels': int(np.count_nonzero(before & after)),
              'net_delta_voxels': int(after.sum()) - int(before.sum())}
    warnings = []
    if not counts['before_voxels'] and not counts['after_voxels']:
        warnings.append('The selected anatomy is absent before and after; no surfaces are available.')
    elif not counts['before_voxels']:
        warnings.append('The selected anatomy is absent before the edit; only the after surface is available.')
    elif not counts['after_voxels']:
        warnings.append('The selected anatomy is absent after the edit; only the before surface is available.')
    if not counts['added_voxels'] and not counts['removed_voxels']:
        warnings.append('No target geometry changed: before and after masks are identical.')
    figure, meshes, surfaces = go.Figure(), [], []
    total_faces = 0
    for key, mask in (('before', before), ('after', after)):
        grid, axes = occupancy_grid(mask, 1, origin, spacing)
        vertices, faces = _surface(grid, axes, bounds)
        total_faces += len(faces)
        if total_faces > MAX_SURFACE_TRIANGLES:
            raise ValueError('Surface overlay exceeds 600,000 native triangles; use a smaller crop')
        meshes.append((key, vertices, faces))
        surfaces.append({'key': key, 'trace_index': len(surfaces), 'color': COLORS[key],
                         'source_voxels': counts[f'{key}_voxels'], 'surface_vertices': len(vertices),
                         'surface_triangles': len(faces), 'mask_sha256': _mask_sha256(mask),
                         'opacity': INITIAL_OPACITY[key], 'visible': True,
                         'bounds_ijk_relative_mm': (np.column_stack((vertices.min(axis=0), vertices.max(axis=0))).tolist()
                                                    if len(vertices) else None)})
        figure.add_trace(go.Mesh3d(x=vertices[:, 0], y=vertices[:, 1], z=vertices[:, 2],
                         i=faces[:, 0], j=faces[:, 1], k=faces[:, 2],
                         color=COLORS[key], opacity=INITIAL_OPACITY[key], visible=True,
                         name=key.capitalize(), showlegend=True, flatshading=False,
                         lighting=dict(ambient=.65, diffuse=.8, specular=.12, roughness=.85),
                         hovertemplate='i-relative mm: %{x:.2f}<br>j-relative mm: %{y:.2f}'
                                       '<br>k-relative mm: %{z:.2f}<extra>%{fullData.name}</extra>'))
    figure.update_layout(scene=dict(xaxis=dict(title='i × spacing (relative mm)', range=bounds[0]),
                                     yaxis=dict(title='j × spacing (relative mm)', range=bounds[1]),
                                     zaxis=dict(title='k × spacing (relative mm)', range=bounds[2]),
                                     aspectmode='data', bgcolor='#f4f6f8', uirevision='surface-overlay-camera',
                                     camera=dict(eye=dict(x=1.6, y=-1.9, z=1.3))),
                         uirevision='surface-overlay-controls', height=700,
                         margin=dict(l=0, r=0, t=20, b=25),
                         legend=dict(x=.01, y=.99, bgcolor='rgba(255,255,255,0.85)', itemdoubleclick=False),
                         template='plotly_white')
    output.mkdir(parents=True, exist_ok=True)
    _write_html(figure, paths['html'], counts, warnings)
    _write_png(meshes, bounds, counts, warnings, paths['png'])
    metadata = {'schema_version': 'fakect-surface-overlay-1',
                **{key: path.name for key, path in paths.items()},
                **{f'{key}_path': str(path) for key, path in paths.items()},
                'source_shape_kji': list(before.shape), 'crop_origin_ijk': origin.tolist(),
                'spacing_ijk_mm': spacing.tolist(), 'crop_bounds_ijk_relative_mm': bounds.tolist(),
                'counts': counts, 'colors': COLORS.copy(), 'opacity_presets': list(OPACITY_PRESETS),
                'initial_opacity': INITIAL_OPACITY.copy(), 'surfaces': surfaces,
                'requested_volume_stride': int(volume_stride), 'volume_stride': 1,
                'sampling': 'native Boolean voxel occupancy; no max pooling or mesh decimation',
                'surface_boundary': 'zero occupancy border closes surfaces on the exact crop boundary',
                'selection_extent': 'selected anatomy throughout the crop, not clipped to the edit ROI',
                'coordinates': 'source index times spacing in mm; physical origin and orientation unverified',
                'interactive_renderer': 'Plotly Mesh3d: registered before and after surfaces in one scene',
                'static_renderer': 'same native marching-cubes meshes and initial opacities in one view',
                'counts_resolution': 'native voxels; changed target membership, not all relabeling or scalar changes',
                'limits': {'source_voxels': MAX_SOURCE_VOXELS, 'combined_surface_triangles': MAX_SURFACE_TRIANGLES},
                'warnings': warnings, 'standalone_plotly_javascript': True,
                'source_arrays_modified': False, 'html_bytes': paths['html'].stat().st_size}
    paths['metadata'].write_text(json.dumps(metadata, indent=2) + '\n', encoding='utf-8')
    return metadata
