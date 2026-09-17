"""Bounded 3D previews of categorical XCAT crops, in native index-relative mm.

The interactive view uses Plotly Volume traces of binary occupancy, not measured
attenuation. Strided previews max-pool each block so a thin label survives. The
displayed occupied blocks can therefore be wider than the original anatomy.
Neither these previews nor their sampled grids are segmentation outputs.
"""
from pathlib import Path
import json
import os

import numpy as np


COLORS = {
    'background': '#000000', 'soft_tissue': '#bcaaa4', 'bone': '#fff0bc',
    'cartilage': '#66bb6a', 'muscle': '#b85c38', 'artery': '#f44336',
    'vein': '#367bf5', 'lung': '#4dd0c8', 'adipose': '#ffd600',
    'nervous_tissue': '#9575cd', 'fluid': '#b3e5fc', 'unknown': '#ff00cc',
}
SELECTION_COLOR = '#00bcd4'
ROI_COLOR = '#ff9800'


def occupancy_grid(mask, stride, crop_origin_ijk, spacing_ijk_mm):
    """Return block-maximum occupancy and block-center coordinates (i, j, k).

    Input/output arrays are kji. Coordinates include the source crop origin and
    use the true center of the last block even when its width is below stride.
    """
    mask = np.asarray(mask)
    if mask.ndim != 3 or any(n < 2 for n in mask.shape):
        raise ValueError('A 3D crop with at least two voxels on each axis is required')
    if isinstance(stride, bool) or not isinstance(stride, (int, np.integer)) or stride < 1:
        raise ValueError('volume_stride must be a positive integer')
    origin = np.asarray(crop_origin_ijk, dtype=float)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if origin.shape != (3,) or spacing.shape != (3,) or not np.all(np.isfinite(origin)):
        raise ValueError('Crop origin and spacing must have three finite components')
    if not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Spacing must be finite and positive')
    result = mask.astype(bool, copy=False)
    for axis, length in enumerate(mask.shape):
        result = np.logical_or.reduceat(result, np.arange(0, length, stride), axis=axis)
    coordinates = []
    for axis, length in enumerate(mask.shape[::-1]):
        first = np.arange(0, length, stride)
        last = np.minimum(first + stride, length) - 1
        coordinates.append(((origin[axis] + (first + last) / 2) * spacing[axis]).astype(np.float32))
    return result, tuple(coordinates)


def _surface(mask, axes_ijk, bounds_ijk_mm):
    """March occupied blocks, retaining the exact outer crop boundary."""
    from skimage.measure import marching_cubes
    if not np.any(mask):
        return np.empty((0, 3)), np.empty((0, 3), dtype=np.int32)
    vertices, faces, _, _ = marching_cubes(np.pad(mask.astype(np.uint8), 1), level=.5)
    points = np.empty_like(vertices)
    for array_axis, physical_axis in enumerate((2, 1, 0)):
        centers = axes_ijk[physical_axis]
        lo, hi = bounds_ijk_mm[physical_axis]
        padded = np.r_[2 * lo - centers[0], centers, 2 * hi - centers[-1]]
        points[:, physical_axis] = np.interp(vertices[:, array_axis], np.arange(len(padded)), padded)
    return points, faces


def _sphere(center, radius, n_longitude=48, n_latitude=25):
    theta = np.linspace(0, 2 * np.pi, n_longitude, endpoint=False)
    phi = np.linspace(0, np.pi, n_latitude)
    xyz = np.stack((np.sin(phi[:, None]) * np.cos(theta),
                    np.sin(phi[:, None]) * np.sin(theta),
                    np.cos(phi[:, None]) * np.ones_like(theta)), axis=-1)
    vertices = xyz.reshape(-1, 3) * radius + center
    faces = []
    for row in range(n_latitude - 1):
        for col in range(n_longitude):
            a, b = row * n_longitude + col, row * n_longitude + (col + 1) % n_longitude
            c, d = a + n_longitude, b + n_longitude
            faces.extend(((a, b, c), (b, d, c)))
    return vertices, np.asarray(faces, dtype=np.int32)


def _box_coordinates(bounds):
    corners = np.asarray([[bounds[0][i], bounds[1][j], bounds[2][k]]
                          for i in range(2) for j in range(2) for k in range(2)])
    values = []
    for n in range(8):
        for bit in (1, 2, 4):
            if not n & bit:
                values.extend((corners[n].tolist(), corners[n | bit].tolist(), [None] * 3))
    return list(zip(*values))


def render_volume_preview(original_labels, tissue_labels, catalog, selection_mask, *,
                          crop_origin_ijk, spacing_ijk_mm, roi_center_ijk, roi_radius_mm,
                          context_tissues, volume_stride, volume_opacity, context_opacity,
                          output_dir):
    """Write self-contained HTML Volume rendering and a static surface PNG.

    ``context_tissues`` accepts category names or numeric IDs. The selected mask
    is excluded from contextual groups before pooling. Hover coordinates and all
    axes use native index * spacing, without asserting patient orientation.
    """
    import plotly.graph_objects as go
    original = np.asarray(original_labels)
    tissues = np.asarray(tissue_labels)
    selected = np.asarray(selection_mask)
    if original.ndim != 3 or tissues.shape != original.shape or selected.shape != original.shape:
        raise ValueError('Original labels, tissue labels, and selection must share a 3D shape')
    if original.size > 16_777_216:
        raise ValueError('Preview is limited to 16,777,216 source voxels; use a smaller crop')
    if not np.all(np.isfinite(original)) or not np.all(original == np.trunc(original)):
        raise ValueError('Original labels must be finite integer-valued IDs')
    if selected.dtype != np.bool_:
        raise ValueError('selection_mask must have Boolean dtype')
    for value, name in ((volume_opacity, 'volume_opacity'), (context_opacity, 'context_opacity')):
        if not np.isfinite(value) or not 0 <= value <= 1:
            raise ValueError(f'{name} must be in [0, 1]')
    center = np.asarray(roi_center_ijk, dtype=float)
    origin = np.asarray(crop_origin_ijk, dtype=float)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if center.shape != (3,) or not np.all(np.isfinite(center)):
        raise ValueError('ROI center must have three finite index coordinates')
    if not np.isfinite(roi_radius_mm) or roi_radius_mm <= 0:
        raise ValueError('ROI radius must be finite and positive')
    pooled, axes = occupancy_grid(selected, volume_stride, origin, spacing)
    if any(n < 2 for n in pooled.shape):
        raise ValueError('volume_stride leaves fewer than two blocks on an axis; reduce it')
    bounds = np.column_stack(((origin - .5) * spacing,
                              (origin + np.asarray(original.shape[::-1]) - .5) * spacing))
    categories = catalog['categories']
    by_key = {c['name']: c for c in categories}
    by_key.update({c['id']: c for c in categories})
    records = {r['original_id']: r for r in catalog['records']}
    selected_ids, selected_counts = np.unique(original[selected], return_counts=True)
    selected_records = [{'original_id': int(label), 'voxel_count': int(count),
                         'original_name': records.get(int(label), {}).get('original_name', 'unnamed')}
                        for label, count in zip(selected_ids, selected_counts)]
    selected_name = '; '.join(f"{r['original_id']} {r['original_name']}" for r in selected_records[:4])
    warnings = []
    if not selected_records:
        selected_name = 'absent from crop'
        warnings.append('No selected label voxels occur in this crop; relocate the crop or change the selection.')
    if len(selected_records) > 4:
        selected_name += f'; +{len(selected_records) - 4} IDs'
    fields = [{'name': f'Selected: {selected_name}', 'grid': pooled,
               'color': SELECTION_COLOR, 'opacity': float(volume_opacity),
               'source_voxels': int(selected.sum()), 'kind': 'selection'}]
    seen = set()
    missing_context = []
    for key in context_tissues:
        if key not in by_key:
            raise ValueError(f'Unknown contextual tissue category: {key!r}')
        category = by_key[key]
        if category['id'] in seen:
            continue
        seen.add(category['id'])
        mask = (tissues == category['id']) & ~selected
        if not np.any(mask):
            missing_context.append(category['name'])
            continue
        grid, _ = occupancy_grid(mask, volume_stride, origin, spacing)
        fields.append({'name': category['name'].replace('_', ' '), 'grid': grid,
                       'color': COLORS.get(category['name'], '#999999'),
                       'opacity': float(context_opacity), 'source_voxels': int(mask.sum()),
                       'kind': 'context', 'tissue_id': int(category['id'])})
    render_cell_count = int(np.prod(np.asarray(pooled.shape) + 2))
    if render_cell_count * len(fields) > 650_000:
        raise ValueError('3D preview has too many display cells; increase volume_stride or reduce context/crop')
    output = Path(output_dir)
    output.mkdir(parents=True, exist_ok=True)
    paths = {key: output / filename for key, filename in (
        ('html', 'roi-volume.html'), ('png', 'roi-surfaces.png'), ('metadata', 'roi-volume.json'))}
    if any(path.exists() for path in paths.values()):
        raise FileExistsError('3D preview output already exists; choose a new output directory')
    # A zero occupancy border closes structures at the crop edge. Mirroring the
    # first/last centers across the edge places the .5 isosurface on that edge,
    # including partial blocks at the upper end. It also renders an all-filled
    # crop, which otherwise contains no internal density transitions.
    render_axes = [np.r_[2 * b[0] - a[0], a, 2 * b[1] - a[-1]].astype(np.float32)
                   for a, b in zip(axes, bounds)]
    zz, yy, xx = np.meshgrid(render_axes[2], render_axes[1], render_axes[0], indexing='ij')
    flatcoords = {'x': xx.ravel(), 'y': yy.ravel(), 'z': zz.ravel()}
    fig = go.Figure()
    for field in fields:
        fig.add_trace(go.Volume(**flatcoords, value=np.pad(field['grid'].astype(np.uint8), 1).ravel(),
                      isomin=.5, isomax=1, opacity=field['opacity'], surface_count=8,
                      opacityscale=[[0, .15], [1, 1]],
                      colorscale=[[0, field['color']], [1, field['color']]],
                      showscale=False, showlegend=True, name=field['name'],
                      caps=dict(x_show=False, y_show=False, z_show=False),
                      hovertemplate='i-relative mm: %{x:.2f}<br>j-relative mm: %{y:.2f}'
                                    '<br>k-relative mm: %{z:.2f}<extra>%{fullData.name}</extra>'))
    sphere_vertices, sphere_faces = _sphere(center * spacing, roi_radius_mm)
    fig.add_trace(go.Mesh3d(x=sphere_vertices[:, 0], y=sphere_vertices[:, 1], z=sphere_vertices[:, 2],
                  i=sphere_faces[:, 0], j=sphere_faces[:, 1], k=sphere_faces[:, 2],
                  color=ROI_COLOR, opacity=.16, name='Editable ROI sphere', showlegend=True,
                  hoverinfo='name', flatshading=False))
    bx, by, bz = _box_coordinates(bounds)
    fig.add_trace(go.Scatter3d(x=bx, y=by, z=bz, mode='lines', line=dict(color='#687783', width=2),
                             name='Crop boundary', showlegend=True, hoverinfo='skip'))
    steps = []
    for label, multiplier in [('Light', .4), ('Configured', 1), ('Dense', 2)]:
        steps.append(dict(method='restyle', label=label,
                          args=[{'opacity': [min(1, f['opacity'] * multiplier) for f in fields]},
                                list(range(len(fields)))]))
    annotation = ('Binary occupancy volume; legend toggles individual labels. Drag to orbit; wheel to zoom.'
                  f'<br>Max-pool stride {volume_stride}: thin labels survive, but displayed support expands.'
                  '<br>Native relative mm; anatomical orientation is unverified. ROI is a planning overlay.')
    if warnings:
        annotation += '<br>' + warnings[0]
    fig.update_layout(title=dict(text='Selected-label 3D volume preview', x=.02),
                      scene=dict(xaxis=dict(title='i × spacing (relative mm)', range=bounds[0]),
                                 yaxis=dict(title='j × spacing (relative mm)', range=bounds[1]),
                                 zaxis=dict(title='k × spacing (relative mm)', range=bounds[2]),
                                 aspectmode='data', bgcolor='#f4f6f8',
                                 camera=dict(eye=dict(x=1.6, y=-1.9, z=1.3))),
                      height=850, margin=dict(l=0, r=0, t=70, b=130),
                      legend=dict(x=.01, y=.99, bgcolor='rgba(255,255,255,0.75)'),
                      annotations=[dict(text=annotation, x=.01, y=-.075, xref='paper', yref='paper',
                                        showarrow=False, align='left', font=dict(size=12))],
                      sliders=[dict(active=1, x=.55, y=-.015, len=.4, steps=steps,
                                    currentvalue=dict(prefix='Volume opacity: '))],
                      template='plotly_white')
    fig.write_html(str(paths['html']), include_plotlyjs=True, full_html=True,
                   config={'responsive': True, 'displaylogo': False}, auto_open=False)

    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    static, static_axes = plt.subplots(1, 2, figsize=(18, 9), subplot_kw={'projection': '3d'})
    trace_metadata = []
    surface_fields = []
    for field in reversed(fields):
        vertices, faces = _surface(field['grid'], axes, bounds)
        alpha = field['opacity']
        if field['kind'] == 'selection' and alpha > 0:
            alpha = min(.8, max(.08, alpha * 2.5))
        surface_fields.append((field, vertices, faces, alpha))
        trace_metadata.append({key: value for key, value in field.items() if key != 'grid'} |
                              {'occupied_display_blocks': int(field['grid'].sum()),
                               'surface_triangles': int(len(faces)), 'static_surface_opacity': alpha})
    panel_titles = ('Selected original labels + ROI', 'Selected labels + tissue context + ROI')
    for panel, ax in enumerate(static_axes):
        for field, vertices, faces, alpha in surface_fields:
            if (panel == 0 and field['kind'] != 'selection') or not len(faces) or alpha == 0:
                continue
            mesh = Poly3DCollection(vertices[faces], facecolor=field['color'], edgecolor='none',
                                    alpha=alpha, linewidth=0, rasterized=True)
            ax.add_collection3d(mesh)
        roi_mesh = Poly3DCollection(sphere_vertices[sphere_faces], facecolor=ROI_COLOR,
                                   edgecolor='none', alpha=.12, linewidth=0, rasterized=True)
        ax.add_collection3d(roi_mesh)
        # Great-circle outlines make the transparent ROI boundary visible even
        # when a vessel crosses its front/back surface.
        angle = np.linspace(0, 2 * np.pi, 97)
        for fixed_axis in range(3):
            circle = np.zeros((len(angle), 3))
            varying_axes = [axis for axis in range(3) if axis != fixed_axis]
            circle[:, varying_axes[0]] = roi_radius_mm * np.cos(angle)
            circle[:, varying_axes[1]] = roi_radius_mm * np.sin(angle)
            circle += center * spacing
            ax.plot(*circle.T, color=ROI_COLOR, alpha=.55, linewidth=.8)
        ax.plot(*(np.asarray(a, dtype=float) for a in (bx, by, bz)),
                color='#687783', alpha=.65, linewidth=.6)
        ax.scatter(*(center * spacing), color=ROI_COLOR, s=25, marker='+', depthshade=False)
        ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2], xlabel='i × spacing (relative mm)',
               ylabel='j × spacing (relative mm)', zlabel='k × spacing (relative mm)')
        ax.set_box_aspect(bounds[:, 1] - bounds[:, 0])
        ax.view_init(elev=22, azim=-55)
        ax.set_title(panel_titles[panel], pad=22)
        handles = [Patch(color=f['color'], alpha=.8, label=f['name']) for f in fields
                   if panel == 1 or f['kind'] == 'selection']
        handles.append(Patch(color=ROI_COLOR, alpha=.35, label='Editable ROI sphere'))
        ax.legend(handles=handles, loc='upper left', bbox_to_anchor=(-.01, 1.02), fontsize=8)
    static.suptitle('Translucent surface rendering of the same crop\n'
                    'Static marching-cubes preview; interactive HTML uses occupancy Volume traces', fontsize=15, y=.97)
    static.text(.04, .025, f'Max-pool stride {volume_stride}; displayed support expands. '
                'Native relative mm; orientation unverified.\n'
                'Selected-surface opacity is enhanced for visibility; context retains configured opacity. '
                'Use 2D overlays for exact ROI placement.', fontsize=9)
    static.subplots_adjust(left=.015, right=.96, top=.85, bottom=.1, wspace=.12)
    static.savefig(paths['png'], dpi=160)
    plt.close(static)
    metadata = {'schema_version': 'fakect-volume-preview-1',
                'html_path': str(paths['html']), 'png_path': str(paths['png']),
                'metadata_path': str(paths['metadata']),
                'source_shape_kji': list(original.shape), 'display_shape_kji': list(pooled.shape),
                'volume_render_shape_kji': [n + 2 for n in pooled.shape],
                'crop_origin_ijk': origin.tolist(), 'spacing_ijk_mm': spacing.tolist(),
                'crop_bounds_ijk_relative_mm': bounds.tolist(), 'roi_center_ijk': center.tolist(),
                'roi_radius_mm': float(roi_radius_mm), 'volume_stride': int(volume_stride),
                'sampling': 'block maximum binary occupancy; occupied display blocks may enlarge anatomy',
                'volume_boundary': 'one zero occupancy block around crop; .5 boundary intersects crop edge',
                'coordinates': 'native index times spacing in mm; physical origin and orientation unverified',
                'interactive_renderer': 'Plotly Volume (binary occupancy, not attenuation)',
                'static_renderer': 'marching-cubes translucent surfaces of the same occupancy grid',
                'static_panels': list(panel_titles),
                'static_opacity': 'selected surfaces enhanced up to .8; context retains configured opacity; zero stays zero',
                'original_selected_labels': selected_records, 'traces': list(reversed(trace_metadata)),
                'empty_context_tissues': missing_context, 'warnings': warnings,
                'html_bytes': paths['html'].stat().st_size,
                'standalone_plotly_javascript': True,
                'source_arrays_modified': False}
    paths['metadata'].write_text(json.dumps(metadata, indent=2) + '\n')
    return metadata
