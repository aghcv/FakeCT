"""Volume-conserving spatial profiles along the original ordered ROI tube.

These are voxel volume per unit path length, not vessel-normal plane areas.
The same nearest-polyline projection as editing keeps distance and path_percent
consistent, including when the tube doubles back along the native axial axis.
"""
import csv
from pathlib import Path

import numpy as np

from fakect_morphology import tube_position_mm


SERIES = {'before': 'target_mask_before', 'after': 'target_mask_after',
          'added': 'added_mask', 'removed': 'removed_mask',
          'blocked': 'blocked_mask', 'unresolved': 'unresolved_mask'}
AREA_SEMANTICS = (
    'Native voxel volume assigned to nearest original-parent centerline arc bins, divided by bin length; '
    'not vessel-normal cross-sectional area, diameter, or measured stenosis. '
    'Voxels projected to the first or last endpoint are reported separately, not piled into the end bins.')


def build_arc_profile(edit, resolved, config, edit_region):
    """Measure current voxel positions on one unchanged full parent reference.

    Intervals use at most 512 equal bins, approximately twice the largest native
    voxel spacing. Each interior voxel contributes exactly once. Endpoint-clamped
    volume is accounted separately because it has no resolved longitudinal span.
    Growth is assigned by its current spatial location, never its ancestor's.
    """
    if resolved.get('roi_kind') != 'tube':
        return None
    nodes = np.asarray(resolved.get('range_parent_nodes_ijk', resolved['roi_nodes_ijk']), dtype=float)
    spacing = np.asarray(resolved['spacing_ijk_mm'], dtype=float)
    region = np.asarray(edit_region, dtype=bool)
    if (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or nodes.ndim != 2 or nodes.shape[1:] != (3,) or not 2 <= len(nodes) <= 512
            or not np.isfinite(nodes).all()):
        raise ValueError('Arc profiles require a finite tube and positive physical spacing')
    if region.ndim != 3 or not region.size or region.size > 2_000_000:
        raise ValueError('Arc profiles require a nonempty crop of at most 2,000,000 voxels')
    lengths = np.linalg.norm(np.diff(nodes * spacing, axis=0), axis=1)
    node_distances = np.r_[0., np.cumsum(lengths)]
    total = float(node_distances[-1])
    if not np.isfinite(total) or total <= 0 or np.any(lengths <= 0):
        raise ValueError('Arc profiles require distinct finite consecutive tube points')
    count = int(min(512, max(1, np.ceil(total / (2 * spacing.max())))))
    edges = np.linspace(0., total, count + 1)
    widths = np.diff(edges)
    position = tube_position_mm(region.shape, resolved['crop_low_ijk'], nodes, spacing)
    # Projection arithmetic can put an exact terminal control point a few ULPs
    # inside the path. Snap only at floating-point precision, not voxel scale.
    endpoint_tolerance = min(total / 4, 32 * np.finfo(float).eps * max(1., total))
    endpoint_start = position <= endpoint_tolerance
    endpoint_end = position >= total - endpoint_tolerance
    interior = ~(endpoint_start | endpoint_end)
    indexes = np.clip(np.searchsorted(edges, position, side='right') - 1, 0, count - 1)
    voxel_volume = float(np.prod(spacing))
    volumes, densities, endpoints = {}, {}, {}
    for name, key in SERIES.items():
        mask = np.asarray(edit[key], dtype=bool)
        if mask.shape != region.shape:
            raise ValueError('Profile masks must share the native crop shape')
        values = np.bincount(indexes[mask & interior], minlength=count) * voxel_volume
        volumes[name] = values.tolist()
        densities[name] = (values / widths).tolist()
        endpoints[name] = [float(np.count_nonzero(mask & end) * voxel_volume)
                           for end in (endpoint_start, endpoint_end)]
    strength = np.asarray(edit.get('strength_mm', np.zeros(region.shape)), dtype=float)
    if strength.shape != region.shape or not np.isfinite(strength).all() or np.any(strength < 0):
        raise ValueError('Profile strength must be a finite nonnegative native crop field')
    maximum = np.zeros(count, dtype=float)
    active = region & interior
    np.maximum.at(maximum, indexes[active], strength[active])
    interval = np.asarray(resolved.get('range_interval_mm', (0., total)), dtype=float)
    if interval.shape != (2,) or not np.isfinite(interval).all() or not 0 <= interval[0] < interval[1] <= total + 1e-8:
        raise ValueError('Profile range must lie along the full parent path')
    if np.allclose(interval, (0., total), atol=1e-10, rtol=1e-12):
        interval = np.array([0., total])
    return {
        'profile_coordinate': 'centerline_arc_length_mm',
        'profile_roi_name': config.get('profile_roi_name', config.get('roi', {}).get('parent_roi', 'main')),
        'profile_length_mm': total,
        'profile_area_semantics': AREA_SEMANTICS,
        'arc_profile': {
            'edges_mm': edges.tolist(), 'centers_mm': ((edges[:-1] + edges[1:]) / 2).tolist(),
            'widths_mm': widths.tolist(), 'volume_mm3': volumes,
            'volume_per_length_mm2': densities, 'endpoint_volume_mm3': endpoints,
            'maximum_strength_mm': maximum.tolist(),
            'endpoint_maximum_strength_mm': [float(strength[region & end].max(initial=0.))
                                            for end in (endpoint_start, endpoint_end)],
            'range_interval_mm': interval.tolist(),
            'node_distance_mm': node_distances.tolist(),
            'voxel_volume_mm3': voxel_volume,
            'coordinate_semantics': 'Distance from the first original full-parent ROI point in supplied order; physical spacing is applied before projection. Same reference as path_percent, not smoothed-spline arc length.',
            'projection_semantics': 'Each current voxel center maps to its nearest original polyline segment; equal-distance ties choose the first segment. Self-approaches can mix branches; this is not an anatomical centerline extraction.',
            'endpoint_semantics': 'Volumes whose closest coordinate is 0 or the total path length (within floating-point projection tolerance) are separate endpoint totals; interior bins plus endpoints conserve each mask volume.',
            'endpoint_tolerance_mm': endpoint_tolerance,
            'bin_semantics': 'Equal physical-length bins, approximately twice the largest native spacing; capped at 512 bins. Voxel-center counts are not subvoxel or interpolated areas.',
            'strength_semantics': 'Maximum requested spatial distance among edit-footprint voxels in each bin, including directional taper, before tissue resistance; recipe uses maximum per-pass request, not a sum or achieved displacement.',
        }}


def render_arc_profile(metadata, config, names, colors, output):
    """Draw a millimetre axis and matching percentage axis; export plotted data."""
    import matplotlib.pyplot as plt

    data = metadata['arc_profile']
    x = np.asarray(data['centers_mm'])
    length = metadata['profile_length_mm']
    roi_name = metadata['profile_roi_name']
    operation = config['edit']['operation']
    recipe = operation == 'recipe'
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    try:
        for key, label, color in [('before', 'Before', '#007c91'), ('after', 'After', '#d45a00')]:
            axes[0].plot(x, data['volume_per_length_mm2'][key], label=label, color=color, lw=1.8)
        axes[0].set_ylabel('Target volume / path length\n(mm³/mm = mm²)')
        axes[0].set_title(f'ROI {roi_name}: current target geometry along the original centerline', pad=52)
        axes[0].legend()
        for key, color in colors.items():
            axes[1].plot(x, data['volume_per_length_mm2'][key.removesuffix('_mask')],
                         label=names[key], color=color)
        axes[1].set(ylabel='Affected volume / path length\n(mm³/mm = mm²)')
        axes[1].legend(ncol=2)
        axes[2].plot(x, data['maximum_strength_mm'], color='#713fb5')
        axes[2].set(ylabel=('Maximum per-pass request\nin arc bin (mm)' if recipe else
                            'Maximum requested distance\nin arc bin (mm)'),
                    xlabel=f'Curvilinear distance from ROI {roi_name} start (mm)')
        interval = data['range_interval_mm']
        for ax in axes:
            ax.grid(alpha=.2)
            ax.set_xlim(0, length)
            if interval != [0., length]:
                ax.axvspan(*interval, color='#e9b949', alpha=.12)
                for boundary in interval:
                    ax.axvline(boundary, color='#a77700', ls=':', lw=.8)
        percent = axes[0].secondary_xaxis('top', functions=(lambda s: s * 100 / length,
                                                           lambda p: p * length / 100))
        percent.set_xlabel('Distance along parent ROI (%) — same reference as path_percent')
        end_before, end_after = (data['endpoint_volume_mm3'][key] for key in ('before', 'after'))
        caption = (f"Bins: {data['widths_mm'][0]:.2f} mm. Target includes tracked growth; values are volume per length, not vessel-normal areas.\n"
                   f'Endpoint-projected volume, excluded from curves (start/end, mm³): before {end_before[0]:g}/{end_before[1]:g}; '
                   f'after {end_after[0]:g}/{end_after[1]:g}.')
        if interval != [0., length]:
            caption += f'\nShading: original selection interval {interval[0]:.2f}–{interval[1]:.2f} mm; the parent reference stays fixed.'
        fig.suptitle(f'{operation.capitalize()}: requested spatial profile and achieved label geometry', fontsize=14)
        fig.text(.5, .013, caption, ha='center', fontsize=9)
        fig.tight_layout(rect=(0, .09, 1, .96))
        fig.savefig(Path(output) / 'edit-profile.png', dpi=145)
    finally:
        plt.close(fig)
    with (Path(output) / 'edit-profile.csv').open('w', newline='', encoding='utf-8') as stream:
        writer = csv.writer(stream)
        writer.writerow(['start_mm', 'end_mm', 'center_mm', 'center_percent',
                         *[name + '_volume_per_length_mm2' for name in SERIES], 'maximum_request_mm'])
        for index, center in enumerate(x):
            writer.writerow([data['edges_mm'][index], data['edges_mm'][index+1], center, center*100/length,
                             *[data['volume_per_length_mm2'][name][index] for name in SERIES],
                             data['maximum_strength_mm'][index]])
    return {'profile_data': 'edit-profile.csv'}
