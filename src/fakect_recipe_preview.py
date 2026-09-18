"""Native-coordinate visual review of named edit regions and sequential passes."""
import os
from pathlib import Path

import numpy as np


ROI_COLORS = ('#ed7c21', '#7060d8', '#16a37b', '#de508c', '#c2a300', '#238ec7')
_PLANES = ((2, 0, 1, 'Axial'), (1, 0, 2, 'Coronal'), (0, 1, 2, 'Sagittal'))


def _focus(mask, target, low):
    """Use an occupied native voxel nearest the region's target centroid."""
    points = np.argwhere(np.asarray(mask, dtype=bool) & np.asarray(target, dtype=bool))
    if not len(points):
        points = np.argwhere(mask)
    if not len(points):
        raise ValueError('Cannot render a named ROI with no effective voxels')
    center = points.mean(axis=0)
    chosen = points[np.argmin(np.sum((points - center) ** 2, axis=1))]
    return tuple(int(v) for v in chosen[::-1] + np.asarray(low))


def _nodes(definition):
    nodes = np.asarray(definition['center_ijk'], dtype=float)
    if nodes.ndim == 1:
        nodes = nodes[None, :]
    radii = np.atleast_1d(definition['radius_mm']).astype(float)
    return nodes, radii


def render_recipe_rois(arrays, resolved, config, masks, output):
    """Render all effective named masks against the original full-crop target.

    ``masks`` maps names to native boolean masks already intersected with the
    immutable main ROI. No morphology is applied by this renderer.
    """
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch
    from mpl_toolkits.mplot3d.art3d import Poly3DCollection
    from fakect_recipe import recipe_regions
    from fakect_volume_preview import occupancy_grid, _surface

    output = Path(output)
    output.mkdir(parents=True, exist_ok=True)
    low = np.asarray(resolved['crop_low_ijk'])
    high = np.asarray(resolved['crop_high_ijk_exclusive'])
    spacing = np.asarray(resolved['spacing_ijk_mm'])
    selection_role = config.get('recipe', {}).get('roi_role') == 'selection'
    target = np.asarray(arrays['candidates'], dtype=bool)
    image = arrays['attenuation_cm_inverse']
    window = (float(image.min()), max(float(image.max()), float(image.min()) + .001))
    definitions = recipe_regions(config, resolved)
    rows = []
    for index, (name, mask) in enumerate(masks.items()):
        definition = definitions[name]
        nodes, radii = _nodes(definition)
        focus = (_focus(mask, target, low) if mask.any() else
                 tuple(int(v) for v in np.clip(np.rint(nodes.mean(axis=0)), low, high-1)))
        rows.append({'name': name, 'color': ROI_COLORS[index % len(ROI_COLORS)],
                     'display_name': definition.get('display_name', name),
                     'parent_roi': definition.get('parent_roi'),
                     'range_metadata': definition.get('range_metadata', {}),
                     'shape': definition['shape'], 'nodes_ijk': nodes.tolist(),
                     'radii_mm': radii.tolist(), 'effective_voxels': int(mask.sum()),
                     'target_voxels': int((mask & target).sum()),
                     'focus_ijk': focus})

    def panel(ax, spec, focus, limits=None, highlighted=None):
        fixed, x, y, view = spec
        level = focus[fixed]
        take = lambda field: np.take(field, int(level - low[fixed]), axis=2-fixed)
        extent = (low[x]-.5, high[x]-.5, low[y]-.5, high[y]-.5)
        ax.imshow(take(image), origin='lower', cmap='gray', extent=extent,
                  aspect=spacing[y]/spacing[x], interpolation='nearest',
                  vmin=window[0], vmax=window[1])
        selected = take(target)
        if selected.any() and not selected.all():
            ax.contour(np.arange(low[x], high[x]), np.arange(low[y], high[y]), selected,
                       levels=[.5], colors=['#00efff'], linewidths=1.05)
        outer = take(arrays['roi'])
        if outer.any() and not outer.all():
            ax.contour(np.arange(low[x], high[x]), np.arange(low[y], high[y]), outer,
                       levels=[.5], colors=['#ffb300' if selection_role else '#eeeeee'],
                       linewidths=1 if selection_role else .65, linestyles='-' if selection_role else '--')
        for row in rows:
            region = take(masks[row['name']])
            if not region.any():
                continue
            rgba = np.zeros((*region.shape, 4))
            rgba[..., :3] = to_rgba(row['color'])[:3]
            rgba[..., 3] = region * config['preview']['overlay_opacity']
            ax.imshow(rgba, origin='lower', extent=extent, aspect=spacing[y]/spacing[x],
                      interpolation='nearest')
            if not region.all():
                ax.contour(np.arange(low[x], high[x]), np.arange(low[y], high[y]), region,
                           levels=[.5], colors=[row['color']], linewidths=1)
            if highlighted is None or highlighted == row['name']:
                points = np.argwhere(region)
                center = points.mean(axis=0)
                ax.text(low[x]+center[1], low[y]+center[0], row['display_name'], fontsize=8,
                        ha='center', color='white', bbox={'facecolor': row['color'], 'alpha': .7, 'pad': 2})
        ax.axvline(focus[x], color='white', lw=.55, ls=':')
        ax.axhline(focus[y], color='white', lw=.55, ls=':')
        ax.set(title=f'{view}: {"ijk"[fixed]}={level}', xlabel=f'{"ijk"[x]} (native index)',
               ylabel=f'{"ijk"[y]} (native index)')
        if limits is not None:
            ax.set(xlim=(limits[0][x]-.5, limits[1][x]-.5),
                   ylim=(limits[0][y]-.5, limits[1][y]-.5))

    legend = [Line2D([0], [0], color='#00b8c9', label='Original full-crop target'),
              Line2D([0], [0], color='#ffb300' if selection_role else '#888888',
                     ls='-' if selection_role else '--',
                     label='Original main selector (growth can extend beyond it)' if selection_role else 'Main ROI: hard edit boundary')]
    legend += [Patch(facecolor=row['color'], alpha=.5, label=row['display_name']) for row in rows]
    fig, axes = plt.subplots(1, 3, figsize=(17, 7))
    for ax, spec in zip(axes, _PLANES):
        panel(ax, spec, resolved['slice_ijk'])
    fig.suptitle(config['study']['name'] + ' | Named regions on the original attenuation\n'
                 + ('Original selectors choose target seeds; offspring can grow beyond them. Cyan: original full-crop target.'
                    if selection_role else 'Colored regions are clipped to the main ROI; cyan shows the full target in this crop.'), fontsize=14)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .02), ncol=min(5, len(legend)), frameon=False)
    fig.tight_layout(rect=(0, .12, 1, .91))
    fig.savefig(output/'recipe-rois.png', dpi=145)
    plt.close(fig)

    fig, axes = plt.subplots(len(rows), 3, figsize=(16, 4.8*len(rows)), squeeze=False)
    for r, row in enumerate(rows):
        points = np.argwhere(masks[row['name']])[:, ::-1] + low
        if not len(points):
            points = np.asarray([row['focus_ijk']])
        margin = np.ceil(5/spacing)
        bounds = (np.maximum(low, points.min(axis=0)-margin),
                  np.minimum(high, points.max(axis=0)+margin+1))
        for ax, spec in zip(axes[r], _PLANES):
            panel(ax, spec, row['focus_ijk'], bounds, row['name'])
            ax.set_title(row['display_name'] + ' | ' + ax.get_title() + '\n'
                         + 'Focus i,j,k=' + ','.join(map(str, row['focus_ijk'])))
    fig.suptitle('Per-region native close-ups | Source anatomy before all edits', fontsize=14)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .015), ncol=min(5, len(legend)), frameon=False)
    fig.tight_layout(rect=(0, .07, 1, .95))
    fig.savefig(output/'recipe-roi-closeups.png', dpi=145)
    plt.close(fig)

    stride = config['preview']['volume_stride']
    bounds = tuple((float(low[a]-.5), float(high[a]-.5)) for a in range(3))
    fig = plt.figure(figsize=(15, 8))
    for plot_index, azimuth in enumerate((-60, 40), 1):
        ax = fig.add_subplot(1, 2, plot_index, projection='3d')
        for mask, color, opacity in [(target, '#00a9ba', .50)] + [
                (masks[row['name']], row['color'], .18) for row in rows]:
            sampled, coordinates = occupancy_grid(mask, stride, low, (1, 1, 1))
            vertices, faces = _surface(sampled, coordinates, bounds)
            if len(faces):
                surface = Poly3DCollection(vertices[faces], facecolors=color, alpha=opacity,
                                           edgecolor='none', linewidth=0, rasterized=True)
                ax.add_collection3d(surface)
        for row in rows:
            nodes = np.asarray(row['nodes_ijk'])
            ax.plot(nodes[:, 0], nodes[:, 1], nodes[:, 2], color=row['color'], linewidth=2)
            center = row['focus_ijk']
            ax.text(*center, row['display_name']+'\n'+','.join(map(str, center)), color=row['color'], fontsize=8)
        ax.set(xlim=bounds[0], ylim=bounds[1], zlim=bounds[2], xlabel='i (native index)',
               ylabel='j (native index)', zlabel='k (native index)')
        ax.set_box_aspect((high-low)*spacing)
        ax.view_init(elev=22, azim=azimuth)
    fig.suptitle('Original full-crop target and transparent effective named regions\n'
                 'Working region names do not verify anatomical orientation.', fontsize=14)
    # The dashed main-ROI outline belongs to the 2D views; the 3D view draws
    # only target and effective named-region surfaces.
    surface_legend = [legend[0], *legend[2:]]
    fig.legend(handles=surface_legend, loc='lower center', bbox_to_anchor=(.5, .065),
               ncol=min(5, len(surface_legend)), frameon=False)
    fig.text(.5, .025, f'Display stride {stride}: occupied blocks can widen thin structures; native masks determine counts. '
             'Axes retain i,j,k indices; physical spacing controls aspect.', ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .12, 1, .90))
    fig.savefig(output/'recipe-roi-surfaces.png', dpi=145)
    plt.close(fig)
    result = {'overview': 'recipe-rois.png', 'closeups': 'recipe-roi-closeups.png',
            'surfaces': 'recipe-roi-surfaces.png', 'rois': rows,
            'target_voxels_in_crop': int(target.sum()), 'display_stride': stride,
            'mask_semantics': ('Colored regions are original target selectors; permitted per-pass edit footprints may extend outside them'
                               if selection_role else 'Each region is intersected with the immutable main ROI; selected tube ranges are also clipped by the parent path coordinate'),
            'name_semantics': 'Region names are working labels; anatomical orientation remains unverified'}
    if selection_role:
        result['roi_role'] = 'selection'
    return result


def render_recipe_step(event, output):
    """Render one applied pass at a focus voxel within its effective named ROI."""
    from fakect_edit_preview import render_edit_comparison
    arrays = event['before_arrays']
    edit = event['result']
    resolved = dict(event['resolved'])
    directory = event.get('artifact_directory',
                          f"steps/{event['index']:02d}-{event['step_name']}-iteration-{event['iteration']:02d}")
    output = Path(output)
    destination = output/directory
    destination.mkdir(parents=True, exist_ok=True)
    selection_role = event['config'].get('recipe', {}).get('roi_role') == 'selection'
    display_mask = np.asarray(arrays['roi'])
    if selection_role:
        display_mask = (display_mask | np.asarray(arrays.get('edit_region', arrays['roi'])) |
                        np.asarray(arrays['selected']) | edit['added_mask'] | edit['removed_mask'])
    if not display_mask.any():
        return {'artifact_directory': str(directory),
                'skipped': 'No effective ROI voxels; no slice comparison was generated'}
    focus = _focus(display_mask, arrays['selected'], resolved['crop_low_ijk'])
    changed = (np.asarray(edit.get('changed_mask', edit['added_mask'] | edit['removed_mask']))
               if selection_role else np.asarray(edit['added_mask']) | np.asarray(edit['removed_mask']))
    if changed.any():
        focus = _focus(display_mask, changed, resolved['crop_low_ijk'])
    resolved['slice_ijk'] = focus
    resolved['focus_ijk'] = focus
    figures = render_edit_comparison(arrays, edit, resolved, event['config'], destination,
                                     before_is_source=False)
    figures['comparison'] = str(Path(directory)/figures['comparison'])
    figures['profile'] = str(Path(directory)/figures['profile'])
    figures['artifact_directory'] = str(directory)
    figures['focus_ijk'] = list(focus)
    figures['focus_semantics'] = ('A changed voxel near the changed-region centroid, including growth outside the original selector; otherwise a tracked target voxel'
                                  if selection_role else 'A changed voxel near the changed-region centroid, or selected ROI center when no changes applied')
    return figures
