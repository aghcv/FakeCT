"""Native-grid before/after figures for bounded morphology trials."""
import os
from pathlib import Path

import numpy as np

from fakect_roi import COLORS


def render_edit_comparison(arrays, edit, resolved, config, output_dir, *, before_is_source=True):
    """Plot labels and applied/proposed changes without displaying a CT proxy as CT."""
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba
    from matplotlib.lines import Line2D
    from matplotlib.patches import Patch

    output = Path(output_dir)
    low = np.asarray(resolved['crop_low_ijk'])
    high = np.asarray(resolved['crop_high_ijk_exclusive'])
    spacing = np.asarray(resolved['spacing_ijk_mm'])
    lut = np.tile(to_rgba(COLORS['unknown']), (256, 1))
    for category in resolved['catalog']['categories']:
        lut[category['id']] = to_rgba(COLORS.get(category['name'], '#999999'))
    cmap = ListedColormap(lut)
    colors = {'added_mask': '#00af77', 'removed_mask': '#d72664',
              'blocked_mask': '#c88c00', 'unresolved_mask': '#7149c6'}
    names = {'added_mask': 'Added target', 'removed_mask': 'Released + reassigned',
             'blocked_mask': 'Blocked proposal', 'unresolved_mask': 'Unresolved; preserved'}
    is_recipe = config['edit']['operation'] == 'recipe'
    selection_role = config.get('recipe', {}).get('roi_role') == 'selection'
    if is_recipe:
        names.update(added_mask='Net added target', removed_mask='Net released target',
                     blocked_mask='Blocked at any pass', unresolved_mask='Unresolved at any pass')
    specs = [(2, 0, 1, 'Axial'), (1, 0, 2, 'Coronal'), (0, 1, 2, 'Sagittal')]
    selection_roi = np.asarray(edit.get('selection_roi_mask', arrays['roi']), dtype=bool)
    edit_region = np.asarray(edit.get('edit_region_mask', arrays.get('edit_region', arrays['roi'])), dtype=bool)
    display_mask = arrays['roi']
    if selection_role:
        display_mask = (selection_roi | edit_region | edit['target_mask_before'] | edit['target_mask_after'] |
                        edit.get('changed_mask', edit['added_mask'] | edit['removed_mask']))
    roi_points = np.argwhere(display_mask)
    # Include the selector, edit footprint and tracked offspring, then retain
    # five millimetres of tissue context around everything being compared.
    display_low = np.maximum(low, roi_points.min(axis=0)[::-1] + low - np.ceil(5 / spacing))
    display_high = np.minimum(high, roi_points.max(axis=0)[::-1] + low + np.ceil(5 / spacing) + 1)
    fig, axes = plt.subplots(3, 3, figsize=(16, 14))
    fig.subplots_adjust(left=.07, right=.98, bottom=.10, top=.87, hspace=.38, wspace=.27)
    for row, (fixed, x, y, name) in enumerate(specs):
        index = resolved['slice_ijk'][fixed]
        take = lambda value: np.take(value, index - low[fixed], axis=2 - fixed)
        roi = take(selection_roi if selection_role else arrays['roi'])
        footprint = take(edit_region) if selection_role else None
        extent = (low[x] - .5, high[x] - .5, low[y] - .5, high[y] - .5)
        for column in range(3):
            ax = axes[row, column]
            if column < 2:
                field = arrays['tissue'] if column == 0 else edit['edited_tissue_labels']
                ax.imshow(take(field), cmap=cmap, vmin=0, vmax=255, origin='lower', extent=extent,
                          aspect=spacing[y] / spacing[x], interpolation='nearest')
                target = take(edit['target_mask_before' if column == 0 else 'target_mask_after'])
                if target.any() and not target.all():
                    ax.contour(np.arange(low[x], high[x]), np.arange(low[y], high[y]), target,
                               levels=[.5], colors=['#00ffff'], linewidths=1)
            else:
                ax.imshow(take(arrays['atn']), cmap='gray', origin='lower', extent=extent,
                          aspect=spacing[y] / spacing[x], interpolation='nearest')
                for key, color in colors.items():
                    mask = take(edit[key])
                    if is_recipe and key in ('blocked_mask', 'unresolved_mask'):
                        # Historical failures must not obscure a later accepted
                        # net change. Keep the full masks for counts/profiles.
                        mask = mask & ~take(edit['added_mask'] | edit['removed_mask'])
                    overlay = np.zeros((*mask.shape, 4))
                    overlay[..., :3] = to_rgba(color)[:3]
                    overlay[..., 3] = .9 * mask
                    ax.imshow(overlay, origin='lower', extent=extent,
                              aspect=spacing[y] / spacing[x], interpolation='nearest')
            if footprint is not None and footprint.any() and not footprint.all():
                ax.contour(np.arange(low[x], high[x]), np.arange(low[y], high[y]), footprint,
                           levels=[.5], colors=['#4169e1'], linewidths=1.1, linestyles='--')
            if roi.any() and not roi.all():
                ax.contour(np.arange(low[x], high[x]), np.arange(low[y], high[y]), roi,
                           levels=[.5], colors=['#ffb300'], linewidths=1.1 if selection_role else .7,
                           linestyles='-' if selection_role else '--')
            heading = [('Before: original tissue' if before_is_source else 'Before: input-state tissue'),
                       'After: reassigned tissue',
                       ('Changes on ORIGINAL attenuation' if before_is_source else 'Changes on input-state attenuation proxy')][column]
            ax.set(title=f'{heading}\n{name}: {"ijk"[fixed]}={index}',
                   xlabel=f'{"ijk"[x]} (native index)', ylabel=f'{"ijk"[y]} (native index)',
                   xlim=(display_low[x]-.5, display_high[x]-.5),
                   ylim=(display_low[y]-.5, display_high[y]-.5))
    operation = config['edit']['operation']
    counts = edit['summary']['counts']
    fig.suptitle(f"{config['study']['name']} | {operation} trial | native labels\n"
                 f"Added {counts['added']:,}; released/reassigned {counts['removed']:,}; "
                 f"blocked {counts['blocked']:,}; unresolved {counts['unresolved']:,}\n"
                 + ('Cyan: tracked target. Orange: original selector. Dashed blue: permitted edit footprint.'
                    if selection_role else 'Cyan: selected target boundary. Dashed orange: immutable ROI boundary.'),
                 fontsize=14, y=.97)
    legend = [Patch(facecolor=color, label=names[key]) for key, color in colors.items()]
    if selection_role:
        legend += [Line2D([0], [0], color='#ffb300', label='Original selection ROI'),
                   Line2D([0], [0], color='#4169e1', ls='--', label='Permitted growth / edit footprint'),
                   Line2D([0], [0], color='#00ffff', label='Tracked selected target')]
    present = set(np.unique(arrays['tissue'])) | set(np.unique(edit['edited_tissue_labels']))
    legend += [Patch(facecolor=COLORS.get(c['name'], '#999999'), label=c['name'].replace('_', ' '))
               for c in resolved['catalog']['categories'] if c['id'] in present]
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .025), ncol=5, fontsize=9, frameon=False)
    fig.text(.5, .009, 'Source volumes are preserved. Geometry/label trial only; no recovered CT is shown.'
             + (' Net changes take display precedence over earlier blocked/unresolved proposals.' if is_recipe else ''),
             ha='center', fontsize=9 if is_recipe else 10)
    fig.savefig(output / 'edit-comparison.png', dpi=145)
    plt.close(fig)

    k = np.arange(low[2], high[2])
    voxel_area = float(spacing[0] * spacing[1])
    before = edit['target_mask_before'].sum(axis=(1, 2)) * voxel_area
    after = edit['target_mask_after'].sum(axis=(1, 2)) * voxel_area
    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
    axes[0].plot(k, before, label='Before', color='#007c91', lw=2)
    axes[0].plot(k, after, label='After', color='#d45a00', lw=1.8)
    axes[0].set(ylabel=('Tracked target area (mm²)' if selection_role else 'Target area inside ROI (mm²)'),
                title=('Selected target and its offspring, including growth outside the selector'
                       if selection_role else 'Native axial area through the ROI'))
    axes[0].legend()
    for key, color in colors.items():
        axes[1].plot(k, edit[key].sum(axis=(1, 2)) * voxel_area, label=names[key], color=color)
    axes[1].set(ylabel='Affected area (mm²)')
    axes[1].legend(ncol=2)
    strength = edit.get('strength_mm')
    if strength is not None:
        axes[2].plot(k, np.where(edit_region if selection_role else arrays['roi'], strength, 0).max(axis=(1, 2)), color='#713fb5')
    region_label = 'edit footprint' if selection_role else 'ROI'
    axes[2].set(ylabel=('Maximum per-pass distance\nin this ' + region_label + ' slice (mm)' if is_recipe else
                        'Maximum requested distance\nin this ' + region_label + ' slice (mm)'), xlabel='k (native axial index)')
    for ax in axes:
        ax.grid(alpha=.2)
        ax.set_xlim(display_low[2], display_high[2])
    fig.suptitle(f'{operation.capitalize()}: requested spatial profile and achieved label geometry', fontsize=14)
    fig.text(.5, .01, ('Areas include tracked target growth outside the original selector; they are not vessel-normal lumen areas.\n'
                     if selection_role else 'Areas are axial ROI intersections, not vessel-normal lumen areas or a measured stenosis percentage.\n') +
             'Six-neighbor distances are weighted by voxel spacing; subvoxel changes may produce no changed labels.',
             ha='center', fontsize=10)
    fig.tight_layout(rect=(0, .065, 1, .955))
    from fakect_arc_profile import build_arc_profile, render_arc_profile
    arc_profile = build_arc_profile(edit, resolved, config, edit_region if selection_role else arrays['roi'])
    fig.savefig(output / ('edit-profile-axial.png' if arc_profile else 'edit-profile.png'), dpi=145)
    plt.close(fig)
    result = {'comparison': 'edit-comparison.png', 'profile': 'edit-profile.png',
            'axial_k': k.tolist(), 'before_roi_axial_area_mm2': before.tolist(),
            'after_roi_axial_area_mm2': after.tolist(),
            'area_semantics': ('Native axial area of the tracked selected lineage, including offspring outside the original selector; not vessel-normal area'
                               if selection_role else 'Native axial target area inside ROI; not vessel-normal area or clinical stenosis')}
    if selection_role:
        result.update(roi_role='selection', selection_roi_voxels=int(selection_roi.sum()),
                      edit_region_voxels=int(edit_region.sum()),
                      display_low_ijk=display_low.tolist(), display_high_ijk_exclusive=display_high.tolist(),
                      before_roi_axial_area_mm2=((edit['target_mask_before'] & selection_roi).sum(axis=(1, 2)) * voxel_area).tolist(),
                      after_roi_axial_area_mm2=((edit['target_mask_after'] & selection_roi).sum(axis=(1, 2)) * voxel_area).tolist(),
                      before_tracked_axial_area_mm2=before.tolist(), after_tracked_axial_area_mm2=after.tolist(),
                      roi_area_semantics='Fixed original-selector intersections, separate from the displayed complete tracked-target areas',
                      mask_semantics='Solid orange is the fixed original selector; dashed blue is the permitted edit footprint; cyan is the tracked target')
    if arc_profile:
        result.update(arc_profile)
        result['axial_profile'] = 'edit-profile-axial.png'
        result.update(render_arc_profile(arc_profile, config, names, colors, output))
    return result
