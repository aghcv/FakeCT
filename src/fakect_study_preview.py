"""Display the full training target separately from the editable ROI."""
import os
from pathlib import Path

import numpy as np


def render_training_target(arrays, resolved, config, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D
    from scipy.ndimage import label

    target = np.isin(arrays['act'], config['train']['target_source_ids'])
    roi = arrays['roi']
    lo, hi = resolved['crop_low_ijk'], resolved['crop_high_ijk_exclusive']
    spacing = resolved['spacing_ijk_mm']
    image = arrays['attenuation_cm_inverse']
    window = (float(image.min()), max(float(image.max()), float(image.min())+.001))
    fig, axes = plt.subplots(2, 3, figsize=(16, 10))
    specs = [(2, 0, 1, 'Axial'), (1, 0, 2, 'Coronal'), (0, 1, 2, 'Sagittal')]
    for column, (fixed, x, y, name) in enumerate(specs):
        n = resolved['slice_ijk'][fixed] - lo[fixed]
        take = lambda value: np.take(value, n, axis=2-fixed)
        mask, region = take(target), take(roi)
        extent = (lo[x]-.5, hi[x]-.5, lo[y]-.5, hi[y]-.5)
        for row in range(2):
            ax = axes[row, column]
            field = take(image) if row == 0 else mask
            ax.imshow(field, cmap='gray', origin='lower', extent=extent,
                      interpolation='nearest', aspect=spacing[y]/spacing[x],
                      vmin=window[0] if row == 0 else 0, vmax=window[1] if row == 0 else 1)
            overlay = np.zeros((*region.shape, 4)); overlay[..., :3] = (1, .6, 0)
            overlay[..., 3] = region * config['preview']['overlay_opacity']
            ax.imshow(overlay, origin='lower', extent=extent, aspect=spacing[y]/spacing[x], interpolation='nearest')
            if mask.any() and not mask.all():
                ax.contour(np.arange(lo[x], hi[x]), np.arange(lo[y], hi[y]), mask,
                           levels=[.5], colors=['#00d9ef'], linewidths=1)
            ax.set(title=f'{name}: {"ijk"[fixed]}={resolved["slice_ijk"][fixed]}\n'
                         + ('Source attenuation (1/cm)' if row == 0 else 'Binary target over the FULL crop'),
                   xlabel=f'{"ijk"[x]} (native index)', ylabel=f'{"ijk"[y]} (native index)')
    fig.suptitle(f"{config['study']['name']} | Pair design before cohort preparation\n"
                 f"Target source IDs {config['train']['target_source_ids']} | "
                 f"{int(target.sum()):,} target voxels in crop; {int((target & roi).sum()):,} inside edit ROI",
                 fontsize=14)
    fig.legend(handles=[Line2D([0],[0],color='#00d9ef',label='Full training-target boundary'),
                        Patch(facecolor='orange',alpha=.3,label='ROI: limits geometry edits only')],
               loc='lower center', bbox_to_anchor=(.5,.04), ncol=2, frameon=False)
    fig.text(.5,.015,'Target is restricted to this exported crop; review source ID coverage and ROI placement. '
             'No complete-anatomy coverage is implied.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.09,1,.92))
    fig.savefig(Path(output)/'training-target.png',dpi=145)
    plt.close(fig)
    return {'target_source_ids': list(config['train']['target_source_ids']),
            'target_voxels_in_crop': int(target.sum()), 'target_voxels_in_roi': int((target & roi).sum()),
            'target_voxels_outside_roi': int((target & ~roi).sum()),
            'target_components_6': int(label(target)[1]),
            'target_definition': 'Membership in specified original IDs over the entire exported crop, not intersected with edit ROI',
            'scope': 'Specified source IDs within this exported crop; no complete-anatomy coverage claim',
            'coordinate_reviewed': config['roi']['coordinate_reviewed'],
            'image_units': 'cm^-1', 'source_image_only': True}
