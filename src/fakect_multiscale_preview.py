"""Validated overview figures independent of frozen training implementations.

Boxes use normalized voxel-face coordinates: center_kji followed by extent_kji.
Only the reference box selects the three review slices. Predicted anatomy and
proposal boxes are drawn at their own unchanged absolute coordinates; bounds
outside the scan FOV are permitted. Dashed rectangles are projections of boxes
that do not intersect the displayed slice.
"""
from pathlib import Path

import numpy as np


def _box(value, name):
    try:
        array = np.asarray(value, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must contain six finite numeric values') from exc
    if array.shape not in ((6,), (1, 6)):
        raise ValueError(f'{name} must contain six values: center_kji followed by extent_kji')
    array = array.reshape(6)
    if not np.isfinite(array).all() or np.any(array[3:] <= 0):
        raise ValueError(f'{name} must be finite with three positive extents')
    return array


def _proposals(values):
    if values is None:
        return []
    try:
        array = np.asarray(values, dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('proposal_boxes must contain six-value boxes') from exc
    if array.size == 0 and (array.shape == (0,) or array.shape == (0, 6)):
        return []
    if array.shape == (6,):
        array = array[None]
    if array.ndim != 2 or array.shape[1] != 6:
        raise ValueError('proposal_boxes must have shape [N,6]')
    return [_box(value, f'proposal_boxes[{index}]') for index, value in enumerate(array)]


def preview(row, ov, box, output, predicted=None, proposal_boxes=None):
    """Render three physical-coordinate slices with validated box overlays.

    ``ov`` contains a 3D image in k,j,i order and positive ``spacing_ijk_mm``.
    Target and predicted boxes accept lists or arrays of shape [6] or [1,6].
    Proposals accept [N,6], one [6] box, an empty list/array, or None. Invalid
    geometry is rejected before creating a figure or output file.

    The return value records the actual planes, box bounds, and intersection
    status. It can be saved alongside a report for auditable visualization.
    """
    reference = _box(box, 'box')
    prediction = None if predicted is None else _box(predicted, 'predicted')
    proposals = _proposals(proposal_boxes)
    image = np.asarray(ov['image'])
    if image.ndim != 3 or any(n <= 0 for n in image.shape) or image.dtype.kind not in 'fiu':
        raise ValueError('Overview image must be a nonempty numeric 3D k,j,i array')
    if not np.isfinite(image).all():
        raise ValueError('Overview image must be finite')
    try:
        spacing = np.asarray(ov['spacing_ijk_mm'], dtype=np.float64)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Overview spacing must contain three positive finite values') from exc
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Overview spacing must contain three positive finite values')
    spacing_kji = spacing[::-1]
    physical_extent = np.asarray(image.shape) * spacing_kji
    if not np.isfinite(physical_extent).all():
        raise ValueError('Overview physical extent is not finite')
    shape = np.asarray(image.shape, dtype=np.int64)
    # Clip in normalized coordinates before multiplying, avoiding integer
    # overflow for otherwise finite reference centers far outside the FOV.
    center = np.minimum((np.clip(reference[:3], 0., 1.) * shape).astype(np.int64), shape - 1)
    planes = (center + .5) * spacing_kji
    overlays = [(reference, '#ffdf00', 'Supervision')]
    if prediction is not None:
        overlays.append((prediction, '#00d8ff', 'Prediction'))
    overlays.extend((value, '#ff7199', 'Detail block ' + str(index + 1)) for index, value in enumerate(proposals))
    overlay_records = []
    for value, color, label in overlays:
        lower = (value[:3] - value[3:] / 2) * physical_extent
        size = value[3:] * physical_extent
        upper = lower + size
        if not np.isfinite(lower).all() or not np.isfinite(size).all() or not np.isfinite(upper).all():
            raise ValueError('Box physical bounds are not finite')
        intersects = (lower <= planes) & (planes <= upper)
        overlay_records.append({'label': label, 'color': color, 'box_kji': value.tolist(),
                                'lower_kji_mm': lower.tolist(), 'extent_kji_mm': size.tolist(),
                                'intersects_slice_kji': intersects.tolist()})

    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.patches import Rectangle
    fig, axes = plt.subplots(1, 3, figsize=(12, 6))
    try:
        for axis, ax in enumerate(axes):
            other = [i for i in range(3) if i != axis]
            ax.imshow(np.take(image, int(center[axis]), axis=axis), cmap='gray', vmin=0, vmax=.5, origin='lower',
                      extent=(0, physical_extent[other[1]], 0, physical_extent[other[0]]))
            for record in overlay_records:
                low, size = record['lower_kji_mm'], record['extent_kji_mm']
                intersects = record['intersects_slice_kji'][axis]
                label = record['label'] if intersects else record['label'] + ' (out-of-plane projection)'
                ax.add_patch(Rectangle((low[other[1]], low[other[0]]), size[other[1]], size[other[0]],
                                      fill=False, edgecolor=record['color'], linewidth=1.5,
                                      linestyle='-' if intersects else '--', label=label))
            # Keep the scan extent fixed even for off-FOV proposal rectangles.
            ax.set_xlim(0, physical_extent[other[1]])
            ax.set_ylim(0, physical_extent[other[0]])
            ax.set_title(('Axial', 'Coronal', 'Sagittal')[axis])
            ax.set_xlabel(('k', 'j', 'i')[other[1]] + ' (mm from source origin)')
            ax.set_ylabel(('k', 'j', 'i')[other[0]] + ' (mm)')
        axes[0].legend(loc='lower left', fontsize=7)
        fig.suptitle(str(row.get('sample_id', 'Overview')) + ' · ' + str(row.get('split', ''))
                     + '\nYellow: supervision; cyan: predicted anatomy; pink: actual detail blocks')
        fig.tight_layout()
        fig.savefig(Path(output), dpi=120)
    finally:
        plt.close(fig)
    return {'output': str(Path(output)), 'slice_indices_kji': center.tolist(),
            'slice_centers_kji_mm': planes.tolist(), 'physical_extent_kji_mm': physical_extent.tolist(),
            'overlays': overlay_records,
            'slice_selection': 'Reference box center; predictions and detail boxes retain their own coordinates'}
