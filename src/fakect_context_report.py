"""Offline anatomical review of all context-cohort baseline masks.

This report displays reference labels and attenuation, never model predictions.
Native, unresampled mask surfaces retain the existing viewer's display limits.
"""
from html import escape
import json
from pathlib import Path

import numpy as np

from fakect_context_data import inspect_dataset
from fakect_context_volume import ContextVolume
from fakect_mask_pair_preview import sha256
from fakect_two_stage_3d import build_scene, render_viewer

FOOTER = ('Provisional atlas target: T12 midpoint to T1 upper bound with proximal connected arterial branches. '
          'Review the inferior cutoff, aortic root and branch coverage for every phantom. Native voxel surfaces have '
          'crop-edge closures and no smoothing. These are reference masks, not AI predictions or proof of physiological connectivity.')


def _baseline_rows(dataset):
    rows = [row for row in dataset['samples'] if row['baseline']]
    cases = {row['case_id'] for row in dataset['samples']}
    if len(rows) != len(cases) or {row['case_id'] for row in rows} != cases:
        raise ValueError('Exactly one baseline per context-cohort case is required')
    return sorted(rows, key=lambda row: row['case_id'])


def _plane_mask(mask, mask_origin_kji, low_kji, shape_kji, axis):
    """Place one native compact mask onto a bounded whole-scan slice."""
    mask, origin = np.asarray(mask, bool), np.asarray(mask_origin_kji, int)
    low, shape = np.asarray(low_kji, int), np.asarray(shape_kji, int)
    if mask.ndim != 3 or low.shape != (3,) or shape.shape != (3,) or shape[axis] != 1:
        raise ValueError('Expected a singleton native slice')
    result = np.zeros(tuple(shape), bool)
    a, b = np.maximum(low, origin), np.minimum(low + shape, origin + mask.shape)
    if np.all(b > a):
        target = tuple(slice(int(x - l), int(y - l)) for x, y, l in zip(a, b, low))
        source = tuple(slice(int(x - o), int(y - o)) for x, y, o in zip(a, b, origin))
        result[target] = mask[source]
    return np.squeeze(result, axis=axis)


def _slice_specs(mask, crop_origin_ijk, source_shape_kji, spacing_ijk_mm, context_mm):
    """Use maximum target-area planes; show full transverse field of view."""
    mask = np.asarray(mask, bool)
    if mask.ndim != 3 or not mask.any():
        raise ValueError('Cannot review an empty anatomical mask')
    origin = np.asarray(crop_origin_ijk, int)[::-1]
    source_shape = np.asarray(source_shape_kji, int)
    spacing = np.asarray(spacing_ijk_mm, float)[::-1]
    points = []
    for axis in range(3):
        area = mask.sum(axis=tuple(n for n in range(3) if n != axis))
        points.append(int(np.argmax(area)) + int(origin[axis]))
    padding = int(np.ceil(context_mm / spacing[0]))
    low_k = max(0, int(origin[0]) - padding)
    high_k = min(int(source_shape[0]), int(origin[0] + mask.shape[0]) + padding)
    specs = []
    for axis, name, native_axis, horizontal, vertical in ((0, 'Axial', 'k', 'i', 'j'),
                                                        (1, 'Coronal', 'j', 'i', 'k'),
                                                        (2, 'Sagittal', 'i', 'j', 'k')):
        low = np.array([low_k, 0, 0])
        shape = np.array([high_k - low_k, source_shape[1], source_shape[2]])
        low[axis], shape[axis] = points[axis], 1
        specs.append({'name': name, 'axis': axis, 'native_axis': native_axis,
                      'native_index': points[axis], 'low_kji': low.tolist(), 'shape_kji': shape.tolist(),
                      'horizontal_axis': horizontal, 'vertical_axis': vertical})
    return specs


def _slice_figure(row, selection, masks, mask_origin_ijk, output, *, context_mm, attenuation_max):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap
    specs = _slice_specs(masks['core'] | masks['branches'], mask_origin_ijk,
                         row['source_shape_kji'], row['spacing_ijk_mm'], context_mm)
    figure, axes = plt.subplots(1, 3, figsize=(15.8, 6.1), constrained_layout=True)
    spacing_kji = np.asarray(row['spacing_ijk_mm'])[::-1]
    with ContextVolume(row) as volume:
        for ax, spec in zip(axes, specs):
            axis = spec['axis']
            low, shape = np.array(spec['low_kji']), np.array(spec['shape_kji'])
            block = volume.read_block(low, shape, row['spacing_ijk_mm'])
            plane = np.squeeze(block['image'], axis=axis)
            plane_axes = [n for n in range(3) if n != axis]
            vertical, horizontal = plane_axes
            extent = [float(low[horizontal] - .5), float(low[horizontal] + shape[horizontal] - .5),
                      float(low[vertical] - .5), float(low[vertical] + shape[vertical] - .5)]
            aspect = float(spacing_kji[vertical] / spacing_kji[horizontal])
            base = ax.imshow(plane, cmap='gray', origin='lower', extent=extent,
                             vmin=0., vmax=attenuation_max, interpolation='nearest', aspect=aspect)
            for key, color in (('core', '#e6503e'), ('branches', '#00c8b4')):
                overlay = _plane_mask(masks[key], np.asarray(mask_origin_ijk)[::-1], low, shape, axis)
                ax.imshow(np.ma.masked_where(~overlay, overlay), cmap=ListedColormap([color]),
                          origin='lower', extent=extent, interpolation='nearest', alpha=.6,
                          vmin=0, vmax=1, aspect=aspect)
            ax.set_title(f"{spec['name']} · {spec['native_axis']} = {spec['native_index']}")
            ax.set_xlabel(spec['horizontal_axis'] + ' · native voxel index')
            ax.set_ylabel(spec['vertical_axis'] + ' · native voxel index')
            spec['attenuation_min_cm_inverse'] = float(plane.min())
            spec['attenuation_max_cm_inverse'] = float(plane.max())
    figure.colorbar(base, ax=list(axes), shrink=.5, label='Attenuation (cm⁻¹)')
    figure.suptitle(f"{row['case_id']} · original whole-body attenuation + provisional thoracic target", fontsize=14)
    figure.savefig(output, dpi=140, facecolor='white')
    plt.close(figure)
    return specs


def _selection(row):
    path = Path(row['selection_manifest'])
    if sha256(path) != row['selection_manifest_sha256']:
        raise ValueError('Selection manifest changed')
    selection = json.loads(path.read_text())
    filename = path.parent / selection['mask_npz']
    if sha256(filename) != selection['mask_npz_sha256']:
        raise ValueError('Anatomical payload changed')
    with np.load(filename, allow_pickle=False) as data:
        masks = {'core': np.asarray(data['core_mask'], bool), 'branches': np.asarray(data['branch_mask'], bool)}
        spacing, origin = data['spacing_ijk_mm'], data['crop_origin_ijk']
        if (np.any(masks['core'] & masks['branches']) or
                not np.array_equal(data['mask'], masks['core'] | masks['branches'])):
            raise ValueError('Anatomical component masks disagree with combined target')
    with np.load(row['mask_npz'], allow_pickle=False) as data:
        if (not np.array_equal(data['before'], data['after']) or
                not np.array_equal(data['after'], masks['core'] | masks['branches']) or
                not np.array_equal(origin, data['crop_origin_ijk'])):
            raise ValueError('Baseline reference differs from its anatomical selection')
    return selection, masks, spacing, origin


def render_report(manifest_path, output_directory, *, context_mm=30., attenuation_max=.3):
    """Create a new portable report with one exact native scene per baseline."""
    if (not np.isfinite(context_mm) or context_mm < 0 or context_mm > 150 or
            not np.isfinite(attenuation_max) or attenuation_max <= 0 or attenuation_max > 10):
        raise ValueError('Use context_mm in [0,150] and attenuation_max in (0,10]')
    dataset = inspect_dataset(manifest_path, verify_payloads=False)
    rows = _baseline_rows(dataset)
    output = Path(output_directory).resolve()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'INCOMPLETE').write_text('Building anatomical review\n')
    cards = []
    def scenes():
        for index, row in enumerate(rows):
            selection, masks, spacing, origin = _selection(row)
            image = f"slices-{row['case_id']}.png"
            specs = _slice_figure(row, selection, masks, origin, output / image,
                                  context_mm=context_mm, attenuation_max=attenuation_max)
            z_low, z_high = selection['z_slab_mm']
            explanation = (f"Phantom {row['case_id']}; native crop origin i,j,k = {origin.tolist()}. "
                           f"RAW-z slab {z_low:.2f}–{z_high:.2f} mm. Proximal branch reach "
                           f"{selection['branch_reach_mm']:.2f} mm from the aortic core. "
                           "Baseline anatomical labels; no supplied edit ROI or model prediction.")
            counts = {key: int(mask.sum()) for key, mask in masks.items()}
            cards.append({'case_id': row['case_id'], 'sample_id': row['sample_id'],
                          'anatomy_family': row['anatomy_family'], 'image': image, 'image_sha256': sha256(output / image),
                          'selection_manifest_sha256': row['selection_manifest_sha256'],
                          'mask_npz_sha256': row['mask_npz_sha256'], 'counts': counts,
                          'z_slab_mm': selection['z_slab_mm'], 'branch_reach_mm': selection['branch_reach_mm'],
                          'spacing_ijk_mm': spacing.tolist(), 'crop_origin_ijk': origin.tolist(),
                          'slices': specs})
            yield build_scene(masks, spacing, sample_id=row['sample_id'], modes=[{
                'key': 'anatomy', 'label': 'Baseline anatomical extent', 'explanation': explanation,
                'layers': [{'key': 'core', 'label': 'Aortic core', 'color': '#e6503e', 'visible': True, 'opacity': .45},
                           {'key': 'branches', 'label': 'Proximal connected branches', 'color': '#00c8b4',
                            'visible': True, 'opacity': .8}]}],
                metadata={'case_id': row['case_id'], 'is_control': True,
                          'selection_sha256': row['selection_manifest_sha256'], 'crop_origin_ijk': origin.tolist()},
                show_metrics=False)
    viewer = render_viewer(output, scenes(), title='Context cohort · review baseline aortic targets',
                           coordinate_description='Native-grid millimeters from lower crop face; native i,j,k origin is recorded per case. Anatomical orientation remains unverified.',
                           footer_note=FOOTER, show_metrics=False)
    table = ''.join('<tr><td>' + escape(card['case_id']) + '</td><td>' + escape(card['anatomy_family']) +
                    '</td><td>' + f"{card['counts']['core']:,}" + '</td><td>' + f"{card['counts']['branches']:,}" +
                    '</td><td>' + f"{card['branch_reach_mm']:.2f}" + '</td></tr>' for card in cards)
    overview = (f'<p class="notice">{len(rows)} baseline phantoms · {dataset["sample_count"]} paired scenarios. '
                'Anatomical extent and family assignments remain provisional. Review the root, inferior thoracic cutoff, '
                'and included/excluded branches before interpreting segmentation performance.</p>'
                '<details><summary>All case extents and provisional families</summary><table><thead><tr><th>Case</th>'
                '<th>Provisional family</th><th>Core voxels</th><th>Branch voxels</th><th>Branch reach · mm</th>'
                '</tr></thead><tbody>' + table + '</tbody></table></details>')
    slices = ('<section id="attenuation"><h2>Attenuation and reference labels</h2><p id="slice-info"></p>'
              '<p><span style="color:#e6503e">■ Aortic core</span> · <span style="color:#00a99a">■ Proximal branches</span>. '
              'Each plane passes through its largest target cross-section. Images preserve native voxel sampling; '
              'the axial view spans the full transverse field of view. Coronal and sagittal views show the '
              'thoracic selection plus vertical context. Click to open the image at full size.</p>'
              '<a id="slice-link" target="_blank" rel="noopener"><img id="slice-image" style="width:100%;height:auto" '
              'alt="Axial, coronal and sagittal attenuation images with anatomical mask overlays"></a></section>')
    cards_json = json.dumps(cards, allow_nan=False).replace('<', '\\u003c')
    script = ('<script>(function(){const cards=' + cards_json + '; const select=document.getElementById("sample");'
              'function update(){const c=cards[Number(select.value)||0];document.getElementById("slice-info").textContent='
              'c.case_id+" · "+c.anatomy_family+" (provisional) · spacing i,j,k: "+c.spacing_ijk_mm.join(", ")+" mm";'
              'document.getElementById("slice-image").src=c.image;document.getElementById("slice-link").href=c.image;}'
              'select.addEventListener("change",update);update();})();</script>')
    index = output / 'index.html'
    text = index.read_text()
    text = text.replace('<div class="toolbar">', overview + '<div class="toolbar">', 1)
    text = text.replace('<p id="scope"', slices + '<p id="scope"', 1)
    text = text.replace('</body>', script + '</body>')
    text = text.replace('Reset shows the entire saved crop, including distant false positives.',
                        'Reset shows the entire saved anatomical crop.')
    text = text.replace('</style>', 'table{border-collapse:collapse;margin:12px 0}th,td{padding:7px 12px;border-bottom:1px solid #d8e2e9;text-align:left}summary{cursor:pointer}#attenuation{margin-top:24px}#attenuation h2{font-size:20px}</style>', 1)
    index.write_text(text)
    if sha256(manifest_path) != dataset['manifest_sha256']:
        raise ValueError('Dataset manifest changed during report generation')
    record = {'schema_version': 'fakect.context-anatomy-review/1', 'dataset_manifest': dataset['manifest_path'],
              'dataset_manifest_sha256': dataset['manifest_sha256'], 'case_count': len(rows),
              'anatomical_extent_reviewed': False, 'context_mm': context_mm,
              'attenuation_display_range_cm_inverse': [0., attenuation_max], 'cases': cards,
              'renderer_sha256': sha256(Path(__file__)),
              'surface_renderer_sha256': sha256(Path(__file__).with_name('fakect_two_stage_3d.py')),
              'report_html_sha256': sha256(index), **viewer}
    (output / 'review-provenance.json').write_text(json.dumps(record, indent=2, allow_nan=False) + '\n')
    (output / 'INCOMPLETE').unlink()
    return record
