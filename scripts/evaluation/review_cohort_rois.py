#!/usr/bin/env python3
"""Read native crops and assemble a manual ROI review without editing anatomy.

Each case gets an immutable JSON/PNG snapshot. Later --cases calls may append
new cases to the same draft review directory; matching existing snapshots are
reused. The aggregate HTML/JSON/montage are refreshed. Nothing sets review flags,
executes edit recipes, freezes data, exports pairs or estimates whole-aorta
coverage from a bounded crop.
"""
from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
from decimal import Decimal
import hashlib
import html
import json
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
import numpy as np
from scipy import ndimage
from fakect_study_config import load_study_config
from fakect_roi import prepare_crop, resolve_preview, selection_diagnostics
from fakect_anatomy import verify_selection_sources

SCHEMA = 'fakect.cohort-roi-review/1'


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def path_from(value):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT / path).resolve()


def load_ini(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    with Path(path).open() as stream:
        parser.read_file(stream)
    return parser


def radius_change(config_path, reference):
    """Compare literal decimals, preserving exact 15% user intent."""
    original, current = load_ini(reference), load_ini(config_path)
    records = []
    original_rois = {s for s in original.sections() if s == 'roi' or s.startswith('roi.')}
    current_rois = {s for s in current.sections() if s == 'roi' or s.startswith('roi.')}
    for section in sorted(original_rois | current_rois):
        if section not in original or section not in current:
            records.append({'section': section, 'same_sections': False})
            continue
        old = [Decimal(x.strip()) for x in original[section]['radius_mm'].split(',')]
        new = [Decimal(x.strip()) for x in current[section]['radius_mm'].split(',')]
        old_points = [Decimal(x.strip()) for x in original[section]['center_ijk'].replace(';', ',').split(',') if x.strip()]
        new_points = [Decimal(x.strip()) for x in current[section]['center_ijk'].replace(';', ',').split(',') if x.strip()]
        records.append({'section': section, 'same_sections': True,
                        'exact_radius_scale_1_15': new == [x * Decimal('1.15') for x in old],
                        'points_unchanged': old_points == new_points,
                        'original_radius_mm': list(map(str, old)), 'radius_mm': list(map(str, new))})
    return {'reference_config': str(reference), 'reference_config_sha256': sha(reference),
            'all_exact_radius_scale_1_15': all(r.get('exact_radius_scale_1_15', False) for r in records),
            'all_points_unchanged': all(r.get('points_unchanged', False) for r in records),
            'regions': records}


def bbox(mask, low):
    cells = np.argwhere(mask)
    if not len(cells):
        return None
    return {'low_ijk': (cells.min(axis=0)[::-1] + low).astype(int).tolist(),
            'high_ijk_exclusive': (cells.max(axis=0)[::-1] + low + 1).astype(int).tolist()}


def native_metrics(arrays, resolved):
    catalog = {int(r['original_id']): r for r in resolved['catalog']['records']}
    audit = resolved['anatomical_selection']
    anchor_ids = [int(r['original_id']) for r in audit['records'] if r['role'] == 'anchor']
    anchor = np.isin(arrays['act'], anchor_ids)
    selected, roi = arrays['selected'], arrays['roi']
    low = np.asarray(resolved['crop_low_ijk'], dtype=int)
    distance = ndimage.distance_transform_edt(~selected, sampling=resolved['spacing_ijk_mm'][::-1])
    nodes = []
    for index, point in enumerate(resolved['roi_nodes_ijk']):
        global_cell = np.floor(np.asarray(point) + .5).astype(int)
        local = (global_cell - low)[::-1]
        inside = bool(np.all(local >= 0) and np.all(local < arrays['act'].shape))
        row = {'node_number_1_based': index + 1, 'center_ijk': list(point),
               'nearest_voxel_ijk': global_cell.tolist(), 'inside_crop': inside}
        if inside:
            cell = tuple(local)
            key = int(arrays['act'][cell]); label = catalog.get(key, {})
            row.update(original_id=key, original_name=label.get('original_name'),
                       tissue=label.get('classification', {}).get('tissue_name'),
                       is_selected=bool(selected[cell]), is_aortic_anchor=bool(anchor[cell]),
                       nearest_selected_distance_mm=float(distance[cell]))
        nodes.append(row)
    diagnostics = selection_diagnostics(arrays)
    anchor_count = int(anchor.sum())
    anchor_inside_roi = int(np.count_nonzero(anchor & roi))
    ids, counts = np.unique(arrays['act'][selected], return_counts=True)
    result = {'selected_original_labels': [
        {'original_id': int(key), 'original_name': catalog[int(key)]['original_name'], 'voxels': int(n)}
        for key, n in zip(ids, counts)],
        'selected_voxels': int(selected.sum()), 'roi_voxels': int(roi.sum()),
        'selected_is_subset_of_roi_and_admitted_ids': bool(np.all(~selected | (roi & arrays['candidates']))),
        'selected_bbox_ijk': bbox(selected, low), 'roi_bbox_ijk': bbox(roi, low),
        'selected_components_6': diagnostics['component_count_6'],
        'selected_component_voxels_6': diagnostics['components_voxels_6'],
        'native_selection_warnings': diagnostics['warnings'],
        'aortic_anchor_ids': anchor_ids, 'aortic_anchor_voxels_in_crop': anchor_count,
        'aortic_anchor_voxels_inside_roi': anchor_inside_roi,
        'fraction_of_cropped_anchor_inside_roi': anchor_inside_roi / anchor_count if anchor_count else None,
        'anchor_fraction_scope': 'Fraction of the atlas aortic anchor within this bounded native crop only; this does not measure whole-aorta coverage or establish completeness of the selected thoracic region.',
        'anatomy_audit_counts': audit['counts'], 'anatomy_audit_warnings': audit['warnings'],
        'unconnected_retained_roi_voxels': audit['counts']['unconnected_retained_roi_voxels'],
        'centerline_nodes': nodes,
        'centerline_nodes_outside_selected_label': [n['node_number_1_based'] for n in nodes if not n.get('is_selected', False)],
        'centerline_nodes_outside_crop': [n['node_number_1_based'] for n in nodes if not n['inside_crop']]}
    return result, anchor


def draw_case(path, arrays, resolved, anchor, case_id, template, metrics):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    low = np.asarray(resolved['crop_low_ijk'])
    high = np.asarray(resolved['crop_high_ijk_exclusive'])
    spacing = resolved['spacing_ijk_mm']
    slices = np.asarray(resolved['slice_ijk'], dtype=int)
    nodes = np.asarray(resolved['roi_nodes_ijk'])
    attenuation = arrays['attenuation_cm_inverse']
    positive = attenuation[attenuation > 0]
    vmax = float(np.percentile(positive, 98)) if positive.size else 1.
    fig, axes = plt.subplots(1, 3, figsize=(14.4, 5.1), constrained_layout=True)
    fig.patch.set_facecolor('#f4f6fa')
    planes = [(0, 1, 2, 'Sagittal'), (1, 0, 2, 'Coronal'), (2, 0, 1, 'Axial')]
    for ax, (fixed, x, y, title) in zip(axes, planes):
        index = int(slices[fixed] - low[fixed])
        plane = lambda a: np.take(a, index, axis=2-fixed)
        extent = (low[x]-.5, high[x]-.5, low[y]-.5, high[y]-.5)
        ax.imshow(plane(attenuation), cmap='gray', origin='lower', extent=extent,
                  vmin=0, vmax=vmax, interpolation='nearest', aspect=spacing[y]/spacing[x])
        shade = np.zeros((*plane(arrays['roi']).shape, 4))
        shade[..., :3] = (0, .88, 1)
        shade[..., 3] = plane(arrays['roi']) * .20
        ax.imshow(shade, origin='lower', extent=extent, interpolation='nearest', aspect=spacing[y]/spacing[x])
        xx = np.arange(low[x], high[x]); yy = np.arange(low[y], high[y])
        for mask, color, width in ((anchor, '#ffd166', 1.0), (arrays['selected'], '#ff39ec', 1.35),
                                   (arrays['roi'], '#00ddff', .65)):
            field = plane(mask)
            if field.any() and not field.all():
                ax.contour(xx, yy, field.astype(float), levels=[.5], colors=[color], linewidths=width)
        # Projection shows the parent path only; the cyan mask is the actual plane intersection.
        ax.plot(nodes[:, x], nodes[:, y], '--', color='white', alpha=.6, lw=.7)
        ax.scatter(nodes[:, x], nodes[:, y], c='white', s=6, alpha=.8)
        for n in (0, len(nodes)-1):
            ax.annotate(str(n+1), nodes[n, [x, y]], color='white', fontsize=8)
        ax.set_xlim(extent[:2]); ax.set_ylim(extent[2:]); ax.grid(alpha=.1)
        ax.set_xlabel('ijk'[x] + ' (native index)'); ax.set_ylabel('ijk'[y] + ' (native index)')
        ax.set_title(f'{title}: {"ijk"[fixed]} = {slices[fixed]}', fontsize=11)
    fig.suptitle(f'{case_id} — {template}\n{metrics["selected_voxels"]:,} selected voxels; '
                 f'{metrics["selected_components_6"]} selected components (6-neighbor); '
                 f'{metrics["unconnected_retained_roi_voxels"]} disconnected voxels retained by ID', fontsize=12)
    fig.supxlabel('Cyan shade/outline: ROI intersection   •   Magenta: selected target   •   Gold: atlas aorta in crop\n'
                  'White dotted line: projected centerline, numbered endpoints; projections need not lie in the displayed slice.', fontsize=9)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor()); plt.close(fig)


def aggregate(output, project, project_path):
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    rows = []
    for case in project['cases']:
        path = output / f'{case["case_id"]}.json'
        if path.exists():
            row = json.loads(path.read_bytes())
            if sha(output / row['image']) != row['image_sha256']:
                raise ValueError(f'Existing case review image changed: {path}')
            rows.append(row)
    if not rows:
        raise ValueError('No case reviews to assemble')
    document = {'schema_version': SCHEMA, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                'project': str(project_path), 'project_sha256': sha(project_path),
                'case_count': len(rows), 'expected_case_count': len(project['cases']),
                'review_approved': False, 'native_edits_executed': False, 'paired_data_generated': False,
                'scope': 'Manual ROI placement review from three native slices and crop-local metrics. Whole-aorta coverage and clinical validity are not established.',
                'cases': [{'case_id': r['case_id'], 'json': f'{r["case_id"]}.json', 'image': r['image'],
                           'selected_voxels': r['metrics']['selected_voxels'],
                           'selected_components_6': r['metrics']['selected_components_6'],
                           'unconnected_retained_roi_voxels': r['metrics']['unconnected_retained_roi_voxels'],
                           'flags': r['flags']} for r in rows]}
    (output / 'review.json').write_text(json.dumps(document, indent=2, allow_nan=False)+'\n')
    fig, axes = plt.subplots(len(rows), 1, figsize=(16, len(rows)*5.67), squeeze=False)
    for ax, row in zip(axes[:, 0], rows):
        ax.imshow(plt.imread(output / row['image'])); ax.axis('off')
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1, hspace=.02)
    fig.savefig(output / 'montage.png', dpi=130, facecolor='white'); plt.close(fig)
    cards = []
    for row in rows:
        flags = '; '.join(row['flags']) or 'No automated geometry flags; manual inspection still required.'
        cards.append(f'<section><h2>{html.escape(row["case_id"])}</h2><p>{html.escape(flags)}</p>'
                     f'<a href="{row["image"]}"><img src="{row["image"]}" alt="Three native views for {row["case_id"]}"></a>'
                     f'<p><a href="{row["case_id"]}.json">Native metrics and centerline-node labels</a></p></section>')
    (output / 'report.html').write_text('<!doctype html><html><head><meta charset="utf-8"><title>CoA ROI review</title>'
        '<style>body{font-family:system-ui;margin:2rem auto;max-width:1500px;background:#f3f5fa;color:#15283d}'
        'section{background:white;padding:1rem;margin:1rem 0;border-radius:8px}img{width:100%}a{color:#135db2}</style></head><body>'
        f'<h1>CoA ROI placement review — {len(rows)} / {len(project["cases"])} phantoms</h1>'
        '<p>Native ACT/attenuation crops, enlarged selectors and resolved anatomical labels. '
        'This report does not approve coordinates or parameters. Cyan is the ROI, magenta the selected target, '
        'and gold the atlas aorta within the crop. Disconnected selected components may be branches; inspect their placement.</p>'
        '<p><a href="montage.png">All-case montage</a> · <a href="review.json">Summary JSON</a></p>'
        + ''.join(cards) + '</body></html>')
    return document


def run(project_path, output, cases=None, reference_directory=None):
    project_path, output = path_from(project_path), path_from(output)
    reference_directory = path_from(reference_directory or 'configs/cohorts/coa')
    project_bytes = project_path.read_bytes(); project = json.loads(project_bytes)
    if project.get('schema_version') != 'fakect.cohort-project/1':
        raise ValueError('Expected fakect.cohort-project/1 manifest')
    case_ids = [str(r['case_id']) for r in project['cases']]
    if len(case_ids) != len(set(case_ids)) or any(not re.fullmatch(r'[A-Za-z0-9_.-]+', s) for s in case_ids):
        raise ValueError('Case identifiers must be safe and unique')
    wanted = set(cases or case_ids)
    if wanted - set(case_ids):
        raise ValueError(f'Unknown case IDs: {sorted(wanted-set(case_ids))}')
    output.mkdir(parents=True, exist_ok=True)
    if (output / 'review.json').exists():
        previous = json.loads((output / 'review.json').read_bytes())
        if previous.get('schema_version') != SCHEMA or previous.get('project_sha256') != hashlib.sha256(project_bytes).hexdigest():
            raise ValueError('Existing draft belongs to a different project revision; choose a new output directory')
    for row in project['cases']:
        case_id = str(row['case_id'])
        if case_id not in wanted:
            continue
        config_path = path_from(row['config']); before = config_path.read_bytes()
        config = load_study_config(config_path); resolved = resolve_preview(config)
        if str(config['input']['case_id']) != case_id:
            raise ValueError(f'Project/config case ID mismatch: {case_id}')
        if 'anatomical_selection' not in resolved:
            raise ValueError(f'Resolve the anatomy artifact before reviewing case {case_id}')
        snapshot = output / f'{case_id}.json'; image = output / f'{case_id}.png'
        identity = {'input_sha256': hashlib.sha256(before).hexdigest(),
                    'anatomy_sha256': resolved['anatomy_selection_sha256'], 'review_code_sha256': sha(__file__)}
        if snapshot.exists():
            old = json.loads(snapshot.read_bytes())
            if any(old.get(k) != v for k, v in identity.items()) or sha(image) != old.get('image_sha256'):
                raise ValueError(f'Case {case_id} review is stale; choose a new output revision')
            print(f'{case_id}: reused verified review snapshot', flush=True)
            continue
        if image.exists():
            raise ValueError(f'Partial case review exists: {image}; choose a new output revision')
        arrays, sources = prepare_crop(resolved, config)
        metrics, anchor = native_metrics(arrays, resolved)
        radius = radius_change(config_path, reference_directory/f'{case_id}.ini')
        flags = []
        if not radius['all_exact_radius_scale_1_15']: flags.append('Selector radii differ from exact 1.15 reference scale')
        if not radius['all_points_unchanged']: flags.append('Centerline points differ from reference placement')
        if not metrics['selected_voxels']: flags.append('ROI selects no target voxels')
        if metrics['selected_components_6'] > 1: flags.append('Selected target has multiple 6-connected components; inspect branches and isolated pieces')
        if metrics['unconnected_retained_roi_voxels']: flags.append('Whole-ID admission retained disconnected ROI voxels')
        if metrics['centerline_nodes_outside_selected_label']: flags.append('Some centerline nodes lie outside the selected label; inspect their distances and surrounding labels')
        if metrics['centerline_nodes_outside_crop']: flags.append('Centerline nodes lie outside the loaded crop')
        if not metrics['selected_is_subset_of_roi_and_admitted_ids']: flags.append('Invalid selected-mask membership')
        draw_case(image, arrays, resolved, anchor, case_id, row.get('template', ''), metrics)
        document = {'schema_version': SCHEMA, 'case_id': case_id, 'config': str(config_path), **identity,
                    'image': image.name, 'image_sha256': sha(image),
                    'generated_at_utc': datetime.now(timezone.utc).isoformat(),
                    'review_approved': False, 'coordinate_reviewed_in_input': config['roi']['coordinate_reviewed'],
                    'parameters_reviewed_in_input': config['train']['parameters_reviewed'],
                    'slice_ijk': list(resolved['slice_ijk']), 'crop_low_ijk': list(resolved['crop_low_ijk']),
                    'crop_high_ijk_exclusive': list(resolved['crop_high_ijk_exclusive']),
                    'spacing_ijk_mm': list(resolved['spacing_ijk_mm']), 'sources': sources,
                    'radius_change': radius, 'metrics': metrics, 'flags': flags}
        verify_selection_sources(resolved)
        if config_path.read_bytes() != before:
            raise ValueError(f'Input changed while reviewing {case_id}; retain partial output and rerun in a fresh directory')
        with snapshot.open('x') as stream:
            stream.write(json.dumps(document, indent=2, allow_nan=False)+'\n')
        print(f'{case_id}: {metrics["selected_voxels"]:,} selected voxels; '
              f'{metrics["selected_components_6"]} components; {len(flags)} review flags', flush=True)
        del arrays, anchor
    if project_path.read_bytes() != project_bytes:
        raise ValueError('Project changed during review')
    result = aggregate(output, project, project_path)
    print(json.dumps({'report': str(output/'report.html'), 'case_count': result['case_count'],
                      'review_approved': False, 'native_edits_executed': False}, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', required=True)
    parser.add_argument('--output', default='outputs/cohorts/coa/radius115/roi-review-r1')
    parser.add_argument('--cases', nargs='+', help='Case IDs, separated by spaces or commas; other cases are untouched')
    parser.add_argument('--reference-directory', default='configs/cohorts/coa')
    args = parser.parse_args()
    cases = [part for value in (args.cases or []) for part in value.split(',') if part]
    try:
        run(args.project, args.output, cases, args.reference_directory)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
