#!/usr/bin/env python3
"""Inspect saved native corner trials using overlays and independent label checks.

This reads diagnostic NPZs and their immutable trial snapshots. It never runs
edits, changes cohort settings, approves a freeze, or renders meshes from RAW.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import sys

import numpy as np
from scipy import ndimage

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(Path(__file__).parent))
from fakect_study_config import load_study_config
from fakect_reassignment import _record_reasons
from fakect_training_data import _array_hash, _file_hash
from review_cohort_outcomes import summarize_sample
from review_internal_fill import conservative_clearance


def parent_arc(points_ijk, nodes_ijk, spacing_ijk_mm):
    """Closest original-polyline physical arc coordinate, not native axial k."""
    spacing = np.asarray(spacing_ijk_mm)
    points = np.asarray(points_ijk) * spacing
    nodes = np.asarray(nodes_ijk) * spacing
    lengths = np.linalg.norm(np.diff(nodes, axis=0), axis=1)
    best = np.full(len(points), np.inf)
    arc = np.zeros(len(points))
    offset = 0.
    for start, delta, length in zip(nodes[:-1], np.diff(nodes, axis=0), lengths):
        t = np.clip(((points - start) @ delta) / length**2, 0, 1)
        distance = np.sum((points - (start + t[:, None] * delta))**2, axis=1)
        take = distance < best
        best[take] = distance[take]
        arc[take] = offset + t[take] * length
        offset += length
    return arc, offset


def component_correspondence(before, after):
    structure = ndimage.generate_binary_structure(3, 1)
    first, nfirst = ndimage.label(before, structure)
    last, nlast = ndimage.label(after, structure)
    sizes = np.bincount(first.ravel())
    pairs = np.unique(first[before & after].astype(np.int64) * (nlast + 1) + last[before & after])
    original = pairs // (nlast + 1)
    survivors = pairs % (nlast + 1)
    rows = []
    for component in range(1, nfirst + 1):
        ids = survivors[original == component]
        rows.append({'original_component': component, 'before_voxels': int(sizes[component]),
            'surviving_component_ids': ids.tolist(), 'split_or_lost': len(ids) != 1})
    merges = [{'after_component': int(component), 'original_components': original[survivors == component].tolist()}
              for component in np.unique(survivors) if np.count_nonzero(survivors == component) > 1]
    return {'before_components': nfirst, 'after_components': nlast, 'original_components': rows,
        'split_or_lost_original_components': sum(row['split_or_lost'] for row in rows),
        'merged_original_components': merges,
        'scope': 'Six-face connectivity of all admitted anatomical IDs inside the saved native crop. Original component survivors are checked separately; this is not a guarantee of minimum vessel caliber.'}


def native_checks(data, document, catalog):
    before, after = data['before_labels'], data['after_labels']
    changed = before != after
    scalar_changed = data['before_attenuation_per_pixel'] != data['after_attenuation_per_pixel']
    ids = document['candidate_source_ids']
    candidates = np.isin(before, ids)
    candidates_after = np.isin(after, ids)
    category = {row['id']: row['name'] for row in catalog['categories']}
    reasons = {row['original_id']: _record_reasons(row, category) for row in catalog['records']}
    bone_ids = [key for key, value in reasons.items() if 'bone' in value]
    skin_ids = [key for key, value in reasons.items() if 'skin' in value and key != 4]
    excluded_arteries = [row['original_id'] for row in catalog['records']
                        if row['classification'].get('tissue_name') == 'artery' and row['original_id'] not in ids]
    checks = {'changed_mask_matches_native_labels': bool(np.array_equal(changed, data['changed_mask'])),
        'scalar_changed_mask_matches_native_attenuation': bool(np.array_equal(scalar_changed, data['scalar_changed_mask'])),
        'changes_outside_edit_region': int(((changed | scalar_changed) & ~data['edit_region_mask']).sum()),
        'unselected_original_target_changed': int((changed & candidates & ~data['before_selected']).sum()),
        'bone_voxels_changed': int(((changed | scalar_changed) & np.isin(before, bone_ids)).sum()),
        'other_skin_identity_voxels_changed': int(((changed | scalar_changed) & np.isin(before, skin_ids)).sum()),
        'excluded_artery_voxels_changed': int(((changed | scalar_changed) & np.isin(before, excluded_arteries)).sum()),
        'source_act_hash_matches': _array_hash(before) == document['source_crop_sha256']['act'],
        'source_atn_hash_matches': _array_hash(data['before_attenuation_per_pixel']) == document['source_crop_sha256']['atn'],
        'source_selected_hash_matches': _array_hash(data['before_selected']) == document['source_crop_sha256']['selected']}
    selected_after = data['after_target']
    checks['after_lineage_mask_has_non_target_labels'] = int((selected_after & ~candidates_after).sum())
    direct = component_correspondence(candidates, candidates_after)
    borrowed = changed & (before == 4)
    if borrowed.any():
        _, lower = conservative_clearance(before, document['geometry']['spacing_ijk_mm'])
        minimum = float(lower[borrowed].min())
    else:
        minimum = None
    checks['borrowed_id4_voxels'] = int(borrowed.sum())
    checks['released_voxels_assigned_id4'] = int((changed & (after == 4)).sum())
    checks['borrowed_id4_boundary_clearance_lower_bound_mm'] = minimum
    checks['borrowed_id4_within_3mm_boundary_margin'] = bool(minimum is not None and minimum <= 3.)
    source, count = np.unique(before[changed], return_counts=True)
    destination, counts = np.unique(after[changed], return_counts=True)
    records = {row['original_id']: row for row in catalog['records']}
    def names(values, amounts):
        return [{'id': int(key), 'name': records.get(int(key), {}).get('original_name'), 'voxels': int(value)}
                for key, value in zip(values, amounts)]
    return {'checks': checks, 'connectivity': direct,
            'changed_original_labels': names(source, count), 'changed_result_labels': names(destination, counts)}


def _slice(array, axis, index):
    return np.take(array, index, axis=axis)


def _panel(ax, image, before, after, roi, low, spacing, axis, fixed, center=None, half_width=None, projection=False):
    # Array slices are kji; x/y labels must preserve native global index axes.
    xy = ((0, 1), (0, 2), (1, 2))[axis]
    x, y = xy
    extent = [low[x]-.5, low[x]+image.shape[1]-.5, low[y]-.5, low[y]+image.shape[0]-.5]
    ax.imshow(image, origin='lower', cmap='gray', vmin=0, vmax=.28, extent=extent)
    added, removed = after & ~before, before & ~after
    overlay = np.zeros(image.shape + (4,))
    overlay[added] = (.15, 1., .25, .85)
    overlay[removed] = (1., .05, .55, .9)
    ax.imshow(overlay, origin='lower', extent=extent)
    coordinates = [np.arange(image.shape[1])+low[x], np.arange(image.shape[0])+low[y]]
    for mask, color, style in ((before, '#00d9ff', '--'), (after, '#ffe45e', '-'), (roi, 'white', ':')):
        if mask.any() and not mask.all():
            ax.contour(*coordinates, mask, levels=[.5], colors=[color], linewidths=.65, linestyles=style)
    ax.set_aspect(spacing[y] / spacing[x])
    ax.set_xlabel('ijk'[x]); ax.set_ylabel('ijk'[y])
    if center is not None:
        ax.set_xlim(center[x] - half_width / spacing[x], center[x] + half_width / spacing[x])
        ax.set_ylim(center[y] - half_width / spacing[y], center[y] + half_width / spacing[y])
    if projection:
        ax.set_title(('Coronal projection', 'Sagittal projection')[axis - 1] + '\nDepth is collapsed; use native checks for connectivity', fontsize=10)
    else:
        ax.set_title(('Axial', 'Coronal', 'Sagittal')[axis] + ' — ' + 'kji'[axis] + '=' + str(fixed), fontsize=10)


def draw_case(output, data, document, config, inspection):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-trial-review-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    before = np.isin(data['before_labels'], document['candidate_source_ids'])
    after = np.isin(data['after_labels'], document['candidate_source_ids'])
    added, removed = after & ~before, before & ~after
    low = np.asarray(document['geometry']['crop_origin_ijk'])
    spacing = np.asarray(document['geometry']['spacing_ijk_mm'])
    nodes = np.asarray(document['geometry']['roi_nodes_ijk'])
    figure, axes = plt.subplots(4, 3, figsize=(16, 18), constrained_layout=True)
    for axis, panel in ((1, axes[0, 0]), (2, axes[0, 1])):
        image = np.max(data['before_attenuation_per_pixel'] / (spacing[0]/10), axis=axis)
        _panel(panel, image, before.any(axis=axis), after.any(axis=axis), data['before_roi'].any(axis=axis),
               low, spacing, axis, None, projection=True)
    # A compact physical arc profile helps locate additions and losses; it is a
    # bin-volume diagnostic, not a clinical cross-sectional stenosis measure.
    profile = axes[0, 2]
    for mask, color, label in ((data['before_selected'], '#008db1', 'Before'),
                               (data['after_target'], '#d48500', 'After')):
        points = np.argwhere(mask)[:, ::-1] + low
        distance, total = parent_arc(points, nodes, spacing)
        bins = np.linspace(0, total, max(4, int(np.ceil(total / max(1., spacing.min()))))+1)
        volume = np.histogram(distance, bins)[0] * np.prod(spacing) / np.diff(bins)
        profile.plot(((bins[1:]+bins[:-1])/2)[1:-1], volume[1:-1], color=color, label=label)
    profile.set_title('Target volume per arc-length bin\nEnd-cap bins omitted; not clinical lumen area', fontsize=10)
    profile.set_xlabel('Distance from parent ROI start (mm)'); profile.set_ylabel('mm³ / mm'); profile.legend()
    maximum = next(row for row in document['samples'] if row['extreme'] in ('maximum', 'minimum_and_maximum'))
    summaries = {row['step_name']: row for row in maximum['step_outcomes']}
    centers = []
    for row_index, name in enumerate(config['recipe']['steps'], 1):
        edit = maximum['edits'][name]
        difference = added if edit['operation'] == 'dilation' else removed
        coordinates = np.argwhere(difference)
        arcs, total = parent_arc(coordinates[:, ::-1] + low, nodes, spacing)
        start, stop = edit.get('path_percent', (0., 100.))
        local = coordinates[(arcs >= total * start/100) & (arcs <= total * stop/100)]
        if not len(local):
            # Explicitly show the planned midpoint when the requested step had
            # no effect; the audit table remains the source of pass attribution.
            segments = np.r_[0., np.cumsum(np.linalg.norm(np.diff(nodes, axis=0)*spacing, axis=1))]
            midpoint = (start+stop)/200 * total
            point = np.asarray([np.interp(midpoint, segments, nodes[:, axis]) for axis in range(3)])
            local = np.rint(point-low)[None, ::-1].astype(int)
        requested = edit['distance_mm'] if edit['operation'] == 'dilation' else 0.
        half_width = max(config['roi']['radius_mm']) + requested + 4
        row_centers = []
        for axis in range(3):
            fixed = int(np.bincount(local[:, axis], minlength=before.shape[axis]).argmax())
            fixed = max(0, min(before.shape[axis]-1, fixed))
            on_plane = local[np.abs(local[:, axis]-fixed) <= 1]
            center = np.median(on_plane if len(on_plane) else local, axis=0)[::-1] + low
            row_centers.append({'plane': 'kji'[axis], 'index': int(fixed+low[2-axis]), 'center_ijk': center.tolist()})
            _panel(axes[row_index, axis], _slice(data['before_attenuation_per_pixel'], axis, fixed)/(spacing[0]/10),
                _slice(before, axis, fixed), _slice(after, axis, fixed), _slice(data['before_roi'], axis, fixed),
                low, spacing, axis, int(fixed+low[2-axis]), center, half_width)
        step = summaries[name]
        axes[row_index, 0].text(-.19, .5, name + '\n+' + str(step['added_voxels']) + ' / −' + str(step['removed_voxels']) +
            '\n' + str(step['unresolved_voxels']) + ' unresolved', transform=axes[row_index, 0].transAxes,
            rotation=90, va='center', ha='center', fontsize=10)
        centers.append({'step_name': name, 'planes': row_centers,
            'selection': 'Native plane with most net changes of this operation inside its parent arc interval; overlapping edits can affect the same view.'})
    labels = [Line2D([], [], color='#00b9e8', linestyle='--', label='Before outline'),
              Line2D([], [], color='#d9a900', label='After outline'),
              Line2D([], [], color='#25d53c', linewidth=5, label='Added target'),
              Line2D([], [], color='#f52d87', linewidth=5, label='Removed target'),
              Line2D([], [], color='gray', linestyle=':', label='Original ROI')]
    figure.legend(handles=labels, loc='lower center', bbox_to_anchor=(.5, -.012), ncol=5)
    figure.suptitle('Phantom ' + document['case_id'] + ' — maximum saved scenario\nNative before/after anatomy overlay; attenuation-copy proxy', fontsize=17)
    figure.savefig(output, dpi=135, bbox_inches='tight')
    plt.close(figure)
    return centers


def inspect_trial(folder, output):
    document = json.loads((folder/'trial.json').read_bytes())
    if document.get('schema_version') != 'fakect.recipe-trial/1' or not document.get('complete') or (folder/'INCOMPLETE').exists():
        raise ValueError('Trial must be complete')
    maximum = next(row for row in document['samples'] if row['extreme'] in ('maximum', 'minimum_and_maximum'))
    payload = folder/maximum['payload']
    if payload.parent.resolve() != folder.resolve() or _file_hash(payload) != maximum['payload_sha256']:
        raise ValueError('Maximum payload path or checksum mismatch')
    if _file_hash(folder/'input.ini') != document['input_config_sha256']:
        raise ValueError('Trial input snapshot checksum mismatch')
    config = load_study_config(folder/'input.ini', repo_root=ROOT)
    catalog_bytes = config['input']['catalog'].read_bytes()
    provenance = json.loads((folder/'data-config.json').read_bytes())
    if hashlib.sha256(catalog_bytes).hexdigest() != provenance['catalog_sha256']:
        raise ValueError('Source catalog changed since trial')
    catalog = json.loads(catalog_bytes)
    with np.load(payload, allow_pickle=False) as saved:
        data = {name: saved[name] for name in saved.files if name not in ('geometry_json', 'trial_metadata_json')}
    inspection = native_checks(data, document, catalog)
    samples = [summarize_sample(row) for row in document['samples']]
    row = {'case_id': document['case_id'], 'trial_directory': str(folder),
        'trial_sha256': _file_hash(folder/'trial.json'), 'payload_sha256': maximum['payload_sha256'],
        'input_config_sha256': document['input_config_sha256'], 'dataset_fingerprint': document['dataset_fingerprint'],
        'source_crop_sha256': document['source_crop_sha256'], 'samples': samples, **inspection,
        'montage': document['case_id']+'.png', 'full_parameter_space_checked': False, 'freeze_approved': False}
    row['local_planes'] = draw_case(output/row['montage'], data, document, config, inspection)
    return row


def render_html(document):
    escape = lambda value: html.escape(str(value))
    content = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Native cohort trial review</title>',
        '<style>body{font:16px system-ui;margin:2rem;background:#f2f5f7;color:#182635}section{background:white;padding:1rem;margin:1rem 0}img{width:100%}table{border-collapse:collapse;width:100%}td,th{border:1px solid #bcc6ce;padding:.4rem;text-align:left}pre{white-space:pre-wrap}</style>',
        '<h1>Native before/after cohort trial review</h1><p>Maximum corner overlays and independent native label checks. '
        'Minimum and maximum outcomes are tabulated. Only two corners were simulated; the full cohort must pass preflight separately. '
        'Projected views collapse depth; component correspondence uses native six-face connectivity. Partial unresolved erosion retains original tissue. '
        'These are attenuation-copy proxies, not physical CT reconstructions or clinically validated stenosis measurements.</p>']
    for row in document['cases']:
        content.append('<section><h2 id="case-'+escape(row['case_id'])+'">Phantom '+escape(row['case_id'])+'</h2>')
        if row.get('error'):
            content.append('<p>'+escape(row['error'])+'</p></section>'); continue
        content.append('<p>Source trial: '+escape(row['trial_directory'])+'</p>')
        content.append('<p>Original components: '+str(row['connectivity']['before_components'])+'; final: '+str(row['connectivity']['after_components'])+
            '; split/lost original components: '+str(row['connectivity']['split_or_lost_original_components'])+
            '; bone, other skin and excluded-artery changed voxels: '+
            ', '.join(str(row['checks'][key]) for key in ('bone_voxels_changed','other_skin_identity_voxels_changed','excluded_artery_voxels_changed'))+'.</p>')
        content.append('<table><tr><th>Corner</th><th>Edit</th><th>Requested → accepted mm</th><th>Added / removed</th><th>Unresolved</th><th>Guard</th><th>Review flags</th></tr>')
        for sample in row['samples']:
            for step in sample['steps']:
                values = [sample.get('extreme', sample['variant_id']), step['step_name'],
                    str(step['requested_distance_mm'])+' → '+str(step['accepted_distance_mm']),
                    str(step['added_voxels'])+' / '+str(step['removed_voxels']), step['unresolved_voxels'],
                    (step['erosion_safeguard'] or {}).get('status','off') + '; retained=' +
                    str((step['erosion_safeguard'] or {}).get('retained_volume_ratio','n/a')),
                    '; '.join(x['reason'] for x in step['issues']) or 'none']
                content.append('<tr>'+''.join('<td>'+escape(value)+'</td>' for value in values)+'</tr>')
        content.append('</table><a href="'+escape(row['montage'])+'"><img loading="lazy" src="'+escape(row['montage'])+'" alt="Native before and after target overlays"></a>')
        content.append('<details><summary>Native checks and changed labels</summary><pre>'+escape(json.dumps({key:row[key] for key in ('checks','connectivity','changed_original_labels','changed_result_labels')},indent=2))+'</pre></details></section>')
    return ''.join(content)+'</html>'


def run(project, trial_root, output, trial_name='trial-r2', overrides=None):
    output=Path(output).resolve(); output.mkdir(parents=True,exist_ok=False)
    entries=json.loads(Path(project).read_bytes())['cases']
    document={'schema_version':'fakect.cohort-trial-visual-review/1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
              'project':str(Path(project).resolve()),'cases':[],'freeze_approved':False,
              'complete':False,'expected_case_count':len(entries),
              'code_sha256':{str(path.relative_to(ROOT)):_file_hash(path) for path in (
                  Path(__file__), Path(__file__).with_name('review_internal_fill.py'),
                  Path(__file__).with_name('review_cohort_outcomes.py'), ROOT/'src/fakect_reassignment.py')},
              'clearance_semantics':'Conservative separation between entire borrowed ID4 voxel cells and background cells, subtracting both half voxel diagonals from center distance; crop boundary is treated as background.'}
    for entry in entries:
        case=str(entry['case_id']); folder=Path(trial_root)/case/(overrides or {}).get(case,trial_name)
        try:
            row=inspect_trial(folder.resolve(),output)
            for source,target in zip(json.loads((folder/'trial.json').read_bytes())['samples'],row['samples']):
                target['extreme']=source['extreme']
        except (OSError,ValueError,KeyError,TypeError) as error:
            row={'case_id':case,'error':str(error),'trial_directory':str(folder)}
        document['cases'].append(row)
        (output/'review.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
        (output/'report.html').write_text(render_html(document))
        print(case, row.get('error','rendered'),flush=True)
    document.update(complete=True,failed_case_count=sum('error' in row for row in document['cases']))
    (output/'review.json').write_text(json.dumps(document,indent=2,allow_nan=False)+'\n')
    (output/'report.html').write_text(render_html(document))
    return document


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project',type=Path,required=True)
    parser.add_argument('--trial-root',type=Path,required=True)
    parser.add_argument('--output',type=Path,required=True)
    parser.add_argument('--trial-name',default='trial-r2')
    parser.add_argument('--case-trial',action='append',default=[],help='Case-specific revision, e.g. 260614=trial-r3')
    args=parser.parse_args()
    try:
        overrides=dict(value.split('=',1) for value in args.case_trial)
        result=run(args.project,args.trial_root,args.output,args.trial_name,overrides)
    except (OSError,ValueError,KeyError,TypeError) as error:
        parser.exit(2,'error: '+str(error)+'\n')
    if any(row.get('error') for row in result['cases']):
        parser.exit(2,'error: some trial reviews failed; inspect review.json\n')


if __name__=='__main__':
    main()
