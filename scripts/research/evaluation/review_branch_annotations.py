#!/usr/bin/env python3
"""Review the corrected anatomical reference, without evaluating any model.

Reads ten frozen baseline annotations and the unchanged r1 baseline CT exports.
Only the three previously selected example cases receive CT/3D illustrations.
"""
from pathlib import Path
import argparse
import csv
from html import escape
import json
import os
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure, label

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_multiscale_experiment import checked_json, sha256
from fakect_anatomy import _role
from fakect_two_stage_3d import build_scene, render_viewer

STRUCTURE = generate_binary_structure(3, 1)
CYAN, MAGENTA = '#00ffff', '#ff00ff'


def checked_file(path, expected, fingerprints):
    path = Path(path).resolve()
    digest = sha256(path)
    if digest != expected:
        raise ValueError('Frozen annotation dependency changed: ' + str(path))
    fingerprints[str(path)] = digest
    return path


def binary(value):
    value = np.asarray(value)
    if value.ndim != 3 or np.any((value != 0) & (value != 1)):
        raise ValueError('Anatomical annotation masks must be three-dimensional and binary')
    return value.astype(bool)


def place(source, source_origin_ijk, origin_ijk, shape):
    result = np.zeros(tuple(shape), bool)
    src_origin, dst_origin = np.asarray(source_origin_ijk)[::-1], np.asarray(origin_ijk)[::-1]
    low = np.maximum(src_origin, dst_origin)
    high = np.minimum(src_origin + source.shape, dst_origin + shape)
    if np.all(high > low):
        dst = tuple(slice(int(a - o), int(b - o)) for a, b, o in zip(low, high, dst_origin))
        src = tuple(slice(int(a - o), int(b - o)) for a, b, o in zip(low, high, src_origin))
        result[dst] = source[src]
    return result


def bbox(mask, padding):
    low, high = [], []
    for axis in range(3):
        indices = np.flatnonzero(mask.any(axis=tuple(a for a in range(3) if a != axis)))
        low.append(max(0, int(indices[0]) - int(padding[axis])))
        high.append(min(mask.shape[axis], int(indices[-1]) + 1 + int(padding[axis])))
    return tuple(slice(a, b) for a, b in zip(low, high))


def components(mask):
    return int(label(mask, STRUCTURE)[1])


def render_slices(image, masks, origin_ijk, spacing, case, destination):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-branch-annotation-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'svg.fonttype': 'none', 'pdf.fonttype': 42})
    core, branch, old = masks['core'], masks['branch'], masks['old_proximal_branch']
    contacts = branch & binary_dilation(core, STRUCTURE)
    if not contacts.any():
        raise ValueError('The selected branch components have no six-connected aortic ostia')
    positions = [int(np.argmax((contacts if axis != 1 else branch).sum(axis=tuple(a for a in range(3) if a != axis))))
                 for axis in range(3)]
    box = bbox(core | branch, np.ceil(8 / spacing[::-1]).astype(int))
    fig, axes = plt.subplots(1, 3, figsize=(15, 6.8))
    metadata = []
    for axis, (ax, name) in enumerate(zip(axes, ('Axial ostia', 'Coronal branch overview', 'Sagittal ostia'))):
        fixed = positions[axis]; section = list(box); section[axis] = fixed
        plane = np.asarray(image.dataobj[tuple(section[::-1])], np.float32).T
        vertical, horizontal = [a for a in range(3) if a != axis]
        x = np.arange(box[horizontal].start, box[horizontal].stop) + origin_ijk[2 - horizontal]
        y = np.arange(box[vertical].start, box[vertical].stop) + origin_ijk[2 - vertical]
        extent = [x[0] - .5, x[-1] + .5, y[0] - .5, y[-1] + .5]
        ax.imshow(plane, cmap='gray', vmin=-200, vmax=500, origin='lower', interpolation='nearest', extent=extent,
                  aspect=spacing[2 - vertical] / spacing[2 - horizontal])
        for source, color, width, style in ((core, CYAN, 2.2, 'solid'), (branch, MAGENTA, 2.2, 'solid'),
                                             (old, '#ffffff', 1.1, 'dashed')):
            local = source[tuple(section)]
            if local.any() and not local.all():
                ax.contour(x, y, local.astype(float), levels=[.5], colors=[color], linewidths=width, linestyles=style)
        native = fixed + origin_ijk[2 - axis]
        centre = np.array([(s.start + s.stop - 1) / 2 for s in box][::-1]); centre[2 - axis] = fixed
        world = (image.affine @ np.r_[centre, 1.])[:3]
        ax.set_title('{} · {}={}\n{}={:.1f} mm RAS'.format(name, 'kji'[axis], native, 'xyz'[2 - axis], world[2 - axis]), fontsize=11)
        ax.set_xlabel('ijk'[2 - horizontal] + ' · source voxel index')
        ax.set_ylabel('ijk'[2 - vertical] + ' · source voxel index')
        metadata.append(dict(view=name, source_axis='kji'[axis], source_index=int(native),
                             export_index=fixed, slice_centre_ras_mm=world.tolist()))
    fig.suptitle('Phantom {} · unmodified baseline CT · anatomical target audit\nCyan: original aortic core · magenta: restored branches · white dashed: former proximal reach'.format(case), fontsize=12)
    fig.tight_layout()
    for extension in ('png', 'pdf', 'svg'):
        fig.savefig(destination / ('baseline-' + case + '.' + extension), dpi=220, bbox_inches='tight')
    plt.close(fig)
    return metadata


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--study', default='outputs/experiments/aorta/pretrained-finetune-branches-r3')
    parser.add_argument('--output', default='annotation-review-r1')
    args = parser.parse_args()
    study = Path(args.study).resolve()
    lock_path = study / 'study-lock.json'; lock = checked_json(lock_path)
    reference_lock_path = Path(lock['target_policy']['reference_study']); reference = checked_json(reference_lock_path)
    reference_study = reference_lock_path.parent
    if lock['grids'] != reference['grids'] or lock['folds'] != reference['folds'] or lock['rosters'] != reference['rosters']:
        raise ValueError('Anatomical correction must retain the exact original boxes, folds and sample rosters')
    if lock['example_sample_ids'] != reference['example_sample_ids']:
        raise ValueError('Illustration cases must remain frozen before model evaluation')
    example_cases = list(dict.fromkeys(s.split(':')[0] for s in lock['example_sample_ids']))
    destination = study / args.output
    if destination.exists(): raise FileExistsError('Do not overwrite an existing annotation review')
    temporary = Path(tempfile.mkdtemp(prefix='.' + args.output + '-', dir=study))
    fingerprints = {str(lock_path): sha256(lock_path), str(reference_lock_path): sha256(reference_lock_path)}
    rows, illustrations, scenes = [], [], []
    try:
        original_rows = {r['case_id']: r for r in lock['samples'] if r['baseline']}
        for case in lock['cases']:
            source = original_rows[case]
            path = study / 'annotations' / case / 'annotation.json'
            annotation = checked_json(path); fingerprints[str(path)] = sha256(path)
            if annotation.get('complete') is not True or annotation['branch_extent'] != 'common_box':
                raise ValueError('Complete annotations extending to the common box are required')
            payload = checked_file(path.parent / annotation['mask_npz'], annotation['mask_npz_sha256'], fingerprints)
            with np.load(payload, allow_pickle=False) as data:
                core, branch, target = (binary(data[key]) for key in ('core', 'branch', 'target'))
                origin = data['crop_origin_ijk'].copy(); spacing = data['spacing_ijk_mm'].copy()
            grid = lock['grids'][case]
            if (list(origin) != grid['native_origin_ijk'] or list(target.shape) != grid['native_shape_kji'] or
                    not np.allclose(spacing, grid['native_spacing_ijk_mm'], atol=1e-9, rtol=0)):
                raise ValueError('Annotation differs from the shared native input box')
            if np.any(core & branch) or not np.array_equal(target, core | branch):
                raise ValueError('Combined target must be the disjoint core and restored branch union')
            counts = {key + '_voxels': int(mask.sum()) for key, mask in (('core', core), ('branch', branch), ('target', target))}
            if any(counts[k] != annotation['counts'][k] for k in counts):
                raise ValueError('Annotation counts do not match its native payload')
            selection_path = checked_file(source['selection_manifest'], source['selection_manifest_sha256'], fingerprints)
            selection = json.loads(selection_path.read_text())
            previous_payload = checked_file(selection_path.parent / selection['mask_npz'], selection['mask_npz_sha256'], fingerprints)
            with np.load(previous_payload, allow_pickle=False) as data:
                old_core = place(binary(data['core_mask']), data['crop_origin_ijk'], origin, target.shape)
                old_branch = place(binary(data['branch_mask']), data['crop_origin_ijk'], origin, target.shape)
            if not np.array_equal(core, old_core) or np.any(old_core & ~target):
                raise ValueError('Original aortic core changed or was removed from the combined target')
            catalog_path = checked_file(selection['catalog_path'], selection['catalog_sha256'], fingerprints)
            catalog = json.loads(catalog_path.read_text())
            by_id = {int(r['original_id']): r for r in catalog['records']}
            roles = {i: _role(r, {'include_branches': True}) for i, r in by_id.items()}
            pulmonary_ids = {i for i, (_, reason) in roles.items() if 'Pulmonary circulation excluded' in reason}
            selected = set(annotation['selected_source_ids']); eligible = set(annotation['eligible_source_ids'])
            if selected - eligible or any(roles[i][0] not in ('anchor', 'branch') for i in selected) or selected & pulmonary_ids or eligible & pulmonary_ids:
                raise ValueError('Pulmonary or otherwise excluded atlas labels entered branch propagation')
            continuation = branch & ~old_branch
            dropped_old = int((old_branch & ~target).sum())
            combined_components = components(target)
            record = dict(case_id=case, core_voxels=counts['core_voxels'], old_proximal_branch_voxels=int(old_branch.sum()),
                restored_branch_voxels=counts['branch_voxels'], continuation_voxels=int(continuation.sum()),
                combined_target_voxels=counts['target_voxels'], old_proximal_branch_not_in_new_target=dropped_old,
                core_subset_passed=True, shared_box_unchanged=True, combined_components=combined_components,
                pulmonary_exclusion_passed=True, selected_source_ids=sorted(selected),
                selected_source_names=[by_id[i]['original_name'] for i in sorted(selected)],
                coronary_source_names=[by_id[i]['original_name'] for i in sorted(selected) if any(
                    word in by_id[i]['original_name'].lower() for word in ('dias_rca', 'dias_lad', 'dias_lcx'))],
                branch_terminal_policy=annotation['branch_terminal_policy'], metadata=annotation['counts'])
            rows.append(record)
            if case in example_cases:
                prepared_path = reference_study / 'cases' / case / 'prepared.json'
                prepared = checked_json(prepared_path); fingerprints[str(prepared_path)] = sha256(prepared_path)
                native = prepared['samples'][case + ':baseline']['native']
                image_path = checked_file(native['input_nifti'], native['input_nifti_sha256'], fingerprints)
                image = nib.load(image_path)
                # NIfTI-1 sform coefficients are float32; compare to that exact
                # header representation rather than the lock's float64 values.
                expected_header_affine = np.asarray(grid['native_affine_ras_ijk'], np.float32).astype(float)
                if image.shape != tuple(target.shape[::-1]) or not np.allclose(image.affine, expected_header_affine, atol=1e-6, rtol=0):
                    raise ValueError('Unmodified baseline CT does not match the new annotation frame')
                masks = dict(core=core, branch=branch, old_proximal_branch=old_branch)
                slice_meta = render_slices(image, masks, origin, spacing, case, temporary)
                illustrations.append(dict(case_id=case, source_baseline_ct=image_path.as_posix(),
                    source_origin_ijk=origin.tolist(), spacing_ijk_mm=spacing.tolist(), slices=slice_meta,
                    image='baseline-' + case + '.png', model_predictions_displayed=False))
                box = bbox(target, np.ceil(3 / spacing[::-1]).astype(int))
                display = {key: value[box] for key, value in masks.items()}
                if np.prod(display['core'].shape) <= 16_777_216 and min(display['core'].shape) >= 2:
                    scenes.append(build_scene(display, spacing, sample_id=case + ':baseline', modes=[dict(key='annotation',
                        label='Corrected reference anatomy', explanation='Reference annotation only; no model prediction or editing is displayed.',
                        layers=[dict(key='core', label='Original aortic core', color=CYAN, visible=True, opacity=.45),
                                dict(key='branch', label='Restored connected branches', color=MAGENTA, visible=True, opacity=.8),
                                dict(key='old_proximal_branch', label='Former proximal branch reach', color='#ffffff', visible=False, opacity=.45)])],
                        metadata=dict(case_id=case, is_control=True, low_export_kji=[s.start for s in box],
                                      source_origin_ijk=origin.tolist()), show_metrics=False))
            print('Audited anatomy', case, counts, flush=True)
        viewer = render_viewer(temporary / 'report-3d', scenes, title='Aorta and restored branches · reference audit',
            coordinate_description='Native millimeters relative to each display crop; global source indices and RAS slice coordinates are recorded in audit.json.',
            footer_note='Reference annotations on unmodified baseline CT. Branches extend to the unchanged CT box and T12–T1 slab; no three-radius endpoint cap is imposed. No AI predictions or accuracy scores are displayed.', show_metrics=False)
        viewer['viewer'] = str(destination / 'report-3d/index.html')
        payload = dict(schema_version='fakect.branch-annotation-review/1', status='complete', rows=rows, illustrations=illustrations,
            original_example_sample_ids=lock['example_sample_ids'], illustrated_baseline_cases=example_cases,
            anatomical_scope=annotation['target_semantics'], foreground_channels=lock['foreground_channels'],
            reference_only=True, original_box_folds_rosters_unchanged=True, viewer=viewer,
            warning='Atlas roles include coronary branches although the clinical Dataset293 foreground group has no dedicated coronary artery class. This audit reviews reference anatomy, not model performance.')
        (temporary / 'audit.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        fields = ['case_id', 'core_voxels', 'old_proximal_branch_voxels', 'restored_branch_voxels', 'continuation_voxels',
                  'combined_target_voxels', 'combined_components', 'old_proximal_branch_not_in_new_target', 'core_subset_passed',
                  'shared_box_unchanged', 'pulmonary_exclusion_passed']
        with (temporary / 'counts.csv').open('w', newline='') as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({k: row[k] for k in fields} for row in rows)
        table = ''.join('<tr><td>' + r['case_id'] + '</td>' + ''.join('<td>{:,}</td>'.format(r[k]) for k in
            ('core_voxels', 'old_proximal_branch_voxels', 'restored_branch_voxels', 'continuation_voxels')) + '</tr>' for r in rows)
        cards = ''.join('<h2>Phantom ' + r['case_id'] + ' · unmodified baseline</h2><p><a href="baseline-' + r['case_id'] + '.pdf">PDF</a> · <a href="baseline-' + r['case_id'] + '.svg">SVG</a></p><img src="' + r['image'] + '" alt="Baseline CT with aortic core and restored branch annotations">' for r in illustrations)
        document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Corrected anatomical target audit</title>'
            '<style>body{font:16px/1.5 system-ui;max-width:1450px;margin:auto;padding:24px;color:#183144}table{border-collapse:collapse;width:100%}th,td{padding:8px;border-bottom:1px solid #ddd;text-align:left}img{max-width:100%}.note{padding:16px;background:#eef4f6}iframe{width:100%;height:1000px;border:0}</style></head><body>'
            '<h1>Aorta and connected branches · corrected reference audit</h1><p class="note">These are anatomical reference annotations, not model predictions. Cyan shows the unchanged aortic core; magenta shows restored connected branches. White dashed contours show the previous proximal branch reach.</p>'
            '<p>All ten cases retain the exact original CT boxes, folds and sample rosters. The original core is unchanged and remains inside the combined reference. Branch components extend to the shared CT box and strict T12–T1 slab; the former three-radius endpoint limit is removed. Atlas-qualified pulmonary labels are excluded before connectivity propagation.</p>'
            '<table><thead><tr><th>Phantom</th><th>Original core voxels</th><th>Former proximal branches</th><th>Restored branches</th><th>Further continuations</th></tr></thead><tbody>' + table + '</tbody></table>'
            '<p>Counts use each phantom’s native grid. The three illustrated cases were fixed in the original study before model evaluation. All images here are their unmodified baseline CTs; no modified scan is substituted and no performance result is selected.</p>'
            + cards + '<h2>Interactive reference surfaces</h2><p>Rotate and zoom, hide/show core and branches independently, and choose 15%, 45% or 80% opacity. Visible cap ends at the box/slab boundary reflect the unchanged field of view.</p><a href="report-3d/index.html">Open 3D anatomy viewer</a><iframe src="report-3d/index.html" title="Original core and restored connected branches"></iframe>'
            '<p class="note">Atlas eligibility also includes connected coronary branches. Dataset293’s selected clinical foreground group has no dedicated coronary artery class; its unchanged pretrained output therefore does not have an exact one-to-one native coronary label. This report establishes the intended anatomical reference only.</p>'
            '<p><a href="counts.csv">All native counts CSV</a> · <a href="audit.json">Checks, source labels and coordinates JSON</a> · <a href="provenance.json">Input and output checksums</a></p></body></html>')
        (temporary / 'report.html').write_text(document)
        shutil.copyfile(Path(__file__), temporary / 'annotation-review-generator.py')
        outputs = {str(p.relative_to(temporary)): sha256(p) for p in temporary.rglob('*') if p.is_file()}
        (temporary / 'provenance.json').write_text(json.dumps(dict(schema_version='fakect.branch-annotation-review/1',
            input_sha256=fingerprints, output_sha256=outputs, generator_sha256=sha256(Path(__file__)),
            models_run=False, surface_metrics_recomputed=False), indent=2) + '\n')
        temporary.rename(destination)
        print(json.dumps(dict(report=str(destination / 'report.html'), cases=len(rows), illustrations=len(illustrations)), indent=2))
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == '__main__': main()
