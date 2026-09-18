#!/usr/bin/env python3
"""Edit one commented INI file, then generate a complete standalone ROI report."""
import argparse
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_config import load_preview_config
from fakect_roi import COLORS, digest, prepare_crop, resolve_preview, selection_diagnostics


def json_value(value):
    if isinstance(value, Path):
        return str(value)
    if isinstance(value, dict):
        return {key: json_value(val) for key, val in value.items()}
    if isinstance(value, (tuple, list)):
        return [json_value(v) for v in value]
    return value


def render_slices(arrays, resolved, config, output):
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, to_rgba
    from matplotlib.patches import Patch
    from matplotlib.lines import Line2D

    low, high = resolved['crop_low_ijk'], resolved['crop_high_ijk_exclusive']
    spacing = resolved['spacing_ijk_mm']
    center = resolved['focus_ijk']
    nodes = np.asarray(resolved['roi_nodes_ijk'], dtype=float)
    radii = np.asarray(resolved['roi_radii_mm'], dtype=float)
    roi_kind = resolved['roi_kind']
    catalog = resolved['catalog']
    lut = np.tile(to_rgba(COLORS['unknown']), (256, 1))
    for cat in catalog['categories']:
        lut[cat['id']] = to_rgba(COLORS.get(cat['name'], '#999999'))
    tissue_cmap = ListedColormap(lut)
    # Native axis numbers i=0,j=1,k=2; array axis numbers are reversed.
    specs = [(2, 0, 1, 'Axial'), (1, 0, 2, 'Coronal'), (0, 1, 2, 'Sagittal')]
    window = [float(arrays['attenuation_cm_inverse'].min()), float(arrays['attenuation_cm_inverse'].max())]
    if window[1] == window[0]:
        window[1] = window[0] + .001
    selections = []

    def panel(ax, fixed_axis, xaxis, yaxis, fixed_value, row, name):
        n = fixed_value - low[fixed_axis]
        raw = arrays['attenuation_cm_inverse'] if row == 0 else arrays['tissue']
        image = np.take(raw, n, axis=2 - fixed_axis)
        selected = np.take(arrays['selected'], n, axis=2 - fixed_axis)
        candidates = np.take(arrays['candidates'], n, axis=2 - fixed_axis)
        roi = np.take(arrays['roi'], n, axis=2 - fixed_axis)
        extent = (low[xaxis] - .5, high[xaxis] - .5, low[yaxis] - .5, high[yaxis] - .5)
        im = ax.imshow(image, origin='lower', extent=extent,
                       aspect=spacing[yaxis] / spacing[xaxis], interpolation='nearest',
                       cmap='gray' if row == 0 else tissue_cmap,
                       vmin=window[0] if row == 0 else 0, vmax=window[1] if row == 0 else 255)
        if candidates.any() and not candidates.all():
            ax.contour(np.arange(low[xaxis], high[xaxis]), np.arange(low[yaxis], high[yaxis]),
                       candidates.astype(float), levels=[.5], colors=['#7ad8e0'], linewidths=.6,
                       linestyles='dotted', alpha=.7)
        if selected.any() and not selected.all():
            ax.contour(np.arange(low[xaxis], high[xaxis]), np.arange(low[yaxis], high[yaxis]),
                       selected.astype(float), levels=[.5], colors=['#00ffff'], linewidths=1.2)
        if roi.any():
            overlay = np.zeros((*roi.shape, 4), dtype=float)
            overlay[..., :3] = to_rgba('#ff9800')[:3]
            overlay[..., 3] = roi * config['preview']['overlay_opacity']
            ax.imshow(overlay, origin='lower', extent=extent, aspect=spacing[yaxis] / spacing[xaxis], interpolation='nearest')
            if not roi.all():
                ax.contour(np.arange(low[xaxis], high[xaxis]), np.arange(low[yaxis], high[yaxis]),
                           roi.astype(float), levels=[.5], colors=['#ffb300'], linewidths=1.1)
        ax.axvline(center[xaxis], color='#ffb300', lw=.7, ls='--', alpha=.9)
        ax.axhline(center[yaxis], color='#ffb300', lw=.7, ls='--', alpha=.9)
        axis_names = 'ijk'
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel=f'{axis_names[xaxis]} (native voxel index)',
               ylabel=f'{axis_names[yaxis]} (native voxel index)',
               title=f'{name}: {axis_names[fixed_axis]}={fixed_value} ({fixed_value * spacing[fixed_axis]:g} relative mm)\n'
                     f'Tissue candidates: {int(candidates.sum())}; selected in ROI: {int(selected.sum())}')
        ax.secondary_xaxis('top', functions=(lambda v, s=spacing[xaxis]: v * s,
                                             lambda v, s=spacing[xaxis]: v / s)).set_xlabel('relative mm')
        if row == 0:
            selections.append({'view': name, 'fixed_axis': axis_names[fixed_axis], 'index': fixed_value,
                               'selected_voxels': int(selected.sum()), 'selected_in_roi_voxels': int((selected & roi).sum()),
                               'candidate_voxels': int(candidates.sum()), 'roi_intersects_plane': bool(roi.any())})
        return im

    legend = [Patch(facecolor='#ff9800', alpha=.3, edgecolor='#ffb300', label=f'{roi_kind} ROI voxel mask'),
              Line2D([0], [0], color='#00ffff', label='selected tissue inside ROI'),
              Line2D([0], [0], color='#7ad8e0', ls=':', label='tissue candidates in crop')]
    legend += [Patch(facecolor=COLORS.get(c['name'], '#999999'), edgecolor='#777777', label=c['name'].replace('_', ' ')) for c in catalog['categories']]
    ids = ','.join(map(str, config['selection']['source_ids']))
    selector = f'original IDs {ids if len(ids) < 55 else ids[:52] + "..."}' if ids else 'all source IDs in group'
    target_title = f"{config['selection']['tissue']}; {selector}; bounded by ROI"
    roi_description = (f'Sphere center i,j,k={tuple(nodes[0])}; radius={radii[0]:g} mm' if roi_kind == 'sphere'
                       else f'Tube: {len(nodes)} ordered center points; radii {radii.min():g}–{radii.max():g} mm; focus i,j,k={center}')
    source_hash = catalog['sources']['atlas']['sha256'][:12] if 'atlas' in catalog['sources'] else 'see provenance'
    fig, axes = plt.subplots(2, 3, figsize=(17, 12))
    fig.subplots_adjust(left=.07, right=.94, bottom=.16, top=.80, hspace=.55, wspace=.32)
    for col, (axis, x, y, name) in enumerate(specs):
        for row in range(2):
            im = panel(axes[row, col], axis, x, y, resolved['slice_ijk'][axis], row, name)
            if row == 0:
                fig.colorbar(im, ax=axes[row, col], shrink=.55, pad=.025, label='1/cm')
    fig.suptitle(f"{config['study']['name']} | XCAT {config['input']['case_id']}, frame {config['input']['frame']} | {target_title}\n"
                 f"{roi_description}. Native crop, no resampling.\n"
                 f"Top: attenuation + ROI. Bottom: proposed tissue groups + ROI.\n"
                 f"FakeCT policy {catalog['policy_version']} | DPI source atlas SHA256 {source_hash}", y=.98, fontsize=14)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .045), ncol=5, frameon=False, fontsize=9)
    fig.text(.5, .015, 'Edit roi.center_ijk and radius_mm to relocate or reshape the ROI; slice_ijk controls the displayed planes.\n'
             'Coordinates are native indices / relative mm; anatomical orientation remains unverified. Magenta labels require review.', ha='center', fontsize=10)
    fig.savefig(output / 'roi-closeups.png', dpi=160)
    plt.close(fig)

    # Sample five axial planes over the physical ROI envelope, including caps.
    first_k = np.min(nodes[:, 2] - radii / spacing[2])
    last_k = np.max(nodes[:, 2] + radii / spacing[2])
    levels = sorted(set(int(np.clip(round(v), low[2], high[2] - 1)) for v in np.linspace(first_k, last_k, 5)))
    fig, axes = plt.subplots(2, len(levels), figsize=(4.4 * len(levels), 10), squeeze=False)
    fig.subplots_adjust(left=.05, right=.98, bottom=.12, top=.80, wspace=.4, hspace=.5)
    for col, k in enumerate(levels):
        for row in range(2):
            panel(axes[row, col], 2, 0, 1, k, row, 'Axial')
    fig.suptitle(f"ROI z-level close-ups | {target_title}\nOrange: native {roi_kind} ROI mask; cyan: final tissue selection\n"
                 f"{roi_description}. Top attenuation, bottom proposed groups.", y=.98, fontsize=14)
    fig.text(.5, .045, 'ROI sections are evaluated on the native grid using physical distances; these views apply no geometry edit.\n'
             'Adjust ordered tube points/radii or sphere settings in the INI and compare a new report.', ha='center', fontsize=11)
    fig.savefig(output / 'roi-z-stack.png', dpi=150)
    plt.close(fig)
    return {'planes': selections, 'z_stack_k': levels, 'attenuation_window_cm_inverse': window}


def run(config_path, validate_only=False, *, config_override=None, training_plan=None, extra_code_files=(), expected_input_bytes=None):
    config_path = Path(config_path).expanduser().resolve()
    config_bytes = Path(config_path).read_bytes()
    if expected_input_bytes is not None and config_bytes != expected_input_bytes:
        raise ValueError('Input changed before rendering; rerun from the saved file')
    config = load_preview_config(config_path) if config_override is None else config_override
    if Path(config_path).read_bytes() != config_bytes:
        raise ValueError('INI changed while loading; rerun with the saved input')
    code_files = [Path(__file__), ROOT/'src/fakect_roi.py', ROOT/'src/fakect_config.py',
                  ROOT/'src/fakect_volume_preview.py', ROOT/'src/fakect_tissues.py', ROOT/'src/fakect_preview_report.py',
                  ROOT/'src/fakect_morphology.py', ROOT/'src/fakect_reassignment.py',
                  ROOT/'src/fakect_edit_preview.py', ROOT/'src/fakect_arc_profile.py', ROOT/'src/fakect_global_preview.py',
                  ROOT/'src/fakect_surface_overlay.py']
    code_files += [ROOT/'src/fakect_direction.py', ROOT/'src/fakect_centerline_frame.py']
    code_files += [Path(p) for p in extra_code_files]
    recipe_requested = 'recipe' in config
    if recipe_requested:
        code_files += [ROOT/'src/fakect_recipe.py', ROOT/'src/fakect_recipe_config.py',
                       ROOT/'src/fakect_recipe_preview.py', ROOT/'src/fakect_tube_range.py',
                       ROOT/'src/fakect_frame_preview.py', ROOT/'src/fakect_growth.py']
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
    resolved = resolve_preview(config)
    edit_requested = config.get('edit', {}).get('operation', 'none') != 'none'
    recipe_plan = None
    if recipe_requested:
        from fakect_recipe import validate_recipe
        recipe_plan = validate_recipe(config, resolved)
    if 'edit' in config:
        from fakect_morphology import validate_edit_geometry
        validate_edit_geometry(resolved, config)
    output = config['output']['directory']
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Output exists and is not empty: {output}; choose a new output.directory')
    if validate_only:
        print(json.dumps(json_value({'status': 'metadata_valid; source voxel presence not checked',
                                     'crop_low_ijk': resolved['crop_low_ijk'],
                                     'crop_high_ijk_exclusive': resolved['crop_high_ijk_exclusive'],
                                     'candidate_id_count': len(resolved['source_ids']),
                                     'roi_shape': resolved['roi_kind'], 'roi_nodes_ijk': resolved['roi_nodes_ijk'],
                                     'roi_radii_mm': resolved['roi_radii_mm'], 'slice_ijk': resolved['slice_ijk'],
                                     'operation': config.get('edit', {}).get('operation', 'none'),
                                     'recipe': recipe_plan,
                                     'output': output}), indent=2))
        return
    arrays, sources = prepare_crop(resolved, config)
    selection = selection_diagnostics(arrays, resolved)
    edit_result = None
    recipe_steps = []
    recipe_figures = None
    frame_figures = []
    if edit_requested:
        from fakect_morphology import apply_morphology
        edit_result = apply_morphology(arrays, resolved, config)
    output.mkdir(parents=True, exist_ok=True)
    if recipe_requested:
        (output / 'INCOMPLETE').write_text('Recipe preview is incomplete; no final report has been published.\n')
        from fakect_recipe import apply_recipe, recipe_masks
        from fakect_recipe_preview import render_recipe_rois, render_recipe_step
        masks = recipe_masks(arrays, resolved, config)
        recipe_figures = render_recipe_rois(arrays, resolved, config, masks, output)

        def capture_step(event):
            relative = f"steps/{event['index']:02d}-{event['step_name']}-iteration-{event['iteration']:02d}"
            event['artifact_directory'] = relative
            figures = render_recipe_step(event, output)
            before, result = event['before_arrays'], event['result']
            record = {**event['pass_summary'],
                      **{key: event[key] for key in ('index', 'step_name', 'roi_name', 'iteration')}}
            record.update(summary=result['summary'], figures=figures,
                          array_artifact=relative+'/step.npz')
            np.savez_compressed(output / record['array_artifact'],
                                **{key: value for key, value in result.items() if isinstance(value, np.ndarray)},
                                before_labels=before['act'], before_attenuation_per_pixel=before['atn'],
                                before_tissue_labels=before['tissue'], roi_mask=before['roi'],
                                step_json=np.array(json.dumps(json_value(record), allow_nan=False)),
                                input_config_sha256=np.array(hashlib.sha256(config_bytes).hexdigest()))
            recipe_steps.append(record)
            counts = result['summary']['counts']
            print(f"Recipe pass {event['index']}: {event['step_name']} / {event['roi_name']} / "
                  f"iteration {event['iteration']}; added={counts['added']}, removed={counts['removed']}", flush=True)

        edit_result = apply_recipe(arrays, resolved, config, on_step=capture_step)
        from fakect_direction import recipe_centerline_frames
        frames = recipe_centerline_frames(config, resolved)
        if frames:
            from fakect_frame_preview import render_centerline_frames
            frame_figures = render_centerline_frames(frames, arrays, resolved, config, output)
    print(f"Read native crop {arrays['act'].shape}; selected voxels={int(arrays['selected'].sum())}", flush=True)
    plot_stats = render_slices(arrays, resolved, config, output)
    from fakect_global_preview import render_global_preview
    global_stats = render_global_preview(resolved, config, output)
    # Whole-phantom context and native local views must come from one source state.
    for row in sources.values():
        stat = Path(row['path']).stat()
        if stat.st_size != row['bytes'] or stat.st_mtime_ns != row['mtime_ns']:
            raise ValueError('Source changed between local and global rendering; rerun to a new directory')
    if training_plan is not None:
        from fakect_study_preview import render_training_target
        target_stats = render_training_target(arrays, resolved, config, output)
    from fakect_volume_preview import render_volume_preview
    volume_stats = render_volume_preview(arrays['act'], arrays['tissue'], resolved['catalog'], arrays['selected'],
        crop_origin_ijk=resolved['crop_low_ijk'], spacing_ijk_mm=resolved['spacing_ijk_mm'],
        roi_center_ijk=resolved['roi_nodes_ijk'][0], roi_radius_mm=resolved['roi_radii_mm'][0],
        roi_shape=resolved['roi_kind'], roi_nodes_ijk=resolved['roi_nodes_ijk'],
        roi_radii_mm=resolved['roi_radii_mm'], roi_mask=arrays['roi'],
        context_tissues=config['preview']['context_tissues'], volume_stride=config['preview']['volume_stride'],
        volume_opacity=config['preview']['volume_opacity'], context_opacity=config['preview']['context_opacity'], output_dir=output)
    if edit_result is not None:
        from fakect_edit_preview import render_edit_comparison
        from fakect_surface_overlay import render_surface_overlay
        surface_overlay = render_surface_overlay(
            arrays['candidates'], np.isin(edit_result['edited_labels'], resolved['source_ids']),
            crop_origin_ijk=resolved['crop_low_ijk'], spacing_ijk_mm=resolved['spacing_ijk_mm'],
            volume_stride=config['preview']['volume_stride'], output_dir=output)
        figure_config = config if not recipe_requested else {**config, 'edit': {'operation': 'recipe'}}
        edit_figures = render_edit_comparison(arrays, edit_result, resolved, figure_config, output)
        after_volume = render_volume_preview(edit_result['edited_labels'], edit_result['edited_tissue_labels'],
            resolved['catalog'], edit_result['target_mask_after'],
            crop_origin_ijk=resolved['crop_low_ijk'], spacing_ijk_mm=resolved['spacing_ijk_mm'],
            roi_center_ijk=resolved['roi_nodes_ijk'][0], roi_radius_mm=resolved['roi_radii_mm'][0],
            roi_shape=resolved['roi_kind'], roi_nodes_ijk=resolved['roi_nodes_ijk'],
            roi_radii_mm=resolved['roi_radii_mm'], roi_mask=arrays['roi'],
            context_tissues=config['preview']['context_tissues'], volume_stride=config['preview']['volume_stride'],
            volume_opacity=config['preview']['volume_opacity'], context_opacity=config['preview']['context_opacity'],
            output_dir=output / 'after')
    unknown = next(c['id'] for c in resolved['catalog']['categories'] if c['name'] == 'unknown')
    missing = sorted(set(map(int, np.unique(arrays['act']))) - {r['original_id'] for r in resolved['catalog']['records']})
    spacing = np.asarray(resolved['spacing_ijk_mm'])
    nodes_mm = np.asarray(resolved['roi_nodes_ijk']) * spacing
    radii = np.asarray(resolved['roi_radii_mm'])[:, None]
    roi_clipped = bool(np.any(np.min(nodes_mm-radii, axis=0) < -.5*spacing) or
                       np.any(np.max(nodes_mm+radii, axis=0) > (np.asarray(resolved['shape_kji'][::-1])-.5)*spacing))
    if any(digest(p) != code_hashes[str(p.relative_to(ROOT))] for p in code_files):
        raise ValueError('Preview source code changed during rendering; retain these partial outputs and rerun to a new directory')
    report = {'schema_version': 'fakect.roi-preview/5',
              'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'preview_only': True, 'geometry_edited': bool(edit_result is not None and edit_result['changed_mask'].any()),
              'source_volumes_modified': False, 'scalar_recovery_performed': False, 'config': json_value(config),
              'input_config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'catalog_sha256': resolved['catalog_sha256'],
              'policy_version': resolved['catalog']['policy_version'],
              'audit_sha256': resolved['audit_sha256'],
              'atlas_sources': resolved['catalog']['sources'], 'source_files': sources,
              'source_ids': list(resolved['source_ids']), 'source_names': resolved['source_names'],
              'source_ids_semantics': 'Candidate dictionary IDs; final selected mask is candidate tissue AND ROI',
              'selection': selection,
              'geometry': {'array_order': 'kji', 'source_shape_kji': resolved['shape_kji'],
                           'crop_origin_ijk': resolved['crop_low_ijk'], 'crop_high_ijk_exclusive': resolved['crop_high_ijk_exclusive'],
                           'crop_shape_kji': list(arrays['act'].shape), 'spacing_ijk_mm': resolved['spacing_ijk_mm'],
                           'roi_kind': resolved['roi_kind'], 'nodes_ijk': resolved['roi_nodes_ijk'],
                           'radii_mm': resolved['roi_radii_mm'], 'focus_ijk': resolved['focus_ijk'],
                           'roi_semantics': 'Sphere, or union of balls whose centers and radii interpolate linearly along each ordered segment; round caps',
                           'orientation': 'native array interpretation; anatomical orientation and physical origin unverified'},
              'roi_clipped_by_source_boundary': roi_clipped, 'crop_voxels': int(arrays['act'].size),
              'candidate_voxels': int(arrays['candidates'].sum()),
              'selected_voxels': int(arrays['selected'].sum()), 'selected_in_roi_voxels': int((arrays['selected'] & arrays['roi']).sum()),
              'roi_voxels': int(arrays['roi'].sum()), 'unknown_group_voxels': int((arrays['tissue'] == unknown).sum()),
              'unknown_group_voxels_in_roi': int(((arrays['tissue'] == unknown) & arrays['roi']).sum()),
              'missing_dictionary_ids': missing,
              'groups': {c['name']: int((arrays['tissue'] == c['id']).sum()) for c in resolved['catalog']['categories']},
              'slices': plot_stats, 'volume': volume_stats, 'global_view': global_stats,
              'html_report': {'path': 'report.html', 'self_contained': True},
              'code_sha256': code_hashes}
    if recipe_requested and config['recipe'].get('roi_role') == 'selection':
        report['source_ids_semantics'] = 'Candidate dictionary IDs; original selection is candidate tissue AND ROI. Final edited selection includes surviving original ancestors and their offspring outside the ROI.'
    if training_plan is not None:
        report['training_plan'] = {**training_plan, 'target_preview': target_stats}
        report['rerun_command'] = 'python3 scripts/train_study.py --config /path/to/study.ini --stage preview'
    if edit_result is not None:
        if recipe_requested:
            report['recipe'] = {**edit_result['summary'], 'figures': recipe_figures,
                                'engine_steps': edit_result['summary'].get('steps', []), 'steps': recipe_steps,
                                'final_figures': edit_figures, 'after_volume': after_volume,
                                'surface_overlay': surface_overlay,
                                'array_artifact': 'edit.npz', 'plan': recipe_plan}
            if frame_figures:
                report['recipe']['centerline_frames'] = frame_figures
        else:
            report['edit'] = {**edit_result['summary'], 'figures': edit_figures, 'after_volume': after_volume,
                              'surface_overlay': surface_overlay,
                              'scope': 'Derived native crop only; source volumes preserved', 'array_artifact': 'edit.npz'}
        metadata_key = 'recipe' if recipe_requested else 'edit'
        np.savez_compressed(output / 'edit.npz',
            **{key: value for key, value in edit_result.items() if isinstance(value, np.ndarray)},
            original_labels=arrays['act'], original_tissue_labels=arrays['tissue'],
            original_attenuation_per_pixel=arrays['atn'], roi_mask=arrays['roi'],
            geometry_json=np.array(json.dumps(json_value(report['geometry']))),
            edit_json=np.array(json.dumps(json_value(report[metadata_key]), allow_nan=False)),
            catalog_json=np.array(resolved['catalog_bytes'].decode('utf-8')))
    if not arrays['candidates'].any():
        report['selection_warning'] = 'Tissue candidates are absent from this crop; relocate the ROI or change tissue selection.'
    elif not arrays['selected'].any():
        report['selection_warning'] = 'Tissue candidates are present in the crop but outside the ROI; relocate or resize the ROI.'
    np.savez_compressed(output / 'crop.npz', original_labels=arrays['act'], tissue_labels=arrays['tissue'],
                        attenuation_per_pixel=arrays['atn'], candidate_mask=arrays['candidates'],
                        selected_mask=arrays['selected'], roi_mask=arrays['roi'],
                        geometry_json=np.array(json.dumps(json_value(report['geometry']))), catalog_sha256=np.array(report['catalog_sha256']),
                        catalog_json=np.array(resolved['catalog_bytes'].decode('utf-8')))
    (output / 'input.ini').write_bytes(config_bytes)
    (output / 'resolved-config.json').write_text(json.dumps(json_value(config), indent=2) + '\n')
    report['artifacts_sha256'] = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file() and p.name != 'INCOMPLETE'}
    (output / 'preview-report.json').write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + '\n')
    from fakect_preview_report import write_preview_report
    html = write_preview_report(output, json_value(report), config_bytes.decode('utf-8-sig'))
    if recipe_requested:
        (output / 'INCOMPLETE').unlink()
    manifest = {str(p.relative_to(output)): digest(p) for p in output.rglob('*') if p.is_file()}
    (output / 'artifact-manifest.json').write_text(json.dumps(manifest, indent=2) + '\n')
    print(json.dumps({'report': html['path'], 'output': str(output), 'selected_voxels': report['selected_voxels'],
                      'candidate_voxels': report['candidate_voxels'], 'connected_components': selection['component_count_6'],
                      'unknown_group_voxels': report['unknown_group_voxels'],
                      'edit': report.get('edit', {}).get('counts'),
                      'recipe': report.get('recipe', {}).get('counts'),
                      'warning': report.get('selection_warning')}, indent=2), flush=True)
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--validate-only', action='store_true', help='Check configuration, metadata and file sizes without reading voxel payloads')
    args = parser.parse_args()
    try:
        run(args.config, args.validate_only)
    except (ValueError, OSError, KeyError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
