#!/usr/bin/env python3
"""Edit one commented INI file, then render full-resolution ROI and 3D previews."""
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
from fakect_roi import COLORS, digest, prepare_crop, resolve_preview


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
    from matplotlib.patches import Ellipse, Patch
    from matplotlib.lines import Line2D

    low, high = resolved['crop_low_ijk'], resolved['crop_high_ijk_exclusive']
    spacing = resolved['spacing_ijk_mm']
    center = config['roi']['center_ijk']
    radius = config['roi']['radius_mm']
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
        roi = np.take(arrays['roi'], n, axis=2 - fixed_axis)
        extent = (low[xaxis] - .5, high[xaxis] - .5, low[yaxis] - .5, high[yaxis] - .5)
        im = ax.imshow(image, origin='lower', extent=extent,
                       aspect=spacing[yaxis] / spacing[xaxis], interpolation='nearest',
                       cmap='gray' if row == 0 else tissue_cmap,
                       vmin=window[0] if row == 0 else 0, vmax=window[1] if row == 0 else 255)
        if selected.any() and not selected.all():
            ax.contour(np.arange(low[xaxis], high[xaxis]), np.arange(low[yaxis], high[yaxis]),
                       selected.astype(float), levels=[.5], colors=['#00ffff'], linewidths=1.2)
        distance_mm = (fixed_value - center[fixed_axis]) * spacing[fixed_axis]
        section_squared = radius ** 2 - distance_mm ** 2
        if section_squared >= 0:
            section_radius = np.sqrt(section_squared)
            ellipse = dict(xy=(center[xaxis], center[yaxis]),
                           width=2 * section_radius / spacing[xaxis], height=2 * section_radius / spacing[yaxis])
            ax.add_patch(Ellipse(**ellipse, facecolor='#ff9800', edgecolor='none', alpha=config['preview']['overlay_opacity']))
            ax.add_patch(Ellipse(**ellipse, facecolor='none', edgecolor='#ffb300', linewidth=1.5))
        ax.axvline(center[xaxis], color='#ffb300', lw=.7, ls='--', alpha=.9)
        ax.axhline(center[yaxis], color='#ffb300', lw=.7, ls='--', alpha=.9)
        axis_names = 'ijk'
        ax.set(xlim=extent[:2], ylim=extent[2:], xlabel=f'{axis_names[xaxis]} (native voxel index)',
               ylabel=f'{axis_names[yaxis]} (native voxel index)',
               title=f'{name}: {axis_names[fixed_axis]}={fixed_value} ({fixed_value * spacing[fixed_axis]:g} relative mm)\n'
                     f'Target: {int(selected.sum())} voxels; inside ROI: {int((selected & roi).sum())}')
        ax.secondary_xaxis('top', functions=(lambda v, s=spacing[xaxis]: v * s,
                                             lambda v, s=spacing[xaxis]: v / s)).set_xlabel('relative mm')
        if row == 0:
            selections.append({'view': name, 'fixed_axis': axis_names[fixed_axis], 'index': fixed_value,
                               'selected_voxels': int(selected.sum()), 'selected_in_roi_voxels': int((selected & roi).sum()),
                               'roi_intersects_plane': bool(section_squared >= 0)})
        return im

    legend = [Patch(facecolor='#ff9800', alpha=.3, edgecolor='#ffb300', label='ROI sphere intersection'),
              Line2D([0], [0], color='#00ffff', label='selected original label boundary')]
    legend += [Patch(facecolor=COLORS.get(c['name'], '#999999'), edgecolor='#777777', label=c['name'].replace('_', ' ')) for c in catalog['categories']]
    ids = ','.join(map(str, resolved['source_ids']))
    target_title = f"{config['selection']['tissue']}; original IDs {ids if len(ids) < 70 else ids[:67] + '...'}"
    source_hash = catalog['sources']['atlas']['sha256'][:12] if 'atlas' in catalog['sources'] else 'see provenance'
    fig, axes = plt.subplots(2, 3, figsize=(17, 12))
    fig.subplots_adjust(left=.07, right=.94, bottom=.16, top=.80, hspace=.55, wspace=.32)
    for col, (axis, x, y, name) in enumerate(specs):
        for row in range(2):
            im = panel(axes[row, col], axis, x, y, resolved['slice_ijk'][axis], row, name)
            if row == 0:
                fig.colorbar(im, ax=axes[row, col], shrink=.55, pad=.025, label='1/cm')
    fig.suptitle(f"{config['study']['name']} | XCAT {config['input']['case_id']}, frame {config['input']['frame']} | {target_title}\n"
                 f"ROI center i,j,k={center}; radius={radius:g} mm. Native crop, no resampling.\n"
                 f"Top: attenuation + ROI. Bottom: proposed tissue groups + ROI.\n"
                 f"FakeCT policy {catalog['policy_version']} | DPI source atlas SHA256 {source_hash}", y=.98, fontsize=14)
    fig.legend(handles=legend, loc='lower center', bbox_to_anchor=(.5, .045), ncol=5, frameon=False, fontsize=9)
    fig.text(.5, .015, 'Move roi.center_ijk in the INI to relocate the sphere; change preview.slice_ijk to inspect other planes.\n'
             'Coordinates are native indices / relative mm; anatomical orientation remains unverified. Magenta labels require review.', ha='center', fontsize=10)
    fig.savefig(output / 'roi-closeups.png', dpi=160)
    plt.close(fig)

    # Five neighboring axial planes within the ROI diameter, rounded to native centers.
    offsets = np.rint(np.linspace(-radius, radius, 5) / spacing[2]).astype(int)
    levels = sorted(set(int(np.clip(center[2] + v, low[2], high[2] - 1)) for v in offsets))
    fig, axes = plt.subplots(2, len(levels), figsize=(4.4 * len(levels), 10), squeeze=False)
    fig.subplots_adjust(left=.05, right=.98, bottom=.12, top=.80, wspace=.4, hspace=.5)
    for col, k in enumerate(levels):
        for row in range(2):
            panel(axes[row, col], 2, 0, 1, k, row, 'Axial')
    fig.suptitle(f"ROI z-level close-ups | {target_title}\nOrange: true sphere cross-section; cyan: selected original IDs\n"
                 f"Full-resolution native crop; center i,j,k={center}, radius={radius:g} mm. Top attenuation, bottom proposed groups.", y=.98, fontsize=14)
    fig.text(.5, .045, 'The spherical overlay narrows away from its center; at the two tips its cross-section becomes a point.\n'
             'Move center_ijk or radius_mm in the INI and use a new output directory to compare versions.', ha='center', fontsize=11)
    fig.savefig(output / 'roi-z-stack.png', dpi=150)
    plt.close(fig)
    return {'planes': selections, 'z_stack_k': levels, 'attenuation_window_cm_inverse': window}


def run(config_path, validate_only=False):
    config_bytes = Path(config_path).read_bytes()
    config = load_preview_config(config_path)
    if Path(config_path).read_bytes() != config_bytes:
        raise ValueError('INI changed while loading; rerun with the saved input')
    code_files = [Path(__file__), ROOT/'src/fakect_roi.py', ROOT/'src/fakect_config.py',
                  ROOT/'src/fakect_volume_preview.py', ROOT/'src/fakect_tissues.py']
    code_hashes = {str(p.relative_to(ROOT)): digest(p) for p in code_files}
    resolved = resolve_preview(config)
    output = config['output']['directory']
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise ValueError(f'Output exists and is not empty: {output}; choose a new output.directory')
    if validate_only:
        print(json.dumps(json_value({'status': 'metadata_valid; source voxel presence not checked',
                                     'crop_low_ijk': resolved['crop_low_ijk'],
                                     'crop_high_ijk_exclusive': resolved['crop_high_ijk_exclusive'],
                                     'source_ids': resolved['source_ids'], 'slice_ijk': resolved['slice_ijk'],
                                     'output': output}), indent=2))
        return
    arrays, sources = prepare_crop(resolved, config)
    output.mkdir(parents=True, exist_ok=True)
    print(f"Read native crop {arrays['act'].shape}; selected voxels={int(arrays['selected'].sum())}", flush=True)
    plot_stats = render_slices(arrays, resolved, config, output)
    from fakect_volume_preview import render_volume_preview
    volume_stats = render_volume_preview(arrays['act'], arrays['tissue'], resolved['catalog'], arrays['selected'],
        crop_origin_ijk=resolved['crop_low_ijk'], spacing_ijk_mm=resolved['spacing_ijk_mm'],
        roi_center_ijk=config['roi']['center_ijk'], roi_radius_mm=config['roi']['radius_mm'],
        context_tissues=config['preview']['context_tissues'], volume_stride=config['preview']['volume_stride'],
        volume_opacity=config['preview']['volume_opacity'], context_opacity=config['preview']['context_opacity'], output_dir=output)
    unknown = next(c['id'] for c in resolved['catalog']['categories'] if c['name'] == 'unknown')
    missing = sorted(set(map(int, np.unique(arrays['act']))) - {r['original_id'] for r in resolved['catalog']['records']})
    roi_clipped = any(config['roi']['center_ijk'][a] * resolved['spacing_ijk_mm'][a] - config['roi']['radius_mm'] < -.5 * resolved['spacing_ijk_mm'][a]
                      or (config['roi']['center_ijk'][a] * resolved['spacing_ijk_mm'][a] + config['roi']['radius_mm'] >
                          (resolved['shape_kji'][2-a] - .5) * resolved['spacing_ijk_mm'][a]) for a in range(3))
    if any(digest(p) != code_hashes[str(p.relative_to(ROOT))] for p in code_files):
        raise ValueError('Preview source code changed during rendering; retain these partial outputs and rerun to a new directory')
    report = {'schema_version': 'fakect.roi-preview/1', 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
              'preview_only': True, 'geometry_edited': False, 'config': json_value(config),
              'input_config_sha256': hashlib.sha256(config_bytes).hexdigest(), 'catalog_sha256': resolved['catalog_sha256'],
              'audit_sha256': resolved['audit_sha256'],
              'atlas_sources': resolved['catalog']['sources'], 'source_files': sources,
              'source_ids': list(resolved['source_ids']), 'source_names': resolved['source_names'],
              'geometry': {'array_order': 'kji', 'source_shape_kji': resolved['shape_kji'],
                           'crop_origin_ijk': resolved['crop_low_ijk'], 'crop_high_ijk_exclusive': resolved['crop_high_ijk_exclusive'],
                           'crop_shape_kji': list(arrays['act'].shape), 'spacing_ijk_mm': resolved['spacing_ijk_mm'],
                           'roi_center_ijk': config['roi']['center_ijk'], 'roi_radius_mm': config['roi']['radius_mm'],
                           'orientation': 'native array interpretation; anatomical orientation and physical origin unverified'},
              'roi_clipped_by_source_boundary': roi_clipped, 'crop_voxels': int(arrays['act'].size),
              'selected_voxels': int(arrays['selected'].sum()), 'selected_in_roi_voxels': int((arrays['selected'] & arrays['roi']).sum()),
              'roi_voxels': int(arrays['roi'].sum()), 'unknown_group_voxels': int((arrays['tissue'] == unknown).sum()),
              'missing_dictionary_ids': missing,
              'groups': {c['name']: int((arrays['tissue'] == c['id']).sum()) for c in resolved['catalog']['categories']},
              'slices': plot_stats, 'volume': volume_stats,
              'code_sha256': code_hashes}
    if not arrays['selected'].any():
        report['selection_warning'] = 'Selected original IDs are absent from this crop; relocate the ROI. No geometry was edited.'
    elif not (arrays['selected'] & arrays['roi']).any():
        report['selection_warning'] = 'Selected IDs are present in the crop but outside the ROI sphere; relocate or resize the ROI.'
    np.savez_compressed(output / 'crop.npz', original_labels=arrays['act'], tissue_labels=arrays['tissue'],
                        attenuation_per_pixel=arrays['atn'], selected_mask=arrays['selected'], roi_mask=arrays['roi'],
                        geometry_json=np.array(json.dumps(json_value(report['geometry']))), catalog_sha256=np.array(report['catalog_sha256']),
                        catalog_json=np.array(resolved['catalog_bytes'].decode('utf-8')))
    (output / 'input.ini').write_bytes(config_bytes)
    (output / 'resolved-config.json').write_text(json.dumps(json_value(config), indent=2) + '\n')
    report['artifacts_sha256'] = {p.name: digest(p) for p in output.iterdir() if p.is_file()}
    (output / 'preview-report.json').write_text(json.dumps(json_value(report), indent=2, allow_nan=False) + '\n')
    print(json.dumps({'output': str(output), 'selected_voxels': report['selected_voxels'],
                      'selected_in_roi_voxels': report['selected_in_roi_voxels'],
                      'unknown_group_voxels': report['unknown_group_voxels'],
                      'warning': report.get('selection_warning')}, indent=2), flush=True)


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
