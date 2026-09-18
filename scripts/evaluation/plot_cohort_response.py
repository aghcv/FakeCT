#!/usr/bin/env python3
"""Plot native aorta cohort response from a completed recipe preflight audit.

Usage:
    python3 scripts/evaluation/plot_cohort_response.py \
        --preflight outputs/studies/thoracic-aorta/cohort-preflight-v1/preflight.json \
        --output docs/integration/2026-09-18/cohort-distance-response.png

Writes the requested PNG and a PDF beside it. Reads recorded outcomes only;
does not simulate edits, load source volumes, modify the audit, or fit a model.
The two axes are arch_outer_expand.distance_mm and coa_narrow.distance_mm;
one variant per distance pair and one pass per named edit are required.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path

import numpy as np


DILATION = 'arch_outer_expand'
EROSION = 'coa_narrow'


def response_data(document):
    if document.get('schema_version') != 'fakect.recipe-preflight/1' or not document.get('execution_valid'):
        raise ValueError('Use a completed, valid fakect.recipe-preflight/1 audit')
    samples = document['samples']
    if not document.get('complete') or any(row.get('status') != 'ok' for row in samples):
        raise ValueError('Every preflight variant must have completed successfully')
    baselines = [row for row in samples if row['operation'] == 'none']
    if len(baselines) != 1:
        raise ValueError('Exactly one original-phantom baseline is required')
    baseline = baselines[0]
    spacing = np.asarray(baseline['geometry']['spacing_ijk_mm'], dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError('Native spacing must contain three positive finite values')
    voxel_mm3 = float(np.prod(spacing))
    records = []
    for sample in samples:
        if sample['operation'] == 'none':
            continue
        if (sample['case_id'], sample['frame'], sample['anatomy_family']) != (
                baseline['case_id'], baseline['frame'], baseline['anatomy_family']):
            raise ValueError('This response chart requires one source anatomy and frame')
        if sample['geometry'] != baseline['geometry']:
            raise ValueError('All samples must share the baseline native geometry')
        steps = {}
        for name in (DILATION, EROSION, 'arch_inner_narrow'):
            rows = [step for step in sample['step_outcomes'] if step['step_name'] == name]
            if len(rows) != 1:
                raise ValueError(f'Exactly one pass of {name} is required for each edited variant')
            steps[name] = rows[0]
        records.append({'dilation_mm': float(sample['edits'][DILATION]['distance_mm']),
                        'erosion_mm': float(sample['edits'][EROSION]['distance_mm']),
                        'delta_mm3': (sample['foreground_voxels'] - baseline['foreground_voxels']) * voxel_mm3,
                        'steps': steps})
    if not records:
        raise ValueError('No edited cohort variants are available')
    pairs = [(row['dilation_mm'], row['erosion_mm']) for row in records]
    if len(set(pairs)) != len(pairs):
        raise ValueError('Extra sweep axes or repeated distance pairs are unsupported by this two-axis chart')
    dilation = sorted({row['dilation_mm'] for row in records})
    erosion = sorted({row['erosion_mm'] for row in records})
    if len(records) != len(dilation) * len(erosion):
        raise ValueError('The distance grid must be complete')
    grid = np.empty((len(dilation), len(erosion)), dtype=float)
    for row in records:
        grid[dilation.index(row['dilation_mm']), erosion.index(row['erosion_mm'])] = row['delta_mm3']
    return baseline, voxel_mm3, records, dilation, erosion, grid


def draw(preflight, output):
    raw = Path(preflight).read_bytes()
    document = json.loads(raw)
    baseline, voxel_mm3, records, dilation, erosion, grid = response_data(document)
    os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-cohort-plot-matplotlib')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    from matplotlib.colors import TwoSlopeNorm

    plt.rcParams.update({'font.family': 'DejaVu Sans', 'font.size': 10,
                         'axes.spines.top': False, 'axes.spines.right': False,
                         'pdf.fonttype': 42, 'savefig.facecolor': 'white'})
    fig = plt.figure(figsize=(13.8, 10.0))
    layout = fig.add_gridspec(2, 2, height_ratios=(1, .9), hspace=.65, wspace=.27,
                            left=.08, right=.94, top=.83, bottom=.19)
    added_ax = fig.add_subplot(layout[0, 0])
    removed_ax = fig.add_subplot(layout[0, 1])
    grid_ax = fig.add_subplot(layout[1, :])

    def curve(ax, distances, coordinate, step, field, label, color, marker):
        groups = [np.asarray([row['steps'][step][field] * voxel_mm3
                              for row in records if row[coordinate] == value]) for value in distances]
        middle = np.asarray([values.mean() for values in groups])
        low = np.asarray([values.min() for values in groups])
        high = np.asarray([values.max() for values in groups])
        ax.plot(distances, middle, color=color, marker=marker, lw=2.2, ms=6, label=label)
        if np.any(high != low):
            ax.fill_between(distances, low, high, color=color, alpha=.18,
                            label=label + ': range across other axis')
        for x, y, lower, upper in zip(distances, middle, low, high):
            value = f'{y:,.0f}' if lower == upper else f'{lower:,.0f}–{upper:,.0f}'
            ax.annotate(value, (x, y), xytext=(0, 9), textcoords='offset points',
                        color=color, ha='center', fontsize=10, fontweight='bold')
        return high.max()

    peak = curve(added_ax, dilation, 'dilation_mm', DILATION, 'added_voxels',
                 'Applied additions', '#007c91', 'o')
    added_ax.set(title='A  Outer-arch dilation: achieved additions',
                 xlabel='Requested outer-arch distance budget (mm)', ylabel='Added target volume (mm³)',
                 xticks=dilation, ylim=(0, max(1., peak * 1.25)))
    peak_removed = curve(removed_ax, erosion, 'erosion_mm', EROSION, 'removed_voxels',
                         'Applied removals', '#cf5b18', 'o')
    peak_unresolved = curve(removed_ax, erosion, 'erosion_mm', EROSION, 'unresolved_voxels',
                            'Unresolved; original label kept', '#7454a0', 's')
    removed_ax.set(title='B  CoA erosion: achieved versus unresolved',
                   xlabel='Requested CoA distance budget (mm)', ylabel='Volume (mm³)',
                   xticks=erosion, ylim=(0, max(1., max(peak_removed, peak_unresolved) * 1.25)))
    removed_ax.legend(loc='upper left', frameon=False, fontsize=9)
    for ax in (added_ax, removed_ax):
        ax.set_axisbelow(True)
        ax.grid(axis='y', color='#d8dde2', lw=.6)
        ax.margins(x=.12)
        ax.title.set_fontweight('bold')
        ax.title.set_fontsize(11)

    extent = max(1., float(np.abs(grid).max()))
    heat = grid_ax.imshow(grid, cmap='RdBu', norm=TwoSlopeNorm(vmin=-extent, vcenter=0, vmax=extent),
                          origin='lower', aspect='auto', interpolation='nearest')
    grid_ax.set(xticks=np.arange(len(erosion)), xticklabels=[f'{v:g}' for v in erosion],
                yticks=np.arange(len(dilation)), yticklabels=[f'{v:g}' for v in dilation],
                xlabel='Requested CoA distance budget (mm)', ylabel='Requested outer-arch\ndistance budget (mm)',
                title='C  Final selected-target volume change from the original phantom')
    grid_ax.title.set_fontweight('bold')
    grid_ax.title.set_fontsize(11)
    for row in range(len(dilation)):
        for column in range(len(erosion)):
            value = grid[row, column]
            grid_ax.text(column, row, f'{value:+,.0f}', ha='center', va='center',
                         color='white' if abs(value) > extent * .60 else '#172331', fontsize=11)
    colorbar = fig.colorbar(heat, ax=grid_ax, fraction=.025, pad=.025)
    colorbar.set_label('Net target volume change (mm³)')
    spacing = baseline['geometry']['spacing_ijk_mm']
    fig.suptitle('Requested edit budgets and achieved cohort geometry', x=.08, ha='left',
                 y=.975, fontsize=18, fontweight='bold', color='#183347')
    subtitle = (f"XCAT {baseline['case_id']}, frame {baseline['frame']}  |  {len(records)} edited variants + original baseline  |  "
                f"{document['unique_geometry_count']} distinct masks\n"
                f"Native grid: {' × '.join(f'{v:g}' for v in spacing)} mm; 1 voxel = {voxel_mm3:g} mm³.  "
                f"Original selected target: {baseline['foreground_voxels']:,} voxels "
                f"({baseline['foreground_voxels'] * voxel_mm3:,.0f} mm³).")
    fig.text(.08, .93, subtitle, ha='left', va='top', fontsize=10.5, linespacing=1.6, color='#415369')
    inner = {(row['steps']['arch_inner_narrow']['requested_distance_mm'],
              row['steps']['arch_inner_narrow']['removed_voxels'],
              row['steps']['arch_inner_narrow']['unresolved_voxels']) for row in records}
    if len(inner) == 1:
        distance, removed, unresolved = next(iter(inner))
        fixed = (f'Fixed inner-arch step: {distance:g} mm requested; {removed:,} voxels removed and '
                 f'{unresolved:,} unresolved in every edited variant.')
    else:
        fixed = 'The inner-arch step is included in each final result; its recorded outcomes vary across combinations.'
    footer = (fixed + '\n'
              'Unresolved releases keep their original labels. Tissue resistance and recipient search constrain achieved changes.\n'
              'Distances are requested voxel-grid budgets, not measured wall displacement, lumen area, or clinical severity.\n'
              'Target: ROI-selected tissue and surviving descendants; nearby selected branches may be included. One source anatomy.')
    fig.text(.08, .115, footer, ha='left', va='top', fontsize=9, linespacing=1.55, color='#415369')
    fig.text(.08, .018, f"Source: {Path(preflight).parent.name}/{Path(preflight).name}  |  "
             f"SHA256 {hashlib.sha256(raw).hexdigest()[:16]}  |  Native preflight; no model performance shown",
             ha='left', fontsize=8, color='#5a6775')
    output = Path(output)
    if output.suffix.lower() != '.png':
        raise ValueError('--output must be a PNG path; a PDF is also written beside it')
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=180)
    fig.savefig(output.with_suffix('.pdf'))
    plt.close(fig)
    return output, output.with_suffix('.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument('--preflight', required=True, type=Path)
    parser.add_argument('--output', required=True, type=Path)
    arguments = parser.parse_args()
    try:
        for path in draw(arguments.preflight, arguments.output):
            print(path)
    except (ValueError, KeyError, OSError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
