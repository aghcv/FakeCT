#!/usr/bin/env python3
"""Minimal 5x5 Boolean morphology and generic-label illustration."""
import argparse
import json
import os
from pathlib import Path

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-pixel-illustration-mpl')
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyArrowPatch, Patch

from illustrate_bitwise_2d import ROOT, boolean_round
from fakect_morphology import _owned_paths

INK = '#203047'
EMPTY = '#eef1f5'
COLORS = ['#5576d1', '#e9b75f', '#65b7a4']


def compute():
    target = np.zeros((5, 5), dtype=bool)
    target[2, 1:4] = True
    target[1:4, 2] = True
    dilated = boolean_round(target, 'dilation')
    eroded = boolean_round(target, 'erosion')
    cross = ndimage.generate_binary_structure(2, 1)
    np.testing.assert_array_equal(dilated, ndimage.binary_dilation(target, structure=cross))
    np.testing.assert_array_equal(eroded, ndimage.binary_erosion(target, structure=cross))
    labels = np.where(np.indices(target.shape)[0] < 3, 2, 3).astype(np.int32)
    labels[target] = 1
    original = labels.copy()
    borrowed = dilated & ~target
    released = target & ~eroded
    grown_labels = labels.copy()
    grown_labels[borrowed] = 1
    # Same deterministic ownership primitive as the current native engine;
    # a one-plane graph has four in-plane neighbors. All recipients have zero
    # extra stiffness and every release is one edge from an eligible seed.
    seeds = ~target & ndimage.binary_dilation(released, structure=cross)
    _, owners = _owned_paths((released | seeds)[None], seeds[None], (1., 1., 1.),
                             1., labels[None])
    owners = owners[0]
    assert np.all(owners[released] >= 0)
    shrunk_labels = labels.copy()
    shrunk_labels[released] = labels.ravel()[owners[released]]
    np.testing.assert_array_equal(labels, original)
    np.testing.assert_array_equal(grown_labels == 1, dilated)
    np.testing.assert_array_equal(shrunk_labels == 1, eroded)
    assert (target.sum(), dilated.sum(), eroded.sum()) == (5, 13, 1)
    return target, dilated, eroded, labels, grown_labels, shrunk_labels


def grid(ax, values, title, binary=False):
    cmap = ListedColormap([EMPTY, COLORS[0]] if binary else COLORS)
    norm = BoundaryNorm([-.5, .5, 1.5] if binary else [.5, 1.5, 2.5, 3.5], cmap.N)
    ax.imshow(values, cmap=cmap, norm=norm, interpolation='nearest')
    ax.set_xticks(np.arange(-.5, 5, 1), minor=True)
    ax.set_yticks(np.arange(-.5, 5, 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2.2)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    for edge in ax.spines.values():
        edge.set_visible(False)
    ax.set_title(title, fontsize=13, color=INK, pad=12, weight='medium')


def generate(output, overwrite=False):
    target, dilated, eroded, labels, grown, shrunk = compute()
    output.mkdir(parents=True, exist_ok=overwrite)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(10, 7.6), facecolor='white')
    fig.text(.06, .965, 'Dilation & erosion', color=INK, fontsize=23, weight='bold', va='top')
    fig.text(.06, .905, '1  Boolean masks', color=INK, fontsize=12, weight='bold')
    fig.text(.94, .905, 'One step · 4 neighbors', color='#738094', fontsize=10, ha='right')
    fig.text(.06, .455, '2  Label reassignment', color=INK, fontsize=12, weight='bold')
    lefts, width, height = [.06, .395, .73], .21, .21*10/7.6
    panels = ((dilated, target, eroded), (grown, labels, shrunk))
    titles = (('Dilation (OR)', 'Original', 'Erosion (AND)'), ('Borrowed', 'Original labels', 'Reassigned'))
    for row, bottom in enumerate((.56, .11)):
        for col, left in enumerate(lefts):
            grid(fig.add_axes([left, bottom, width, height]), panels[row][col],
                 titles[row][col], binary=row == 0)
        # Both outcomes derive from the central original, not from each other.
        y = bottom + height/2
        for start, end in ((.38, .285), (.62, .715)):
            fig.add_artist(FancyArrowPatch((start, y), (end, y), transform=fig.transFigure,
                arrowstyle='-|>', mutation_scale=18, linewidth=1.7, color=INK))
    fig.legend(handles=[Patch(facecolor=color, label=f'Label {index}')
                        for index, color in enumerate(COLORS, 1)],
               loc='lower center', bbox_to_anchor=(.5, .025), ncol=3,
               frameon=False, fontsize=11, columnspacing=2.5, handlelength=1.4)
    stem = 'bitwise-morphology-2d-simple'
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'{stem}.{suffix}', dpi=220, facecolor='white')
    plt.close(fig)
    evidence = {'schema_version': 'fakect.simple-pixel-illustration/1',
        'grid_shape': [5, 5], 'selected_label': 1, 'connectivity': 4, 'iterations': 1,
        'counts': {'original': 5, 'dilated': 13, 'eroded': 1, 'borrowed': 8, 'reassigned': 4},
        'original_labels': labels.tolist(), 'dilated_labels': grown.tolist(),
        'eroded_labels': shrunk.tolist(),
        'semantics': 'Each row branches from its central original. All neighbors are eligible, with zero extra resistance. Released cells copy a face-adjacent original label; equal path costs use the lower original label ID then seed coordinate. This intentionally simplified example omits stiffness, ROI and safeguards.',
        'checks': {'boolean_matches_scipy': True, 'source_unchanged': True,
                   'label_membership_matches_boolean_result': True,
                   'every_release_has_native_kernel_owner': True}}
    (output/'evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    svg = (output/f'{stem}.svg').read_text()
    svg = svg[svg.index('<svg'):]
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Simple pixel morphology</title>'
    page += '<style>body{margin:20px;font:15px system-ui;color:#203047}svg{display:block;max-width:1100px;width:100%;height:auto;margin:auto}nav{display:flex;justify-content:center;gap:24px}</style><nav>'
    page += ' '.join(f'<a href="{stem}.{suffix}">{suffix.upper()}</a>' for suffix in ('png', 'pdf', 'svg'))
    page += '</nav>' + svg + '</html>'
    (output/'report.html').write_text(page)
    print(json.dumps({'report': str(output/'report.html'), 'counts': evidence['counts']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/illustrations/bitwise-2d-simple')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    generate(args.output.resolve(), args.overwrite)
