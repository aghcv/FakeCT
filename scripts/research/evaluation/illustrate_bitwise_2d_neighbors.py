#!/usr/bin/env python3
"""Small pixel examples of resistance-aware growth and neighbor reassignment."""
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
import matplotlib.patheffects as pe
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import FancyArrowPatch, Patch, Rectangle
from matplotlib.lines import Line2D

from illustrate_bitwise_2d import ROOT, boolean_round
from fakect_morphology import _owned_paths

INK = '#203047'
MUTED = '#8090a3'
COLORS = ['#5576d1', '#e9b75f', '#65b7a4', '#a4abb5']
EMPTY = '#f7f8fa'
STEM = 'bitwise-morphology-2d-neighbors'


def compute():
    target = np.zeros((5, 5), dtype=bool)
    target[2, 1:4] = True
    target[1:4, 2] = True
    labels = np.where(np.indices(target.shape)[0] < 3, 2, 3).astype(np.int32)
    labels[target] = 1
    labels[2, 0] = 4
    original = labels.copy()
    eligible = (labels != 1) & (labels != 4)
    cross = ndimage.generate_binary_structure(2, 1)
    dilated = boolean_round(target, 'dilation')
    eroded = boolean_round(target, 'erosion')
    np.testing.assert_array_equal(dilated, ndimage.binary_dilation(target, structure=cross))
    np.testing.assert_array_equal(eroded, ndimage.binary_erosion(target, structure=cross))

    # A single plane in the native six-face graph has four in-plane neighbors.
    # Unit edge lengths, one pass, zero resistance except Label 4 (rigid).
    proposed = dilated & ~target
    _, growth_owners = _owned_paths((target | (proposed & eligible))[None],
        target[None], (1., 1., 1.), 1., labels[None])
    borrowed = proposed & (growth_owners[0] >= 0)
    blocked = proposed & ~borrowed
    grown = labels.copy()
    grown[borrowed] = labels.ravel()[growth_owners[0][borrowed]]

    released = target & ~eroded
    seeds = eligible & ndimage.binary_dilation(released, structure=cross)
    distances, owners = _owned_paths((released | seeds)[None], seeds[None],
        (1., 1., 1.), 1., labels[None])
    owners, distances = owners[0], distances[0]
    assert np.all(owners[released] >= 0)
    shrunk = labels.copy()
    shrunk[released] = labels.ravel()[owners[released]]
    focal = (2, 3)
    winner = tuple(int(v) for v in np.unravel_index(owners[focal], labels.shape))

    # Insets show local decisions, not additional rounds of whole-mask edits.
    # Dilation inset covers the source pixel and its up/down/left neighbors.
    growth_before = labels[1:4, 0:3].copy()
    growth_after = grown[1:4, 0:3].copy()
    # The blank center is a temporary illustration of the selected release;
    # other cells retain immutable original labels during this one decision.
    release_search = labels[1:4, 2:5].copy()
    release_search[1, 1] = 0
    release_after = release_search.copy()
    release_after[1, 1] = labels[winner]

    np.testing.assert_array_equal(labels, original)
    np.testing.assert_array_equal(grown == 1, target | borrowed)
    np.testing.assert_array_equal(shrunk == 1, eroded)
    np.testing.assert_array_equal(grown[labels == 4], labels[labels == 4])
    np.testing.assert_array_equal(shrunk[labels == 4], labels[labels == 4])
    assert (target.sum(), proposed.sum(), borrowed.sum(), blocked.sum(), released.sum()) == (5, 8, 7, 1, 4)
    assert blocked[2, 0] and winner == (1, 3) and shrunk[focal] == 2
    assert distances[focal] == 1 and not seeds[2, 2]
    assert growth_after[1, 0] == 4 and growth_after[0, 1] == growth_after[2, 1] == 1
    return {
        'labels': labels, 'grown': grown, 'shrunk': shrunk,
        'growth_before': growth_before, 'growth_after': growth_after,
        'release_search': release_search, 'release_after': release_after,
        'released': released, 'borrowed': borrowed, 'blocked': blocked,
        'owners': owners, 'winner': winner,
    }


def grid(ax, values, title):
    cmap = ListedColormap([EMPTY] + COLORS)
    ax.imshow(values, cmap=cmap, norm=BoundaryNorm(np.arange(-.5, 5, 1), cmap.N),
              interpolation='nearest', origin='upper')
    # Automatic major ticks can coincide with, and suppress, minor grid lines
    # on a 3x3 inset. Clear them so every pixel remains visibly separate.
    ax.set_xticks([])
    ax.set_yticks([])
    ax.set_xticks(np.arange(-.5, values.shape[1], 1), minor=True)
    ax.set_yticks(np.arange(-.5, values.shape[0], 1), minor=True)
    ax.grid(which='minor', color='white', linewidth=2.5)
    ax.tick_params(which='both', bottom=False, left=False, labelbottom=False, labelleft=False)
    for edge in ax.spines.values():
        edge.set_visible(False)
    ax.set_title(title, fontsize=12, color=INK, pad=12)


def highlight(ax, x, y, *, dashed=False):
    rectangle = Rectangle((x-.40, y-.40), .8, .8, fill=False, linewidth=1.7,
                          edgecolor=INK, linestyle='--' if dashed else '-', zorder=6)
    rectangle.set_path_effects([pe.Stroke(linewidth=3.7, foreground='white'), pe.Normal()])
    ax.add_patch(rectangle)


def arrow(ax, start, end, *, faint=False):
    line = FancyArrowPatch(start, end, arrowstyle='-|>', mutation_scale=14,
        linewidth=1.3 if faint else 2, color=MUTED if faint else INK,
        linestyle=':' if faint else '-', zorder=8)
    if not faint:
        line.set_path_effects([pe.Stroke(linewidth=4, foreground='white'), pe.Normal()])
    ax.add_patch(line)


def figure_arrow(fig, start, end):
    fig.add_artist(FancyArrowPatch(start, end, transform=fig.transFigure,
        arrowstyle='-|>', mutation_scale=17, linewidth=1.7, color=INK))


def generate(output, overwrite=False):
    data = compute()
    output.mkdir(parents=True, exist_ok=overwrite)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig = plt.figure(figsize=(11, 7.4), facecolor='white')
    fig.text(.06, .965, 'Dilation, erosion & neighboring labels', fontsize=22,
             color=INK, weight='bold', va='top')
    fig.text(.94, .905, 'One step · 4 neighbors', fontsize=10, color=MUTED, ha='right')
    top_axes = []
    width, height, bottom = .215, .215*11/7.4, .54
    for left, key, title in zip((.06, .3925, .725), ('grown', 'labels', 'shrunk'),
                               ('Dilation', 'Original', 'Erosion')):
        ax = fig.add_axes([left, bottom, width, height])
        grid(ax, data[key], title)
        top_axes.append(ax)
    y = bottom + height/2
    figure_arrow(fig, (.373, y), (.293, y))
    figure_arrow(fig, (.627, y), (.707, y))

    # This divider associates the two examples with the outward operations;
    # it is a visual guide, not an ROI or a restriction on either full edit.
    top_axes[1].plot([2, 2], [-.5, 4.5], color=INK, linewidth=1.3,
        linestyle=(0, (3, 3)), path_effects=[pe.Stroke(linewidth=3.5, foreground='white'), pe.Normal()])
    highlight(top_axes[1], 0, 2)
    highlight(top_axes[0], 0, 2)
    highlight(top_axes[1], 3, 2, dashed=True)
    highlight(top_axes[2], 3, 2, dashed=True)

    # Short leaders connect each half of the original to its extracted example.
    for xs in ((.445, .445, .267), (.555, .555, .737)):
        fig.add_artist(Line2D(xs, (.527, .512, .483), transform=fig.transFigure,
            color='#bac3ce', linewidth=1.1))
    fig.text(.267, .449, 'Borrow where allowed', fontsize=14, weight='bold',
             color=INK, ha='center')
    fig.text(.737, .449, 'Reassign a released pixel', fontsize=14, weight='bold',
             color=INK, ha='center')

    inset_width, inset_height, inset_bottom = .155, .155*11/7.4, .142
    inset_axes = []
    for left, key, title in zip((.06, .32, .53, .79),
            ('growth_before', 'growth_after', 'release_search', 'release_after'),
            ('Check neighbors', 'Grow', 'Search neighbors', 'Copy label')):
        ax = fig.add_axes([left, inset_bottom, inset_width, inset_height])
        grid(ax, data[key], title)
        inset_axes.append(ax)
    mid = inset_bottom + inset_height/2
    figure_arrow(fig, (.235, mid), (.3, mid))
    figure_arrow(fig, (.705, mid), (.77, mid))

    # Left: upward/downward growth is possible; a rigid left cell stays put.
    for ax in inset_axes[:2]:
        highlight(ax, 0, 1)
    arrow(inset_axes[0], (1, .8), (1, .03))
    arrow(inset_axes[0], (1, 1.2), (1, 1.97))
    inset_axes[0].plot([.55, .84], [1, 1], color=INK, linewidth=2.2, zorder=9)
    inset_axes[0].plot([.54, .54], [.73, 1.27], color=INK, linewidth=3.2, zorder=9,
        path_effects=[pe.Stroke(linewidth=5.2, foreground='white'), pe.Normal()])

    # Right: only surrounding non-target, non-rigid cells are recipients.
    # Show eligible paths faintly; the native owner is the upper Label 2 cell.
    highlight(inset_axes[2], 1, 1, dashed=True)
    highlight(inset_axes[3], 1, 1, dashed=True)
    arrow(inset_axes[2], (1.9, 1), (1.15, 1), faint=True)
    arrow(inset_axes[2], (1, 1.9), (1, 1.15), faint=True)
    arrow(inset_axes[2], (1, .1), (1, .89))
    fig.text(.267, .102, 'Resistant pixel stays unchanged', ha='center', color=MUTED, fontsize=10)
    fig.text(.737, .102, 'Choose an eligible neighbor', ha='center', color=MUTED, fontsize=10)
    names = ['Label 1 · target', 'Label 2', 'Label 3', 'Label 4 · resistant']
    fig.legend(handles=[Patch(facecolor=color, label=name) for color, name in zip(COLORS, names)],
        loc='lower center', bbox_to_anchor=(.5, .015), ncol=4, frameon=False,
        fontsize=11, columnspacing=2.2, handlelength=1.3)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'{STEM}.{suffix}', dpi=220, facecolor='white')
    plt.close(fig)

    evidence = {
        'schema_version': 'fakect.neighbor-pixel-illustration/1',
        'grid_shape': [5, 5], 'connectivity': 4, 'iterations': 1,
        'target_label': 1, 'resistant_label': 4,
        'counts': {'original_target': 5, 'proposed_borrowing': 8, 'borrowed': 7,
                   'blocked': 1, 'dilated_target': 12, 'released_and_reassigned': 4, 'eroded_target': 1},
        'original_labels': data['labels'].tolist(),
        'dilated_labels': data['grown'].tolist(), 'eroded_labels': data['shrunk'].tolist(),
        'dilation_example': {'blocked_pixel_row_col': [2, 0], 'source_pixel_row_col': [2, 1],
            'crop_rows': [1, 4], 'crop_columns': [0, 3],
            'before': data['growth_before'].tolist(), 'after': data['growth_after'].tolist()},
        'erosion_example': {'released_pixel_row_col': [2, 3],
            'selected_owner_row_col': list(data['winner']), 'selected_label': 2,
            'eligible_face_neighbors': [{'row_col': [1, 3], 'label': 2},
                {'row_col': [2, 4], 'label': 2}, {'row_col': [3, 3], 'label': 3}],
            'ineligible_target_neighbor_row_col': [2, 2],
            'search': data['release_search'].tolist(), 'single_pixel_result': data['release_after'].tolist()},
        'semantics': [
            'The top results independently edit the entire original target; the divider only separates the two examples visually.',
            'Boolean OR/AND is applied to the target mask, not integer label IDs. Dilation then filters proposals by eligibility and native reachable paths.',
            'Unit spacing, one step. Label 4 has stiffness 1 and cannot be borrowed or act as an erosion recipient; other surrounding labels have stiffness 0.',
            'Releases copy original labels through the native _owned_paths graph. Equal costs choose lower label ID, then original seed index, not a majority vote.',
            'Inset crops use half-open row/column ranges. The erosion inset illustrates only the marked pixel decision, using unchanged original neighbors; the blank center is illustrative and is never saved to the final label map.',
            'Faint erosion arrows show other eligible candidates. The solid arrow indicates the chosen upper neighbor. There is no eligible arrow from the retained target at left.',
            'This intentionally minimal example omits ROI, variable stiffness and geometric safeguards.',
        ],
        'checks': {'boolean_matches_scipy': True, 'native_ownership_used': True,
            'source_unchanged': True, 'resistant_label_preserved': True,
            'all_releases_resolved': True, 'inset_coordinates_verified': True},
    }
    (output/'evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    svg = (output/f'{STEM}.svg').read_text()
    svg = svg[svg.index('<svg'):]
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>Neighbor label examples</title>'
    page += '<style>body{margin:20px;font:15px system-ui;color:#203047}svg{display:block;max-width:1200px;width:100%;height:auto;margin:auto}nav{display:flex;justify-content:center;gap:24px}details{max-width:850px;margin:20px auto;line-height:1.6}</style><nav>'
    page += ' '.join(f'<a href="{STEM}.{suffix}">{suffix.upper()}</a>' for suffix in ('png', 'pdf', 'svg'))
    page += '</nav>'+svg+'<details><summary>How the example is computed</summary>'
    page += '<p>The split line separates two examples; both upper results independently edit the full original target. One four-neighbor step proposes eight new pixels. Seven become Label 1; the resistant Label 4 pixel stays unchanged.</p>'
    page += '<p>Erosion releases four pixels. The lower-right inset isolates one release: it searches eligible original neighbors and copies Label 2 from the upper neighbor. Equal path costs use lower label ID, then seed position; this is not a majority vote. Faint arrows show other candidates. The adjacent retained target is excluded.</p>'
    page += '<p>The blank cell is only an explanatory intermediate state. The final upper-right result contains no unassigned pixels. The example uses the same ownership primitive as the edit engine, with unit pixel spacing, zero extra resistance for Labels 2 and 3, and Label 4 fully protected.</p>'
    page += '<p><a href="evidence.json">Exact arrays and validation</a></p></details></html>'
    (output/'report.html').write_text(page)
    print(json.dumps({'report': str(output/'report.html'), 'counts': evidence['counts']}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/illustrations/bitwise-2d-neighbors')
    parser.add_argument('--overwrite', action='store_true')
    args = parser.parse_args()
    generate(args.output.resolve(), args.overwrite)
