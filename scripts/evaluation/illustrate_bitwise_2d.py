#!/usr/bin/env python3
"""Render a pixel teaching example using the actual FakeCT editing kernel.

The inspected plane is inside an extruded synthetic 3D object. At unit spacing,
its two-round four-neighbor Boolean masks equal the native six-neighbor proposals.
No phantom inputs, frozen cohorts, or data-generating modules are changed.
"""
import argparse
import hashlib
import json
import os
from pathlib import Path
import sys

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-pixel-illustration-mpl')
import numpy as np
from scipy import ndimage
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import to_rgba
from matplotlib.patches import Circle, Patch, Rectangle

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from fakect_morphology import apply_morphology, _owned_paths
from fakect_reassignment import reassignment_masks, stiffness_field
from fakect_roi import sphere_mask
from fakect_tissues import coarse_labels

INK, MUTED = '#172d42', '#597083'
COLORS = {10: '#dc6273', 30: '#e8c998', 40: '#8fdbd6', 90: '#7d899b'}
NAMES = {10: 'A', 30: 'S', 40: 'L', 90: 'B'}
ADD, RELEASE, BLOCK, UNRESOLVED = '#2eaa73', '#d7479c', '#c78916', '#794eac'
K, LOW, HIGH = 15, 9, 22
VIEW = (K, slice(LOW, HIGH), slice(LOW, HIGH))


def shift(mask, dj, di):
    """Nonwrapping shifts: outside the teaching grid is false."""
    padded = np.pad(mask, 1)
    return padded[1+dj:1+dj+mask.shape[0], 1+di:1+di+mask.shape[1]]


def boolean_round(mask, operation):
    values = [mask] + [shift(mask, *offset) for offset in ((0, 1), (0, -1), (1, 0), (-1, 0))]
    return (np.logical_or.reduce(values) if operation == 'dilation'
            else np.logical_and.reduce(values))


def fixture():
    shape = (31, 31, 31)
    k, j, i = np.indices(shape)
    labels = np.where(j >= 15, 40, 30).astype(np.int32)
    labels[:, 10:21, 19] = 90
    labels[(i-15)**2 + (j-15)**2 <= 3.6**2] = 10
    catalog = {'categories': [{'id': c, 'name': n} for c, n in
        ((1, 'artery'), (2, 'soft_tissue'), (3, 'lung'), (4, 'bone'), (255, 'unknown'))],
        'records': [{'original_id': original, 'original_name': name,
                     'classification': {'tissue_id': category, 'tissue_name': name}}
                    for original, category, name in ((10, 1, 'artery'), (30, 2, 'soft_tissue'),
                                                    (40, 3, 'lung'), (90, 4, 'bone'))]}
    roi = sphere_mask(shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), 3.7)
    target = labels == 10
    arrays = {'act': labels, 'atn': (np.arange(labels.size).reshape(shape)+1).astype(float)/1e5,
              'tissue': coarse_labels(labels, catalog), 'candidates': target,
              'selected': target & roi, 'roi': roi,
              'edit_region': sphere_mask(shape, (0, 0, 0), (15, 15, 15), (1, 1, 1), 6.2)}
    resolved = {'crop_low_ijk': (0, 0, 0), 'crop_high_ijk_exclusive': shape[::-1],
                'spacing_ijk_mm': (1, 1, 1), 'roi_kind': 'sphere',
                'roi_nodes_ijk': ((15, 15, 15),), 'roi_radii_mm': (3.7,),
                'catalog': catalog, 'source_ids': (10,)}
    config = {'selection': {'tissue': 'artery', 'source_ids': (10,)},
              'recipe': {'roi_role': 'selection'},
              'edit': {'operation': 'dilation', 'distance_mm': 2.2, 'profile': 'uniform'},
              'reassignment': {'mode': 'stiffness', 'allowed_tissues': (),
                  'max_distance_mm': 3., 'unresolved': 'preserve',
                  'stiffness': {'default': 0., 'labels': {},
                      'tissues': {'artery': .05, 'soft_tissue': .4, 'lung': 0., 'bone': 1., 'skin': .95}}}}
    return arrays, resolved, config


def compute():
    arrays, resolved, config = fixture()
    source = arrays['act'].copy()
    dilation = apply_morphology(arrays, resolved, config)
    erosion_config = {**config, 'edit': {**config['edit'], 'operation': 'erosion'}}
    erosion = apply_morphology(arrays, resolved, erosion_config)
    target = arrays['selected'][K]
    d1 = boolean_round(target, 'dilation'); d2 = boolean_round(d1, 'dilation')
    e1 = boolean_round(target, 'erosion'); e2 = boolean_round(e1, 'erosion')
    np.testing.assert_array_equal(dilation['proposed_added_mask'][K], d2 & ~target)
    np.testing.assert_array_equal(erosion['proposed_removed_mask'][K], target & ~e2)
    eligible, protected, _ = reassignment_masks(source, arrays['candidates'], resolved['catalog'], config['reassignment'])
    released = erosion['proposed_removed_mask']
    seeds = eligible & ndimage.binary_dilation(released, structure=ndimage.generate_binary_structure(3, 1))
    stiffness = stiffness_field(source, resolved['catalog'], config['reassignment'])
    costs, owners = _owned_paths(released | seeds, seeds, (1, 1, 1), 3., source, initial=3*stiffness)
    assigned = released & (owners >= 0)
    np.testing.assert_array_equal(assigned, erosion['removed_mask'])
    np.testing.assert_array_equal(erosion['edited_labels'][assigned], source.ravel()[owners[assigned]])
    np.testing.assert_array_equal(arrays['act'], source)
    for result in (dilation, erosion):
        assert not (result['changed_mask'] & protected).any()
    assert dilation['blocked_mask'][VIEW].any() and erosion['unresolved_mask'][VIEW].any()
    # A highlighted released pixel chooses a more distant, less stiff lung seed.
    p = (K, 14, 12)
    assert released[p] and source[p] == 10 and erosion['edited_labels'][p] == 40
    winner = tuple(int(x) for x in np.unravel_index(owners[p], source.shape))
    assert winner == (K, 15, 11) and np.isclose(costs[p], 2.)
    return arrays, dilation, erosion, seeds, costs, owners


def colors(labels, fade=1.):
    rgba = np.empty((*labels.shape, 4))
    for key, color in COLORS.items():
        rgba[labels == key] = to_rgba(color)
    rgba[..., :3] = fade*rgba[..., :3] + (1-fade)
    return rgba


def base(ax, labels, title, caption, fade=1., letters=False):
    ax.imshow(colors(labels, fade), origin='lower', extent=(LOW-.5, HIGH-.5, LOW-.5, HIGH-.5),
              interpolation='nearest')
    ax.set(xlim=(LOW-.5, HIGH-.5), ylim=(LOW-.5, HIGH-.5), aspect='equal')
    ax.set_xticks(np.arange(LOW-.5, HIGH, 1), minor=True)
    ax.set_yticks(np.arange(LOW-.5, HIGH, 1), minor=True)
    ax.grid(which='minor', color='white', lw=.75)
    ax.tick_params(which='minor', length=0)
    ax.set_xticks([10, 15, 20]); ax.set_yticks([10, 15, 20])
    ax.tick_params(labelsize=8, colors=MUTED, length=2)
    ax.set_xlabel('i  (pixel column)', fontsize=8, color=MUTED, labelpad=1)
    ax.set_ylabel('j', fontsize=8, color=MUTED, rotation=0, labelpad=7)
    ax.set_title(title, loc='left', fontsize=12, color=INK, weight='bold', pad=10)
    ax.text(0, -.17, caption, transform=ax.transAxes, va='top', color=INK, fontsize=9, linespacing=1.4)
    for edge in ax.spines.values(): edge.set_visible(False)
    if letters:
        for (j, i), value in np.ndenumerate(labels):
            ax.text(i+LOW, j+LOW, NAMES[int(value)], ha='center', va='center', fontsize=6.5, color=INK)


def mark(ax, mask, color, glyph=None, fill=True):
    for j, i in np.argwhere(mask):
        ax.add_patch(Rectangle((i+LOW-.45, j+LOW-.45), .9, .9,
                     facecolor=color if fill else 'none', edgecolor=color, lw=1.3, zorder=4))
        if glyph:
            ax.text(i+LOW, j+LOW, glyph, color='white' if fill else color, fontsize=8,
                    weight='bold', ha='center', va='center', zorder=5)


def generate(output, overwrite=False):
    output.mkdir(parents=True, exist_ok=overwrite)
    arrays, dilation, erosion, seeds, costs, owners = compute()
    source = arrays['act'][VIEW]; target = arrays['selected'][VIEW]
    borrowed = dilation['added_mask'][VIEW]; blocked = dilation['blocked_mask'][VIEW]
    proposed = dilation['proposed_added_mask'][VIEW]
    released = erosion['proposed_removed_mask'][VIEW]; assigned = erosion['removed_mask'][VIEW]
    unresolved = erosion['unresolved_mask'][VIEW]
    counts = {'original_target': int(target.sum()), 'dilation_proposed': int(proposed.sum()),
              'borrowed': int(borrowed.sum()), 'blocked': int(blocked.sum()),
              'released_proposed': int(released.sum()), 'reassigned': int(assigned.sum()),
              'unresolved_preserved': int(unresolved.sum())}
    assert counts['dilation_proposed'] == counts['borrowed'] + counts['blocked']
    assert counts['released_proposed'] == counts['reassigned'] + counts['unresolved_preserved']
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'pdf.fonttype': 42, 'svg.fonttype': 'none'})
    fig, axes = plt.subplots(2, 4, figsize=(18, 12))
    fig.subplots_adjust(left=.045, right=.982, bottom=.24, top=.80, wspace=.24, hspace=.58)
    fig.text(.035, .968, 'Bitwise morphology → borrowing and label reassignment',
             size=23, weight='bold', color=INK, va='top')
    fig.text(.035, .929, 'A pixelated teaching example computed with the current FakeCT kernel. '
             'Dilation and erosion each start from the same original grid.', size=11, color=MUTED)
    fig.text(.035, .897, 'Boolean mask T: 1 = selected artery, 0 = everything else.  '
             'Shifted copies N, S, E, W use face neighbors; original label IDs are kept separately.', size=10, color=INK)
    fig.text(.035, .865, 'Dilation:  D₁ = T | N | S | E | W       Erosion:  E₁ = T & N & S & E & W'
             '       | = OR     & = AND     ~ = NOT', size=12, color=INK, weight='bold')
    fig.text(.035, .839, 'Two rounds are shown (D₂, E₂). Pixels are 1 mm; edit budget = 2.2 mm; '
             'recipient search cap = 3 mm. No Gaussian taper or direction bias in this example.', size=10, color=MUTED)

    base(axes[0, 0], source, '1  Original labels + selection',
         f'{counts["original_target"]} artery pixels form T. Dashed circle =\nselection ROI; edits may grow beyond it.', letters=True)
    axes[0, 0].add_patch(Circle((15, 15), 3.7, fill=False, ls='--', lw=2, color=INK))
    base(axes[0, 1], source, '2  Dilation proposes new pixels',
         f'B* = D₂ & ~T  →  {counts["dilation_proposed"]} candidates.\nBoolean growth alone does not choose donors.', fade=.4)
    mark(axes[0, 1], target, COLORS[10]); mark(axes[0, 1], proposed, '#51b9ce', '+')
    base(axes[0, 2], source, '3  Borrow only permitted pixels',
         f'{counts["borrowed"]} borrowed (+); {counts["blocked"]} blocked (×).\nStiffness and reachable paths limit growth.', fade=.35)
    mark(axes[0, 2], target, COLORS[10]); mark(axes[0, 2], borrowed, ADD, '+')
    mark(axes[0, 2], blocked, BLOCK, '×')
    base(axes[0, 3], dilation['edited_labels'][VIEW], '4  Dilation updates the label map',
         'Borrowed S / L pixels become A. Their original\nlabels remain recoverable in the source snapshot.', letters=True)
    mark(axes[0, 3], borrowed, ADD, fill=False)

    base(axes[1, 0], source, '5  Erosion proposes releases',
         f'R* = T & ~E₂  →  {counts["released_proposed"]} proposed releases.\nPink marks geometry to remove, not a new tissue.', fade=.45)
    mark(axes[1, 0], target & ~released, COLORS[10]); mark(axes[1, 0], released, RELEASE, 'R')
    base(axes[1, 1], source, '6  Find surrounding label seeds',
         'Outlined S / L neighbors seed the search.\nPaths enter R*; they do not cross rigid bone.', fade=.6)
    mark(axes[1, 1], released, RELEASE, 'R')
    mark(axes[1, 1], seeds[VIEW], INK, fill=False)
    base(axes[1, 2], erosion['edited_labels'][VIEW], '7  Choose a reachable seed',
         'P chooses lung via Q: 2 + 3×0 = 2.0 mm.\nNearest soft seed: 1 + 3×0.4 = 2.2 mm.', fade=.75)
    mark(axes[1, 2], assigned, INK, fill=False)
    for i, j, letter in ((12, 14, 'P'), (11, 15, 'Q')):
        axes[1, 2].text(i, j, letter, color=INK, fontsize=12, weight='bold', ha='center', va='center', zorder=8,
                        bbox={'facecolor': 'white', 'edgecolor': INK, 'boxstyle': 'circle,pad=.10'})
    axes[1, 2].plot([11, 12, 12], [15, 15, 14], color=INK, lw=2, zorder=6)
    base(axes[1, 3], erosion['edited_labels'][VIEW], '8  Erosion finishes reassignment',
         f'{counts["reassigned"]} pixels → S / L. {counts["unresolved_preserved"]} unresolved pixels\n(purple outlines) stay A; no unlabeled holes.', letters=True)
    mark(axes[1, 3], unresolved, UNRESOLVED, fill=False)

    handles = [Patch(facecolor=COLORS[k], label=t) for k, t in
        ((10, 'A  artery'), (30, 'S  soft tissue'), (40, 'L  lung'), (90, 'B  rigid bone'))]
    handles += [Patch(facecolor=ADD, label='Borrowed'), Patch(facecolor=RELEASE, label='Proposed release'),
                Patch(facecolor=BLOCK, label='Blocked'), Patch(facecolor='none', edgecolor=UNRESOLVED, label='Unresolved → keep A')]
    fig.legend(handles=handles, loc='lower center', bbox_to_anchor=(.5, .135), ncol=8, frameon=False, fontsize=9)
    fig.text(.035, .101, 'Reassignment cost = path length + (search cap × recipient stiffness). '
             'Choose the lowest cost within the cap; ties use original ID, then seed coordinate.', size=10, color=INK)
    fig.text(.035, .077, 'Stiffness in this teaching grid: artery 0.05, soft tissue 0.40, lung 0, bone 1. '
             'Propagation reads the pre-edit label map and copies the winning seed label.', size=10, color=INK)
    fig.text(.035, .053, 'No recipient within the cap → preserve the original artery label. '
             'With assign_surrounding_tissue=false, releases instead get a diagnostic marker, not a training-ready label.', size=9.5, color=MUTED)
    fig.text(.035, .030, 'Toy anatomy. The inspected plane is inside an extruded 3D object: 4 face neighbors here correspond to '
             '6 in a volume. Production also uses ROI ranges, taper, direction and optional erosion safeguards.', size=9.5, color=MUTED)
    for suffix in ('png', 'pdf', 'svg'):
        fig.savefig(output/f'bitwise-morphology-2d.{suffix}', dpi=190, facecolor='white')
    plt.close(fig)
    evidence = {'schema_version': 'fakect.pixel-illustration/1', 'counts_in_displayed_plane': counts,
        'description': 'Independent dilation and erosion from one original synthetic grid; native kernel execution, not patient data.',
        'source_slice_k': K, 'display_i_j_range_inclusive': [LOW, HIGH-1],
        'source_labels': source.tolist(), 'dilation_labels': dilation['edited_labels'][VIEW].tolist(),
        'erosion_labels': erosion['edited_labels'][VIEW].tolist(),
        'borrowed_mask': borrowed.astype(int).tolist(), 'blocked_mask': blocked.astype(int).tolist(),
        'proposed_released_mask': released.astype(int).tolist(), 'unresolved_mask': unresolved.astype(int).tolist(),
        'highlight': {'pixel_ijk': [12, 14, K], 'winning_seed_ijk': [11, 15, K],
                      'winning_label_id': 40, 'winning_cost_mm': 2., 'nearest_soft_seed_cost_mm': 2.2},
        'checks': {'boolean_proposals_equal_native_kernel': True, 'source_unchanged': True,
                   'protected_labels_unchanged': True, 'assigned_labels_match_native_seed_owners': True,
                   'requested_equals_achieved_plus_blocked_or_unresolved': True},
        'code_sha256': {name: hashlib.sha256((ROOT/'src'/name).read_bytes()).hexdigest()
                       for name in ('fakect_morphology.py', 'fakect_reassignment.py')}}
    (output/'evidence.json').write_text(json.dumps(evidence, indent=2)+'\n')
    # One self-contained viewing page; vector/print formats remain separate exports.
    svg = (output/'bitwise-morphology-2d.svg').read_text()
    svg = svg[svg.index('<svg'):]
    page = '<!doctype html><html lang="en"><meta charset="utf-8"><title>FakeCT pixel morphology</title>'
    page += '<style>body{font:16px system-ui;margin:24px;color:#172d42}svg{width:100%;height:auto}.toolbar{display:flex;gap:20px;flex-wrap:wrap}</style>'
    page += '<nav class="toolbar"><a href="bitwise-morphology-2d.png">PNG</a><a href="bitwise-morphology-2d.pdf">PDF</a><a href="bitwise-morphology-2d.svg">Editable SVG</a><a href="evidence.json">Exact pixel counts and labels</a></nav>'
    page += svg + '<p>Counts refer only to the displayed plane. Bitwise operations act on Boolean membership masks; they do not overwrite integer label IDs directly.</p></html>'
    (output/'report.html').write_text(page)
    print(json.dumps({'report': str(output/'report.html'), 'counts': counts}, indent=2))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output', type=Path, default=ROOT/'outputs/illustrations/bitwise-2d')
    parser.add_argument('--overwrite', action='store_true', help='Regenerate the six illustration artifacts in an existing output directory')
    args = parser.parse_args()
    generate(args.output.resolve(), args.overwrite)
