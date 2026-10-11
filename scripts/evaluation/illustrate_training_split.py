#!/usr/bin/env python3
"""Poster flowchart of the actual family-held-out segmentation experiment."""
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch

ROOT = Path(__file__).resolve().parents[2]
SOURCE = ROOT / 'outputs/experiments/aorta/pretrained-finetune-branches-r3/study-lock.json'
OUT = ROOT / 'outputs/poster/bmes2026/training-split-flowchart-r1'


def main():
    if OUT.exists():
        raise FileExistsError('Use a new figure revision')
    lock = json.loads(SOURCE.read_text())
    folds, rosters = lock['folds'], lock['rosters']
    assert len(folds) == 5 and len(lock['cases']) == 10
    for fold in folds:
        assert len(fold['train_cases']) == 8 and len(fold['test_cases']) == 2
        assert sum(len(rosters['training_siblings'][c]) for c in fold['train_cases']) == 136
        assert sum(len(rosters['evaluation'][c]) for c in fold['test_cases']) == 26
    evaluation = {sid for ids in rosters['evaluation'].values() for sid in ids}
    assert len(evaluation) == 130
    OUT.mkdir(parents=True)
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none', 'pdf.fonttype': 42})
    fig, ax = plt.subplots(figsize=(11.8, 10.2))
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, 100); ax.set_ylim(-7, 102); ax.axis('off')
    ink, cyan, magenta = '#183144', '#009db8', '#ce009e'

    def box(x, y, width, height, title, detail='', border=ink, fill='#f3f6f8', size=12):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle='round,pad=0.45,rounding_size=1.2', linewidth=2.1,
            edgecolor=border, facecolor=fill, zorder=2))
        if detail:
            ax.text(x+width/2, y+height*.74, title, ha='center', va='center',
                fontsize=size, fontweight='bold', color=ink, zorder=3)
            ax.text(x+width/2, y+height*.32, detail, ha='center', va='center',
                fontsize=size-1, color=ink, linespacing=1.35, zorder=3)
        else:
            ax.text(x+width/2, y+height/2, title, ha='center', va='center',
                fontsize=size, fontweight='bold', color=ink, zorder=3)

    def arrow(points, color=ink, dashed=False):
        for start, stop in zip(points[:-2], points[1:-1]):
            ax.plot([start[0],stop[0]], [start[1],stop[1]], color=color,
                linewidth=1.9, linestyle='--' if dashed else '-', zorder=1)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
            mutation_scale=16, color=color, linewidth=1.9,
            linestyle='--' if dashed else '-', zorder=1))

    ax.text(50, 99, 'Training and held-out evaluation', ha='center', va='center',
        fontsize=19, fontweight='bold', color=ink)
    ax.text(50, 95, 'Actual segmentation experiment · five family rotations',
        ha='center', va='center', fontsize=11.5, color='#506677')
    box(13, 81, 47, 10, '12 reference phantoms', '6 provisional anatomy families')
    box(71, 81, 27, 10, '2 infant references', 'Excluded from this run', border='#8797a0', fill='#f0f1f2', size=10.5)
    arrow([(60.5,86),(70.3,86)], '#8797a0', True)
    box(13, 64, 47, 12, '10 phantoms · 5 families',
        '256 siblings + 1 original per phantom\n2,570-volume generation pool', size=11.5)
    arrow([(36.5,80.3),(36.5,76.8)])
    box(18, 49, 64, 10, 'Select one family to hold out',
        'Keep related phantoms and their siblings together', size=11.5)
    arrow([(36.5,63.3),(36.5,61.5),(50,61.5),(50,59.8)])
    box(3, 30, 44, 13, 'TRAINING',
        '4 families · 8 phantoms\n8 × (1 original + 16 selected siblings)\n136 training volumes',
        cyan, '#e9faff', 11.5)
    box(55, 30, 43, 13, 'TEST / EVALUATION',
        '1 family · 2 unseen phantoms\n2 × (1 original + 12 selected siblings)\n26 held-out volumes',
        magenta, '#fff0fb', 11.5)
    arrow([(50,48.3),(50,46.4),(25,46.4),(25,43.8)], cyan)
    arrow([(50,48.3),(50,46.4),(76.5,46.4),(76.5,43.8)], magenta)
    box(3, 15, 44, 10, 'Fine-tune TotalSegmentator', 'Training CT / reference-mask pairs', cyan, '#e9faff', 11.5)
    box(55, 15, 43, 10, 'Predict the held-out CTs', 'Pretrained + Fine-tuned → reference comparison', magenta, '#fff0fb', 10.5)
    arrow([(25,29.3),(25,25.8)], cyan)
    arrow([(76.5,29.3),(76.5,25.8)], magenta)
    arrow([(47.5,20),(54.3,20)], cyan)
    box(23, 0, 75, 10, 'Rotate the held-out family five times',
        '5 × 26 = 130 evaluated volumes: 10 originals + 120 siblings\nAverage held-out Dice across addition / removal burden', size=11.5)
    arrow([(76.5,14.3),(76.5,10.8)], magenta)
    ax.text(50, -4, 'Test and evaluation use the same held-out set. No separate validation set in this fixed-budget run.',
        ha='center', va='center', fontsize=10, color='#506677')
    fig.subplots_adjust(left=.025, right=.985, top=.99, bottom=.035)
    for suffix in ('png','pdf','svg'):
        fig.savefig(OUT/('training-testing-evaluation-flowchart.'+suffix), dpi=240,
            facecolor='white', bbox_inches='tight')
    plt.close(fig)
    receipt = dict(schema_version='fakect.training-split-figure/1',
        source_study=str(SOURCE), source_sha256=hashlib.sha256(SOURCE.read_bytes()).hexdigest(),
        library_reference_phantoms=12, active_reference_phantoms=10, active_families=5,
        excluded_reference_ids=['260612','260617'], train_volumes_per_fold=136,
        evaluation_volumes_per_fold=26, evaluation_unique_volumes=130,
        no_separate_validation_set=True, generator_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        output_sha256={p.name:hashlib.sha256(p.read_bytes()).hexdigest() for p in OUT.iterdir() if p.is_file()})
    (OUT/'provenance.json').write_text(json.dumps(receipt,indent=2)+'\n')
    print(OUT/'training-testing-evaluation-flowchart.png')


if __name__ == '__main__':
    main()
