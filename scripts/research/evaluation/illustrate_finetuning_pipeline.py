#!/usr/bin/env python3
"""Render the verified CT-only fine-tuning pipeline without loading a model.

The SVG retains native editable text. Output directories are immutable; use a
new revision or a temporary directory for another rendering. This script reads
only small source/configuration/provenance files and runs no training/inference.
"""
from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import re
import xml.etree.ElementTree as ET
from xml.sax.saxutils import escape

os.environ.setdefault('MPLCONFIGDIR', '/tmp/fakect-finetuning-figure-matplotlib')
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


ROOT = Path(__file__).resolve().parents[3]
DEFAULT_OUT = ROOT / 'outputs/poster/bmes2026/finetuning-pipeline-r1'
CONFIG = ROOT / 'archive/research/configs/models/aorta-pretrained-finetune-branches-v3.ini'
STUDY = ROOT / 'outputs/experiments/aorta/pretrained-finetune-branches-r3/study-lock.json'
WIDTH, HEIGHT = 1800, 1280
INK, CYAN, MAGENTA = '#183144', '#007d95', '#b6007f'
MUTED, CYAN_FILL, MAGENTA_FILL = '#526575', '#effafd', '#fff1fa'
STEM = 'finetuning-pipeline'


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_facts():
    config = configparser.ConfigParser(inline_comment_prefixes=('#',))
    config.read(CONFIG)
    lock = json.loads(STUDY.read_text())
    metadata_directory = Path(lock['checkpoint']).parents[1]
    dataset_path = metadata_directory / 'dataset.json'
    plans_path = metadata_directory / 'plans.json'
    labels = json.loads(dataset_path.read_text())['labels']
    names = [name.strip() for name in config['target']['foreground_labels'].split(',')]
    grouped = {name: labels[name] for name in names}
    assert sorted(grouped.values()) == lock['foreground_channels'] == [2, 4, 5, 6, 7, 8]
    assert len(labels) == 19 and set(labels.values()) == set(range(19))
    assert config['training'].getint('updates') == lock['config']['training']['updates'] == 2000
    assert config['training'].getfloat('learning_rate') == 1e-4
    assert config['input'].getfloat('spacing_mm') == 1.5
    sources = [CONFIG, STUDY, dataset_path, plans_path,
               ROOT / 'src/fakect_pretrained_learning.py',
               ROOT / 'src/fakect_pretrained_data.py',
               ROOT / 'src/fakect_pretrained_study.py']
    return {
        'schema_version': 'fakect.finetuning-pipeline-figure/1',
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'purpose': 'Poster schematic of the branch-inclusive region-assisted development comparison; no new measured results.',
        'sources_sha256': {str(path.relative_to(ROOT)): digest(path) for path in sources},
        'method': {
            'training_arms': ['original CT only', 'original CT plus digital siblings'],
            'training_inputs': 'CT only, one input channel; masks and edit maps never enter the network forward pass.',
            'geometry': 'Matched frozen case-specific physical box; canonical RAS model grid; 1.5 mm isotropic sampling.',
            'ct_preparation': 'Attenuation-to-HU calibration; native int32 HU; linear resampling; int32 then float32 official Dataset293 normalization.',
            'normalization': {'clip_hu': [-101.0, 813.0], 'mean': 142.1977119916391, 'std': 154.74032651622053},
            'target': config['target']['policy'],
            'branch_extent': config['target']['branch_extent'],
            'reference_masks': 'Nearest-neighbor resampling on the identical model grid, with a separate annotated-domain validity mask.',
            'architecture': 'Complete original cardiac PlainConvUNet with 19 output classes, including all pretrained output heads.',
            'initialization': 'Original architecture reconstructed and full network state loaded with strict=True; original optimizer/scheduler/epoch state not restored.',
            'foreground_channels_by_name': grouped,
            'binary_training_logit': 'logsumexp(foreground logits) - logsumexp(complement logits)',
            'loss': 'Masked binary cross entropy plus foreground Dice, with deep supervision. The coarsest supervision level has zero default loss weight.',
            'training': {'optimizer': 'fresh AdamW', 'learning_rate': 1e-4,
                         'weight_decay': config['training'].getfloat('weight_decay'),
                         'batch_size': 1, 'patch_shape': [128, 128, 128],
                         'updates_per_arm_per_fold': 2000,
                         'all_parameters_trainable': True,
                         'checkpoint': 'Final fixed-budget weights; no held-out checkpoint selection or early stopping.'},
            'evaluation': 'Held-out-family original/sibling CTs enter pretrained and final fine-tuned models. All arms use identical box, normalization, Gaussian overlap tiling, no mirroring, and foreground extraction.',
            'prediction_union': '19-way argmax in channels [2,4,5,6,7,8], not thresholding the summed foreground probability.',
            'metric': 'Native-grid Dice against the corresponding reference target within the annotated evaluation domain; no numerical performance is invented.',
            'scope': 'Region-assisted development comparison with provisional atlas-derived references; not an untouched clinical test.',
        },
        'visual_encoding': {'CT_and_predictions': CYAN, 'reference_masks_and_loss': MAGENTA,
                            'gradient_update': MUTED},
        'canvas': {'svg_width_px': WIDTH, 'svg_height_px': HEIGHT,
                   'png_width_px': WIDTH * 2, 'png_height_px': HEIGHT * 2,
                   'main_body_font_px': 36, 'minimum_font_px': 28,
                   'svg_text_is_editable': True},
        'generator_sha256': digest(__file__),
    }


def render(out):
    if out.exists():
        raise FileExistsError(f'Refusing to overwrite {out}; choose a new revision.')
    facts = load_facts()
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'axes.unicode_minus': False})
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=72)
    ax = fig.add_axes([0, 0, 1, 1])
    fig.patch.set_facecolor('white')
    ax.set_xlim(0, WIDTH)
    ax.set_ylim(HEIGHT, 0)
    ax.axis('off')
    text_checks = []

    def label(x, y, text, *, size=36, color=INK, bold=False, ha='center', area=None):
        artist = ax.text(x, y, text, fontsize=size, color=color, ha=ha, va='center',
                         fontweight='bold' if bold else 'normal', zorder=5)
        text_checks.append((artist, area))
        return artist

    def box(x, y, width, height, color=CYAN, fill=CYAN_FILL):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle='round,pad=0,rounding_size=16', linewidth=3,
            edgecolor=color, facecolor=fill, zorder=2))
        return (x + 12, y + 8, x + width - 12, y + height - 8)

    def arrow(points, color=CYAN, *, width=3.8):
        for start, stop in zip(points[:-2], points[1:-1]):
            ax.plot([start[0], stop[0]], [start[1], stop[1]], color=color,
                    linewidth=width, solid_capstyle='round', zorder=3)
        ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
            mutation_scale=24, linewidth=width, color=color, shrinkA=0, shrinkB=0,
            zorder=3))

    label(60, 62, 'CT-only fine-tuning and held-out evaluation', size=52, bold=True, ha='left')
    ax.plot([63, 110], [125, 125], color=CYAN, linewidth=6)
    label(128, 125, 'CT / predictions', size=30, color=CYAN, ha='left')
    ax.plot([535, 582], [125, 125], color=MAGENTA, linewidth=6)
    label(600, 125, 'Reference masks / loss', size=30, color=MAGENTA, ha='left')

    label(60, 199, 'TRAIN', size=40, bold=True, ha='left')
    label(290, 199, 'Original CTs + digital siblings', size=36, ha='left')
    ax.plot([60, 1740], [225, 225], color='#d9e1e7', linewidth=2)

    ct = box(60, 250, 480, 230)
    label(300, 291, 'Training CT', size=40, bold=True, color=CYAN, area=ct)
    label(300, 344, 'Matched physical box', area=ct)
    label(300, 394, '1.5 mm isotropic', area=ct)
    label(300, 444, 'Official HU normalization', area=ct)

    network = box(640, 250, 540, 230)
    label(910, 291, 'PlainConvUNet', size=40, bold=True, area=network)
    label(910, 344, 'CT only · 19 output classes', area=network)
    label(910, 394, 'Strict pretrained loading', area=network)
    label(910, 444, 'All weights + all heads', area=network)

    prediction = box(1280, 250, 460, 230)
    label(1510, 291, 'Binary foreground', size=40, bold=True, color=CYAN, area=prediction)
    label(1510, 354, '19-class logits', area=prediction)
    label(1510, 407, 'Aorta + branches', area=prediction)
    arrow([(559, 365), (619, 365)])
    arrow([(1199, 365), (1259, 365)])

    masks = box(60, 530, 480, 155, MAGENTA, MAGENTA_FILL)
    label(300, 572, 'Reference masks', size=40, bold=True, color=MAGENTA, area=masks)
    label(300, 621, 'Aorta + branches', area=masks)
    label(300, 660, 'Matched grid + validity', size=28, area=masks)

    loss = box(1280, 530, 460, 155, MAGENTA, MAGENTA_FILL)
    label(1510, 572, 'Masked BCE + Dice', size=40, bold=True, color=MAGENTA, area=loss)
    label(1510, 621, 'Deep supervision', area=loss)
    label(1510, 660, 'Annotated voxels only', size=28, area=loss)
    arrow([(559, 606), (1259, 606)], MAGENTA)
    label(910, 564, 'Supervision only', size=36, color=MAGENTA)
    arrow([(1510, 490), (1510, 520)])
    arrow([(1280, 542), (1230, 505), (1125, 480)], MUTED, width=3)
    label(887, 514, 'Update network weights', size=32, color=MUTED)

    label(900, 735, 'Fresh AdamW · lr 1e−4 · batch 1 · 128³ patches · 2,000 updates', size=36)
    label(900, 781, 'Fixed training budget; no held-out checkpoint selection or early stopping.', size=28, color=MUTED)

    ax.plot([60, 1740], [823, 823], color='#d9e1e7', linewidth=2)
    label(60, 859, 'HELD-OUT EVALUATION', size=40, bold=True, ha='left')
    label(1740, 859, 'Originals + siblings', size=36, ha='right')

    heldout = box(60, 916, 440, 141)
    label(280, 958, 'Held-out CT', size=40, bold=True, color=CYAN, area=heldout)
    label(280, 1017, 'Same box + scaling', area=heldout)

    pretrained = box(610, 908, 430, 68)
    label(825, 942, 'Pretrained', size=38, bold=True, area=pretrained)
    tuned = box(610, 1008, 430, 68)
    label(825, 1042, 'Fine-tuned', size=38, bold=True, area=tuned)
    ax.plot([524, 555, 555], [987, 987, 942], color=CYAN, linewidth=3.8)
    ax.plot([555, 555], [987, 1042], color=CYAN, linewidth=3.8)
    arrow([(555, 942), (589, 942)])
    arrow([(555, 1042), (589, 1042)])
    ax.plot([1060, 1100, 1100], [942, 942, 987], color=CYAN, linewidth=3.8)
    ax.plot([1060, 1100, 1100], [1042, 1042, 987], color=CYAN, linewidth=3.8)
    arrow([(1100, 987), (1149, 987)])

    inference = box(1170, 916, 570, 141)
    label(1455, 954, 'Identical inference', size=40, bold=True, color=CYAN, area=inference)
    label(1455, 1011, 'Same tiling + class union', area=inference)

    heldout_truth = box(60, 1120, 440, 85, MAGENTA, MAGENTA_FILL)
    label(280, 1162, 'Reference mask', size=38, bold=True, color=MAGENTA, area=heldout_truth)
    metric = box(1170, 1120, 570, 85, MAGENTA, MAGENTA_FILL)
    label(1455, 1162, 'Dice vs. reference', size=40, bold=True, area=metric)
    arrow([(519, 1162), (1149, 1162)], MAGENTA)
    label(835, 1117, 'Evaluation only', size=28, color=MAGENTA)
    arrow([(1455, 1076), (1455, 1100)])
    label(900, 1250, 'Region-assisted development comparison · Reference masks never enter the network.', size=28, color=MUTED)

    # Geometric validation protects poster readability without testing runtime.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    violations = []
    for artist, area in text_checks:
        extent = artist.get_window_extent(renderer)
        if extent.x0 < 20 or extent.x1 > WIDTH - 20 or extent.y0 < 8 or extent.y1 > HEIGHT - 8:
            violations.append(f'Canvas overflow: {artist.get_text()}')
        if area:
            corners = ax.transData.transform([(area[0], area[1]), (area[2], area[3])])
            x0, x1 = sorted(corners[:, 0]); y0, y1 = sorted(corners[:, 1])
            if extent.x0 < x0 or extent.x1 > x1 or extent.y0 < y0 or extent.y1 > y1:
                violations.append(f'Box overflow: {artist.get_text()}')
    if violations:
        plt.close(fig)
        raise ValueError('\n'.join(violations))

    out.mkdir(parents=True, exist_ok=False)
    for extension in ('svg', 'pdf', 'png'):
        fig.savefig(out / f'{STEM}.{extension}', dpi=144, facecolor='white',
                    metadata={'Creator': 'FakeCT poster pipeline schematic'} if extension == 'pdf' else None)
    fig.savefig(out / f'{STEM}-preview.png', dpi=72, facecolor='white')
    plt.close(fig)

    svg_path = out / f'{STEM}.svg'
    svg = svg_path.read_text()
    root_start = svg.index('<svg')
    root_end = svg.index('>', root_start) + 1
    root_tag = svg[root_start:root_end]
    root_tag = re.sub(r'width="[^"]+"', f'width="{WIDTH}px"', root_tag, count=1)
    root_tag = re.sub(r'height="[^"]+"', f'height="{HEIGHT}px"', root_tag, count=1)
    svg = svg[:root_start] + root_tag + svg[root_end:]
    accessible_title = 'CT-only fine-tuning and held-out evaluation'
    accessible_description = (
        'Training CT is resampled and normalized before the strictly initialized 19-class PlainConvUNet. '
        'Separate reference masks feed only masked BCE and Dice with deep supervision. '
        'Held-out CT feeds pretrained and fine-tuned models under identical inference; reference masks feed only Dice evaluation.')
    root_end = svg.index('>', svg.index('<svg')) + 1
    embedded = ('\n <title>' + escape(accessible_title) + '</title>\n <desc>'
                + escape(accessible_description) + '</desc>\n <metadata id="fakect-method-provenance">'
                + escape(json.dumps(facts, ensure_ascii=False)) + '</metadata>')
    svg = svg[:root_end] + embedded + svg[root_end:]
    svg_path.write_text(svg)
    tree = ET.fromstring(svg)
    assert tree.get('width') == f'{WIDTH}px' and tree.get('height') == f'{HEIGHT}px'
    assert tree.get('viewBox') == f'0 0 {WIDTH} {HEIGHT}'
    native_text_count = len(tree.findall('.//{http://www.w3.org/2000/svg}text'))
    assert native_text_count == len(text_checks), 'Expected editable SVG text for every label.'

    caption = (
        'CT-only fine-tuning of the complete pretrained cardiac PlainConvUNet. '
        'Original and digital-sibling CTs use a matched physical box, 1.5 mm isotropic grid, '
        'and the original HU normalization. All pretrained weights and output heads are strictly loaded. '
        'Aorta and connected-branch reference masks are separate supervision, used only by a masked binary BCE + Dice loss '
        'with deep supervision. Fresh AdamW optimizes 2,000 batch-one 128³ patch updates per fold. '
        'All parameters are trainable; the coarsest default supervision weight is zero. '
        'The final fixed-budget checkpoint is used without held-out selection or early stopping. '
        'Held-out CT is processed identically by pretrained and fine-tuned models; a common 19-way argmax foreground union '
        'is scored with Dice against the corresponding native-grid reference. '
        'This is a region-assisted development comparison with provisional atlas-derived references.\n')
    (out / 'caption.txt').write_text(caption)
    facts['validation'] = {'native_svg_text_elements': native_text_count,
                           'text_within_canvas_and_boxes': True,
                           'source_configuration_checked': True,
                           'new_training_or_inference_run': False}
    facts['artifacts_sha256'] = {p.name: digest(p) for p in sorted(out.iterdir()) if p.is_file()}
    (out / 'provenance.json').write_text(json.dumps(facts, indent=2) + '\n')
    print(out / f'{STEM}.svg')
    print(out / f'{STEM}.png')
    print(out / f'{STEM}.pdf')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-dir', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    render(args.output_dir.resolve())


if __name__ == '__main__':
    main()
