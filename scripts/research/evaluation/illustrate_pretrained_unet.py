#!/usr/bin/env python3
"""Draw the verified Dataset293 3D U-net without constructing or running a model.

The complete 19-channel pretrained graph is preserved for grouped binary
supervision. Geometry is read from the source plans and checked against an
actual fine-tuning initialization receipt. SVG text stays editable and every
visible label is at least 24 px in the 1800 x 1310 SVG coordinate system.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import xml.etree.ElementTree as ET

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.patches import Circle, FancyArrowPatch, FancyBboxPatch, Polygon


ROOT = Path(__file__).resolve().parents[3]
MODEL = ROOT / ('outputs/pretrained/totalseg-2.18.0/nnunet/results/'
                'Dataset293_TotalSegmentator_part3_cardiac_1559subj/'
                'nnUNetTrainerNoMirroring__nnUNetPlans__3d_fullres')
STUDY = ROOT / 'outputs/experiments/aorta/pretrained-finetune-branches-r3'
PACKAGES = ROOT / 'outputs/environments/totalseg-r1/lib/python3.10/site-packages'
DEFAULT_OUT = ROOT / 'outputs/poster/bmes2026/unet-architecture-r1'
WIDTH, HEIGHT = 1800, 1310
INK, MUTED = '#183144', '#526879'
CYAN, MAGENTA = '#00a2bf', '#ce1199'
ENC_FILL, DEC_FILL = '#e8f9fc', '#fff0fa'


def sha256(path):
    return hashlib.sha256(path.read_bytes()).hexdigest()


def line_of(path, token):
    for i, line in enumerate(path.read_text().splitlines(), 1):
        if token in line:
            return i
    raise ValueError(f'Missing source evidence {token!r} in {path}')


def verify_sources():
    plans_path, labels_path = MODEL / 'plans.json', MODEL / 'dataset.json'
    init_path = STUDY / 'folds/xcat_reference_adult/siblings/initialization.json'
    source_path = ROOT / 'src/fakect_pretrained_learning.py'
    plans, dataset, initial = [json.loads(p.read_text())
                               for p in (plans_path, labels_path, init_path)]
    model_lock_path = ROOT / 'outputs/pretrained/totalseg-2.18.0/model-lock.json'
    model_lock = json.loads(model_lock_path.read_text())
    assert model_lock['versions']['TotalSegmentator'] == '2.18.0'
    assert model_lock['versions']['nnunetv2'] == '2.8.1'
    for source in (plans_path, labels_path):
        assert sha256(source) == model_lock['weights'][str(source)]
    checkpoint_path = str(MODEL / 'fold_0/checkpoint_final.pth')
    assert initial['source_checkpoint_sha256'] == model_lock['weights'][checkpoint_path]
    cfg = plans['configurations']['3d_fullres']
    n = len(cfg['n_conv_per_stage_encoder'])
    features = [min(cfg['UNet_base_num_features'] * 2 ** i,
                    cfg['unet_max_num_features']) for i in range(n)]
    grids = []
    shape = cfg['patch_size'][:]
    for stride in cfg['pool_op_kernel_sizes']:
        shape = [x // s for x, s in zip(shape, stride)]
        grids.append(shape[:])
    labels = dataset['labels']
    foreground = initial['foreground_channels']
    assert cfg['UNet_class_name'] == initial['architecture'] == 'PlainConvUNet'
    assert features == [32, 64, 128, 256, 320, 320]
    assert grids == [[r] * 3 for r in [128, 64, 32, 16, 8, 4]]
    assert cfg['n_conv_per_stage_encoder'] == [2] * 6
    assert cfg['n_conv_per_stage_decoder'] == [2] * 5
    assert cfg['conv_kernel_sizes'] == [[3, 3, 3]] * 6
    assert cfg['pool_op_kernel_sizes'] == [[1, 1, 1]] + [[2, 2, 2]] * 5
    assert cfg['spacing'] == initial['spacing_mm'] == [1.5] * 3
    assert cfg['patch_size'] == initial['checkpoint_patch_size'] == [128] * 3
    assert dataset['channel_names'] == {'0': 'CT'}
    assert set(labels.values()) == set(range(19))
    assert labels == initial['source_labels']
    assert foreground == [2, 4, 5, 6, 7, 8]
    assert [labels[k] for k in initial['foreground_names']] == foreground
    heads = []
    for index, channels in enumerate(features[-2::-1]):
        key = f'decoder.seg_layers.{index}.weight'
        expected = [19, channels, 1, 1, 1]
        assert initial['head_shapes'][key] == expected
        assert initial['head_shapes'][f'decoder.seg_layers.{index}.bias'] == [19]
        heads.append({'index': index, 'grid': grids[-2 - index],
                      'input_channels': channels, 'output_channels': 19,
                      'kernel': [1] * 3, 'weight_shape': expected})
    sources = [
        (model_lock_path, '"versions"', 'Pinned package versions and original source-file hashes'),
        (plans_path, '"3d_fullres"', 'Geometry, channels, stage counts and kernels'),
        (labels_path, '"labels"', 'All 19 source class labels'),
        (init_path, '"head_shapes"', 'Actual strict-loaded model and all five head shapes'),
        (source_path, 'def build_model(', 'Strict loading of the complete pretrained graph'),
        (source_path, 'def aorta_loss(', 'Grouped BCE + Dice and default deep-supervision weights'),
        (source_path, 'def predict_tiled(', 'Full-resolution 19-way inference and grouped extraction'),
        (PACKAGES / 'nnunetv2/utilities/plans_handling/plans_handler.py',
         '"features_per_stage"', 'Old plans conversion: capped doubling of features and normalization'),
        (PACKAGES / 'dynamic_network_architectures/building_blocks/plain_conv_encoder.py',
         "pool: str = 'conv'", 'Downsampling uses strided Conv3d'),
        (PACKAGES / 'dynamic_network_architectures/building_blocks/unet_decoder.py',
         'torch.cat((x, skips', 'ConvTranspose3d, skip concatenation, five retained heads'),
        (PACKAGES / 'dynamic_network_architectures/building_blocks/simple_conv_blocks.py',
         'class StackedConvBlocks', 'First convolution uses stage stride, second uses stride 1'),
        (STUDY / 'study-lock.json', '"foreground_channels"', 'Actual grouped target definition'),
    ]
    evidence = [{'path': str(p), 'relative_path': str(p.relative_to(ROOT)),
                 'sha256': sha256(p), 'line': line_of(p, token), 'supports': meaning}
                for p, token, meaning in sources]
    return dict(architecture='PlainConvUNet', configuration='3d_fullres',
                source_package_versions=model_lock['versions'],
                parameters=initial['parameters'], input_channels=1,
                output_channels=19, spacing_mm=cfg['spacing'], patch=cfg['patch_size'],
                encoder_features=features, encoder_grids=grids,
                encoder_strides=cfg['pool_op_kernel_sizes'],
                encoder_convolutions_per_stage=cfg['n_conv_per_stage_encoder'],
                decoder_convolutions_per_stage=cfg['n_conv_per_stage_decoder'],
                decoder_concat_channels=[2 * f for f in features[-2::-1]],
                decoder_features=features[-2::-1], decoder_heads=heads,
                normalization='InstanceNorm3d(eps=1e-5, affine=True)',
                activation='LeakyReLU(inplace=True; default negative_slope=0.01)',
                dropout=None, convolution_bias=True, convolution_padding=[1] * 3,
                downsampling='First 3x3x3 convolution in each E1..E5 stage has stride 2',
                upsampling='ConvTranspose3d(kernel_size=2, stride=2)',
                output_head='Conv3d(kernel_size=1, stride=1, bias=True)',
                loss='masked binary BCE + foreground Dice on grouped logits',
                deep_supervision_grids_fine_to_coarse=grids[:-1],
                deep_supervision_raw_weights_fine_to_coarse=[1, .5, .25, .125, 0],
                deep_supervision_normalized_weights_fine_to_coarse=[8/15, 4/15, 2/15, 1/15, 0],
                inference_deep_supervision=False, source_labels=labels,
                source_checkpoint_sha256=initial['source_checkpoint_sha256'],
                foreground_channels=foreground, foreground_names=initial['foreground_names'],
                probability=initial['probability'], extraction=initial['extraction'],
                binary_logit=initial['binary_logit'],
                coronary_scope=initial['scope_note'], evidence=evidence,
                verification='Metadata/source inspection only; no model instantiated or run')


def draw_figure(meta):
    plt.rcParams.update({'font.family': 'DejaVu Sans', 'svg.fonttype': 'none',
                         'pdf.fonttype': 42, 'font.size': 26})
    fig = plt.figure(figsize=(WIDTH / 72, HEIGHT / 72), dpi=72, facecolor='white')
    ax = fig.add_axes([0, 0, 1, 1])
    ax.set(xlim=(0, WIDTH), ylim=(HEIGHT, 0)); ax.axis('off')
    labels = []

    def text(x, y, label, size=26, bold=False, color=INK, align='center'):
        assert size >= 24
        item = ax.text(x, y, label, fontsize=size, fontweight='bold' if bold else 'normal',
                       color=color, ha=align, va='center', zorder=5)
        labels.append(item)
        return item

    def box(x, y, width, height, color, fill, radius=9):
        ax.add_patch(FancyBboxPatch((x, y), width, height,
            boxstyle=f'round,pad=0,rounding_size={radius}', linewidth=2.8,
            edgecolor=color, facecolor=fill, zorder=3))

    def tensor(x, center_y, width, color, fill, first, second):
        y, height, depth = center_y - 44, 88, 10
        ax.add_patch(Polygon([(x, y), (x+depth, y-depth), (x+width+depth, y-depth),
                             (x+width, y)], closed=True, facecolor=fill,
                             edgecolor=color, linewidth=2, zorder=2))
        ax.add_patch(Polygon([(x+width, y), (x+width+depth, y-depth),
                             (x+width+depth, y+height-depth), (x+width, y+height)],
                            closed=True, facecolor=fill, edgecolor=color, linewidth=2, zorder=2))
        box(x, y, width, height, color, fill, radius=5)
        text(x+width/2, center_y-17, first, size=30, bold=True)
        text(x+width/2, center_y+23, second, size=26)

    def arrow(points, color=INK, dashed=False, width=2.8, end=True):
        style = '--' if dashed else '-'
        for a, b in zip(points[:-2], points[1:-1]):
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                    linewidth=width, linestyle=style, zorder=1)
        if end:
            ax.add_patch(FancyArrowPatch(points[-2], points[-1], arrowstyle='-|>',
                mutation_scale=20, color=color, linewidth=width, linestyle=style,
                shrinkA=0, shrinkB=0, zorder=2))
        else:
            a, b = points[-2:]
            ax.plot([a[0], b[0]], [a[1], b[1]], color=color,
                    linewidth=width, linestyle=style, zorder=1)

    text(900, 55, 'Fine-tuned 3D U-net architecture', size=50, bold=True)
    text(900, 113, 'TotalSegmentator cardiac model · PlainConvUNet · all 19 output channels retained', size=29)
    text(242, 177, 'ENCODER', size=33, bold=True, color=CYAN)
    text(1085, 177, 'DECODER', size=33, bold=True, color=MAGENTA)
    text(1490, 177, 'OUTPUT HEADS', size=33, bold=True)
    box(55, 215, 365, 61, INK, '#f2f6f8')
    text(237.5, 246, 'Input CT: 128³ × 1', size=29, bold=True)
    text(656, 224, '1.5 mm isotropic voxels', size=26, color=MUTED)
    text(656, 263, 'CT-normalized HU', size=26, color=MUTED)
    text(1085, 238, '↑ transposed convolution', size=24, color=MAGENTA)
    text(1490, 238, '1³ Conv3d → 19 logits', size=26)

    rows = [335, 460, 585, 710, 835, 960]
    encoder_x, encoder_w = 55, 365
    decoder_x, decoder_w = 905, 350
    head_x, head_w, concat_x = 1345, 330, 862
    arrow([(237.5, 276), (237.5, 281)], CYAN)
    for stage, (row, channels, grid) in enumerate(zip(rows, meta['encoder_features'], meta['encoder_grids'])):
        tensor(encoder_x, row, encoder_w, CYAN, ENC_FILL,
               f'E{stage} · {grid[0]}³', f'{channels} feature channels')
        if stage:
            arrow([(237.5, rows[stage-1]+44), (237.5, row-54)], CYAN)
            text(314, (rows[stage-1]+row)/2, 'stride 2', size=24, color=CYAN)

    # Rows run fine -> coarse; decoder indices run coarse -> fine.
    for scale, row in enumerate(rows[:-1]):
        features = meta['encoder_features'][scale]
        decoder_index = 4 - scale
        grid = meta['encoder_grids'][scale][0]
        arrow([(encoder_x+encoder_w+10, row), (concat_x-21, row)], '#7c939f')
        text(635, row-26, 'skip · concatenate', size=26, color=MUTED)
        ax.add_patch(Circle((concat_x, row), 20, facecolor='white', edgecolor=MAGENTA,
                            linewidth=2.8, zorder=4))
        text(concat_x, row, 'C', size=29, bold=True, color=MAGENTA)
        arrow([(concat_x+21, row), (decoder_x, row)], MAGENTA)
        tensor(decoder_x, row, decoder_w, MAGENTA, DEC_FILL,
               f'D{decoder_index} · {grid}³', f'concat {features*2} → {features}')
        if scale < 4:
            # Feature output from the next coarser decoder stage goes through
            # an upsampling convolution before joining the encoder skip.
            below = rows[scale+1]
            bend = (row+below)/2
            arrow([(1080, below-54), (1080, bend), (concat_x, bend), (concat_x, row+21)], MAGENTA)
            text(816, bend+8, '×2', size=26, bold=True, color=MAGENTA)
        dashed = scale > 0
        color = MUTED if scale == 4 else MAGENTA
        arrow([(decoder_x+decoder_w+10, row), (head_x, row)], color, dashed=dashed)
        fill = '#f4f6f7' if scale == 4 else DEC_FILL
        box(head_x, row-43, head_w, 86, color, fill)
        text(head_x+head_w/2, row-17, f'{grid}³ × 19 logits', size=28, bold=True)
        weights = ['1', '1/2', '1/4', '1/8', '0 (retained)']
        text(head_x+head_w/2, row+23, f'Loss weight {weights[scale]}', size=25, color=color)

    # U-shaped bottleneck-to-decoder route, with no skip from the bottleneck.
    arrow([(encoder_x+encoder_w+10, rows[-1]), (concat_x, rows[-1]),
           (concat_x, rows[-2]+21)], MAGENTA)
    text(642, 927, 'Bottleneck → upsample ×2', size=27, bold=True, color=MAGENTA)
    text(242, 1025, 'Six encoder stages', size=27, bold=True, color=CYAN)
    text(1080, 1025, 'Five decoder stages', size=27, bold=True, color=MAGENTA)

    # Only the full-resolution logits feed the inference output.
    arrow([(head_x+head_w, rows[0]), (1745, rows[0]), (1745, 1042),
           (1510, 1042), (1510, 1080)], INK)
    text(1510, 979, 'Full-resolution inference', size=26, bold=True)
    box(1300, 1080, 455, 195, INK, '#f0f6f8')
    text(1527.5, 1112, 'Aorta + arch branches', size=28, bold=True)
    text(1527.5, 1157, 'Group {2, 4, 5, 6, 7, 8}', size=29, bold=True)
    text(1527.5, 1200, 'Mask: 19-way argmax in group', size=24)
    text(1527.5, 1243, 'Probability: Σ group softmax', size=25)

    box(55, 1080, 1180, 195, '#b9c9d0', '#f7fafb')
    text(645, 1110, 'Fine-tuning: grouped binary supervision', size=30, bold=True)
    text(645, 1150, 'Every stage: 2 × (3³ Conv3d → InstanceNorm3d → LeakyReLU)', size=26)
    text(645, 1190, 'Down: 3³ conv, stride 2 · Up: 2³ transposed conv, stride 2', size=26)
    text(645, 1230, 'Masked BCE + foreground Dice · four active scales · five heads retained', size=26)

    # Verify rendered labels stay inside the canvas before producing artifacts.
    fig.canvas.draw()
    renderer = fig.canvas.get_renderer()
    bounds = []
    for label in labels:
        extent = label.get_window_extent(renderer)
        if extent.x0 < 0 or extent.x1 > WIDTH or extent.y0 < 0 or extent.y1 > HEIGHT:
            raise ValueError(f'Text outside canvas: {label.get_text()}')
        bounds.append(dict(text=label.get_text(), font_size_px=label.get_fontsize(),
                           x=label.get_position()[0], y=label.get_position()[1]))
    return fig, bounds


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output-directory', type=Path, default=DEFAULT_OUT)
    args = parser.parse_args()
    out = args.output_directory.resolve()
    if out.exists():
        raise FileExistsError('Use a new figure revision; existing artifacts are immutable')
    meta = verify_sources()
    fig, label_bounds = draw_figure(meta)
    out.mkdir(parents=True)
    stem = 'pretrained-unet-architecture'
    fig.savefig(out / f'{stem}.svg', facecolor='white')
    fig.savefig(out / f'{stem}.pdf', facecolor='white')
    fig.savefig(out / f'{stem}.png', dpi=144, facecolor='white')
    fig.savefig(out / f'{stem}-preview.png', dpi=72, facecolor='white')
    plt.close(fig)
    svg_path = out / f'{stem}.svg'
    svg = svg_path.read_text()
    svg = svg.replace('width="1800pt" height="1310pt"', 'width="1800px" height="1310px"')
    # Descriptive provenance is machine readable without a dense visual footnote.
    svg = svg.replace('<defs>', '<metadata id="fakect-provenance">' +
                      json.dumps({'architecture': meta['architecture'],
                                  'configuration': meta['configuration'],
                                  'evidence': meta['evidence'],
                                  'coronary_scope': meta['coronary_scope']}).replace('&', '&amp;').replace('<', '&lt;') +
                      '</metadata>\n <defs>', 1)
    svg_path.write_text(svg)
    xml = ET.fromstring(svg)
    svg_text = xml.findall('.//{http://www.w3.org/2000/svg}text')
    assert len(svg_text) == len(label_bounds)
    assert min(p['font_size_px'] for p in label_bounds) >= 24
    assert 'viewBox="0 0 1800 1310"' in svg
    # Validate each SVG text style also retained the requested font size.
    sizes = [float(re.search(r'([\d.]+)px', t.attrib['style']).group(1))
             for t in svg_text]
    assert min(sizes) >= 24
    receipt = dict(schema_version='fakect.pretrained-unet-figure/1', **meta,
                   canvas_px=[WIDTH, HEIGHT], minimum_font_size_px=24,
                   all_svg_text_editable=True, label_count=len(label_bounds),
                   generator_path=str(Path(__file__).resolve()),
                   generator_sha256=sha256(Path(__file__)),
                   output_sha256={p.name:sha256(p) for p in sorted(out.iterdir()) if p.is_file()})
    (out / 'provenance.json').write_text(json.dumps(receipt, indent=2)+'\n')
    (out / 'text-layout.json').write_text(json.dumps(label_bounds, indent=2)+'\n')
    print(out / f'{stem}.svg')
    print(out / f'{stem}-preview.png')


if __name__ == '__main__':
    main()
