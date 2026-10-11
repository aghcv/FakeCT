#!/usr/bin/env python3
"""Build an offline native-grid 3D review of atlas masks and draft edit pairs."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
from fakect_mask_pair_preview import sha256
from fakect_two_stage_3d import build_scene, render_viewer


def verified_json(path, schema):
    path = Path(path).resolve()
    digest = sha256(path)
    if path.with_suffix('.json.sha256').read_text().strip() != digest:
        raise ValueError(f'Manifest checksum changed: {path}')
    document = json.loads(path.read_text())
    if document.get('schema_version') != schema:
        raise ValueError(f'Unexpected manifest schema: {path}')
    return document, digest


def layer(key, label, color, opacity=.8, visible=True):
    return dict(key=key, label=label, color=color, opacity=opacity, visible=visible)


def scenes(selection_path, pairs_path=None):
    selection, selection_hash = verified_json(selection_path, 'fakect.phantom-anatomy/1')
    if not selection.get('complete'):
        raise ValueError('Automatic selection is incomplete')
    mask_path = Path(selection_path).resolve().parent/selection['mask_npz']
    if sha256(mask_path) != selection['mask_npz_sha256']:
        raise ValueError('Automatic mask checksum changed')
    with np.load(mask_path, allow_pickle=False) as data:
        masks = {'before': data['mask'], 'core': data['core_mask'], 'branches': data['branch_mask']}
        spacing = data['spacing_ijk_mm']
        origin = data['crop_origin_ijk']
    if (np.any(masks['core'] & masks['branches']) or
            not np.array_equal(masks['before'], masks['core'] | masks['branches'])):
        raise ValueError('Automatic core/branch masks disagree with the combined mask')
    limits = selection['z_slab_mm']
    explanation = (f"Atlas extraction without an input ROI. Native crop origin (i,j,k): {origin.tolist()}. "
                   f"RAW-z slab {limits[0]:.2f}–{limits[1]:.2f} mm; branch reach "
                   f"{selection['branch_reach_mm']:.2f} mm from the aorta core. "
                   "Review the distal cut and proximal branches. No model prediction is shown.")
    yield build_scene(masks, spacing, sample_id=f"{selection['case_id']}:atlas-anatomy", modes=[{
        'key': 'anatomy', 'label': 'Automatic anatomical selection', 'explanation': explanation,
        'layers': [layer('core', 'Aortic core', '#d24a39', .45),
                   layer('branches', 'Proximal connected branches', '#00a69a'),
                   layer('before', 'Combined anatomical mask', '#76889e', .15, False)]}],
        metadata={'case_id': selection['case_id'], 'is_control': True,
                  'selection_sha256': selection_hash, 'crop_origin_ijk': origin.tolist()})
    if pairs_path is None:
        return
    pairs, _ = verified_json(pairs_path, 'fakect.mask-pair-preview/1')
    if (pairs['case_id'] != selection['case_id'] or
            pairs['source_anatomy_sha256'] != selection['mask_npz_sha256'] or
            pairs['source_selection_manifest_sha256'] != selection_hash or
            Path(pairs['source_anatomy_npz']).resolve() != mask_path.resolve()):
        raise ValueError('Draft pairs refer to a different anatomical selection')
    for row in pairs['samples']:
        if row['status'] != 'ready_for_visual_review':
            continue  # All failures remain explicit in the companion review record.
        path = Path(pairs_path).resolve().parent/row['path']
        if sha256(path) != row['sha256']:
            raise ValueError('Draft mask-pair checksum changed')
        with np.load(path, allow_pickle=False) as data:
            masks = {key: data[key] for key in ('before', 'after', 'added', 'removed')}
            spacing, origin = data['spacing_ijk_mm'], data['crop_origin_ijk']
            if str(data['sample_id']) != row['sample_id']:
                raise ValueError('Draft pair belongs to another sample')
        if (not np.array_equal(masks['added'], masks['after'] & ~masks['before']) or
                not np.array_equal(masks['removed'], masks['before'] & ~masks['after'])):
            raise ValueError('Draft pair change masks disagree with before/after')
        yield build_scene(masks, spacing, sample_id=row['sample_id'], modes=[{
            'key': 'pairs', 'label': 'Broader reference and synthetic edits',
            'explanation': (f"Review-only transferred edit pair. Native crop origin (i,j,k): {origin.tolist()}. "
                            "Existing edit deltas are applied to the automatic anatomical reference. "
                            "These are ground-truth masks, not model predictions; anatomical extent is provisional."),
            'layers': [layer('before', 'Original reference', '#76889e', .15),
                       layer('after', 'Edited observed mask', '#e8b74b', .45),
                       layer('added', 'True added voxels', '#d5006d'),
                       layer('removed', 'True removed voxels', '#006fd6')]}],
            metadata={'case_id': row['case_id'], 'is_control': row['baseline'],
                      'crop_origin_ijk': origin.tolist(), 'counts': row['counts']})


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--selection', required=True, type=Path)
    parser.add_argument('--pairs', type=Path)
    parser.add_argument('--output', required=True, type=Path)
    args = parser.parse_args(argv)
    try:
        args.output.mkdir(parents=True, exist_ok=False)
        pending = args.output/'INCOMPLETE'
        pending.write_text('Building native-mask 3D review\n')
        result = render_viewer(args.output, scenes(args.selection, args.pairs),
            title='Thoracic aorta · automatic anatomy and synthetic mask pairs',
            coordinate_description='native-grid mm from lower crop face; crop origin recorded per scene; orientation unverified',
            show_metrics=False,
            footer_note='Provisional atlas selection: T12 midpoint to T1 upper bound, with proximal connected arterial branches. Review anatomical coverage before training. Surfaces use native Boolean masks with crop-edge closures and no smoothing. These are reference/edited masks, not model predictions.')
        record = {'selection': str(args.selection.resolve()), 'selection_sha256': sha256(args.selection),
                  'pairs': str(args.pairs.resolve()) if args.pairs else None,
                  'pairs_sha256': sha256(args.pairs) if args.pairs else None,
                  'renderer_sha256': sha256(ROOT/'src/fakect_two_stage_3d.py'),
                  'review_script_sha256': sha256(__file__), **result}
        if args.pairs:
            pairs, _ = verified_json(args.pairs, 'fakect.mask-pair-preview/1')
            record['requires_review'] = [r for r in pairs['samples'] if r['status'] != 'ready_for_visual_review']
        (args.output/'review-provenance.json').write_text(json.dumps(record, indent=2)+'\n')
        pending.unlink()
        print(json.dumps(record, indent=2))
        return 0
    except (OSError, ValueError, KeyError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    raise SystemExit(main())
