#!/usr/bin/env python3
"""Preview, plan, prepare, or fit an explicitly staged FakeCT segmentation study."""
import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts'))
from fakect_study_config import STAGES, load_study_config
from fakect_roi import digest, resolve_preview
from fakect_training_data import (IMAGE_WARNING, SPLIT_WARNING, dataset_fingerprint,
                                  prepare_training_dataset, validate_dataset, validate_study_plan)
from preview_roi import json_value


def study_plan(config, resolved, variants):
    shape = [int(b-a) for a, b in zip(resolved['crop_low_ijk'], resolved['crop_high_ijk_exclusive'])][::-1]
    return json_value({
        'schema_version': 'fakect.study-plan/1', 'stage': config['train']['stage'],
        'dataset_prepared': False, 'model_fitted': False,
        'case_id': config['input']['case_id'], 'frame': config['input']['frame'],
        'anatomy_family': config['train']['anatomy_family'],
        'target_source_ids': config['train']['target_source_ids'],
        'target_definition': 'Original target ID membership across the entire exported crop after editing; never clipped to ROI',
        'crop_shape_kji': shape, 'spacing_ijk_mm': resolved['spacing_ijk_mm'],
        'dataset_fingerprint': dataset_fingerprint(config, resolved),
        'variant_count': len(variants), 'variants': variants,
        'planned_split_counts': dict(Counter(v['split'] for v in variants)),
        'split_assignment_status': 'Provisional: identical masks are grouped during preparation; baseline-equivalent masks stay in train',
        'image_method': config['train']['image_method'], 'image_warning': IMAGE_WARNING,
        'split_mode': config['train']['split_mode'], 'split_warning': SPLIT_WARNING,
        'model': config['model'], 'dataset_directory': config['train']['dataset_directory'],
        'model_directory': config['train']['model_directory'],
    })


def run(config_path, stage=None, validate_only=False):
    config_path = Path(config_path).expanduser().resolve()
    input_bytes = config_path.read_bytes()
    config = load_study_config(config_path)
    configured_stage = config['train']['stage']
    if config_path.read_bytes() != input_bytes:
        raise ValueError('Input changed while being parsed; rerun from the saved file')
    if stage is not None:
        if stage not in STAGES:
            raise ValueError('Unsupported study stage')
        config['train']['stage'] = stage
    stage = config['train']['stage']
    resolved = resolve_preview(config)
    variants = validate_study_plan(config, resolved)
    plan = study_plan(config, resolved, variants)
    plan['input_config_sha256'] = hashlib.sha256(input_bytes).hexdigest()
    plan['stage_override'] = stage if stage != configured_stage else None
    if config_path.read_bytes() != input_bytes:
        raise ValueError('Input changed during validation; rerun from the saved file')
    if validate_only:
        result = {'validated': True, 'stage': stage, 'variant_count': len(variants),
                  'planned_split_counts': plan['planned_split_counts'], 'crop_shape_kji': plan['crop_shape_kji'],
                  'outputs_written': False, 'voxel_payloads_read': False}
    elif stage == 'preview':
        from preview_roi import run as preview
        return preview(config_path, config_override=config, training_plan=plan, expected_input_bytes=input_bytes,
                       extra_code_files=[Path(__file__), ROOT/'src/fakect_study_config.py',
                                         ROOT/'src/fakect_study_preview.py', ROOT/'src/fakect_training_data.py',
                                         ROOT/'src/fakect_segmentation.py'])
    elif stage == 'plan':
        output = config['output']['directory']
        if output.exists() and (not output.is_dir() or any(output.iterdir())):
            raise FileExistsError('Plan output directory is populated; choose a new output.directory')
        output.mkdir(parents=True, exist_ok=True)
        plan['generated_at_utc'] = datetime.now(timezone.utc).isoformat()
        plan['code_sha256'] = {str(path.relative_to(ROOT)): digest(path) for path in
                              [Path(__file__), ROOT/'src/fakect_study_config.py', ROOT/'src/fakect_config.py']}
        (output/'input.ini').write_bytes(input_bytes)
        (output/'resolved-config.json').write_text(json.dumps(json_value(config), indent=2)+'\n')
        (output/'plan.json').write_text(json.dumps(plan, indent=2, allow_nan=False)+'\n')
        inventory = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
        (output/'artifact-manifest.json').write_text(json.dumps(inventory, indent=2)+'\n')
        result = {'stage': stage, 'plan': str(output/'plan.json'), 'variant_count': len(variants),
                  'planned_split_counts': plan['planned_split_counts'], 'voxel_payloads_read': False}
    elif stage == 'prepare':
        result = prepare_training_dataset(config, resolved, input_bytes=input_bytes)
        result = {key: value for key, value in result.items() if key != 'samples'}
        result['stage'] = stage
    else:
        manifest_path = config['train']['dataset_directory']/'dataset-manifest.json'
        validate_dataset(config, resolved, manifest_path)
        from fakect_segmentation import train_segmentation
        metadata = train_segmentation(config, manifest_path)
        output = config['train']['model_directory']
        (output/'input.ini').write_bytes(input_bytes)
        (output/'resolved-config.json').write_text(json.dumps(json_value(config), indent=2)+'\n')
        inventory = {path.name: digest(path) for path in output.iterdir() if path.is_file()}
        (output/'artifact-manifest.json').write_text(json.dumps(inventory, indent=2)+'\n')
        result = {'stage': stage, 'model_directory': str(output), 'best_epoch': metadata['best_epoch'],
                  'best_validation_loss': metadata['best_validation_loss'], 'test_evaluated': False}
    print(json.dumps(json_value(result), indent=2, allow_nan=False), flush=True)
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--stage', choices=sorted(STAGES), help='Explicitly override train.stage for this run')
    parser.add_argument('--validate-only', action='store_true', help='Check metadata and all planned edit bounds without reading voxels or writing outputs')
    args = parser.parse_args()
    try:
        run(args.config, args.stage, args.validate_only)
    except (ValueError, OSError, KeyError, ImportError) as exc:
        parser.exit(2, f'error: {exc}\n')


if __name__ == '__main__':
    main()
