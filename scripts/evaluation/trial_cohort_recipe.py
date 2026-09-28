#!/usr/bin/env python3
"""Run only the minimum and maximum sweep corners for manual recipe diagnosis.

This is a diagnostic subset, never a native preflight or a freeze prerequisite.
The maximum trial retains native before/after arrays for geometry inspection.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_study_config import load_study_config
from fakect_roi import resolve_preview
from fakect_recipe import apply_recipe
from fakect_recipe_study_config import plan_recipe_variants, variant_recipe_config
from fakect_recipe_training import _source, _checked_result, _step_outcomes, _fingerprint, recipe_provenance
from fakect_training_data import (_array_hash, _file_hash, _geometry, _canonical,
                                  _json_value, _write_anatomical_selection_snapshot)

SCHEMA = 'fakect.recipe-trial/1'


def select_extremes(config, variants):
    """Select complete all-axis corners; numeric values need not be ordered."""
    axes = [(name, parameter, values) for name, parameters in sorted(config['sweeps'].items())
            for parameter, values in sorted(parameters.items())]
    edited = [row for row in variants if row['operation'] != 'none']
    result = []
    for title, extreme in (('minimum', min), ('maximum', max)):
        wanted = {(name, parameter): extreme(values) for name, parameter, values in axes}
        matches = [row for row in edited if all(row['parameters'].get(name, {}).get(parameter) == value
                                               for (name, parameter), value in wanted.items())]
        if len(matches) != 1:
            raise ValueError(f'Expected exactly one {title} all-axis recipe corner')
        if result and result[0]['variant']['variant_id'] == matches[0]['variant_id']:
            result[0]['extreme'] = 'minimum_and_maximum'
        else:
            result.append({'extreme': title, 'variant': matches[0]})
    return result


def _write_json(path, value):
    Path(path).write_text(json.dumps(_json_value(value), indent=2, allow_nan=False) + '\n')


def run(config_path, output, *, repo_root=ROOT):
    config_path = Path(config_path).expanduser().resolve()
    output = Path(output).expanduser().resolve()
    input_bytes = config_path.read_bytes()
    output.mkdir(parents=True, exist_ok=False)
    (output / 'INCOMPLETE').write_text('Diagnostic recipe trial incomplete; this is not a preflight or paired dataset.\n')
    (output / 'input.ini').write_bytes(input_bytes)
    document = {'schema_version': SCHEMA, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'complete': False, 'execution_valid': False, 'full_parameter_space_checked': False,
        'freeze_eligible': False, 'dataset_prepared': False, 'model_fitted': False,
        'input_config_path': str(config_path), 'input_config_sha256': hashlib.sha256(input_bytes).hexdigest(),
        'trial_script_sha256': _file_hash(__file__), 'samples': [],
        'scope': 'Only the all-axis minimum and maximum recipe corners, independently from the same original crop. No full-grid geometry validation. This diagnostic schema cannot substitute for fakect.recipe-preflight/1 or approve freezing.'}
    try:
        config = load_study_config(config_path, repo_root=repo_root)
        if config['study']['schema_version'] != 'fakect.recipe-cohort/1':
            raise ValueError('Trial input must be a generation-only fakect.recipe-cohort/1 INI')
        if config_path.read_bytes() != input_bytes:
            raise ValueError('INI changed while being parsed')
        resolved = resolve_preview(config)
        variants = plan_recipe_variants(config)
        selected = select_extremes(config, variants)
        fingerprint = _fingerprint(config, resolved)
        document.update(case_id=config['input']['case_id'], frame=config['input']['frame'],
            dataset_fingerprint=fingerprint, planned_variant_count=len(variants),
            selected_variant_ids=[row['variant']['variant_id'] for row in selected],
            trial_count=len(selected), candidate_source_ids=list(resolved['source_ids']),
            geometry=_geometry(resolved), variants_start_from_original=True)
        _write_json(output / 'resolved-config.json', config)
        _write_json(output / 'data-config.json', recipe_provenance(config, resolved))
        _write_anatomical_selection_snapshot(output, resolved)
        # _source calls prepare_crop once, validates ordinary attenuation targets,
        # and marks the shared source arrays read-only before independent trials.
        arrays, sources = _source(config, resolved)
        hashes = {key: _array_hash(arrays[key]) for key in ('act', 'atn', 'selected', 'roi')}
        document.update(source_crop_sha256=hashes, source_files=sources,
            source_array_order='kji', source_dtypes={key: str(arrays[key].dtype) for key in hashes})
        _write_json(output / 'trial.json', document)
        baseline_hash = _array_hash(arrays['selected'].astype(np.uint8))
        for item in selected:
            variant = item['variant']
            record = {**variant, 'extreme': item['extreme']}
            result = None
            try:
                result = apply_recipe(arrays, resolved, variant_recipe_config(config, variant))
                image, mask = _checked_result(arrays, resolved, result)
                record.update(status='ok', mask_sha256=_array_hash(mask), image_sha256=_array_hash(image),
                    foreground_voxels=int(mask.sum()), changed_voxels=int(result['changed_mask'].sum()),
                    scalar_changed_voxels=int(result['scalar_changed_mask'].sum()),
                    zero_change=not bool(result['changed_mask'].any()),
                    baseline_equivalent=_array_hash(mask) == baseline_hash,
                    step_outcomes=_step_outcomes(result['summary']), morphology=result['summary'],
                    geometry=document['geometry'], dataset_fingerprint=fingerprint)
                if item['extreme'] in ('maximum', 'minimum_and_maximum'):
                    payload = output / 'max-trial.npz'
                    with payload.open('xb') as stream:
                        np.savez_compressed(stream,
                            before_labels=arrays['act'], before_attenuation_per_pixel=arrays['atn'],
                            before_selected=arrays['selected'], before_roi=arrays['roi'],
                            after_labels=result['edited_labels'],
                            after_attenuation_per_pixel=result['attenuation_proxy_per_pixel'],
                            after_target=result['target_mask_after'], changed_mask=result['changed_mask'],
                            scalar_changed_mask=result['scalar_changed_mask'],
                            edit_region_mask=result['edit_region_mask'],
                            geometry_json=np.array(_canonical(document['geometry'])),
                            trial_metadata_json=np.array(_canonical({'schema_version': SCHEMA,
                                'variant_id': variant['variant_id'], 'extreme': item['extreme'],
                                'dataset_fingerprint': fingerprint, 'source_crop_sha256': hashes})))
                    record.update(payload=payload.name, payload_sha256=_file_hash(payload))
                del image, mask, result
            except (ValueError, RuntimeError, OSError) as error:
                record.update(status='failed', error_type=type(error).__name__, error=str(error))
            finally:
                result = None
            document['samples'].append(record)
            _write_json(output / 'trial.json', document)
            print(f"Recipe trial {item['extreme']}: {variant['variant_id']}; {record['status']}", flush=True)
        if any(_array_hash(arrays[key]) != value for key, value in hashes.items()):
            raise ValueError('A trial modified the shared original source crop')
        if _fingerprint(config, resolved) != fingerprint:
            raise ValueError('Data-generating inputs or code changed during diagnostic trials')
        if config_path.read_bytes() != input_bytes:
            raise ValueError('INI changed during diagnostic trials; snapshots describe the original input bytes')
        document['failed_trials'] = sum(row['status'] != 'ok' for row in document['samples'])
        if document['failed_trials']:
            raise ValueError(f"{document['failed_trials']} diagnostic trial(s) failed; inspect {output / 'trial.json'}")
        document.update(complete=True, execution_valid=True, source_arrays_unchanged=True)
        _write_json(output / 'trial.json', document)
        (output / 'INCOMPLETE').unlink()
        return document
    except Exception as error:
        document.update(complete=False, execution_valid=False, error_type=type(error).__name__, error=str(error))
        _write_json(output / 'trial.json', document)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='New, non-existing diagnostic directory')
    args = parser.parse_args()
    try:
        result = run(args.config, args.output)
    except (ValueError, RuntimeError, OSError, KeyError, TypeError) as error:
        parser.exit(2, 'error: ' + str(error) + '\n')
    print(json.dumps({key: result[key] for key in ('case_id', 'planned_variant_count', 'trial_count',
        'selected_variant_ids', 'complete', 'full_parameter_space_checked', 'freeze_eligible')}, indent=2))


if __name__ == '__main__':
    main()
