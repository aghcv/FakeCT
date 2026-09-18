"""Execute independent ordered recipes as a reviewed synthetic cohort.

Native preflight measures every variant without exporting image/mask pairs.
Preparation repeats the same checks and publishes the existing training schema
only after native execution and geometry-group split checks succeed.
"""
import csv
import hashlib
import html
import io
import json
import math
from pathlib import Path
import zipfile

import numpy as np

from fakect_growth import lineage_selection
from fakect_recipe import apply_recipe
from fakect_recipe_study_config import (validate_recipe_study_plan,
                                        variant_recipe_config)
from fakect_released import RELEASED_LABEL_ID
from fakect_roi import prepare_crop
from fakect_training_data import (SCHEMA, IMAGE_WARNING, SPLIT_WARNING, _array_hash,
                                  _canonical, _file_hash, _geometry, _json_value,
                                  _write_json, _write_npz)


PREFLIGHT_SCHEMA = 'fakect.recipe-preflight/1'
TARGET_SEMANTICS = (
    'Surviving selected target ancestry and its offspring. The original target is '
    'selected tissue inside the main ROI; descendants outside that selector remain '
    'positive, while unrelated same-tissue anatomy stays negative.')
PREFLIGHT_WARNING = (
    'Native execution and split checks do not establish anatomical or clinical realism. '
    'Review the requested and achieved geometry before preparing a cohort.')
_TRAIN_DATA_KEYS = ('target_scope', 'anatomy_family', 'include_baseline', 'max_variants',
                    'image_method', 'split_mode', 'validation_fraction', 'test_fraction', 'split_seed')
_MODULES = ('fakect_recipe_training.py', 'fakect_recipe_study_config.py', 'fakect_training_data.py',
            'fakect_recipe.py', 'fakect_recipe_config.py', 'fakect_config.py', 'fakect_study_config.py',
            'fakect_morphology.py', 'fakect_erosion_guard.py', 'fakect_reassignment.py',
            'fakect_released.py', 'fakect_roi.py', 'fakect_tissues.py', 'fakect_growth.py',
            'fakect_direction.py', 'fakect_centerline_frame.py', 'fakect_tube_range.py')


def recipe_provenance(config, resolved):
    """Bind all data-generating recipe settings, source state and runtime code."""
    sources = {}
    for channel, filename in sorted(resolved['source_files'].items()):
        path = Path(filename).resolve()
        stat = path.stat()
        sources[channel] = {'path': str(path), 'bytes': stat.st_size, 'mtime_ns': stat.st_mtime_ns}
    for key in ('catalog', 'audit'):
        if _file_hash(config['input'][key]) != resolved[key + '_sha256']:
            raise ValueError(f'{key} changed after source resolution')
    metadata = {str(resolved['case'][key]): _file_hash(resolved['case'][key])
                for key in ('par_path', 'log_path') if key in resolved.get('case', {})}
    return {'study_schema': config['study']['schema_version'],
            **{key: config[key] for key in ('input', 'selection', 'roi', 'rois', 'recipe', 'edits', 'reassignment')},
            'sweeps': config.get('sweeps', {}), 'centerline': config.get('centerline'),
            'train': {key: config['train'][key] for key in _TRAIN_DATA_KEYS},
            'geometry': _geometry(resolved), 'candidate_source_ids': list(resolved['source_ids']),
            'target_scope': 'selected_lineage', 'target_semantics': TARGET_SEMANTICS,
            'catalog_sha256': resolved['catalog_sha256'], 'audit_sha256': resolved['audit_sha256'],
            'source_files': sources, 'source_metadata_sha256': metadata,
            'source_integrity_scope': 'File size/mtime plus saved crop hashes; full source volumes are not hashed',
            'code_sha256': {name: _file_hash(Path(__file__).with_name(name)) for name in _MODULES}}


def _fingerprint(config, resolved):
    return hashlib.sha256(_canonical(recipe_provenance(config, resolved)).encode()).hexdigest()


def _source(config, resolved):
    arrays, sources = prepare_crop(resolved, config)
    if np.any(arrays['act'] == RELEASED_LABEL_ID):
        raise ValueError('Diagnostic released voxels have unassigned attenuation and cannot form training pairs')
    if not arrays['selected'].any():
        raise ValueError('Selected-lineage training target is absent from the main ROI')
    # Catch unintended mutation immediately. Each recipe is responsible for its
    # own working copies, and every variant receives this identical snapshot.
    for value in arrays.values():
        if isinstance(value, np.ndarray):
            value.setflags(write=False)
    return arrays, sources


def _reserve(config, resolved, output, input_bytes, provenance, *, prepare):
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').write_text('Recipe cohort evaluation is incomplete; do not train from these files.\n')
    if input_bytes is not None:
        with (output/'input.ini').open('xb') as stream:
            stream.write(input_bytes)
    _write_json(output/'resolved-config.json', config)
    _write_json(output/'data-config.json', provenance)
    (output/'source-label-catalog.json').write_bytes(resolved['catalog_bytes'])
    (output/'source-audit.json').write_bytes(Path(config['input']['audit']).read_bytes())
    if prepare:
        (output/'variants').mkdir()


def _checked_result(arrays, resolved, result):
    labels = np.asarray(result['edited_labels'])
    proxy = np.asarray(result['attenuation_proxy_per_pixel'])
    mask = np.asarray(result['target_mask_after'])
    shape = arrays['act'].shape
    if labels.shape != shape or proxy.shape != shape or mask.shape != shape or mask.dtype.kind != 'b':
        raise ValueError('Recipe outputs do not share the native crop geometry')
    if np.any(labels == RELEASED_LABEL_ID) or np.any(result.get('attenuation_unassigned_mask', False)):
        raise ValueError('Diagnostic released voxels have unassigned attenuation and cannot form training pairs')
    if not np.all(np.isfinite(proxy)):
        raise ValueError('Recipe returned nonfinite attenuation')
    changed, scalar_changed = result['changed_mask'], result['scalar_changed_mask']
    if not np.array_equal(changed, labels != arrays['act']):
        raise ValueError('Recipe changed_mask disagrees with final versus original labels')
    if not np.array_equal(scalar_changed, proxy != arrays['atn']):
        raise ValueError('Recipe scalar_changed_mask disagrees with final versus original attenuation')
    if np.any((changed | scalar_changed) & ~result['edit_region_mask']):
        raise ValueError('Recipe changed a label/scalar outside its permitted edit region')
    if not np.array_equal(mask, lineage_selection(result['target_origin_index'], arrays['selected'])):
        raise ValueError('Recipe training target disagrees with original selected ancestry')
    if np.any(mask & ~np.isin(labels, resolved['source_ids'])):
        raise ValueError('Recipe lineage target includes a non-target label')
    image = np.asarray(proxy / (resolved['spacing_ijk_mm'][0] / 10), dtype=np.float32)
    if not np.all(np.isfinite(image)):
        raise ValueError('Nonfinite attenuation after inverse-centimetre conversion')
    return image, mask.astype(np.uint8)


def _step_outcomes(summary):
    rows = []
    for step in summary['steps']:
        guard = step.get('erosion_safeguard', {})
        planned = next((row for row in summary.get('planned_steps', [])
                        if row['name'] == step['step_name']), {})
        requested = guard.get('requested_distance_mm', planned.get('edit', {}).get('distance_mm', step['distance_mm']))
        rows.append({'step_name': step['step_name'], 'iteration': step['iteration'],
                     'operation': step.get('requested_operation', step['operation']),
                     'status': step['status'], 'requested_distance_mm': requested,
                     'accepted_distance_mm': guard.get('accepted_distance_mm',
                         0. if step['operation'] == 'none' else step['distance_mm']),
                     'added_voxels': step['counts']['added'], 'removed_voxels': step['counts']['removed'],
                     'proposed_added_voxels': step['counts']['proposed_added'],
                     'proposed_removed_voxels': step['counts']['proposed_removed'],
                     'blocked_voxels': step['counts']['blocked'], 'unresolved_voxels': step['counts']['unresolved'],
                     'erosion_safeguard': guard or None})
    return rows


def _assign_geometry_splits(samples, config, baseline_hash):
    """Assign whole realized geometry groups after every variant has run."""
    groups = {}
    for sample in samples:
        if sample['status'] == 'ok':
            groups.setdefault(sample['mask_sha256'], []).append(sample)
    seed = config['train']['split_seed']
    ranked = sorted((key for key in groups if key != baseline_hash),
                    key=lambda key: hashlib.sha256(f'{seed}:{key}'.encode()).digest())
    fractions = [config['train'][key] for key in ('validation_fraction', 'test_fraction')]
    required = sum(fraction > 0 for fraction in fractions)
    need_training = int(baseline_hash not in groups)
    group_splits = dict.fromkeys(groups, 'train')
    if len(ranked) >= required + need_training:
        counts = [max(1, math.floor(len(ranked)*fraction)) if fraction else 0 for fraction in fractions]
        capacity = len(ranked)-need_training
        while sum(counts) > capacity:
            index = max(range(2), key=lambda i: counts[i])
            counts[index] -= 1
        for key in ranked[:counts[0]]:
            group_splits[key] = 'validation'
        for key in ranked[counts[0]:sum(counts)]:
            group_splits[key] = 'test'
    for key, rows in groups.items():
        first = rows[0]['variant_id']
        for index, row in enumerate(rows):
            row.update(split=group_splits[key], duplicate_of=first if index else None)


def _split_diagnostics(samples, config):
    successful = [row for row in samples if row['status'] == 'ok']
    split_counts = {split: sum(row['split'] == split for row in successful)
                    for split in ('train', 'validation', 'test')}
    geometry_counts = {split: len({row['mask_sha256'] for row in successful if row['split'] == split})
                       for split in split_counts}
    errors = []
    if not geometry_counts['train']:
        errors.append('No distinct training geometry remains after native execution and grouping')
    for split, key in (('validation', 'validation_fraction'), ('test', 'test_fraction')):
        if config['train'][key] > 0 and not geometry_counts[split]:
            errors.append(f'No distinct {split} geometry remains after native execution and grouping')
    return {'split_counts': split_counts, 'geometry_group_counts': geometry_counts,
            'unique_geometry_count': len({row['mask_sha256'] for row in successful}),
            'duplicate_variant_count': sum(row.get('duplicate_of') is not None for row in successful),
            'baseline_equivalent_count': sum(row['baseline_equivalent'] for row in successful),
            'zero_change_count': sum(row['zero_change'] for row in successful),
            'split_errors': errors,
            'fit_ready': not errors and bool(geometry_counts['validation'])}


def _write_review(output, document):
    """Write inspectable summaries, including native failures and every pass."""
    _write_json(output/'preflight.json', document)
    with (output/'preflight.csv').open('x', newline='', encoding='utf-8') as stream:
        fields = ['variant_id', 'status', 'planned_split', 'split', 'duplicate_of',
                  'foreground_voxels', 'changed_voxels', 'scalar_changed_voxels',
                  'zero_change', 'step_name', 'iteration', 'operation',
                  'requested_distance_mm', 'accepted_distance_mm', 'added_voxels',
                  'removed_voxels', 'proposed_added_voxels', 'proposed_removed_voxels',
                  'blocked_voxels', 'unresolved_voxels', 'error']
        writer = csv.DictWriter(stream, fieldnames=fields, extrasaction='ignore')
        writer.writeheader()
        for sample in document['samples']:
            for step in sample.get('step_outcomes', []) or [{}]:
                writer.writerow({**sample, **{key: value for key, value in step.items() if key != 'status'}})
    escape = lambda value: html.escape(str(value))
    rows = []
    for sample in document['samples']:
        steps = '; '.join(f"{step['step_name']} #{step['iteration']}: {step['requested_distance_mm']:g} → "
                          f"{step['accepted_distance_mm']:g} mm, +{step['added_voxels']}/−{step['removed_voxels']}, "
                          f"{step['unresolved_voxels']} unresolved"
                          for step in sample.get('step_outcomes', []))
        values = [sample['variant_id'], sample['status'], sample.get('split', ''),
                  sample.get('foreground_voxels', ''), sample.get('changed_voxels', ''),
                  sample.get('duplicate_of') or '', steps or sample.get('error', '')]
        rows.append('<tr>' + ''.join('<td>' + escape(value) + '</td>' for value in values) + '</tr>')
    reasons = ''.join('<li>' + escape(error) + '</li>' for error in document['split_errors'])
    content = ('<!doctype html><html lang="en"><meta charset="utf-8"><title>Recipe cohort preflight</title>'
               '<style>body{font:16px system-ui,sans-serif;margin:2rem;color:#1d2630}table{border-collapse:collapse;width:100%}'
               'td,th{border:1px solid #ccd3da;padding:.55rem;text-align:left}th{background:#edf2f6}</style>'
               '<h1>Recipe cohort preflight</h1><p>' + escape(TARGET_SEMANTICS) + '</p><p>' +
               escape(IMAGE_WARNING) + ' ' + escape(SPLIT_WARNING) + '</p><p>' + escape(PREFLIGHT_WARNING) + '</p>'
               f"<p>{document['successful_variants']} successful variants; {document['failed_variants']} failures; "
               f"{document['unique_geometry_count']} distinct target geometries. "
               f"Training readiness: {'ready' if document['fit_ready'] and not document['failed_variants'] else 'not ready'}.</p>"
               '<ul>' + reasons + '</ul><table><thead><tr>' + ''.join('<th>'+title+'</th>' for title in
                 ['Variant', 'Execution', 'Actual split', 'Target voxels', 'Changed labels', 'Duplicate of',
                  'Requested → accepted distance; additions/removals']) + '</tr></thead><tbody>' +
               ''.join(rows) + '</tbody></table><p>preflight.json contains full native checks and safeguard attempts; '
               'preflight.csv contains one row per variant/pass. Accepted distance is an edit budget, '
               'not an achieved displacement. Unresolved releases retain their original target label.</p></html>')
    (output/'preflight.html').write_text(content, encoding='utf-8')


def _run(config, resolved, input_bytes, *, prepare):
    if input_bytes is not None and not isinstance(input_bytes, bytes):
        raise ValueError('input_bytes must contain the exact captured input bytes')
    variants = validate_recipe_study_plan(config, resolved)
    provenance = recipe_provenance(config, resolved)
    fingerprint = hashlib.sha256(_canonical(provenance).encode()).hexdigest()
    key = 'dataset_directory' if prepare else 'preflight_directory'
    output = Path(config['train'][key]).expanduser().resolve()
    if output.exists():
        raise FileExistsError(f'Recipe {key} exists; choose a new directory')
    arrays, sources = _source(config, resolved)
    original_hashes = {key: _array_hash(arrays[key]) for key in ('act', 'atn', 'selected')}
    original_mask = arrays['selected'].astype(np.uint8)
    baseline_hash = _array_hash(original_mask)
    geometry = _geometry(resolved)
    _reserve(config, resolved, output, input_bytes, provenance, prepare=prepare)
    snapshot = {'source_crop_sha256': original_hashes, 'array_order': 'kji',
                'shape_kji': list(arrays['act'].shape),
                'dtypes': {key: str(arrays[key].dtype) for key in original_hashes},
                'semantics': 'SHA256 of contiguous native crop array bytes; selected is the original main-ROI lineage target.'}
    _write_json(output/'source-crop.json', snapshot)
    if prepare:
        _write_npz(output/'source.npz', original_labels=arrays['act'],
                   original_attenuation_per_pixel=arrays['atn'], original_tissue_labels=arrays['tissue'],
                   roi_mask=arrays['roi'], target_mask=original_mask,
                   geometry_json=np.array(_canonical(geometry)),
                   catalog_json=np.array(resolved['catalog_bytes'].decode('utf-8')))
    samples = []
    for index, variant in enumerate(variants, 1):
        record = {**variant, 'planned_split': variant['split']}
        result = None
        try:
            result = apply_recipe(arrays, resolved, variant_recipe_config(config, variant))
            image, mask = _checked_result(arrays, resolved, result)
            mask_hash = _array_hash(mask)
            record.update(status='ok', split=variant['split'], duplicate_of=None,
                geometry_group=f"{config['train']['anatomy_family']}:{mask_hash}",
                mask_sha256=mask_hash, image_sha256=_array_hash(image),
                baseline_equivalent=mask_hash == baseline_hash, foreground_voxels=int(mask.sum()),
                changed_voxels=int(result['changed_mask'].sum()),
                scalar_changed_voxels=int(result['scalar_changed_mask'].sum()),
                zero_change=not bool(result['changed_mask'].any()),
                step_outcomes=_step_outcomes(result['summary']),
                image_method='attenuation_copy_proxy', image_units='cm^-1', image_warning=IMAGE_WARNING,
                target_scope='selected_lineage', mask_semantics=TARGET_SEMANTICS,
                geometry=geometry, dataset_fingerprint=fingerprint, morphology=result['summary'],
                label_checks={'unassigned_voxels': 0, 'target_matches_selected_lineage': True,
                              'changes_within_edit_region': True, 'source_arrays_unchanged': True})
            if prepare:
                path = output/'variants'/(variant['variant_id']+'.npz')
                _write_npz(path, image=image, mask=mask, edited_labels=result['edited_labels'],
                           changed_mask=result['changed_mask'], scalar_changed_mask=result['scalar_changed_mask'],
                           target_origin_index=result['target_origin_index'], edit_region_mask=result['edit_region_mask'])
                record.update(path=str(path.relative_to(output)))
            del image, mask
        except (ValueError, RuntimeError, OSError) as exc:
            record.update(status='failed', error_type=type(exc).__name__, error=str(exc))
        finally:
            del result
        samples.append(record)
        print(f"Recipe cohort {index}/{len(variants)}: {variant['variant_id']}; {record['status']}; "
              + (f"changed={record['changed_voxels']}, target={record['foreground_voxels']}"
                 if record['status'] == 'ok' else record['error']), flush=True)
    if any(_array_hash(arrays[key]) != value for key, value in original_hashes.items()):
        raise ValueError('Recipe modified its immutable shared source snapshot')
    if _fingerprint(config, resolved) != fingerprint:
        raise ValueError('Data-generating inputs or code changed during recipe cohort execution; partial outputs retained')
    _assign_geometry_splits(samples, config, baseline_hash)
    if prepare:
        for record in samples:
            if record['status'] != 'ok':
                continue
            # Append the one final metadata member after native geometry has
            # determined the split. Image arrays need not be recompressed.
            data = io.BytesIO()
            np.lib.format.write_array(data, np.array(_canonical(record)), allow_pickle=False)
            path = output/record['path']
            with zipfile.ZipFile(path, 'a', compression=zipfile.ZIP_DEFLATED) as archive:
                archive.writestr('metadata_json.npy', data.getvalue())
            record['sha256'] = _file_hash(path)
    split = _split_diagnostics(samples, config)
    failed = sum(sample['status'] != 'ok' for sample in samples)
    document = {'schema_version': PREFLIGHT_SCHEMA, 'complete': True, 'execution_complete': True,
                'execution_valid': failed == 0, 'dataset_prepared': False, 'model_fitted': False,
                'dataset_fingerprint': fingerprint, 'target_scope': 'selected_lineage',
                'target_semantics': TARGET_SEMANTICS, 'source_files': sources,
                'source_crop_sha256': original_hashes,
                'successful_variants': len(samples)-failed, 'failed_variants': failed,
                'variant_count': len(samples), 'samples': samples, **split,
                'input_config_sha256': hashlib.sha256(input_bytes).hexdigest() if input_bytes is not None else None,
                'split_assignment_semantics': 'After native execution, distinct non-baseline target-mask hashes are ranked deterministically using split_seed. Identical geometry stays together; baseline-equivalent geometry stays in train.',
                'accepted_distance_semantics': 'The accepted safeguarded edit budget is not measured achieved displacement; review added/removed, unresolved voxels and geometry.',
                'warnings': [IMAGE_WARNING, SPLIT_WARNING, PREFLIGHT_WARNING]}
    document['fit_ready'] = document['fit_ready'] and failed == 0
    if any(row.get('baseline_equivalent') for row in samples if row['operation'] != 'none'):
        document['warnings'].append('Some edited variants preserve baseline target geometry; grid spacing, policy or safeguards can collapse distinct requested settings.')
    if document['duplicate_variant_count']:
        document['warnings'].append('Repeated target geometry is retained for audit but grouped into one split; variant count exceeds the number of distinct target anatomies.')
    if any(row.get('foreground_voxels') == 0 for row in samples if row['status'] == 'ok'):
        document['warnings'].append('Some variants have an empty selected-lineage target; review those complete target losses before training.')
    if any(step['unresolved_voxels'] for row in samples for step in row.get('step_outcomes', [])):
        document['warnings'].append('Some requested releases remain unresolved and keep their original labels; ordinary reassignment can achieve different geometry from diagnostic erosion.')
    _write_review(output, document)
    if not prepare:
        (output/'INCOMPLETE').unlink()
        inventory = {str(path.relative_to(output)): _file_hash(path)
                     for path in sorted(output.rglob('*')) if path.is_file()}
        _write_json(output/'artifact-manifest.json', inventory)
        return {**document, 'stage': 'preflight', 'preflight_path': str(output/'preflight.json'),
                'report_path': str(output/'preflight.html'), 'preflight_directory': str(output)}
    if failed or split['split_errors']:
        problems = ([f'{failed} recipe variants failed native checks'] if failed else []) + split['split_errors']
        raise ValueError('; '.join(problems) + f'; inspect {output / "preflight.html"}. Dataset remains INCOMPLETE')
    artifacts = {str(path.relative_to(output)): _file_hash(path)
                 for path in sorted(output.rglob('*')) if path.is_file() and path.name != 'INCOMPLETE'}
    manifest = {'schema_version': SCHEMA, 'complete': True, 'dataset_fingerprint': fingerprint,
                'input_snapshot': 'input.ini' if input_bytes is not None else None,
                'input_config_sha256': document['input_config_sha256'],
                'anatomy_family': config['train']['anatomy_family'], 'case_id': config['input']['case_id'],
                'frame': config['input']['frame'], 'split_mode': 'scenario_only', 'split_warning': SPLIT_WARNING,
                'image_method': 'attenuation_copy_proxy', 'image_units': 'cm^-1', 'image_warning': IMAGE_WARNING,
                'target_scope': 'selected_lineage', 'target_semantics': TARGET_SEMANTICS,
                'candidate_source_ids': list(resolved['source_ids']), 'geometry': geometry, 'source_files': sources,
                'source_crop_sha256': original_hashes,
                'catalog_sha256': resolved['catalog_sha256'], 'audit_sha256': resolved['audit_sha256'],
                'provenance': provenance, 'samples': samples, 'sample_count': len(samples), **split,
                'original_sources_modified': False, 'variants_start_from_original': True,
                'warnings': document['warnings'], 'artifacts_sha256': artifacts}
    path = output/'dataset-manifest.json'
    _write_json(path, manifest)
    checksum = _file_hash(path)
    (output/'dataset-manifest.sha256').write_text(checksum+'\n')
    (output/'INCOMPLETE').unlink()
    return {'manifest_path': str(path), 'manifest_sha256': checksum, 'dataset_directory': str(output),
            'dataset_fingerprint': fingerprint, 'sample_count': len(samples), **split,
            'target_scope': 'selected_lineage', 'warnings': document['warnings'], 'samples': samples}


def preflight_recipe_dataset(config, resolved, *, input_bytes=None):
    """Evaluate all native variants and write review artifacts, without pair NPZs."""
    return _run(config, resolved, input_bytes, prepare=False)


def prepare_recipe_dataset(config, resolved, *, input_bytes=None):
    """Prepare independently edited image/lineage-mask pairs after native checks."""
    return _run(config, resolved, input_bytes, prepare=True)
