"""Pin prepared native pairs for CT segmentation and reference-shape recovery.

This adapter never runs morphology or changes a frozen cohort. The observed mask
A and original reference B define net added=A&~B and removed=B&~A. Net changes
describe the final reconstruction problem, not per-step intervention history.
Family groupings remain explicitly provisional; this is an engineering dataset.
"""
from __future__ import annotations

import argparse
import configparser
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re

import numpy as np

from fakect_cohort_registry import (verify_prepared_manifest, _hash, _array_hash,
                                   _json, _binary, _npz, _safe_file)

SCHEMA = 'fakect.two-stage-dataset/1'
ROOT = Path(__file__).resolve().parents[1]
_ID = re.compile(r'^[A-Za-z0-9][A-Za-z0-9_.-]*$')
_ARRAY_KEYS = {'before': 'target_mask', 'after': 'mask', 'image': 'image',
               'added': 'added_mask', 'removed': 'removed_mask'}


def _write(path, document):
    with Path(path).open('x', encoding='utf-8') as handle:
        handle.write(json.dumps(document, indent=2, allow_nan=False) + '\n')


def _absolute(path, root=ROOT):
    path = Path(path).expanduser()
    return (path if path.is_absolute() else root/path).resolve()


def _identifier(value, label):
    if not isinstance(value, str) or not _ID.fullmatch(value) or '..' in value:
        raise ValueError(f'Invalid {label}: {value!r}')
    return value


def _project(path):
    path = _absolute(path)
    raw = path.read_bytes()
    project = _json(raw.decode())
    if (project.get('schema_version') != 'fakect.cohort-project/1'
            or not isinstance(project.get('cases'), list) or not project['cases']):
        raise ValueError('Expected a nonempty fakect.cohort-project/1')
    rows, ids = [], set()
    for case in project['cases']:
        case_id = _identifier(case.get('case_id'), 'case_id')
        if case_id in ids:
            raise ValueError('Duplicate project case ID')
        ids.add(case_id)
        family = _identifier(case.get('provisional_family'), 'provisional_family')
        if 'dataset_manifest' in case:
            manifest = _absolute(case['dataset_manifest'])
        else:
            parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
            parser.read_string(_absolute(case['config']).read_text())
            if parser.get('input', 'case_id') != case_id:
                raise ValueError('Project and cohort configuration case IDs disagree')
            manifest = _absolute(parser.get('train', 'dataset_directory'))/'dataset-manifest.json'
        rows.append({'case_id': case_id, 'anatomy_family': family,
                     'prepared_manifest': str(manifest)})
    return path, raw, rows


def _prepared_case(row, *, verify_payloads):
    prepared = verify_prepared_manifest(row['prepared_manifest'], verify_payloads=verify_payloads)
    if (prepared['case_id'] != row['case_id']
            or prepared['anatomy_family'] != row['anatomy_family']):
        raise ValueError('Project case/family disagrees with prepared cohort')
    if prepared.get('split_mode') != 'unassigned' or prepared.get('target_scope') != 'selected_lineage':
        raise ValueError('Two-stage datasets require unassigned selected-lineage cohorts')
    return prepared


def _source_mask(prepared):
    with _npz(Path(prepared['_manifest_path']).parent/'source.npz') as source:
        before = source['target_mask']
    _binary(before, tuple(prepared['geometry']['crop_shape_kji']), 'Before mask')
    return before.astype(np.uint8, copy=False)


def _descriptor(case, prepared, sample, before_hash):
    variant_id = _identifier(sample['variant_id'], 'variant_id')
    return {
        'sample_id': case['case_id'] + ':' + variant_id,
        'case_id': case['case_id'], 'anatomy_family': case['anatomy_family'],
        'family_verified': False, 'family_status': 'provisional',
        'variant_id': variant_id, 'baseline': variant_id == 'baseline',
        'baseline_equivalent': sample.get('baseline_equivalent', variant_id == 'baseline'),
        'split': 'unassigned',
        'source_npz': str(Path(prepared['_manifest_path']).parent/'source.npz'),
        'source_npz_sha256': prepared['artifacts_sha256']['source.npz'],
        'variant_npz': sample['_path'], 'variant_npz_sha256': sample['sha256'],
        'before_mask_sha256': before_hash, 'mask_sha256': sample['mask_sha256'],
        'image_sha256': sample['image_sha256'], 'geometry': prepared['geometry'],
        'image_units': prepared['image_units'], 'image_method': prepared['image_method'],
        'prepared_manifest': prepared['_manifest_path'],
        'prepared_manifest_sha256': prepared['manifest_sha256'],
        'dataset_fingerprint': prepared['dataset_fingerprint'],
        'parameters': sample.get('parameters', {}),
        'step_outcomes': sample.get('step_outcomes', []),
    }


def _changes(before, after):
    _binary(after, before.shape, 'After mask')
    return ((after.astype(bool) & ~before.astype(bool)).astype(np.uint8),
            (before.astype(bool) & ~after.astype(bool)).astype(np.uint8))


def prepare_dataset(project_path, output_dir, *, write_changes=True):
    """Create a new immutable manifest and optional compressed net-change masks.

Prepared image/reference arrays remain in their original immutable cohort
directories; they are hash-pinned, not duplicated or resampled. Existing output
directories are never overwritten, even when an earlier attempt is incomplete.
"""
    if not isinstance(write_changes, bool):
        raise ValueError('write_changes must be Boolean')
    project_path, raw, planned = _project(project_path)
    output = _absolute(output_dir)
    if output.exists():
        raise FileExistsError('Two-stage dataset directory exists; choose a new revision')
    # Fail missing/stale metadata before reserving an output. Full array checks
    # follow per case, in bounded memory, after the INCOMPLETE marker is present.
    for row in planned:
        _prepared_case(row, verify_payloads=False)
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').write_text('Two-stage dataset export is incomplete; do not train.\n')
    (output/'project.json').write_bytes(raw)
    if write_changes:
        (output/'changes').mkdir()
    cases, samples, inventory = [], [], {'project.json': _hash(output/'project.json')}
    for row in planned:
        prepared = _prepared_case(row, verify_payloads=True)
        before = _source_mask(prepared)
        before_hash = _array_hash(before)
        case = {**row, 'prepared_manifest_sha256': prepared['manifest_sha256'],
                'dataset_fingerprint': prepared['dataset_fingerprint'],
                'family_verified': False, 'family_status': 'provisional',
                'sample_count': prepared['sample_count'],
                'before_mask_sha256': before_hash,
                'source_crop_sha256': prepared.get('source_crop_sha256', {}),
                'geometry': prepared['geometry'],
                'freeze_sha256': prepared['artifacts_sha256'].get('cohort-freeze.json')}
        cases.append(case)
        for source_sample in prepared['samples']:
            sample = _descriptor(case, prepared, source_sample, before_hash)
            with _npz(sample['variant_npz']) as data:
                after = data['mask']
            added, removed = _changes(before, after)
            sample.update(added_voxels=int(added.sum()), removed_voxels=int(removed.sum()),
                          before_voxels=int(before.sum()), after_voxels=int(after.sum()),
                          added_mask_sha256=_array_hash(added), removed_mask_sha256=_array_hash(removed))
            if sample['baseline'] and (sample['added_voxels'] or sample['removed_voxels']):
                raise ValueError('Baseline is not identical to its reference mask')
            if write_changes:
                name = 'changes/' + row['case_id'] + '__' + sample['variant_id'] + '.npz'
                path = output/name
                with path.open('xb') as stream:
                    np.savez_compressed(stream, added_mask=added, removed_mask=removed)
                inventory[name] = _hash(path)
                sample.update(changes_npz=str(path), changes_npz_sha256=inventory[name])
            samples.append(sample)
        print(f"Two-stage data: {row['case_id']}, {prepared['sample_count']} native pairs verified", flush=True)
    document = {
        'schema_version': SCHEMA, 'complete': True,
        'created_at_utc': datetime.now(timezone.utc).isoformat(),
        'scope': 'engineering_only', 'family_status': 'provisional', 'family_verified': False,
        'split_mode': 'unassigned', 'case_count': len(cases), 'sample_count': len(samples),
        'baseline_count': sum(sample['baseline'] for sample in samples),
        'family_count': len({case['anatomy_family'] for case in cases}),
        'project_path': str(project_path), 'project_sha256': hashlib.sha256(raw).hexdigest(),
        'array_keys': _ARRAY_KEYS, 'change_semantics': 'net_final_vs_original',
        'model1': {'input': 'image', 'target': 'after'},
        'model2': {'input': 'after', 'target': 'before', 'auxiliary_targets': ['added', 'removed']},
        'warnings': [
            'Original phantoms are reference shapes under the study assumption, not clinically certified normal anatomy.',
            'Anatomy families are provisional. Keep each whole family together; results are engineering-only.',
            'Native target masks describe the reviewed ROI-selected aorta and its descendants, not independently validated whole-thoracic coverage.',
            'Images are attenuation-copy proxies in cm^-1, not HU or validated clinical CT.',
            'Net masks omit transient per-step additions/removals later undone by another edit.',
        ],
        'cases': cases, 'samples': samples, 'artifacts_sha256': inventory,
    }
    path = output/'dataset-manifest.json'
    _write(path, document)
    (output/'dataset-manifest.sha256').write_text(_hash(path)+'\n')
    (output/'INCOMPLETE').unlink()
    return {**document, 'manifest_path': str(path), 'manifest_sha256': _hash(path)}


def inspect_dataset(manifest_path, verify_payloads=True):
    """Verify references and optionally every native payload/change-mask contract.

Metadata-only inspection does not decode any NPZ, including held-out anatomy.
It still verifies JSON hashes, sample identities and bindings to prepared data.
"""
    path = _absolute(manifest_path)
    output = path.parent
    if path.name != 'dataset-manifest.json' or path.is_symlink() or (output/'INCOMPLETE').exists():
        raise ValueError('Expected a complete regular two-stage dataset-manifest.json')
    checksum = (output/'dataset-manifest.sha256').read_text().strip()
    if _hash(path) != checksum:
        raise ValueError('Two-stage manifest checksum mismatch')
    document = _json(path.read_text())
    if (document.get('schema_version') != SCHEMA or document.get('complete') is not True
            or document.get('scope') != 'engineering_only'
            or document.get('family_status') != 'provisional'
            or document.get('family_verified') is not False
            or document.get('split_mode') != 'unassigned'
            or document.get('array_keys') != _ARRAY_KEYS
            or document.get('change_semantics') != 'net_final_vs_original'):
        raise ValueError('Unsupported two-stage dataset contract')
    inventory = document.get('artifacts_sha256', {})
    if inventory.get('project.json') != document.get('project_sha256'):
        raise ValueError('Project snapshot hash is missing or inconsistent')
    for name, expected in inventory.items():
        artifact = _safe_file(output, name)
        if (verify_payloads or artifact.suffix != '.npz') and _hash(artifact) != expected:
            raise ValueError('Two-stage artifact checksum mismatch: '+name)
    cases, expected_samples, before_masks = {}, {}, {}
    for case in document.get('cases', []):
        if (case['case_id'] in cases or case.get('family_verified') is not False
                or case.get('family_status') != 'provisional'):
            raise ValueError('Invalid case identity/family status')
        prepared = _prepared_case(case, verify_payloads=verify_payloads)
        if (prepared['manifest_sha256'] != case['prepared_manifest_sha256']
                or prepared['sample_count'] != case['sample_count']
                or prepared['geometry'] != case['geometry']
                or prepared['dataset_fingerprint'] != case['dataset_fingerprint']):
            raise ValueError('Prepared cohort changed since two-stage export')
        expected_before_hash = prepared.get('source_crop_sha256', {}).get('selected')
        if expected_before_hash is not None and expected_before_hash != case['before_mask_sha256']:
            raise ValueError('Before mask hash disagrees with prepared source metadata')
        cases[case['case_id']] = case
        if verify_payloads:
            before_masks[case['case_id']] = _source_mask(prepared)
            if _array_hash(before_masks[case['case_id']]) != case['before_mask_sha256']:
                raise ValueError('Before mask checksum mismatch')
        for source_sample in prepared['samples']:
            expected = _descriptor(case, prepared, source_sample, case['before_mask_sha256'])
            expected_samples[expected['sample_id']] = expected
    identities = set()
    for sample in document.get('samples', []):
        identity = sample.get('sample_id')
        expected = expected_samples.get(identity)
        if identity in identities or expected is None or any(sample.get(k) != v for k, v in expected.items()):
            raise ValueError('Two-stage sample disagrees with pinned prepared sample')
        identities.add(identity)
        if 'changes_npz' in sample:
            change_path = Path(sample['changes_npz'])
            try:
                relative = str(change_path.relative_to(output))
            except ValueError as exc:
                raise ValueError('Change mask path escapes two-stage dataset') from exc
            if inventory.get(relative) != sample.get('changes_npz_sha256'):
                raise ValueError('Change mask hash disagrees with inventory')
        if verify_payloads:
            before = before_masks[sample['case_id']]
            with _npz(sample['variant_npz']) as data:
                after = data['mask']
            added, removed = _changes(before, after)
            for name, array in (('added', added), ('removed', removed)):
                if (sample.get(name+'_mask_sha256') != _array_hash(array)
                        or sample.get(name+'_voxels') != int(array.sum())):
                    raise ValueError('Net change metadata disagrees with paired masks')
            if sample.get('before_voxels') != int(before.sum()) or sample.get('after_voxels') != int(after.sum()):
                raise ValueError('Recorded foreground count disagrees with paired masks')
            if 'changes_npz' in sample:
                with _npz(sample['changes_npz']) as changes:
                    if (set(changes.files) != {'added_mask', 'removed_mask'}
                            or not np.array_equal(changes['added_mask'], added)
                            or not np.array_equal(changes['removed_mask'], removed)):
                        raise ValueError('Saved net change masks disagree with paired masks')
    if (identities != set(expected_samples) or document.get('sample_count') != len(identities)
            or document.get('case_count') != len(cases)
            or document.get('baseline_count') != sum(s['baseline'] for s in document['samples'])
            or document.get('family_count') != len({c['anatomy_family'] for c in cases.values()})):
        raise ValueError('Two-stage dataset inventory counts disagree')
    return {**document, 'manifest_path': str(path), 'manifest_sha256': checksum,
            'payloads_verified': bool(verify_payloads)}


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    commands = parser.add_subparsers(dest='command', required=True)
    prepare = commands.add_parser('prepare')
    prepare.add_argument('--project', required=True)
    prepare.add_argument('--output', required=True)
    prepare.add_argument('--no-change-sidecars', action='store_true')
    inspect = commands.add_parser('inspect')
    inspect.add_argument('--manifest', required=True)
    inspect.add_argument('--metadata-only', action='store_true')
    args = parser.parse_args(argv)
    result = (prepare_dataset(args.project, args.output, write_changes=not args.no_change_sidecars)
              if args.command == 'prepare' else inspect_dataset(args.manifest, not args.metadata_only))
    print(json.dumps({key: value for key, value in result.items()
                     if key not in ('samples', 'cases', 'artifacts_sha256')}, indent=2))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
