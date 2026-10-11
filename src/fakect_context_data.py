"""Immutable broader-aorta masks referencing original whole-body attenuation.

The thoracic landmark policy and anatomy families remain provisional. Preparation
never copies a whole-body sibling image or changes an existing frozen cohort.
"""
from pathlib import Path
import configparser
import copy
import io
import json
import re

import numpy as np

import fakect_phantom_anatomy as anatomy
from fakect_mask_pair_preview import sha256, write_preview
from fakect_two_stage_data import inspect_dataset as inspect_source

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.context-dataset/1'
CODE_FILES = ('fakect_context_data.py', 'fakect_mask_pair_preview.py',
              'fakect_phantom_anatomy.py', 'fakect_anatomy.py',
              'fakect_anatomy_surfaces.py', 'fakect_anatomy_centerlines.py',
              'fakect_tissues.py', 'fakect_roi.py')


def _absolute(value, root=ROOT):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _write(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + '\n')
    path.with_suffix('.json.sha256').write_text(sha256(path) + '\n')


def _read(path, expected=None):
    path = Path(path)
    checksum = sha256(path)
    if expected is not None and checksum != expected:
        raise ValueError('Changed manifest: ' + str(path))
    if path.with_suffix('.json.sha256').read_text().strip() != checksum:
        raise ValueError('Manifest checksum mismatch: ' + str(path))
    return json.loads(path.read_text())


def _identifier(value):
    if not isinstance(value, str) or not re.fullmatch(r'[A-Za-z0-9_-]+', value):
        raise ValueError('Invalid case identifier')
    return value


def load_config(path):
    path = _absolute(path)
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    parser.read_string(path.read_text())
    allowed = {'schema_version', 'source_manifest', 'anatomy_template', 'cases',
               'output_directory', 'surface_inventory_pattern'}
    if (parser.defaults() or parser.sections() != ['cohort'] or
            set(parser['cohort']) - allowed):
        raise ValueError('Expected only the documented [cohort] fields')
    if parser.get('cohort', 'schema_version') != SCHEMA:
        raise ValueError('Expected schema_version=' + SCHEMA)
    cases = parser.get('cohort', 'cases', fallback='all').strip()
    cases = None if cases == 'all' else [_identifier(x.strip()) for x in cases.split(',')]
    if cases is not None and len(cases) != len(set(cases)):
        raise ValueError('Duplicate requested case')
    pattern = parser.get('cohort', 'surface_inventory_pattern', fallback='').strip()
    if pattern and (pattern.count('{case_id}') != 1 or
                    re.search(r'[{}]', pattern.replace('{case_id}', ''))):
        raise ValueError('Surface inventory pattern needs exactly one {case_id} placeholder')
    return {'config_path': path, 'config_sha256': sha256(path),
            'source_manifest': _absolute(parser.get('cohort', 'source_manifest')),
            'anatomy_template': _absolute(parser.get('cohort', 'anatomy_template')),
            'output_directory': _absolute(parser.get('cohort', 'output_directory')),
            'cases': cases, 'surface_inventory_pattern': pattern, 'resume': False}


def _identity(raw, *, check_stat=True):
    if (not isinstance(raw, dict) or not isinstance(raw.get('path'), str) or
            type(raw.get('bytes')) is not int or raw['bytes'] <= 0 or
            type(raw.get('mtime_ns')) is not int or raw['mtime_ns'] <= 0):
        raise ValueError('Invalid whole-body source identity')
    result = {**{k: raw[k] for k in ('bytes', 'mtime_ns')}, 'path': str(_absolute(raw['path']))}
    if check_stat:
        stat = Path(result['path']).stat()
        if stat.st_size != result['bytes'] or stat.st_mtime_ns != result['mtime_ns']:
            raise ValueError('Whole-body source changed: ' + result['path'])
    return result


def _case_config(config, case_id, output):
    value = copy.deepcopy(anatomy.load_config(config['anatomy_template']))
    value['input']['case_id'] = case_id
    value['name'] = 'thoracic-aorta-' + case_id
    value['output_directory'] = output / 'anatomy'
    pattern = config['surface_inventory_pattern']
    value['input']['surface_inventory'] = (_absolute(pattern.format(case_id=case_id)) if pattern else None)
    return value


def _write_case_input(config, path):
    parser = configparser.ConfigParser(interpolation=None)
    parser['anatomy'] = {'schema_version': anatomy.SCHEMA, 'name': config['name'],
                         'output_directory': str(config['output_directory'])}
    parser['input'] = {k: '' if v is None else str(v) for k, v in config['input'].items()}
    parser['selection'] = {k: '' if v is None else str(v).lower() if isinstance(v, bool) else str(v)
                           for k, v in config['selection'].items()}
    content = io.StringIO()
    parser.write(content)
    if path.exists() and path.read_text() != content.getvalue():
        raise ValueError('Pinned case input changed')
    if not path.exists():
        path.write_text(content.getvalue())
    return anatomy.load_config(path)


def _inputs(config):
    if sha256(config['config_path']) != config['config_sha256']:
        raise ValueError('Context INI changed after loading')
    source = inspect_source(config['source_manifest'], verify_payloads=False)
    selected = config['cases'] or sorted({r['case_id'] for r in source['samples']})
    available = {r['case_id'] for r in source['samples']}
    if not selected or not set(selected) <= available:
        raise ValueError('Requested cases are absent from source cohort')
    rows = [r for r in source['samples'] if r['case_id'] in selected]
    if len({r['sample_id'] for r in rows}) != len(rows):
        raise ValueError('Duplicate source sample')
    prepared = {}
    for row in rows:
        filename = row['prepared_manifest']
        if filename not in prepared:
            if sha256(filename) != row['prepared_manifest_sha256']:
                raise ValueError('Prepared source manifest changed')
            prepared[filename] = json.loads(Path(filename).read_text())
        item = prepared[filename]
        if item['case_id'] != row['case_id'] or item['anatomy_family'] != row['anatomy_family']:
            raise ValueError('Prepared case/family identity mismatch')
        _identity(item['source_files']['atn'])
        _identity(item['source_files']['act'])
    request = {'schema_version': SCHEMA, 'config_sha256': config['config_sha256'],
               'source_manifest': str(config['source_manifest']),
               'source_manifest_sha256': source['manifest_sha256'],
               'anatomy_template': str(config['anatomy_template']),
               'anatomy_template_sha256': sha256(config['anatomy_template']),
               'cases': selected, 'sample_count': len(rows),
               'code_sha256': {name: sha256(ROOT / 'src' / name) for name in CODE_FILES}}
    return source, rows, prepared, request


def validate(config):
    """Validate registered RAW/atlas and source metadata; do not decode native volumes."""
    _, rows, prepared, request = _inputs(config)
    reports = []
    for case in request['cases']:
        selected = [r for r in rows if r['case_id'] == case]
        report = anatomy.validate(_case_config(config, case, config['output_directory'] / case))
        for row in selected:
            item = prepared[row['prepared_manifest']]
            if (report['frame'] != item['frame'] or
                    report['act_source'] != _identity(item['source_files']['act']) or
                    report['geometry']['shape_kji'] != row['geometry']['source_shape_kji'] or
                    report['geometry']['spacing_ijk_mm'] != row['geometry']['spacing_ijk_mm']):
                raise ValueError('Anatomy and frozen edit source geometry/frame differ')
        reports.append({'case_id': case, 'sample_count': len(selected), 'frame': report['frame'],
                        'status': report['status'], 'scope': report['scope'],
                        'geometry': report['geometry'], 'provisional': True})
    return {'schema_version': SCHEMA, 'status': 'metadata_validated', 'scope': 'engineering_only',
            'anatomical_extent_reviewed': False, 'case_count': len(reports),
            'sample_count': len(rows), 'cases': reports, 'request': request,
            'note': 'Native transfer and payload verification occur during prepare; metadata validation cannot guarantee anatomical coverage.'}


def _descriptor(source, prepared, preview_sample, selection, case_dir):
    geometry = source['geometry']
    return {'sample_id': source['sample_id'], 'case_id': source['case_id'],
            'variant_id': source['variant_id'], 'anatomy_family': source['anatomy_family'],
            'family_status': 'provisional', 'family_verified': False,
            'baseline': source['baseline'], 'baseline_equivalent': source['baseline_equivalent'],
            'split': 'unassigned', 'frame': prepared['frame'],
            'original_reference_mask_sha256': source['before_mask_sha256'],
            'source_atn': _identity(prepared['source_files']['atn']),
            'source_act': _identity(prepared['source_files']['act']),
            'source_shape_kji': geometry['source_shape_kji'],
            'spacing_ijk_mm': geometry['spacing_ijk_mm'],
            'source_npz': source['source_npz'], 'source_npz_sha256': source['source_npz_sha256'],
            'variant_npz': source['variant_npz'], 'variant_npz_sha256': source['variant_npz_sha256'],
            'edit_crop_origin_ijk': geometry['crop_origin_ijk'],
            'edit_crop_shape_kji': geometry['crop_shape_kji'],
            'mask_npz': str(case_dir / 'pairs' / preview_sample['path']),
            'mask_npz_sha256': preview_sample['sha256'], 'mask_counts': preview_sample['counts'],
            'prepared_manifest': source['prepared_manifest'],
            'prepared_manifest_sha256': source['prepared_manifest_sha256'],
            'selection_manifest': str(case_dir / 'anatomy' / 'selection.json'),
            'selection_manifest_sha256': sha256(case_dir / 'anatomy' / 'selection.json'),
            'anatomical_scope': selection['scope'], 'anatomical_extent_reviewed': False,
            'image_units': source['image_units'], 'image_method': source['image_method'],
            'source_atn_units': 'per_pixel', 'source_atn_dtype': '<f4',
            'parameters': source.get('parameters', {}), 'step_outcomes': source.get('step_outcomes', [])}


def _completed_case(case_dir, request, selected, prepared):
    selection_path, preview_path = case_dir / 'anatomy/selection.json', case_dir / 'pairs/preview-manifest.json'
    selection, preview = _read(selection_path), _read(preview_path)
    if ((selection_path.parent / 'INCOMPLETE').exists() or (preview_path.parent / 'INCOMPLETE').exists() or
            selection.get('complete') is not True):
        raise ValueError('Incomplete per-case anatomy or mask transfer')
    if (selection.get('schema_version') != 'fakect.phantom-anatomy/1' or
            selection.get('provisional') is not True or selection.get('coordinate_reviewed') is not False or
            selection.get('status') != 'complete_provisional_review' or
            preview.get('schema_version') != 'fakect.mask-pair-preview/1' or
            preview.get('source_selection_manifest') != str(selection_path) or
            preview.get('source_paired_manifest') != request['source_manifest'] or
            preview['source_selection_manifest_sha256'] != sha256(selection_path) or
            preview['source_paired_manifest_sha256'] != request['source_manifest_sha256'] or
            preview.get('source_anatomy_npz') != str(case_dir / 'anatomy/mask.npz') or
            preview.get('source_anatomy_sha256') != selection['mask_npz_sha256'] or
            sha256(case_dir / 'anatomy/mask.npz') != selection['mask_npz_sha256']):
        raise ValueError('Case manifest binding changed')
    for row in selected:
        item = prepared[row['prepared_manifest']]
        if (selection['case_id'] != row['case_id'] or preview['case_id'] != row['case_id'] or
                selection['frame'] != item['frame'] or preview['frame'] != item['frame'] or
                selection['act_source'] != _identity(item['source_files']['act']) or
                preview['source_act'] != selection['act_source'] or
                selection['geometry']['shape_kji'] != row['geometry']['source_shape_kji'] or
                selection['geometry']['spacing_ijk_mm'] != row['geometry']['spacing_ijk_mm']):
            raise ValueError('Case selection frame/geometry/source provenance disagrees with edits')
    pairs = {s['sample_id']: s for s in preview['samples']}
    if len(pairs) != len(preview['samples']) or set(pairs) != {s['sample_id'] for s in selected}:
        raise ValueError('Case transfer omitted or duplicated a source scenario')
    for row in selected:
        pair = pairs[row['sample_id']]
        if any(pair.get(key) != row[key] for key in ('sample_id', 'case_id', 'variant_id', 'baseline')):
            raise ValueError('Transferred pair identity changed')
        if pair.get('status') == 'ready_for_visual_review':
            name = pair.get('path')
            if not isinstance(name, str) or Path(name).name != name or not name.endswith('.npz'):
                raise ValueError('Transferred pair path must stay inside its case directory')
    failed = [s for s in preview['samples'] if s['status'] != 'ready_for_visual_review']
    if failed:
        return {'case_id': selection['case_id'], 'status': 'requires_review', 'failures': failed}, []
    rows = [_descriptor(s, prepared[s['prepared_manifest']], pairs[s['sample_id']], selection, case_dir) for s in selected]
    for row in rows:
        if sha256(row['mask_npz']) != row['mask_npz_sha256']:
            raise ValueError('Transferred mask changed')
    summary = {'case_id': selection['case_id'], 'frame': selection['frame'],
               'anatomy_family': rows[0]['anatomy_family'], 'sample_count': len(rows),
               'status': 'complete_provisional_review', 'anatomical_extent_reviewed': False,
               'anatomical_scope': selection['scope'], 'settings': selection['settings'],
               'selection_manifest': str(selection_path), 'selection_manifest_sha256': sha256(selection_path),
               'preview_manifest': str(preview_path), 'preview_manifest_sha256': sha256(preview_path)}
    return summary, rows


def prepare(config):
    """Prepare every requested sample, resuming only explicitly matched revisions."""
    metadata = validate(config)
    _, source_rows, prepared, request = _inputs(config)
    output = config['output_directory']
    if output.exists():
        if not config.get('resume'):
            raise ValueError('Output exists; choose a new revision or explicitly use --resume')
        if _read(output / 'request.json') != request:
            raise ValueError('Resume input/code fingerprint differs; use a new output revision')
        if (output / 'dataset-manifest.json').exists() and not (output / 'INCOMPLETE').exists():
            return inspect_dataset(output / 'dataset-manifest.json')
    else:
        output.mkdir(parents=True)
        (output / 'INCOMPLETE').write_text('Context cohort preparation in progress\n')
        (output / 'input.ini').write_bytes(config['config_path'].read_bytes())
        (output / 'anatomy-template.ini').write_bytes(config['anatomy_template'].read_bytes())
        _write(output / 'request.json', request)
        _write(output / 'metadata-validation.json', metadata)
    summaries, rows = [], []
    for case in request['cases']:
        case_dir = output / case
        case_dir.mkdir(exist_ok=True)
        selected = [r for r in source_rows if r['case_id'] == case]
        try:
            case_config = _write_case_input(_case_config(config, case, case_dir), case_dir / 'anatomy.ini')
            if not (case_dir / 'anatomy').exists():
                anatomy.run(case_config)
            elif (case_dir / 'anatomy/INCOMPLETE').exists():
                raise ValueError('Incomplete anatomy extraction requires a new output revision')
            selection = _read(case_dir / 'anatomy/selection.json')
            if selection['config_sha256'] != sha256(case_dir / 'anatomy.ini'):
                raise ValueError('Cached anatomy uses a different case input')
            if not (case_dir / 'pairs').exists():
                write_preview(case_dir / 'anatomy/mask.npz', case_id=case,
                              dataset_manifest=config['source_manifest'], output_directory=case_dir / 'pairs')
            summary, case_rows = _completed_case(case_dir, request, selected, prepared)
            summaries.append(summary)
            rows.extend(case_rows)
        except (ValueError, OSError, KeyError) as error:
            summaries.append({'case_id': case, 'status': 'failed', 'reason': str(error)})
        _write(output / 'preparation-status.json', {'cases': summaries, 'expected_samples': len(source_rows),
                                                   'prepared_samples': len(rows), 'complete': False})
    if any(c['status'] != 'complete_provisional_review' for c in summaries) or len(rows) != len(source_rows):
        raise ValueError('Context cohort remains INCOMPLETE; review preparation-status.json. No failed scenarios were omitted from a training release.')
    if _inputs(config)[3] != request:
        raise ValueError('Context inputs changed during preparation')
    document = {'schema_version': SCHEMA, 'complete': True, 'scope': 'engineering_only',
                'status': 'complete_provisional_review', 'anatomical_extent_reviewed': False,
                'family_status': 'provisional', 'family_verified': False, 'split_mode': 'unassigned',
                'source_manifest': request['source_manifest'], 'source_manifest_sha256': request['source_manifest_sha256'],
                'request_sha256': sha256(output / 'request.json'), 'case_count': len(summaries),
                'sample_count': len(rows), 'baseline_count': sum(r['baseline'] for r in rows),
                'family_count': len({r['anatomy_family'] for r in rows}), 'cases': summaries, 'samples': rows,
                'image_storage': 'original_whole_body_atn_plus_frozen_native_crop_overlay',
                'source_integrity': 'Original full-volume path, byte length and mtime; full source volumes are not content-hashed.',
                'mask_semantics': 'ROI-free landmark-limited thoracic aorta and proximal branches, with frozen native edit deltas.',
                'limitations': ['Anatomical boundaries and branch coverage require visual review.',
                                'Family assignments are provisional; siblings must never cross splits.',
                                'Attenuation-copy proxy is not a simulated clinical CT reconstruction.']}
    _write(output / 'dataset-manifest.json', document)
    # Inspect everything before removing the incomplete marker. The internal flag
    # is not exposed on the public inspection interface.
    _inspect(output / 'dataset-manifest.json', verify_payloads=True, preparing=True)
    (output / 'INCOMPLETE').unlink()
    _write(output / 'preparation-status.json', {'cases': summaries, 'expected_samples': len(source_rows),
                                               'prepared_samples': len(rows), 'complete': True})
    return inspect_dataset(output / 'dataset-manifest.json', verify_payloads=False)


def _vector(value, kind):
    array = np.asarray(value)
    if array.shape != (3,) or not np.isfinite(array).all():
        raise ValueError('Invalid native geometry vector')
    if kind == 'origin':
        if array.dtype.kind not in 'iu' or np.any(array < 0):
            raise ValueError('Invalid native crop origin')
    elif kind == 'shape':
        if array.dtype.kind not in 'iu' or np.any(array <= 0):
            raise ValueError('Invalid native shape')
    elif np.any(array <= 0):
        raise ValueError('Invalid native spacing')
    return array


def _inspect(path, verify_payloads, preparing=False):
    path = _absolute(path)
    if path.name != 'dataset-manifest.json' or path.is_symlink() or ((path.parent / 'INCOMPLETE').exists() and not preparing):
        raise ValueError('Expected a complete context dataset-manifest.json')
    document = _read(path)
    if (document.get('schema_version') != SCHEMA or document.get('complete') is not True or
            document.get('scope') != 'engineering_only' or document.get('anatomical_extent_reviewed') is not False or
            document.get('family_status') != 'provisional' or document.get('family_verified') is not False or
            document.get('split_mode') != 'unassigned' or
            document.get('image_storage') != 'original_whole_body_atn_plus_frozen_native_crop_overlay'):
        raise ValueError('Unsupported context dataset contract')
    request = _read(path.parent / 'request.json', document['request_sha256'])
    if (request.get('schema_version') != SCHEMA or not request.get('cases') or
            len(request['cases']) != len(set(request['cases']))):
        raise ValueError('Invalid pinned context request')
    source = inspect_source(document['source_manifest'], verify_payloads=False)
    if (source['manifest_sha256'] != document['source_manifest_sha256'] or
            request['source_manifest_sha256'] != source['manifest_sha256'] or
            request['source_manifest'] != document['source_manifest'] or
            sha256(path.parent / 'input.ini') != request['config_sha256'] or
            sha256(path.parent / 'anatomy-template.ini') != request['anatomy_template_sha256']):
        raise ValueError('Context source/snapshot fingerprint mismatch')
    originals = {r['sample_id']: r for r in source['samples'] if r['case_id'] in request['cases']}
    prepared = {r['prepared_manifest']: json.loads(Path(r['prepared_manifest']).read_text()) for r in originals.values()}
    expected_rows, expected_cases = {}, []
    for case in request['cases']:
        summary, case_rows = _completed_case(path.parent / case, request,
            [r for r in originals.values() if r['case_id'] == case], prepared)
        if summary['status'] != 'complete_provisional_review':
            raise ValueError('Case contains failed transfers')
        expected_cases.append(summary)
        expected_rows.update({r['sample_id']: r for r in case_rows})
    if document.get('cases') != expected_cases:
        raise ValueError('Case scope or provenance changed')
    identities, verified = set(), set()
    for row in document.get('samples', []):
        identity = row.get('sample_id')
        if identity in identities or identity not in expected_rows or row != expected_rows[identity]:
            raise ValueError('Context sample disagrees with pinned source or mask transfer')
        identities.add(identity)
        source_shape = _vector(row['source_shape_kji'], 'shape')
        edit_shape = _vector(row['edit_crop_shape_kji'], 'shape')
        edit_origin = _vector(row['edit_crop_origin_ijk'], 'origin')
        spacing = _vector(row['spacing_ijk_mm'], 'spacing')
        if np.any(edit_origin + edit_shape[::-1] > source_shape[::-1]):
            raise ValueError('Edit crop leaves source volume')
        if row['source_atn']['bytes'] != int(np.prod(source_shape)) * 4:
            raise ValueError('ATN byte count disagrees with native float32 geometry')
        if verify_payloads:
            for key in ('variant_npz', 'source_npz', 'mask_npz'):
                filename = row[key]
                if filename not in verified:
                    if sha256(filename) != row[key + '_sha256']:
                        raise ValueError('Changed context payload: ' + filename)
                    verified.add(filename)
            with np.load(row['variant_npz'], allow_pickle=False) as data:
                if data['image'].shape != tuple(edit_shape) or not np.isfinite(data['image']).all():
                    raise ValueError('Invalid frozen attenuation overlay')
            with np.load(row['mask_npz'], allow_pickle=False) as data:
                masks = [np.asarray(data[k]) for k in ('before', 'after', 'added', 'removed')]
                before, after, added, removed = masks
                origin = _vector(data['crop_origin_ijk'], 'origin')
                if (before.ndim != 3 or not before.size or any(m.shape != before.shape or not np.all((m == 0) | (m == 1)) for m in masks) or
                        not np.array_equal(spacing, data['spacing_ijk_mm']) or
                        data['sample_id'].shape != () or str(data['sample_id']) != identity or
                        np.any(origin + np.asarray(before.shape)[::-1] > source_shape[::-1])):
                    raise ValueError('Invalid compact mask geometry/identity')
                before, after, added, removed = [m.astype(bool) for m in masks]
                if not np.array_equal(added, after & ~before) or not np.array_equal(removed, before & ~after):
                    raise ValueError('Compact added/removed masks disagree with reference and observed masks')
                for name, mask in zip(('before', 'after', 'added', 'removed'), masks):
                    if int(mask.sum()) != row['mask_counts'][name]:
                        raise ValueError('Compact mask count mismatch')
    if (identities != set(originals) or document.get('sample_count') != len(identities) or
            document.get('case_count') != len(expected_cases) or
            document.get('family_count') != len({r['anatomy_family'] for r in expected_rows.values()}) or
            document.get('baseline_count') != sum(r['baseline'] for r in expected_rows.values())):
        raise ValueError('Context dataset inventory counts disagree')
    return {**document, 'manifest_path': str(path), 'manifest_sha256': sha256(path),
            'payloads_verified': bool(verify_payloads)}


def inspect_dataset(path, verify_payloads=True):
    """Verify immutable provenance and optionally every compact native payload."""
    return _inspect(path, verify_payloads)
