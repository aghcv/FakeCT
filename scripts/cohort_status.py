#!/usr/bin/env python3
"""Regenerate a mutable, metadata-only dashboard for phantom cohort review.

This index does not validate current source volumes, execute edits, freeze an
input, publish a dataset, or assign model splits. Case artifacts remain read-only.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path, PurePosixPath
import re
import shlex
import sys
import tempfile
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_study_config import load_study_config
from fakect_recipe_study_config import plan_recipe_variants
from fakect_cohort_registry import load_registry_entry, verify_prepared_manifest

SCHEMA = 'fakect.cohort-status/1'
HEX = re.compile(r'[0-9a-f]{64}\Z')


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def _json(path):
    value = json.loads(Path(path).read_text())
    if not isinstance(value, dict):
        raise ValueError(f'Expected a JSON object: {path}')
    return value


def _resolve(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def _artifact(folder, name):
    parts = PurePosixPath(name)
    if (not name or '\\' in name or parts.is_absolute() or
            any(part in ('', '.', '..') for part in name.split('/'))):
        raise ValueError(f'Unsafe artifact name: {name}')
    path = folder.joinpath(*parts.parts)
    current = folder
    for part in parts.parts:
        current /= part
        if current.is_symlink():
            raise ValueError(f'Symlink artifact: {name}')
    if not path.is_file():
        raise ValueError(f'Missing artifact: {name}')
    return path


def _inventory(folder, required):
    inventory = _json(folder / 'artifact-manifest.json')
    if not set(required).issubset(inventory):
        raise ValueError('Artifact inventory does not cover required report/input files')
    skipped = 0
    for name, checksum in inventory.items():
        if not isinstance(checksum, str) or not HEX.fullmatch(checksum):
            raise ValueError(f'Invalid artifact checksum: {name}')
        path = _artifact(folder, name)
        if path.suffix.lower() in ('.npz', '.npy', '.bin', '.raw'):
            skipped += 1
        elif _sha(path) != checksum:
            raise ValueError(f'Artifact checksum mismatch: {name}')
    return {'integrity': 'metadata_checked', 'payload_files_not_read': skipped}


def _missing(folder, document):
    result = {'directory': str(folder), 'document': str(folder / document),
              'state': 'missing', 'integrity': 'not_checked', 'matches_current_input': None}
    if (folder / 'INCOMPLETE').exists():
        result['state'] = 'incomplete'
    elif folder.exists() and not (folder / document).is_file():
        result['state'] = 'incomplete'
    return result


def _report_status(folder, current_hash, kind):
    name = 'preview-report.json' if kind == 'preview' else 'preflight.json'
    report_name = 'report.html' if kind == 'preview' else 'preflight.html'
    result = _missing(folder, name)
    result['report'] = str(folder / report_name)
    if result['state'] == 'incomplete' or not (folder / name).is_file():
        return result
    try:
        document = _json(folder / name)
        result.update(_inventory(folder, (name, report_name, 'input.ini')))
        saved = _sha(folder / 'input.ini')
        if document.get('input_config_sha256') != saved:
            raise ValueError('Report input hash disagrees with its saved input.ini')
        matches = saved == current_hash
        result.update(state='complete' if matches else 'stale', matches_current_input=matches,
                      input_sha256=saved)
        if kind == 'preflight':
            valid = (document.get('complete') is True and document.get('execution_valid') is True
                     and document.get('failed_variants') == 0)
            result.update(execution_valid=valid, variant_count=document.get('variant_count'),
                          successful_variants=document.get('successful_variants'),
                          failed_variants=document.get('failed_variants'),
                          dataset_fingerprint=document.get('dataset_fingerprint'))
            if not valid:
                result['state'] = 'failed'
        else:
            result['selected_voxels'] = document.get('selected_voxels')
    except (OSError, ValueError, TypeError) as error:
        result.update(state='invalid', integrity='failed', error=str(error))
    return result


def _freeze_status(folder, current_hash):
    result = _missing(folder, 'freeze.json')
    if result['state'] == 'incomplete' or not (folder / 'freeze.json').is_file():
        return result
    try:
        lock = _json(_artifact(folder, 'freeze.json'))
        if (_sha(folder / 'freeze.json') != _artifact(folder, 'freeze.sha256').read_text().strip()
                or lock.get('schema_version') != 'fakect.cohort-freeze/1'
                or _sha(_artifact(folder, 'input.ini')) != lock.get('input_sha256')
                or _sha(_artifact(folder, 'preflight.json')) != lock.get('preflight_sha256')
                or lock.get('coordinate_reviewed') is not True
                or lock.get('parameters_reviewed') is not True):
            raise ValueError('Freeze metadata checksum or review flags are invalid')
        matches = lock['input_sha256'] == current_hash
        result.update(state='complete' if matches else 'stale', integrity='metadata_checked',
                      matches_current_input=matches, input_sha256=lock['input_sha256'],
                      variant_count=lock.get('variant_count'),
                      dataset_fingerprint=lock.get('dataset_fingerprint'))
    except (OSError, ValueError, TypeError) as error:
        result.update(state='invalid', integrity='failed', error=str(error))
    return result


def _prepared_status(folder, current_hash):
    result = _missing(folder, 'dataset-manifest.json')
    if result['state'] == 'incomplete' or not (folder / 'dataset-manifest.json').is_file():
        return result
    try:
        manifest = verify_prepared_manifest(folder / 'dataset-manifest.json', verify_payloads=False)
        matches = manifest.get('input_config_sha256') == current_hash
        result.update(state='complete' if matches else 'stale', integrity='metadata_checked',
                      matches_current_input=matches, input_sha256=manifest.get('input_config_sha256'),
                      sample_count=manifest['sample_count'], split_mode=manifest['split_mode'],
                      dataset_fingerprint=manifest['dataset_fingerprint'])
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(state='invalid', integrity='failed', error=str(error))
    return result


def _published_status(registry, name, current_hash):
    result = {'name': name, 'state': 'not_published', 'integrity': 'not_checked',
              'matches_current_input': None,
              'entry': str(registry / 'entries' / f'{name}.json')}
    try:
        entry = load_registry_entry(registry, name, verify=False)
        manifest = entry['manifest']
        matches = manifest.get('input_config_sha256') == current_hash
        result.update(state='published', integrity='metadata_checked', matches_current_input=matches,
                      input_sha256=manifest.get('input_config_sha256'),
                      manifest=entry['manifest_path'], sample_count=entry['sample_count'],
                      family_verified=entry['family_verified'], anatomy_family=entry['anatomy_family'],
                      original_manifest=entry.get('original_manifest_path'),
                      snapshot_relation='current_input' if matches else 'different_input_snapshot')
    except FileNotFoundError:
        if (registry / 'entries' / f'{name}.json').exists():
            result.update(state='invalid', integrity='failed', error='Published entry has missing artifacts')
    except (OSError, ValueError, TypeError, KeyError) as error:
        result.update(state='invalid', integrity='failed', error=str(error))
    return result


def _command_path(path, root):
    try:
        return str(path.relative_to(root))
    except ValueError:
        return str(path)


def _commands(config_path, config, case, registry, published, root):
    base = ['python3', 'scripts/prepare_cohort.py', '--config', _command_path(config_path, root)]
    commands = {stage: shlex.join(base + ['--stage', stage])
                for stage in ('preview', 'preflight', 'freeze', 'prepare')}
    commands['validate'] = shlex.join(base + ['--validate-only'])
    name = case['registry_name']
    # The imported first case already occupies r1. New drafts need an unused
    # revision name; the dashboard must not suggest replacing the snapshot.
    if published['state'] != 'not_published' and not published.get('matches_current_input'):
        stem = re.sub(r'-r\d+$', '', name)
        revision = 1
        while (registry / 'entries' / f'{stem}-r{revision}.json').exists():
            revision += 1
        name = f'{stem}-r{revision}'
    commands['register'] = shlex.join([
        'python3', 'scripts/cohort_registry.py', 'register', '--registry', _command_path(registry, root),
        '--dataset', _command_path(config['train']['dataset_directory'] / 'dataset-manifest.json', root),
        '--name', name, '--anatomy-family', config['train']['anatomy_family']])
    return commands


def collect_status(project_path, *, repo_root=ROOT):
    """Read metadata only. No case, registry, source, or output file is written."""
    root = Path(repo_root).resolve()
    project_path = _resolve(project_path, root)
    project = _json(project_path)
    if project.get('schema_version') != 'fakect.cohort-project/1' or not isinstance(project.get('cases'), list):
        raise ValueError('Expected a fakect.cohort-project/1 case manifest')
    registry = _resolve(project['registry'], root)
    cases = []
    for case in project['cases']:
        config_path = _resolve(case['config'], root)
        row = {key: case.get(key) for key in ('case_id', 'template', 'provisional_family', 'registry_name')}
        row.update(config=str(config_path), config_valid=False)
        try:
            raw = config_path.read_bytes()
            current_hash = hashlib.sha256(raw).hexdigest()
            row['input_sha256'] = current_hash
            config = load_study_config(config_path, repo_root=root)
            if str(config['input']['case_id']) != str(case['case_id']):
                raise ValueError('Project case_id disagrees with INI input.case_id')
            variants = plan_recipe_variants(config)
            if config_path.read_bytes() != raw:
                raise ValueError('Input changed while building the dashboard; regenerate the index')
            row.update(config_valid=True, planned_variants=len(variants),
                       coordinate_reviewed=config['roi']['coordinate_reviewed'],
                       named_roi_review={name: roi.get('coordinate_reviewed', False)
                                         for name, roi in config.get('rois', {}).items()},
                       parameters_reviewed=config['train'].get('parameters_reviewed', False),
                       selected_tissue=config['selection']['tissue'],
                       source_ids=list(config['selection']['source_ids']),
                       anatomy_family=config['train']['anatomy_family'],
                       preview=_report_status(config['output']['directory'], current_hash, 'preview'),
                       preflight=_report_status(config['train']['preflight_directory'], current_hash, 'preflight'),
                       prepare=_prepared_status(config['train']['dataset_directory'], current_hash))
            if 'freeze_directory' in config['train']:
                row['freeze'] = _freeze_status(config['train']['freeze_directory'], current_hash)
            else:
                row['freeze'] = {'state': 'not_applicable', 'integrity': 'not_checked'}
            row['published'] = _published_status(registry, case['registry_name'], current_hash)
            row['commands'] = _commands(config_path, config, case, registry, row['published'], root)
            if case.get('localization'):
                row['localization'] = str(_resolve(case['localization'], root))
        except (OSError, ValueError, TypeError, KeyError) as error:
            row.update(config_valid=False, error=str(error))
        cases.append(row)
    return {'schema_version': SCHEMA, 'name': project.get('name', 'Cohort review'),
            'generated_at_utc': datetime.now(timezone.utc).isoformat(), 'project': str(project_path),
            'repo_root': str(root), 'registry': str(registry),
            'mutable_review_index': True, 'voxel_payloads_read': False,
            'generation_dependencies_checked': False, 'cases': cases,
            'case_count': len(cases),
            'published_case_count': sum(row.get('published', {}).get('state') == 'published' for row in cases)}


def render_html(status, output):
    """Render links relative to the dashboard, including native report tabs."""
    output = Path(output).resolve()
    escape = lambda value: html.escape(str(value), quote=True)

    def link(path, label, fragment=''):
        if not Path(path).is_file():
            return escape(label) + ' (not available)'
        href = quote(os.path.relpath(path, output), safe='/') + fragment
        return f'<a href="{escape(href)}">{escape(label)}</a>'

    def state(item):
        value = item['state'].replace('_', ' ')
        detail = f'<span class="badge {escape(item["state"])}">{escape(value)}</span>'
        if item.get('matches_current_input') is True:
            detail += '<small>Current input snapshot; source/code identity not checked.</small>'
        if item.get('matches_current_input') is False:
            detail += '<small>Saved INI differs from current input.</small>'
        if item.get('error'):
            detail += '<small>' + escape(item['error']) + '</small>'
        return detail

    cards = []
    for case in status['cases']:
        title = f'<h2>Case {escape(case["case_id"])}</h2>'
        details = (f'<p>{escape(case["template"])} · provisional family: '
                   f'{escape(case["provisional_family"])}</p>')
        details += '<p>' + link(case['config'], 'Edit case INI') + '</p>'
        if not case['config_valid']:
            cards.append('<section>' + title + details + '<p class="error">' + escape(case['error']) + '</p></section>')
            continue
        review = 'reviewed' if case['coordinate_reviewed'] else 'needs review'
        parameters = 'reviewed' if case['parameters_reviewed'] else 'needs review'
        details += (f'<p><strong>{case["planned_variants"]} planned variants</strong> · '
                    f'ROI: {review} · Parameter outcomes: {parameters} · '
                    f'Tissue: {escape(case["selected_tissue"])}</p>')
        if case['named_roi_review']:
            details += '<p>Named ROI review: ' + ', '.join(
                escape(name) + (' reviewed' if reviewed else ' needs review')
                for name, reviewed in case['named_roi_review'].items()) + '</p>'
        preview = case['preview']['report']
        details += '<p class="links">' + ' · '.join([
            link(preview, 'Full report'), link(preview, 'Global view', '#panel-global'),
            link(preview, 'Local view', '#panel-local'), link(preview, 'Before / after', '#panel-edits')]) + '</p>'
        details += '<table><thead><tr><th>Active draft stage</th><th>Status</th><th>Artifact</th></tr></thead><tbody>'
        for key, label in [('preview', 'ROI preview'), ('preflight', 'Native parameter preflight'),
                           ('freeze', 'Reviewed input freeze'), ('prepare', 'Paired data')]:
            item = case[key]
            artifact = item.get('report', item.get('document'))
            details += f'<tr><td>{label}</td><td>{state(item)}</td><td>' + (
                link(artifact, 'Open') if artifact else '—') + '</td></tr>'
        details += '</tbody></table>'
        published = case['published']
        details += '<div class="published"><strong>Published registry snapshot: </strong>' + state(published)
        if published['state'] == 'published':
            details += '<p>' + link(published['entry'], published['name']) + f' · {published["sample_count"]} pairs · '
            details += 'family identity ' + ('verified' if published['family_verified'] else 'not verified') + '</p>'
            if not published['matches_current_input']:
                details += '<p>This is a separate saved revision; it does not publish the active draft above.</p>'
            if published.get('original_manifest'):
                details += '<p>Imported from ' + escape(published['original_manifest']) + '</p>'
        details += '</div><details><summary>Commands for this case</summary>'
        details += ('<p>Run from the checkout below. Use unused versioned output directories after changes. '
                    'Validate checks metadata; preflight executes all combinations. Review ROI and achieved '
                    'changes, set review flags, then freeze and prepare. Register only completed pairs; '
                    'family verification requires a separate provenance review.</p>')
        details += '<pre>' + escape('cd ' + shlex.quote(status['repo_root'])) + '</pre>'
        for command in ('preview', 'validate', 'preflight', 'freeze', 'prepare', 'register'):
            details += f'<p>{command.capitalize()}</p><pre>' + escape(case['commands'][command]) + '</pre>'
        details += '</details>'
        cards.append('<section>' + title + details + '</section>')
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
            '<title>' + escape(status['name']) + '</title><style>'
            'body{font:16px/1.5 system-ui,sans-serif;background:#eef2f5;color:#1f303c;margin:0}'
            'header,main,footer{max-width:1150px;margin:auto;padding:24px}h1,h2{line-height:1.2}'
            'section{background:white;border:1px solid #d5dde3;border-radius:10px;padding:24px;margin:20px 0}'
            'a{color:#096b91}table{width:100%;border-collapse:collapse}td,th{text-align:left;padding:10px;border-bottom:1px solid #dce3e8}'
            'small{display:block;color:#596776}.badge{display:inline-block;padding:2px 9px;border-radius:15px;background:#e8edf2}'
            '.complete,.published.badge{background:#dcf2e8}.stale,.incomplete{background:#fff0ce}.invalid,.failed,.error{color:#9a2727;background:#ffeded}'
            '.published{padding:12px 16px;background:#f4f7fa;margin:16px 0}pre{padding:12px;background:#edf3f6;white-space:pre-wrap;overflow-wrap:anywhere}'
            'details{margin-top:16px}summary{cursor:pointer;font-weight:600}th{background:#f3f6f8}'
            '@media(max-width:700px){header,main,footer{padding:12px}section{padding:14px}table{font-size:13px}}'
            '</style><header><h1>' + escape(status['name']) + '</h1><p>'
            f'{status["case_count"]} case inputs · {status["published_case_count"]} published snapshots</p>'
            '<p>This is a mutable review index. Refresh it after saving inputs or completing a stage. '
            'Status checks metadata checksums and artifact existence without reading raw volumes or NPZ payloads. '
            'It does not revalidate current generation dependencies or replace native preflight.</p><p>'
            + link(Path(status['repo_root']) / 'docs/MULTIPHANTOM_COHORTS.md', 'Workflow guide')
            + '</p><p>Updated ' + escape(status['generated_at_utc']) + '</p></header><main>'
            + ''.join(cards) + '</main><footer>Review flags are declarations in each INI, not automatic anatomical verification. '
            'No freeze, publication, training, or split assignment is performed by this dashboard.</footer></html>')


def _atomic_write(path, text):
    descriptor, temporary = tempfile.mkstemp(prefix='.' + path.name + '.', dir=path.parent)
    try:
        with os.fdopen(descriptor, 'w', encoding='utf-8') as stream:
            stream.write(text)
        os.replace(temporary, path)
    finally:
        if os.path.exists(temporary):
            os.unlink(temporary)


def write_status(status, output):
    """Replace only the two mutable index files, never immutable case outputs."""
    output = Path(output).expanduser().resolve()
    protected = [Path(status['registry'])]
    for case in status['cases']:
        for stage in ('preview', 'preflight', 'freeze', 'prepare'):
            directory = case.get(stage, {}).get('directory')
            if directory:
                protected.append(Path(directory))
    if any(output == path or path in output.parents for path in protected):
        raise ValueError('Dashboard output must be outside immutable case artifacts and registry directories')
    output.mkdir(parents=True, exist_ok=True)
    _atomic_write(output / 'status.json', json.dumps(status, indent=2, allow_nan=False) + '\n')
    _atomic_write(output / 'index.html', render_html(status, output))


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, default=Path('configs/cohorts/coa/project.json'))
    parser.add_argument('--output', type=Path, default=Path('outputs/cohorts/coa'))
    args = parser.parse_args(argv)
    try:
        status = collect_status(args.project)
        output = _resolve(args.output, ROOT)
        write_status(status, output)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps({'report': str(output / 'index.html'), 'case_count': status['case_count'],
                      'published_case_count': status['published_case_count'],
                      'voxel_payloads_read': False}, indent=2))


if __name__ == '__main__':
    main()
