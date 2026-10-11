#!/usr/bin/env python3
"""Resolve anatomy and execute native cohort preflight, with conservative resume.

Existing case artifacts are never removed or overwritten. Incomplete, stale or
failed outputs require a reviewed new revision. This runner does not freeze an
input, export training pairs, publish a dataset, or train a model.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
import fcntl
import hashlib
import json
import os
from pathlib import Path, PurePosixPath
import re
import signal
import subprocess
import sys
import threading
import time
import uuid

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_study_config import load_study_config
from fakect_roi import resolve_preview
from fakect_recipe_training import _fingerprint

SCHEMA = 'fakect.cohort-batch/1'
HEX = re.compile(r'[0-9a-f]{64}\Z')
CASE_ID = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')


def now():
    return datetime.now(timezone.utc).isoformat()


def sha(path):
    digest = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def resolve_path(value, root):
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_project(path, root=ROOT, cases=None):
    path = resolve_path(path, root)
    raw = path.read_bytes()
    project = json.loads(raw)
    if (not isinstance(project, dict) or project.get('schema_version') != 'fakect.cohort-project/1'
            or not isinstance(project.get('cases'), list) or not project['cases']):
        raise ValueError('Expected a nonempty fakect.cohort-project/1 manifest')
    selected = set(cases or ())
    rows, identifiers, paths = [], set(), set()
    for row in project['cases']:
        if not isinstance(row, dict):
            raise ValueError('Project cases must be objects')
        identifier = str(row.get('case_id', ''))
        if not CASE_ID.fullmatch(identifier) or identifier in identifiers:
            raise ValueError(f'Invalid or duplicate case_id: {identifier}')
        config = resolve_path(row['config'], root)
        if config in paths:
            raise ValueError('Project cases must have distinct configuration paths')
        identifiers.add(identifier)
        paths.add(config)
        if not selected or identifier in selected:
            rows.append({'case_id': identifier, 'config': str(config)})
    if selected - identifiers:
        raise ValueError(f'Unknown requested case IDs: {sorted(selected - identifiers)}')
    return project, rows, hashlib.sha256(raw).hexdigest()


def verify_preflight(folder, config, resolved):
    """Verify all existing report files and current generation identity; no voxels."""
    folder = Path(folder).resolve()
    if (folder / 'INCOMPLETE').exists():
        raise ValueError('Preflight is INCOMPLETE; preserve it and choose a new output revision')
    inventory = json.loads((folder / 'artifact-manifest.json').read_bytes())
    required = {'preflight.json', 'preflight.html', 'input.ini', 'data-config.json', 'source-crop.json'}
    if not isinstance(inventory, dict) or not required.issubset(inventory):
        raise ValueError('Preflight inventory does not cover required artifacts')
    for name, checksum in inventory.items():
        parts = PurePosixPath(name)
        if (not name or '\\' in name or parts.is_absolute()
                or any(part in ('', '.', '..') for part in name.split('/'))
                or not isinstance(checksum, str) or not HEX.fullmatch(checksum)):
            raise ValueError(f'Unsafe preflight artifact entry: {name}')
        path = folder
        for part in parts.parts:
            path /= part
            if path.is_symlink():
                raise ValueError(f'Symlink preflight artifact: {name}')
        if not path.is_file() or sha(path) != checksum:
            raise ValueError(f'Preflight artifact missing or changed: {name}')
    document = json.loads((folder / 'preflight.json').read_bytes())
    if (document.get('schema_version') != 'fakect.recipe-preflight/1'
            or document.get('complete') is not True or document.get('execution_complete') is not True):
        raise ValueError('Preflight is incomplete or has an unsupported schema')
    if document.get('input_config_sha256') != sha(folder / 'input.ini'):
        raise ValueError('Preflight input snapshot hash disagrees with its report')
    if document.get('dataset_fingerprint') != _fingerprint(config, resolved):
        raise ValueError('Preflight settings, sources or generation code are stale; choose a new revision')
    fields = ('variant_count', 'successful_variants', 'failed_variants', 'execution_valid',
              'cohort_ready', 'fit_ready', 'duplicate_variant_count', 'dataset_fingerprint')
    summary = {key: document.get(key) for key in fields}
    summary['report'] = str(folder / 'preflight.html')
    summary['ready'] = (document.get('execution_valid') is True
                        and document.get('failed_variants') == 0
                        and (document.get('cohort_ready') is True or document.get('fit_ready') is True))
    return summary


def execute(command, log_path, stop, root=ROOT):
    """Bound numerical worker threads; each native command keeps a separate log."""
    environment = dict(os.environ)
    for key in ('OMP_NUM_THREADS', 'OPENBLAS_NUM_THREADS', 'MKL_NUM_THREADS', 'NUMEXPR_NUM_THREADS'):
        environment[key] = '1'
    with Path(log_path).open('x', encoding='utf-8') as log:
        log.write(json.dumps({'command': command, 'started_at': now()}) + '\n')
        log.flush()
        process = subprocess.Popen(command, cwd=root, env=environment, stdout=log,
                                   stderr=subprocess.STDOUT, start_new_session=True)
        while True:
            try:
                return process.wait(timeout=1)
            except subprocess.TimeoutExpired:
                if stop.is_set():
                    os.killpg(process.pid, signal.SIGTERM)
                    try:
                        return process.wait(timeout=10)
                    except subprocess.TimeoutExpired:
                        os.killpg(process.pid, signal.SIGKILL)
                        return process.wait()


def run_case(row, output, run_id, through, python, update, stop, root=ROOT):
    case_id, path = row['case_id'], Path(row['config'])
    started = time.monotonic()
    stage = 'config'
    try:
        update(case_id, state='running', started_at=now())
        raw = path.read_bytes()
        config = load_study_config(path, repo_root=root)
        if (str(config['input']['case_id']) != case_id
                or config['study']['schema_version'] != 'fakect.recipe-cohort/1'):
            raise ValueError('Project case ID or generation-only cohort schema disagrees with INI')
        if not config['selection'].get('anatomy') or not config['selection'].get('anatomy_file'):
            raise ValueError('Batch anatomy resolution requires selection.anatomy and anatomy_file')
        if path.read_bytes() != raw:
            raise ValueError('Input changed while loading')
        update(case_id, input_sha256=hashlib.sha256(raw).hexdigest())
        anatomy = Path(config['selection']['anatomy_file'])
        stage = 'anatomy'
        if stop.is_set():
            raise InterruptedError('Batch interrupted before anatomy resolution')
        if (anatomy.parent / 'INCOMPLETE').exists():
            raise ValueError('Anatomy selection is INCOMPLETE; preserve it and choose a new revision')
        if anatomy.exists():
            resolved = resolve_preview(config)
            update(case_id, anatomy={'state': 'reused', 'path': str(anatomy),
                                    'selected_source_ids': list(resolved['source_ids'])})
        else:
            if anatomy.parent.exists() and (not anatomy.parent.is_dir() or any(anatomy.parent.iterdir())):
                raise ValueError('Anatomy output folder is populated without a valid selection; choose a new revision')
            command = [python, str(root / 'scripts/resolve_anatomy.py'), '--config', str(path)]
            log = output / f'{case_id}.{run_id}.anatomy.log'
            update(case_id, anatomy={'state': 'running', 'log': str(log), 'command': command})
            code = execute(command, log, stop, root)
            if code:
                update(case_id, anatomy={'state': 'failed', 'exit_code': code, 'log': str(log)})
                raise RuntimeError(f'Anatomy command exited {code}; inspect {log}')
            if path.read_bytes() != raw:
                raise ValueError('Input changed during anatomy execution; preserve saved artifacts and rerun a new revision')
            resolved = resolve_preview(config)
            update(case_id, anatomy={'state': 'complete', 'exit_code': code, 'log': str(log),
                                    'path': str(anatomy), 'selected_source_ids': list(resolved['source_ids'])})
        if through == 'preflight':
            stage = 'preflight'
            if stop.is_set():
                raise InterruptedError('Batch interrupted before native preflight')
            folder = Path(config['train']['preflight_directory'])
            if folder.exists():
                summary = verify_preflight(folder, config, resolved)
                update(case_id, preflight={'state': 'reused' if summary['ready'] else 'failed', **summary})
                if not summary['ready']:
                    raise RuntimeError('Existing matching preflight contains unsuccessful native outcomes; review a new revision')
            else:
                command = [python, str(root / 'scripts/prepare_cohort.py'), '--config', str(path), '--stage', 'preflight']
                log = output / f'{case_id}.{run_id}.preflight.log'
                update(case_id, preflight={'state': 'running', 'log': str(log), 'command': command})
                code = execute(command, log, stop, root)
                # A complete but unsuccessful preflight is still useful for review.
                try:
                    summary = verify_preflight(folder, config, resolved)
                except (ValueError, OSError, KeyError, TypeError) as error:
                    update(case_id, preflight={'state': 'failed', 'exit_code': code, 'log': str(log)})
                    raise RuntimeError(f'Native preflight exit {code}, output could not be verified: {error}; inspect {log}') from error
                update(case_id, preflight={'state': 'complete' if code == 0 and summary['ready'] else 'failed',
                                           'exit_code': code, 'log': str(log), **summary})
                if code or not summary['ready']:
                    raise RuntimeError(f'Native preflight needs review (exit {code}); inspect {summary["report"]}')
        if path.read_bytes() != raw:
            raise ValueError('Input changed during execution; outputs describe their captured input.ini')
        update(case_id, state='complete', completed_at=now(), elapsed_seconds=round(time.monotonic() - started, 3))
    except Exception as error:
        # Isolate failed phantoms so other cases can still produce review evidence.
        update(case_id, state='interrupted' if stop.is_set() else 'failed', failed_stage=stage,
               error_type=type(error).__name__, error=str(error), completed_at=now(),
               elapsed_seconds=round(time.monotonic() - started, 3))


def run(project_path, *, output=None, cases=None, workers=1, through='preflight', python=sys.executable, root=ROOT):
    if workers not in (1, 2) or through not in ('anatomy', 'preflight'):
        raise ValueError('workers must be 1 or 2; through must be anatomy or preflight')
    root = Path(root).resolve()
    project_path = resolve_path(project_path, root)
    project, rows, project_sha = load_project(project_path, root, cases)
    # Reject shared destinations before scheduling either writer. A malformed INI
    # remains an isolated per-case failure and does not prevent valid cases running.
    destinations = {}
    for row in rows:
        try:
            config = load_study_config(row['config'], repo_root=root)
        except (ValueError, OSError, KeyError, TypeError):
            continue
        selected = config.get('selection', {}).get('anatomy_file')
        folders = ([Path(selected).parent] if selected else [])
        if through == 'preflight' and config.get('train', {}).get('preflight_directory'):
            folders.append(config['train']['preflight_directory'])
        for folder in folders:
            folder = Path(folder).resolve()
            if folder in destinations:
                raise ValueError(f'Cases {destinations[folder]} and {row["case_id"]} share an output directory: {folder}')
            destinations[folder] = row['case_id']
    directory = resolve_path(output or project.get('batch_directory')
                             or f'outputs/cohorts/coa/{project_path.stem}-batch', root)
    directory.mkdir(parents=True, exist_ok=True)
    run_id = datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%S') + '-' + uuid.uuid4().hex[:8]
    manifest = {'schema_version': SCHEMA, 'run_id': run_id, 'project': str(project_path),
                'project_sha256': project_sha, 'started_at': now(), 'through': through,
                'workers': workers, 'state': 'running', 'dataset_prepared': False, 'input_frozen': False,
                'cases': {row['case_id']: {**row, 'state': 'pending'} for row in rows}}
    mutex, stop = threading.Lock(), threading.Event()
    with (directory / '.batch.lock').open('a') as lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as error:
            raise ValueError('Another batch owns this status directory; choose a separate --output for each Slurm task') from error

        def save():
            encoded = json.dumps(manifest, indent=2, allow_nan=False) + '\n'
            temporary = directory / f'.{run_id}.tmp'
            temporary.write_text(encoded)
            os.replace(temporary, directory / 'status.json')
            # Keep every run's final/intermediate history separate from current status.
            temporary.write_text(encoded)
            os.replace(temporary, directory / f'status-{run_id}.json')

        def update(identifier, **values):
            with mutex:
                manifest['cases'][identifier].update(values)
                manifest['updated_at'] = now()
                save()
            if values.get('state') in ('complete', 'failed', 'interrupted'):
                print(f'{identifier}: {values["state"]}' + (f' — {values["error"]}' if 'error' in values else ''), flush=True)

        save()
        previous_handlers = {}
        if threading.current_thread() is threading.main_thread():
            def interrupted(signum, frame):
                stop.set()
            for signum in (signal.SIGINT, signal.SIGTERM):
                previous_handlers[signum] = signal.signal(signum, interrupted)
        try:
            with ThreadPoolExecutor(max_workers=workers) as pool:
                futures = [pool.submit(run_case, row, directory, run_id, through, python, update, stop, root)
                           for row in rows]
                for future in as_completed(futures):
                    future.result()
        finally:
            for signum, previous in previous_handlers.items():
                signal.signal(signum, previous)
        if sha(project_path) != project_sha:
            manifest['error'] = 'Project changed during execution; this run used the saved case list and project hash'
        failed = sum(row['state'] != 'complete' for row in manifest['cases'].values())
        manifest.update(state='interrupted' if stop.is_set() else 'failed' if failed or 'error' in manifest else 'complete',
                        failed_cases=failed, completed_at=now())
        save()
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, help='Status/log directory; use a distinct directory per Slurm array task')
    parser.add_argument('--cases', nargs='+', help='Optional case IDs; remaining project cases are not executed')
    parser.add_argument('--workers', type=int, choices=(1, 2), default=1)
    parser.add_argument('--through', choices=('anatomy', 'preflight'), default='preflight')
    parser.add_argument('--python', default=sys.executable, help='Python interpreter for native child commands')
    args = parser.parse_args()
    try:
        result = run(args.project, output=args.output, cases=args.cases, workers=args.workers,
                     through=args.through, python=args.python)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')
    print(json.dumps({key: result[key] for key in ('run_id', 'state', 'failed_cases', 'through')}), flush=True)
    if result['state'] != 'complete':
        parser.exit(2)


if __name__ == '__main__':
    main()
