"""Second pilot revision: stable training, explicit resource and anatomy gates.

Reuse the immutable data/annotation contracts from revision one without
changing its pinned implementation or results.
"""
import configparser
from copy import deepcopy
import hashlib
import html
import io
import json
import math
from pathlib import Path
import tempfile

import fakect_multiscale_experiment as base
from fakect_multiscale_experiment import (ROOT, SCORE, sha256, write_json, checked_json,
    bbox_target, proposal_origins, coverage, normalized_input, verify_no_cancellation)

SCHEMA = 'fakect.multiscale-pilot/2'
TRAINING_FIELDS = {'locator_learning_rate', 'minimum_target_coverage',
                   'host_memory_limit_mb', 'maximum_traces'}
CODE_FILES = base.CODE_FILES + (
    'src/fakect_multiscale_experiment_v2.py', 'src/fakect_multiscale_learning_v2.py',
    'src/fakect_multiscale_network_v2.py', 'src/fakect_tensorflow_steps.py',
    'scripts/research/train_multiscale_coa_v2.py', 'scripts/research/slurm/fakect_multiscale_v2_gpu.sh')


def load_config(path):
    p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                  empty_lines_in_values=False)
    p.read_string(Path(path).read_text())
    if (p.defaults() or set(p.sections()) != set(base.FIELDS) | {'training'} or
            set(p['training']) != TRAINING_FIELDS or p['experiment']['schema_version'] != SCHEMA):
        raise ValueError('Expected pilot schema /2 and the documented [training] fields')
    training = dict(p['training'])
    p.remove_section('training')
    p['experiment']['schema_version'] = base.SCHEMA
    stream = io.StringIO()
    p.write(stream)
    # Reuse the exact original input validation rather than relax its parser.
    with tempfile.TemporaryDirectory(prefix='fakect-pilot-config-') as directory:
        adapted = Path(directory)/'input.ini'
        adapted.write_text(stream.getvalue())
        c = base.load_config(adapted)
    c['experiment']['schema_version'] = SCHEMA
    c['training'] = {
        'locator_learning_rate':float(training['locator_learning_rate']),
        'minimum_target_coverage':float(training['minimum_target_coverage']),
        'host_memory_limit_mb':int(training['host_memory_limit_mb']),
        'maximum_traces':int(training['maximum_traces'])}
    t = c['training']
    if (not math.isfinite(t['locator_learning_rate']) or t['locator_learning_rate'] <= 0 or
            not 0 < t['minimum_target_coverage'] <= 1 or
            not 512 <= t['host_memory_limit_mb'] <= 12288 or t['maximum_traces'] != 1):
        raise ValueError('Invalid training rate, coverage, host memory cap or trace limit')
    return c


def plan(c):
    result = base.plan(c)
    result.update(schema_version=SCHEMA, training_contract={
        'locator_initialization':'Mean bounds from ORIGINAL TRAINING cases only; compare with this constant prior.',
        'locator_loss':'Stable soft-target cross entropy on raw bounded-box parameters plus direct coordinate error.',
        'training_steps':'Fixed TensorSpec GradientTape functions; optimizer slots built before tracing.',
        'coverage_gate':'Minimum before/after anatomy coverage for EVERY sample, plus full removal coverage and change fraction.',
        'resource_guard':'Linux resident memory sampled around each batch; stop at configured host threshold, not a kernel-enforced host limit.'})
    return result


def freeze(c, input_bytes):
    document = plan(c)
    sources = {name:(ROOT/name).read_bytes() for name in CODE_FILES}
    out = Path(c['experiment']['output_directory'])
    out.mkdir(parents=True, exist_ok=False)
    (out/'INCOMPLETE').touch()
    (out/'input.ini').write_bytes(input_bytes)
    hashes = {}
    for name, content in sources.items():
        target = out/'source'/name
        target.parent.mkdir(parents=True, exist_ok=True)
        target.write_bytes(content)
        hashes[name] = hashlib.sha256(content).hexdigest()
    document.update(code_sha256=hashes, input_sha256=hashlib.sha256(input_bytes).hexdigest())
    write_json(out/'experiment-lock.json', document)
    (out/'INCOMPLETE').unlink()
    return document


def inspect_lock(c, input_bytes):
    out = Path(c['experiment']['output_directory'])
    lock = checked_json(out/'experiment-lock.json')
    if (lock['schema_version'] != SCHEMA or lock['config'] != c or
            lock['input_sha256'] != hashlib.sha256(input_bytes).hexdigest() or
            sha256(out/'input.ini') != lock['input_sha256']):
        raise ValueError('Configuration changed; use a new revision')
    if set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Source inventory changed')
    for name,digest in lock['code_sha256'].items():
        if sha256(ROOT/name) != digest or sha256(out/'source'/name) != digest:
            raise ValueError('Pinned implementation changed: '+name)
    for key,name in [('manifest','dataset_sha256'),('annotations','annotation_sha256')]:
        checked_json(c['data'][key],lock[name])
    return lock


def coverage_gate(record, c):
    minimum = c['training']['minimum_target_coverage']
    return (record['target_coverage'] is not None and record['target_coverage'] >= minimum and
            record['before_coverage'] is not None and record['before_coverage'] >= minimum and
            (record['changed_coverage'] is None or record['changed_coverage'] >= c['pilot']['minimum_changed_coverage']) and
            (record['removed_coverage'] is None or record['removed_coverage'] == 1.0))


def report(c):
    out = Path(c['experiment']['output_directory'])
    path = Path(base.report(c))
    additional = []
    for filename,title in [('locator-diagnostics.json','Locator compared with training prior'),
                           ('training-runtime.json','Memory and tracing'),
                           ('runtime-smoke.json','GPU compatibility and trace check')]:
        f = out/filename
        if f.exists():
            value = checked_json(f)
            additional.append('<h2>'+html.escape(title)+'</h2><pre>'+html.escape(json.dumps(value,indent=2))+'</pre>')
    path.write_text(path.read_text()+''.join(additional))
    return str(path)
