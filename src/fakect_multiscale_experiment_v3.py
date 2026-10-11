"""Third pilot revision: stable training, explicit resource and anatomy gates.

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
import fakect_multiscale_experiment_v2 as previous
from fakect_multiscale_experiment import (ROOT, SCORE, sha256, write_json, checked_json,
    bbox_target, proposal_origins, coverage, normalized_input, verify_no_cancellation)

SCHEMA = 'fakect.multiscale-pilot/3'
TRAINING_FIELDS = {'locator_learning_rate', 'minimum_target_coverage',
                   'host_memory_limit_mb', 'maximum_traces'}
AUGMENTATION_FIELDS = {'translate_fraction_kji', 'scale_range', 'crop_fraction_max', 'samples_per_case'}
ROBUSTNESS_FIELDS = {'translation_fraction', 'maximum_relative_error'}
CODE_FILES = previous.CODE_FILES + (
    'src/fakect_multiscale_experiment_v3.py', 'src/fakect_multiscale_learning_v3.py',
    'src/fakect_multiscale_network_v3.py', 'src/fakect_multiscale_augmentation.py',
    'scripts/train_multiscale_coa_v3.py', 'scripts/slurm/fakect_multiscale_v3_gpu.sh')


def load_config(path):
    p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                  empty_lines_in_values=False)
    p.read_string(Path(path).read_text())
    if (p.defaults() or set(p.sections()) != set(base.FIELDS) | {'training', 'augmentation', 'robustness'} or
            set(p['training']) != TRAINING_FIELDS or p['experiment']['schema_version'] != SCHEMA or
            set(p['augmentation']) != AUGMENTATION_FIELDS or set(p['robustness']) != ROBUSTNESS_FIELDS or
            set(p['input.detail']) != base.FIELDS['input.detail'] | {'margin_mm_ijk'}):
        raise ValueError('Expected pilot schema /3 and exactly the documented input fields')
    def numbers(section, key, count):
        values = [float(v.strip()) for v in p[section][key].split(',')]
        if len(values) != count or not all(math.isfinite(v) for v in values):
            raise ValueError(f'{section}.{key} needs {count} finite numbers')
        return values
    margin = numbers('input.detail', 'margin_mm_ijk', 3)
    spacing = numbers('input.detail', 'spacing_ijk_mm', 3)
    base_shape = [int(v.strip()) for v in p['input.detail']['shape_kji'].split(',')]
    if len(base_shape) != 3 or any(v < 16 or v % 8 for v in base_shape) or min(margin) < 0 or min(spacing) <= 0:
        raise ValueError('Invalid base detail shape, margin or spacing')
    # Symmetric increments rounded outward to four voxels per side maintain
    # center parity and divisibility by eight. Budget applies AFTER padding.
    pad = [4 * math.ceil(m / s / 4 - 1e-12) for m,s in zip(margin[::-1], spacing[::-1])]
    effective = [n + 2*h for n,h in zip(base_shape,pad)]
    p['input.detail']['shape_kji'] = ','.join(map(str,effective))
    del p['input.detail']['margin_mm_ijk']
    aug = {'translate_fraction_kji':numbers('augmentation','translate_fraction_kji',3),
           'scale_range':numbers('augmentation','scale_range',2),
           'crop_fraction_max':float(p['augmentation']['crop_fraction_max']),
           'samples_per_case':int(p['augmentation']['samples_per_case'])}
    rob = {key:float(p['robustness'][key]) for key in ROBUSTNESS_FIELDS}
    if (not all(0 <= v <= .2 for v in aug['translate_fraction_kji']) or
            not .5 <= aug['scale_range'][0] <= aug['scale_range'][1] <= 1.5 or
            not 0 <= aug['crop_fraction_max'] <= .25 or not 1 <= aug['samples_per_case'] <= 4 or
            not .01 <= rob['translation_fraction'] <= .1 or not 0 < rob['maximum_relative_error'] <= .75):
        raise ValueError('Invalid augmentation or robustness limits')
    p.remove_section('augmentation'); p.remove_section('robustness')
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
    c['input.detail'].update(base_shape_kji=base_shape, margin_mm_ijk=margin,
                            effective_margin_mm_ijk=[h*s for h,s in zip(pad[::-1],spacing)])
    c['augmentation'], c['robustness'] = aug, rob
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
        'locator_initialization':'Only extent bias uses ORIGINAL TRAINING means. Centers come from a spatial heatmap.',
        'locator_loss':'Spatial Gaussian center supervision, extent and coordinate regression.',
        'augmentation':'Only original training overviews are augmented; image, validity and supervised bounds share one transform.',
        'robustness':'Six fixed translation probes per development original; inverse-map predictions for native coverage QA; no test images.',
        'detail_margin':'Symmetric physical margin is rounded outward per axis, and final shape is checked against the input budget.',
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
                           ('runtime-smoke.json','GPU compatibility and trace check'),
                           ('augmentation-audit.json','Original training views and geometric transforms'),
                           ('locator-robustness.json','Image dependence and fixed translation checks')]:
        f = out/filename
        if f.exists():
            value = checked_json(f)
            additional.append('<h2>'+html.escape(title)+'</h2><pre>'+html.escape(json.dumps(value,indent=2))+'</pre>')
    status = checked_json(out/'results.json') if (out/'results.json').exists() else {}
    smoke = checked_json(out/'runtime-smoke.json') if (out/'runtime-smoke.json').exists() else {}
    cov = checked_json(out/'proposal-audit.json') if (out/'proposal-audit.json').exists() else {}
    rows = cov.get('samples',[])
    summary = '<section><h2>Revision three: current result</h2>'
    summary += '<p>Status: <strong>'+html.escape(status.get('status','Awaiting completed results'))+'</strong>. '
    summary += 'CoA classifier trained: '+str(status.get('classifier_trained',False))+'. Test evaluated: false.</p>'
    if rows:
        summary += f'<p>Native coverage passed for {sum(r["coverage_passed"] for r in rows)}/{len(rows)} development scenarios.</p>'
        summary += '<table><tr><th>Scenario</th><th>Before anatomy</th><th>After anatomy</th><th>Removed</th><th>Pass</th></tr>'
        def percent(value):
            return 'N/A' if value is None else f'{100*value:.1f}%'
        for row in rows:
            summary += '<tr>'+''.join('<td>'+html.escape(str(v))+'</td>' for v in
                [row['sample_id'],percent(row['before_coverage']),percent(row['target_coverage']),
                 percent(row['removed_coverage']),row['coverage_passed']])+'</tr>'
        summary += '</table>'
    summary += '<p>Pink outlines in locator figures show actual detail blocks, including the configured symmetric margin. Cyan shows predicted anatomy bounds; yellow is supervision for review.</p>'
    summary += '<p>Fixed translation and blank-attenuation checks use development cases. They are engineering screens, not independent test performance.</p>'
    if smoke:
        summary += '<p>Full-size GPU compatibility check: '+html.escape(smoke.get('status','unknown'))+'.</p>'
    summary += '</section>'
    content = path.read_text()
    content = content.replace('<body>','<body>'+summary,1) if '<body>' in content else content.replace('<h1>',summary+'<h1>',1)
    extra = ''.join(additional)
    content = content.replace('</body>',extra+'</body>',1) if '</body>' in content else content+extra
    path.write_text(content)
    return str(path)
