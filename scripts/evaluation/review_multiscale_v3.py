#!/usr/bin/env python3
"""Recover evaluation from a frozen v3 locator without retraining or overwrites.

The original run may have failed while rendering previews after training. This
tool verifies its pinned sources, input and preparation, evaluates the already
selected checkpoint on the same development panel and writes a separate review.
No test payloads, optimizers, training steps or target-directed crops are used.
"""
import argparse
from copy import deepcopy
from datetime import datetime, timezone
import json
import html
from pathlib import Path
import sys
import time
import traceback

import numpy as np

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import fakect_multiscale_experiment_v3 as exp
import fakect_multiscale_learning_v3 as learning
from fakect_context_volume import ContextVolume


def report(c, provenance):
    path = Path(exp.report(c))
    note = ('<section><h2>Recovered checkpoint evaluation</h2><p>The original job completed locator training '
        'and then failed while drawing detail blocks. This separate review evaluates its saved checkpoint '
        'without retraining. No CoA classifier has been trained in this review.</p><p>Selected epoch: '
        +str(provenance['selected_epoch'])+'. Original experiment: '
        +html.escape(provenance['parent_experiment'])+'</p><details><summary>Recovery provenance</summary><pre>'
        +html.escape(json.dumps(provenance,indent=2))+'</pre></details></section>')
    path.write_text(path.read_text().replace('<h1>',note+'<h1>',1))


def review(config_path, output_directory, *, cpu=False):
    config_path, output_directory = Path(config_path), Path(output_directory).resolve()
    c = exp.load_config(config_path)
    parent = Path(c['experiment']['output_directory'])
    lock = exp.inspect_lock(c,config_path.read_bytes())
    prepared = exp.checked_json(parent/'preparation.json')
    history = exp.checked_json(parent/'locator-history.json')
    if not prepared.get('complete') or prepared['lock_sha256'] != exp.sha256(parent/'experiment-lock.json'):
        raise ValueError('Preparation does not match the frozen experiment')
    rows = lock['samples']
    if not rows or any(r['split'] not in ('train','validation') for r in rows):
        raise ValueError('Recovery evaluation accepts only development samples')
    ids = [r['sample_id'] for r in rows]
    prepared_ids = [r['sample_id'] for r in prepared['samples']]
    if len(set(ids)) != len(ids) or len(set(prepared_ids)) != len(prepared_ids) or set(ids) != set(prepared_ids):
        raise ValueError('Prepared samples must match the selected panel exactly')
    if output_directory == parent or output_directory.exists():
        raise ValueError('Use a new, unused review directory')
    checkpoint = parent/'model/locator.keras'
    checkpoint_hash = exp.sha256(checkpoint)
    best = min(history['epochs'],key=lambda r:r['validation_loss'])
    targets = {r['sample_id']:np.asarray(r['target_box_kji'],np.float32)[None] for r in prepared['samples']}
    extents = {r['sample_id']:np.asarray(c['input.overview']['shape_kji'])*
               np.asarray(r['overview_spacing_ijk_mm'][::-1]) for r in prepared['samples']}
    output_directory.mkdir(parents=True)
    (output_directory/'previews').mkdir()
    # This copy is provenance for reporting, not a new trainable experiment.
    exp.write_json(output_directory/'experiment-lock.json',lock)
    sources = ['scripts/evaluation/review_multiscale_v3.py','src/fakect_multiscale_preview.py',
               'scripts/slurm/fakect_multiscale_review_gpu.sh']
    for name in sources:
        path = output_directory/'source'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes((ROOT/name).read_bytes())
    provenance = {'schema_version':'fakect.multiscale-review/1','created_utc':datetime.now(timezone.utc).isoformat(),
        'parent_experiment':str(parent),'parent_lock_sha256':exp.sha256(parent/'experiment-lock.json'),
        'checkpoint':str(checkpoint),'checkpoint_sha256':checkpoint_hash,'selected_epoch':best['epoch'],
        'source_sha256':{name:exp.sha256(ROOT/name) for name in sources},
        'evaluation_only':True,'test_evaluated':False,'cpu_requested':cpu,
        'note':'Preserved original failed run. Reused its selected checkpoint and exact coverage/robustness gates; no retraining.'}
    exp.write_json(output_directory/'review-provenance.json',provenance)
    report_config = deepcopy(c)
    report_config['experiment']['output_directory'] = str(output_directory)
    runtime_config = deepcopy(report_config)
    if cpu:
        runtime_config['pilot']['require_gpu'] = False
    started = time.monotonic()
    try:
        tf,network = learning.configure_tensorflow(runtime_config)
        guard = learning.RuntimeGuard(runtime_config,tf)
        guard.check('review-start',save=True)
        locator = network.load_model(checkpoint)
        if not all(np.isfinite(v.numpy()).all() for v in locator.weights):
            raise ValueError('Nonfinite locator checkpoint weights')
        inputs = learning.Inputs(c)
        predictions,records,failures = {},[],[]
        for row in rows:
            key = row['sample_id']; ov,x = inputs.coarse(row)
            predicted = network.predict_boxes(locator,x)[0]
            predictions[key] = predicted
            origins,boxes = exp.proposal_origins(predicted,ov,c)
            with ContextVolume(row,verify_hashes=True) as volume:
                exp.verify_no_cancellation(volume,row)
                coverage = exp.coverage(volume,origins,c)
            passed = exp.coverage_gate(coverage,c)
            if not passed:
                failures.append(key)
            records.append({'sample_id':key,'split':row['split'],'predicted_box':predicted.tolist(),
                'origins_kji':origins,'proposal_boxes':boxes.tolist(),**coverage,'coverage_passed':passed,
                'center_error_mm':float(np.linalg.norm((predicted[:3]-targets[key][0,:3])*extents[key]))})
            guard.check('review-proposal',save=True)
            print(json.dumps({'stage':'review-coverage','sample_id':key,'passed':passed,
                              'completed':len(records),'total':len(rows)}),flush=True)
        exp.write_json(output_directory/'proposal-audit.json',{'passed':not failures,'failed_sample_ids':failures,
            'samples':records,'proof':'Original no-cancellation and full-removal gates unchanged.',
            'inference':'Saved image-only locator. Native masks audit, never recenter, its proposals.'})
        originals = [r for r in rows if r['baseline']]
        diagnostics = exp.checked_json(parent/'locator-diagnostics.json')
        for split,key in [('train','learned_training'),('validation','learned_validation')]:
            diagnostics[key] = learning.box_metrics([r for r in originals if r['split']==split],
                                                   predictions,targets,extents)
            recorded = best['training_geometry' if split=='train' else 'validation_geometry']
            if abs(recorded['mean_center_error_mm']-diagnostics[key]['mean_center_error_mm']) > .01:
                raise ValueError('Loaded checkpoint differs from the selected history geometry')
        diagnostics['selected_epoch'] = best['epoch']
        exp.write_json(output_directory/'locator-diagnostics.json',diagnostics)
        robustness = learning.robustness_audit(locator,network,originals,inputs,targets,extents,predictions,c,guard=guard)
        exp.write_json(output_directory/'locator-robustness.json',robustness)
        ready = not failures and robustness['passed']
        result = {'status':'locator_ready_for_classifier' if ready else 'locator_review_failed',
            'evaluation_only':True,'classifier_trained':False,'test_evaluated':False,
            'coverage_passed':not failures,'passing_samples':len(rows)-len(failures),'total_samples':len(rows),
            'failed_samples':failures,'robustness_passed':robustness['passed'],'selected_epoch':best['epoch'],
            'training':diagnostics['learned_training'],'validation':diagnostics['learned_validation'],
            'tensorflow_version':tf.__version__,'elapsed_seconds':time.monotonic()-started,
            'checkpoint_sha256':checkpoint_hash,**learning.memory_snapshot(tf)}
        # Commit numerical results BEFORE optional rendering. A figure failure
        # must not discard the completed native-space audit again.
        exp.write_json(output_directory/'results.json',result)
        from fakect_multiscale_preview import preview
        by_id = {r['sample_id']:r for r in records}
        errors = []
        for row in originals:
            try:
                ov,_ = inputs.coarse(row)
                proposal = by_id[row['sample_id']]
                preview(row,ov,targets[row['sample_id']][0],
                    output_directory/'previews'/(row['case_id']+'-locator.png'),
                    predicted=proposal['predicted_box'],proposal_boxes=proposal['proposal_boxes'])
            except Exception as error:
                errors.append({'sample_id':row['sample_id'],'type':type(error).__name__,'message':str(error)})
        result['preview_errors'] = errors
        guard.check('review-complete',save=True)
        exp.inspect_lock(c,config_path.read_bytes())
        if exp.sha256(checkpoint) != checkpoint_hash:
            raise ValueError('Checkpoint changed during evaluation')
        for name,digest in provenance['source_sha256'].items():
            if exp.sha256(ROOT/name) != digest or exp.sha256(output_directory/'source'/name) != digest:
                raise ValueError('Recovery source changed during evaluation: '+name)
        result.update(elapsed_seconds=time.monotonic()-started,**learning.memory_snapshot(tf))
        exp.write_json(output_directory/'results.json',result)
        report(report_config,provenance)
        return result
    except Exception as error:
        exp.write_json(output_directory/'failure.json',{'type':type(error).__name__,'message':str(error),
            'traceback':traceback.format_exc(),'evaluation_only':True})
        if (output_directory/'results.json').exists():
            incomplete = exp.checked_json(output_directory/'results.json')
            incomplete.update(status='review_incomplete',error=str(error))
            exp.write_json(output_directory/'results.json',incomplete)
        report(report_config,provenance)
        raise


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config',required=True)
    parser.add_argument('--output-dir',required=True)
    parser.add_argument('--cpu',action='store_true',help='Allow CPU inference for small development tests')
    args = parser.parse_args()
    result = review(args.config,args.output_dir,cpu=args.cpu)
    print(json.dumps(result,indent=2))


if __name__ == '__main__':
    main()
