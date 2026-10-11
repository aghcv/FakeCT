#!/usr/bin/env python3
"""Partial positive-voxel coverage of corrected arterial targets.

No image, model, surface-distance or distance-transform computation is needed.
The three completed held-out families are fixed before this audit executes.
"""
from pathlib import Path
import csv
import json
import shutil
import sys
import tempfile

import nibabel as nib
import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_multiscale_experiment import checked_json, sha256

STUDY = ROOT / 'outputs/experiments/aorta/pretrained-finetune-branches-r3'
FAMILIES = ('xcat_reference_10yr', 'xcat_reference_15yr', 'xcat_reference_1yr')
ARMS = ('pretrained', 'siblings')
CHANNELS = [2, 4, 5, 6, 7, 8]


def read_mask(path, digest, fingerprints, reference=None):
    path = Path(path)
    if sha256(path) != digest: raise ValueError('Frozen coverage input changed: ' + str(path))
    fingerprints[str(path)] = digest
    image = nib.load(path)
    if len(image.shape) != 3 or (reference is not None and (image.shape != reference.shape or
            not np.allclose(image.affine, reference.affine, rtol=0, atol=1e-5))):
        raise ValueError('Coverage inputs must use identical native NIfTI grids and affines')
    value = np.asanyarray(image.dataobj)
    if value.dtype.kind in 'bui':
        if value.min() < 0 or value.max() > 1: raise ValueError('Native masks must be binary')
    elif np.any((value != 0) & (value != 1)):
        raise ValueError('Native masks must be binary')
    return image, value.astype(bool)


def summarize(rows):
    result = {}
    for arm in ARMS:
        records = [r for r in rows if r['arm'] == arm]
        result[arm] = {}
        for component in ('core', 'branches'):
            denom = sum(r[component + '_truth_voxels'] for r in records)
            hits = sum(r[component + '_covered_voxels'] for r in records)
            case_means = [float(np.mean([r[component + '_recall'] for r in records if r['case_id'] == case]))
                          for case in sorted({r['case_id'] for r in records})]
            result[arm][component] = dict(scans=len(records), cases=len(case_means),
                mean_scan_recall=float(np.mean([r[component + '_recall'] for r in records])),
                case_balanced_recall=float(np.mean(case_means)),
                pooled_voxel_recall=hits / denom, truth_voxels=denom, covered_voxels=hits)
    return result


def main():
    lock_path = STUDY / 'study-lock.json'
    lock = checked_json(lock_path); lock_hash = sha256(lock_path)
    if lock['foreground_channels'] != CHANNELS: raise ValueError('Unexpected corrected foreground group')
    destination = STUDY / 'coverage-preview-r1'
    if destination.exists(): raise FileExistsError('Do not overwrite an existing coverage preview')
    temporary = Path(tempfile.mkdtemp(prefix='.coverage-preview-r1-', dir=STUDY))
    fingerprints = {str(lock_path): lock_hash}
    rows = []
    try:
        for family in FAMILIES:
            fold = next(f for f in lock['folds'] if f['id'] == family)
            predictions = {}
            for arm in ARMS:
                path = STUDY / 'evaluation' / family / arm / 'results.json'
                done = checked_json(path)
                if (done.get('complete') is not True or (path.parent / 'INCOMPLETE').exists() or
                        done['study_lock_sha256'] != lock_hash or done['arm'] != arm or done['fold'] != family):
                    raise ValueError('Completed, matched and checksum-verified prediction sets are required')
                fingerprints[str(path)] = sha256(path)
                predictions[arm] = {r['sample_id']: r for r in done['samples']}
            for case in fold['test_cases']:
                path = STUDY / 'cases' / case / 'prepared.json'; prepared = checked_json(path)
                fingerprints[str(path)] = sha256(path)
                for sample_id in lock['rosters']['evaluation'][case]:
                    native = prepared['samples'][sample_id]['native']
                    components = native['target_components']
                    core_image, core = read_mask(components['aorta_core']['path'], components['aorta_core']['sha256'], fingerprints)
                    _, branches = read_mask(components['connected_branches']['path'], components['connected_branches']['sha256'], fingerprints, core_image)
                    if np.any(core & branches): raise ValueError('Core and connected branches must be disjoint')
                    core_count, branch_count = int(core.sum()), int(branches.sum())
                    if core_count == 0 or branch_count == 0: raise ValueError('Both anatomical components must be present')
                    for arm in ARMS:
                        record = predictions[arm][sample_id]
                        if record['status'] != 'completed' or record['inference']['foreground_channels'] != CHANNELS:
                            raise ValueError('Each completed arm must extract the same corrected foreground group')
                        _, prediction = read_mask(record['prediction_nifti'], record['prediction_nifti_sha256'], fingerprints, core_image)
                        core_hits, branch_hits = int(prediction[core].sum()), int(prediction[branches].sum())
                        rows.append(dict(sample_id=sample_id, case_id=case, family=family,
                            baseline=bool(native['baseline']), change_kind=native['change_kind'], arm=arm,
                            core_truth_voxels=core_count, core_covered_voxels=core_hits, core_recall=core_hits / core_count,
                            branches_truth_voxels=branch_count, branches_covered_voxels=branch_hits,
                            branches_recall=branch_hits / branch_count, frame_match_passed=True,
                            foreground_channels=CHANNELS))
                    del core, branches, prediction
                print('Audited component coverage', family, case, flush=True)
        unique = {r['sample_id'] for r in rows}
        if len(unique) != 78 or len(rows) != 156: raise ValueError('The fixed partial audit must contain 78 paired scans')
        families = {f: summarize([r for r in rows if r['family'] == f]) for f in FAMILIES}
        payload = dict(schema_version='fakect.partial-branch-coverage/1', status='PARTIAL',
            paired_scans=78, paired_cases=6, completed_families=list(FAMILIES), planned_study_scans=130,
            foreground_channels=CHANNELS, scope='Positive-voxel recall on corrected native aortic core and connected branches. '
                'Known-region synthetic benchmark; precision, Dice, surface metrics and clinical accuracy are not evaluated here.',
            metric_definitions={'core_recall': 'Predicted-positive core voxels / native reference core voxels',
                                'branches_recall': 'Predicted-positive branch voxels / native reference branch voxels'},
            rows=rows, summary=summarize(rows), families=families,
            baseline_controls=summarize([r for r in rows if r['baseline']]),
            modified_siblings=summarize([r for r in rows if not r['baseline']]),
            independence='Equal 13-scan evaluation rosters per case. Siblings are correlated; no patient-level confidence intervals are calculated.')
        (temporary / 'coverage.json').write_text(json.dumps(payload, indent=2, allow_nan=False) + '\n')
        with (temporary / 'coverage.csv').open('w', newline='') as stream:
            fields = [k for k in rows[0] if k != 'foreground_channels']
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
            writer.writerows({k: r[k] for k in fields} for r in rows)
        shutil.copyfile(Path(__file__), temporary / 'coverage-preview-generator.py')
        outputs = {p.name: sha256(p) for p in temporary.iterdir() if p.is_file()}
        (temporary / 'provenance.json').write_text(json.dumps(dict(schema_version='fakect.partial-branch-coverage/1',
            input_sha256=fingerprints, output_sha256=outputs, generator_sha256=sha256(Path(__file__)),
            models_run=False, images_read=False, surface_metrics_computed=False, distance_transforms_computed=False), indent=2) + '\n')
        temporary.rename(destination)
        print(json.dumps(dict(directory=str(destination), summary=payload['summary'], families=families), indent=2), flush=True)
    except Exception:
        shutil.rmtree(temporary, ignore_errors=True)
        raise


if __name__ == '__main__': main()
