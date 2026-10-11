#!/usr/bin/env python3
"""Diagnose coarse localization using the unchanged official fine aorta model."""
import argparse
import configparser
import hashlib
import json
import os
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
sys.path.insert(0, str(ROOT / 'scripts/research'))
from benchmark_pretrained_aorta import load_config, inspect
from fakect_multiscale_experiment import checked_json, sha256, write_json
from fakect_pretrained_report import write_report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--config', required=True)
    args = parser.parse_args()
    path = Path(args.config)
    p = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',))
    p.read_string(path.read_text())
    if p.defaults() or p.sections() != ['diagnostic'] or set(p['diagnostic']) != {
            'parent_directory', 'output_directory', 'cases', 'xy_context_mm'}:
        raise ValueError('Unexpected focus diagnostic fields')
    c = dict(p['diagnostic'])
    for key in ('parent_directory', 'output_directory'):
        value = Path(c[key]); c[key] = str((value if value.is_absolute() else ROOT / value).resolve())
    cases = [v.strip() for v in c['cases'].split(',')]
    context = float(c['xy_context_mm'])
    if not np.isfinite(context) or not 10 <= context <= 80 or len(set(cases)) != len(cases):
        raise ValueError('Use distinct cases and fixed xy_context_mm in [10,80]')
    parent = Path(c['parent_directory'])
    pc = load_config(parent / 'input.ini')
    lock = inspect(pc, (parent / 'input.ini').read_bytes(), check_model=True)
    completed = checked_json(parent / 'benchmark-receipt.json')
    if not completed['complete'] or not set(cases) <= set(pc['benchmark']['cases']):
        raise ValueError('Completed parent benchmark with matching development cases required')
    import nibabel as nib
    import torch
    if not torch.cuda.is_available(): raise ValueError('GPU required')
    os.environ['TOTALSEG_HOME_DIR'] = pc['model']['cache_directory']
    from totalsegmentator.config import setup_nnunet
    setup_nnunet()
    from totalsegmentator.nnunet import nnUNet_predict_image
    out = Path(c['output_directory']); out.mkdir(parents=True, exist_ok=False)
    (out / 'INCOMPLETE').touch()
    (out / 'input.ini').write_bytes(path.read_bytes())
    (out / 'probe-source.py').write_bytes(Path(__file__).read_bytes())
    records = []; crop_metadata = {}
    for case in cases:
        base = next(r for r in lock['samples'] if r['case_id'] == case and r['baseline'])
        prepared = checked_json(parent / 'cases' / case / 'prepared.json')
        row = next(r for r in prepared['samples'] if r['baseline'])
        for key in ('input_nifti', 'truth_nifti', 'domain_nifti', 'changed_nifti'):
            if sha256(row[key]) != row[key + '_sha256']: raise ValueError('Pinned source export changed')
        selection = json.loads(Path(base['selection_manifest']).read_text())
        geometry = selection['geometry']; spacing = np.asarray(base['spacing_ijk_mm'])
        origin = np.asarray(row['export']['origin_ijk'])
        image = nib.load(row['input_nifti'])
        lo = np.asarray(geometry['crop_low_ijk'])[:2] - np.ceil(context / spacing[:2]).astype(int) - origin[:2]
        hi = np.asarray(geometry['crop_high_ijk_exclusive'])[:2] + np.ceil(context / spacing[:2]).astype(int) - origin[:2]
        lo = np.maximum(lo, 0); hi = np.minimum(hi, image.shape[:2])
        if np.any(hi <= lo): raise ValueError('Supplied atlas-region rectangle is empty')
        box = np.zeros(image.shape, np.uint8)
        box[lo[0]:hi[0], lo[1]:hi[1], :] = 1
        crop = nib.Nifti1Image(box, image.affine)
        folder = out / case; folder.mkdir()
        crop_metadata[case] = {'low_ij': lo.tolist(), 'high_ij_exclusive': hi.tolist(),
                              'context_mm': context, 'z_scope': 'Entire frozen thoracic input with z context',
                              'source': 'Frozen original atlas region plus fixed physical padding; no truth mask shape'}
        print('Known-region diagnostic', case, crop_metadata[case], flush=True)
        nnUNet_predict_image(Path(row['input_nifti']), folder, [293], model='3d_fullres', folds=[0],
             trainer='nnUNetTrainerNoMirroring', tta=False, multilabel_image=False,
             resample=1.5, crop=crop, crop_addon=[0,0,0], roi_subset=['aorta'], task_name='total',
             nr_threads_resampling=2, nr_threads_saving=1, device='cuda', resampling_order=1)
        prediction = folder / 'aorta.nii.gz'
        inferred = nib.load(str(prediction))
        if inferred.shape != image.shape or not np.allclose(inferred.affine, image.affine, rtol=0, atol=1e-5):
            raise ValueError('Diagnostic prediction failed native grid roundtrip')
        records.append(dict(row, status='completed', prediction_nifti=str(prediction),
                            prediction_nifti_sha256=sha256(prediction), diagnostic_crop=crop_metadata[case]))
    write_json(out / 'manifest.json', {'samples': records, 'models_trained': False,
        'parent_benchmark_lock_sha256': sha256(parent / 'benchmark-lock.json'),
        'predictor': {'model': lock['model'], 'scope': 'Supplied atlas-region diagnostic; default 6 mm crop bypassed',
                      'fine_checkpoint_unchanged': True, 'internal_spacing_mm': 1.5}})
    result = write_report(out / 'manifest.json', out / 'review-r1', settings=pc['evaluation'])
    write_json(out / 'receipt.json', {'complete': True, 'models_trained': False, 'report': result,
        'parent_benchmark_lock_sha256': sha256(parent / 'benchmark-lock.json'), 'crops': crop_metadata,
        'source_sha256': hashlib.sha256(Path(__file__).read_bytes()).hexdigest()})
    (out / 'INCOMPLETE').unlink()
    print(json.dumps(result, indent=2))


if __name__ == '__main__': main()
