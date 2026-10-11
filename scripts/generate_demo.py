#!/usr/bin/env python3
"""Initialize a portable mathematical CoA-shaped example; optionally smoke-test it."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT/'src'))
sys.path.insert(0, str(ROOT/'scripts'))
from fakect_roi import tube_mask


def sha(path):
    """Hash the complete contents of one fixture artifact.

    Parameters
    ----------
    path : str or pathlib.Path
        Existing file whose bytes are read without modification.

    Returns
    -------
    str
        Hexadecimal SHA-256 digest.
    """
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, data):
    """Serialize finite JSON fixture metadata to a file.

    Parameters
    ----------
    path : pathlib.Path
        Destination file; its parent must already exist.
    data : JSON-compatible object
        Metadata to serialize; nonfinite floats are rejected.

    Returns
    -------
    None
        Writes the requested files or prints CLI output; no return value.

    Notes
    -----
    An existing destination is replaced; callers create fresh demo revisions.
    """
    path.write_text(json.dumps(data, indent=2, allow_nan=False)+'\n')


def initialize(output):
    """Create three mathematical volume fixtures and reusable cohort/model inputs.

    Parameters
    ----------
    output : str or pathlib.Path
        New output directory; existing revisions are refused.

    Returns
    -------
    dict
        Project metadata listing cases, INI paths, source units and template hash.

    Notes
    -----
    Creates ACT/ATN arrays, audit metadata, a tissue catalog and editable INIs.
    Arrays use k,j,i order and unit spacing in mm. Native ATN values encode
    attenuation in cm^-1 multiplied by the i voxel width in cm.
    An INCOMPLETE marker remains if initialization fails; sources are not XCAT data.
    """
    output = Path(output).expanduser().resolve()
    template = (ROOT/'configs/templates/coa-template.ini').read_text()
    # INI paths are literal. '#' and line breaks cannot round-trip with the
    # strict comment-aware parser; fail before creating a partial directory.
    if any(c in str(output) for c in ('#', '\n', '\r', '\0')):
        raise ValueError('Demo directory cannot contain # or a line break')
    output.mkdir(parents=True, exist_ok=False)
    (output/'INCOMPLETE').write_text('Demo fixture initialization incomplete\n')
    shape = (56, 56, 56)
    spacing = (1., 1., 1.)
    source_root = output/'phantoms'
    source_root.mkdir()
    catalog = {'policy_version': 'synthetic-demo/1', 'sources': {},
               'categories': [{'id': code, 'name': name} for code, name in
                              ((0,'background'),(1,'soft_tissue'),(2,'bone'),(5,'artery'),(255,'unknown'))],
               'records': [{'original_id': label, 'original_name': name,
                            'classification': {'tissue_id': code, 'tissue_name': group}}
                           for label,name,code,group in
                           ((0,'background',0,'background'),(20,'soft',1,'soft_tissue'),
                            (30,'protected_material',2,'bone'),(-7,'main_tube',5,'artery'),
                            (8,'connected_branch',5,'artery'),(9,'neighbor_tube',5,'artery'))]}
    write_json(output/'catalog.json', catalog)
    source_cases, metadata_hashes, cases = [], {}, []
    for index in range(3):
        case_id = f'demo{index+1:02d}'
        source = source_root/case_id
        source.mkdir()
        nodes = np.array(((18,28,9),(17,28,27),(24,28,41),(34,28,40),(39,28,27),(39,28,9)),float)
        nodes[:,0] += index-1
        nodes[2:4,2] += index-1
        radius = 3.0+index*.5
        labels = np.full(shape,20,dtype='<f4')
        main = tube_mask(shape,(0,0,0),nodes,spacing,[radius]*len(nodes))
        branch_nodes = np.array((nodes[2],nodes[2]+(0,8,4)))
        branch = tube_mask(shape,(0,0,0),branch_nodes,spacing,[1.5,1.5])
        nearby = tube_mask(shape,(0,0,0),((49,28,12),(49,28,36)),spacing,[1.5,1.5])
        labels[nearby] = 9
        labels[branch] = 8
        labels[main] = -7
        labels[2:8,2:8,2:8] = 30
        labels[[0,-1],:,:] = 0
        # XCAT-like native ATN stores mu times pixel width in cm. The pipeline
        # converts this back to cm^-1; these are simple software fixture values.
        mu = np.full(shape,.19,dtype=np.float32)
        mu[np.isin(labels,(-7,8,9))] = .26
        mu[labels==30] = .45
        mu[labels==0] = 0
        labels.tofile(source/f'{case_id}_act_1.bin')
        (mu*(spacing[0]/10)).astype('<f4').tofile(source/f'{case_id}_atn_1.bin')
        par, log = source/f'{case_id}.par', source/f'{case_id}.log'
        par.write_text('Mathematical software fixture, not an XCAT parameter file.\n')
        log.write_text('Fixture geometry is explicitly specified in audit.json.\n')
        source_cases.append({'case_id':case_id,'directory':str(source),'shape_kji':list(shape),
                             'spacing_ijk_mm':list(spacing),'par_path':str(par),'log_path':str(log)})
        metadata_hashes.update({str(p):sha(p) for p in (par,log)})
        folder = output/'cases'/case_id
        folder.mkdir(parents=True)
        text = template.format(case_id=case_id,source_root=source_root,audit=output/'audit.json',
                               catalog=output/'catalog.json',case_output=folder,
                               centers=' ; '.join(','.join(f'{v:g}' for v in row) for row in nodes),
                               radii=','.join(['5.5']*len(nodes)))
        (folder/'cohort.ini').write_text(text)
        cases.append({'case_id':case_id,'cohort_ini':str(folder/'cohort.ini'),
                      'anatomy_family':f'synthetic-{case_id}','registry_name':f'{case_id}-r1'})
    write_json(output/'audit.json',{'cases':source_cases,'source_metadata_sha256':metadata_hashes})
    (output/'model.ini').write_text(f'''# Software demonstration only: three mathematical geometries, not clinical validation.
[experiment]
schema_version = fakect.model-experiment/1
registry = {output/'registry'}
datasets = demo01-r1,demo02-r1,demo03-r1
output_directory = {output/'model-r1'}
[split]
mode = anatomy_family
validation_fraction = 0.2
test_fraction = 0.2
seed = 42
[evaluation]
strategy = holdout
case_counts = 1,1,1
seeds = 42
threshold = 0.5
[preprocess]
target_spacing_mm = 1,1,1
[model]
architecture = unet2d
patch_size = 32,32
slice_axis = k
normalization = fixed_clip
clip_min = 0
clip_max = 0.5
epochs = 1
batch_size = 4
learning_rate = 0.001
seed = 42
''')
    project = {'schema_version':'fakect.demo/1','purpose':'Mathematical software demonstration; not clinical or biological validation',
               'source_units':'ATN = attenuation cm^-1 times i pixel width in cm',
               'review_required':True,'cases':cases,'model_ini':str(output/'model.ini'),
               'template_sha256':sha(ROOT/'configs/templates/coa-template.ini')}
    write_json(output/'project.json',project)
    write_json(output/'source-manifest.json',{'schema_version':'fakect.demo-source/1',
               'artifacts_sha256':{str(p.relative_to(output)):sha(p) for p in sorted(output.rglob('*'))
                                   if p.is_file() and p.name!='INCOMPLETE'}})
    (output/'INCOMPLETE').unlink()
    return project


def smoke_test(project):
    """Exercise native preparation and registry validation on a fresh mathematical demo.

    Parameters
    ----------
    project : dict
        Metadata returned by initialize, pointing to its new fixture files.

    Returns
    -------
    dict
        Receipt with per-case counts, source-integrity checks and model readiness.

    Notes
    -----
    Generates ROI reports, marks only deterministic fixture inputs reviewed,
    runs native preflight, freezes recipes and exports/registers nine samples per
    case. It validates the model plan without fitting an ML model.
    Never use these automated review assertions for externally supplied anatomy.
    """
    from train_study import run
    from fakect_cohort_registry import register_dataset, load_registry_entry
    from model_experiment import run as model_run
    output = Path(project['model_ini']).parent
    source = json.loads((output/'source-manifest.json').read_text())
    baseline_hashes = {p:h for p,h in source['artifacts_sha256'].items() if p.endswith('.bin')}
    results = []
    # Four-panel report is shown for the first case. All three use the same
    # native preparation path and remain separated at the model-family split.
    first = Path(project['cases'][0]['cohort_ini'])
    run(first,stage='preview',preview_mode='roi-selection')
    for case in project['cases']:
        config = Path(case['cohort_ini'])
        text = config.read_text().replace('coordinate_reviewed = false','coordinate_reviewed = true')
        text = text.replace('parameters_reviewed = false','parameters_reviewed = true')
        config.write_text(text)
        run(config,stage='preflight',validate_only=True)
        preflight = run(config,stage='preflight')
        if not preflight.get('execution_valid') or not preflight.get('cohort_ready'):
            raise ValueError('Demo native preflight failed; inspect retained artifacts')
        run(config,stage='freeze')
        prepared = run(config,stage='prepare')
        manifest = json.loads((config.parent/'pairs-r1/dataset-manifest.json').read_text())
        if (manifest['sample_count'] != 9 or not any(row['changed_voxels'] > 0 for row in manifest['samples'])
                or not manifest['original_sources_modified'] is False):
            raise ValueError('Demo must export nine recipes with actual morphological changes and preserved sources')
        entry = register_dataset(output/'registry',config.parent/'pairs-r1/dataset-manifest.json',
                                 case['registry_name'],case['anatomy_family'],family_verified=True)
        load_registry_entry(output/'registry',case['registry_name'])
        results.append({'case_id':case['case_id'],'sample_count':entry['sample_count'],
                        'execution_valid':preflight['execution_valid'],
                        'unique_geometry_count':manifest['unique_geometry_count'],
                        'changed_siblings':sum(row['changed_voxels'] > 0 for row in manifest['samples'])})
    assert all(sha(output/p)==h for p,h in baseline_hashes.items()), 'Source arrays changed'
    model_validation = model_run(project['model_ini'],validate_only=True)
    receipt = {'schema_version':'fakect.demo-smoke/1','cases':results,
               'source_arrays_unchanged':True,'model_fitted':False,
               'review_scope':'Deterministic mathematical software fixture only',
               'model_ready':model_validation.get('ready')}
    write_json(output/'smoke-test.json',receipt)
    return receipt


def main(argv=None):
    """Initialize a demo revision and optionally run its pipeline smoke test.

    Parameters
    ----------
    argv : sequence of str or None, optional
        Arguments excluding the executable name; None uses sys.argv[1:].

    Returns
    -------
    int
        Zero on success; argparse raises SystemExit(2) for input/execution failures.
    """
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--output',required=True,type=Path,help='New directory; existing revisions are refused')
    parser.add_argument('--smoke-test',action='store_true',
                        help='Initialize and exercise the deterministic fixture through ROI, preflight, freeze, pairs and model validation; no fitting')
    args = parser.parse_args(argv)
    try:
        project = initialize(args.output)
        result = smoke_test(project) if args.smoke_test else project
    except (ValueError,OSError,KeyError,ImportError) as exc:
        parser.exit(2,f'error: {exc}\n')
    print(json.dumps(result,indent=2,allow_nan=False))
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
