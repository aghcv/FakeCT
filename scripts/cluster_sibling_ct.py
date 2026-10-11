#!/usr/bin/env python3
"""Freeze/extract baseline-relative CT descriptors and plot exploratory clusters."""
import argparse
import configparser
import json
from pathlib import Path
import sys

import numpy as np

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))
from fakect_multiscale_experiment import sha256,write_json,checked_json
from fakect_ct_delta_features import prepare_case_geometry,extract_variant,feature_names

SCHEMA='fakect.ct-delta-study/1'
CODE_FILES=('scripts/cluster_sibling_ct.py','src/fakect_ct_delta_features.py',
            'src/fakect_ct_delta_clustering.py','src/fakect_ct_delta_report.py',
            'src/fakect_centerline_frame.py','src/fakect_multiscale_experiment.py')


def load_config(path):
    p=configparser.ConfigParser(interpolation=None,inline_comment_prefixes=('#',))
    p.read_string(Path(path).read_text())
    if p.defaults() or set(p.sections())!={'analysis','representation','clustering'}:
        raise ValueError('Expected analysis, representation and clustering sections')
    c={s:dict(p[s]) for s in p.sections()}
    if c['analysis']['schema_version']!=SCHEMA:raise ValueError('Unsupported CT study schema')
    for k in ('manifest','output_directory'):
        v=Path(c['analysis'][k]).expanduser();c['analysis'][k]=str((v if v.is_absolute() else ROOT/v).resolve())
    for k in c['representation']:
        c['representation'][k]=(int if k.endswith('_bins') else float)(c['representation'][k])
    integers={'seed','max_components','k_min','k_max','n_init','kmeans_iterations','silhouette_sample_size'}
    for k in c['clustering']:c['clustering'][k]=(int if k in integers else float)(c['clustering'][k])
    return c


def plan(c,content):
    source=checked_json(c['analysis']['manifest'])
    if source.get('complete') is not True:raise ValueError('Completed population required')
    rows=source['samples'];cases=sorted({r['case_id'] for r in rows})
    if set(cases)&{'260612','260617'}:raise ValueError('Reserved infant data cannot enter this study')
    dependencies={c['analysis']['manifest']:sha256(c['analysis']['manifest'])};metadata={}
    for case in cases:
        baseline=[r for r in rows if r['case_id']==case and r['baseline']]
        if len(baseline)!=1:raise ValueError('One original baseline per case required')
        row=baseline[0]
        recipe=ROOT/'outputs/cohorts/coa/radius115'/case/'pairs-r2/resolved-config.json'
        selected=Path(row['selection_manifest'])
        if sha256(selected)!=row['selection_manifest_sha256']:raise ValueError('Anatomical selection changed')
        for f in (recipe,selected,Path(row['source_npz'])):dependencies[str(f)]=sha256(f)
        if dependencies[row['source_npz']]!=row['source_npz_sha256']:raise ValueError('Original native source changed')
        metadata[case]={'resolved_config':str(recipe),'selection_manifest':str(selected)}
    out=Path(c['analysis']['output_directory']);out.mkdir(parents=True,exist_ok=False)
    (out/'input.ini').write_bytes(content)
    code={name:sha256(ROOT/name) for name in CODE_FILES}
    for name in CODE_FILES:
        target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
    lock={'schema_version':SCHEMA,'config':c,'samples':rows,'cases':cases,'case_metadata':metadata,
          'dependency_sha256':dependencies,'code_sha256':code,'input_sha256':sha256(out/'input.ini'),
          'feature_inputs':'Paired attenuation only, on baseline geometric coordinates; no sibling labels/edit parameters',
          'reserved_test_evaluated':False,'models_trained':False}
    write_json(out/'analysis-lock.json',lock);return lock


def inspect(c,content):
    out=Path(c['analysis']['output_directory']);lock=checked_json(out/'analysis-lock.json')
    import hashlib
    if lock['config']!=c or hashlib.sha256(content).hexdigest()!=lock['input_sha256']:
        raise ValueError('Frozen analysis input changed')
    for name,digest in lock['code_sha256'].items():
        if sha256(ROOT/name)!=digest or sha256(out/'source'/name)!=digest:raise ValueError('Frozen analysis source changed')
    for name,digest in lock['dependency_sha256'].items():
        if sha256(name)!=digest:raise ValueError('Frozen CT source/metadata changed')
    return lock


def extract_case(c,lock,case):
    if case not in lock['cases']:raise ValueError('Unknown source case')
    out=Path(c['analysis']['output_directory'])/'cases'/case;out.mkdir(parents=True,exist_ok=False)
    (out/'INCOMPLETE').touch()
    rows=sorted([r for r in lock['samples'] if r['case_id']==case],key=lambda r:r['sample_id'])
    base=next(r for r in rows if r['baseline']);m=lock['case_metadata'][case]
    geometry=prepare_case_geometry(base,json.loads(Path(m['resolved_config']).read_text()),
                json.loads(Path(m['selection_manifest']).read_text()),**c['representation'])
    features=[];records=[]
    for i,row in enumerate(rows):
        feature,summary=extract_variant(geometry,row)
        if row['baseline'] and np.any(feature):raise ValueError('Original baseline has nonzero CT-difference features')
        records.append({'sample_id':row['sample_id'],'case_id':case,'anatomy_family':row['anatomy_family'],
            'baseline':row['baseline'],**summary,'ct_relative_l1':summary['normalized_positive_ct_energy']+summary['normalized_negative_ct_energy'],
            'added_voxels':row['mask_counts']['added'],'removed_voxels':row['mask_counts']['removed'],
            'annotation_use':'Native mask counts for posthoc interpretation only; excluded from feature extraction'})
        features.append(feature)
        if i%32==0:print(case,i+1,'/',len(rows),flush=True)
    with (out/'features.npz').open('xb') as f:np.savez_compressed(f,features=np.stack(features),feature_names=np.asarray(feature_names(geometry)))
    write_json(out/'features.json',{'records':records,'geometry':geometry.metadata,'features_sha256':sha256(out/'features.npz'),
                                   'source_lock_sha256':sha256(Path(c['analysis']['output_directory'])/'analysis-lock.json')})
    (out/'INCOMPLETE').unlink();return {'case':case,'rows':len(rows),'features':geometry.feature_count}


def analyse(c,lock):
    from fakect_ct_delta_clustering import analyse as clustering,transform_features
    from fakect_ct_delta_report import write_report
    out=Path(c['analysis']['output_directory']);rows=[];matrices=[];feature_hashes={};common_names=None
    lock_hash=sha256(out/'analysis-lock.json')
    for case in lock['cases']:
        folder=out/'cases'/case;meta=checked_json(folder/'features.json')
        if meta['source_lock_sha256']!=lock_hash:raise ValueError('Features belong to a different frozen analysis')
        if sha256(folder/'features.npz')!=meta['features_sha256']:raise ValueError('Extracted CT features changed')
        expected=sorted(r['sample_id'] for r in lock['samples'] if r['case_id']==case)
        if [r['sample_id'] for r in meta['records']]!=expected:raise ValueError('Extracted sample order/completeness changed')
        rows+=meta['records']
        with np.load(folder/'features.npz',allow_pickle=False) as data:
            features=data['features']
            names=data['feature_names']
            if common_names is None:common_names=names.copy()
            elif not np.array_equal(common_names,names):raise ValueError('Features use different coordinate/channel ordering')
            if features.shape!=(len(expected),2*c['representation']['arc_bins']*c['representation']['radial_bins']*c['representation']['angular_bins']):
                raise ValueError('Extracted feature dimensions changed')
            matrices.append(features)
        feature_hashes[case]={'matrix_sha256':meta['features_sha256'],'metadata_sha256':sha256(folder/'features.json')}
    matrix=np.vstack(matrices);edited=np.asarray([not r['baseline'] for r in rows]);result=clustering(matrix[edited],c['clustering'])
    controls=transform_features(result,matrix[~edited]);records=[r for r in rows if not r['baseline']];baselines=[r for r in rows if r['baseline']]
    result['analysis']['representation']={'method':'Baseline-relative CT signed cylindrical physical integral',**c['representation'],
        'baseline_attenuation_scale':'Median original target attenuation; original segmentation used for reference only',
        'geometry_scope':'Original main aorta path, anatomical thoracic slab/crop and fixed relative radial extent',
        'feature_uses_sibling_labels_or_edit_parameters':False,
        'mean_delta_energy_coverage':float(np.mean([r['ct_delta_energy_coverage_fraction'] for r in records])),
        'minimum_delta_energy_coverage':float(min(r['ct_delta_energy_coverage_fraction'] for r in records)),
        'ct_near_baseline_siblings':sum(r['ct_near_baseline'] for r in records)}
    result['analysis']['models_trained']=False
    result['analysis']['provenance']={'source_lock_sha256':lock_hash,'input_sha256':lock['input_sha256'],
                                    'features':feature_hashes,'code_sha256':lock['code_sha256']}
    result['analysis']['scope']='Descriptive all-development-cohort clustering; future model uses require training-partition refitting'
    with (out/'pca-arrays.npz').open('xb') as f:np.savez_compressed(f,features=matrix[edited],scores=result['scores'],
        components=result['components'],mean=result['mean'],labels=result['labels'],candidate_labels=result['candidate_labels'],
        baseline_scores=controls,candidate_centroid_features=result['candidate_centroid_features'])
    report=write_report(out/'review-r1',result['analysis'],records,result['scores'],result['candidate_labels'],baselines,controls)
    write_json(out/'analysis-receipt.json',{'complete':True,'source_lock_sha256':lock_hash,
               'pca_arrays_sha256':sha256(out/'pca-arrays.npz'),'report':report,
               'siblings':len(records),'baseline_controls':len(baselines)})
    return report


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--config',required=True)
    p.add_argument('--stage',choices=('plan','list','features','analyse'),default='plan');p.add_argument('--case-id');p.add_argument('--ids-only',action='store_true')
    a=p.parse_args();path=Path(a.config);c=load_config(path);content=path.read_bytes()
    if a.stage=='plan':result={'cases':plan(c,content)['cases']}
    else:
        lock=inspect(c,content)
        if a.stage=='list':
            if a.ids_only:print('\n'.join(lock['cases']));return
            result={'cases':lock['cases']}
        elif a.stage=='features':result=extract_case(c,lock,a.case_id)
        else:result=analyse(c,lock)
    print(json.dumps(result,indent=2,default=str))


if __name__=='__main__':main()
