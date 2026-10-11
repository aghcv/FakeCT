"""Repair failed native siblings without regenerating or altering good data.

Each failing recipe is replayed from its immutable original snapshot. A callback
checks the same selected-lineage trunk rule after every pass. Only the first
offending erosion distance is reduced; its location, profile, direction and
existing local safeguards remain unchanged. Requested and accepted parameters,
all rejected attempts and the mapping to a new accepted-content ID are saved.
The conservative ROI rule is not proof of broader anatomical disconnection.
"""
from collections import Counter
from copy import deepcopy
import configparser
import hashlib
import json
import math
from pathlib import Path

import numpy as np
from scipy import ndimage

import fakect_sibling_population as population
from fakect_growth import lineage_selection
from fakect_mask_pair_preview import transfer_pair
from fakect_recipe import apply_recipe
from fakect_recipe_training import _checked_result, _step_outcomes
from fakect_multiscale_experiment import checked_json, sha256, write_json

ROOT = population.ROOT
SCHEMA = 'fakect.sibling-repair/1'
CODE_FILES = tuple(dict.fromkeys(population.CODE_FILES + (
    'src/fakect_sibling_repair.py','scripts/repair_sibling_population.py')))
_FIELDS = {'schema_version','parent_directory','output_directory','backoff_factor','max_backoff_steps'}


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None,inline_comment_prefixes=('#',),
                                      empty_lines_in_values=False)
    parser.read_string(Path(path).read_text())
    if parser.defaults() or parser.sections()!=['repair'] or set(parser['repair'])!=_FIELDS:
        raise ValueError('Expected exactly the documented repair section')
    c = dict(parser['repair'])
    if c['schema_version']!=SCHEMA or any('\n' in v or '\r' in v for v in c.values()):
        raise ValueError('Unsupported repair schema or multiline input')
    for key in ('parent_directory','output_directory'):
        value=Path(c[key]).expanduser()
        c[key]=str((value if value.is_absolute() else ROOT/value).resolve())
    parent,out=Path(c['parent_directory']),Path(c['output_directory'])
    if parent==out or parent in out.parents or out in parent.parents:
        raise ValueError('Repair output must be separate from the preserved parent')
    c['backoff_factor']=float(c['backoff_factor'])
    c['max_backoff_steps']=int(c['max_backoff_steps'])
    if (not math.isfinite(c['backoff_factor']) or not 0<c['backoff_factor']<1
            or not 0<=c['max_backoff_steps']<=16):
        raise ValueError('Backoff must be in(0,1), with0..16 reductions per erosion step')
    return c


def _read_parent_case(path):
    """Read a signed diagnostic receipt even when its whole case is incomplete.

    This is deliberately narrower than checked_json: only plan() calls it, and
    verifies every record and selected payload before admitting reusable rows.
    No incomplete case is passed directly to model training.
    """
    path=Path(path)
    if sha256(path)!=Path(str(path)+'.sha256').read_text().strip():
        raise ValueError('Parent diagnostic case receipt checksum changed')
    return json.loads(path.read_text())


def plan(c):
    parent=Path(c['parent_directory'])
    parent_c=population.load_config(parent/'input.ini')
    old=population.inspect_lock(parent_c,(parent/'input.ini').read_bytes())
    parent_digest=sha256(parent/'population-lock.json')
    dependencies={str(parent/'population-lock.json'):parent_digest,
                  str(parent/'input.ini'):sha256(parent/'input.ini')}
    cases={}
    for case,context in old['cases'].items():
        path=parent/'cases'/case/'dataset-manifest.json'
        document=_read_parent_case(path)
        dependencies[str(path)]=sha256(path)
        expected={r['sample_id']:r for r in context['recipes']}
        records={r['sample_id']:r for r in document['records']}
        if (document.get('schema_version')!=population.CASE_SCHEMA or document.get('trial') is not False
                or document.get('source_lock_sha256')!=parent_digest or set(records)!=set(expected)
                or len(document['records'])!=len(expected) or document['case_id']!=case):
            raise ValueError('Parent case receipt does not cover its exact frozen recipe panel')
        failed=[r for r in document['records'] if r['status']!='ok']
        if any(r.get('error')!='Native edited trunk disconnected or completely removed' for r in failed):
            raise ValueError('This revision repairs only recorded selected-trunk connectivity failures')
        source_rows={r['sample_id']:r for r in document['samples']}
        successful={sid for sid,r in records.items() if r['status']=='ok'}
        if (len(source_rows)!=len(document['samples'])
                or set(source_rows)!=(successful | {context['baseline']['sample_id']})
                or document['failed_siblings']!=len(failed)
                or document['successful_siblings']!=len(successful)):
            raise ValueError('Parent successful payload panel disagrees with diagnostic records')
        payloads={}
        for sid in sorted(successful):
            row=source_rows[sid]
            if (row['case_id']!=case or row['baseline'] or row['anatomy_family']!=context['baseline']['anatomy_family']
                    or row['parameters']!=expected[sid]['parameters']):
                raise ValueError('Parent reusable row identity/parameters changed')
            for key in ('variant_npz','mask_npz'):
                filename=row[key]
                if sha256(filename)!=row[key+'_sha256']:
                    raise ValueError('Parent reusable payload changed: '+sid)
                payloads[filename]=row[key+'_sha256']
        compact_context={
            'baseline':context['baseline'],'recipe':context['recipe'],'directory':context['directory'],
            'prepared':{key:context['prepared'][key] for key in ('geometry','candidate_source_ids',
                'source_crop_sha256','catalog_sha256','audit_sha256')},
            'provenance':{'anatomical_protected_ids':context['provenance'].get('anatomical_protected_ids',[])}}
        cases[case]={'context':compact_context,'parent_receipt':str(path),
            'parent_receipt_sha256':sha256(path),'reused_rows':[source_rows[s] for s in sorted(successful)],
            'reused_records':[records[s] for s in sorted(successful)],
            'failed_recipes':[expected[r['sample_id']] for r in failed],
            'failed_records':failed,'payload_sha256':payloads,
            'requested_siblings':len(expected),'baseline':context['baseline']}
    return {'schema_version':SCHEMA,'config':deepcopy(c),'cases':cases,
        'dependency_sha256':dependencies,'parent_population_lock_sha256':parent_digest,
        'reused_siblings':sum(len(r['reused_rows']) for r in cases.values()),
        'repair_siblings':sum(len(r['failed_recipes']) for r in cases.values()),
        'reserved_case_ids':list(population.RESERVED_CASES),'reserved_test_evaluated':False,
        'guard_policy':'Original selected-lineage largest trunk descendants must occupy one target component after every pass; no weakening of parent rule',
        'parameter_policy':'Only the first failing erosion distance may be reduced; dilation and all other settings are unchanged',
        'scope':'Additive conservative native repair; an ROI fragment is not necessarily a broader anatomical disconnection'}


def freeze(c,content):
    lock=plan(c)
    out=Path(c['output_directory'])
    out.mkdir(parents=True,exist_ok=False)
    (out/'INCOMPLETE').touch()
    (out/'input.ini').write_bytes(content)
    lock['input_sha256']=hashlib.sha256(content).hexdigest()
    lock['code_sha256']={}
    for name in CODE_FILES:
        raw=(ROOT/name).read_bytes()
        path=out/'source'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
        lock['code_sha256'][name]=hashlib.sha256(raw).hexdigest()
    write_json(out/'repair-lock.json',lock)
    (out/'INCOMPLETE').unlink()
    return lock


def inspect_lock(c,content):
    out=Path(c['output_directory'])
    lock=checked_json(out/'repair-lock.json')
    if lock['config']!=c or lock['input_sha256']!=hashlib.sha256(content).hexdigest():
        raise ValueError('Frozen repair input changed; use another revision')
    if sha256(out/'input.ini')!=lock['input_sha256'] or set(lock['code_sha256'])!=set(CODE_FILES):
        raise ValueError('Frozen repair input/source inventory changed')
    for name,digest in lock['code_sha256'].items():
        if sha256(ROOT/name)!=digest or sha256(out/'source'/name)!=digest:
            raise ValueError('Frozen repair implementation changed: '+name)
    for name,digest in lock['dependency_sha256'].items():
        if sha256(name)!=digest:
            raise ValueError('Parent repair dependency changed: '+name)
    parent=Path(c['parent_directory'])
    population.inspect_lock(population.load_config(parent/'input.ini'),(parent/'input.ini').read_bytes())
    for case in lock['cases'].values():
        policy=case['context']['recipe'].get('reassignment',{}).get('stiffness',{})
        if policy.get('labels'):
            policy['labels']={int(k):v for k,v in policy['labels'].items()}
    return lock


class _GlobalFailure(Exception):
    def __init__(self,record):
        self.record=record
        super().__init__('Selected-lineage global continuity failed after '+record['step_name'])


class RepairFailure(ValueError):
    """A failed finite replay retaining all completed pass/attempt diagnostics."""
    def __init__(self,message,audit):
        self.audit=audit
        super().__init__(message)


def _event_audit(arrays,event):
    origins=event['result']['target_origin_index']
    mask=lineage_selection(origins,arrays['selected'])
    connectivity=population.trunk_connectivity(arrays['selected'],mask,origins)
    record={'step_name':event['step_name'],'iteration':int(event['iteration']),
            'operation':event['config']['edit']['operation'],
            'configured_distance_mm':float(event['config']['edit']['distance_mm']),
            'connectivity':connectivity}
    if not connectivity['passed']:
        components,count=ndimage.label(mask,ndimage.generate_binary_structure(3,1))
        details=[]
        for label in range(1,count+1):
            inside=components==label
            ancestral=arrays['act'].ravel()[origins[inside]]
            ids,n=np.unique(ancestral,return_counts=True)
            details.append({'voxels':int(inside.sum()),
                'ancestor_original_ids':{str(int(i)):int(v) for i,v in zip(ids,n)}})
        record['selected_component_details']=sorted(details,key=lambda d:d['voxels'],reverse=True)
        record['native_erosion_safeguard']=deepcopy(event['result']['summary'].get('erosion_safeguard'))
    return record


def repair_recipe(arrays,resolved,config,*,backoff_factor=.5,max_backoff_steps=8):
    """Return result, accepted config and finite replay audit, or explicit failure."""
    if not 0<backoff_factor<1 or type(max_backoff_steps) is not int or not 0<=max_backoff_steps<=16:
        raise ValueError('Invalid global repair backoff policy')
    accepted=deepcopy(config)
    reductions=Counter()
    attempts=[]
    def failure_audit():
        return {'passed':False,'attempts':deepcopy(attempts),
            'distance_reductions':dict(reductions),'backoff_factor':backoff_factor,
            'max_backoff_steps_per_erosion':max_backoff_steps,
            'only_erosion_distance_changed':True}
    maximum=1+max_backoff_steps*sum(e['operation']=='erosion' for e in config['edits'].values())
    for attempt in range(maximum):
        pass_audits=[]
        def observe(event):
            record=_event_audit(arrays,event)
            pass_audits.append(record)
            if not record['connectivity']['passed']:
                raise _GlobalFailure(record)
        try:
            result=apply_recipe(arrays,resolved,accepted,on_step=observe)
            attempts.append({'attempt':attempt+1,'accepted':True,
                'configured_distances_mm':{n:e['distance_mm'] for n,e in accepted['edits'].items()},
                'passes':pass_audits})
            return result,accepted,{'passed':True,'attempts':attempts,
                'distance_reductions':dict(reductions),'backoff_factor':backoff_factor,
                'max_backoff_steps_per_erosion':max_backoff_steps,
                'only_erosion_distance_changed':True}
        except _GlobalFailure as failure:
            name=failure.record['step_name']
            attempts.append({'attempt':attempt+1,'accepted':False,'first_failing_step':name,
                'configured_distances_mm':{n:e['distance_mm'] for n,e in accepted['edits'].items()},
                'passes':pass_audits})
            if accepted['edits'][name]['operation']!='erosion':
                raise RepairFailure('Global continuity failed during a non-erosion pass; dilation is never altered',
                                    failure_audit())
            if reductions[name]>=max_backoff_steps:
                raise RepairFailure('Global erosion backoff exhausted for '+name,failure_audit())
            reductions[name]+=1
            accepted['edits'][name]['distance_mm']=round(config['edits'][name]['distance_mm']*
                                                       backoff_factor**reductions[name],9)
    raise RepairFailure('Finite repair attempt budget exhausted',failure_audit())


def repair_case(c,lock,case_id):
    if case_id not in lock['cases'] or case_id in population.RESERVED_CASES:
        raise ValueError('Unknown or reserved repair case')
    case=lock['cases'][case_id]
    out=Path(c['output_directory'])/'cases'/case_id
    out.mkdir(parents=True,exist_ok=False)
    (out/'INCOMPLETE').touch()
    rows=[deepcopy(case['baseline'])]+deepcopy(case['reused_rows'])
    records=deepcopy(case['reused_records'])
    # Every inherited sample is independently bound; never consume the old
    # incomplete case as an already released training cohort.
    for filename,digest in case['payload_sha256'].items():
        if sha256(filename)!=digest:
            raise ValueError('Inherited successful payload changed')
    repairs=[]
    arrays=resolved=broad=broad_origin=None
    if case['failed_recipes']:
        arrays,resolved,broad,broad_origin=population._load_source(case['context'])
        (out/'variants').mkdir()
        (out/'masks').mkdir()
    for request in case['failed_recipes']:
        record={'requested_sample_id':request['sample_id'],'requested_variant_id':request['variant_id'],
                'case_id':case_id,'group':request['group'],'status':'failed'}
        try:
            config=deepcopy(case['context']['recipe'])
            config['edits']=deepcopy(request['edits'])
            result,accepted,audit=repair_recipe(arrays,resolved,config,
                backoff_factor=c['backoff_factor'],max_backoff_steps=c['max_backoff_steps'])
            image,mask=_checked_result(arrays,resolved,result)
            pair=transfer_pair(broad,broad_origin,arrays['selected'],mask,
                               resolved['crop_low_ijk'],result['target_origin_index'])
            fingerprint={'requested_sample_id':request['sample_id'],'accepted_edits':accepted['edits'],
                         'parent_population_lock_sha256':lock['parent_population_lock_sha256']}
            variant='repair-'+hashlib.sha256(population._canonical(fingerprint).encode()).hexdigest()[:16]
            sample_id=case_id+':'+variant
            parameters=deepcopy(request['parameters'])
            for name in parameters:
                parameters[name]['distance_mm']=accepted['edits'][name]['distance_mm']
            connectivity=population.trunk_connectivity(arrays['selected'],mask.astype(bool),
                                                      result['target_origin_index'])
            record.update(status='ok',sample_id=sample_id,variant_id=variant,
                requested_parameters=deepcopy(request['parameters']),accepted_parameters=parameters,
                requested_edits=deepcopy(request['edits']),accepted_edits=accepted['edits'],
                repair_audit=audit,mask_counts=pair['counts'],trunk_connectivity=connectivity,
                step_outcomes=_step_outcomes(result['summary']),
                after_mask_sha256=population._array_hash(pair['after'].astype(np.uint8)),
                image_sha256=population._array_hash(image),
                scalar_changed_voxels=int(result['scalar_changed_mask'].sum()),
                baseline_equivalent=not bool(pair['added'].any() or pair['removed'].any()))
            variant_path=out/'variants'/(variant+'.npz')
            with variant_path.open('xb') as stream:
                np.savez_compressed(stream,image=image,mask=mask,edited_labels=result['edited_labels'],
                    changed_mask=result['changed_mask'],scalar_changed_mask=result['scalar_changed_mask'],
                    target_origin_index=result['target_origin_index'],edit_region_mask=result['edit_region_mask'],
                    metadata_json=np.asarray(population._canonical(record)))
            mask_path=out/'masks'/(variant+'.npz')
            with mask_path.open('xb') as stream:
                np.savez_compressed(stream,before=pair['before'].astype(np.uint8),after=pair['after'].astype(np.uint8),
                    added=pair['added'].astype(np.uint8),removed=pair['removed'].astype(np.uint8),
                    crop_origin_ijk=pair['crop_origin_ijk'],spacing_ijk_mm=case['baseline']['spacing_ijk_mm'],
                    sample_id=np.asarray(sample_id))
            row=deepcopy(case['baseline'])
            row.update(sample_id=sample_id,variant_id=variant,baseline=False,
                baseline_equivalent=record['baseline_equivalent'],split='unassigned',
                variant_npz=str(variant_path),variant_npz_sha256=sha256(variant_path),
                mask_npz=str(mask_path),mask_npz_sha256=sha256(mask_path),mask_counts=pair['counts'],
                parameters=parameters,requested_parameters=deepcopy(request['parameters']),
                requested_sample_id=request['sample_id'],requested_variant_id=request['variant_id'],
                step_outcomes=record['step_outcomes'],population_group=request['group'],
                trunk_connectivity=connectivity,repair_audit=audit)
            rows.append(row)
            records.append(record)
        except (ValueError,RuntimeError,OSError) as error:
            record.update(status='failed',error_type=type(error).__name__,error=str(error))
            if isinstance(error,RepairFailure):
                record['repair_audit']=error.audit
        repairs.append(record)
        write_json(out/'repair-progress.json',{'repairs':repairs})
        print(case_id,request['variant_id'],record['status'],flush=True)
    failed=sum(r['status']!='ok' for r in repairs)
    after_hashes=[r['after_mask_sha256'] for r in records if r.get('after_mask_sha256')]
    document={'schema_version':population.CASE_SCHEMA,'complete':failed==0,'trial':False,
        'case_id':case_id,'anatomy_family':case['baseline']['anatomy_family'],'samples':rows,'records':records,
        'requested_siblings':case['requested_siblings'],'successful_siblings':len(rows)-1,
        'failed_siblings':failed,'reused_siblings':len(case['reused_rows']),'repaired_siblings':len(repairs)-failed,
        'unique_edited_geometry_count':len(set(after_hashes)),
        'baseline_equivalent_siblings':sum(r.get('baseline_equivalent',False) for r in rows if not r['baseline']),
        'repair_records':repairs,'parent_receipt':case['parent_receipt'],
        'parent_receipt_sha256':case['parent_receipt_sha256'],
        'source_lock_sha256':sha256(Path(c['output_directory'])/'repair-lock.json'),
        'original_sources_modified':False,'full_body_siblings_written':False,
        'classifier_started':False,'guard_policy':lock['guard_policy'],'scope':lock['scope']}
    write_json(out/'dataset-manifest.json',document)
    if failed or len(rows)-1!=case['requested_siblings']:
        raise ValueError('Repair case incomplete; preserve diagnostics and use a new revision')
    (out/'INCOMPLETE').unlink()
    return document


def collect(c,lock):
    out=Path(c['output_directory'])
    if (out/'dataset-manifest.json').exists():
        raise FileExistsError('Repaired population already released')
    rows,cases=[],[]
    for case in lock['cases']:
        path=out/'cases'/case/'dataset-manifest.json'
        d=checked_json(path)
        if (d['complete'] is not True or d['failed_siblings']!=0
                or d['source_lock_sha256']!=sha256(out/'repair-lock.json')
                or d['successful_siblings']!=lock['cases'][case]['requested_siblings']):
            raise ValueError('Repair case did not cover every original requested sibling')
        for r in d['samples']:
            for key in ('variant_npz','mask_npz'):
                if sha256(r[key])!=r[key+'_sha256']:
                    raise ValueError('Released repaired/reused payload changed')
            row=deepcopy(r)
            if not row['baseline']:
                row.update(prepared_manifest=str(path),prepared_manifest_sha256=sha256(path))
            rows.append(row)
        cases.append({'case_id':case,'manifest':str(path),'sha256':sha256(path),
            'reused_siblings':d['reused_siblings'],'repaired_siblings':d['repaired_siblings']})
    if len({r['sample_id'] for r in rows})!=len(rows):
        raise ValueError('Repair repeated accepted sample identities')
    document={'schema_version':population.SCHEMA,'complete':True,'cases':cases,'case_ids':list(lock['cases']),
        'samples':rows,'sample_count':len(rows),'edited_siblings':sum(not r['baseline'] for r in rows),
        'original_baselines':sum(r['baseline'] for r in rows),'image_units':'cm^-1',
        'image_method':'attenuation_copy_proxy','source_lock_sha256':sha256(out/'repair-lock.json'),
        'reused_siblings':lock['reused_siblings'],'repaired_siblings':lock['repair_siblings'],
        'parent_population_lock_sha256':lock['parent_population_lock_sha256'],
        'reserved_case_ids':lock['reserved_case_ids'],'reserved_test_evaluated':False,
        'full_body_siblings_written':False,'classifier_started':False,
        'registry_policy':'Reference-based immutable manifest; portable registry export requires an adapter',
        'scope':lock['scope']}
    write_json(out/'dataset-manifest.json',document)
    return document
