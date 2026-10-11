"""Immutable, family-held-out matched adaptation of a complete pretrained aorta head."""
from __future__ import annotations
import configparser
import gc
import hashlib
import importlib.metadata
import json
import os
from pathlib import Path
import resource
import time
from contextlib import ExitStack

import numpy as np

from fakect_multiscale_experiment import sha256, checked_json, write_json
from fakect_pretrained_evidence import case_evidence
from fakect_pretrained_data import (NativeCoreTemplate, PretrainedCaseVolume, fixed_case_grid,
    choose_nested_rosters, audit_resampling_ceiling, export_native_reference, model_to_native_mask)
from fakect_pretrained_adapter import export_mask_nifti, _open_writer

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.pretrained-finetune/1'
ARMS = ('baseline_only', 'siblings')
FIELDS = {
 'experiment': {'schema_version','output_directory','target_review_status'},
 'population': {'manifest','core_census','train_siblings_per_case','evaluation_siblings_per_case','family_reference'},
 'model': {'model_lock','environment_directory'},
 'input': {'spacing_mm','padding_xy_mm','context_z_mm'},
 'training': {'updates','learning_rate','weight_decay','seed','control_fraction','foreground_fraction','report_every'},
 'evaluation': {'surface_tolerance_mm','boundary_margin_mm','local_context_mm'},
 'runtime': {'maximum_gpu_allocated_mib','maximum_host_rss_mib'},
}
TARGET_FIELDS = {'policy','branch_extent','foreground_labels','reference_study'}
CODE_FILES = ('src/fakect_pretrained_study.py','src/fakect_pretrained_data.py','src/fakect_pretrained_learning.py',
 'src/fakect_pretrained_comparison_report.py','src/fakect_pretrained_report.py','src/fakect_pretrained_adapter.py',
 'src/fakect_pretrained_evidence.py','src/fakect_context_volume.py','src/fakect_multiscale_experiment.py',
 'src/fakect_anatomy_surfaces.py','src/fakect_anatomy_centerlines.py','src/fakect_two_stage_3d.py',
 'src/fakect_surface_overlay.py','src/fakect_volume_preview.py','scripts/research/finetune_aorta.py',
 'scripts/research/slurm/fakect_finetune_prepare.sh','scripts/research/slurm/fakect_finetune_gpu.sh','scripts/research/slurm/fakect_finetune_report.sh')


def target_settings(c):
 return c.get('target',{'policy':'aorta_core','branch_extent':'proximal','foreground_labels':['aorta'],'reference_study':None})


def _template(c,lock,base,selection,evidence,grid):
 if target_settings(c)['policy']=='aorta_core':return NativeCoreTemplate(base,selection,evidence)
 from fakect_pretrained_targets import BranchInclusiveTemplate
 return BranchInclusiveTemplate(base,selection,evidence,grid,
     branch_extent=target_settings(c)['branch_extent'],
     annotation_directory=Path(c['experiment']['output_directory'])/'annotations'/base['case_id'])


def load_config(path):
 p=configparser.ConfigParser(interpolation=None,inline_comment_prefixes=('#',));p.read_string(Path(path).read_text())
 if p.defaults() or set(p.sections()) not in (set(FIELDS),set(FIELDS)|{'target'}):raise ValueError('Unexpected fine-tuning sections')
 c={s:dict(p[s]) for s in FIELDS}
 for s,fields in FIELDS.items():
  if set(c[s])!=fields:raise ValueError('Wrong fields in '+s)
 if p.has_section('target'):
  if set(p['target'])!=TARGET_FIELDS:raise ValueError('Wrong fields in target')
  t=dict(p['target'])
  if t['policy']!='thoracic_aorta_and_connected_branches' or t['branch_extent'] not in ('common_box','proximal'):
   raise ValueError('Unsupported branch-inclusive target policy or extent')
  t['foreground_labels']=[v.strip() for v in t['foreground_labels'].split(',') if v.strip()]
  reference=Path(t['reference_study']).expanduser();t['reference_study']=str((reference if reference.is_absolute() else ROOT/reference).resolve())
  if len(t['foreground_labels'])<2 or len(t['foreground_labels'])!=len(set(t['foreground_labels'])):raise ValueError('Distinct grouped aorta/branch output labels required')
  c['target']=t
 if c['experiment']['schema_version']!=SCHEMA or c['experiment']['target_review_status']!='engineering_proxy':
  raise ValueError('Explicit engineering-proxy fine-tuning schema required')
 for section,keys in [('experiment',['output_directory']),('population',['manifest','core_census','family_reference']),('model',['model_lock','environment_directory'])]:
  for key in keys:
   value=Path(c[section][key]).expanduser();c[section][key]=str((value if value.is_absolute() else ROOT/value).resolve())
 for key in ('train_siblings_per_case','evaluation_siblings_per_case'):c['population'][key]=int(c['population'][key])
 for section in ('input','training','evaluation','runtime'):
  for key,value in c[section].items():c[section][key]=(int if key in ('updates','seed','report_every') else float)(value)
 if c['input']['spacing_mm']!=1.5 or c['population']['train_siblings_per_case']!=16 or c['population']['evaluation_siblings_per_case']!=12:
  raise ValueError('This controlled study fixes 1.5 mm / 16 TRAIN / 12 held-out siblings')
 if c['training']['updates']<1 or c['training']['report_every']<1 or c['training']['learning_rate']<=0:
  raise ValueError('Positive update, report and learning budgets required')
 for key in ('control_fraction','foreground_fraction'):
  if not 0<=c['training'][key]<=1:raise ValueError('Training fractions must lie in [0,1]')
 if any(not np.isfinite(v) for s in ('input','training','evaluation','runtime') for v in c[s].values()):raise ValueError('Finite settings required')
 return c


def _census(path,manifest,manifest_path):
 a=checked_json(path)
 if not a.get('complete'):raise ValueError('Completed core extent census required')
 extents=a.get('extents',a.get('extent_by_sample_id'))
 if isinstance(extents,list):extents={r['sample_id']:r for r in extents}
 if not isinstance(extents,dict) or set(extents)!={r['sample_id'] for r in manifest['samples']}:raise ValueError('Core census inventory mismatch')
 digest=a.get('source_manifest_sha256',a.get('manifest_sha256'))
 if digest!=sha256(manifest_path):raise ValueError('Core census source identity mismatch')
 for r in manifest['samples']:
  e=extents[r['sample_id']]
  if e['case_id']!=r['case_id'] or e['baseline']!=r['baseline'] or min(e['added_voxels'],e['removed_voxels'])<0:
   raise ValueError('Core census sample provenance mismatch')
 return extents


def plan(c,content,*,validate_only=False):
 manifest=checked_json(c['population']['manifest']);model=checked_json(c['model']['model_lock'])
 if manifest.get('complete') is not True:raise ValueError('Complete sibling population required')
 rows=manifest['samples'];cases=sorted({r['case_id'] for r in rows})
 if set(cases)&{'260612','260617'}:raise ValueError('Reserved infant cases cannot enter this study')
 extents=_census(c['population']['core_census'],manifest,c['population']['manifest'])
 rosters=choose_nested_rosters(rows,extents,train_siblings=c['population']['train_siblings_per_case'],
     evaluation_siblings=c['population']['evaluation_siblings_per_case'],seed=c['training']['seed'])
 parent=checked_json(c['population']['family_reference']);folds=[]
 for stage in parent['stages']:
  train,test=stage['train_case_ids'],stage['final_test_case_ids']
  if len(train)!=8 or len(test)!=2 or len(set(train+test))!=10 or set(train)&set(test) or set(train+test)!=set(cases):raise ValueError('Family fold assignment mismatch')
  if {r['anatomy_family'] for r in rows if r['case_id'] in train}&{r['anatomy_family'] for r in rows if r['case_id'] in test}:
   raise ValueError('Related anatomy family leaked across a fold')
  folds.append({'id':stage['outer_fold_id'],'train_cases':train,'test_cases':test})
 if len(folds)!=5 or len({f['id'] for f in folds})!=5:raise ValueError('Five conservative family folds required')
 if sorted(case for fold in folds for case in fold['test_cases'])!=cases:raise ValueError('Every case must be held out exactly once')
 reference=None
 if target_settings(c)['reference_study']:
  reference=checked_json(target_settings(c)['reference_study'])
  if reference['cases']!=cases or reference['folds']!=folds or reference['rosters']!=rosters:
   raise ValueError('Branch correction must preserve original cases, family folds and sibling identities')
 checkpoint=[p for p in model['weights'] if 'Dataset293_' in p and p.endswith('/fold_0/checkpoint_final.pth')]
 if len(checkpoint)!=1:raise ValueError('One original complete cardiac checkpoint required')
 if sha256(checkpoint[0])!=model['weights'][checkpoint[0]]:raise ValueError('Original cardiac checkpoint changed')
 evidence={};grids={};dependencies={}
 for path in (c['population']['manifest'],c['population']['core_census'],c['population']['family_reference'],c['model']['model_lock']):dependencies[path]=sha256(path)
 if reference:
  dependencies[target_settings(c)['reference_study']]=sha256(target_settings(c)['reference_study'])
 for case in cases:
  base=next(r for r in rows if r['case_id']==case and r['baseline']);evidence[case]=case_evidence(base)
  selected=json.loads(Path(base['selection_manifest']).read_text());grids[case]=fixed_case_grid(base,selected,evidence[case],**c['input'])
  dependencies[base['selection_manifest']]=sha256(base['selection_manifest'])
  if reference and grids[case]!=reference['grids'][case]:raise ValueError('Corrected target must retain the original physical CT bounding box and model grid')
 foreground_channels=[2]
 if 'target' in c:
  from fakect_pretrained_learning import resolve_foreground_channels
  labels_path=Path(checkpoint[0]).parents[1]/'dataset.json'
  labels=json.loads(labels_path.read_text())['labels']
  foreground_channels=list(resolve_foreground_channels(labels,c['target']['foreground_labels']))
  dependencies[str(labels_path)]=sha256(labels_path)
 if validate_only:return {'validated':True,'cases':cases,'folds':[f['id'] for f in folds],
     'population_samples':len(rows),'evaluation_samples':sum(len(v) for v in rosters['evaluation'].values()),
     'updates_per_training_arm':c['training']['updates'],'model_checkpoint_verified':True,'geometry_verified':True,
     'target_policy':target_settings(c),'foreground_channels':foreground_channels,'reference_grid_parity':reference is not None}
 out=Path(c['experiment']['output_directory']);out.mkdir(parents=True,exist_ok=False);(out/'INCOMPLETE').touch()
 (out/'input.ini').write_bytes(content);code={}
 files=CODE_FILES+('src/fakect_pretrained_targets.py','src/fakect_phantom_anatomy.py','src/fakect_anatomy.py','src/fakect_roi.py','src/fakect_tissues.py') if 'target' in c else CODE_FILES
 for name in files:
  code[name]=sha256(ROOT/name);target=out/'source'/name;target.parent.mkdir(parents=True,exist_ok=True);target.write_bytes((ROOT/name).read_bytes())
 lock={'schema_version':SCHEMA,'config':c,'samples':rows,'cases':cases,'folds':folds,'rosters':rosters,
   'extents':extents,'grids':grids,'evidence':evidence,'model':model,'checkpoint':checkpoint[0],
   'checkpoint_sha256':model['weights'][checkpoint[0]],'dependencies':dependencies,'code_sha256':code,
   'input_sha256':sha256(out/'input.ini'),'reserved_evaluated':False,
   'scope':'Development family-held-out region-assisted matched adaptation; no clinical grades',
   'comparators':'All arms use identical physical CT grids, normalization, tiling and the same foreground class extraction',
   'foreground_channels':foreground_channels,'target_policy':target_settings(c)}
 from fakect_pretrained_comparison_report import select_examples
 evaluation_ids=[sid for case in cases for sid in rosters['evaluation'][case]]
 lock['example_sample_ids']=reference['example_sample_ids'] if reference else select_examples([extents[sid] for sid in evaluation_ids])
 write_json(out/'study-lock.json',lock);(out/'INCOMPLETE').unlink();return lock


def inspect(c,content,*,software=False):
 out=Path(c['experiment']['output_directory']);lock=checked_json(out/'study-lock.json')
 if lock['config']!=c or hashlib.sha256(content).hexdigest()!=lock['input_sha256']:raise ValueError('Frozen study input changed')
 for name,digest in lock['code_sha256'].items():
  if sha256(ROOT/name)!=digest or sha256(out/'source'/name)!=digest:raise ValueError('Frozen study source changed: '+name)
 for name,digest in lock['dependencies'].items():
  if sha256(name)!=digest:raise ValueError('Frozen study dependency changed')
 for ev in lock['evidence'].values():
  for d in ev['dependencies']:
   stat=Path(d['path']).stat()
   if stat.st_size!=d['bytes'] or stat.st_mtime_ns!=d['mtime_ns']:raise ValueError('Frozen source metadata changed')
 if sha256(lock['checkpoint'])!=lock['checkpoint_sha256']:raise ValueError('Initial complete checkpoint changed')
 if software:
  for name,version in lock['model']['versions'].items():
   if importlib.metadata.version(name)!=version:raise ValueError('Pinned software version changed: '+name)
 return lock


def _name(sid):return hashlib.sha256(sid.encode()).hexdigest()[:20]


def prepare_case(c,lock,case):
 out=Path(c['experiment']['output_directory'])/'cases'/case;out.mkdir(parents=True,exist_ok=False);(out/'INCOMPLETE').touch()
 rows={r['sample_id']:r for r in lock['samples'] if r['case_id']==case};base=next(r for r in rows.values() if r['baseline'])
 selected=json.loads(Path(base['selection_manifest']).read_text());evidence=lock['evidence'][case];grid=lock['grids'][case]
 template=_template(c,lock,base,selected,evidence,grid)
 with PretrainedCaseVolume(base,grid,selected,evidence,template=template) as volume:
  original=volume.full_image()[0];masks=volume.masks_full()
 if 'target' in c:
  reference_folder=Path(c['target']['reference_study']).parent/'cases'/case
  reference=checked_json(reference_folder/'prepared.json')
  if sha256(reference_folder/'baseline-model.npz')!=reference['baseline_model_sha256']:raise ValueError('Reference CT cache changed')
  with np.load(reference_folder/'baseline-model.npz',allow_pickle=False) as previous:
   if not np.array_equal(previous['image'],original):raise ValueError('Target correction changed the model CT input')
   old_core_model=previous['target']
  if np.any(old_core_model & ~masks['target']):raise ValueError('Branch-inclusive model target lost the original aorta core')
  branch_model=(masks['target']&~old_core_model).astype(np.uint8)
  restored_branch=model_to_native_mask(branch_model,grid)
  native_branch=template.baseline_branch_mask
  intersection=int(np.count_nonzero(restored_branch & native_branch))
  branch_sampling={'native_branch_voxels':int(native_branch.sum()),'model_branch_voxels':int(branch_model.sum()),
      'roundtrip_branch_voxels':int(restored_branch.sum()),'retained_native_branch_voxels':intersection,
      'branch_roundtrip_dice':2*intersection/(int(restored_branch.sum())+int(native_branch.sum())) if restored_branch.any() or native_branch.any() else 1.,
      'branch_roundtrip_recall':intersection/int(native_branch.sum()) if native_branch.any() else None,'unchanged_model_ct':True}
 points=np.argwhere(masks['target']&masks['domain'])
 if len(points)>8192:points=points[np.linspace(0,len(points)-1,8192).astype(int)]
 with (out/'baseline-model.npz').open('xb') as f:np.savez_compressed(f,image=original,target=masks['target'],domain=masks['domain'],valid=masks['valid'],points=points)
 record={'baseline':base['sample_id'],'baseline_model_sha256':sha256(out/'baseline-model.npz'),'samples':{},'roundtrip':{},
         'target_policy':target_settings(c),'target_metadata':getattr(template,'metadata',{}),
         'study_lock_sha256':sha256(out.parents[1]/'study-lock.json')}
 if getattr(template,'annotation_manifest_path',None):
  annotation=Path(template.annotation_manifest_path);record['annotation_manifest']=str(annotation)
  record['annotation_manifest_sha256']=sha256(annotation)
  metadata=checked_json(annotation);record['annotation_mask']=str(annotation.parent/metadata['mask_npz'])
  record['annotation_mask_sha256']=metadata['mask_npz_sha256']
  record['branch_sampling']=branch_sampling
 wanted=sorted(set(lock['rosters']['training_siblings'][case]+lock['rosters']['evaluation'][case]))
 for sid in wanted:
  row=rows[sid];folder=out/_name(sid);folder.mkdir();item={'sample_id':sid,'baseline':row['baseline'],'model_delta':None,
      'target_extent':template.extent(row)}
  with PretrainedCaseVolume(row,grid,selected,evidence,template=template) as volume:
   if not row['baseline']:
    image=volume.full_image()[0];current=volume.masks_full()
    if not np.array_equal(current['valid'],masks['valid']) or not np.array_equal(current['domain'],masks['domain']):raise ValueError('Sibling model domain changed')
    changed=np.flatnonzero((image!=original)|(current['target']!=masks['target']))
    path=folder/'model-delta.npz'
    with path.open('xb') as f:np.savez_compressed(f,indices=changed,image=image.ravel()[changed],target=current['target'].ravel()[changed])
    item.update(model_delta=str(path),model_delta_sha256=sha256(path),model_delta_voxels=len(changed))
   if sid in lock['rosters']['evaluation'][case]:
    record['roundtrip'][sid]=audit_resampling_ceiling(volume,
        max_native_voxels=64_000_000 if 'target' in c else 32_000_000)
    item['native']=export_native_reference(row,grid,selected,evidence,folder/'native',template=template,export_baseline=True)
    if 'target' in c:
     # Native component references document the recovered branch territory.
     branch_provider=volume.provider;component_paths={}
     shape=tuple(grid['native_shape_kji']);origin=np.asarray(grid['native_origin_ijk'])
     affine=np.asarray(grid['native_affine_ras_ijk'])
     with ExitStack() as stack:
      writers={};staged=[]
      for component in ('aorta_core','connected_branches'):
       path=folder/'native'/(component+'.nii.gz')
       writer,temp,final=_open_writer(stack,path,shape,affine,np.uint8,'Matched '+component+' reference')
       writers[component]=writer;staged.append((temp,final));component_paths[component]=str(path)
      for z in range(0,shape[0],8):
       block=(min(8,shape[0]-z),shape[1],shape[2]);point=origin+[0,0,z]
       core=branch_provider.core_provider.read_block(point,block)['target']
       selected_target=branch_provider.read_block(point,block)['target']
       if np.any(core&~selected_target):raise ValueError('Core truth is not contained in branch-inclusive target')
       writers['aorta_core'].write(core.astype(np.uint8).tobytes(order='C'))
       writers['connected_branches'].write((selected_target&~core).astype(np.uint8).tobytes(order='C'))
     for temp,final in staged:os.link(temp,final);temp.unlink()
     item['native']['target_components']={name:{'path':path,'sha256':sha256(path)} for name,path in component_paths.items()}
  record['samples'][sid]=item;print('Prepared',case,sid,flush=True)
 write_json(out/'prepared.json',record);(out/'INCOMPLETE').unlink();return {'case':case,'samples':len(wanted)}


class ModelCache:
 def __init__(self,c,lock,cases=None):
  self.out=Path(c['experiment']['output_directory']);self.lock=lock;self.baselines={};self.records={}
  for case in (lock['cases'] if cases is None else cases):
   folder=self.out/'cases'/case;record=checked_json(folder/'prepared.json')
   if record['study_lock_sha256']!=sha256(self.out/'study-lock.json') or sha256(folder/'baseline-model.npz')!=record['baseline_model_sha256']:
    raise ValueError('Prepared model baseline identity mismatch')
   for key in ('annotation_manifest','annotation_mask'):
    if key in record and sha256(record[key])!=record[key+'_sha256']:raise ValueError('Frozen target annotation changed')
   self.records[case]=record
   with np.load(folder/'baseline-model.npz',allow_pickle=False) as d:self.baselines[case]={k:d[k] for k in d.files}
   expected=set(lock['rosters']['training_siblings'][case]+lock['rosters']['evaluation'][case]) if 'rosters' in lock else set(record['samples'])
   if set(record['samples'])!=expected:raise ValueError('Prepared sample inventory changed')
   base=self.baselines[case];shape=tuple(lock['grids'][case]['shape_kji']) if 'grids' in lock else base['image'].shape
   if base['image'].shape!=shape or not np.isfinite(base['image']).all():raise ValueError('Model baseline image contract failed')
   for key in ('target','domain','valid'):
    if base[key].shape!=shape or not np.all((base[key]==0)|(base[key]==1)):raise ValueError('Model baseline mask contract failed')
   points=base['points']
   if not len(points) or points.ndim!=2 or points.shape[1]!=3 or np.any(points<0) or np.any(points>=np.array(shape)):
    raise ValueError('Model baseline foreground sampling contract failed')
 def volume(self,sid):
  case=sid.split(':')[0];base=self.baselines[case];r=self.records[case]['samples'][sid]
  image,target=base['image'].copy(),base['target'].copy()
  if r['model_delta']:
   if sha256(r['model_delta'])!=r['model_delta_sha256']:raise ValueError('Sparse model delta changed')
   with np.load(r['model_delta'],allow_pickle=False) as d:
    indices=d['indices']
    if indices.ndim!=1 or indices.dtype.kind not in 'iu' or np.any(indices<0) or np.any(indices>=image.size) or len(np.unique(indices))!=len(indices):
     raise ValueError('Sparse model delta index contract failed')
    if d['image'].shape!=indices.shape or d['target'].shape!=indices.shape or not np.isfinite(d['image']).all() or not np.all((d['target']==0)|(d['target']==1)):
     raise ValueError('Sparse model delta value contract failed')
    image.ravel()[d['indices']]=d['image'];target.ravel()[d['indices']]=d['target']
  return image,target,base['domain']&base['valid']
 def patch(self,sid,offset,shape=(128,128,128)):
  image,target,valid=self.volume(sid)
  result=[np.zeros(shape,np.float32) for _ in range(3)]
  low=np.maximum(offset,0);high=np.minimum(np.asarray(offset)+shape,image.shape)
  if np.any(high<=low):raise ValueError('Training patch has no image intersection')
  src=tuple(slice(int(l),int(h)) for l,h in zip(low,high));dst=tuple(slice(int(l-o),int(h-o)) for l,h,o in zip(low,high,offset))
  for output,array in zip(result,(image,target,valid)):output[dst]=array[src]
  if not np.any(result[2]):raise ValueError('Training patch has no annotated region')
  return result


def schedule(c,lock,cache,fold):
 rng=np.random.default_rng(c['training']['seed']);cases=fold['train_cases'];items=[];order=[];positions={case:0 for case in cases}
 for index in range(c['training']['updates']):
  if not order:order=list(rng.permutation(cases))
  case=str(order.pop());base=cache.baselines[case];shape=np.asarray(base['image'].shape);patch=np.full(3,128)
  if rng.random()<c['training']['foreground_fraction']:
   points=base['points'];point=points[int(rng.integers(len(points)))];offset=point-rng.integers(0,128,3)
  else:offset=np.asarray([rng.integers(max(1,int(n-128+1))) for n in shape])
  offset=np.asarray([-(128-int(n))//2 if n<128 else np.clip(o,0,n-128) for n,o in zip(shape,offset)],int)
  baseline=cache.records[case]['baseline'];siblings=lock['rosters']['training_siblings'][case][1:]
  sid=baseline
  if rng.random()>=c['training']['control_fraction']:
   sid=siblings[positions[case]%len(siblings)];positions[case]+=1
  items.append({'case':case,'offset_kji':offset.tolist(),'baseline_only':baseline,'siblings':sid})
 return items


def _memory(c):
 import torch
 rss=resource.getrusage(resource.RUSAGE_SELF).ru_maxrss/1024
 gpu=torch.cuda.max_memory_allocated()/1024**2 if torch.cuda.is_available() else 0.
 if rss>c['runtime']['maximum_host_rss_mib'] or gpu>c['runtime']['maximum_gpu_allocated_mib']:raise MemoryError('Production study memory guard exceeded')
 return {'host_peak_rss_mib':rss,'gpu_peak_allocated_mib':gpu}


def smoke(c,lock):
 from fakect_pretrained_learning import production_smoke
 out=Path(c['experiment']['output_directory']);fold=lock['folds'][0];cache=ModelCache(c,lock,fold['train_cases'])
 choices=schedule(c,lock,cache,fold);first=next(x for x in choices if x['case']=='260613')
 patch=cache.patch(first['baseline_only'],first['offset_kji']);start=time.monotonic()
 receipt=production_smoke(lock['checkpoint'],*patch,out/'smoke',steps=2,learning_rate=c['training']['learning_rate'],
     foreground_channels=lock.get('foreground_channels',[2]))
 receipt.update(_memory(c));receipt.update(study_lock_sha256=sha256(out/'study-lock.json'),train_case=first['case'],
     train_sample=first['baseline_only'],patch_offset_kji=first['offset_kji'],elapsed_seconds=time.monotonic()-start)
 write_json(out/'runtime-smoke.json',receipt);return receipt


def fit(c,lock,fold_id,arm):
 import torch
 from fakect_pretrained_learning import build_model,make_optimizer,make_scaler,train_step,model_metadata,save_checkpoint,weights_sha256
 if arm not in ARMS:raise ValueError('Unknown training arm')
 out=Path(c['experiment']['output_directory']);smoke_receipt=checked_json(out/'runtime-smoke.json')
 if not smoke_receipt['passed'] or smoke_receipt['study_lock_sha256']!=sha256(out/'study-lock.json'):raise ValueError('Completed permitted production smoke required')
 fold=next(f for f in lock['folds'] if f['id']==fold_id);cache=ModelCache(c,lock,fold['train_cases']);steps=schedule(c,lock,cache,fold)
 path=out/'folds'/fold_id/arm;path.mkdir(parents=True,exist_ok=False);(path/'INCOMPLETE').touch()
 torch.manual_seed(c['training']['seed']);np.random.seed(c['training']['seed']);torch.cuda.reset_peak_memory_stats()
 model=build_model(lock['checkpoint'],'cuda',True);initial=model_metadata(model,foreground_channels=lock.get('foreground_channels',[2]))
 if initial['weights_sha256']!=smoke_receipt['initial_weights_sha256']:raise ValueError('Complete pretrained initialization changed')
 optimizer=make_optimizer(model,c['training']['learning_rate'],c['training']['weight_decay']);scaler=make_scaler('cuda')
 write_json(path/'initialization.json',initial);write_json(path/'schedule.json',steps);history=[];started=time.monotonic()
 for index,item in enumerate(steps):
  patch=cache.patch(item[arm],item['offset_kji']);metrics=train_step(model,optimizer,*patch,scaler=scaler,
      foreground_channels=lock.get('foreground_channels',[2]))
  if (index+1)%c['training']['report_every']==0 or index==0:
   memory=_memory(c);history.append({'update':index+1,**metrics,**memory,'elapsed_seconds':time.monotonic()-started})
   write_json(path/'history.json',history);print(fold_id,arm,index+1,metrics['loss'],flush=True)
 saved=save_checkpoint(path/'checkpoint-final.pth',model,{'fold':fold_id,'arm':arm,'updates':len(steps),'test_evaluated':False,
     'target_policy':lock.get('target_policy'),'foreground_channels':lock.get('foreground_channels',[2])},
     foreground_channels=lock.get('foreground_channels',[2]))
 write_json(path/'training.json',{'complete':True,'updates':len(steps),'fold':fold_id,'arm':arm,'initial_weights_sha256':initial['weights_sha256'],
     'schedule_sha256':sha256(path/'schedule.json'),'checkpoint':saved,'study_lock_sha256':sha256(out/'study-lock.json'),
     'scope':'Fixed budget TRAIN-only adaptation; no held-out checkpoint selection',**_memory(c)})
 (path/'INCOMPLETE').unlink();return {'fold':fold_id,'arm':arm,'checkpoint':saved}


def evaluate(c,lock,fold_id,arm):
 import torch
 from fakect_pretrained_learning import build_model,predict_tiled
 out=Path(c['experiment']['output_directory']);fold=next(f for f in lock['folds'] if f['id']==fold_id)
 checkpoint=lock['checkpoint']
 if arm in ARMS:
  trained=checked_json(out/'folds'/fold_id/arm/'training.json');checkpoint=trained['checkpoint']['path']
  if not trained['complete'] or trained['fold']!=fold_id or trained['arm']!=arm or trained['updates']!=c['training']['updates'] or trained['study_lock_sha256']!=sha256(out/'study-lock.json'):
   raise ValueError('Complete fixed-budget matching fold/arm training required')
  if sha256(checkpoint)!=trained['checkpoint']['sha256']:raise ValueError('Frozen final checkpoint changed')
 elif arm!='pretrained':raise ValueError('Unknown evaluation arm')
 folder=out/'evaluation'/fold_id/arm;folder.mkdir(parents=True,exist_ok=False);(folder/'INCOMPLETE').touch()
 cache=ModelCache(c,lock,fold['test_cases']);model=build_model(checkpoint,'cuda',False);records=[]
 for case in fold['test_cases']:
  for sid in lock['rosters']['evaluation'][case]:
   image,_,_=cache.volume(sid);predicted=predict_tiled(model,image,temporary_directory=str(folder),
       foreground_channels=lock.get('foreground_channels',[2]))
   native=model_to_native_mask(predicted['mask'],lock['grids'][case]);path=folder/(_name(sid)+'.nii.gz')
   if native.size<=32_000_000:
    export_mask_nifti(native,path,origin_ijk=lock['grids'][case]['native_origin_ijk'],evidence=lock['evidence'][case])
   else:
    if native.size>128_000_000:raise ValueError('Native prediction exceeds fixed ROI export budget')
    with ExitStack() as stack:
     writer,temp,final=_open_writer(stack,path,native.shape,np.asarray(lock['grids'][case]['native_affine_ras_ijk']),np.uint8,'Matched aorta native ROI prediction')
     for z in range(0,native.shape[0],8):writer.write(native[z:z+8].astype(np.uint8).tobytes(order='C'))
    os.link(temp,final);temp.unlink()
   records.append({'sample_id':sid,'case_id':case,'status':'completed','prediction_nifti':str(path),
       'prediction_nifti_sha256':sha256(path),'inference':predicted['metadata']});_memory(c)
   print('Evaluated',fold_id,arm,sid,flush=True)
 write_json(folder/'results.json',{'complete':True,'samples':records,'checkpoint_sha256':sha256(checkpoint),
       'study_lock_sha256':sha256(out/'study-lock.json'),'arm':arm,'fold':fold_id})
 (folder/'INCOMPLETE').unlink();return {'fold':fold_id,'arm':arm,'samples':len(records)}


def report(c,lock,revision):
 from fakect_pretrained_comparison_report import write_report,select_examples
 out=Path(c['experiment']['output_directory']);samples=[];allrows={r['sample_id']:r for r in lock['samples']}
 for fold in lock['folds']:
  predictions={}
  for arm in ('pretrained',*ARMS):
   path=out/'evaluation'/fold['id']/arm/'results.json'
   if path.exists() and not (path.parent/'INCOMPLETE').exists():
    done=checked_json(path)
    expected=[sid for case in fold['test_cases'] for sid in lock['rosters']['evaluation'][case]]
    if not done['complete'] or done['fold']!=fold['id'] or done['arm']!=arm or done['study_lock_sha256']!=sha256(out/'study-lock.json') or [r['sample_id'] for r in done['samples']]!=expected:
     raise ValueError('Evaluation fold/arm inventory or source identity mismatch')
    predictions[arm]={r['sample_id']:r for r in done['samples']}
  for case in fold['test_cases']:
   prepared_path=out/'cases'/case/'prepared.json';prepared=checked_json(prepared_path) if prepared_path.exists() else None
   for sid in lock['rosters']['evaluation'][case]:
    row={**lock['extents'][sid],'split':'held_out_family','fold':fold['id'],'predictions':{}}
    if prepared:row.update(prepared['samples'][sid]['native'])
    for arm in ('pretrained',*ARMS):row['predictions'][arm]=predictions.get(arm,{}).get(sid,{'status':'pending'})
    samples.append(row)
 if 'target' in c:
  # Broad target extents are newly measured for the fixed prepared panel.
  # The immutable full-population core census remains a historical reference.
  histogram=[]
  for case in lock['cases']:
   prepared=checked_json(out/'cases'/case/'prepared.json')
   histogram.extend(item['target_extent'] for item in prepared['samples'].values())
 else:histogram=list(lock['extents'].values())
 manifest={'schema_version':SCHEMA,'samples':samples,'cohort_extents':histogram,
           'example_sample_ids':lock['example_sample_ids'],
           'scope':lock['scope'],'study_lock_sha256':sha256(out/'study-lock.json'),'config':c,
           'protocol':{'scope':lock['scope'],'comparators':lock['comparators'],'config':c,
                       'study_lock_sha256':sha256(out/'study-lock.json')}}
 if 'target' in c:
  manifest.update(target_policy=lock['target_policy'],
      histogram_scope='Fixed prepared TRAIN/EVAL subset: 280 edited siblings and ten originals',
      display_scope='Matched atlas-qualified thoracic aorta and connected systemic/cerebrovascular/coronary arterial branches, continued within one shared physical CT box and thoracic slab. Both models use the same aorta/major-arch output-class union; pulmonary labels are excluded. The pretrained cardiac head has no dedicated coronary class. Synthetic edit burden is not a clinical severity grade.')
 name='comparison-'+revision+'.json';write_json(out/name,manifest)
 result=write_report(out/name,out/revision,settings=c['evaluation']);return result
