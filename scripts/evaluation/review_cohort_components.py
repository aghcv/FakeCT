#!/usr/bin/env python3
"""Explain baseline ROI fragments using native labels, without executing edits.

Writes a new component-details.json supplement; existing ROI review snapshots
are preserved. Full-crop candidate connectivity and eligible atlas connectivity
are both reported relative to the main selected component's aortic anchor.
"""
import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT/'src'))
import numpy as np
from scipy import ndimage
from fakect_study_config import load_study_config
from fakect_roi import prepare_crop, resolve_preview
from fakect_anatomy import verify_selection_sources
from fakect_recipe import recipe_masks, _region_key
from fakect_morphology import tube_position_mm


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def supplement(review_directory, max_coordinates=100):
    folder = Path(review_directory).expanduser().resolve()
    output = folder/'component-details.json'
    if output.exists():
        raise ValueError('Component supplement already exists; retain it and use a new review revision')
    reviews = sorted(folder.glob('260*.json'))
    if not reviews:
        raise ValueError('No completed per-case ROI review snapshots')
    results=[]
    for review_path in reviews:
        review=json.loads(review_path.read_bytes()); case_id=review['case_id']
        summary={'case_id':case_id,'roi_review_json':review_path.name,'roi_review_sha256':sha(review_path),
                 'selected_component_voxels_6':review['metrics']['selected_component_voxels_6']}
        if review['metrics']['selected_components_6'] <= 1:
            summary.update(native_crop_reread=False, minor_components=[])
            results.append(summary); continue
        config_path=Path(review['config']); config_bytes=config_path.read_bytes()
        if hashlib.sha256(config_bytes).hexdigest()!=review['input_sha256']:
            raise ValueError(f'Input changed since ROI review: {case_id}')
        config=load_study_config(config_path);resolved=resolve_preview(config)
        if resolved['anatomy_selection_sha256']!=review['anatomy_sha256']:
            raise ValueError(f'Anatomy changed since ROI review: {case_id}')
        arrays,_=prepare_crop(resolved,config)
        structure=ndimage.generate_binary_structure(3,1)
        components,count=ndimage.label(arrays['selected'],structure=structure)
        sizes=np.bincount(components.ravel()); sizes[0]=0; main_id=int(sizes.argmax())
        if sorted(map(int,sizes[1:]),reverse=True)!=review['metrics']['selected_component_voxels_6']:
            raise ValueError(f'Native components changed since ROI review: {case_id}')
        audit=resolved['anatomical_selection']; catalog={int(r['original_id']):r for r in resolved['catalog']['records']}
        anchor_ids=[int(r['original_id']) for r in audit['records'] if r['role']=='anchor']
        eligible_ids=[int(r['original_id']) for r in audit['records'] if r['role']!='excluded']
        main_anchor=(components==main_id)&np.isin(arrays['act'],anchor_ids)
        candidate_cc,_=ndimage.label(arrays['candidates'],structure=structure)
        eligible_cc,_=ndimage.label(np.isin(arrays['act'],eligible_ids),structure=structure)
        candidate_anchor_cc=set(map(int,np.unique(candidate_cc[main_anchor])))-{0}
        eligible_anchor_cc=set(map(int,np.unique(eligible_cc[main_anchor])))-{0}
        low=np.asarray(resolved['crop_low_ijk'])
        total_length=float(np.linalg.norm(np.diff(np.asarray(resolved['roi_nodes_ijk'])*resolved['spacing_ijk_mm'],axis=0),axis=1).sum())
        details=[]
        for label in sorted((k for k in range(1,count+1) if k!=main_id),key=lambda k:-sizes[k]):
            native=np.argwhere(components==label); global_ijk=native[:,::-1]+low
            keys,counts=np.unique(arrays['act'][components==label],return_counts=True)
            candidate_keys=set(map(int,np.unique(candidate_cc[components==label])))-{0}
            eligible_keys=set(map(int,np.unique(eligible_cc[components==label])))-{0}
            via_candidates=bool(candidate_keys) and candidate_keys.issubset(candidate_anchor_cc)
            via_eligible=bool(eligible_keys) and eligible_keys.issubset(eligible_anchor_cc)
            row={'component_label':int(label),'voxels':int(sizes[label]),
                 'labels':[{'original_id':int(k),'original_name':catalog[int(k)]['original_name'],'voxels':int(v)} for k,v in zip(keys,counts)],
                 'bbox_ijk':{'low_ijk':global_ijk.min(axis=0).tolist(),'high_ijk_exclusive':(global_ijk.max(axis=0)+1).tolist()},
                 'centroid_ijk':global_ijk.mean(axis=0).tolist(),
                 'connects_to_main_aortic_anchor_through_native_admitted_ids':via_candidates,
                 'connects_to_main_aortic_anchor_through_native_atlas_eligible_ids':via_eligible,
                 'native_contact_connectivity':6,
                 'connection_must_leave_roi':via_candidates,
                 'interpretation':('Baseline ROI clipping: disconnected inside the selector, but joins the main aortic anchor through native admitted labels outside the ROI.' if via_candidates else
                                   'Native atlas-eligible connection exists but traverses IDs excluded from the final target.' if via_eligible else
                                   'No native connection to the main selected aortic anchor found within this crop; inspect this component.')}
            if len(native)<=max_coordinates:
                voxels=[]
                for cell,ijk in zip(native,global_ijk):
                    cell=tuple(cell); point=tuple(map(int,ijk))
                    arc=float(tube_position_mm((1,1,1),point,resolved['roi_nodes_ijk'],resolved['spacing_ijk_mm'])[0,0,0])
                    local_resolved={**resolved,'crop_low_ijk':point,'crop_high_ijk_exclusive':tuple(v+1 for v in point)}
                    local_arrays={'act':np.asarray([[[arrays['act'][cell]]]]),'roi':np.ones((1,1,1),dtype=bool)}
                    masks=recipe_masks(local_arrays,local_resolved,config)
                    memberships={name:bool(masks[_region_key(name,edit)][0,0,0]) for name,edit in config['edits'].items()}
                    voxels.append({'ijk':list(point),'original_id':int(arrays['act'][cell]),
                                   'parent_arc_distance_mm':arc,'parent_path_percent':100*arc/total_length,
                                   'edit_selector_membership':memberships})
                row['voxels_native']=voxels
                row['parent_path_percent_range']=[min(v['parent_path_percent'] for v in voxels),max(v['parent_path_percent'] for v in voxels)]
                row['edit_selector_voxel_counts']={name:sum(v['edit_selector_membership'][name] for v in voxels) for name in config['edits']}
                row['edit_membership_scope']='Original voxel membership in exact named/ranged selectors; Gaussian and angular weights, reassignment, lineage and safeguards are not executed.'
            else:
                row['voxels_native']=None
                row['coordinate_limit_note']=f'Component exceeds {max_coordinates} coordinates; use its IDs and native bounding box for detailed review.'
            details.append(row)
        verify_selection_sources(resolved)
        if config_path.read_bytes()!=config_bytes: raise ValueError(f'Input changed during component review: {case_id}')
        summary.update(native_crop_reread=True,main_selected_component_voxels=int(sizes[main_id]),
                       main_selected_aortic_anchor_voxels=int(main_anchor.sum()),
                       anatomy_selection_sha256=resolved['anatomy_selection_sha256'],minor_components=details)
        results.append(summary)
        print(case_id,[(r['voxels'],r.get('parent_path_percent_range'),r.get('edit_selector_voxel_counts'),r['connection_must_leave_roi']) for r in details],flush=True)
    document={'schema_version':'fakect.cohort-roi-component-review/1','generated_at_utc':datetime.now(timezone.utc).isoformat(),
              'code_sha256':sha(__file__),'max_coordinates_per_component':max_coordinates,
              'review_approved':False,'native_edits_executed':False,'case_count':len(results),
              'scope':'Baseline selection fragments only. Connectivity is native voxel contact inside the loaded crop, not independent proof of physiological connectivity. No existing report/config/dataset was altered.',
              'cases':results}
    with output.open('x') as stream: stream.write(json.dumps(document,indent=2,allow_nan=False)+'\n')
    print(output)


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review-directory',required=True)
    parser.add_argument('--max-coordinates',type=int,default=100)
    args=parser.parse_args()
    if not 1<=args.max_coordinates<=10000: parser.error('--max-coordinates must be1..10000')
    try: supplement(args.review_directory,args.max_coordinates)
    except (ValueError,OSError,KeyError,TypeError) as e:parser.exit(2,f'error: {e}\n')


if __name__=='__main__':main()
