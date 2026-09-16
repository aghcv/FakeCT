"""Bounded audit of extracted notebook functions; never executes notebook top level."""
import argparse
import ast
import contextlib
import io
import json
from pathlib import Path
import numpy as np
from scipy.ndimage import binary_dilation, distance_transform_edt

ROOT = Path(__file__).resolve().parents[2]
parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--dilation', type=Path, default=ROOT/'references/notebooks/dilation_export.py.txt')
parser.add_argument('--erosion', type=Path, default=ROOT/'references/notebooks/erosion_export.py.txt')
parser.add_argument('--output', type=Path, default=Path('outputs/evaluation/notebook-results.json'))
args = parser.parse_args()
DIL, ERO = args.dilation, args.erosion
report = {'method': 'AST-selected function definitions; no top-level notebook code, installs, downloads, or writes to data', 'versions': {'numpy': np.__version__}}
source = DIL.read_text()
try:
    ast.parse(source)
    report['dilation_full_export_parses'] = True
except SyntaxError as exc:
    report['dilation_full_export_parses'] = {'line': exc.lineno, 'text': exc.text.strip(), 'error': str(exc)}
# Correct one source-export typo in memory solely to obtain function ASTs.
dtree = ast.parse(source.replace('\n-import numpy as np\n', '\nimport numpy as np\n'))
etree = ast.parse(ERO.read_text())
report['dilation_function_definitions'] = sum(isinstance(n, ast.FunctionDef) for n in dtree.body)
report['erosion_function_definitions'] = sum(isinstance(n, ast.FunctionDef) for n in etree.body)
def extract(tree, names, env):
    defs = [n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name in names]
    exec(compile(ast.Module(body=defs, type_ignores=[]), '<extracted notebook functions>', 'exec'), env)
    return {n.name: n.lineno for n in defs}

denv = {'np': np, 'binary_dilation': binary_dilation, 'distance_transform_edt': distance_transform_edt, 'plot_test_results': lambda *a, **k: None}
dnames = {'borrow_pixels_2d','borrow_pixels_3d','vti_dilate6','vti_make_sphere','vti_find_illustrative_seed','vti_shape_domain','vti_borrow_round','vti_make_teaching_bands'}
report['selected_dilation_function_lines'] = extract(dtree, dnames, denv)
dtest_names = {'test_unlisted_and_multi_counts', 'test_disallowed_donor', 'test_overlapping_allowed_and_protected', 'test_self_borrowing', 'test_empty_request_mask','test_full_artery_grid','test_empty_allowed_donors','test_single_element_and_1d_like_grids','test_large_integer_dtypes','test_disjoint_request_mask'}
extract(dtree, dtest_names, denv)
eenv = {'np': np}
enames = {'release_pixels_2d','run_recipient_policy_tests','run_erosion_2d_tests','run_additional_erosion_test','run_expanded_search_test'}
report['selected_erosion_function_lines'] = extract(etree, enames, eenv)
report['notebook_test_functions'] = {}
for name in sorted(dtest_names):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            denv[name]()
        report['notebook_test_functions'][name] = 'pass'
    except Exception as exc:
        report['notebook_test_functions'][name] = type(exc).__name__ + ': ' + str(exc)
for name in sorted(enames - {'release_pixels_2d'}):
    try:
        with contextlib.redirect_stdout(io.StringIO()):
            eenv[name](*([eenv['release_pixels_2d']] if name == 'run_erosion_2d_tests' else []))
        report['notebook_test_functions'][name] = 'pass'
    except Exception as exc:
        report['notebook_test_functions'][name] = type(exc).__name__ + ': ' + str(exc)

# Adversarial self-donor accounting case not covered by notebook tests.
for dim, shape in [(2,(1,2)),(3,(1,1,2))]:
    grid = np.array([9,2], dtype=np.uint8).reshape(shape)
    out, counts, blocked = denv[f'borrow_pixels_{dim}d'](grid,np.ones(shape,dtype=bool),9,{9,2},set())
    report[f'self_donor_{dim}d'] = {'changed':int((out!=grid).sum()),'sum_donor_counts':sum(counts.values()),'donor_counts':counts, 'blocked':int(blocked.sum())}

# Property check: the ordinary 3D policy preserves protected voxels and exact audit.
rng = np.random.default_rng(20260916)
pass_trials=0
for _ in range(30):
    a=rng.integers(0,6,size=(9,11,13),dtype=np.uint8)
    req=rng.random(a.shape)<.25
    out,counts,blocked=denv['borrow_pixels_3d'](a,req,5,{1,2,3,4},{4})
    assert np.array_equal(out[~req],a[~req])
    assert np.array_equal(out[a==4],a[a==4])
    assert int((out!=a).sum()) == sum(counts.values())
    assert not np.any(blocked & (out!=a))
    pass_trials+=1
report['random_3d_donor_invariant_trials_passed']=pass_trials

# Erosion fallback can see across protected barriers; unspecified recipient IDs accepted.
release = eenv['release_pixels_2d']
a=np.full((5,5),2,dtype=np.uint8);a[1:4,1:4]=4;a[2,2]=9
req=np.zeros(a.shape,bool);req[2,2]=True
out,unresolved=release(a,req,9,[2],{4},max_search_radius=2)
report['erosion_protected_ring']={'old_center':int(a[2,2]),'new_center':int(out[2,2]),'unresolved':bool(unresolved[2,2]),'protected_changed':int(((out!=a)&(a==4)).sum())}
a=np.full((3,3),7,dtype=np.uint8);a[1,1]=9
req=np.zeros(a.shape,bool);req[1,1]=True
out,u=release(a,req,9,[2,3],set(),max_search_radius=0)
report['erosion_recipient_not_in_priority']={'assigned_label':int(out[1,1]),'priority':[2,3]}
# Invalid request currently can edit protected non-target centers.
a=np.full((3,3),2,dtype=np.uint8);a[1,1]=4
out,u=release(a,req,9,[2],{4},max_search_radius=0)
report['erosion_invalid_release_mask']={'old_center':int(a[1,1]),'new_center':int(out[1,1]),'protected_label':4}
try:
    release(np.ones((3,3,3),dtype=np.uint8),np.zeros((3,3,3),bool),9,[2],set())
    report['erosion_3d']='accepted'
except Exception as exc:
    report['erosion_3d']=type(exc).__name__+': '+str(exc)

# Six-neighbor morphology agrees with reference SciPy for bounded random volumes.
structure=np.zeros((3,3,3),bool)
structure[1,1,:]=True;structure[1,:,1]=True;structure[:,1,1]=True
checks=0
for _ in range(30):
    mask=rng.random((7,9,11))<.1
    assert np.array_equal(denv['vti_dilate6'](mask),binary_dilation(mask,structure=structure))
    checks+=1
report['dilate6_scipy_equivalence_trials_passed']=checks

# Target-domain radius and growth-round heuristic disagree even without barriers.
growth=[]
for spacing, seed_radius,target_radius in [((1.,1.,1.),1.,3.),((2.,1.,1.),2.,6.),((1.,1.,1.),4.,10.)]:
    shape=(41,41,41);center=(20,20,20)
    seed=denv['vti_make_sphere'](shape,center,seed_radius,spacing)
    target=denv['vti_make_sphere'](shape,center,target_radius,spacing)
    rounds=max(1,int(np.ceil((target_radius-seed_radius)/min(spacing))))
    current=seed.copy()
    for _ in range(rounds):current=denv['vti_dilate6'](current)&target
    growth.append({'spacing_zyx':spacing,'seed_radius':seed_radius,'target_radius':target_radius,'rounds':rounds,'target_voxels':int(target.sum()),'actual_voxels':int(current.sum()),'target_fill_fraction':float(current.sum()/target.sum())})
report['rounds_vs_requested_sphere']=growth

# Accepted growth cannot cross a full protected plane.
denv.update(VTI_BAND_POLICY={2:{'borrowable':True,'resistance':1.},4:{'borrowable':False,'resistance':float('inf')}},VTI_BORROW_FORCE=3.)
a=np.full((9,9,9),2,dtype=np.uint8);a[:,:,4]=4;a[4,4,2]=9
current=a.copy()
for _ in range(20):
    frontier=denv['vti_dilate6'](current==9)&(current!=9)
    current,claimed,blocked,counts=denv['vti_borrow_round'](current,frontier,9)
    if not claimed.any():break
report['dilation_full_protected_plane']={'target_voxels_far_side':int((current[:,:,5:]==9).sum()),'protected_changed':int(((a!=current)&(a==4)).sum()),'grown_voxels':int((current==9).sum())}

denv.update(VTI_BAND_EDGES=(-np.inf,0.,4.5,5.5,100.,700.,1400.,np.inf),VTI_BACKGROUND_VALUE=0.,VTI_BACKGROUND_ATOL=1e-6)
vals=np.array([-np.inf,-1.,0.,5e-7,4.5,5.5,100.,700.,1400.,np.inf,np.nan])
report['scalar_bands']={'input_repr':[str(x) for x in vals],'labels':denv['vti_make_teaching_bands'](vals).tolist()}
try:
    denv['vti_find_illustrative_seed'](np.full((9,9,9),2,dtype=np.uint8),5,(1.,1.,1.),4.,24.)
    report['inspection_auto_seed_missing_target']='accepted'
except Exception as exc:
    report['inspection_auto_seed_missing_target']=type(exc).__name__+': '+str(exc)

args.output.parent.mkdir(parents=True, exist_ok=True)
args.output.write_text(json.dumps(report,indent=2)+'\n')
print(json.dumps(report,indent=2))
