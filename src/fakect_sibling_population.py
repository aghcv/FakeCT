"""Generate independent bounded aorta siblings from frozen native snapshots.

The original full-body ATN is never copied. Each sibling contains one edited
native attenuation/lineage crop and one broader before/after mask pair. Sources,
ROI, atlas selection, tissue stiffness and erosion safeguards remain pinned.
Requested parameters, native outcomes, collapsed/duplicate geometries and trunk
connectivity are recorded. Native failures invalidate the entire case release.
"""
from collections import Counter
from copy import deepcopy
import configparser
import hashlib
import json
import math
from pathlib import Path
import re

import numpy as np
from scipy import ndimage

from fakect_context_volume import _array_header
from fakect_mask_pair_preview import transfer_pair
from fakect_recipe import apply_recipe, recipe_regions, _roi_geometry
from fakect_recipe_training import _checked_result, _step_outcomes, _MODULES
from fakect_morphology import validate_edit_geometry, MAX_MORPHOLOGY_VOXELS
from fakect_tissues import coarse_labels
from fakect_multiscale_experiment import checked_json, sha256, write_json

ROOT = Path(__file__).resolve().parents[1]
SCHEMA = 'fakect.sibling-population/1'
CASE_SCHEMA = 'fakect.sibling-case/1'
RESERVED_CASES = ('260612', '260617')
EDIT_NAMES = ('arch_outer_expand', 'arch_inner_narrow', 'coa_narrow')
CODE_FILES = tuple(dict.fromkeys(('src/fakect_sibling_population.py',
    'scripts/research/prepare_sibling_population.py', 'src/fakect_mask_pair_preview.py',
    'src/fakect_context_volume.py', 'src/fakect_multiscale_experiment.py') +
    tuple('src/' + name for name in _MODULES)))
_IDENTIFIER = re.compile(r'[A-Za-z0-9][A-Za-z0-9_.-]*\Z')
_FIELDS = {
    'population': {'schema_version', 'output_directory', 'source_manifest',
                   'prepared_root', 'case_ids', 'variants_per_case', 'seed'},
    'variation': {'distance_fraction', 'shape_k', 'window_length_fraction',
                  'directions', 'angular_width_deg'},
    **{'region.' + name: {'path_percent', 'length_percent', 'symmetric_distance_scale'}
       for name in EDIT_NAMES},
}


def _canonical(value):
    return json.dumps(value, sort_keys=True, separators=(',', ':'), allow_nan=False)


def _array_hash(value):
    return hashlib.sha256(np.ascontiguousarray(value).tobytes()).hexdigest()


def _pair(value, name, minimum=0., maximum=100.):
    pair = [float(v.strip()) for v in value.split(',')]
    if (len(pair) != 2 or not np.isfinite(pair).all()
            or not minimum <= pair[0] <= pair[1] <= maximum):
        raise ValueError('Invalid range: ' + name)
    return pair


def load_config(path):
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=('#',),
                                      empty_lines_in_values=False)
    parser.read_string(Path(path).read_text())
    if parser.defaults() or set(parser.sections()) != set(_FIELDS):
        raise ValueError('Expected population, variation and three named region sections')
    c = {name: dict(parser[name]) for name in _FIELDS}
    if any(set(c[name]) != fields for name, fields in _FIELDS.items()):
        raise ValueError('Unknown or missing population fields')
    if any('\n' in v or '\r' in v for section in c.values() for v in section.values()):
        raise ValueError('Each input must occupy one line')
    p = c['population']
    if p['schema_version'] != SCHEMA:
        raise ValueError('Unsupported population schema')
    for key in ('source_manifest', 'prepared_root', 'output_directory'):
        value = Path(p[key]).expanduser()
        p[key] = str((value if value.is_absolute() else ROOT/value).resolve())
    p['case_ids'] = [v.strip() for v in p['case_ids'].split(',')]
    if (not p['case_ids'] or len(set(p['case_ids'])) != len(p['case_ids'])
            or any(not _IDENTIFIER.fullmatch(v) or '..' in v for v in p['case_ids'])
            or set(p['case_ids']) & set(RESERVED_CASES)):
        raise ValueError('Case IDs must be unique development IDs; reserved infants stay closed')
    p['variants_per_case'], p['seed'] = int(p['variants_per_case']), int(p['seed'])
    if not 7 <= p['variants_per_case'] <= 512 or not 0 <= p['seed'] < 2**31:
        raise ValueError('Variants must be7..512 and seed0..2147483647')
    out, prepared = Path(p['output_directory']), Path(p['prepared_root'])
    if out == prepared or out in prepared.parents or prepared in out.parents:
        raise ValueError('Output must be separate from frozen prepared cohorts')
    v = c['variation']
    v['distance_fraction'] = _pair(v['distance_fraction'], 'distance_fraction', .05, 1.)
    v['shape_k'] = _pair(v['shape_k'], 'shape_k', .1, 30.)
    v['window_length_fraction'] = _pair(v['window_length_fraction'], 'window_length_fraction', .1, 1.)
    v['directions'] = [x.strip() for x in v['directions'].split(',')]
    v['angular_width_deg'] = [float(x.strip()) for x in v['angular_width_deg'].split(',')]
    if (not v['directions'] or set(v['directions']) - {'inner', 'outer', 'all'}
            or len(set(v['directions'])) != len(v['directions'])
            or not v['angular_width_deg'] or not np.isfinite(v['angular_width_deg']).all()
            or any(not 0 < w <= 180 for w in v['angular_width_deg'])):
        raise ValueError('Directions are inner/outer/all; asymmetric widths must be<=180 (all means360)')
    for name in EDIT_NAMES:
        r = c['region.' + name]
        r['path_percent'] = _pair(r['path_percent'], name + '.path_percent')
        r['length_percent'] = _pair(r['length_percent'], name + '.length_percent', .1)
        if r['length_percent'][1] > r['path_percent'][1] - r['path_percent'][0]:
            raise ValueError('Length exceeds its allowed parent-path region')
        r['symmetric_distance_scale'] = float(r['symmetric_distance_scale'])
        if not math.isfinite(r['symmetric_distance_scale']) or not 0 < r['symmetric_distance_scale'] <= 1:
            raise ValueError('Symmetric distance scale must be in(0,1]')
    return c


def _source_cases(c):
    p = c['population']
    source = checked_json(p['source_manifest'])
    if source.get('complete') is not True or source.get('schema_version') != 'fakect.context-dataset/1':
        raise ValueError('Expected a completed context-aorta dataset manifest')
    rows = source['samples']
    contexts = {}
    deps = {p['source_manifest']: sha256(p['source_manifest'])}
    for case in p['case_ids']:
        baseline = [r for r in rows if r['case_id'] == case and r['baseline']]
        if len(baseline) != 1:
            raise ValueError('Missing unique original baseline: ' + case)
        baseline = deepcopy(baseline[0])
        folder = Path(p['prepared_root'])/case/'pairs-r2'
        prepared_path = folder/'dataset-manifest.json'
        if prepared_path.resolve() != Path(baseline['prepared_manifest']).resolve():
            raise ValueError('Prepared source path disagrees with context baseline')
        if sha256(prepared_path) != baseline['prepared_manifest_sha256']:
            raise ValueError('Prepared manifest checksum changed: ' + case)
        prepared = json.loads(prepared_path.read_text())
        config = json.loads((folder/'resolved-config.json').read_text())
        # JSON object keys stringify the signed-ID stiffness overrides. Restore
        # the runtime type expected by the unchanged strict morphology engine.
        labels = config.get('reassignment',{}).get('stiffness',{}).get('labels',{})
        if labels:
            config['reassignment']['stiffness']['labels'] = {int(k):v for k,v in labels.items()}
        provenance = json.loads((folder/'data-config.json').read_text())
        if (prepared.get('complete') is not True or (folder/'INCOMPLETE').exists()
                or prepared['case_id'] != case or config['input']['case_id'] != case
                or config['recipe']['steps'] != list(EDIT_NAMES)
                or prepared['anatomy_family'] != baseline['anatomy_family']
                or config['recipe']['roi_role'] != 'selection'
                or config['roi']['shape'] != 'tube'):
            raise ValueError('Unsupported or incomplete original recipe source')
        for key in ('source_atn', 'source_act'):
            identity = baseline[key]
            stat = Path(identity['path']).stat()
            if (stat.st_size, stat.st_mtime_ns) != (identity['bytes'], identity['mtime_ns']):
                raise ValueError('Original source stat changed: ' + case)
        for name in ('dataset-manifest.json', 'resolved-config.json', 'data-config.json',
                     'source-label-catalog.json', 'source-crop.json'):
            path = folder/name
            deps[str(path)] = sha256(path)
            if name != 'dataset-manifest.json' and deps[str(path)] != prepared['artifacts_sha256'][name]:
                raise ValueError('Frozen prepared metadata changed: ' + name)
        contexts[case] = {'baseline': baseline, 'prepared': prepared, 'recipe': config,
                          'provenance': provenance, 'directory': str(folder)}
    return contexts, deps


def _resolved(context):
    g, p = context['prepared']['geometry'], context['provenance']
    path = Path(context['directory'])/'source-label-catalog.json'
    catalog_bytes = path.read_bytes()
    return {'crop_low_ijk': g['crop_origin_ijk'],
        'crop_high_ijk_exclusive': g['crop_high_ijk_exclusive'],
        'spacing_ijk_mm': g['spacing_ijk_mm'], 'shape_kji': g['source_shape_kji'],
        'source_ids': context['prepared']['candidate_source_ids'],
        'catalog': json.loads(catalog_bytes), 'catalog_bytes': catalog_bytes,
        'anatomical_protected_ids': p.get('anatomical_protected_ids', []),
        'catalog_sha256': context['prepared']['catalog_sha256'],
        'audit_sha256': context['prepared']['audit_sha256']}


def _quick_halos(config, resolved):
    """Conservative parameter check without allocating native voxel masks."""
    regions = recipe_regions(config, resolved)
    previous = []
    for name in config['recipe']['steps']:
        edit = config['edits'][name]
        geometry = _roi_geometry(regions['edit:' + name], resolved)
        inherited = sum(distance for lo, hi, distance in previous
                        if max(lo, edit['path_percent'][0]) <= min(hi, edit['path_percent'][1]))
        geometry['growth_reach_mm'] = inherited
        step_config = {**config, 'edit': {k:v for k,v in edit.items()
                        if k not in ('roi', 'iterations', 'path_percent')}}
        validate_edit_geometry(geometry, step_config)
        if edit['operation'] == 'dilation':
            previous.append((*edit['path_percent'], edit['distance_mm']))


def _disabled(edit):
    result = deepcopy(edit)
    result.update(operation='none', distance_mm=0.)
    for key in ('assign_surrounding_tissue', 'min_volume_ratio', 'preserve_connectivity',
                'backoff_factor', 'max_backoff_steps'):
        result.pop(key, None)
    return result


def sample_recipes(c, original, *, source_identity=''):
    """Stratified reproducible parameters, balanced by change-class presence."""
    count, seed = c['population']['variants_per_case'], c['population']['seed']
    case = original['input']['case_id']
    derived = int.from_bytes(hashlib.sha256(f'{seed}:{case}'.encode()).digest()[:8], 'big')
    rng = np.random.default_rng(derived)
    strata = np.column_stack([(rng.permutation(count) + rng.random(count))/count
                              for _ in range(18)])
    groups = ('enlargement_only', 'narrowing_only', 'mixed')
    narrowing = ((1,), (2,), (1,2))
    mixed = ((0,1), (0,2), (0,1,2))
    records = []
    group_counts = Counter()
    for index in range(count):
        # Any remainder goes to the mixed group:256 ->85/85/86.
        group = groups[(index + 2) % 3]
        occurrence = group_counts[group]
        group_counts[group] += 1
        active = (0,) if group == 'enlargement_only' else (
            narrowing[occurrence % 3] if group == 'narrowing_only' else mixed[occurrence % 3])
        edits, parameters = {}, {}
        for j, name in enumerate(EDIT_NAMES):
            old, region = original['edits'][name], c['region.' + name]
            if j not in active:
                edits[name] = _disabled(old)
                continue
            u = strata[index, j*6:j*6+6]
            interpolate = lambda bounds, q: bounds[0] + q*(bounds[1]-bounds[0])
            span = interpolate(region['length_percent'], u[1])
            lo = interpolate([region['path_percent'][0], region['path_percent'][1]-span], u[2])
            direction = c['variation']['directions'][int(u[3]*len(c['variation']['directions']))]
            fraction = interpolate(c['variation']['distance_fraction'], u[0])
            maximum = max([old['distance_mm']] + original.get('sweeps',{}).get(name,{}).get('distance_mm',[]))
            distance = maximum * fraction * (region['symmetric_distance_scale'] if direction == 'all' else 1.)
            window = interpolate(c['variation']['window_length_fraction'], u[5])
            edit = deepcopy(old)
            edit.update(distance_mm=round(distance,6), iterations=1, profile='gaussian',
                profile_axis='tube', path_percent=[round(lo,6),round(lo+span,6)],
                shape_k=round(interpolate(c['variation']['shape_k'],u[4]),6),
                shape_window=[round((1-window)/2,6),round((1+window)/2,6)], direction=direction)
            if direction == 'all':
                edit.pop('angular_width_deg',None)
            else:
                widths = c['variation']['angular_width_deg']
                edit['angular_width_deg'] = widths[int(rng.integers(len(widths)))]
            edits[name] = edit
            parameters[name] = {k:deepcopy(edit[k]) for k in ('distance_mm','path_percent',
                'shape_k','shape_window','direction','angular_width_deg') if k in edit}
        identity = {'case_id':case,'edits':edits,'source_identity':source_identity,
                    'selection':original['selection'],'roi':original['roi'],
                    'reassignment':original['reassignment'],'centerline':original.get('centerline')}
        digest = hashlib.sha256(_canonical(identity).encode()).hexdigest()[:16]
        records.append({'variant_id':'sibling-'+digest,'sample_id':case+':sibling-'+digest,
            'case_id':case,'anatomy_family':original['train']['anatomy_family'],
            'group':group,'active_steps':[EDIT_NAMES[j] for j in active],
            'edits':edits,'parameters':parameters,'split':'unassigned', 'sampling_index':index})
    return records


def plan(c):
    contexts, deps = _source_cases(c)
    cases = {}
    for case, context in contexts.items():
        original, baseline = context['recipe'], context['baseline']
        recipes = sample_recipes(c, original, source_identity=baseline['source_npz_sha256'])
        failures = []
        resolved = _resolved(context)
        for recipe in recipes:
            config = deepcopy(original)
            config['edits'] = recipe['edits']
            try:
                _quick_halos(config,resolved)
            except ValueError as error:
                failures.append({'variant_id':recipe['variant_id'],'error':str(error)})
        if failures:
            raise ValueError(f'{case}: {len(failures)} planned recipes exceed frozen crop halos; '
                             f'first={failures[0]}. Reduce distance/length ranges before freezing.')
        if len({r['variant_id'] for r in recipes}) != len(recipes):
            raise ValueError('Sampler repeated requested recipes')
        cases[case] = {**context, 'recipes':recipes,
            'requested_group_counts':dict(Counter(r['group'] for r in recipes)),
            'baseline_control_policy':'reuse one immutable original; do not count repeated controls as diversity'}
    return {'schema_version':SCHEMA,'config':deepcopy(c),'cases':cases,
        'reserved_case_ids':list(RESERVED_CASES),'reserved_test_evaluated':False,
        'dependency_sha256':deps,'complete':False,
        'scope':'Engineering siblings from existing development families; no new independent patients',
        'registry_policy':'Reference-based immutable manifest; legacy portable registry export requires its own adapter'}


def freeze(c, content):
    lock = plan(c)
    out = Path(c['population']['output_directory'])
    out.mkdir(parents=True,exist_ok=False)
    (out/'INCOMPLETE').touch()
    (out/'input.ini').write_bytes(content)
    lock['input_sha256'] = hashlib.sha256(content).hexdigest()
    lock['code_sha256'] = {}
    for name in CODE_FILES:
        raw = (ROOT/name).read_bytes()
        path = out/'source'/name
        path.parent.mkdir(parents=True,exist_ok=True)
        path.write_bytes(raw)
        lock['code_sha256'][name] = hashlib.sha256(raw).hexdigest()
    write_json(out/'population-lock.json',lock)
    (out/'INCOMPLETE').unlink()
    return lock


def inspect_lock(c, content):
    out = Path(c['population']['output_directory'])
    lock = checked_json(out/'population-lock.json')
    if lock['config'] != c or lock['input_sha256'] != hashlib.sha256(content).hexdigest():
        raise ValueError('Frozen population input changed; use a new revision')
    if sha256(out/'input.ini') != lock['input_sha256'] or set(lock['code_sha256']) != set(CODE_FILES):
        raise ValueError('Frozen population input/source inventory changed')
    for name, digest in lock['code_sha256'].items():
        if sha256(ROOT/name) != digest or sha256(out/'source'/name) != digest:
            raise ValueError('Population implementation changed: '+name)
    for name, digest in lock['dependency_sha256'].items():
        if sha256(name) != digest:
            raise ValueError('Population source metadata changed: '+name)
    for case in lock['cases'].values():
        labels = case['recipe'].get('reassignment',{}).get('stiffness',{}).get('labels',{})
        if labels:
            case['recipe']['reassignment']['stiffness']['labels'] = {int(k):v for k,v in labels.items()}
        for key in ('source_atn','source_act'):
            source = case['baseline'][key]
            stat = Path(source['path']).stat()
            if (stat.st_size,stat.st_mtime_ns) != (source['bytes'],source['mtime_ns']):
                raise ValueError('Original source stat changed')
    return lock


def _load_source(context):
    baseline, prepared = context['baseline'], context['prepared']
    source_path = Path(baseline['source_npz'])
    if sha256(source_path) != baseline['source_npz_sha256']:
        raise ValueError('Frozen native source snapshot changed')
    shape = tuple(prepared['geometry']['crop_shape_kji'])
    if int(np.prod(shape)) > MAX_MORPHOLOGY_VOXELS:
        raise ValueError('Native source exceeds the morphology budget')
    for key in ('original_labels','original_attenuation_per_pixel','target_mask','roi_mask'):
        if _array_header(source_path,key)[0] != shape:
            raise ValueError('Native source array geometry changed')
    with np.load(source_path,allow_pickle=False) as data:
        labels, atn = data['original_labels'], data['original_attenuation_per_pixel']
        selected, roi = data['target_mask'].astype(bool), data['roi_mask'].astype(bool)
    for key,value in (('act',labels),('atn',atn),('selected',selected)):
        if _array_hash(value) != prepared['source_crop_sha256'][key]:
            raise ValueError('Frozen native array checksum changed: '+key)
    resolved = _resolved(context)
    candidates = np.isin(labels,resolved['source_ids'])
    if not np.array_equal(selected,candidates & roi):
        raise ValueError('Original selected ancestry disagrees with pinned IDs/ROI')
    arrays = {'act':labels,'atn':atn,'selected':selected,'roi':roi,
        'tissue':coarse_labels(labels,resolved['catalog']),'candidates':candidates,
        'attenuation_cm_inverse':atn/(resolved['spacing_ijk_mm'][0]/10)}
    for a in arrays.values():
        a.flags.writeable = False
    path = Path(baseline['mask_npz'])
    if sha256(path) != baseline['mask_npz_sha256']:
        raise ValueError('Frozen broad baseline pair changed')
    shape,_ = _array_header(path,'before')
    if math.prod(shape) > 32_000_000:
        raise ValueError('Broad baseline exceeds the mask budget')
    with np.load(path,allow_pickle=False) as data:
        broad = data['before'].astype(bool)
        origin = data['crop_origin_ijk']
        if not np.array_equal(data['before'],data['after']):
            raise ValueError('Original baseline contains net edits')
    # The existing descriptor retains the original selected-lineage reference
    # hash, preceding the later broad thoracic mask transfer. The broad mask is
    # independently bound by mask_npz_sha256; do not reinterpret that hash.
    if _array_hash(selected.astype(np.uint8)) != baseline['original_reference_mask_sha256']:
        raise ValueError('Original selected anatomical reference mask changed')
    return arrays,resolved,broad,origin


def trunk_connectivity(before, after, origins):
    """Count final components containing descendants of the original trunk.

    The largest initial component is the trunk. Small existing ROI fragments
    remain documented rather than forcing the complete selected mask connected.
    """
    structure = ndimage.generate_binary_structure(3,1)
    components,n = ndimage.label(before,structure)
    if not n:
        raise ValueError('Original selected mask has no trunk')
    sizes = np.bincount(components.ravel())[1:]
    trunk = int(np.argmax(sizes))+1
    descendants = (origins>=0) & (components.ravel()[np.maximum(origins,0)]==trunk) & after
    final,count = ndimage.label(after,structure)
    ids = np.unique(final[descendants])
    ids = ids[ids!=0]
    return {'passed':len(ids)==1,'original_components':int(n),
        'original_trunk_voxels':int(sizes[trunk-1]),'final_components':int(count),
        'trunk_descendant_components':int(len(ids)),
        'trunk_descendant_voxels':int(descendants.sum()),
        'interpretation':'native6-neighbor continuity, not a minimum clinically plausible lumen caliber'}


def _generate(c,lock,case_id,output,recipes,*,trial=False):
    if case_id not in lock['cases'] or case_id in RESERVED_CASES:
        raise ValueError('Unknown or reserved case')
    context = lock['cases'][case_id]
    arrays,resolved,broad,broad_origin = _load_source(context)
    output.mkdir(parents=True,exist_ok=False)
    (output/'INCOMPLETE').touch()
    (output/'variants').mkdir()
    (output/'masks').mkdir()
    rows,records,groups = [],[],{}
    baseline = deepcopy(context['baseline'])
    rows.append(baseline)
    initial_hashes = {name:_array_hash(arrays[name]) for name in ('act','atn','selected')}
    for index,recipe in enumerate(recipes):
        record = deepcopy(recipe)
        record['status'] = 'failed'
        result = None
        try:
            config = deepcopy(context['recipe'])
            config['edits'] = deepcopy(recipe['edits'])
            result = apply_recipe(arrays,resolved,config)
            image,mask = _checked_result(arrays,resolved,result)
            continuity = trunk_connectivity(arrays['selected'],mask.astype(bool),result['target_origin_index'])
            record['trunk_connectivity'] = continuity
            if not continuity['passed']:
                raise ValueError('Native edited trunk disconnected or completely removed')
            pair = transfer_pair(broad,broad_origin,arrays['selected'],mask,
                resolved['crop_low_ijk'],result['target_origin_index'])
            mask_hash = _array_hash(pair['after'].astype(np.uint8))
            duplicate = groups.get(mask_hash)
            groups.setdefault(mask_hash,recipe['sample_id'])
            record.update(status='ok',mask_counts=pair['counts'],
                image_sha256=_array_hash(image),after_mask_sha256=mask_hash,
                duplicate_of=duplicate,baseline_equivalent=not bool(pair['added'].any() | pair['removed'].any()),
                step_outcomes=_step_outcomes(result['summary']),
                scalar_changed_voxels=int(result['scalar_changed_mask'].sum()))
            variant_path = output/'variants'/(recipe['variant_id']+'.npz')
            with variant_path.open('xb') as stream:
                np.savez_compressed(stream,image=image,mask=mask,
                    edited_labels=result['edited_labels'],changed_mask=result['changed_mask'],
                    scalar_changed_mask=result['scalar_changed_mask'],
                    target_origin_index=result['target_origin_index'],edit_region_mask=result['edit_region_mask'],
                    metadata_json=np.asarray(_canonical(record)))
            mask_path = output/'masks'/(recipe['variant_id']+'.npz')
            with mask_path.open('xb') as stream:
                np.savez_compressed(stream,before=pair['before'].astype(np.uint8),
                    after=pair['after'].astype(np.uint8),added=pair['added'].astype(np.uint8),
                    removed=pair['removed'].astype(np.uint8),crop_origin_ijk=pair['crop_origin_ijk'],
                    spacing_ijk_mm=baseline['spacing_ijk_mm'],sample_id=np.asarray(recipe['sample_id']))
            row = deepcopy(baseline)
            row.update(sample_id=recipe['sample_id'],variant_id=recipe['variant_id'],baseline=False,
                baseline_equivalent=record['baseline_equivalent'],split='unassigned',
                variant_npz=str(variant_path),variant_npz_sha256=sha256(variant_path),
                mask_npz=str(mask_path),mask_npz_sha256=sha256(mask_path),mask_counts=pair['counts'],
                parameters=deepcopy(recipe['parameters']),step_outcomes=record['step_outcomes'],
                population_group=recipe['group'],duplicate_of=duplicate,
                trunk_connectivity=continuity)
            rows.append(row)
        except (ValueError,RuntimeError,OSError) as error:
            record.update(status='failed',error_type=type(error).__name__,error=str(error))
        finally:
            del result
        records.append(record)
        with (output/'progress.jsonl').open('a') as stream:
            stream.write(_canonical(record)+'\n')
        print(f'{case_id} {index+1}/{len(recipes)} {recipe["variant_id"]} {record["status"]}',flush=True)
    if any(_array_hash(arrays[name])!=value for name,value in initial_hashes.items()):
        raise ValueError('Recipe mutated its immutable source snapshot')
    failed = sum(r['status']!='ok' for r in records)
    document = {'schema_version':CASE_SCHEMA,'complete':failed==0,'trial':trial,
        'case_id':case_id,'anatomy_family':baseline['anatomy_family'],'samples':rows,'records':records,
        'requested_siblings':len(recipes),'successful_siblings':len(rows)-1,'failed_siblings':failed,
        'unique_edited_geometry_count':len(groups),
        'duplicate_siblings':sum(r.get('duplicate_of') is not None for r in records),
        'baseline_equivalent_siblings':sum(r.get('baseline_equivalent',False) for r in records),
        'requested_group_counts':dict(Counter(r['group'] for r in recipes)),
        'source_lock_sha256':sha256(Path(c['population']['output_directory'])/'population-lock.json'),
        'original_sources_modified':False,'full_body_siblings_written':False,
        'source_snapshot_reused':baseline['source_npz'],'classifier_started':False,
        'scope':'Native engineering generation; masks/attenuation-copy proxies are not clinical CT validation'}
    path = output/'dataset-manifest.json'
    write_json(path,document)
    if failed:
        raise ValueError(f'{case_id}: {failed} native failures; case remains INCOMPLETE; inspect {path}')
    # Rows bind the completed case receipt, without a recursive self hash. The
    # signed case receipt retains original prepared metadata as source provenance.
    (output/'INCOMPLETE').unlink()
    return document


def prepare_case(c,lock,case_id):
    out = Path(c['population']['output_directory'])/'cases'/case_id
    return _generate(c,lock,case_id,out,lock['cases'][case_id]['recipes'])


def trial_case(c,lock,case_id,count=7):
    if type(count) is not int or not 1<=count<=21:
        raise ValueError('Trial count must be1..21')
    out = Path(c['population']['output_directory'])/'trials'/case_id/f'trial-{count}'
    recipes = lock['cases'][case_id]['recipes']
    # First exercise distinct active-step patterns, then fill additional trial
    # requests. A seven-recipe smoke covers all seven patterns in the full plan.
    chosen,patterns = [],set()
    for recipe in recipes:
        pattern = tuple(recipe['active_steps'])
        if pattern not in patterns:
            patterns.add(pattern)
            chosen.append(recipe)
            if len(chosen)==count:
                break
    for recipe in recipes:
        if len(chosen)==count:
            break
        if recipe not in chosen:
            chosen.append(recipe)
    return _generate(c,lock,case_id,out,chosen,trial=True)


def collect(c,lock):
    out = Path(c['population']['output_directory'])
    rows,case_receipts = [],[]
    for case,context in lock['cases'].items():
        path = out/'cases'/case/'dataset-manifest.json'
        document = checked_json(path)
        if (document.get('complete') is not True or document.get('trial') is not False
                or (path.parent/'INCOMPLETE').exists()
                or document['source_lock_sha256']!=sha256(out/'population-lock.json')
                or document['requested_siblings']!=c['population']['variants_per_case']
                or document['failed_siblings']!=0):
            raise ValueError('Incomplete or incompatible generated case: '+case)
        for row in document['samples']:
            for key in ('variant_npz','mask_npz'):
                if sha256(row[key])!=row[key+'_sha256']:
                    raise ValueError('Generated payload changed: '+row['sample_id'])
            row = deepcopy(row)
            if not row['baseline']:
                row['prepared_manifest'] = str(path)
                row['prepared_manifest_sha256'] = sha256(path)
            rows.append(row)
        case_receipts.append({'case_id':case,'manifest':str(path),'sha256':sha256(path),
            'anatomy_family':context['baseline']['anatomy_family'],
            'unique_edited_geometry_count':document['unique_edited_geometry_count'],
            'duplicate_siblings':document['duplicate_siblings'],
            'baseline_equivalent_siblings':document['baseline_equivalent_siblings']})
    ids = [r['sample_id'] for r in rows]
    if len(set(ids))!=len(ids):
        raise ValueError('Repeated sibling identity in population')
    manifest = {'schema_version':SCHEMA,'complete':True,'case_ids':list(lock['cases']),
        'cases':case_receipts,'samples':rows,'sample_count':len(rows),
        'edited_siblings':sum(not r['baseline'] for r in rows),
        'original_baselines':sum(r['baseline'] for r in rows),
        'source_lock_sha256':sha256(out/'population-lock.json'),
        'image_units':'cm^-1','image_method':'attenuation_copy_proxy',
        'reserved_case_ids':list(RESERVED_CASES),'reserved_test_evaluated':False,
        'full_body_siblings_written':False,'classifier_started':False,
        'registry_policy':lock['registry_policy'],'scope':lock['scope']}
    if (out/'dataset-manifest.json').exists():
        raise FileExistsError('Population already collected; preserve its immutable receipt')
    write_json(out/'dataset-manifest.json',manifest)
    return manifest
