"""Sibling diversity, immutable export, failure and family/source contracts."""
from contextlib import redirect_stdout
from copy import deepcopy
import io
import json
from pathlib import Path
import sys
import unittest
from unittest.mock import patch

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT/'src'))

import fakect_sibling_population as population
from fakect_context_volume import ContextVolume
from fakect_roi import resolve_preview, prepare_crop
from fakect_training_data import _geometry,_json_value
from fakect_multiscale_experiment import write_json,sha256
import test_training_data


class PopulationTests(unittest.TestCase):
    def setUp(self):
        fixture = test_training_data.TrainingDataTests()
        fixture.setUp()
        self.addCleanup(fixture.doCleanups)
        self.root = fixture.root
        self.c = population.load_config(ROOT/'archive/research/configs/cohorts/aorta-diversity.ini')
        self.c['population'].update(output_directory=str(self.root/'population'),
            source_manifest=str(self.root/'context.json'),prepared_root=str(self.root/'prepared'),
            case_ids=['001'])
        original = deepcopy(fixture.config)
        original.pop('edit')
        original['study']['schema_version'] = 'fakect.recipe-cohort/1'
        original['roi'].update(shape='tube',center_ijk=[[15,15,9],[15,15,12],[15,15,18],[15,15,21]],
                               radius_mm=[3,3,3,3],crop_half_width_mm=20.,coordinate_reviewed=True)
        original['rois'] = {}
        original['recipe'] = {'steps':list(population.EDIT_NAMES),'overlap':'sequential','roi_role':'selection'}
        original['edits'] = {}
        for name in population.EDIT_NAMES:
            operation = 'dilation' if name=='arch_outer_expand' else 'erosion'
            original['edits'][name] = dict(roi='main',iterations=1,operation=operation,distance_mm=2.,
                profile='gaussian',profile_axis='tube',shape_k=6.,shape_window=[.05,.95],
                path_percent=[10.,90.],direction='all')
            if operation=='erosion':
                original['edits'][name].update(assign_surrounding_tissue=True,min_volume_ratio=.2,
                    preserve_connectivity=True,backoff_factor=.5,max_backoff_steps=8)
        original['sweeps'] = {'arch_outer_expand':{'distance_mm':[1.,2.]},'coa_narrow':{'distance_mm':[1.,2.]}}
        original['train'].update(target_scope='selected_lineage',split_mode='unassigned')
        self.original = json.loads(json.dumps(original,default=str))
        resolved = resolve_preview(original)
        arrays,sources = prepare_crop(resolved,original)
        geometry = _json_value(_geometry(resolved))
        folder = self.root/'prepared/001/pairs-r2'
        folder.mkdir(parents=True)
        source_path = folder/'source.npz'
        np.savez_compressed(source_path,original_labels=arrays['act'],
            original_attenuation_per_pixel=arrays['atn'],target_mask=arrays['selected'].astype(np.uint8),
            roi_mask=arrays['roi'],geometry_json=np.asarray(json.dumps(geometry)),
            catalog_json=np.asarray(json.dumps(resolved['catalog'])))
        mask_path = self.root/'baseline-masks.npz'
        broad = np.isin(arrays['act'],resolved['source_ids']).astype(np.uint8)
        np.savez_compressed(mask_path,before=broad,after=broad,added=np.zeros_like(broad),
            removed=np.zeros_like(broad),crop_origin_ijk=geometry['crop_origin_ijk'],
            spacing_ijk_mm=geometry['spacing_ijk_mm'],sample_id=np.asarray('001:baseline'))
        variant = self.root/'baseline.npz'
        np.savez_compressed(variant,image=(arrays['atn']/(geometry['spacing_ijk_mm'][0]/10)).astype(np.float32))
        source_hashes = {k:population._array_hash(arrays[k]) for k in ('act','atn','selected')}
        prepared = {'complete':True,'case_id':'001','anatomy_family':original['train']['anatomy_family'],
            'geometry':geometry,'candidate_source_ids':list(resolved['source_ids']),
            'source_crop_sha256':source_hashes,'catalog_sha256':resolved['catalog_sha256'],
            'audit_sha256':resolved['audit_sha256']}
        write_json(folder/'resolved-config.json',self.original)
        write_json(folder/'data-config.json',{'anatomical_protected_ids':[]})
        (folder/'source-label-catalog.json').write_bytes(resolved['catalog_bytes'])
        write_json(folder/'source-crop.json',{'source_crop_sha256':source_hashes})
        prepared['artifacts_sha256'] = {p.name:sha256(p) for p in folder.iterdir() if p.is_file()}
        write_json(folder/'dataset-manifest.json',prepared)
        atn,act = Path(resolved['source_files']['atn']),Path(resolved['source_files']['act'])
        identity = lambda p:dict(path=str(p),bytes=p.stat().st_size,mtime_ns=p.stat().st_mtime_ns)
        self.baseline = dict(sample_id='001:baseline',case_id='001',variant_id='baseline',baseline=True,
            anatomy_family=original['train']['anatomy_family'],baseline_equivalent=True,split='unassigned',
            family_status='provisional',family_verified=False,frame=1,
            source_atn=identity(atn),source_act=identity(act),source_shape_kji=geometry['source_shape_kji'],
            spacing_ijk_mm=geometry['spacing_ijk_mm'],source_npz=str(source_path),source_npz_sha256=sha256(source_path),
            variant_npz=str(variant),variant_npz_sha256=sha256(variant),
            mask_npz=str(mask_path),mask_npz_sha256=sha256(mask_path),
            edit_crop_origin_ijk=geometry['crop_origin_ijk'],edit_crop_shape_kji=geometry['crop_shape_kji'],
            original_reference_mask_sha256=population._array_hash(arrays['selected'].astype(np.uint8)),
            prepared_manifest=str(folder/'dataset-manifest.json'),
            prepared_manifest_sha256=sha256(folder/'dataset-manifest.json'))
        write_json(self.root/'context.json',dict(schema_version='fakect.context-dataset/1',complete=True,
                                               samples=[self.baseline]))
        self.c['variation']['directions'] = ['all']
        self.c['variation']['distance_fraction'] = [.5,1.]

    # Purpose: Verify recipe sampling is deterministic, balanced and preserves source settings.
    # Inputs: Frozen source recipe, seed-based 256-sibling sampling and alternate seed.
    # Outputs: 85/85/86 groups, unique varied recipes, bounded settings, copied safeguards and unchanged source.
    def test_sampler_is_reproducible_balanced_varied_and_preserves_source_settings(self):
        original = deepcopy(self.original)
        rows = population.sample_recipes(self.c,original,source_identity='sourcehash')
        self.assertEqual(rows,population.sample_recipes(self.c,original,source_identity='sourcehash'))
        self.assertEqual(original,self.original)
        self.assertEqual(len(rows),256)
        counts = {g:sum(r['group']==g for r in rows) for g in ('enlargement_only','narrowing_only','mixed')}
        self.assertEqual(counts,dict(enlargement_only=85,narrowing_only=85,mixed=86))
        self.assertEqual(len({r['sample_id'] for r in rows}),256)
        self.assertEqual(len({tuple(r['active_steps']) for r in rows}),7)
        for row in rows:
            for name,e in row['edits'].items():
                if e['operation']=='none':
                    self.assertEqual(e['distance_mm'],0.)
                    self.assertNotIn('preserve_connectivity',e)
                    continue
                self.assertEqual(e['operation'],original['edits'][name]['operation'])
                self.assertEqual(e['iterations'],1)
                r = self.c['region.'+name]
                self.assertLessEqual(e['distance_mm'],2.*r['symmetric_distance_scale'])
                self.assertGreaterEqual(e['path_percent'][0],r['path_percent'][0])
                self.assertLessEqual(e['path_percent'][1],r['path_percent'][1])
                self.assertNotIn('angular_width_deg',e)
                if e['operation']=='erosion':
                    for key in ('min_volume_ratio','preserve_connectivity','backoff_factor','max_backoff_steps'):
                        self.assertEqual(e[key],original['edits'][name][key])
        other = deepcopy(self.c);other['population']['seed']+=1
        self.assertNotEqual(rows,population.sample_recipes(other,original,source_identity='sourcehash'))

    # Purpose: Verify sampled directional widths are valid engine inputs.
    # Inputs: Variation directions inner, outer and all.
    # Outputs: All requested directions represented; all has no width and directed widths at most 180 degrees.
    def test_direction_widths_are_real_engine_inputs_not_unsupported_approximations(self):
        c = deepcopy(self.c);c['variation']['directions']=['inner','outer','all']
        rows = population.sample_recipes(c,self.original)
        self.assertEqual({e['direction'] for r in rows for e in r['edits'].values() if e['operation']!='none'},
                         {'inner','outer','all'})
        for r in rows:
            for e in r['edits'].values():
                if e['direction']=='all':
                    self.assertNotIn('angular_width_deg',e)
                else:
                    self.assertLessEqual(e['angular_width_deg'],180)

    # Purpose: Verify trunk continuity tolerates old fragments but rejects new breaks.
    # Inputs: Connected trunk plus old island, then removal of the trunk midpoint.
    # Outputs: Original passes; broken trunk fails with two descendant components.
    def test_trunk_continuity_allows_old_fragments_but_rejects_new_break(self):
        before = np.zeros((9,5,5),bool);before[1:8,2,2]=True;before[0,0,0]=True
        origins = np.where(before,np.arange(before.size).reshape(before.shape),-1)
        self.assertTrue(population.trunk_connectivity(before,before,origins)['passed'])
        after = before.copy();after[4,2,2]=False
        removed_origins = origins.copy();removed_origins[4,2,2]=-1
        audit = population.trunk_connectivity(before,after,removed_origins)
        self.assertFalse(audit['passed']);self.assertEqual(audit['trunk_descendant_components'],2)

    # Purpose: Verify population plans avoid native images and pin source metadata.
    # Inputs: Metadata-only 256-recipe plan, unknown case and frozen source-crop mutation.
    # Outputs: No plan outputs/reads, unknown case rejected and changed source metadata invalidates lock.
    def test_metadata_plan_never_loads_images_and_pins_changed_sources(self):
        with patch.object(population,'_load_source',side_effect=AssertionError('native read')):
            lock = population.plan(self.c)
        self.assertEqual(len(lock['cases']['001']['recipes']),256)
        self.assertFalse(Path(self.c['population']['output_directory']).exists())
        bad = deepcopy(self.c);bad['population']['case_ids']=['260612']
        with self.assertRaises(ValueError):
            population.plan(bad)
        self.c['population']['variants_per_case']=21
        content=b'# bounded fixture input\n'
        population.freeze(self.c,content)
        lock = population.inspect_lock(self.c,content)
        (self.root/'prepared/001/pairs-r2/source-crop.json').write_text('{}')
        with self.assertRaisesRegex(ValueError,'source metadata'):
            population.inspect_lock(self.c,content)

    # Purpose: Verify native sibling export is readable and preserves original payloads.
    # Inputs: Frozen 21-variant plan and one selected maximum enlargement trial.
    # Outputs: Baseline plus sibling, finite image reads, exact change masks, unchanged sources and no overwrite.
    def test_single_native_export_is_readable_and_original_files_unchanged(self):
        self.c['population']['variants_per_case']=21
        content=b'# bounded fixture native trial\n'
        lock=population.freeze(self.c,content)
        native=population._load_source(lock['cases']['001'])
        self.assertTrue(all(not a.flags.writeable for a in native[0].values()))
        oldhashes={name:sha256(self.baseline[name]) for name in ('source_npz','mask_npz','variant_npz')}
        recipes=lock['cases']['001']['recipes']
        selected=[max((r for r in recipes if r['group']=='enlargement_only'),
                      key=lambda r:r['edits']['arch_outer_expand']['distance_mm'])]
        directory=self.root/'native-export'
        with redirect_stdout(io.StringIO()):
            result=population._generate(self.c,lock,'001',directory,selected,trial=True)
        self.assertTrue(result['complete']);self.assertEqual(len(result['samples']),2)
        row=result['samples'][1]
        with ContextVolume(row) as volume:
            block=volume.read_block(row['edit_crop_origin_ijk'][::-1],row['edit_crop_shape_kji'],row['spacing_ijk_mm'])
            self.assertTrue(np.isfinite(block['image']).all())
        with np.load(row['mask_npz']) as z:
            np.testing.assert_array_equal(z['added'],(z['after']==1)&(z['before']==0))
            np.testing.assert_array_equal(z['removed'],(z['before']==1)&(z['after']==0))
        self.assertEqual(row['original_reference_mask_sha256'],self.baseline['original_reference_mask_sha256'])
        self.assertEqual(oldhashes,{name:sha256(self.baseline[name]) for name in oldhashes})
        self.assertFalse((directory/'INCOMPLETE').exists())
        with self.assertRaises(FileExistsError):
            population._generate(self.c,lock,'001',directory,selected,trial=True)

    # Purpose: Verify native failures prevent population release while retaining evidence.
    # Inputs: Two sibling recipes with apply_recipe raising bad-native-edit ValueError.
    # Outputs: Two failure records, incomplete manifest/marker and collect refuses release.
    def test_native_failure_is_recorded_and_prevents_population_release(self):
        self.c['population']['variants_per_case']=21
        lock=population.freeze(self.c,b'# native failure fixture\n')
        directory=Path(self.c['population']['output_directory'])/'cases/001'
        recipes=lock['cases']['001']['recipes'][:2]
        with redirect_stdout(io.StringIO()),patch.object(population,'apply_recipe',side_effect=ValueError('bad native edit')):
            with self.assertRaisesRegex(ValueError,'2 native failures'):
                population._generate(self.c,lock,'001',directory,recipes)
        document=json.loads((directory/'dataset-manifest.json').read_text())
        self.assertFalse(document['complete']);self.assertEqual(document['failed_siblings'],2)
        self.assertEqual(len(document['records']),2);self.assertTrue((directory/'INCOMPLETE').exists())
        with self.assertRaisesRegex(ValueError,'Incomplete'):
            population.collect(self.c,lock)


if __name__=='__main__':
    unittest.main()
