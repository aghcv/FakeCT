"""Portable 64³/96³ CT-like crops, generated without XCAT or production geometry.

This is a mathematical software fixture, not sampled patient/XCAT anatomy or a
physics-validated CT. ACT IDs and per-pixel ATN retain the native file contract.
The same physical field of view is sampled at two anisotropic resolutions.
"""
import configparser
import hashlib
import json
from pathlib import Path

import numpy as np


def _sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_voxel_fixture(root, size=64, reviewed=False):
    """Write a complete generation-only example and return independent oracles.

    Coordinates in the INI are local native i,j,k. ``crop_origin_ijk`` maps that
    local source to the original logical volume; it never changes local indices.
    Paired float32 ACT + ATN payloads are exactly 2 MiB / 6.75 MiB.
    """
    if size not in (64, 96):
        raise ValueError('Portable voxel fixture size must be 64 or 96')
    root = Path(root).resolve()
    root.mkdir(parents=True, exist_ok=True)
    case_id = f'voxel{size}'
    case = root / case_id
    case.mkdir()
    shape = (size, size, size)
    spacing = tuple(v * 64 / size for v in (.8, .9, 1.2))
    crop_origin = (111, 211, 311)
    original_shape = (512, 512, 512)
    k, j, i = np.ogrid[:size, :size, :size]
    x, y, z = i * spacing[0], j * spacing[1], k * spacing[2]

    # Direct analytic transverse disks about a gently curved centerline. This
    # deliberately does not call FakeCT's tube_mask or morphology algorithms.
    cz = np.clip(z, 21., 53.)
    cx = 25. + 3.5 * np.sin(np.pi * (cz - 21.) / 32.)
    cy = 28. + .1 * (cz - 21.)
    main = ((x - cx) ** 2 + (y - cy) ** 2 + (z - cz) ** 2 <= 2.25 ** 2)
    branch = np.zeros(shape, dtype=bool)
    for t in np.linspace(0., 1., 65):
        center = (28.5 + 2.5*t, 29.6 - 9.6*t, 37. + t)
        branch |= ((x-center[0])**2 + (y-center[1])**2 + (z-center[2])**2 <= 1.35**2)
    branch &= ~main
    neighbor = np.broadcast_to(((x-36.5)**2 + (y-30.)**2 <= 1.5**2) & (z >= 21) & (z <= 53), shape).copy()
    vein = np.broadcast_to(((x-18.5)**2 + (y-29.)**2 <= 1.6**2) & (z >= 25) & (z <= 49), shape).copy()
    bone = np.broadcast_to((x >= cx+2.5) & (x <= cx+3.8) & (abs(y-cy) <= 1.1) & (z >= 25) & (z <= 31), shape).copy()
    resistant_skin = np.broadcast_to((x <= cx-2.5) & (x >= cx-3.8) & (abs(y-cy) <= 1.1) & (z >= 25) & (z <= 31), shape).copy()
    outside = np.broadcast_to((i < 2) | (j < 2) | (k < 2) | (i >= size-2) | (j >= size-2) | (k >= size-2), shape)
    shell = np.broadcast_to((i == 2) | (j == 2) | (k == 2) | (i == size-3) | (j == size-3) | (k == size-3), shape) & ~outside
    lung = np.broadcast_to((x > 40) & (y > 25) & (z > 15) & (z < 60), shape).copy()
    unknown = np.broadcast_to((x > 10) & (x < 12) & (y > 15) & (y < 18) & (z > 34) & (z < 38), shape).copy()

    labels = np.broadcast_to(np.where(y < 30., 20., 21.), shape).astype('<f4', copy=True)
    for mask, code in ((lung,22), (unknown,99), (vein,-10), (neighbor,9),
                       (branch,8), (main,-7), (bone,30), (resistant_skin|shell,40), (outside,0)):
        labels[mask] = code
    # Every label's attenuation is known independently. Arteries intentionally
    # share a signal so geometry/identity, not intensity alone, must protect them.
    signals = {0:0., -7:.24, 8:.24, 9:.24, -10:.22, 20:.18,
               21:.20, 22:.035, 30:.43, 40:.19, 99:.13}
    attenuation = np.empty(shape, dtype='<f4')
    for code, signal in signals.items():
        attenuation[labels == code] = signal
    per_pixel = (attenuation * np.float32(spacing[0]/10)).astype('<f4')
    act_path, atn_path = case/f'{case_id}_act_1.bin', case/f'{case_id}_atn_1.bin'
    labels.tofile(act_path)
    per_pixel.tofile(atn_path)
    par, log = case/f'{case_id}.par', case/f'{case_id}.log'
    provenance = dict(crop_origin_ijk=crop_origin, original_shape_kji=original_shape,
                      coordinate_semantics='Local crop voxel centers; original ijk = local ijk + crop_origin_ijk',
                      fixture_kind='analytic-software-phantom-not-licensed-XCAT')
    par.write_text(json.dumps(dict(shape_kji=shape, spacing_ijk_mm=spacing, **provenance), indent=2)+'\n')
    log.write_text('ACT signed anatomical IDs; ATN cm^-1 * i spacing(cm); mathematical software fixture\n')
    categories = [(0,'background'),(1,'soft_tissue'),(2,'bone'),(5,'artery'),
                  (6,'vein'),(7,'lung'),(10,'skin'),(255,'unknown')]
    definitions = [(0,'background',0,'background'), (-7,'analytic_aorta',5,'artery'),
                   (8,'connected_branch',5,'artery'),(9,'nearby_artery',5,'artery'),
                   (-10,'nearby_vein',6,'vein'),(20,'soft_a',1,'soft_tissue'),
                   (21,'soft_b',1,'soft_tissue'),(22,'lung',7,'lung'),
                   (30,'bone_obstacle',2,'bone'),(40,'skin_obstacle',10,'skin'),
                   (99,'known_unclassified',255,'unknown')]
    catalog = dict(policy_version='portable-voxel-fixture/1', sources={}, crop_provenance=provenance,
                   categories=[dict(id=c,name=n) for c,n in categories],
                   records=[dict(original_id=c,original_name=n,
                                 classification=dict(tissue_id=t,tissue_name=g,structure_type=g))
                            for c,n,t,g in definitions])
    catalog_path = root/'catalog.json'
    catalog_path.write_text(json.dumps(catalog, indent=2)+'\n')
    audit = dict(cases=[dict(case_id=case_id, directory=str(case), shape_kji=shape,
                            spacing_ijk_mm=spacing, par_path=str(par), log_path=str(log), **provenance)],
                 source_metadata_sha256={str(p):_sha(p) for p in (par,log)})
    audit_path = root/'audit.json'
    audit_path.write_text(json.dumps(audit, indent=2)+'\n')
    physical_nodes = [(25.,28.,21.), (27.4748737,28.8,29.), (28.5,29.6,37.),
                      (27.4748737,30.4,45.), (25.,31.2,53.)]
    nodes = tuple(tuple(p[a]/spacing[a] for a in range(3)) for p in physical_nodes)
    flag = 'true' if reviewed else 'false'
    edit = dict(roi='main', iterations='1', profile='uniform', profile_axis='tube',
                shape_k='6', shape_window='0,1', direction='all')
    sections = {
        'study':dict(schema_version='fakect.recipe-cohort/1', name=f'portable-{case_id}'),
        'input':dict(root=str(root),case_id=case_id,frame='1',audit=str(audit_path),catalog=str(catalog_path)),
        'selection':dict(tissue='artery',source_ids=''),
        'roi':dict(shape='tube',center_ijk=' ; '.join(','.join(f'{c:.9g}' for c in n) for n in nodes),
                   radius_mm='2.8,2.8,2.8,2.8,2.8',crop_half_width_mm='10',coordinate_reviewed=flag),
        'preview':dict(mode='roi-selection',content='attenuation',slice_ijk='roi',overlay_opacity='.2',
                       volume_opacity='.25',context_tissues='artery,soft_tissue,bone,skin',context_opacity='.08',volume_stride='3'),
        'recipe':dict(roi_role='selection',steps='expansion,local_narrowing',overlap='sequential'),
        'edit.expansion':dict(edit, operation='dilation',path_percent='10,40',distance_mm='2'),
        'edit.local_narrowing':dict(edit, operation='erosion',path_percent='65,85',distance_mm='1.2',
                                    assign_surrounding_tissue='true',preserve_connectivity='true',
                                    min_volume_ratio='.4',backoff_factor='.5',max_backoff_steps='6'),
        'reassignment':dict(mode='stiffness',allowed_tissues='',max_distance_mm='3',unresolved='preserve'),
        'stiffness':dict(default='.05',bone='1',skin='.95',soft_tissue='.05',artery='.05',vein='.05',lung='.05',unknown='.05'),
        'train':dict(stage='preview',preflight_directory=str(root/'preflight'),freeze_directory=str(root/'freeze'),
                     dataset_directory=str(root/'pairs'),anatomy_family='analytic-voxel-configuration',target_scope='selected_lineage',
                     include_baseline='true',max_variants='8',image_method='attenuation_copy_proxy',parameters_reviewed=flag),
        'sweep.expansion':dict(distance_mm='0,2'),
        'sweep.local_narrowing':dict(distance_mm='0,1.2'),
        'output':dict(directory=str(root/'review'))}
    parser = configparser.ConfigParser(interpolation=None)
    parser.read_dict(sections)
    cohort_ini = root/'cohort.ini'
    with cohort_ini.open('w') as stream:
        stream.write('# Generated mathematical CT-like crop; no licensed or clinical data.\n')
        parser.write(stream)
    names = dict(main=-7,branch=8,neighbor=9,vein=-10,bone=30,skin=40,
                 soft_a=20,soft_b=21,lung=22,unknown=99,background=0)
    inputs = (act_path,atn_path,par,log,catalog_path,audit_path,cohort_ini)
    return dict(cohort_ini=cohort_ini,input_hashes={p:_sha(p) for p in inputs},
                shape_kji=shape,spacing_ijk_mm=spacing,nodes_ijk=nodes,
                crop_origin_ijk=crop_origin,original_shape_kji=original_shape,
                masks={name:labels==code for name,code in names.items()},labels=labels,
                attenuation_cm_inverse=attenuation,attenuation_per_pixel=per_pixel,
                label_signals_cm_inverse=signals,audit_path=audit_path,catalog_path=catalog_path,
                source_paths=dict(act=act_path,atn=atn_path),sections=sections)
