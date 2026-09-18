"""Derived diagnostic labels for erosion with deferred tissue assignment.

The reserved marker belongs only to generated outputs, never the XCAT atlas.
It carries no material identity or recovered attenuation and is kept out of
subsequent donor/recipient propagation until explicitly resolved in future work.
"""
from copy import deepcopy

import numpy as np
from scipy import ndimage

RELEASED_LABEL_ID = -(2**31)
RELEASED_TISSUE_ID = 254
RELEASED_NAME = 'released'
RELEASED_COLOR = '#ff2ea6'
EXTENSION = 'fakect.diagnostic-released/1'
SCALAR_STATUS = ('Released voxels retain their input-state attenuation only as an unassigned placeholder. '
                 'No surrounding tissue or attenuation has been inferred; use attenuation_unassigned_mask to identify them.')


def uses_diagnostic_release(config):
    return any(edit.get('assign_surrounding_tissue', True) is False
               for edit in [config.get('edit', {}), *config.get('edits', {}).values()])


def released_catalog(catalog):
    """Copy and extend a catalog, rejecting collisions with source identities."""
    from fakect_tissues import _validated_catalog_mapping
    _validated_catalog_mapping(catalog)
    categories = [row for row in catalog['categories']
                  if row['id'] == RELEASED_TISSUE_ID or row['name'] == RELEASED_NAME]
    records = [row for row in catalog['records'] if row['original_id'] == RELEASED_LABEL_ID]
    if categories or records:
        if (catalog.get('diagnostic_extension') != EXTENSION or len(categories) != 1 or len(records) != 1
                or categories[0]['id'] != RELEASED_TISSUE_ID or categories[0]['name'] != RELEASED_NAME
                or records[0].get('derived_label') != EXTENSION
                or records[0]['classification']['tissue_id'] != RELEASED_TISSUE_ID):
            raise ValueError('Diagnostic released label/category collides with an existing catalog identity')
        return deepcopy(catalog)
    result = deepcopy(catalog)
    result['categories'].append({'id': RELEASED_TISSUE_ID, 'name': RELEASED_NAME,
                                 'description': 'Eroded target awaiting material assignment; diagnostic only'})
    result['records'].append({'original_id': RELEASED_LABEL_ID, 'original_name': RELEASED_NAME,
                              'derived_label': EXTENSION,
                              'classification': {'tissue_id': RELEASED_TISSUE_ID, 'tissue_name': RELEASED_NAME,
                                                 'structure_type': 'diagnostic',
                                                 'hierarchy_path': 'FakeCT diagnostic/Released',
                                                 'canonical_name': RELEASED_NAME,
                                                 'confidence': 'not_applicable', 'rule': 'diagnostic_erosion'}})
    result['diagnostic_extension'] = EXTENSION
    result['diagnostic_extension_semantics'] = ('Source categories and records are preserved. The released marker '
        'is a generated output label, not an anatomical XCAT ID or a validated tissue classification.')
    _validated_catalog_mapping(result)
    return result


def release_assignment_metadata(labels_before, release_mask, catalog, policy):
    """Observe the six-neighbor shell; never reinterpret its anatomical labels."""
    from fakect_reassignment import stiffness_coefficients
    labels, mask = np.asarray(labels_before), np.asarray(release_mask, dtype=bool)
    if labels.shape != mask.shape or mask.ndim != 3:
        raise ValueError('Released diagnostic mask must match the native 3D label crop')
    shell = ndimage.binary_dilation(mask, structure=ndimage.generate_binary_structure(3, 1)) & ~mask
    ids, counts = np.unique(labels[shell], return_counts=True)
    records = {row['original_id']: row for row in catalog['records']}
    coefficients = stiffness_coefficients(catalog, policy) if policy.get('mode') == 'stiffness' else {}
    rows = []
    for value, count in zip(ids, counts):
        value = int(value)
        record = records.get(value, {})
        row = {'original_id': value, 'original_name': record.get('original_name', 'unmapped'),
               'tissue_name': record.get('classification', {}).get('tissue_name', 'unknown'),
               'count': int(count)}
        row.update(coefficients.get(value, {}))
        rows.append(row)
    return {'mode': 'diagnostic_label', 'label_name': RELEASED_NAME, 'label_id': RELEASED_LABEL_ID,
            'tissue_id': RELEASED_TISSUE_ID, 'diagnostic_only': True,
            'newly_released_voxels': int(mask.sum()), 'current_released_voxels': int(mask.sum()),
            'scalar_status': SCALAR_STATUS, 'surrounding_labels': rows,
            'surrounding_semantics': 'Unique input-state voxels sharing a face with a newly released voxel, excluding the released set; includes remaining target. Catalog classifications are observations, not verified material identities.',
            'subsequent_pass_semantics': 'Released markers remain visible and protected from donor/recipient propagation; no material is inferred from their retained attenuation.'}
