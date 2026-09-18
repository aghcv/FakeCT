"""Explicit donor/recipient eligibility that preserves anatomical identities.

The permissive policy protects whole catalogued bone and skin labels, including
skin surfaces that belong to the coarse soft-tissue group. It does not reinterpret
or modify the atlas. Coarse ``unknown`` with a dictionary identity is distinct
from an original ID missing entirely from the dictionary.
"""
import re

import numpy as np

from fakect_tissues import _validated_labels, coarse_labels


MODES = ('allowlist', 'permissive_except_bone_skin', 'stiffness')


def _normalized(value):
    return re.sub(r'[^a-z0-9]+', '_', str(value or '').lower()).strip('_')


def _record_reasons(record, categories):
    """Conservative whole-label exclusions, using applied atlas fields only."""
    classification = record.get('classification', {})
    group = classification.get('tissue_name', categories.get(classification.get('tissue_id'), 'unknown'))
    structure = _normalized(classification.get('structure_type'))
    reasons = []
    if record.get('derived_label') == 'fakect.diagnostic-released/1' or group == 'released':
        reasons.append('diagnostic_released')
    if int(record['original_id']) == 0 or group == 'background':
        reasons.append('background')
    if group == 'bone' or categories.get(classification.get('tissue_id')) == 'bone' or structure == 'bone':
        reasons.append('bone')
    hierarchy = [_normalized(part) for part in str(classification.get('hierarchy_path', '')).split('/')]
    hierarchy_skin = any(re.sub(r'^\d+_', '', part) == 'skin' for part in hierarchy)
    names = (record.get('original_name'), classification.get('original_name'),
             classification.get('canonical_name'), classification.get('normalized_name'))
    explicit_skin = any(re.search(r'(?:^|_)skin\d*(?:_|$)', _normalized(name)) for name in names)
    if group == 'skin' or structure == 'skin' or hierarchy_skin or explicit_skin:
        reasons.append('skin')
    return reasons


def validate_reassignment_policy(catalog, policy, target_tissue=None, target_source_ids=()):
    """Validate policy and metadata-only target protection without reading voxels."""
    mode = policy.get('mode', 'allowlist')
    if mode not in MODES:
        raise ValueError('reassignment.mode must be allowlist, permissive_except_bone_skin, or stiffness')
    allowed = tuple(policy.get('allowed_tissues', ()))
    categories = {category['name']: category['id'] for category in catalog['categories']}
    if len(set(allowed)) != len(allowed):
        raise ValueError('reassignment.allowed_tissues must not repeat a tissue')
    if mode == 'allowlist':
        if any(name not in categories or name in ('unknown', 'released') for name in allowed):
            raise ValueError('Allowed tissues must be known catalog groups; unknown can never be allowlisted, nor diagnostic released markers')
        if target_tissue in allowed:
            raise ValueError('Allowed reassignment tissues must be disjoint from the target tissue')
    elif allowed:
        raise ValueError(f'{mode} requires empty reassignment.allowed_tissues')
    records = {int(record['original_id']): record for record in catalog['records']}
    for value in target_source_ids:
        if int(value) in records and 'diagnostic_released' in _record_reasons(records[int(value)], {v: k for k, v in categories.items()}):
            raise ValueError('Diagnostic released markers cannot be anatomical edit targets')
    if target_tissue == 'released':
        raise ValueError('Diagnostic released markers cannot be anatomical edit targets')
    if mode == 'stiffness':
        stiffness = policy.get('stiffness', {})
        tissues = stiffness.get('tissues', {})
        overrides = stiffness.get('labels', {})
        if 'default' not in stiffness or not {'bone', 'skin'} <= set(tissues):
            raise ValueError('Stiffness requires default, bone, and skin factors')
        if set(tissues) - (set(categories) | {'skin'}):
            raise ValueError('Stiffness refers to an unknown tissue group')
        for value in (stiffness['default'], *tissues.values(), *overrides.values()):
            if isinstance(value, bool) or not np.isfinite(value) or not 0 <= value <= 1:
                raise ValueError('Stiffness factors must be finite numbers between 0 and 1')
        for value in overrides:
            if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or int(value) not in records:
                raise ValueError(f'Stiffness label override must identify a catalogued signed original ID: {value}')
            if 'diagnostic_released' in _record_reasons(records[int(value)], {v: k for k, v in categories.items()}):
                raise ValueError('Diagnostic released markers cannot have a tissue stiffness override')
        if 'released' in tissues:
            raise ValueError('Diagnostic released markers cannot have a tissue stiffness override')
        for value in target_source_ids:
            value = int(value)
            if value not in records or 'background' in _record_reasons(records[value], {v: k for k, v in categories.items()}):
                raise ValueError(f'Stiffness forbids background/unmapped target ID {value}')
    if mode == 'permissive_except_bone_skin':
        groups = {value: key for key, value in categories.items()}
        for value in target_source_ids:
            value = int(value)
            reasons = _record_reasons(records[value], groups) if value in records else ['unmapped_original_id']
            if reasons:
                raise ValueError(f'Permissive reassignment forbids protected target ID {value}: {", ".join(reasons)}')
    metadata = {'mode': mode, 'allowed_tissues': list(allowed),
            'identity_source': 'Original IDs and applied classification fields in the preserved source label catalog; atlas classifications are not changed.',
            'catalog_sources': catalog.get('sources', {}),
            'eligibility_semantics': (
                'Catalogued non-target IDs except bone, skin, and background. Catalogued unknown tissues and other vessels can be reassigned; unmapped IDs remain protected. Whole identified skin labels are protected even when their coarse tissue is soft_tissue.'
                if mode == 'permissive_except_bone_skin' else
                'Only explicitly allowed coarse tissue groups are eligible; target candidates and unknown/unmapped labels are excluded.'),
            'skin_identity_rules': ['hierarchy segment Skin', 'structure_type skin', 'explicit canonical/original skin name'] if mode != 'allowlist' else [],
            'bone_identity_rules': ['coarse tissue bone', 'structure_type bone'] if mode != 'allowlist' else []}
    if mode == 'stiffness':
        metadata.update(stiffness=policy['stiffness'],
                        eligibility_semantics='Catalogued non-target IDs with stiffness below 1 are eligible. Background and unmapped original IDs remain protected. Bone and skin use their original anatomical identity, independent of the coarse tissue group. All supplied factors, including bone and skin, are editable.',
                        stiffness_semantics='Dimensionless editing resistance, not an elastic modulus or a biomechanical model. Per-ID override takes precedence over semantic bone/skin or coarse tissue factors, then the default. A factor of 1 is rigid; 0 imposes no additional resistance.',
                        stiffness_math='Dilation local donor budget = requested spatial distance * (1 - donor stiffness). Erosion release budget = requested spatial distance * (1 - target stiffness). Recipient seed initial cost = reassignment.max_distance_mm * recipient stiffness; physical edge costs are then added up to the global max_distance_mm cap.',
                        recipient_ownership='Minimum stiffness penalty plus physical six-edge path distance; ties use lowest signed original ID then earliest input-state seed coordinate.')
    return metadata


def stiffness_coefficients(catalog, policy):
    """Map catalogued original IDs to effective factors and their decision basis."""
    groups = {category['id']: category['name'] for category in catalog['categories']}
    spec = policy['stiffness']
    result = {}
    for record in catalog['records']:
        key = int(record['original_id'])
        reasons = _record_reasons(record, groups)
        classification = record['classification']
        semantic_group = ('bone' if 'bone' in reasons else 'skin' if 'skin' in reasons
                          else classification.get('tissue_name', 'unknown'))
        if 'diagnostic_released' in reasons:
            value, basis = 1., 'diagnostic_unassigned_protection'
        elif 'background' in reasons:
            value, basis = 1., 'categorical_background_protection'
        elif key in spec.get('labels', {}):
            value, basis = spec['labels'][key], 'original_id_override'
        elif semantic_group in spec['tissues']:
            value, basis = spec['tissues'][semantic_group], 'anatomical_group'
        else:
            value, basis = spec['default'], 'default'
        result[key] = {'stiffness': float(value), 'stiffness_group': semantic_group,
                       'stiffness_basis': basis}
    return result


def stiffness_field(labels, catalog, policy):
    """Native coefficient field; uncatalogued identities are categorically rigid."""
    labels = _validated_labels(labels)
    coefficients = stiffness_coefficients(catalog, policy)
    values, inverse = np.unique(labels, return_inverse=True)
    lookup = np.asarray([coefficients.get(int(value), {'stiffness': 1.})['stiffness'] for value in values])
    return lookup[inverse].reshape(labels.shape)


def reassignment_masks(labels, candidates, catalog, policy):
    """Return ``eligible, protected, metadata`` for this pass's input state.

    Candidates are excluded from erosion recipient seeds as well as donors.
    ``protected`` excludes current candidates, so it can also enforce invariance
    of originally protected non-target voxels across a complete edit recipe.
    """
    labels = _validated_labels(labels)
    candidates = np.asarray(candidates)
    if candidates.shape != labels.shape or candidates.dtype.kind != 'b':
        raise ValueError('Reassignment candidates must be a Boolean mask matching the labels')
    target_ids = tuple(map(int, np.unique(labels[candidates])))
    metadata = validate_reassignment_policy(catalog, policy, target_source_ids=target_ids)
    records = {int(record['original_id']): record for record in catalog['records']}
    categories = {category['id']: category['name'] for category in catalog['categories']}
    present_ids, present_counts = np.unique(labels, return_counts=True)
    present = {int(value): int(count) for value, count in zip(present_ids, present_counts)}
    reasons = {}
    if metadata['mode'] == 'allowlist':
        codes = [category['id'] for category in catalog['categories']
                 if category['name'] in metadata['allowed_tissues']]
        eligible = np.isin(coarse_labels(labels, catalog), codes) & ~candidates
        for value in present:
            if value not in records:
                reasons[value] = ['unmapped_original_id']
            elif records[value]['classification']['tissue_name'] not in metadata['allowed_tissues']:
                reasons[value] = ['outside_tissue_allowlist']
    elif metadata['mode'] == 'permissive_except_bone_skin':
        for value in present:
            reasons[value] = (_record_reasons(records[value], categories) if value in records
                              else ['unmapped_original_id'])
        eligible_ids = [value for value, exclusions in reasons.items() if not exclusions]
        eligible = np.isin(labels, eligible_ids) & ~candidates
    else:
        coefficients = stiffness_coefficients(catalog, policy)
        eligible_ids = [value for value in present if value in coefficients and coefficients[value]['stiffness'] < 1]
        eligible = np.isin(labels, eligible_ids) & ~candidates
        for value in present:
            if value not in records:
                reasons[value] = ['unmapped_original_id']
            elif coefficients[value]['stiffness'] >= 1:
                reasons[value] = _record_reasons(records[value], categories) + ['stiffness_one']
            else:
                reasons[value] = []
    protected = ~eligible & ~candidates

    def rows(mask, with_reasons=False):
        values, counts = np.unique(labels[mask], return_counts=True)
        result = []
        for value, count in zip(values, counts):
            key = int(value)
            record = records.get(key, {})
            row = {'original_id': key,
                   'original_name': record.get('original_name', f'Unmapped original ID {key}'),
                   'tissue_name': record.get('classification', {}).get('tissue_name', 'unknown'),
                   'count': int(count)}
            if with_reasons:
                row['reasons'] = reasons.get(key, ['outside_tissue_allowlist'])
            if metadata['mode'] == 'stiffness':
                row.update(coefficients.get(key, {'stiffness': 1., 'stiffness_group': 'unmapped',
                                                 'stiffness_basis': 'categorical_unmapped_protection'}))
            result.append(row)
        return result

    metadata.update(eligible_voxels=int(eligible.sum()), protected_voxels=int(protected.sum()),
                    candidate_voxels=int(candidates.sum()), eligible_input_labels=rows(eligible),
                    protected_input_labels=rows(protected, True))
    if metadata['mode'] == 'stiffness':
        metadata['effective_input_labels'] = rows(np.ones_like(candidates))
    return eligible, protected, metadata
