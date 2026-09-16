"""Reversible, provenance-preserving semantic groups for signed XCAT labels.

This adapter loads DPI's explicit YAML aliases/rules and reviewed atlas CSV. It
is not a port of DPI's additional built-in C++ rules. Categories describe anatomy,
not guaranteed voxel materials: a skin-bounded volume may contain adipose and a
skull interior may contain marrow. Never infer original IDs from surface indices,
and never discard original per-voxel labels after producing a coarse view.
"""
from __future__ import annotations

import csv
import hashlib
import json
from pathlib import Path
import re
from typing import Mapping

import numpy as np

SCHEMA_VERSION = 'fakect.tissue-catalog/1'
_ACTIVE_ACTIONS = {'', 'corrected', 'override', 'manual_override', 'approved', 'accepted', 'active'}
_FIELDS = ('normalized_name', 'canonical_name', 'system', 'subsystem',
           'anatomical_region', 'structure_type', 'laterality', 'piece_number',
           'temporal_phase', 'hierarchy_path', 'secondary_systems')


def normalize_name(value: str) -> str:
    """DPI normalization: case/space/hyphen folding, without stripping digits."""
    return re.sub(r'_+', '_', re.sub(r'[\s-]', '_', value.strip().lower())).strip('_')


def _source(path):
    path = Path(path)
    return {'path': str(path.resolve()), 'sha256': hashlib.sha256(path.read_bytes()).hexdigest()}


def _strip_comment(line):
    quote = None
    for i, char in enumerate(line):
        if char in "\"'":
            if quote == char:
                quote = None
            elif quote is None:
                quote = char
        elif char == '#' and quote is None:
            return line[:i]
    return line


def _load_hierarchy(path):
    """Read the scalar-only YAML subset supported by DPI; preserve regex slashes.

    Deliberately reject other YAML constructs instead of interpreting unsupported
    values. DPI strips quotes without YAML escape processing, which matters for
    regex word boundaries in double-quoted fields.
    """
    aliases, rules = {}, []
    section = current = None
    for lineno, raw in enumerate(Path(path).read_text().splitlines(), 1):
        line = _strip_comment(raw).rstrip()
        if not line.strip():
            continue
        if '\t' in line[:len(line)-len(line.lstrip())]:
            raise ValueError(f'{path}:{lineno}: tabs are not supported')
        indent, text = len(line)-len(line.lstrip()), line.strip()
        if indent == 0:
            if text not in ('aliases:', 'rules:'):
                raise ValueError(f'{path}:{lineno}: unsupported hierarchy section')
            section, current = text[:-1], None
            continue
        if section == 'aliases' and indent == 2 and text.endswith(':'):
            key = normalize_name(text[:-1])
            if key in aliases:
                raise ValueError(f'{path}:{lineno}: duplicate alias {key}')
            current = aliases[key] = {}
            continue
        if section == 'rules' and indent == 2 and text.startswith('- '):
            current = {}
            rules.append(current)
            text = text[2:]
        elif indent != 4 or current is None:
            raise ValueError(f'{path}:{lineno}: unsupported hierarchy syntax')
        if ':' not in text:
            raise ValueError(f'{path}:{lineno}: expected scalar field')
        key, value = (part.strip() for part in text.split(':', 1))
        if value[:1] in ('"', "'"):
            if len(value) < 2 or value[-1] != value[0]:
                raise ValueError(f'{path}:{lineno}: unclosed quoted scalar')
            value = value[1:-1]
        elif value[:1] in ('[', '{', '|', '>'):
            raise ValueError(f'{path}:{lineno}: only scalar fields are supported')
        if key in current:
            raise ValueError(f'{path}:{lineno}: duplicate field {key}')
        current[key] = value
    ids = set()
    for rule in rules:
        if not rule.get('id') or rule['id'] in ids or not rule.get('pattern'):
            raise ValueError('Hierarchy rules require unique IDs and nonempty patterns')
        ids.add(rule['id'])
        rule['priority'] = int(rule.get('priority', 0))
        rule['_regex'] = re.compile(rule['pattern'], re.IGNORECASE)
    return aliases, sorted(rules, key=lambda row: (-row['priority'], row['id']))


def _signature(row):
    return tuple(row.get(field, '') for field in _FIELDS)


class AtlasClassifier:
    """Classify names using explicit DPI files and a versioned grouping policy."""

    def __init__(self, atlas_path, hierarchy_path, policy_path):
        self.sources = {key: _source(path) for key, path in (
            ('atlas', atlas_path), ('hierarchy', hierarchy_path), ('policy', policy_path))}
        self.aliases, self.rules = _load_hierarchy(hierarchy_path)
        self.policy = json.loads(Path(policy_path).read_text())
        self.categories = self.policy['categories']
        self._categories = {row['name']: row for row in self.categories}
        category_ids = [row['id'] for row in self.categories]
        if (len(self._categories) != len(self.categories) or
                len(set(category_ids)) != len(category_ids) or
                any(type(value) is not int or not 0 <= value <= 255 for value in category_ids)):
            raise ValueError('Policy categories require unique names and uint8 integer IDs')
        if self._categories.get('unknown', {}).get('id') != self.policy['unknown_id']:
            raise ValueError('Policy unknown category must match unknown_id')
        self._policy_rules = sorted(self.policy['rules'], key=lambda row: (-row.get('priority', 0), row['id']))
        for rule in self._policy_rules:
            if rule['tissue'] not in self._categories:
                raise ValueError(f"Unknown policy tissue: {rule['tissue']}")
            for key, value in rule['match'].items():
                field = key[:-6] if key.endswith('_regex') else key
                if key == 'hierarchy_prefix':
                    continue
                if field not in _FIELDS:
                    raise ValueError(f'Unsupported policy field: {key}')
                if key.endswith('_regex'):
                    re.compile(value)
        self.warnings, self.atlas_rows, self._by_name = [], [], {}
        with Path(atlas_path).open(newline='') as stream:
            reader = csv.reader(stream)
            header = next(reader)
            if len(set(header)) != len(header) or not {'original_name', 'review_action'} <= set(header):
                raise ValueError('Atlas requires unique columns, original_name and review_action')
            for values in reader:
                if not values:
                    continue
                row = dict(zip(header, values))
                row['_line'] = reader.line_num
                row['_extra_columns'] = values[len(header):]
                row['_valid_width'] = len(values) == len(header)
                self.atlas_rows.append(row)
                if not row['_valid_width']:
                    self.warnings.append({'kind': 'atlas_row_width', 'line': reader.line_num,
                                          'expected': len(header), 'actual': len(values),
                                          'original_name': row.get('original_name', '')})
                if row.get('original_name'):
                    self._by_name.setdefault(row['original_name'], []).append(row)
        self.counts = {'atlas_rows': len(self.atlas_rows),
                       'atlas_valid_active_rows': sum(self._active(row) for row in self.atlas_rows),
                       'hierarchy_aliases': len(self.aliases), 'hierarchy_rules': len(self.rules)}

    @staticmethod
    def _active(row):
        return row['_valid_width'] and normalize_name(row.get('review_action', '')) in _ACTIVE_ACTIONS

    def _tissue(self, metadata):
        matches = []
        for rule in self._policy_rules:
            valid = True
            for key, expected in rule['match'].items():
                if key == 'hierarchy_prefix':
                    path = metadata.get('hierarchy_path', '')
                    valid = any(path == prefix or path.startswith(prefix + '/') for prefix in expected)
                elif key.endswith('_regex'):
                    valid = re.search(expected, metadata.get(key[:-6], ''), re.IGNORECASE) is not None
                else:
                    valid = metadata.get(key, '') in expected
                if not valid:
                    break
            if valid:
                matches.append(rule)
        if not matches:
            return self._categories['unknown'], []
        top = [row for row in matches if row.get('priority', 0) == matches[0].get('priority', 0)]
        names = {row['tissue'] for row in top}
        if len(names) != 1:
            return self._categories['unknown'], [{'kind': 'policy_conflict', 'rules': [row['id'] for row in top]}]
        return self._categories[names.pop()], [{'kind': 'tissue_policy', 'rule': row['id']} for row in top]

    def classify(self, name, source_file=None, block_index=None):
        """Return hierarchy, grouping and evidence without conflating ID namespaces.

        block_index is an actual DPI original_block_index, never an act ID or a
        local ordinal guessed while parsing one raw file. Scoped overrides match
        exact provided fields; example_* columns are provenance only.
        """
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Anatomical name must be a nonempty string')
        if block_index is not None and (isinstance(block_index, bool) or
                not isinstance(block_index, (int, np.integer)) or block_index < 0):
            raise ValueError('block_index must be a nonnegative integer DPI surface index')
        if source_file is not None:
            source_file = str(source_file)
        result = {field: '' for field in _FIELDS}
        result.update(original_name=name, hierarchy_path='99_Unclassified',
                      confidence='unclassified', status='unknown', evidence=[],
                      atlas_rows=[], matched_rules=[], classifier_scope='explicit_dpi_files_only',
                      material_type=None, material_status='unverified_anatomical_group')
        raw = normalize_name(name)
        working = raw
        def apply(row):
            for field in _FIELDS:
                if row.get(field):
                    result[field] = row[field]
        used_aliases = set()
        def alias(key):
            nonlocal working
            if key in self.aliases and key not in used_aliases:
                entry = self.aliases[key]
                apply(entry)
                working = normalize_name(entry.get('normalized_name') or entry.get('canonical_name') or working)
                used_aliases.add(key)
                result['evidence'].append({'kind': 'hierarchy_alias', 'key': key})
        alias(working)
        for prefix, phase in [('diastole_', 'diastole'), ('dias_', 'diastole'), ('systole_', 'systole'), ('sys_', 'systole')]:
            if working.startswith(prefix):
                result['temporal_phase'], working = phase, working[len(prefix):]
                break
        def laterality():
            nonlocal working
            for position in ('prefix', 'suffix'):
                for side, forms in [('left', ('left', 'lt', 'l')), ('right', ('right', 'rt', 'r'))]:
                    for form in forms:
                        if position == 'prefix' and working.startswith(form + '_'):
                            result['laterality'], working = side, working[len(form)+1:]
                            return
                        if position == 'suffix' and working.endswith('_' + form):
                            result['laterality'], working = side, working[:-len(form)-1]
                            return
        laterality()
        piece = re.search(r'(?<=[^0-9])([0-9]+)$', working)
        if piece:
            result['piece_number'], working = piece.group(1), working[:piece.start()].rstrip('_')
        if not result['laterality']:
            laterality()
        alias(working)
        result['normalized_name'] = result['normalized_name'] or working
        result['canonical_name'] = result['canonical_name'] or result['normalized_name']
        search = ' '.join((result['normalized_name'], result['canonical_name'], raw))
        matches = [rule for rule in self.rules if rule['_regex'].search(search)]
        hierarchy_conflict = False
        if matches:
            top = [rule for rule in matches if rule['priority'] == matches[0]['priority']]
            candidates = []
            for rule in top:
                candidate = {**result, **{key: value for key, value in rule.items() if key in _FIELDS and value}}
                candidates.append(_signature(candidate))
            hierarchy_conflict = len(set(candidates)) > 1
            result['matched_rules'] = [rule['id'] for rule in top]
            result['evidence'].append({'kind': 'hierarchy_rules', 'priority': top[0]['priority'],
                                       'rules': result['matched_rules'], 'conflict': hierarchy_conflict})
            if not hierarchy_conflict:
                apply(top[0])
                result['confidence'] = top[0].get('confidence', 'unspecified')
        rows = self._by_name.get(name, [])
        result['atlas_rows'] = rows
        active = [row for row in rows if self._active(row)]
        exact, scoped = [], []
        for row in active:
            index, source = row.get('original_block_index', ''), row.get('source_file', '')
            if index or source:
                scoped.append(row)
                try:
                    index_match = not index or block_index is not None and int(index) == int(block_index)
                except (ValueError, TypeError):
                    index_match = False
                if index_match and (not source or source_file is not None and source == str(source_file)):
                    exact.append(row)
            else:
                exact.append(row)
        if exact:
            # Most specific applicable override wins; conflicting equal scopes fail closed.
            rank = lambda row: bool(row.get('source_file')) + bool(row.get('original_block_index'))
            best = [row for row in exact if rank(row) == max(map(rank, exact))]
            if len({_signature(row) for row in best}) > 1:
                result['status'] = 'ambiguous_atlas'
                result['evidence'].append({'kind': 'atlas_conflict', 'lines': [row['_line'] for row in best]})
                return self._finish(result, force_unknown=True)
            # CSV overrides carry explicit column values, including blank fields.
            for field in _FIELDS:
                if field in best[0]:
                    result[field] = best[0][field]
            result['confidence'] = best[0].get('classification_confidence', '')
            result['evidence'].append({'kind': 'atlas_override', 'lines': [row['_line'] for row in best]})
            return self._finish(result)
        if scoped:
            reviewed = [self._finish({**row, 'evidence': [], 'status': 'unknown',
                                      'material_status': 'unverified_anatomical_group'}) for row in scoped]
            categories = [self._categories[row['tissue_name']] for row in reviewed]
            result['evidence'].append({'kind': 'unresolved_atlas_scope', 'lines': [row['_line'] for row in scoped],
                                       'source_file': source_file, 'block_index': block_index})
            if len({row['id'] for row in categories}) == 1 and categories[0]['name'] != 'unknown':
                # The tissue is invariant across alternatives, but fine anatomy is not identified.
                result.update({field: '' for field in _FIELDS})
                result.update(canonical_name=raw, normalized_name=raw, hierarchy_path='99_Unclassified',
                              status='tissue_consensus', tissue_id=categories[0]['id'], tissue_name=categories[0]['name'])
                result['evidence'].append({'kind': 'scoped_tissue_consensus', 'tissue_id': categories[0]['id']})
                return result
            result['status'] = 'scope_required'
            return self._finish(result, force_unknown=True)
        if hierarchy_conflict:
            result['status'] = 'ambiguous_hierarchy'
            return self._finish(result, force_unknown=True)
        if rows and not active:
            result['evidence'].append({'kind': 'atlas_rows_not_applied', 'lines': [row['_line'] for row in rows]})
        return self._finish(result)

    def _finish(self, result, force_unknown=False):
        for rule in self.policy.get('hierarchy_review_rules', []):
            structure_type = result.get('structure_type')
            review_type = (('allowed_structure_types' in rule and
                            structure_type not in rule['allowed_structure_types']) or
                           structure_type in rule.get('disallowed_structure_types', []))
            if re.search(rule['original_name_regex'], normalize_name(result['original_name'])) and review_type:
                result['status'] = rule.get('status', 'hierarchy_review_required')
                result['evidence'].append({'kind': 'hierarchy_review', 'rule': rule['id'], 'reason': rule['reason']})
                force_unknown = True
        for rule in self.policy.get('material_review_rules', []):
            if all((re.search(expected, result.get(key[:-6], ''), re.IGNORECASE) is not None
                    if key.endswith('_regex') else result.get(key, '') in expected)
                   for key, expected in rule['match'].items()):
                result['status'] = 'material_review_required'
                result['material_status'] = 'review_required'
                result['evidence'].append({'kind': 'material_review', 'rule': rule['id'], 'reason': rule['reason']})
                force_unknown = True
        category, evidence = (self._categories['unknown'], []) if force_unknown else self._tissue(result)
        result.update(tissue_id=category['id'], tissue_name=category['name'])
        result['evidence'].extend(evidence)
        if not force_unknown:
            result['status'] = 'classified' if category['name'] != 'unknown' else 'unknown'
        return result


def read_organ_table(path) -> dict[int, str]:
    """Read XCAT name = signed_id records, preserving repeated names across IDs."""
    table = {}
    for lineno, line in enumerate(Path(path).read_text().splitlines(), 1):
        line = line.split('#', 1)[0].strip()
        if not line:
            continue
        match = re.fullmatch(r'(.+?)\s*=\s*([+-]?\d+)\s*', line)
        if not match:
            raise ValueError(f'{path}:{lineno}: expected name = signed_integer')
        name, identifier = match.group(1).strip(), int(match.group(2))
        if not -(2**31) <= identifier < 2**31:
            raise ValueError(f'{path}:{lineno}: ID is outside int32 range')
        if identifier in table and table[identifier] != name:
            raise ValueError(f'{path}:{lineno}: conflicting names for ID {identifier}')
        table[identifier] = name
    return table


def build_catalog(classifier: AtlasClassifier, organ_table) -> dict:
    sources = dict(classifier.sources)
    if isinstance(organ_table, (str, Path)):
        sources['organ_table'] = _source(organ_table)
        organ_table = read_organ_table(organ_table)
    table = dict(organ_table)
    for identifier, name in table.items():
        if isinstance(identifier, bool) or not isinstance(identifier, (int, np.integer)) or not -(2**31) <= identifier < 2**31:
            raise ValueError('Organ dictionary keys must be signed int32 integers')
        if not isinstance(name, str) or not name.strip():
            raise ValueError('Organ dictionary values must be nonempty names')
    records = []
    backgrounds = classifier.policy.get('background_original_ids', [0])
    for identifier in backgrounds:
        if identifier in table and normalize_name(table[identifier]) not in ('background', 'exterior', 'air'):
            raise ValueError(f'Original ID {identifier} conflicts with the reserved background convention')
        table.setdefault(identifier, 'background')
    for identifier, name in sorted(table.items()):
        if identifier in backgrounds:
            category = classifier._categories['background']
            classification = {'tissue_id': category['id'], 'tissue_name': category['name'],
                              'status': 'reserved_background', 'hierarchy_path': '', 'structure_type': '',
                              'canonical_name': 'background', 'evidence': [{'kind': 'policy_background_convention'}],
                              'atlas_rows': [], 'matched_rules': [], 'material_type': None,
                              'material_status': 'background_convention'}
        else:
            classification = classifier.classify(name)
        records.append({'original_id': int(identifier), 'original_name': name, 'classification': classification})
    return {'schema_version': SCHEMA_VERSION, 'policy_version': classifier.policy['version'],
            'taxonomy_status': classifier.policy.get('status', 'proposed'),
            'categories': classifier.categories, 'records': records, 'sources': sources,
            'warnings': classifier.warnings,
            'counts': {**classifier.counts, 'original_ids': len(records),
                       'unknown_ids': sum(row['classification']['tissue_name'] == 'unknown' for row in records)},
            'semantics': classifier.policy.get('semantics', 'proposed_anatomical_groups_not_material_ground_truth'),
            'reversibility': 'Keep signed original per-voxel IDs; this many-to-one catalog cannot invert a coarse volume.'}


def _validated_labels(original):
    values = np.asarray(original)
    if values.dtype.kind not in 'iuf':
        raise ValueError('Original labels must be real numeric int32 IDs, not bool/object/complex')
    if not np.all(np.isfinite(values)) or np.any(values < -(2**31)) or np.any(values >= 2**31):
        raise ValueError('Original labels must be finite and within signed int32 range')
    if values.dtype.kind == 'f' and np.any(values != np.floor(values)):
        raise ValueError('Original labels must be integral; interpolation is not a label ID')
    return values.astype(np.int32, copy=False)


def coarse_labels(original, catalog: Mapping, strict=False) -> np.ndarray:
    """Return a new uint8 group volume; signed source labels remain untouched.

    strict rejects IDs absent from the catalog AND catalog entries marked unknown.
    Arrays can be supplied one slice/chunk at a time to bound working memory.
    """
    values = _validated_labels(original)
    mapping, unknown, _ = _validated_catalog_mapping(catalog)
    unique, inverse = np.unique(values, return_inverse=True)
    unresolved = [int(value) for value in unique if mapping.get(int(value), unknown) == unknown]
    if strict and unresolved:
        raise ValueError(f'Unresolved original IDs: {unresolved}')
    converted = np.asarray([mapping.get(int(value), unknown) for value in unique], dtype=np.uint8)
    return converted[inverse].reshape(values.shape)


def subtype_mask(original, catalog: Mapping, tissue_id: int, original_ids=None) -> np.ndarray:
    """Select fine anatomical IDs inside one coarse group from preserved labels.

    This requires the original voxel array. It never tries to reconstruct fine
    anatomy from uint8 coarse labels, which do not contain that information.
    """
    values = _validated_labels(original)
    mapping, _, categories = _validated_catalog_mapping(catalog)
    allowed = {original_id for original_id, group_id in mapping.items() if group_id == tissue_id}
    if tissue_id not in categories:
        raise ValueError(f'Unknown tissue group ID: {tissue_id}')
    selected = allowed if original_ids is None else set(original_ids)
    if not selected <= allowed:
        raise ValueError(f'Original IDs do not belong to tissue {tissue_id}: {sorted(selected-allowed)}')
    return np.isin(values, list(selected))


def _validated_catalog_mapping(catalog):
    """Reject edited catalogs that could truncate IDs or wrap uint8 codes."""
    categories, names = {}, set()
    for row in catalog['categories']:
        code, name = row['id'], row['name']
        if (isinstance(code, bool) or not isinstance(code, (int, np.integer)) or
                not 0 <= code <= 255 or code in categories or name in names):
            raise ValueError('Catalog categories require unique names and uint8 integer IDs')
        categories[code] = name
        names.add(name)
    unknowns = [code for code, name in categories.items() if name == 'unknown']
    if len(unknowns) != 1:
        raise ValueError('Catalog requires one unknown category')
    mapping = {}
    for row in catalog['records']:
        original_id = row['original_id']
        if (isinstance(original_id, bool) or not isinstance(original_id, (int, np.integer)) or
                not -(2**31) <= original_id < 2**31 or original_id in mapping):
            raise ValueError('Catalog original IDs must be unique signed int32 integers')
        code = row['classification']['tissue_id']
        if (isinstance(code, bool) or not isinstance(code, (int, np.integer)) or
                code not in categories):
            raise ValueError('Catalog record has an invalid tissue category ID')
        if row['classification'].get('tissue_name', categories[code]) != categories[code]:
            raise ValueError('Catalog tissue name disagrees with category ID')
        mapping[original_id] = code
    return mapping, unknowns[0], categories
