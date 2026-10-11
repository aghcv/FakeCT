#!/usr/bin/env python3
"""Summarize native cohort outcomes for manual review without approving a freeze.

Reads project INIs and preflight JSON only. Does not execute recipes, inspect
voxel payloads, alter settings, publish pairs, or establish clinical realism.
"""
import argparse
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(ROOT / 'src'))
from fakect_study_config import load_study_config

SCHEMA = 'fakect.cohort-outcome-review/1'


def _issue(level, reason, **details):
    return {'level': level, 'reason': reason, **details}


def _state(issues):
    return ('reject' if any(row['level'] == 'reject' for row in issues) else
            'needs_review' if issues else 'no_flags')


def summarize_step(step, native=None):
    """Retain compact measured outcomes, distinguishing resistance from failure."""
    native = native or {}
    guard = step.get('erosion_safeguard') or native.get('erosion_safeguard') or {}
    counts = native.get('counts', {})
    operation = step.get('operation', native.get('requested_operation', native.get('operation')))
    requested = step.get('requested_distance_mm', guard.get('requested_distance_mm', native.get('distance_mm', 0)))
    accepted = step.get('accepted_distance_mm', guard.get('accepted_distance_mm', requested))
    result = {key: step.get(key, native.get(key)) for key in ('step_name', 'iteration', 'status')}
    result.update(operation=operation, requested_distance_mm=requested, accepted_distance_mm=accepted)
    for key in ('added', 'removed', 'proposed_added', 'proposed_removed', 'blocked', 'unresolved'):
        result[key + '_voxels'] = step.get(key + '_voxels', counts.get(key, 0))
    for key in ('before', 'after', 'requested_removed_before_stiffness', 'release_suppressed_by_target_stiffness'):
        result[key + '_voxels'] = counts.get(key)
    for key in ('components_before', 'components_after', 'full_target_components_before',
                'full_target_components_after', 'full_target_before', 'full_target_after'):
        result[key] = native.get(key)
    for actual, proposed, key in (('removed_voxels', 'proposed_removed_voxels', 'release_assignment_fraction'),
                                  ('added_voxels', 'proposed_added_voxels', 'dilation_acceptance_fraction')):
        result[key] = result[actual] / result[proposed] if result[proposed] else None
    direction = native.get('direction') or {}
    result['direction'] = {key: direction[key] for key in ('direction', 'angular_width_deg',
        'roi_voxels', 'unreliable_roi_voxels', 'angular_supported_roi_voxels') if key in direction}
    result['erosion_safeguard'] = ({key: guard[key] for key in ('status', 'requested_distance_mm',
        'accepted_distance_mm', 'retained_volume_ratio', 'min_volume_ratio', 'preserve_connectivity',
        'baseline_target_voxels', 'connectivity_ok', 'reasons', 'component_details') if key in guard}
        | {'attempt_count': len(guard.get('attempts', []))}) if guard else None
    issues = []
    active = operation in ('dilation', 'erosion') and requested > 0
    changed = result['added_voxels'] if operation == 'dilation' else result['removed_voxels']
    if active and (not changed or result['status'] in ('empty_target', 'no_change')):
        issues.append(_issue('reject', 'active_edit_has_no_effect'))
    if result['unresolved_voxels']:
        issues.append(_issue('needs_review', 'unresolved_releases_preserve_target',
                             voxels=result['unresolved_voxels']))
    # Blocking is a normal consequence of protected labels and tissue resistance.
    # Report its counts/fraction without interpreting every blocked voxel as failure.
    if active and direction and direction.get('angular_supported_roi_voxels') == 0:
        issues.append(_issue('reject', 'no_reliable_directional_support'))
    if active and direction.get('unreliable_roi_voxels', 0):
        issues.append(_issue('needs_review', 'some_directional_support_is_unreliable'))
    before, after = result['full_target_components_before'], result['full_target_components_after']
    if active and before is not None and after is not None and before != after:
        issues.append(_issue('needs_review', 'full_target_component_count_changed', before=before, after=after,
            interpretation='Count changes require component correspondence review; totals alone cannot prove which branch split, merged or vanished.'))
    if guard:
        if guard.get('connectivity_ok') is False or guard.get('reasons'):
            issues.append(_issue('reject', 'accepted_result_violates_erosion_guard'))
        if active and (guard.get('status') == 'skipped' or accepted == 0):
            issues.append(_issue('reject', 'erosion_guard_skipped_edit'))
        elif active and accepted < requested:
            issues.append(_issue('needs_review', 'erosion_guard_reduced_budget', fraction=accepted / requested))
        if active and guard.get('baseline_target_voxels') == 0:
            issues.append(_issue('reject', 'empty_erosion_guard_reference'))
    result.update(issues=issues, review_state=_state(issues))
    return result


def summarize_sample(sample):
    result = {key: sample.get(key) for key in ('variant_id', 'status', 'operation', 'parameters',
        'edits', 'mask_sha256', 'image_sha256', 'baseline_equivalent', 'zero_change', 'duplicate_of',
        'foreground_voxels', 'changed_voxels', 'scalar_changed_voxels', 'error')}
    geometry = sample.get('geometry', {})
    result['voxel_geometry'] = {key: geometry.get(key) for key in ('crop_shape_kji', 'spacing_ijk_mm')}
    issues = []
    if sample.get('status') != 'ok':
        issues.append(_issue('reject', 'native_execution_failed', error=sample.get('error')))
    else:
        if sample.get('foreground_voxels') == 0:
            issues.append(_issue('reject', 'empty_target_mask'))
        if sample.get('operation') != 'none' and (sample.get('baseline_equivalent') or sample.get('zero_change')):
            issues.append(_issue('reject', 'edited_variant_preserves_baseline'))
        if sample.get('duplicate_of'):
            issues.append(_issue('needs_review', 'duplicate_variant_geometry', duplicate_of=sample['duplicate_of']))
        checks = sample.get('label_checks', {})
        if checks.get('unassigned_voxels', 0) or any(checks.get(key) is False for key in
                ('target_matches_selected_lineage', 'changes_within_edit_region', 'source_arrays_unchanged')):
            issues.append(_issue('reject', 'label_or_lineage_invariant_failed'))
    native = {(row['step_name'], row['iteration']): row for row in sample.get('morphology', {}).get('steps', [])}
    result['steps'] = [summarize_step(row, native.get((row['step_name'], row['iteration'])))
                       for row in sample.get('step_outcomes', [])]
    if sample.get('status') == 'ok' and sample.get('operation') != 'none' and not result['steps']:
        issues.append(_issue('reject', 'missing_native_step_outcomes'))
    for step in result['steps']:
        issues.extend({**issue, 'step_name': step['step_name'], 'iteration': step['iteration']} for issue in step['issues'])
    result.update(issues=issues, review_state=_state(issues))
    return result


def distance_responses(samples):
    """Compare one distance axis while holding every other swept setting fixed.

    Counts are accepted additions/removals summed over that edit's passes, not
    nominal distances or final net activity from a combination of edits.
    """
    groups = defaultdict(list)
    for sample in samples:
        if sample.get('status') != 'ok' or sample.get('operation') == 'none':
            continue
        parameters = {(name, key): value for name, values in (sample.get('parameters') or {}).items()
                      for key, value in values.items()}
        for (name, parameter), distance in parameters.items():
            if parameter != 'distance_mm':
                continue
            held = {f'{other}.{key}': value for (other, key), value in parameters.items()
                    if (other, key) != (name, parameter)}
            passes = [step for step in sample['steps'] if step['step_name'] == name]
            operations = {step['operation'] for step in passes if step['operation'] != 'none'}
            operation = next(iter(operations)) if len(operations) == 1 else sample.get('edits', {}).get(name, {}).get('operation')
            if operation not in ('dilation', 'erosion'):
                continue
            count_key = 'added_voxels' if operation == 'dilation' else 'removed_voxels'
            guard = (passes[-1].get('erosion_safeguard') or {}) if passes else {}
            groups[(name, operation, json.dumps(held, sort_keys=True))].append({
                'requested_distance_mm': distance, 'changed_voxels_for_edit': sum(step[count_key] for step in passes),
                'accepted_distance_mm_by_pass': [step['accepted_distance_mm'] for step in passes],
                'unresolved_voxels_for_edit': sum(step['unresolved_voxels'] for step in passes),
                'retained_volume_ratio': guard.get('retained_volume_ratio'),
                'variant_id': sample['variant_id'], 'mask_sha256': sample.get('mask_sha256')})
    responses = []
    for (name, operation, held), rows in sorted(groups.items()):
        rows.sort(key=lambda row: row['requested_distance_mm'])
        if len(rows) < 2:
            continue
        counts = [row['changed_voxels_for_edit'] for row in rows]
        ratios = [row['retained_volume_ratio'] for row in rows]
        monotonic = all(a <= b for a, b in zip(counts, counts[1:]))
        strict = all(a < b for a, b in zip(counts, counts[1:]))
        ratio_monotonic = (all(b <= a + 1e-12 for a, b in zip(ratios, ratios[1:]))
                           if all(value is not None for value in ratios) else None)
        unique_masks = len({row['mask_sha256'] for row in rows if row['mask_sha256']})
        issues = []
        if not monotonic:
            issues.append(_issue('needs_review', 'nonmonotonic_achieved_distance_response'))
        elif not strict:
            issues.append(_issue('needs_review', 'equal_change_counts_at_distinct_distances'))
        if ratio_monotonic is False:
            issues.append(_issue('needs_review', 'nonmonotonic_retained_volume_ratio'))
        if unique_masks < len(rows):
            issues.append(_issue('needs_review', 'distance_axis_contains_duplicate_target_masks'))
        responses.append({'edit_name': name, 'operation': operation, 'parameter': 'distance_mm',
            'held_sweep_parameters': json.loads(held), 'levels': rows, 'level_count': len(rows),
            'nondecreasing_change_voxels': monotonic, 'strictly_increasing_change_voxels': strict,
            'nonincreasing_retained_volume_ratio': ratio_monotonic, 'unique_final_target_masks': unique_masks,
            'issues': issues, 'review_state': _state(issues)})
    return responses


def summarize_preflight(document, case_id):
    if document.get('schema_version') != 'fakect.recipe-preflight/1':
        raise ValueError('Expected fakect.recipe-preflight/1')
    samples = [summarize_sample(row) for row in document.get('samples', [])]
    issues = []
    if not document.get('complete') or not document.get('execution_valid'):
        issues.append(_issue('reject', 'native_preflight_incomplete_or_failed'))
    if document.get('variant_count') != len(samples) or not samples:
        issues.append(_issue('reject', 'preflight_variant_count_mismatch_or_empty'))
    result = {key: document.get(key) for key in ('dataset_fingerprint', 'source_crop_sha256',
        'variant_count', 'successful_variants', 'failed_variants', 'unique_geometry_count',
        'duplicate_variant_count', 'baseline_equivalent_count', 'zero_change_count', 'input_config_sha256')}
    responses = distance_responses(samples)
    for response in responses:
        issues.extend({**issue, 'edit_name': response['edit_name'],
            'held_sweep_parameters': response['held_sweep_parameters']} for issue in response['issues'])
    result.update(case_id=str(case_id), samples=samples, distance_responses=responses, issues=issues)
    return result


def _resolve(path, root):
    path = Path(path).expanduser()
    return (root / path if not path.is_absolute() else path).resolve()


def _geometry_key(sample):
    geometry = sample.get('voxel_geometry', {})
    if any(not isinstance(geometry.get(key), (list, tuple)) or len(geometry[key]) != 3
           or any(not isinstance(value, (int, float)) or value <= 0 for value in geometry[key])
           for key in ('crop_shape_kji', 'spacing_ijk_mm')):
        return None
    return json.dumps(geometry, sort_keys=True)


def collect_project(project_path, repo_root=ROOT):
    root = Path(repo_root).resolve()
    project_path = _resolve(project_path, root)
    project = json.loads(project_path.read_bytes())
    if project.get('schema_version') != 'fakect.cohort-project/1':
        raise ValueError('Expected fakect.cohort-project/1')
    cases = []
    for entry in project['cases']:
        config_path = _resolve(entry['config'], root)
        row = {'case_id': str(entry['case_id']), 'config': str(config_path), 'samples': [], 'issues': []}
        try:
            config = load_study_config(config_path, repo_root=root)
            folder = config['train']['preflight_directory']
            path = folder / 'preflight.json'
            row['preflight_path'] = str(path)
            row['preflight_report'] = str(folder / 'preflight.html')
            row.update(summarize_preflight(json.loads(path.read_bytes()), entry['case_id']))
            row['config'] = str(config_path)
            row['preflight_path'] = str(path)
            row['preflight_report'] = str(folder / 'preflight.html')
            row['preflight_sha256'] = hashlib.sha256(path.read_bytes()).hexdigest()
            if (folder / 'INCOMPLETE').exists():
                row['issues'].append(_issue('reject', 'preflight_directory_marked_incomplete'))
            current = hashlib.sha256(config_path.read_bytes()).hexdigest()
            row['matches_current_input_bytes'] = current == row['input_config_sha256']
            if not row['matches_current_input_bytes']:
                row['issues'].append(_issue('needs_review', 'input_bytes_differ_from_preflight',
                    interpretation='Comments and review flags also alter bytes. This report does not recompute the native generation fingerprint.'))
        except (OSError, ValueError, KeyError, TypeError) as error:
            row['issues'].append(_issue('reject', 'preflight_unavailable_or_invalid', error=str(error)))
        cases.append(row)
    source_groups, mask_groups = defaultdict(list), defaultdict(list)
    for row in cases:
        hashes = row.get('source_crop_sha256') or {}
        samples = row['samples']
        if hashes.get('act') and hashes.get('atn') and samples and _geometry_key(samples[0]) is not None:
            source_groups[(hashes['act'], hashes['atn'], _geometry_key(samples[0]))].append(row['case_id'])
        for sample in samples:
            if sample.get('mask_sha256') and _geometry_key(sample) is not None:
                mask_groups[(sample['mask_sha256'], _geometry_key(sample))].append(
                    {'case_id': row['case_id'], 'variant_id': sample['variant_id']})
    duplicate_sources = [{'case_ids': values, 'act_sha256': key[0], 'atn_sha256': key[1]}
                         for key, values in source_groups.items() if len(set(values)) > 1]
    duplicate_masks = [{'mask_sha256': key[0], 'members': values} for key, values in mask_groups.items()
                       if len(values) > 1]
    for row in cases:
        if any(row['case_id'] in group['case_ids'] for group in duplicate_sources):
            row['issues'].append(_issue('needs_review', 'source_crop_matches_another_case'))
        if any(len({member['case_id'] for member in group['members']}) > 1 and
               any(member['case_id'] == row['case_id'] for member in group['members']) for group in duplicate_masks):
            row['issues'].append(_issue('needs_review', 'target_mask_matches_another_case'))
        all_issues = row['issues'] + [issue for sample in row['samples'] for issue in sample['issues']]
        row['review_state'] = _state(all_issues)
        row['flagged_variants'] = sum(sample['review_state'] != 'no_flags' for sample in row['samples'])
    issues = [issue for row in cases for issue in row['issues']] + [issue for row in cases
        for sample in row['samples'] for issue in sample['issues']]
    return {'schema_version': SCHEMA, 'generated_at_utc': datetime.now(timezone.utc).isoformat(),
        'project': str(project_path), 'name': project.get('name', ''), 'cases': cases,
        'review_state': _state(issues), 'case_count': len(cases),
        'variant_count': sum(len(row['samples']) for row in cases),
        'reject_issue_count': sum(row['level'] == 'reject' for row in issues),
        'review_issue_count': sum(row['level'] == 'needs_review' for row in issues),
        'duplicate_source_crops': duplicate_sources, 'duplicate_target_masks': duplicate_masks,
        'freeze_approved': False, 'voxel_payloads_read': False,
        'interpretation': 'Reject flags identify failed requested edits or invalid execution; needs_review flags require interpretation. No flags is not manual or clinical approval. Source/mask hash matches apply only to recorded crops with matching shape and spacing, not entire phantoms. Component totals do not prove component correspondence. Accepted distance is a budget, not measured vessel displacement. Artifact inventories and current native source fingerprints are not verified by this summary.'}


def render_html(document):
    escape = lambda value: html.escape(str(value))
    parts = ['<!doctype html><html lang="en"><meta charset="utf-8"><title>Cohort outcome review</title>',
        '<style>body{font:15px system-ui;margin:2rem;color:#182635}table{border-collapse:collapse;width:100%;margin:1rem 0}td,th{border:1px solid #ccd3da;padding:.45rem;text-align:left}th{background:#edf2f6}.reject{color:#aa2020}.needs_review{color:#865500}summary{cursor:pointer}code{white-space:normal}</style>',
        '<h1>' + escape(document['name']) + ': native outcome review</h1>',
        '<p>' + escape(document['interpretation']) + '</p>',
        f'<p>{document["case_count"]} phantoms; {document["variant_count"]} variants; '
        f'{document["reject_issue_count"]} reject flags; {document["review_issue_count"]} review flags. Freeze is not approved by this report.</p>']
    for case in document['cases']:
        parts.append('<h2>Case ' + escape(case['case_id']) + ' — ' + escape(case['review_state']) + '</h2>')
        parts.append('<p>Preflight: <code>' + escape(case.get('preflight_path', 'unavailable')) + '</code></p>')
        for issue in case['issues']:
            parts.append('<p class="' + issue['level'] + '">' + escape(issue['reason']) + ': ' + escape(issue.get('error', '')) + '</p>')
        if case.get('distance_responses'):
            parts.append('<details><summary>Achieved additions/removals along each distance axis</summary><table><tr><th>Edit</th><th>Held settings</th><th>Requested mm</th><th>Changed voxels for edit</th><th>Local retained ratios</th><th>Strictly increasing</th></tr>')
            for response in case['distance_responses']:
                levels = response['levels']
                values = [response['edit_name'], response['held_sweep_parameters'],
                    [level['requested_distance_mm'] for level in levels],
                    [level['changed_voxels_for_edit'] for level in levels],
                    [level['retained_volume_ratio'] for level in levels], response['strictly_increasing_change_voxels']]
                parts.append('<tr>'+''.join('<td>'+escape(value)+'</td>' for value in values)+'</tr>')
            parts.append('</table></details>')
        for sample in case['samples']:
            parts.append('<details' + (' open' if sample['review_state'] != 'no_flags' else '') + '><summary class="' + sample['review_state'] + '">' +
                escape(sample['variant_id']) + ' — ' + escape(sample['review_state']) + '</summary>')
            parts.append('<p>' + escape('; '.join(issue['reason'] + (' [' + issue['step_name'] + ']' if 'step_name' in issue else '') for issue in sample['issues']) or 'No automated flags.') + '</p>')
            if sample.get('error'):
                parts.append('<p>' + escape(sample['error']) + '</p>')
            parts.append('<table><tr><th>Pass</th><th>Operation</th><th>Requested → accepted mm</th><th>Added / removed</th><th>Unresolved / blocked</th><th>Full components before → after</th><th>Guard</th></tr>')
            for step in sample['steps']:
                guard = step['erosion_safeguard'] or {}
                values = [str(step['step_name']) + ' #' + str(step['iteration']), step['operation'],
                    str(step['requested_distance_mm']) + ' → ' + str(step['accepted_distance_mm']),
                    str(step['added_voxels']) + ' / ' + str(step['removed_voxels']),
                    str(step['unresolved_voxels']) + ' / ' + str(step['blocked_voxels']),
                    str(step['full_target_components_before']) + ' → ' + str(step['full_target_components_after']),
                    str(guard.get('status', 'off')) + '; retained=' + str(guard.get('retained_volume_ratio', 'n/a'))]
                parts.append('<tr>' + ''.join('<td>' + escape(value) + '</td>' for value in values) + '</tr>')
            parts.append('</table></details>')
    parts.append('<h2>Repeated source crops</h2><pre>' + escape(json.dumps(document['duplicate_source_crops'], indent=2)) + '</pre></html>')
    return ''.join(parts)


def write_review(document, output, overwrite=False):
    output = Path(output).expanduser().resolve()
    if output.suffix != '.json':
        raise ValueError('--output must end in .json; HTML is written beside it')
    paths = (output, output.with_suffix('.html'))
    if not overwrite and any(path.exists() for path in paths):
        raise FileExistsError('Review output exists; choose a new name or explicitly use --overwrite')
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2, allow_nan=False) + '\n')
    output.with_suffix('.html').write_text(render_html(document))
    return paths


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--project', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True, help='JSON report path; sibling HTML is also written')
    parser.add_argument('--overwrite', action='store_true', help='Replace this summary only; case artifacts remain read-only')
    args = parser.parse_args()
    try:
        document = collect_project(args.project)
        paths = write_review(document, args.output, args.overwrite)
    except (OSError, ValueError, KeyError, TypeError) as error:
        parser.exit(2, 'error: ' + str(error) + '\n')
    print(json.dumps({key: document[key] for key in ('case_count', 'variant_count', 'review_state',
        'reject_issue_count', 'review_issue_count', 'freeze_approved')} | {'reports': [str(path) for path in paths]}, indent=2))


if __name__ == '__main__':
    main()
