"""Portable, self-contained HTML reports for the bounded XCAT ROI workbench.

The report embeds existing preview assets. It does not read source volumes,
change labels, or equate connected components with anatomical vessels.
"""
import base64
import hashlib
import html
from html.parser import HTMLParser
import json
from pathlib import Path


_ASSETS = (
    ('roi-closeups.png', 'Native orthogonal close-ups',
     'Attenuation and tissue groups at the inspection crosshair. Orange marks the ROI; '
     'the selected target is the tissue candidate mask intersected with that ROI.'),
    ('roi-z-stack.png', 'Neighboring axial levels',
     'Follow the ROI across native k levels. Read native i, j and k indices from the '
     'axes before adjusting the centerline.'),
    ('roi-surfaces.png', 'Static 3D context',
     'Transparent surfaces provide spatial context. Display sampling can widen thin '
     'structures; use native 2D overlays for precise placement.'),
)


def _escape(value):
    return html.escape(str(value), quote=True)


def _number(value):
    if value is None:
        return 'Not recorded'
    try:
        return format(int(value), ',')
    except (ValueError, TypeError, OverflowError):
        return _escape(value)


def _text_value(value):
    if value is None:
        return 'Not recorded'
    if isinstance(value, (list, tuple)):
        return ', '.join(_text_value(v) for v in value)
    if isinstance(value, dict):
        return _escape(json.dumps(value, sort_keys=True, ensure_ascii=False))
    return _escape(value)


def _table(headers, rows, empty='No entries recorded.'):
    head = ''.join('<th scope="col">' + _escape(h) + '</th>' for h in headers)
    body = ''.join('<tr>' + ''.join('<td>' + cell + '</td>' for cell in row) + '</tr>' for row in rows)
    if not body:
        body = '<tr><td colspan="' + str(len(headers)) + '">' + _escape(empty) + '</td></tr>'
    return '<div class="table-scroll"><table><thead><tr>' + head + '</tr></thead><tbody>' + body + '</tbody></table></div>'


def _measurement(value):
    if value is None:
        return 'Not recorded'
    try:
        return format(float(value), ',.6g')
    except (ValueError, TypeError, OverflowError):
        return _escape(value)


def _embed_png(output, filename, heading, caption, embedded):
    path = output / filename
    if not path.is_file():
        return '<p class="subtle">' + _escape(heading) + ': preview not generated.</p>'
    raw = path.read_bytes()
    encoded = base64.b64encode(raw).decode('ascii')
    embedded[filename] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return ('<figure><h3>' + _escape(heading) + '</h3><img loading="lazy" src="data:image/png;base64,' +
            encoded + '" alt="' + _escape(heading) + '"><figcaption>' + _escape(caption) + '</figcaption></figure>')


def _embed_volume(output, filename, title, embedded):
    path = output / filename
    if not path.is_file():
        return '<p class="subtle">' + _escape(title) + ' was not generated.</p>'
    raw = path.read_bytes()
    document = raw.decode('utf-8')
    parser = _AssetReferences()
    parser.feed(document)
    if parser.references:
        raise ValueError('The 3D document must embed its resources; external or relative asset links were found')
    embedded[filename] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
    return ('<iframe title="' + _escape(title) + '" sandbox="allow-scripts" loading="lazy" srcdoc="' +
            _escape(document) + '"></iframe>')


class _AssetReferences(HTMLParser):
    """Reject linked resources in the otherwise trusted generated 3D document."""
    def __init__(self):
        super().__init__()
        self.references = []

    def handle_starttag(self, tag, attrs):
        attrs = dict(attrs)
        keys = ('src', 'poster', 'data') if tag != 'link' else ('href',)
        for key in keys:
            if key in attrs and attrs[key] and not attrs[key].startswith(('data:', 'blob:', '#')):
                self.references.append((tag, key, attrs[key]))
        if attrs.get('srcset'):
            self.references.append((tag, 'srcset', attrs['srcset']))


_STYLE = '''
:root {color-scheme:light;--ink:#172a3b;--muted:#526575;--line:#d9e2e8;--accent:#006b79;--paper:#fff;--wash:#eef3f6}
*{box-sizing:border-box}html{scroll-behavior:smooth}body{margin:0;background:var(--wash);color:var(--ink);font:16px/1.55 system-ui,-apple-system,BlinkMacSystemFont,"Segoe UI",sans-serif}
header,main,footer{max-width:1240px;margin:auto}header{padding:38px 30px 24px}h1{font-size:clamp(1.9rem,4vw,2.7rem);line-height:1.18;margin:10px 0 14px;overflow-wrap:anywhere}h2{font-size:1.5rem;margin:0 0 14px}h3{font-size:1.1rem;margin:20px 0 8px}p{margin:8px 0 16px}.eyebrow{color:var(--accent);font-size:.8rem;font-weight:750;letter-spacing:.14em;text-transform:uppercase}.subtle,figcaption{color:var(--muted)}
nav{display:flex;gap:8px;flex-wrap:wrap;margin-top:22px}nav a,button,.download{border:1px solid #b7cad3;background:white;color:var(--accent);border-radius:7px;padding:8px 12px;font:inherit;text-decoration:none;cursor:pointer}nav a:hover,button:hover,.download:hover{background:#e6f3f4}main{padding:0 20px}section{background:var(--paper);border:1px solid var(--line);border-radius:12px;padding:28px;margin:0 0 20px;scroll-margin-top:20px}.metrics{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px;margin:20px 0}.metric{padding:16px;border:1px solid var(--line);border-radius:8px;background:#f9fbfc}.metric strong{display:block;font-size:1.9rem;line-height:1.25}.metric span{color:var(--muted);font-size:.9rem}.rule{border-left:4px solid var(--accent);background:#eef8f9;padding:14px 18px}.warning{border-left:4px solid #c6841b;background:#fff8e8;padding:12px 18px;margin:12px 0}.status{display:inline-block;border-radius:20px;background:#e2f1eb;color:#23583d;padding:3px 11px;font-size:.85rem;font-weight:650}.table-scroll{max-width:100%;overflow:auto}table{width:100%;border-collapse:collapse;font-size:.92rem;margin:12px 0}th,td{text-align:left;vertical-align:top;padding:9px 12px;border-bottom:1px solid var(--line)}th{background:#edf3f6;white-space:nowrap}td{overflow-wrap:anywhere}figure{margin:22px 0 32px}figure img{display:block;width:100%;height:auto;border:1px solid var(--line);border-radius:6px}figcaption{font-size:.9rem;margin:8px 0}iframe{width:100%;height:880px;border:1px solid var(--line);border-radius:8px;background:#f4f6f8}details{margin-top:15px}summary{cursor:pointer;font-weight:650;color:var(--accent)}code,pre,textarea{font-family:ui-monospace,SFMono-Regular,Consolas,monospace;font-size:.86rem}code{overflow-wrap:anywhere}pre{white-space:pre-wrap;overflow-wrap:anywhere;padding:15px;border-radius:6px;background:#f2f6f8}textarea{width:100%;min-height:420px;resize:vertical;background:#f8fbfc;color:var(--ink);border:1px solid #b7cad3;border-radius:6px;padding:15px;tab-size:4;line-height:1.5}.actions{display:flex;align-items:center;gap:10px;flex-wrap:wrap;margin:12px 0}#input-status{color:var(--muted);font-size:.9rem}footer{padding:12px 30px 38px;font-size:.85rem;color:var(--muted)}.two-column{display:grid;grid-template-columns:1fr 1fr;gap:20px}.hash{word-break:break-all}.legend-key{display:inline-block;width:.85em;height:.85em;margin-right:.4em;border:1px solid #667;vertical-align:baseline}
@media(max-width:700px){header{padding:25px 20px}main{padding:0 10px}section{padding:18px}.two-column{grid-template-columns:1fr}iframe{height:740px}.metric strong{font-size:1.6rem}th,td{padding:8px}}
@media print{body{background:#fff}nav,.actions,iframe{display:none}section{break-inside:auto;border:0;padding:15px 0}figure{break-inside:avoid}textarea{height:480px}header,main{max-width:none}details>*{display:block}details{break-inside:avoid}}
'''

_SCRIPT = '''
(function(){
  const input = document.getElementById('captured-input');
  const status = document.getElementById('input-status');
  document.getElementById('download-input').addEventListener('click', function(){
    const url = URL.createObjectURL(new Blob([input.value], {type:'text/plain;charset=utf-8'}));
    const link = document.createElement('a');
    link.href = url; link.download = 'xcat-roi.ini';
    document.body.appendChild(link); link.click(); link.remove();
    setTimeout(function(){ URL.revokeObjectURL(url); }, 1000);
    status.textContent = 'Downloaded the displayed INI. Run the preview command to generate a new report.';
  });
  document.getElementById('copy-input').addEventListener('click', async function(){
    try {
      if (!navigator.clipboard) throw new Error('Clipboard unavailable');
      await navigator.clipboard.writeText(input.value);
      status.textContent = 'INI copied.';
    } catch(error) {
      input.focus(); input.select();
      status.textContent = 'INI selected. Press Ctrl+C (or Command+C) to copy.';
    }
  });
})();
'''


def _control_points(report):
    geometry = report.get('geometry', {})
    roi = report.get('config', {}).get('roi', {})
    kind = geometry.get('roi_kind', roi.get('shape', roi.get('kind', roi.get('type', 'sphere'))))
    nodes = geometry.get('nodes_ijk', geometry.get('roi_nodes_ijk', roi.get('nodes_ijk')))
    radii = geometry.get('radii_mm', geometry.get('roi_radii_mm', roi.get('radii_mm')))
    if nodes is None:
        center = geometry.get('roi_center_ijk', roi.get('center_ijk'))
        nodes = center if center and isinstance(center[0], (list, tuple)) else ([center] if center else [])
    if radii is None:
        radius = geometry.get('roi_radius_mm', roi.get('radius_mm'))
        radii = radius if isinstance(radius, (list, tuple)) else [radius] * len(nodes)
    if not isinstance(radii, (list, tuple)):
        radii = [radii] * len(nodes)
    rows = []
    for index, node in enumerate(nodes):
        values = list(node) if isinstance(node, (list, tuple)) else [node]
        values = (values + [None, None, None])[:3]
        radius = radii[index] if index < len(radii) else None
        rows.append([str(index + 1)] + [_text_value(v) for v in values] + [_text_value(radius)])
    return str(kind), rows


def _selection(report):
    selection = report.get('selection', report.get('selection_diagnostics', {}))
    candidate = selection.get('candidate_voxels', report.get('candidate_voxels', report.get('selected_voxels')))
    selected = selection.get('selected_voxels', report.get('selected_in_roi_voxels'))
    roi = selection.get('roi_voxels', report.get('roi_voxels'))
    count = selection.get('component_count_6')
    return selection, candidate, selected, roi, count


def _selected_rows(report, selection):
    names = {str(k): v for k, v in report.get('source_names', {}).items()}
    volume_records = report.get('volume', {}).get('original_selected_labels', [])
    for entry in volume_records:
        names.setdefault(str(entry.get('original_id')), entry.get('original_name', 'Unnamed'))
    counts = selection.get('selected_original_id_counts', {})
    ids = selection.get('selected_original_ids', [])
    if isinstance(counts, dict):
        pairs = list(counts.items())
    elif isinstance(counts, list) and counts and isinstance(counts[0], dict):
        pairs = [(r.get('original_id', r.get('id')), r.get('voxel_count', r.get('count'))) for r in counts]
    elif isinstance(counts, list):
        pairs = list(zip(ids, counts))
    else:
        pairs = []
    if not pairs and ids:
        pairs = [(label, None) for label in ids]
    if not pairs and not selection:
        # Legacy volume metadata counts the whole crop, not ROI intersection.
        return [], 'Per-ID counts inside the ROI were not recorded by this report version.'
    pairs.sort(key=lambda pair: int(pair[0]))
    return [[_escape(label), _escape(names.get(str(label), 'Unnamed / not present in catalog')), _number(count)]
            for label, count in pairs], 'Original signed IDs observed in the selected tissue–ROI intersection.'


def _morphology_section(output, report, embedded):
    """Render applied label changes without treating a scalar proxy as recovered CT."""
    edit = report['edit']
    config = report.get('config', {})
    edit_config = config.get('edit', {})
    reassignment = config.get('reassignment', {})
    counts = edit.get('counts', {})
    volumes = edit.get('volume_mm3', {})
    operation = edit.get('operation', edit_config.get('operation', 'Not recorded'))
    metrics = ''.join('<div class="metric"><strong>' + _number(counts.get(key)) + '</strong><span>' + label + '</span></div>'
                      for key, label in (('before', 'Target voxels before edit'), ('after', 'Target voxels after edit'),
                                         ('added', 'Applied additions'), ('removed', 'Applied removals')))
    change_rows = [[label, _number(counts.get(key))] for key, label in (
        ('proposed_added', 'Proposed additions'), ('proposed_removed', 'Proposed removals'),
        ('added', 'Applied additions'), ('removed', 'Applied removals'),
        ('blocked', 'Blocked changes'), ('unresolved', 'Unresolved reassignments'))]
    volume_rows = [[label, _measurement(volumes.get(key))] for key, label in (
        ('before', 'Target volume before edit'), ('after', 'Target volume after edit'),
        ('added', 'Applied added volume'), ('removed', 'Applied removed volume'))]
    transition_rows = [[_text_value(row.get('original_id')), _text_value(row.get('new_id')), _number(row.get('count'))]
                       for row in edit.get('transitions', [])]
    settings = []
    for key, label in (('operation', 'Operation'), ('distance_mm', 'Distance (mm)'),
                       ('profile', 'Profile'), ('profile_axis', 'Profile axis'),
                       ('shape_k', 'Profile shape k'), ('shape_window', 'Profile window')):
        value = edit.get(key, edit_config.get(key))
        if value is not None:
            settings.append([label, _text_value(value)])
    for key, label in (('allowed_tissues', 'Allowed reassignment tissues'),
                       ('max_distance_mm', 'Maximum reassignment distance (mm)'),
                       ('unresolved', 'Unresolved reassignment policy')):
        if key in reassignment:
            settings.append([label, _text_value(reassignment[key])])
    components = _table(['Mask', 'Components'], [
        ['Target before edit', _text_value(edit.get('components_before'))],
        ['Target after edit', _text_value(edit.get('components_after'))]])
    warnings = ''.join('<p class="warning">' + _escape(warning) + '</p>' for warning in edit.get('warnings', []))
    figures = _embed_png(output, 'edit-comparison.png', 'Before, after and difference',
                         'Compare the original target and edited result at the same native coordinates. '
                         'Read the difference legend for applied additions and removals; source images remain preserved.', embedded)
    figures += _embed_png(output, 'edit-profile.png', 'Achieved cross-sections and edit profile',
                          'Measured cross-sections describe this edited crop. A requested distance or profile is an '
                          'input to the trial; assess the achieved change shown here before using it for a cohort.', embedded)
    after = ''
    if (output / 'after/roi-volume.html').is_file() or (output / 'after/roi-surfaces.png').is_file():
        after = '<h3>After-edit 3D context</h3><p>The target and tissue context below use the edited labels. '
        after += 'Display sampling may widen thin structures; native masks determine the measurements.</p>'
        if (output / 'after/roi-volume.html').is_file():
            after += _embed_volume(output, 'after/roi-volume.html', 'After-edit interactive 3D ROI and tissue volume', embedded)
        if (output / 'after/roi-surfaces.png').is_file():
            after += _embed_png(output, 'after/roi-surfaces.png', 'After-edit static 3D context',
                                'Transparent surfaces of the edited labels, with the same ROI planning overlay.', embedded)
    return ('<section id="edit"><h2>Morphology trial: ' + _escape(operation) + '</h2>'
            '<p class="rule">Target counts and volumes below refer to the target <strong>inside the ROI</strong>, '
            'before and after this trial. Applied additions and removals describe actual label changes.</p>'
            '<div class="metrics">' + metrics + '</div>' + warnings +
            '<p><strong>Strength:</strong> ' + _text_value(edit.get('strength_semantics')) + '</p>' +
            '<p><strong>Scalar image status:</strong> ' + _text_value(edit.get('scalar_status')) + '</p>'
            '<p>Any edited scalar image is a provisional attenuation proxy. It is <strong>not AI background recovery '
            'or a reconstructed CT image</strong>. Original source arrays are preserved alongside the separate '
            'edited result in <code>edit.npz</code>; source volumes remain at their recorded paths.</p>' + figures + after +
            '<div class="two-column"><div><h3>Change accounting</h3>' +
            _table(['Change', 'Voxels'], change_rows) + '</div><div><h3>Physical volumes</h3>' +
            _table(['Quantity inside ROI', 'Volume (mm³)'], volume_rows) + '</div></div>' +
            '<h3>Applied original-label transitions</h3><p>Each row records how many voxels changed from one '
            'original signed label to another. The before-edit anatomical identities remain available in the '
            'original selection table below and in the preserved catalog.</p>' +
            _table(['Original signed ID', 'New signed ID', 'Changed voxels'], transition_rows,
                   'No original-label transitions were recorded.') +
            '<details><summary>Trial settings and component counts</summary>' +
            _table(['Setting', 'Value'], settings) + components +
            '<p>Engine: <code>' + _text_value(edit.get('engine')) + '</code>. Component counts are spatial '
            'diagnostics, not anatomical vessel counts.</p></details></section>')


def _recipe_section(output, report, embedded):
    """Account for ordered passes separately from original-to-final changes."""
    recipe = report['recipe']
    config = report.get('config', {})
    figures = recipe.get('figures', {})
    counts = recipe.get('counts', {})
    activity = recipe.get('activity_counts', {})
    policy = recipe.get('reassignment_policy', {})
    policy_content = '<h3>Tissue resistance and reassignment</h3><p>Mode: <code>' + _text_value(
        policy.get('mode', config.get('reassignment', {}).get('mode', 'allowlist'))) + '</code>. ' + _text_value(
        policy.get('eligibility_semantics', '')) + '</p>'
    if policy.get('mode') == 'stiffness':
        spec = policy.get('stiffness', {})
        factors = [['Default', _measurement(spec.get('default'))]]
        factors += [[_text_value(name), _measurement(value)] for name, value in spec.get('tissues', {}).items()]
        factors += [['Original ID ' + _text_value(name), _measurement(value)]
                    for name, value in spec.get('labels', {}).items()]
        effective = [[_text_value(row.get('original_id')), _text_value(row.get('original_name')),
                      _text_value(row.get('tissue_name')), _text_value(row.get('stiffness_group')),
                      _measurement(row.get('stiffness')), _text_value(row.get('stiffness_basis')),
                      _number(row.get('count'))] for row in policy.get('effective_input_labels', [])]
        policy_content += ('<p>' + _text_value(policy.get('stiffness_semantics')) + '</p>' +
                           _table(['INI tissue or override', 'Factor (0–1)'], factors) +
                           '<p><strong>How factors affect edits:</strong> ' + _text_value(policy.get('stiffness_math')) + '</p>' +
                           '<p>For example, at skin = 0.95, only 5% of the requested distance remains. With a 2 mm request '
                           'and 1 mm voxels this is below one voxel step. Larger requests or smaller voxels '
                           'can change skin; a factor of 1 makes a label rigid. Skin identity conservatively '
                           'covers the whole surface-associated original label, not an isolated dermal layer.</p>' +
                           '<details><summary>Effective factors for original labels present in the crop</summary>' +
                           _table(['Original ID', 'Anatomical name', 'Coarse tissue', 'Resistance group',
                                   'Factor', 'Factor source', 'Original voxels'], effective) + '</details>')
    steps = recipe.get('steps', [])
    roi_rows = []
    for row in figures.get('rois', []):
        coverage = recipe.get('roi_coverage', {}).get(row.get('name'), {})
        roi_rows.append([_text_value(row.get('name')), _text_value(row.get('shape')),
                         _text_value(row.get('nodes_ijk')), _text_value(row.get('radii_mm')),
                         _number(row.get('effective_voxels')), _number(row.get('target_voxels')),
                         _number(coverage.get('clipped_by_outer_roi_voxels'))])
    step_rows = []
    details = []
    for position, step in enumerate(steps, 1):
        summary = step.get('summary', step)
        current_counts = summary.get('counts', {})
        name = step.get('name', step.get('step_name'))
        roi = step.get('roi', step.get('roi_name'))
        operation = summary.get('requested_operation', summary.get('operation'))
        step_rows.append([_number(step.get('index', position)), _text_value(name), _text_value(roi),
                          _number(step.get('iteration')), _text_value(operation),
                          _number(current_counts.get('before')) + ' → ' + _number(current_counts.get('after')),
                          _number(current_counts.get('added')), _number(current_counts.get('removed')),
                          _number(current_counts.get('blocked')), _number(current_counts.get('unresolved'))])
        step_figures = step.get('figures', {})
        content = []
        for key, title, caption in (
                ('comparison', 'Input state, output state and changes',
                 'The input state includes preceding passes. The difference overlay uses the input-state '
                 'attenuation proxy. These views are focused within this named ROI; coordinates may differ between passes.'),
                ('profile', 'Requested profile and achieved axial areas',
                 'Areas belong to this effective named ROI for this pass; they are not vessel-normal lumen areas.')):
            if step_figures.get(key):
                content.append(_embed_png(output, step_figures[key], title, caption, embedded))
        metadata_rows = [[label, _text_value(value)] for label, value in (
            ('Status', step.get('status', summary.get('status'))),
            ('Distance per pass (mm)', summary.get('distance_mm')),
            ('Profile', summary.get('profile')), ('Profile axis', summary.get('profile_axis')),
            ('Requested release before target resistance (voxels)', current_counts.get('requested_removed_before_stiffness')),
            ('Release suppressed by target resistance (voxels)', current_counts.get('release_suppressed_by_target_stiffness')),
            ('Components inside this named ROI: before', summary.get('components_before')),
            ('Components inside this named ROI: after', summary.get('components_after')),
            ('Components of full-crop target: before', summary.get('full_target_components_before')),
            ('Components of full-crop target: after', summary.get('full_target_components_after')),
            ('Display focus (i, j, k)', step_figures.get('focus_ijk')),
            ('Input label SHA256', step.get('labels_before_sha256')),
            ('Output label SHA256', step.get('labels_after_sha256')),
            ('Pass arrays', step.get('artifact', step.get('array_artifact')))) if value is not None]
        warnings = ''.join('<p class="warning">'+_escape(w)+' </p>' for w in summary.get('warnings', []))
        blocked_rows = [[_text_value(row.get('original_id')), _text_value(row.get('original_name')),
                         _text_value(row.get('tissue_name')), _number(row.get('count'))]
                        for row in summary.get('blocked_input_labels', [])]
        blocked_labels = ('<h4>Blocked proposals by input-state label</h4>' +
                          _table(['Original label ID', 'Anatomical name', 'Tissue group', 'Blocked voxels'], blocked_rows) +
                          '<p>These are the labels encountered by blocked growth proposals. Tissue eligibility, '
                          'protected barriers and accepted-path budgets determine whether growth can reach them.</p>'
                          if blocked_rows else '')
        details.append('<details><summary>Pass ' + _number(step.get('index', position)) + ': ' +
                       _text_value(name) + ' · ROI ' + _text_value(roi) + ' · iteration ' +
                       _number(step.get('iteration')) + '</summary>' +
                       _table(['Pass setting', 'Value'], metadata_rows) + warnings + blocked_labels + ''.join(content) + '</details>')
    visualizations = []
    for key, heading, caption in (
            ('overview', 'Named regions at the study crosshair',
             'Colored overlays show each named ROI intersected with the main ROI. Cyan outlines the original '
             'target throughout the crop. Multiple regions can overlap.'),
            ('closeups', 'Per-region close-ups',
             'Each row uses a native target voxel near that region’s centroid. Read the i, j and k '
             'coordinates in each title before moving the corresponding named ROI.'),
            ('surfaces', 'Named regions in 3D',
             'Transparent colored surfaces show effective named regions around the original full-crop target. '
             'Display sampling can widen structures; native masks determine measurements.')):
        if figures.get(key):
            visualizations.append(_embed_png(output, figures[key], heading, caption, embedded))
    final_figures = recipe.get('final_figures', {})
    final = ''
    for key, heading, caption in (
            ('comparison', 'Original anatomy versus final recipe result',
             'Net additions and removals compare the original source with the final result. A later pass can '
             'reverse an earlier label change; this figure does not sum intermediate changes. Net changes take '
             'visual precedence over earlier blocked/unresolved proposals at the same voxel.'),
            ('profile', 'Final axial geometry and recipe activity',
             'Original and final target areas are measured within the main ROI. Blocked and unresolved '
             'masks are unions across passes; the requested profile is the maximum per-pass request, not a sum.')):
        if final_figures.get(key):
            final += _embed_png(output, final_figures[key], heading, caption, embedded)
    if (output/'after/roi-volume.html').is_file():
        final += '<h3>Final interactive 3D context</h3>' + _embed_volume(
            output, 'after/roi-volume.html', 'Final recipe target and tissue volume', embedded)
    if (output/'after/roi-surfaces.png').is_file():
        final += _embed_png(output, 'after/roi-surfaces.png', 'Final static 3D context',
                            'Final edited labels and tissue context within the original main ROI.', embedded)
    net_rows = [[label, _number(counts.get(key))] for key, label in (
        ('before', 'Original target inside main ROI'), ('after', 'Final target inside main ROI'),
        ('added', 'Net target additions'), ('removed', 'Net target removals'),
        ('changed', 'Original-to-final label differences'), ('ever_changed', 'Unique voxels changed in any pass'),
        ('scalar_changed', 'Original-to-final attenuation proxy differences'))]
    net_rows += [[label, _number(activity.get(key))] for key, label in (
        ('added', 'Sum of additions across passes'), ('removed', 'Sum of removals across passes'),
        ('changed', 'Sum of label changes across passes'))]
    component_rows = [[label, _number(recipe.get(key))] for key, label in (
        ('components_before', 'Original target inside main ROI'),
        ('components_after', 'Final target inside main ROI'),
        ('full_target_components_before', 'Original full-crop target'),
        ('full_target_components_after', 'Final full-crop target')) if key in recipe]
    overlaps = [[_text_value(row.get('roi_a')), _text_value(row.get('roi_b')),
                 _number(row.get('effective_overlap_voxels'))] for row in recipe.get('roi_overlaps', [])]
    transition_rows = [[_text_value(row.get('original_id')), _text_value(row.get('new_id')), _number(row.get('count'))]
                       for row in recipe.get('transitions', [])]
    warnings = ''.join('<p class="warning">'+_escape(w)+'</p>' for w in recipe.get('warnings', []))
    return ('<section id="recipe"><h2>Named-region edit recipe</h2>'
            '<p class="rule">Each pass consumes the previous pass’s labels and attenuation proxy. '
            '<strong>Every named ROI is clipped to the fixed main ROI.</strong> '
            'The source arrays remain preserved; regions stay at their configured native coordinates.</p>'
            '<p>Configured order: <code>' + _text_value(config.get('recipe', {}).get('steps', [])) +
            '</code>. Overlap policy: <code>' + _text_value(recipe.get('overlap')) + '</code>. '
            'With <code>sequential</code>, later edits operate on earlier results in overlapping regions. '
            'With <code>error</code>, overlapping active regions prevent execution.</p>'
            '<p>Region names such as “ascending” and “descending” are working labels; they do not establish '
            'anatomical orientation. Confirm placement from the native coordinates and source anatomy.</p>' + warnings +
            policy_content +
            '<h3>Named ROI definitions</h3>' +
            _table(['Name', 'Shape', 'Ordered centers (i, j, k)', 'Radii (mm)', 'Effective ROI voxels',
                    'Original target voxels', 'Voxels clipped by main ROI'], roi_rows) + ''.join(visualizations) +
            '<h3>Execution order and per-pass accounting</h3>' +
            _table(['Pass', 'Edit', 'ROI', 'Iteration', 'Operation', 'Target before → after',
                    'Added', 'Removed', 'Blocked', 'Unresolved'], step_rows) +
            '<p>Per-pass counts refer to that effective named ROI. Repeating an edit recomputes distances '
            'from the current geometry; multiple small passes need not equal one larger pass.</p>' + ''.join(details) +
            '<h3>Final result versus original</h3><p>Net changes, unique changed voxels, and sums across passes '
            'answer different questions. A voxel can be edited more than once or restored by a later pass.</p>' +
            _table(['Quantity', 'Voxels'], net_rows) + final +
            '<p><strong>Geometry status:</strong> ' + _text_value(recipe.get('topology_status')) + '</p>'
            '<p>Six-neighbor component counts below are spatial diagnostics. Clipping a target to a '
            'named ROI or crop can split it into pieces; components do not identify anatomical vessels '
            'and do not establish preservation of all topological properties.</p>' +
            _table(['Target scope', 'Connected components'], component_rows) +
            '<p><strong>Image status:</strong> ' + _text_value(recipe.get('scalar_status')) + '</p>'
            '<p>Original and final arrays: <code>' + _text_value(recipe.get('array_artifact', 'edit.npz')) + '</code>. '
            'Per-pass snapshots remain in the output directory. This report embeds the visualizations, not the voxel arrays.</p>'
            '<details><summary>Overlap counts and final original-label transitions</summary>' +
            _table(['ROI A', 'ROI B', 'Effective overlap (voxels)'], overlaps) +
            _table(['Original signed ID', 'Final signed ID', 'Changed voxels'], transition_rows) + '</details>'
            '<p class="warning">This recipe is an edit preview. Recipe sweeps are not yet connected to '
            'training-cohort preparation; no paired population or model is generated by this run.</p></section>')


def _training_section(output, report, embedded):
    plan = report['training_plan']
    target = plan.get('target_preview', {})
    rows = [[_text_value(v.get('variant_id')), _text_value(v.get('operation')),
             _measurement(v.get('distance_mm')), _measurement(v.get('shape_k')),
             _text_value(v.get('split'))] for v in plan.get('variants', [])]
    target_rows = [[label, _number(target.get(key))] for key, label in (
        ('target_voxels_in_crop', 'Full binary target in exported crop'),
        ('target_voxels_in_roi', 'Target inside editable ROI'),
        ('target_voxels_outside_roi', 'Unchanged target outside ROI'),
        ('target_components_6', 'Target components (six-neighbor)'))]
    model = plan.get('model', {})
    model_rows = [[_escape(key), _text_value(value)] for key, value in model.items()]
    figure = _embed_png(output, 'training-target.png', 'Image and segmentation-target design',
                        'Source attenuation above; full-crop binary target below. Cyan outlines the target; '
                        'transparent orange is the region where geometry edits are permitted. '
                        'This figure shows the original anatomy, including during a single-edit preview.', embedded)
    return ('<section id="training"><h2>Training study: review before preparation</h2>'
            '<p class="rule"><strong>No cohort or model was generated by this preview.</strong> '
            'The draft sweep contains <strong>' + _number(plan.get('variant_count')) + '</strong> variants. '
            'Each future variant starts independently from the original phantom.</p>'
            '<p><strong>Binary target:</strong> original IDs <code>' + _text_value(plan.get('target_source_ids')) +
            '</code> over the entire exported crop after editing. The ROI limits edits; it does not clip the '
            'segmentation label. The changed-voxel mask is a separate diagnostic.</p>'
            '<p>' + _text_value(target.get('scope')) + '</p>' +
            _table(['Target definition', 'Native voxels / components'], target_rows) + figure +
            '<h3>Draft edit sweep</h3><p>Distances are physical grid displacements, not stenosis percentages. '
            'Native sampling and protected tissues may prevent small requests from changing any voxels. '
            'Identical binary-mask geometries stay in one split; baseline-equivalent variants stay in training. '
            'The provisional assignments below can therefore change during preparation.</p>' +
            _table(['Variant', 'Operation', 'Distance (mm)', 'Shape k', 'Provisional split'], rows) +
            '<p class="warning">' + _text_value(plan.get('split_warning')) + '</p>'
            '<p class="warning">' + _text_value(plan.get('image_warning')) + '</p>'
            '<details><summary>Starter TensorFlow experiment and output locations</summary>' +
            _table(['Model setting', 'Value'], model_rows) +
            '<p>Future paired data: <code>' + _text_value(plan.get('dataset_directory')) + '</code><br>'
            'Future model: <code>' + _text_value(plan.get('model_directory')) + '</code></p>'
            '<p>The initial 2D U-Net maps attenuation patches to the binary target. Fixed clipping bounds '
            'are saved with the model. Validation selects the checkpoint; test images and masks are reserved '
            'from fitting and selection.</p></details>'
            '<h3>Continue from the same INI</h3><ol><li>Keep <code>[train] stage = preview</code>; adjust '
            'ROI nodes, radii and the inspection crosshair. To inspect a single deformation, set '
            '<code>[edit] operation = erosion</code> or <code>dilation</code>.</li>'
            '<li>Review the full target, realized edit and ranges. <code>stage = plan</code> writes a metadata-only plan.</li>'
            '<li>When ready, <code>stage = prepare</code> creates pairs using the draft sweep. '
            '<code>stage = fit</code> separately trains from that frozen dataset.</li></ol></section>')


def write_preview_report(output_dir, report, input_ini_text):
    """Write ``report.html`` with all preview assets embedded, refusing overwrite.

    ``report.selection`` may contain candidate/ROI/selected voxel counts,
    six-connected component sizes and selected original-ID counts. Geometry
    accepts ``roi_kind``, ``nodes_ijk`` and ``radii_mm`` or legacy sphere fields.
    Optional ``report.edit`` adds morphology accounting and before/after figures.
    Missing optional assets/diagnostics are shown as unavailable, not invented.
    The report can be copied alone and opened without a server or network.
    """
    output = Path(output_dir)
    destination = output / 'report.html'
    if destination.exists():
        raise FileExistsError('report.html already exists; choose a new output directory')
    if not isinstance(input_ini_text, str):
        raise TypeError('input_ini_text must be the captured INI text')
    if not isinstance(report, dict):
        raise TypeError('report must be a dictionary')
    edit_enabled = isinstance(report.get('edit'), dict)
    recipe_enabled = isinstance(report.get('recipe'), dict)
    changed_preview = edit_enabled or recipe_enabled
    config = report.get('config', {})
    study = config.get('study', {}).get('name', 'XCAT ROI preview')
    case = config.get('input', {}).get('case_id', 'Not recorded')
    frame = config.get('input', {}).get('frame', 'Not recorded')
    tissue = config.get('selection', {}).get('tissue', 'Not recorded')
    explicit_ids = config.get('selection', {}).get('source_ids', [])
    geometry = report.get('geometry', {})
    kind, nodes = _control_points(report)
    selection, candidate, selected, roi, component_count = _selection(report)
    warnings = []
    for value in [report.get('selection_warning'), *selection.get('warnings', []),
                  *report.get('volume', {}).get('warnings', [])]:
        if value and value not in warnings:
            warnings.append(str(value))
    if report.get('roi_clipped_by_source_boundary'):
        warnings.append('The ROI reaches beyond the source volume boundary; the displayed selection is clipped.')
    if selected == 0 and not any('no selected' in w.lower() or 'empty' in w.lower() for w in warnings):
        warnings.append('The tissue–ROI intersection is empty. Relocate the ROI, review its radii, or review the tissue choice.')
    if component_count is not None and component_count > 1 and not any('component' in w.lower() for w in warnings):
        warnings.append('The selected mask has multiple six-connected components. Inspect them before treating the selection as one target.')
    embedded = {}
    figures = [_embed_png(output, filename, heading + (' — before edit' if changed_preview else ''),
                          caption, embedded) for filename, heading, caption in _ASSETS]
    if (output / 'roi-volume.html').is_file():
        volume_content = _embed_volume(output, 'roi-volume.html',
                                        ('Before-edit ' if changed_preview else '') + 'Interactive 3D ROI and tissue volume', embedded)
    else:
        volume_content = '<p class="subtle">Interactive 3D preview was not generated.</p>'
    edit_section = _morphology_section(output, report, embedded) if edit_enabled else ''
    recipe_section = _recipe_section(output, report, embedded) if recipe_enabled else ''
    training_section = _training_section(output, report, embedded) if isinstance(report.get('training_plan'), dict) else ''
    selected_rows, selected_description = _selected_rows(report, selection)
    components = selection.get('components_voxels_6', [])
    component_rows = []
    for index, component in enumerate(components):
        if isinstance(component, dict):
            size = component.get('voxel_count', component.get('voxels', component.get('size')))
        else:
            size = component
        component_rows.append([str(index + 1), _number(size)])
    group_rows = [[_escape(name.replace('_', ' ')), _number(count)] for name, count in report.get('groups', {}).items() if count]
    provenance_rows = []
    for name, entry in report.get('atlas_sources', {}).items():
        if isinstance(entry, dict):
            provenance_rows.append([_escape(name), _escape(entry.get('path', 'Not recorded')),
                                    '<code class="hash">' + _escape(entry.get('sha256', 'Not recorded')) + '</code>'])
    hashes = {key: report[key] for key in ('input_config_sha256', 'catalog_sha256', 'audit_sha256') if key in report}
    source_rows = []
    for name, entry in report.get('source_files', {}).items():
        if isinstance(entry, dict):
            source_rows.append([_escape(name), _escape(entry.get('path', 'Not recorded')),
                                '<code class="hash">' + _escape(entry.get('crop_sha256', 'Not recorded')) + '</code>',
                                _escape(entry.get('integrity_scope', 'Not recorded'))])
    provenance = {'schema_version': report.get('schema_version'), 'generated_at_utc': report.get('generated_at_utc'),
                  'hashes': hashes, 'code_sha256': report.get('code_sha256', {}), 'embedded_assets': embedded}
    metric_html = ''.join('<div class="metric"><strong>' + _number(value) + '</strong><span>' + label + '</span></div>'
                          for label, value in [('Tissue candidates in crop', candidate), ('ROI voxels in crop', roi),
                                               ('Selected voxels inside ROI', selected), ('Selected components (6-neighbor)', component_count)])
    warning_html = ''.join('<p class="warning">' + _escape(w) + '</p>' for w in warnings)
    selection_filter = ('All original IDs mapped to this tissue are candidates.' if not explicit_ids else
                        'The tissue candidates are additionally restricted to ' + str(len(explicit_ids)) + ' configured original ID(s).')
    kind_explanation = ('The tube follows straight segments between ordered control points, with linearly varying radius and round caps. '
                        'The order is the path order: points are not sorted by coordinate.' if kind == 'tube' else
                        'The center and physical radius define a sphere in native voxel coordinates.')
    policy = report.get('policy_version', report.get('catalog_policy_version', 'See pinned policy source below'))
    caption = ('Case ' + _escape(case) + ' · frame ' + _escape(frame) + ' · ' + _escape(tissue.replace('_', ' ')) +
               ' · ' + _escape(kind) + ' ROI')
    status = ('Named-region recipe preview · source volumes preserved' if recipe_enabled else
              ('Morphology trial · source volumes preserved' if edit_enabled else 'Preview only · source labels preserved'))
    if training_section:
        status += ' · cohort not prepared'
    edit_nav = '<a href="#edit">Morphology trial</a>' if edit_enabled else ''
    if recipe_enabled:
        edit_nav += '<a href="#recipe">Named-region recipe</a>'
    if training_section:
        edit_nav += '<a href="#training">Training target and plan</a>'
    before_suffix = ' — before edit' if changed_preview else ''
    original_notice = ('<p class="subtle">This overview and the original 2D, 3D and selection-detail sections '
                       'describe the source <strong>before edit</strong>. The ' +
                       ('recipe' if recipe_enabled else 'morphology') + ' section shows the applied trial.</p>'
                       if changed_preview else '')
    edit_instruction = ('<li>Adjust <code>[edit]</code> operation, distance and profile, and review '
                        '<code>[reassignment]</code> rules before the next trial.</li>' if edit_enabled else '')
    if recipe_enabled:
        edit_instruction = ('<li>Adjust named <code>[roi.NAME]</code> regions and <code>[edit.NAME]</code> '
                            'operations. Set their order in <code>[recipe] steps</code>; '
                            '<code>iterations</code> repeats one edit before the next named step.</li>')
    # Plotly's bundled regl compiler constructs functions dynamically. Its inline
    # WebGL renderer therefore needs unsafe-eval as well as inline scripts.
    # The iframe stays sandboxed without same-origin access; network requests
    # and linked resources remain blocked.
    csp = "default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline' 'unsafe-eval' blob:; frame-src 'self' about: data:; font-src data:; worker-src blob:; connect-src 'none'; base-uri 'none'; form-action 'none'"
    document = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
                '<meta http-equiv="Content-Security-Policy" content="' + _escape(csp) + '">',
                '<title>' + _escape(study) + ' — FakeCT ROI report</title><style>' + _STYLE + '</style></head><body>',
                '<header><div class="eyebrow">FakeCT · ROI planning report</div><h1>' + _escape(study) + '</h1>',
                '<p class="subtle">' + caption + '</p><span class="status">' + status + '</span>',
                '<nav aria-label="Report sections"><a href="#overview">Overview</a><a href="#roi">ROI definition</a>' + edit_nav + '<a href="#slices">2D views</a><a href="#volume">3D views</a><a href="#selection">Selection detail</a><a href="#input">Edit input</a><a href="#provenance">Provenance</a></nav></header><main>',
                '<section id="overview"><h2>Selection overview' + before_suffix + '</h2>' + original_notice + '<p class="rule"><strong>Selected target = tissue candidates ∩ ROI.</strong> ' + _escape(selection_filter) +
                ' A narrow tube can follow one nearby artery while leaving another outside the ROI.</p><div class="metrics">' + metric_html + '</div>' + warning_html,
                '<p>Unknown or review-required voxels inside the ROI: <strong>' + _number(report.get('unknown_group_voxels_in_roi')) + '</strong>. Inspect magenta regions when reviewing the tissue selection.</p>',
                '<p>Counts use native voxels in this crop. A single connected component does not prove that only one anatomical vessel is included; nearby vessels may meet or share an original label. Inspect the overlays and original-label table before changing geometry.</p></section>',
                '<section id="roi"><h2>ROI definition</h2><p>' + _escape(kind_explanation) + '</p>',
                _table(['Point in path order', 'i', 'j', 'k', 'Radius (mm)'], nodes),
                '<div class="two-column"><div><h3>Native coordinates</h3><p>Index order: <code>i, j, k</code>. Spacing (mm): <code>' + _text_value(geometry.get('spacing_ijk_mm')) + '</code>.</p><p class="subtle">' + _escape(geometry.get('orientation', 'Anatomical orientation and physical origin are unverified.')) + '</p></div>',
                '<div><h3>Crop bounds</h3><p>Lower index: <code>' + _text_value(geometry.get('crop_origin_ijk')) + '</code><br>Upper index (exclusive): <code>' + _text_value(geometry.get('crop_high_ijk_exclusive')) + '</code><br>Array shape (k, j, i): <code>' + _text_value(geometry.get('crop_shape_kji')) + '</code></p></div></div></section>',
                edit_section,
                recipe_section,
                training_section,
                '<section id="slices"><h2>Native 2D inspection' + before_suffix + '</h2><p>Inspect the transparent ROI against the tissue boundaries. Keep the intended target inside the overlay and adjacent structures outside it.</p>' + ''.join(figures[:2]) + '</section>',
                '<section id="volume"><h2>Three-dimensional context' + before_suffix + '</h2><p>Drag to rotate, scroll to zoom, click the legend to toggle structures, and use the opacity controls. The interactive figure is embedded in this report and works without a network connection.</p>' + volume_content + figures[2],
                '<p class="subtle">The volume represents binary label occupancy, not measured attenuation. Display sampling may expand thin structures; native masks determine the reported counts.</p></section>',
                '<section id="selection"><h2>What is inside the ROI?' + before_suffix + '</h2><p>' + _escape(selected_description) + ' Fine anatomical identity remains available through the original labels and catalog; grouping is a view of those labels.</p>',
                _table(['Original signed ID', 'Original anatomical name', 'Selected voxels inside ROI'], selected_rows, 'No original IDs recorded inside the ROI.'),
                '<details><summary>Connected components and crop composition</summary><p>Components use six-neighbor connectivity on the selected native mask. They are spatial diagnostics, not vessel identities.</p>',
                _table(['Component', 'Voxels'], component_rows, 'Component sizes not recorded, or selection is empty.'),
                '<h3>All tissue groups in the crop</h3><p>These counts include context outside the ROI.</p>', _table(['Tissue group', 'Crop voxels'], group_rows),
                '<p>Unknown or review-required voxels in the crop: <strong>' + _number(report.get('unknown_group_voxels')) + '</strong>. Missing dictionary IDs: <code>' + _text_value(report.get('missing_dictionary_ids', [])) + '</code>.</p></details></section>',
                '<section id="input"><h2>Adjust and regenerate</h2><ol><li>Edit <code>center_ijk</code> and <code>radius_mm</code> below. For a tube, keep control points in path order and provide exactly one radius for each point.</li>' + edit_instruction + '<li>Set a new <code>output.directory</code> to preserve this comparison.</li><li>Download the edited INI and run the command below from the FakeCT checkout.</li></ol>',
                '<pre>' + _escape(report.get('rerun_command', 'python3 scripts/preview_roi.py --config /path/to/xcat-roi.ini')) + '</pre>',
                '<p class="subtle">Editing this text does not change the displayed figures. They remain the captured result until the preview command is run again.</p>',
                '<label for="captured-input"><strong>Captured input, editable for the next run</strong></label><textarea id="captured-input" spellcheck="false" aria-describedby="input-status">\n' + _escape(input_ini_text) + '</textarea>',
                '<div class="actions"><button type="button" id="download-input">Download edited INI</button><button type="button" id="copy-input">Copy INI</button><span id="input-status" role="status">The figures reflect the original captured input.</span></div></section>',
                '<section id="provenance"><h2>Source and mapping provenance</h2><p>FakeCT grouping policy: <strong>' + _escape(policy) + '</strong>. Its version identifies the tissue grouping policy, not the DPI atlas. DPI sources are identified separately by their content hashes below.</p><p>Anatomical tissue groups are distinct from attenuation material classes. The original signed labels and anatomical names remain the basis for finer selections.</p>',
                _table(['Mapping source', 'Pinned path', 'SHA256'], provenance_rows),
                '<details><summary>Volume sources and reproducibility hashes</summary>',
                _table(['Source', 'Path', 'Crop SHA256', 'Integrity scope'], source_rows),
                '<pre>' + _escape(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False)) + '</pre></details></section></main>',
                '<footer>Generated ' + _escape(report.get('generated_at_utc', 'at an unrecorded time')) + '. This single HTML file contains its figures and captured input. Source volumes remain at their recorded paths; reversible crop data remain in the output directory.</footer>',
                '<script>' + _SCRIPT + '</script></body></html>']
    encoded = '\n'.join(document).encode('utf-8')
    output.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        handle.write(encoded)
    return {'path': str(destination), 'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(),
            'embedded_assets': embedded}
