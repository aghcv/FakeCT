"""Portable, self-contained HTML reports for the bounded XCAT ROI workbench.

The report embeds existing preview assets. It does not read source volumes,
change labels, or equate connected components with anatomical vessels.
"""
import base64
import hashlib
import html
from html.parser import HTMLParser
import json
import math
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


def _embed_edit_profile(output, figures, filename, legacy_heading, legacy_caption,
                        arc_heading, arc_context, embedded):
    """Describe the recorded profile coordinate, retaining legacy axial reports."""
    heading, caption = legacy_heading, legacy_caption
    if figures.get('profile_coordinate') == 'centerline_arc_length_mm':
        heading = arc_heading
        reference = figures.get('profile_roi_name')
        caption = ('The horizontal axis is physical distance in millimeters from the start of the full '
                   'parent ROI, following its original unsmoothed control-point polyline. This is the '
                   'same path used by path_percent; it is not the smoothed direction-reference spline. ')
        if reference is not None:
            caption += 'Profile ROI: ' + str(reference) + '. '
        length = figures.get('profile_length_mm')
        if length is not None:
            try:
                length = format(float(length), ',.6g')
            except (TypeError, ValueError, OverflowError):
                length = str(length)
            caption += 'Full parent path length: ' + length + ' mm. '
        caption += ('Voxels are grouped by their nearest centerline arc bin; assigned voxel volume divided '
                    'by bin length gives an area equivalent (mm²), not a vessel-normal cross-section. ')
        if figures.get('profile_area_semantics'):
            caption += str(figures['profile_area_semantics']) + ' '
        caption += arc_context
    content = _embed_png(output, filename, heading, caption, embedded)
    endpoint_volumes = figures.get('arc_profile', {}).get('endpoint_volume_mm3', {})
    if endpoint_volumes:
        labels = {'before': 'Target before', 'after': 'Target after', 'added': 'Added target',
                  'removed': 'Removed target', 'blocked': 'Blocked proposals', 'unresolved': 'Unresolved proposals'}
        rows = [[_escape(labels.get(key, key)), _measurement(values[0]), _measurement(values[1])]
                for key, values in endpoint_volumes.items() if isinstance(values, (list, tuple)) and len(values) == 2]
        content += ('<details><summary>Endpoint-projected volumes excluded from the curves</summary>'
                    '<p>Voxels whose nearest path coordinate is exactly the start or end have no resolved '
                    'longitudinal span. These volumes are kept separate from the interior bin curves; '
                    'interior bins plus both endpoint totals conserve each mask’s volume.</p>' +
                    _table(['Quantity', 'Start endpoint (mm³)', 'End endpoint (mm³)'], rows) + '</details>')
    if figures.get('profile_data'):
        content += ('<p class="subtle">Plotted bin data: <code>' + _text_value(figures['profile_data']) +
                    '</code> in the output directory. The CSV contains the interior bin values; '
                    'endpoint totals are recorded separately in the profile metadata.</p>')
    if figures.get('axial_profile'):
        primary_note = (' The primary plot above follows distance along the ROI centerline.'
                        if figures.get('profile_coordinate') == 'centerline_arc_length_mm' else '')
        content += ('<details><summary>Native axial diagnostic (k slices)</summary>' +
                    _embed_png(output, figures['axial_profile'], 'Native axial profile diagnostic',
                               'This saved diagnostic uses native k-index slices. Its areas are intersections '
                               'with fixed axial planes, not vessel-normal cross-sections.' + primary_note,
                               embedded) + '</details>')
    return content


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
.report-tabs a[aria-selected="true"]{color:white;background:var(--accent);border-color:var(--accent)}.report-tabs a:focus-visible{outline:3px solid #e5a130;outline-offset:3px}.report-panel:focus{outline:none}.section-links{display:flex;gap:14px;flex-wrap:wrap;margin:0 0 20px}.section-links a{color:var(--accent)}.global-view iframe{height:1050px}.report-panel[hidden]{display:none}
.frame-preview iframe{height:1250px}
@media(max-width:700px){header{padding:25px 20px}main{padding:0 10px}section{padding:18px}.two-column{grid-template-columns:1fr}iframe{height:740px}.global-view iframe{height:1250px}.metric strong{font-size:1.6rem}th,td{padding:8px}}
@media print{body{background:#fff}nav,.actions,iframe{display:none}section{break-inside:auto;border:0;padding:15px 0}figure{break-inside:avoid}textarea{height:480px}header,main{max-width:none}details>*{display:block}details{break-inside:avoid}}
@media print{.report-panel[hidden]{display:block!important}.section-links{display:none}}
.report-tabs{position:sticky;top:0;z-index:10;max-width:1200px;margin:0 auto 20px;padding:10px;background:var(--wash);border-bottom:1px solid var(--line)}
.report-panel,section{scroll-margin-top:85px}
@media(max-width:700px){.report-panel,section{scroll-margin-top:135px}}
'''

_SCRIPT = '''
(function(){
  const tabs = Array.from(document.querySelectorAll('.report-tabs [role="tab"]'));
  const panels = Array.from(document.querySelectorAll('.report-panel'));
  function selectPanel(panel, moveFocus) {
    if (!panel) return;
    panels.forEach(function(item) { item.hidden = item !== panel; });
    tabs.forEach(function(tab) {
      const selected = tab.getAttribute('aria-controls') === panel.id;
      tab.setAttribute('aria-selected', String(selected));
      tab.tabIndex = selected ? 0 : -1;
      if (selected && moveFocus) tab.focus();
    });
  }
  function followHash() {
    let target;
    try { target = document.getElementById(decodeURIComponent(window.location.hash.slice(1))); }
    catch(error) { target = null; }
    const panel = target && target.closest('.report-panel');
    if (!panel) return false;
    selectPanel(panel, false);
    requestAnimationFrame(function() { target.scrollIntoView({block:'start'}); });
    return true;
  }
  function activateTab(tab, moveFocus) {
    const panel = document.getElementById(tab.getAttribute('aria-controls'));
    selectPanel(panel, moveFocus);
    // A native fragment navigation focuses the tabpanel in Firefox, which
    // prevents the next arrow key from reaching the selected tab.
    history.replaceState(null, '', tab.getAttribute('href'));
    panel.scrollIntoView({block:'start'});
  }
  tabs.forEach(function(tab, index) {
    tab.addEventListener('click', function(event) {
      event.preventDefault();
      activateTab(tab, true);
    });
    tab.addEventListener('keydown', function(event) {
      let next = index;
      if (event.key === 'ArrowRight') next = (index + 1) % tabs.length;
      else if (event.key === 'ArrowLeft') next = (index + tabs.length - 1) % tabs.length;
      else if (event.key === 'Home') next = 0;
      else if (event.key === 'End') next = tabs.length - 1;
      else if (event.key === ' ' || event.key === 'Enter') next = index;
      else return;
      event.preventDefault();
      activateTab(tabs[next], true);
    });
  });
  window.addEventListener('hashchange', followHash);
  if (!followHash()) {
    const navigation = document.querySelector('.report-tabs');
    selectPanel(document.getElementById(navigation.dataset.defaultPanel), false);
  }
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


def _main_tube_points(report):
    """Show one-based path nodes with cumulative physical distance and percent."""
    kind, rows = _control_points(report)
    if kind != 'tube' or not rows:
        return ''
    geometry = report.get('geometry', {})
    config = report.get('config', {})
    roi = config.get('roi', {})
    nodes = geometry.get('nodes_ijk', geometry.get('roi_nodes_ijk',
                         roi.get('nodes_ijk', roi.get('center_ijk'))))
    spacing = geometry.get('spacing_ijk_mm', config.get('input', {}).get('spacing_ijk_mm'))
    positions = None
    try:
        spacing = [float(value) for value in spacing]
        points = [[float(value) for value in point] for point in nodes]
        if (len(spacing) == 3 and all(math.isfinite(value) and value > 0 for value in spacing)
                and len(points) == len(rows) and all(len(point) == 3 for point in points)
                and all(math.isfinite(value) for point in points for value in point)):
            positions = [0.0]
            for before, after in zip(points, points[1:]):
                positions.append(positions[-1] + math.sqrt(sum(
                    ((after[axis] - before[axis]) * spacing[axis]) ** 2 for axis in range(3))))
    except (TypeError, ValueError, OverflowError):
        positions = None
    total = positions[-1] if positions else None
    path_rows = []
    for index, row in enumerate(rows):
        distance = positions[index] if positions is not None else None
        percent = 100 * distance / total if total and distance is not None else None
        path_rows.append(row + [_measurement(distance), _measurement(percent)])
    return ('<h3>Main tube point reference</h3><p>Point numbers start at 1 and follow the configured '
            'path order. Percent is cumulative physical centerline length, using voxel spacing; '
            'it is not the fraction of the point count. Total length: <strong>' + _measurement(total) +
            ' mm</strong>.</p>' +
            _table(['Point', 'i', 'j', 'k', 'Radius (mm)', 'Cumulative length (mm)', 'Path (%)'], path_rows))


def _range_description(metadata):
    selector = metadata.get('selector', 'full')
    values = metadata.get('selector_values')
    if selector == 'path_percent':
        return '<code>path_percent = ' + _text_value(values) + '</code> (%)'
    if selector == 'point_range':
        return '<code>point_range = ' + _text_value(values) + '</code> (1-based points)'
    if selector == 'full':
        return 'Full base ROI'
    return _text_value(selector) + ': ' + _text_value(values)


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


def _surface_overlay_section(output, edit, embedded, *, recipe=False):
    """One registered scene comparing original and final full-crop anatomy."""
    if not (output/'edit-overlay.html').is_file() and not (output/'edit-overlay.png').is_file():
        return ''
    metadata = edit.get('surface_overlay', {})
    after = 'the final result of the complete recipe' if recipe else 'the result of this edit'
    content = ('<div id="surface-overlay"><h3>Before / after surface overlay</h3>'
               '<p>Compare the original anatomy with ' + after + ' in one shared 3D view. '
               '<strong>Blue: before. Orange: after.</strong> Toggle each surface independently and '
               'choose its opacity: <strong>15%, 45%, or 80%</strong>. Drag to rotate and scroll to zoom.</p>'
               '<p class="subtle">Both surfaces use the same native coordinates and the full selected '
               'anatomy within this crop. They are not clipped to the editing ROI. Shared regions '
               'overlap; separated boundaries reveal expansion or erosion.</p>')
    if metadata.get('context_source') == 'final_tissue_labels':
        content += ('<p>Each final tissue category has its own visibility and opacity controls. '
                    'Context categories start hidden, except <strong>released</strong> diagnostic markers, '
                    'which appear in bright pink at 80% opacity. Hide either anatomy surface to inspect '
                    'these markers and enable neighboring tissue categories as needed. Ordinary context '
                    'surfaces may be sampled for display; released markers retain native resolution.</p>')
    if (output/'edit-overlay.html').is_file():
        content += _embed_volume(output, 'edit-overlay.html', 'Before and after surfaces in one 3D view', embedded)
    if (output/'edit-overlay.png').is_file():
        content += ('<details><summary>Static before / after overlay</summary>' +
                    _embed_png(output, 'edit-overlay.png', 'Static before / after surface overlay',
                               'Registered blue original and orange final surfaces. Interactive opacity '
                               'changes do not modify this saved image or any label data.', embedded) + '</details>')
    if metadata:
        content += '<details><summary>Surface counts and native geometry</summary><pre>' + _escape(
            json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)) + '</pre></details>'
    return content + '</div>'


def _centerline_frames_section(output, recipe, embedded):
    frames = recipe.get('centerline_frames', [])
    if not frames:
        return ''
    content = ['<div id="centerline-frames" class="frame-preview"><h3>Centerline directions and curvature</h3>',
               '<p class="rule">The original ROI points and masks remain unchanged. A smoothed spline '
               'provides the direction reference for inner and outer edits. '
               '<strong>T</strong> follows the configured point order; <strong>N</strong> points toward '
               'the local curvature center (inner), <strong>−N</strong> points outward, and '
               '<strong>B = T × N</strong>. Binormal sign depends on point order.</p>',
               '<p>Gray samples have undefined or unreliable inner/outer directions, including low-curvature '
               'or ambiguous parts of the fitted path. N, −N and B arrows are omitted there. '
               'These are mathematical directions of the ROI reference, not independently verified anatomical '
               'wall identities. Arrow length is a display scale, not the requested edit distance.</p>']
    for frame in frames:
        parent = _text_value(frame.get('parent_roi'))
        counts = [[label, _number(frame.get(key))] for key, label in (
            ('sample_count', 'Spline samples'), ('reliable_samples', 'Reliable inner/outer samples'),
            ('unreliable_samples', 'Undefined or unreliable samples'))]
        content += ['<h4>Base ROI: ' + parent + '</h4>', _table(['Frame diagnostic', 'Count'], counts)]
        content += ['<p class="warning">' + _escape(warning) + '</p>' for warning in frame.get('warnings', [])]
        if frame.get('html'):
            content.append(_embed_volume(output, frame['html'],
                                          'Centerline direction reference — ' + str(frame.get('parent_roi')), embedded))
        if frame.get('figure'):
            content += ['<details><summary>Static centerline directions</summary>',
                        _embed_png(output, frame['figure'], 'Centerline directions — ' + str(frame.get('parent_roi')),
                                   'Original ROI points and polyline, the fitted direction reference, and sparse '
                                   'trusted N, −N, B and T arrows. Translucent target context comes from original '
                                   'labels; sampling may widen thin structures.', embedded), '</details>']
        if frame.get('curvature_figure'):
            content.append(_embed_png(output, frame['curvature_figure'],
                                       'Curvature and reliability — ' + str(frame.get('parent_roi')),
                                       'Curvature is plotted against cumulative physical path percentage of '
                                       'the original parent ROI. Gray samples do not supply trusted inner/outer directions.', embedded))
        settings = [[_text_value(key), _text_value(value)] for key, value in frame.get('settings', {}).items()]
        settings += [['Arrow display length (mm)', _measurement(frame.get('arrow_length_mm'))],
                     ['Target context display stride', _number(frame.get('target_context_stride'))]]
        artifact = frame.get('artifacts', {}).get('frame_artifact', {})
        content += ['<details><summary>Spline fit, settings and complete frame data</summary>',
                    _table(['Setting', 'Value'], settings),
                    '<p>The complete sampled frame, original ROI, fitting metadata, transported fallback '
                    'normals and reliability flags remain in <code>' + _text_value(frame.get('frame_artifact')) +
                    '</code> in the output directory. This HTML embeds the views; it does not embed the full '
                    'frame JSON.</p><p>Frame JSON SHA256: <code class="hash">' +
                    _text_value(artifact.get('sha256')) + '</code>.</p><pre>' +
                    _escape(json.dumps(frame.get('metadata', {}), indent=2, ensure_ascii=False, allow_nan=False)) +
                    '</pre></details>']
    return ''.join(content) + '</div>'


def _release_assignment_section(output, summary, figures, embedded, *, scope='edit'):
    """Distinguish diagnostic release markers from assigned surrounding tissue."""
    release = summary.get('release_assignment', {})
    if not release or (release.get('mode') != 'diagnostic_label' and not release.get('current_released_voxels')):
        return ''
    new_label = {'recipe': 'Unique voxels marked released during this recipe',
                 'pass': 'Newly marked released in this pass',
                 'edit': 'Newly marked released in this edit'}[scope]
    content = ('<div class="diagnostic-release"><h3>Released diagnostic voxels' +
               (' — final recipe state' if scope == 'recipe' else '') + '</h3>'
               '<p><span class="legend-key" style="background:#ff2ea6"></span><strong>Hot pink: released.</strong> '
               'With <code>assign_surrounding_tissue = false</code>, erosion removes target geometry and '
               'marks the released voxels with a separate diagnostic label. It does not assign a surrounding '
               'tissue. This marker is a derived editing label, not an XCAT anatomical identity.</p>' +
               _table(['Diagnostic quantity', 'Voxels'], [
                   [new_label, _number(release.get('newly_released_voxels'))],
                   ['Current released markers', _number(release.get('current_released_voxels'))]]) +
               '<p class="warning">Released voxels retain their previous attenuation as an unassigned '
               'placeholder. Their intensity has not been reassigned to a surrounding tissue or recovered '
               'by AI. The <code>attenuation_unassigned_mask</code> identifies these voxels; the categorical '
               'geometry change alone does not complete an attenuation image.</p>'
               '<p>Label name: <code>' + _text_value(release.get('label_name')) + '</code>; derived signed ID: '
               '<code>' + _text_value(release.get('label_id')) + '</code>; diagnostic category: <code>' +
               _text_value(release.get('tissue_id')) + '</code>.</p>')
    if release.get('scalar_status'):
        content += '<p><strong>Attenuation status:</strong> ' + _text_value(release['scalar_status']) + '</p>'
    if figures.get('released_neighborhood'):
        content += _embed_png(output, figures['released_neighborhood'], 'Released voxels and surrounding native tissues',
                              'A native before/after close-up around an actual released voxel. The pink outline '
                              'marks the same locations in both states; hot-pink pixels after editing are diagnostic '
                              'markers. Colors represent labels, not attenuation.', embedded)
    neighbors = release.get('surrounding_labels', [])
    if neighbors:
        rows = [[_text_value(row.get('original_id')), _text_value(row.get('original_name')),
                 _text_value(row.get('tissue_name')), _number(row.get('count')),
                 _text_value(row.get('stiffness_group')), _measurement(row.get('stiffness'))] for row in neighbors]
        content += ('<h4>Observed labels around released voxels</h4>'
                    '<p>These are existing labels in the input-state boundary neighborhood. Remaining target '
                    'artery can appear alongside neighboring tissues. The observations do not establish which '
                    'tissue should replace the released voxels, or validate their anatomical identity.</p>' +
                    _table(['Input signed ID', 'Recorded anatomical name', 'Tissue group', 'Neighbor voxels',
                            'Resistance group', 'Stiffness factor'], rows))
    return content + '</div>'


def _erosion_safeguard_section(summary):
    """Explain accepted erosion geometry and the fixed reference used by retries."""
    guard = summary.get('erosion_safeguard', {})
    if not isinstance(guard, dict) or not guard.get('enabled'):
        return ''

    def percent(value):
        if value is None:
            return 'Not recorded'
        try:
            return _measurement(float(value) * 100) + '%'
        except (TypeError, ValueError, OverflowError):
            return _text_value(value)

    def connectivity(value):
        if not guard.get('preserve_connectivity'):
            return 'Not requested'
        return {True: 'Passed', False: 'Failed', None: 'Not evaluated'}.get(value, _text_value(value))

    status = guard.get('status')
    explanation = {
        'accepted': 'The requested distance passed the configured erosion checks.',
        'reduced': 'A smaller erosion distance passed the configured checks and was applied.',
        'skipped': 'No attempted erosion was accepted. The original input state of this pass was kept; '
                   'earlier accepted passes remain in place.',
    }.get(status, 'Review the recorded outcome and attempts below.')
    rows = [
        ['Outcome', _text_value(status)],
        ['Requested erosion distance (mm)', _measurement(guard.get('requested_distance_mm'))],
        ['Accepted erosion distance (mm)', _measurement(guard.get('accepted_distance_mm'))],
        ['Local volume retained', percent(guard.get('retained_volume_ratio'))],
        ['Minimum local retention', percent(guard.get('min_volume_ratio'))],
        ['Target voxels in the fixed baseline', _number(guard.get('baseline_target_voxels'))],
        ['Retained target voxels', _number(guard.get('retained_target_voxels'))],
        ['Preserve connectivity', _text_value(guard.get('preserve_connectivity'))],
        ['Connectivity check', connectivity(guard.get('connectivity_ok'))],
        ['Affected components', _text_value(guard.get('affected_components'))],
    ]
    rows += [[label, _measurement(guard[key])] for key, label in (
        ('baseline_volume_mm3', 'Local target volume in the fixed baseline (mm³)'),
        ('retained_volume_mm3', 'Retained local target volume (mm³)')) if key in guard]
    content = ('<div class="erosion-safeguard"><h3>Erosion safeguard</h3><p><strong>' +
               _escape(explanation) + '</strong></p>' + _table(['Check', 'Result'], rows) +
               '<p><strong>Fixed reference:</strong> the baseline is captured before this edit step '
               'and remains constant across its iterations and retry attempts. Each attempt starts from '
               'the current pass input; rejected trials do not accumulate changes.</p>')
    if guard.get('scope_semantics'):
        content += '<p><strong>Measured region:</strong> ' + _text_value(guard['scope_semantics']) + '</p>'
    if guard.get('reference_semantics'):
        content += '<p><strong>Recorded reference:</strong> ' + _text_value(guard['reference_semantics']) + '</p>'
    attempts = []
    reason_labels = {'local_volume_below_floor': 'Local volume below configured minimum',
                     'affected_component_split_or_lost': 'An affected target component split or disappeared'}
    show_fragments = any(attempt.get('component_details') for attempt in guard.get('attempts', []))
    for index, attempt in enumerate(guard.get('attempts', []), 1):
        accepted = attempt.get('accepted')
        outcome = 'Accepted' if accepted is True else 'Rejected' if accepted is False else 'Not recorded'
        reasons = attempt.get('reasons')
        if isinstance(reasons, (list, tuple)):
            reasons = [reason_labels.get(reason, reason) if isinstance(reason, str) else reason for reason in reasons]
        elif isinstance(reasons, str):
            reasons = reason_labels.get(reasons, reasons)
        attempts.append([_number(index), _measurement(attempt.get('distance_mm')),
                         percent(attempt.get('retained_volume_ratio')),
                         _number(attempt.get('retained_target_voxels')),
                         connectivity(attempt.get('connectivity_ok')), outcome,
                         _text_value(reasons) if reasons else 'Checks passed' if accepted else 'Not recorded'])
        if show_fragments:
            fragments = ['Component ' + _number(row.get('component_id')) + ': 1 → ' +
                         _number(row.get('surviving_components')) for row in attempt.get('component_details', [])]
            attempts[-1].append('; '.join(fragments) if fragments else 'Not evaluated')
    if attempts:
        content += ('<details><summary>Erosion attempts and retry reasons</summary>' +
                    _table(['Attempt', 'Distance (mm)', 'Local volume retained', 'Retained target voxels',
                            'Connectivity', 'Decision', 'Reasons'] +
                           (['Baseline → surviving components'] if show_fragments else []), attempts) + '</details>')
    return content + ('<p class="subtle">These are voxel-level geometry checks. Preserved connectivity '
                      'does not bound minimum cross-sectional area; a connected vessel can still be '
                      'very narrow. Review the resulting geometry in the previews.</p></div>')


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
                       ('shape_k', 'Profile shape k'), ('shape_window', 'Profile window'),
                       ('assign_surrounding_tissue', 'Assign surrounding tissue')):
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
    figures += _embed_edit_profile(output, edit.get('figures', {}), 'edit-profile.png',
                                   'Achieved cross-sections and edit profile',
                                   'Measured cross-sections describe this edited crop. A requested distance or profile is an '
                                   'input to the trial; assess the achieved change shown here before using it for a cohort.',
                                   'Edit profile and achieved geometry along the ROI centerline',
                                   'A requested edit distance is an input to the trial; assess the achieved change '
                                   'before using it for a cohort.', embedded)
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
            + _surface_overlay_section(output, edit, embedded) +
            '<p class="rule">Target counts and volumes below refer to the target <strong>inside the ROI</strong>, '
            'before and after this trial. Applied additions and removals describe actual label changes.</p>'
            '<div class="metrics">' + metrics + '</div>' + warnings +
            _erosion_safeguard_section(edit) +
            _release_assignment_section(output, edit, edit.get('figures', {}), embedded) +
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
    selection_role = recipe.get('roi_role', config.get('recipe', {}).get('roi_role')) == 'selection'
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
    region_definitions = {row.get('name'): row for row in figures.get('rois', [])}
    range_enabled = any(row.get('range_metadata') for row in region_definitions.values())
    roi_rows = []
    for row in figures.get('rois', []):
        coverage = recipe.get('roi_coverage', {}).get(row.get('name'), {})
        range_metadata = row.get('range_metadata', {})
        region_columns = [_text_value(row.get('display_name', row.get('name')))]
        if range_enabled:
            region_columns += [_text_value(row.get('parent_roi') or 'Independent named ROI'),
                               _range_description(range_metadata),
                               _measurement(range_metadata.get('selected_length_mm'))]
        roi_rows.append(region_columns + [_text_value(row.get('shape')),
                         _text_value(row.get('nodes_ijk')), _text_value(row.get('radii_mm')),
                         _number(row.get('effective_voxels')), _number(row.get('target_voxels')),
                         _number(coverage.get('clipped_by_outer_roi_voxels'))])
    roi_headers = ['Name'] + (['Base ROI', 'Selected range', 'Length (mm)'] if range_enabled else [])
    roi_headers += ['Shape', 'Ordered centers (i, j, k)', 'Radii (mm)',
                    'Original selector voxels' if selection_role else 'Effective ROI voxels',
                    'Original selected target voxels' if selection_role else 'Original target voxels',
                    'Voxels clipped by main ROI']
    step_rows = []
    details = []
    safeguards_present = any(step.get('summary', step).get('erosion_safeguard', {}).get('enabled')
                             for step in steps)
    for position, step in enumerate(steps, 1):
        summary = step.get('summary', step)
        current_counts = summary.get('counts', {})
        name = step.get('name', step.get('step_name'))
        roi = step.get('roi', step.get('roi_name'))
        region = region_definitions.get(roi, {})
        roi_display = region.get('display_name', roi)
        range_metadata = step.get('range_metadata', region.get('range_metadata', {}))
        parent_roi = step.get('parent_roi', region.get('parent_roi'))
        roi_description = _text_value(roi_display)
        if parent_roi is not None:
            roi_description += ' (base ' + _text_value(parent_roi) + ')'
        operation = summary.get('requested_operation', summary.get('operation'))
        step_rows.append([_number(step.get('index', position)), _text_value(name), roi_description,
                          _number(step.get('iteration')), _text_value(operation),
                          _number(current_counts.get('before')) + ' → ' + _number(current_counts.get('after')),
                          _number(current_counts.get('added')), _number(current_counts.get('removed')),
                          _number(current_counts.get('blocked')), _number(current_counts.get('unresolved'))])
        if safeguards_present:
            guard = summary.get('erosion_safeguard', {})
            step_rows[-1] += ([_text_value(guard.get('status')),
                              _measurement(guard.get('requested_distance_mm')),
                              _measurement(guard.get('accepted_distance_mm'))] if guard.get('enabled') else
                             ['Not enabled', '—', '—'])
        step_figures = step.get('figures', {})
        content = []
        for key, title, caption in (
                ('comparison', 'Input state, output state and changes',
                 'The input state includes preceding passes. The difference overlay uses the input-state '
                 'attenuation proxy. ' + ('Solid orange is the fixed original selector; dashed blue is the permitted '
                 'growth/edit footprint. Views include selected offspring outside the selector; coordinates may differ between passes.'
                 if selection_role else 'These views are focused within this named ROI; coordinates may differ between passes.')),
                ('profile', 'Requested profile and achieved axial areas',
                 ('Areas include the tracked selected lineage and its offspring outside the original selector; '
                  'they are not vessel-normal lumen areas.' if selection_role else
                  'Areas belong to this effective named ROI for this pass; they are not vessel-normal lumen areas.'))):
            if step_figures.get(key):
                if key == 'profile':
                    content.append(_embed_edit_profile(output, step_figures, step_figures[key], title, caption,
                                                       'Requested profile and achieved geometry along the ROI centerline',
                                                       caption, embedded))
                else:
                    content.append(_embed_png(output, step_figures[key], title, caption, embedded))
        metadata_rows = [[label, _text_value(value)] for label, value in (
            ('Status', step.get('status', summary.get('status'))),
            ('Base ROI', parent_roi),
            ('Distance per pass (mm)', summary.get('distance_mm')),
            ('Profile', summary.get('profile')), ('Profile axis', summary.get('profile_axis')),
            ('Requested release before target resistance (voxels)', current_counts.get('requested_removed_before_stiffness')),
            ('Release suppressed by target resistance (voxels)', current_counts.get('release_suppressed_by_target_stiffness')),
            ('Tracked target components: before' if selection_role else 'Components inside this named ROI: before', summary.get('components_before')),
            ('Tracked target components: after' if selection_role else 'Components inside this named ROI: after', summary.get('components_after')),
            ('Components of full-crop target: before', summary.get('full_target_components_before')),
            ('Components of full-crop target: after', summary.get('full_target_components_after')),
            ('Display focus (i, j, k)', step_figures.get('focus_ijk')),
            ('Input label SHA256', step.get('labels_before_sha256')),
            ('Output label SHA256', step.get('labels_after_sha256')),
            ('Pass arrays', step.get('artifact', step.get('array_artifact')))) if value is not None]
        if selection_role:
            metadata_rows += [[label, _number(current_counts.get(key))] for key, label in (
                ('changed_outside_selection_roi', 'Changed voxels outside the original selector'),
                ('added_outside_selection_roi', 'Added target voxels outside the original selector'),
                ('removed_outside_selection_roi', 'Removed target voxels outside the original selector'),
                ('unselected_target_contact_voxels', 'Added voxels contacting unselected input-state target'))]
            metadata_rows.append(['ROI role', 'Original target selection; descendants remain tracked outside the selector'])
        if range_metadata:
            metadata_rows += [
                ['Selected range', _range_description(range_metadata)],
                ['Selected path length (mm)', _measurement(range_metadata.get('selected_length_mm'))],
                ['Distance along base path (mm)', _measurement(range_metadata.get('start_distance_mm')) +
                 ' → ' + _measurement(range_metadata.get('end_distance_mm'))],
                ['Original base point numbers retained',
                 _text_value(range_metadata.get('original_node_numbers_retained'))]]
            if summary.get('profile_axis') == 'tube':
                metadata_rows.append(['Local tube profile coordinate',
                                      'u = 0 at the selected range start; u = 1 at its end. '
                                      'Gaussian shape_window uses this local coordinate.'])
        direction = summary.get('direction', {})
        if isinstance(direction, dict) and direction:
            metadata_rows += [[label, _text_value(direction.get(key))] for key, label in (
                ('direction', 'Circumferential edit direction'),
                ('angular_width_deg', 'Angular width (degrees)'),
                ('weight_semantics', 'Angular weighting'),
                ('unreliable_roi_voxels', 'Edit-footprint voxels without a reliable inner/outer frame' if selection_role else 'ROI voxels without a reliable inner/outer frame'),
                ('angular_supported_roi_voxels', 'Edit-footprint voxels with angular support' if selection_role else 'ROI voxels with angular support')) if direction.get(key) is not None]
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
                       _text_value(name) + ' · ROI ' + roi_description + ' · iteration ' +
                       _number(step.get('iteration')) + '</summary>' +
                       _table(['Pass setting', 'Value'], metadata_rows) + warnings +
                       _erosion_safeguard_section(summary) +
                       _release_assignment_section(output, summary, step_figures, embedded, scope='pass') +
                       blocked_labels + ''.join(content) + '</details>')
    visualizations = []
    for key, heading, caption in (
            ('overview', 'Named regions at the study crosshair',
             ('Colored overlays select original target seeds. They are fixed selectors, not growth walls; '
              'per-pass figures show the permitted footprints. Cyan outlines the original target throughout the crop.'
              if selection_role else 'Colored overlays show each named ROI intersected with the main ROI. Cyan outlines the original '
              'target throughout the crop. Multiple regions can overlap.')),
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
             ('Original and final areas include all tracked selected lineages and their offspring outside the selector. '
              if selection_role else 'Original and final target areas are measured within the main ROI. ') + 'Blocked and unresolved '
             'masks are unions across passes; the requested profile is the maximum per-pass request, not a sum.')):
        if final_figures.get(key):
            if key == 'profile':
                final += _embed_edit_profile(output, final_figures, final_figures[key], heading, caption,
                                             'Final geometry and recipe activity along the ROI centerline',
                                             caption, embedded)
            else:
                final += _embed_png(output, final_figures[key], heading, caption, embedded)
    if (output/'after/roi-volume.html').is_file():
        final += '<h3>Final interactive 3D context</h3>' + _embed_volume(
            output, 'after/roi-volume.html', 'Final recipe target and tissue volume', embedded)
    if (output/'after/roi-surfaces.png').is_file():
        final += _embed_png(output, 'after/roi-surfaces.png', 'Final static 3D context',
                            ('Final edited labels and tissue context within the native crop; selected offspring can lie outside the original selector.'
                             if selection_role else 'Final edited labels and tissue context within the original main ROI.'), embedded)
    net_rows = [[label, _number(counts.get(key))] for key, label in (
        ('before', 'Original selected target (all tracked lineages)' if selection_role else 'Original target inside main ROI'),
        ('after', 'Final tracked target (including growth outside the selector)' if selection_role else 'Final target inside main ROI'),
        ('added', 'Net target additions'), ('removed', 'Net target removals'),
        ('changed', 'Original-to-final label differences'), ('ever_changed', 'Unique voxels changed in any pass'),
        ('scalar_changed', 'Original-to-final attenuation proxy differences'))]
    net_rows += [[label, _number(activity.get(key))] for key, label in (
        ('added', 'Sum of additions across passes'), ('removed', 'Sum of removals across passes'),
        ('changed', 'Sum of label changes across passes'))]
    if selection_role:
        net_rows += [[label, _number(counts.get(key))] for key, label in (
            ('changed_outside_selection_roi', 'Net changed voxels outside the original main selector'),
            ('added_outside_selection_roi', 'Net added target outside the original main selector'),
            ('removed_outside_selection_roi', 'Net removed target outside the original main selector'))]
        net_rows += [[label, _number(counts[key])] for key, label in (
            ('changed_outside_edit_selectors', 'Net changed voxels outside the union of original edit selectors'),
            ('unselected_target_contact_voxels', 'Contact with unselected current target')) if key in counts]
    component_rows = [[label, _number(recipe.get(key))] for key, label in (
        ('components_before', 'Original selected lineages' if selection_role else 'Original target inside main ROI'),
        ('components_after', 'Final tracked lineages including offspring' if selection_role else 'Final target inside main ROI'),
        ('full_target_components_before', 'Original full-crop target'),
        ('full_target_components_after', 'Final full-crop target')) if key in recipe]
    overlaps = [[_text_value(region_definitions.get(row.get('roi_a'), {}).get('display_name', row.get('roi_a'))),
                 _text_value(region_definitions.get(row.get('roi_b'), {}).get('display_name', row.get('roi_b'))),
                 _number(row.get('effective_overlap_voxels'))] for row in recipe.get('roi_overlaps', [])]
    transition_rows = [[_text_value(row.get('original_id')), _text_value(row.get('new_id')), _number(row.get('count'))]
                       for row in recipe.get('transitions', [])]
    warnings = ''.join('<p class="warning">'+_escape(w)+'</p>' for w in recipe.get('warnings', []))
    range_content = ''
    if range_enabled:
        range_content = (
            '<h3>Select edits along an existing tube</h3><p>Each edit can reuse the main tube or a named '
            'base ROI and select an interval with <code>path_percent</code> or <code>point_range</code>. '
            'For example, <code>path_percent = 30, 75</code> selects 30% through 75% of the base tube’s '
            'physical centerline length. Separate edits can reuse that tube with different ranges.</p>'
            + ('<p><strong>The range selects original ancestors:</strong> <code>path_percent</code> and '
               '<code>point_range</code> choose original target voxels along the base path. Their offspring '
               'may grow radially or beyond the original range end faces inside the permitted footprint. '
               'A uniform edit can cross those end faces; the selected interval is not an offspring boundary.</p>'
               if selection_role else '<p><strong>Range boundaries are strict:</strong> effective voxels must lie '
               'inside the main ROI and within the selected interval of the base tube’s closest physical path coordinate. '
               'Rounded endpoint spheres do not extend the edit past that interval.</p>') +
            '<p>For <code>profile_axis = tube</code>, the selected interval is remapped to '
            '<code>u = 0…1</code>: 0 is its start and 1 is its end. '
            '<code>shape_window = 0, 1</code> applies the Gaussian across the whole selected interval; '
            'a narrower shape window is measured within that interval, not across the full base tube.' +
            (' The Gaussian request is zero outside its local shape window even when the permitted growth '
             'footprint extends farther.' if selection_role else '') + '</p>')
    roi_rule = ('<p class="rule"><strong>ROI role: original target selection.</strong> Solid orange identifies '
                'the fixed selector; dashed blue outlines the permitted growth/edit footprint in per-pass slices. '
                'The original ROI selects anatomy from the source volume. Its selected voxels and their '
                'expanded offspring remain tracked in later passes, including outside the selector. '
                'The selector is not a growth wall. Growth remains bounded by the edit footprint, crop and '
                'tissue resistance.</p><p>Other original voxels of the same tissue category do not seed the edit. '
                'Growing anatomy can still contact unselected current target, including offspring of other '
                'selected regions. Per-pass contact counts identify added voxels next to input-state target '
                'outside that pass’s selected lineage; they do not establish anatomical separation. '
                'The before/after surface overlay includes all target tissue in the crop for context, while '
                'per-pass counts follow the selected lineages. Each pass consumes the previous pass’s labels '
                'and attenuation proxy; original source arrays remain preserved.</p>' if selection_role else
                '<p class="rule">Each pass consumes the previous pass’s labels and attenuation proxy. '
                '<strong>Every named ROI is clipped to the fixed main ROI.</strong> '
                'The source arrays remain preserved; regions stay at their configured native coordinates.</p>')
    growth_summary = ''
    if selection_role:
        metrics = ''.join('<div class="metric"><strong>' + _number(counts.get(key)) + '</strong><span>' + label + '</span></div>'
                          for key, label in (('added', 'Net added tracked target'),
                                             ('removed', 'Net released tracked target'),
                                             ('changed_outside_selection_roi', 'Net label changes beyond the original main selector')))
        growth_summary = ('<div id="selection-growth-summary"><p><strong>The ROI selects original anatomy; '
                          'tracked offspring can grow outside it.</strong> Net counts below include that growth.</p>'
                          '<div class="metrics">' + metrics + '</div></div>')
    return ('<section id="recipe"><h2>Named-region edit recipe</h2>' +
            growth_summary +
            _release_assignment_section(output, recipe, final_figures, embedded, scope='recipe') +
            _surface_overlay_section(output, recipe, embedded, recipe=True) +
            _centerline_frames_section(output, recipe, embedded) +
            roi_rule +
            '<p>Configured order: <code>' + _text_value(config.get('recipe', {}).get('steps', [])) +
            '</code>. Overlap policy: <code>' + _text_value(recipe.get('overlap')) + '</code>. '
            'With <code>sequential</code>, later edits operate on earlier results in overlapping regions. '
            'With <code>error</code>, overlapping active regions prevent execution.</p>'
            '<p>Region names such as “ascending” and “descending” are working labels; they do not establish '
            'anatomical orientation. Confirm placement from the native coordinates and source anatomy.</p>' + warnings +
            policy_content +
            range_content + _main_tube_points(report) +
            '<h3>Named ROI definitions</h3>' +
            _table(roi_headers, roi_rows) + ''.join(visualizations) +
            '<h3>Execution order and per-pass accounting</h3>' +
            _table(['Pass', 'Edit', 'Original selector' if selection_role else 'ROI', 'Iteration', 'Operation', 'Target before → after',
                    'Added', 'Removed', 'Blocked', 'Unresolved'] +
                   (['Erosion safeguard', 'Requested distance (mm)', 'Accepted distance (mm)']
                    if safeguards_present else []), step_rows) +
            '<p>' + ('Per-pass counts follow the selected target and its tracked offspring, including growth outside the original selector. '
                     if selection_role else 'Per-pass counts refer to that effective named ROI. ') + 'Repeating an edit recomputes distances '
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


def _recipe_training_section(output, report, embedded):
    """Describe recipe cohorts without confusing the original selector with anatomy."""
    plan = report['training_plan']
    config = report.get('config', {})
    target = plan.get('target_preview', {})
    order = plan.get('recipe_steps', config.get('recipe', {}).get('steps', []))
    definitions = config.get('edits', {})
    step_rows = []
    for index, name in enumerate(order, 1):
        edit = definitions.get(name, {})
        range_key = next((key for key in ('path_percent', 'point_range') if key in edit), None)
        interval = (_escape(range_key) + ' = ' + _text_value(edit[range_key]) if range_key else 'Whole base ROI')
        step_rows.append([_number(index), _text_value(name), _text_value(edit.get('operation')),
                          _text_value(edit.get('roi', 'main')), interval,
                          _measurement(edit.get('distance_mm')), _number(edit.get('iterations'))])
    sweeps = plan.get('sweeps', config.get('sweeps', {}))
    sweep_rows = [[_text_value(name), _text_value(parameter), _text_value(values)]
                  for name, axes in sweeps.items() for parameter, values in axes.items()]
    variants = []
    for variant in plan.get('variants', []):
        parameters = variant.get('parameters', {})
        values = ['<code>' + _escape(name) + '.' + _escape(parameter) + '</code> = ' + _text_value(value)
                  for name, fields in parameters.items() for parameter, value in fields.items()]
        variants.append([_text_value(variant.get('variant_id')), _text_value(variant.get('operation')),
                         '<br>'.join(values) if values else 'Original phantom; no edits' if variant.get('operation') == 'none' else 'Configured recipe settings',
                         _text_value(variant.get('split'))])
    target_rows = [[label, _number(target.get(key))] for key, label in (
        ('target_voxels_in_crop', 'Original selected target in exported crop'),
        ('target_voxels_in_roi', 'Original target inside main selector'),
        ('target_voxels_outside_roi', 'Original selected target outside main selector'),
        ('target_components_6', 'Original selected components (six-neighbor)'))]
    model_rows = [[_escape(key), _text_value(value)] for key, value in plan.get('model', {}).items()]
    cohort_only = plan.get('split_mode') == 'unassigned'
    split_explanation = ('Model splits are unassigned here. A separate model experiment keeps each anatomy family '
                         'in one train, validation or test partition.' if cohort_only else
                         'Identical target masks stay in one split; baseline-equivalent masks stay in training, '
                         'so provisional assignments can change.')
    next_step = ('set <code>train.parameters_reviewed = true</code>, run <code>stage = freeze</code>, then '
                 '<code>stage = prepare</code>. Register the completed dataset for later model experiments.' if cohort_only else
                 '<code>stage = prepare</code> exports pairs and <code>stage = fit</code> trains from the frozen dataset.')
    figure = _embed_png(output, 'training-target.png', 'Original selected training target',
                        'Source attenuation above; original ROI-selected binary target below. Cyan outlines '
                        'selected tissue; orange is the fixed ancestor selector. Final pairs track surviving '
                        'selected ancestors and their descendants, including growth outside this selector. '
                        'This source figure does not display a completed cohort variant.', embedded)
    return ('<section id="training"><h2>Recipe training study: review before preparation</h2>'
            '<p class="rule"><strong>No cohort or model was generated by this preview.</strong> '
            'The draft contains <strong>' + _number(plan.get('variant_count')) + '</strong> variants. '
            'Each recipe variant starts independently from the original phantom and applies its steps in order.</p>'
            '<p><strong>Binary target: selected lineage.</strong> ' + _text_value(plan.get('target_definition')) +
            ' All selected tissue-category voxels inside the main ROI are included, including any neighboring '
            'branches captured by that ROI. Their surviving descendants stay positive after growth outside it. '
            'Unselected same-category anatomy stays negative. This is not a complete or anatomically exclusive '
            'aorta annotation. The changed-voxel mask remains a separate diagnostic.</p><p>' +
            _text_value(target.get('scope')) + '</p>' +
            _table(['Source target definition', 'Native voxels / components'], target_rows) + figure +
            '<h3>Ordered recipe</h3>' +
            _table(['Order', 'Edit', 'Operation', 'Base ROI', 'Original selection range',
                    'Configured distance (mm)', 'Iterations'], step_rows) +
            '<h3>Draft sweep axes</h3><p>The Cartesian sweep combines the values below; settings without a '
            'sweep keep their configured values. Each variant lists its requested parameters. Erosion safeguards '
            'may reduce an accepted distance, and tissue resistance or unresolved reassignment can limit the '
            'achieved geometry.</p>' + _table(['Edit', 'Parameter', 'Requested values'], sweep_rows) +
            _table(['Variant', 'Operation', 'Requested recipe parameters', 'Provisional split'], variants) +
            '<h3>Plan checks and native preflight</h3><p>This preview measures the source selection and '
            'the configured recipe trial. The draft sweep is a metadata plan; this preview has not executed '
            'every listed variant.</p><p>' + _text_value(plan.get('validation_scope')) + '</p>'
            '<p>Run <code>stage = preflight</code> to execute every planned combination on the native crop '
            'and review actual edits, accepted safeguard distances, unresolved voxels and duplicate geometries '
            'before exporting training pairs. ' + split_explanation + '</p>'
            '<p class="warning">' + _text_value(plan.get('split_warning')) + '</p>'
            '<p class="warning">' + _text_value(plan.get('image_warning')) + '</p>'
            '<details><summary>Output locations and model settings, if applicable</summary>' +
            _table(['Model setting', 'Value'], model_rows) +
            '<p>Native preflight audit: <code>' + _text_value(plan.get('preflight_directory')) + '</code><br>'
            'Future paired data: <code>' + _text_value(plan.get('dataset_directory')) + '</code><br>'
            'Future model: <code>' + _text_value(plan.get('model_directory') or 'Separate model-experiment INI') + '</code></p>'
            '<p>The 2D U-Net maps normalized attenuation patches to the binary selected-lineage target. '
            'Validation selects the checkpoint; test images and masks remain reserved from fitting and selection.</p></details>'
            '<h3>Continue from the same INI</h3><ol><li>Use <code>stage = preview</code> while reviewing the '
            'main ROI and ordered <code>[edit.NAME]</code> settings.</li><li>Adjust <code>[sweep.NAME]</code> '
            'values. <code>stage = plan</code> writes the metadata plan; <code>stage = preflight</code> '
            'measures every combination.</li><li>After reviewing the native results, ' + next_step + '</li></ol></section>')


def _training_section(output, report, embedded):
    plan = report['training_plan']
    if plan.get('schema_version') == 'fakect.recipe-study-plan/1' or plan.get('target_scope') == 'selected_lineage':
        return _recipe_training_section(output, report, embedded)
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


def _global_section(output, report, embedded):
    """Embed bounded whole-phantom sampling separately from native crop results."""
    metadata = report.get('global_view')
    metadata = metadata if isinstance(metadata, dict) else {}
    html_available = (output / 'roi-global.html').is_file()
    png_available = (output / 'roi-global.png').is_file()
    instruction = (
        '<p class="rule"><strong>Start with a tissue type; surface IDs are optional.</strong> '
        'Choose <code>[selection] tissue</code> and leave <code>source_ids =</code> blank or omit it to '
        'consider every original label in that tissue group. Use a sphere or a narrow tube '
        'to isolate the structure you want when nearby vessels share the same tissue type.</p>')
    workflow = ('<details><summary>Use global coordinates to refine the ROI</summary><ol><li>Explore the whole phantom with the global slice controls. Read native '
        '<code>i, j, k</code> coordinates and use the temporary ROI guide to estimate a location '
        'and radius.</li><li>Copy useful coordinates into <code>[roi]</code> or a named '
        '<code>[roi.NAME]</code> in the <a href="#input">Inputs tab</a>. A tube uses an ordered '
        'list of center points and one radius for each point.</li><li>Save the INI, choose a '
        'fresh output directory, and rerun the preview command. Switch to '
        '<a href="#slices">Local view</a> to refine the ROI against native-resolution '
        'slices, then repeat as needed.</li></ol>'
        '<p>Global crosshairs and temporary guides change this browser view only. '
        'They do not update the INI, the captured ROI, local figures, or applied edits. '
        'Local views and edit results reflect the captured input until the preview is rerun.</p></details>')
    viewer = (_embed_volume(output, 'roi-global.html', 'Interactive global phantom exploration', embedded)
              if html_available else
              '<p class="subtle">A global phantom view was not generated for this report. '
              'Regenerate the preview with the current code to add whole-phantom exploration. '
              'The Local view contains the captured crop.</p>')
    figure = (_embed_png(output, 'roi-global.png', 'Whole-phantom overview',
                        'Sampled source views locate the captured ROI within the full phantom. '
                        'Global sampling can hide thin structures; use local native slices for final placement.',
                        embedded) if png_available else '')
    details = ('<details><summary>Global sampling and source metadata</summary><pre>' +
               _escape(json.dumps(metadata, indent=2, ensure_ascii=False, allow_nan=False)) +
               '</pre></details>') if metadata else ''
    return ('<section id="global" class="global-view"><h2>Global phantom view</h2>' + instruction +
            viewer + workflow + figure + details + '</section>'), bool(html_available or png_available)


def _tab_navigation(edit_enabled, recipe_enabled, training_enabled, global_available):
    items = [('global', 'Global view'), ('local', 'Local view')]
    if edit_enabled or recipe_enabled:
        items.append(('edits', 'Edits / recipe' if recipe_enabled else 'Edits'))
    if training_enabled:
        items.append(('training', 'Training'))
    items += [('input', 'Inputs'), ('provenance', 'Provenance')]
    default = 'global' if global_available else 'local'
    links = ''.join('<a href="#panel-' + key + '" id="tab-' + key +
                    '" role="tab" aria-controls="panel-' + key + '" aria-selected="' +
                    ('true' if key == default else 'false') + '">' + label + '</a>'
                    for key, label in items)
    return ('<nav class="report-tabs" role="tablist" aria-label="Report views" '
            'data-default-panel="panel-' + default + '">' + links + '</nav>')


def _panel_start(name):
    return ('<div class="report-panel" id="panel-' + name + '" role="tabpanel" '
            'aria-labelledby="tab-' + name + '" tabindex="0">')


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
    global_section, global_available = _global_section(output, report, embedded)
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
    tab_navigation = _tab_navigation(edit_enabled, recipe_enabled, bool(training_section), global_available)
    edit_panel = (_panel_start('edits') + '<div class="section-links">' + edit_nav + '</div>' +
                  edit_section + recipe_section + '</div>') if changed_preview else ''
    training_panel = (_panel_start('training') + training_section + '</div>') if training_section else ''
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
                '</header>' + tab_navigation + '<main><noscript><p>JavaScript is disabled. All report views are shown below; the links jump between them. Static images remain available.</p></noscript>',
                _panel_start('global') + global_section + '</div>',
                _panel_start('local') + '<div class="section-links"><a href="#overview">Overview</a><a href="#roi">ROI definition</a><a href="#slices">2D views</a><a href="#volume">3D views</a><a href="#selection">Selection detail</a></div>',
                '<section id="slices"><h2>Native 2D inspection' + before_suffix + '</h2><p>Inspect the transparent ROI against the tissue boundaries. Keep the intended target inside the overlay and adjacent structures outside it.</p>' + ''.join(figures[:2]) + '</section>',
                '<section id="overview"><h2>Selection overview' + before_suffix + '</h2>' + original_notice + '<p class="rule"><strong>Selected target = tissue candidates ∩ ROI.</strong> ' + _escape(selection_filter) +
                ' A narrow tube can follow one nearby artery while leaving another outside the ROI.</p><div class="metrics">' + metric_html + '</div>' + warning_html,
                '<p>Unknown or review-required voxels inside the ROI: <strong>' + _number(report.get('unknown_group_voxels_in_roi')) + '</strong>. Inspect magenta regions when reviewing the tissue selection.</p>',
                '<p>Counts use native voxels in this crop. A single connected component does not prove that only one anatomical vessel is included; nearby vessels may meet or share an original label. Inspect the overlays before changing geometry. The original-label table is optional anatomical detail, not required input.</p></section>',
                '<section id="roi"><h2>ROI definition</h2><p>' + _escape(kind_explanation) + '</p>',
                _table(['Point in path order', 'i', 'j', 'k', 'Radius (mm)'], nodes),
                '<div class="two-column"><div><h3>Native coordinates</h3><p>Index order: <code>i, j, k</code>. Spacing (mm): <code>' + _text_value(geometry.get('spacing_ijk_mm')) + '</code>.</p><p class="subtle">' + _escape(geometry.get('orientation', 'Anatomical orientation and physical origin are unverified.')) + '</p></div>',
                '<div><h3>Crop bounds</h3><p>Lower index: <code>' + _text_value(geometry.get('crop_origin_ijk')) + '</code><br>Upper index (exclusive): <code>' + _text_value(geometry.get('crop_high_ijk_exclusive')) + '</code><br>Array shape (k, j, i): <code>' + _text_value(geometry.get('crop_shape_kji')) + '</code></p></div></div></section>',
                '<section id="volume"><h2>Three-dimensional context' + before_suffix + '</h2><p>Drag to rotate, scroll to zoom, click the legend to toggle structures, and use the opacity controls. The interactive figure is embedded in this report and works without a network connection.</p>' + volume_content + figures[2],
                '<p class="subtle">The volume represents binary label occupancy, not measured attenuation. Display sampling may expand thin structures; native masks determine the reported counts.</p></section>',
                '<section id="selection"><h2>What is inside the ROI?' + before_suffix + '</h2><p>' + _escape(selected_description) + ' Fine anatomical identity remains available through the original labels and catalog; grouping is a view of those labels.</p>',
                _table(['Original signed ID', 'Original anatomical name', 'Selected voxels inside ROI'], selected_rows, 'No original IDs recorded inside the ROI.'),
                '<details><summary>Connected components and crop composition</summary><p>Components use six-neighbor connectivity on the selected native mask. They are spatial diagnostics, not vessel identities.</p>',
                _table(['Component', 'Voxels'], component_rows, 'Component sizes not recorded, or selection is empty.'),
                '<h3>All tissue groups in the crop</h3><p>These counts include context outside the ROI.</p>', _table(['Tissue group', 'Crop voxels'], group_rows),
                '<p>Unknown or review-required voxels in the crop: <strong>' + _number(report.get('unknown_group_voxels')) + '</strong>. Missing dictionary IDs: <code>' + _text_value(report.get('missing_dictionary_ids', [])) + '</code>.</p></details></section></div>',
                edit_panel,
                training_panel,
                _panel_start('input') + '<section id="input"><h2>Adjust and regenerate</h2><ol><li>Choose a tissue in <code>[selection]</code>. Leave <code>source_ids =</code> blank for tissue-only selection. Original surface IDs are an optional extra filter; individual overrides in <code>[stiffness.labels]</code> are optional too.</li><li>Edit <code>center_ijk</code> and <code>radius_mm</code> below. For a tube, keep control points in path order and provide exactly one radius for each point.</li>' + edit_instruction + '<li>Set a new <code>output.directory</code> to preserve this comparison.</li><li>Download the edited INI and run the command below from the FakeCT checkout.</li></ol>',
                '<pre>' + _escape(report.get('rerun_command', 'python3 scripts/preview_roi.py --config /path/to/xcat-roi.ini')) + '</pre>',
                '<p class="subtle">Editing this text does not change the displayed figures. They remain the captured result until the preview command is run again.</p>',
                '<label for="captured-input"><strong>Captured input, editable for the next run</strong></label><textarea id="captured-input" spellcheck="false" aria-describedby="input-status">\n' + _escape(input_ini_text) + '</textarea>',
                '<div class="actions"><button type="button" id="download-input">Download edited INI</button><button type="button" id="copy-input">Copy INI</button><span id="input-status" role="status">The figures reflect the original captured input.</span></div></section></div>',
                _panel_start('provenance') + '<section id="provenance"><h2>Source and mapping provenance</h2><p>FakeCT grouping policy: <strong>' + _escape(policy) + '</strong>. Its version identifies the tissue grouping policy, not the DPI atlas. DPI sources are identified separately by their content hashes below.</p><p>Anatomical tissue groups are distinct from attenuation material classes. The original signed labels and anatomical names remain the basis for finer selections.</p>',
                _table(['Mapping source', 'Pinned path', 'SHA256'], provenance_rows),
                '<details><summary>Volume sources and reproducibility hashes</summary>',
                _table(['Source', 'Path', 'Crop SHA256', 'Integrity scope'], source_rows),
                '<pre>' + _escape(json.dumps(provenance, indent=2, ensure_ascii=False, allow_nan=False)) + '</pre></details></section></div></main>',
                '<footer>Generated ' + _escape(report.get('generated_at_utc', 'at an unrecorded time')) + '. This single HTML file contains its figures and captured input. Source volumes remain at their recorded paths; reversible crop data remain in the output directory.</footer>',
                '<script>' + _SCRIPT + '</script></body></html>']
    encoded = '\n'.join(document).encode('utf-8')
    output.mkdir(parents=True, exist_ok=True)
    with destination.open('xb') as handle:
        handle.write(encoded)
    return {'path': str(destination), 'bytes': len(encoded), 'sha256': hashlib.sha256(encoded).hexdigest(),
            'embedded_assets': embedded}
