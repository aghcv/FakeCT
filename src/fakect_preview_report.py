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


def write_preview_report(output_dir, report, input_ini_text):
    """Write ``report.html`` with all preview assets embedded, refusing overwrite.

    ``report.selection`` may contain candidate/ROI/selected voxel counts,
    six-connected component sizes and selected original-ID counts. Geometry
    accepts ``roi_kind``, ``nodes_ijk`` and ``radii_mm`` or legacy sphere fields.
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
    figures = []
    for filename, heading, caption in _ASSETS:
        path = output / filename
        if path.is_file():
            raw = path.read_bytes()
            encoded = base64.b64encode(raw).decode('ascii')
            embedded[filename] = {'bytes': len(raw), 'sha256': hashlib.sha256(raw).hexdigest()}
            figures.append('<figure><h3>' + _escape(heading) + '</h3><img loading="lazy" src="data:image/png;base64,' + encoded + '" alt="' + _escape(heading) + '"><figcaption>' + _escape(caption) + '</figcaption></figure>')
        else:
            figures.append('<p class="subtle">' + _escape(heading) + ': preview not generated.</p>')
    volume_path = output / 'roi-volume.html'
    if volume_path.is_file():
        volume_bytes = volume_path.read_bytes()
        volume_document = volume_bytes.decode('utf-8')
        parser = _AssetReferences()
        parser.feed(volume_document)
        if parser.references:
            raise ValueError('The 3D document must embed its resources; external or relative asset links were found')
        embedded[volume_path.name] = {'bytes': len(volume_bytes), 'sha256': hashlib.sha256(volume_bytes).hexdigest()}
        volume_content = '<iframe title="Interactive 3D ROI and tissue volume" sandbox="allow-scripts" loading="lazy" srcdoc="' + _escape(volume_document) + '"></iframe>'
    else:
        volume_content = '<p class="subtle">Interactive 3D preview was not generated.</p>'
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
    # Plotly's bundled regl compiler constructs functions dynamically. Its inline
    # WebGL renderer therefore needs unsafe-eval as well as inline scripts.
    # The iframe stays sandboxed without same-origin access; network requests
    # and linked resources remain blocked.
    csp = "default-src 'none'; img-src data: blob:; style-src 'unsafe-inline'; script-src 'unsafe-inline' 'unsafe-eval' blob:; frame-src 'self' about: data:; font-src data:; worker-src blob:; connect-src 'none'; base-uri 'none'; form-action 'none'"
    document = ['<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">',
                '<meta http-equiv="Content-Security-Policy" content="' + _escape(csp) + '">',
                '<title>' + _escape(study) + ' — FakeCT ROI report</title><style>' + _STYLE + '</style></head><body>',
                '<header><div class="eyebrow">FakeCT · ROI planning report</div><h1>' + _escape(study) + '</h1>',
                '<p class="subtle">' + caption + '</p><span class="status">Preview only · source labels preserved</span>',
                '<nav aria-label="Report sections"><a href="#overview">Overview</a><a href="#roi">ROI definition</a><a href="#slices">2D views</a><a href="#volume">3D views</a><a href="#selection">Selection detail</a><a href="#input">Edit input</a><a href="#provenance">Provenance</a></nav></header><main>',
                '<section id="overview"><h2>Selection overview</h2><p class="rule"><strong>Selected target = tissue candidates ∩ ROI.</strong> ' + _escape(selection_filter) +
                ' A narrow tube can follow one nearby artery while leaving another outside the ROI.</p><div class="metrics">' + metric_html + '</div>' + warning_html,
                '<p>Unknown or review-required voxels inside the ROI: <strong>' + _number(report.get('unknown_group_voxels_in_roi')) + '</strong>. Inspect magenta regions when reviewing the tissue selection.</p>',
                '<p>Counts use native voxels in this crop. A single connected component does not prove that only one anatomical vessel is included; nearby vessels may meet or share an original label. Inspect the overlays and original-label table before changing geometry.</p></section>',
                '<section id="roi"><h2>ROI definition</h2><p>' + _escape(kind_explanation) + '</p>',
                _table(['Point in path order', 'i', 'j', 'k', 'Radius (mm)'], nodes),
                '<div class="two-column"><div><h3>Native coordinates</h3><p>Index order: <code>i, j, k</code>. Spacing (mm): <code>' + _text_value(geometry.get('spacing_ijk_mm')) + '</code>.</p><p class="subtle">' + _escape(geometry.get('orientation', 'Anatomical orientation and physical origin are unverified.')) + '</p></div>',
                '<div><h3>Crop bounds</h3><p>Lower index: <code>' + _text_value(geometry.get('crop_origin_ijk')) + '</code><br>Upper index (exclusive): <code>' + _text_value(geometry.get('crop_high_ijk_exclusive')) + '</code><br>Array shape (k, j, i): <code>' + _text_value(geometry.get('crop_shape_kji')) + '</code></p></div></div></section>',
                '<section id="slices"><h2>Native 2D inspection</h2><p>Inspect the transparent ROI against the tissue boundaries. Keep the intended target inside the overlay and adjacent structures outside it.</p>' + ''.join(figures[:2]) + '</section>',
                '<section id="volume"><h2>Three-dimensional context</h2><p>Drag to rotate, scroll to zoom, click the legend to toggle structures, and use the opacity controls. The interactive figure is embedded in this report and works without a network connection.</p>' + volume_content + figures[2],
                '<p class="subtle">The volume represents binary label occupancy, not measured attenuation. Display sampling may expand thin structures; native masks determine the reported counts.</p></section>',
                '<section id="selection"><h2>What is inside the ROI?</h2><p>' + _escape(selected_description) + ' Fine anatomical identity remains available through the original labels and catalog; grouping is a view of those labels.</p>',
                _table(['Original signed ID', 'Original anatomical name', 'Selected voxels inside ROI'], selected_rows, 'No original IDs recorded inside the ROI.'),
                '<details><summary>Connected components and crop composition</summary><p>Components use six-neighbor connectivity on the selected native mask. They are spatial diagnostics, not vessel identities.</p>',
                _table(['Component', 'Voxels'], component_rows, 'Component sizes not recorded, or selection is empty.'),
                '<h3>All tissue groups in the crop</h3><p>These counts include context outside the ROI.</p>', _table(['Tissue group', 'Crop voxels'], group_rows),
                '<p>Unknown or review-required voxels in the crop: <strong>' + _number(report.get('unknown_group_voxels')) + '</strong>. Missing dictionary IDs: <code>' + _text_value(report.get('missing_dictionary_ids', [])) + '</code>.</p></details></section>',
                '<section id="input"><h2>Adjust and regenerate</h2><ol><li>Edit <code>center_ijk</code> and <code>radius_mm</code> below. For a tube, keep control points in path order and provide exactly one radius for each point.</li><li>Set a new <code>output.directory</code> to preserve this comparison.</li><li>Download the edited INI and run the command below from the FakeCT checkout.</li></ol>',
                '<pre>python3 scripts/preview_roi.py --config /path/to/xcat-roi.ini</pre>',
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
