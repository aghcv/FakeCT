#!/usr/bin/env python3
"""Render a local HTML acceptance report from a completed manual-review record.

Reads files and verifies links plus the recorded preflight/freeze hashes. It
never executes morphology, freezes inputs, extracts pairs or trains a model.
Existing report files are preserved; choose a fresh output path for a revision.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import html
import json
import os
from pathlib import Path
import re
from urllib.parse import quote

ROOT = Path(__file__).resolve().parents[2]
SCHEMA = 'fakect.cohort-manual-acceptance/1'
PATH_FIELDS = ('config', 'preflight_path', 'freeze_path', 'visual_report', 'visual_image', 'roi_report')
SHA256 = re.compile(r'[0-9a-fA-F]{64}\Z')


def escape(value):
    return html.escape(str(value), quote=True)


def source_path(value):
    if not isinstance(value, str) or not value or '\x00' in value:
        raise ValueError('Artifact paths must be nonempty strings without NUL characters')
    path = Path(value).expanduser()
    return (path if path.is_absolute() else ROOT/path).resolve()


def digest(path):
    result = hashlib.sha256()
    with Path(path).open('rb') as stream:
        for block in iter(lambda: stream.read(1024*1024), b''):
            result.update(block)
    return result.hexdigest()


def href(path, output):
    """Encode local relative paths before HTML escaping, including # and spaces."""
    relative = Path(os.path.relpath(Path(path), output.parent)).as_posix()
    return escape(quote(relative, safe='/.-_~'))


def integer(value, name):
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f'{name} must be a nonnegative integer')
    return value


def strings(value, name):
    if not isinstance(value, list) or any(not isinstance(item, str) for item in value):
        raise ValueError(f'{name} must be a list of strings')
    return value


def validate(document):
    if not isinstance(document, dict) or document.get('schema_version') != SCHEMA:
        raise ValueError(f'Expected {SCHEMA}')
    for key in ('complete', 'paired_dataset_prepared'):
        if not isinstance(document.get(key), bool):
            raise ValueError(f'{key} must be Boolean')
    cases = document.get('cases')
    if not isinstance(cases, list) or not cases:
        raise ValueError('cases must be a nonempty list')
    count = integer(document.get('case_count'), 'case_count')
    total = integer(document.get('variant_count'), 'variant_count')
    if count != len(cases):
        raise ValueError('case_count disagrees with the case list')
    strings(document.get('summary_notes'), 'summary_notes')
    seen, variants, file_set = set(), 0, set()
    for row in cases:
        if not isinstance(row, dict) or not isinstance(row.get('case_id'), str) or not row['case_id']:
            raise ValueError('Every case requires a nonempty string case_id')
        case_id = row['case_id']
        if case_id in seen:
            raise ValueError(f'Duplicate case_id: {case_id}')
        seen.add(case_id)
        if not isinstance(row.get('decision'), str) or not row['decision']:
            raise ValueError(f'Case {case_id}: decision must be nonempty text')
        strings(row.get('review_notes'), f'Case {case_id}: review_notes')
        n = integer(row.get('variant_count'), f'Case {case_id}: variant_count')
        distinct = integer(row.get('unique_geometry_count'), f'Case {case_id}: unique_geometry_count')
        if distinct > n:
            raise ValueError(f'Case {case_id}: unique_geometry_count exceeds variant_count')
        variants += n
        ratios = row.get('local_coa_retained_ratio_min_max')
        if (not isinstance(ratios, list) or len(ratios) != 2 or
                any(isinstance(v, bool) or not isinstance(v, (int, float)) or not 0 <= v <= 1 for v in ratios) or
                ratios[0] > ratios[1]):
            raise ValueError(f'Case {case_id}: expected ordered finite retained ratios in [0,1]')
        for key in ('dataset_fingerprint', 'preflight_sha256', 'freeze_sha256'):
            if not isinstance(row.get(key), str) or not SHA256.fullmatch(row[key]):
                raise ValueError(f'Case {case_id}: {key} must be a SHA256 hex digest')
        for key in PATH_FIELDS:
            path = source_path(row.get(key))
            if not path.is_file():
                raise ValueError(f'Case {case_id}: missing {key}: {path}')
            file_set.add(path)
        for prefix in ('preflight', 'freeze'):
            if digest(source_path(row[prefix+'_path'])) != row[prefix+'_sha256'].lower():
                raise ValueError(f'Case {case_id}: {prefix} artifact hash disagrees with the review record')
    if variants != total:
        raise ValueError('variant_count disagrees with the sum of per-case counts')
    return file_set


def link(path, title, output):
    return f'<a href="{href(path, output)}">{escape(title)}</a>'


def notes(items):
    return '<ul>'+''.join(f'<li>{escape(item)}</li>' for item in items)+'</ul>' if items else '<p>No additional review notes.</p>'


def artifact_links(row, output):
    preflight = source_path(row['preflight_path'])
    native_report = preflight.with_suffix('.html')
    # The JSON remains separately accessible when the native HTML report exists.
    primary = native_report if native_report.is_file() else preflight
    entries = [(source_path(row['config']), 'Current INI'), (primary, 'Native scenario audit'),
               (source_path(row['freeze_path']), 'Freeze record'),
               (source_path(row['visual_report']), 'Before/after report'),
               (source_path(row['roi_report']), 'ROI review')]
    return '<nav aria-label="Case artifacts">'+''.join(link(p, name, output) for p, name in entries)+'</nav>'


def retained(row):
    lo, hi = row['local_coa_retained_ratio_min_max']
    return f'{lo*100:.2f}–{hi*100:.2f}%'


def render(document, review, review_sha256, output, verified_files):
    complete = document['complete']
    paired = document['paired_dataset_prepared']
    status = 'Manual acceptance complete' if complete else 'Draft manual review'
    native_label = 'Native scenarios executed' if complete else 'Scenarios in this review'
    pair_label = 'Prepared, as recorded in the manifest' if paired else 'Awaiting paired-array extraction'
    pair_description = ('The acceptance manifest records paired arrays as prepared.' if paired else
        'Native preflight executed the configured scenarios and retained their audits. '
        'The paired image/target arrays have not been extracted. Freeze records preserve '
        'the accepted inputs and native audits for subsequent reproducible preparation.')
    table_rows, cards = [], []
    for index, row in enumerate(document['cases'], 1):
        case_id = escape(row['case_id']); anchor = f'case-{index}'
        decision = escape(row['decision'])
        table_rows.append(f'<tr><th scope="row"><a href="#{anchor}">{case_id}</a></th>'
            f'<td>{row["variant_count"]:,}</td><td>{row["unique_geometry_count"]:,}</td>'
            f'<td>{retained(row)}</td><td>{decision}</td></tr>')
        figure = source_path(row['visual_image'])
        cards.append(f'<section class="case" id="{anchor}"><div class="case-heading">'
            f'<h2>Phantom {case_id}</h2><span class="decision">{decision}</span></div>'
            f'{artifact_links(row, output)}{notes(row["review_notes"])}'
            f'<figure><a href="{href(figure, output)}"><img loading="lazy" decoding="async" '
            f'src="{href(figure, output)}" alt="Maximum scenario before and after for phantom {case_id}"></a>'
            '<figcaption>Maximum scenario before/after figure. Open the image for full resolution; '
            'use the native scenario audit for every parameter combination.</figcaption></figure>'
            f'<p>Local CoA retained-volume range: <strong>{retained(row)}</strong>. '
            'This is the affected target subregion, not the minimum cross-sectional lumen area.</p>'
            '<details><summary>Recorded artifact fingerprints</summary><dl>'
            f'<dt>Dataset fingerprint</dt><dd><code>{escape(row["dataset_fingerprint"])}</code></dd>'
            f'<dt>Native preflight SHA256</dt><dd><code>{escape(row["preflight_sha256"])}</code> '
            f'{link(source_path(row["preflight_path"]), "JSON audit", output)}</dd>'
            f'<dt>Freeze SHA256</dt><dd><code>{escape(row["freeze_sha256"])}</code></dd>'
            '</dl></details></section>')
    generated = datetime.now(timezone.utc).isoformat()
    return f'''<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>CoA cohort acceptance report</title><style>
:root{{color-scheme:light;--ink:#172c43;--muted:#50657a;--line:#dbe3eb;--navy:#143c62;--blue:#095c9b}}
*{{box-sizing:border-box}}body{{margin:0;background:#f3f6fa;color:var(--ink);font:16px/1.55 system-ui,sans-serif}}
main{{max-width:1480px;margin:auto;padding:24px}}header{{background:var(--navy);color:white;border-radius:12px;padding:24px 28px}}
h1{{font-size:clamp(1.65rem,3vw,2.2rem);line-height:1.2;margin:8px 0 14px}}h2{{margin:0;font-size:1.4rem}}p{{margin:12px 0}}
.status{{font-size:.9rem;letter-spacing:.04em}}.metrics{{display:grid;grid-template-columns:repeat(3,1fr);gap:14px;margin:20px 0}}
.metric,.summary,.case{{background:white;border:1px solid var(--line);border-radius:10px;padding:20px}}
.metric strong{{display:block;font-size:1.9rem;line-height:1.25}}.metric span{{color:var(--muted);font-size:.9rem}}
.metric.extraction strong{{font-size:1.15rem;margin:6px 0}}.summary{{margin:18px 0}}.case{{margin:26px 0;scroll-margin-top:12px}}
.case-heading{{display:flex;align-items:center;justify-content:space-between;gap:16px;flex-wrap:wrap}}
.decision{{font-size:.88rem;background:#eaf1f7;border-radius:20px;padding:5px 12px}}a{{color:var(--blue);text-underline-offset:3px}}
nav{{display:flex;gap:10px 20px;flex-wrap:wrap;margin:14px 0;font-size:.94rem}}.table-scroll{{overflow-x:auto}}
table{{width:100%;border-collapse:collapse}}th,td{{padding:12px;text-align:left;border-bottom:1px solid var(--line);vertical-align:top}}
thead{{background:#eaf0f6}}tbody th{{white-space:nowrap}}figure{{margin:20px 0 8px}}img{{display:block;width:100%;height:auto;border:1px solid var(--line)}}
figcaption,.caption,footer{{color:var(--muted);font-size:.88rem}}li{{margin:6px 0}}details{{border-top:1px solid var(--line);padding-top:12px;margin-top:16px}}
summary{{cursor:pointer}}dt{{font-weight:600;margin-top:12px}}dd{{margin:3px 0}}code{{font-size:.83rem;overflow-wrap:anywhere}}footer{{padding:12px 0 28px}}
@media(max-width:700px){{main{{padding:12px}}header{{padding:20px}}.metrics{{grid-template-columns:1fr}}.case,.summary{{padding:16px}}th,td{{padding:9px}}}}
@media print{{body{{background:white}}main{{max-width:none;padding:0}}header{{color:black;background:white;border:1px solid #999}}.case{{break-inside:avoid}}nav a{{font-size:.8rem}}}}
</style></head><body><main>
<header><div class="status">{status}</div><h1>CoA synthetic cohort review and freeze record</h1>
<p>{escape(pair_description)}</p></header>
<div class="metrics"><div class="metric"><strong>{document['case_count']:,}</strong><span>Phantoms reviewed</span></div>
<div class="metric"><strong>{document['variant_count']:,}</strong><span>{native_label}</span></div>
<div class="metric extraction"><strong>{pair_label}</strong><span>Paired training data</span></div></div>
<section class="summary"><h2>Review conclusions</h2>{notes(document['summary_notes'])}
<p class="caption">Scope: manual technical acceptance of synthetic phantom scenarios. Counts include each configured baseline.
Unique geometry counts are measured within each phantom and do not establish independence between phantoms.</p></section>
<section class="summary"><h2>Phantom overview</h2><div class="table-scroll"><table><thead><tr>
<th>Phantom</th><th>Scenarios</th><th>Unique geometries within phantom</th><th>Local CoA retained volume</th><th>Decision</th>
</tr></thead><tbody>{''.join(table_rows)}</tbody></table></div>
<p class="caption">Retained-volume ranges use the local affected target region. They are not whole-ROI volume ratios or minimum lumen areas.</p></section>
{''.join(cards)}
<footer><p>{link(review, 'Manual acceptance JSON', output)} · Generated {escape(generated)}</p>
<p>Review SHA256: <code>{escape(review_sha256)}</code><br>
Verified {len(verified_files)} referenced files and {document['case_count']*2} recorded preflight/freeze hashes before rendering.
This renderer does not execute edits, freeze inputs or extract training pairs.</p></footer>
</main></body></html>'''


def run(review, output):
    review, output = source_path(review), source_path(output)
    if output.suffix.lower() not in ('.html', '.htm'):
        raise ValueError('--output must name an HTML file')
    if output.exists():
        raise ValueError('Output already exists; choose a fresh report path for this revision')
    raw = review.read_bytes(); document = json.loads(raw)
    verified = validate(document)
    # Also check the optional HTML audit paths used by the link builder.
    for row in document['cases']:
        native_html = source_path(row['preflight_path']).with_suffix('.html')
        if native_html.is_file():
            verified.add(native_html)
    checksum = hashlib.sha256(raw).hexdigest()
    page = render(document, review, checksum, output, verified)
    if review.read_bytes() != raw:
        raise ValueError('Manual acceptance input changed while rendering')
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open('x', encoding='utf-8') as stream:
        stream.write(page)
    result = {'report': str(output), 'review_sha256': checksum,
              'complete': document['complete'], 'case_count': document['case_count'],
              'variant_count': document['variant_count'],
              'paired_dataset_prepared': document['paired_dataset_prepared'],
              'verified_link_file_count': len(verified), 'verified_artifact_hash_count': document['case_count']*2}
    print(json.dumps(result, indent=2))
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--review', required=True, help='Manual acceptance JSON; absolute or repo-relative path')
    parser.add_argument('--output', required=True, help='Fresh HTML filename; absolute or repo-relative path')
    args = parser.parse_args()
    try:
        run(args.review, args.output)
    except (ValueError, OSError, KeyError, TypeError) as error:
        parser.exit(2, f'error: {error}\n')


if __name__ == '__main__':
    main()
