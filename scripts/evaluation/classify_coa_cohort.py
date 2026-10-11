#!/usr/bin/env python3
"""Write provisional CoA benchmark labels without modifying frozen cohort data.

The score is achieved LOCAL volume loss during one final, guarded CoA edit.
It is not minimum cross-sectional narrowing or a clinical diagnostic criterion.
No image arrays, learned models, or patient labels are produced by this script.
"""

from __future__ import annotations

import argparse
import configparser
import csv
from collections import defaultdict
from datetime import datetime, timezone
import hashlib
import html
import json
import math
from pathlib import Path
import shutil
import sys


ROOT = Path(__file__).resolve().parents[2]
SCHEMA = "fakect.coa-label-review/1"
SCORE_NAME = "coa_local_target_volume_loss_fraction"
LIMITATIONS = [
    "Draft engineering benchmark labels require review before training; they are not clinical CoA diagnoses.",
    "The score is lost target volume divided by pre-CoA target volume in the positive longitudinal profile support; it is not minimum cross-sectional area or diameter reduction.",
    "The reference is immediately before the CoA step, after earlier edits. This is not the original unedited phantom and does not measure combined effects of other conditions.",
    "Below-threshold edited siblings retain a CoA edit and companion abnormalities; benchmark-negative does not mean healthy or no narrowing.",
    "Original phantoms are non-CoA by the user's study assumption, not independent clinical adjudication.",
    "Every currently edited sibling also has arch narrowing and enlargement; isolated and independently varied controls are still required for a specific CoA study.",
    "This metadata audit validates recorded counts, spacing and manifest provenance; it does not remeasure geometry or reread image arrays.",
]


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def checked_hash(path: Path, expected: str) -> dict:
    observed = sha256(path)
    if observed != expected:
        raise ValueError(f"SHA-256 mismatch: {path}")
    return {"path": str(path.resolve()), "sha256": observed, "bytes": path.stat().st_size}


def resolve_path(value: str) -> Path:
    path = Path(value).expanduser()
    return (ROOT / path).resolve() if not path.is_absolute() else path.resolve()


def finite_fraction(value: str, name: str) -> float:
    number = float(value)
    if not math.isfinite(number) or not 0 <= number <= 1:
        raise ValueError(f"{name} must be finite and between 0 and 1")
    return number


def load_config(path: Path) -> dict:
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",))
    with path.open() as stream:
        parser.read_file(stream)
    allowed = {
        "review": {"dataset_manifest", "output_directory", "status"},
        "classification": {"step_name", "positive_loss_fraction", "review_half_width",
                           "sensitivity_thresholds", "original_baselines_non_coa"},
    }
    if set(parser.sections()) != set(allowed) or parser.defaults():
        raise ValueError("Use exactly [review] and [classification], with no defaults")
    for section, keys in allowed.items():
        if set(parser[section]) != keys:
            raise ValueError(f"Unexpected or missing parameters in [{section}]")
    if parser["review"]["status"] != "draft":
        raise ValueError("This review script writes draft labels only")
    if not parser.getboolean("classification", "original_baselines_non_coa"):
        raise ValueError("Original baseline non-CoA assumption must be explicit")
    section = parser["classification"]
    threshold = finite_fraction(section["positive_loss_fraction"], "positive_loss_fraction")
    if threshold <= 0:
        raise ValueError("positive_loss_fraction must be greater than zero")
    band = finite_fraction(section["review_half_width"], "review_half_width")
    sensitivity = [finite_fraction(x.strip(), "sensitivity_thresholds")
                   for x in section["sensitivity_thresholds"].split(",")]
    if not sensitivity or any(x <= 0 for x in sensitivity) or len(set(sensitivity)) != len(sensitivity):
        raise ValueError("Provide unique positive sensitivity thresholds")
    return {
        "config_path": str(path.resolve()), "config_sha256": sha256(path),
        "dataset_manifest": str(resolve_path(parser["review"]["dataset_manifest"])),
        "output_directory": str(resolve_path(parser["review"]["output_directory"])),
        "step_name": section["step_name"], "positive_loss_fraction": threshold,
        "review_half_width": band, "sensitivity_thresholds": sensitivity,
        "original_baselines_non_coa": True, "status": "draft",
    }


def exact_count(value, name: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < 0:
        raise ValueError(f"{name} must be a nonnegative integer")
    return value


def close(actual, expected, name: str) -> None:
    if isinstance(actual, bool) or not isinstance(actual, (float, int)) or not math.isfinite(actual):
        raise ValueError(f"{name} is not finite")
    if not math.isclose(actual, expected, rel_tol=1e-9, abs_tol=1e-12):
        raise ValueError(f"{name} disagrees with recorded voxel counts/spacing")


def classify_sample(row: dict, config: dict) -> dict:
    """Missing or inconsistent outcome metadata always becomes unknown, never negative."""
    result = {
        "sample_id": row["sample_id"], "case_id": row["case_id"],
        "variant_id": row["variant_id"], "anatomy_family": row.get("anatomy_family"),
        "original_baseline": row.get("baseline") is True,
        "coa_edit_applied": None, "coa_benchmark_positive": None,
        "class_description": "unknown", SCORE_NAME: None,
        "baseline_target_voxels": None, "retained_target_voxels": None,
        "coa_removed_voxels": None, "baseline_volume_mm3": None,
        "retained_volume_mm3": None, "requested_distance_mm": None,
        "accepted_distance_mm": None, "review_near_threshold": False,
        "other_applied_steps": [], "errors": [],
        "scope_semantics": None, "reference_semantics": None,
    }
    try:
        steps = row["step_outcomes"]
        if not isinstance(steps, list) or not steps:
            raise ValueError("Missing step outcomes")
        result["other_applied_steps"] = [s["step_name"] for s in steps
                                         if s["step_name"] != config["step_name"] and s["status"] == "applied"]
        if result["original_baseline"]:
            if row.get("baseline_equivalent") is not True:
                raise ValueError("Original baseline is not recorded as baseline-equivalent")
            if any(s["status"] != "no_change" or s["operation"] != "none"
                   or s["added_voxels"] != 0 or s["removed_voxels"] != 0 for s in steps):
                raise ValueError("Original baseline contains an edit")
            result.update(coa_edit_applied=False, coa_benchmark_positive=False,
                          class_description="reference_non_coa_assumed")
            result[SCORE_NAME] = 0.0
            result["reference_semantics"] = "Original phantom: zero induced CoA change; non-CoA assumed by study design."
            return result
        selected = [s for s in steps if s["step_name"] == config["step_name"]]
        if len(selected) != 1 or selected[0] is not steps[-1] or selected[0]["iteration"] != 1:
            raise ValueError("Requires exactly one CoA pass, iteration 1, as the final recipe step")
        step = selected[0]
        if step["operation"] != "erosion" or step["status"] != "applied":
            raise ValueError("Nonbaseline CoA step must be a successfully applied erosion")
        result["coa_edit_applied"] = True
        guard = step.get("erosion_safeguard")
        if not isinstance(guard, dict) or not all(guard.get(k) is True for k in ("enabled", "accepted", "connectivity_ok")):
            raise ValueError("Missing or unaccepted erosion safeguard metadata")
        if guard.get("status") != "accepted":
            raise ValueError("Erosion safeguard status is not accepted")
        before = exact_count(guard["baseline_target_voxels"], "baseline_target_voxels")
        after = exact_count(guard["retained_target_voxels"], "retained_target_voxels")
        removed = exact_count(step["removed_voxels"], "removed_voxels")
        if before <= 0 or not 0 <= after < before or removed != before - after or step["added_voxels"] != 0:
            raise ValueError("CoA loss is inconsistent with local pre-step/retained target counts")
        spacing = row["spacing_ijk_mm"]
        if len(spacing) != 3 or any(not math.isfinite(x) or x <= 0 for x in spacing):
            raise ValueError("Invalid spacing_ijk_mm")
        voxel_mm3 = math.prod(spacing)
        close(guard["retained_volume_ratio"], after / before, "retained_volume_ratio")
        close(guard["baseline_volume_mm3"], before * voxel_mm3, "baseline_volume_mm3")
        close(guard["retained_volume_mm3"], after * voxel_mm3, "retained_volume_mm3")
        scope, reference = guard["scope_semantics"], guard["reference_semantics"]
        if "positive longitudinal profile support" not in scope or "immediately before this edit step" not in reference:
            raise ValueError("Unrecognized local reference/scope semantics")
        for field in ("requested_distance_mm", "accepted_distance_mm"):
            close(guard[field], step[field], field)
            if step[field] <= 0:
                raise ValueError(f"{field} must be positive")
            result[field] = step[field]
        attempts = guard.get("attempts", [])
        if not attempts or attempts[-1].get("accepted") is not True:
            raise ValueError("Missing final accepted safeguard attempt")
        last = attempts[-1]
        close(last["distance_mm"], step["accepted_distance_mm"], "final attempt distance")
        close(last["retained_volume_ratio"], after / before, "final attempt retained ratio")
        if last["retained_target_voxels"] != after or last.get("connectivity_ok") is not True:
            raise ValueError("Final accepted safeguard attempt disagrees with result")
        loss = removed / before
        positive = loss >= config["positive_loss_fraction"]
        result.update(coa_benchmark_positive=positive,
                      class_description="above_threshold_induced_coa" if positive else "below_threshold_induced_coa",
                      baseline_target_voxels=before, retained_target_voxels=after,
                      coa_removed_voxels=removed, baseline_volume_mm3=guard["baseline_volume_mm3"],
                      retained_volume_mm3=guard["retained_volume_mm3"],
                      scope_semantics=scope, reference_semantics=reference,
                      review_near_threshold=abs(loss - config["positive_loss_fraction"]) <= config["review_half_width"] + 1e-12)
        result[SCORE_NAME] = loss
    except (ValueError, KeyError, TypeError) as exc:
        result["errors"].append(str(exc))
    return result


def verified_rows(config: dict) -> tuple[list, list]:
    """Verify the signed input and copied outcome metadata against each frozen parent."""
    path = Path(config["dataset_manifest"])
    sidecar = Path(str(path) + ".sha256")
    provenance = [checked_hash(path, sidecar.read_text().strip().split()[0])]
    data = json.loads(path.read_text())
    if data.get("complete") is not True or not data.get("samples"):
        raise ValueError("Source dataset is not complete or has no samples")
    rows = data["samples"]
    if len({r["sample_id"] for r in rows}) != len(rows):
        raise ValueError("Duplicate source sample IDs")
    if len(rows) != data["sample_count"]:
        raise ValueError("Source sample count does not match manifest")
    provenance.append(checked_hash(Path(data["source_manifest"]), data["source_manifest_sha256"]))
    groups = defaultdict(list)
    for row in rows:
        groups[row["prepared_manifest"]].append(row)
    for source_path, group in sorted(groups.items()):
        expected = {r["prepared_manifest_sha256"] for r in group}
        if len(expected) != 1:
            raise ValueError("Inconsistent parent manifest hashes")
        source = Path(source_path)
        provenance.append(checked_hash(source, expected.pop()))
        parent = json.loads(source.read_text())
        if parent.get("complete") is not True:
            raise ValueError(f"Incomplete prepared manifest: {source}")
        samples = {r["variant_id"]: r for r in parent["samples"]}
        for row in group:
            original = samples.get(row["variant_id"])
            if not original or original["case_id"] != row["case_id"]:
                raise ValueError(f"Missing parent sample: {row['sample_id']}")
            if original["step_outcomes"] != row["step_outcomes"] or original["parameters"] != row["parameters"]:
                raise ValueError(f"Copied edit metadata disagrees with parent: {row['sample_id']}")
            if original["geometry"]["spacing_ijk_mm"] != row["spacing_ijk_mm"]:
                raise ValueError(f"Copied spacing disagrees with parent: {row['sample_id']}")
    cases = defaultdict(list)
    for row in rows:
        cases[row["case_id"]].append(row)
    if any(sum(r.get("baseline") is True for r in group) != 1 for group in cases.values()):
        raise ValueError("Each case must contain exactly one original baseline")
    return rows, provenance


def summarize(rows: list, config: dict) -> dict:
    def counts(group):
        return {
            "samples": len(group),
            "benchmark_positive": sum(r["coa_benchmark_positive"] is True for r in group),
            "benchmark_negative": sum(r["coa_benchmark_positive"] is False for r in group),
            "unknown": sum(r["coa_benchmark_positive"] is None for r in group),
            "original_reference": sum(r["original_baseline"] for r in group),
            "below_threshold_edited": sum(r["class_description"] == "below_threshold_induced_coa" for r in group),
            "review_near_threshold": sum(r["review_near_threshold"] for r in group),
        }
    by_case = defaultdict(list)
    for row in rows:
        by_case[row["case_id"]].append(row)
    values = sorted(r[SCORE_NAME] for r in rows if r["coa_edit_applied"] and r[SCORE_NAME] is not None)
    n = len(values)
    median = (values[(n - 1) // 2] + values[n // 2]) / 2 if n else None
    sensitivity = []
    for threshold in config["sensitivity_thresholds"]:
        known = [r for r in rows if r[SCORE_NAME] is not None]
        positive = sum(not r["original_baseline"] and r[SCORE_NAME] >= threshold for r in known)
        sensitivity.append({"threshold": threshold, "benchmark_positive": positive,
                            "benchmark_negative": len(known) - positive, "unknown": len(rows) - len(known)})
    return {**counts(rows), "case_count": len(by_case),
            "edited_score_range": [values[0], values[-1]] if n else None,
            "edited_score_median": median,
            "per_case": {case: counts(group) for case, group in sorted(by_case.items())},
            "threshold_sensitivity": sensitivity}


def histogram_svg(rows, threshold):
    bins = [0] * 20
    for row in rows:
        value = row[SCORE_NAME]
        if row["coa_edit_applied"] and value is not None:
            bins[min(19, int(value * 20))] += 1
    top = max(bins) or 1
    bars = []
    for index, count in enumerate(bins):
        height = 150 * count / top
        x = 45 + index * 30
        bars.append(f'<rect x="{x}" y="{180-height:.2f}" width="28" height="{height:.2f}" fill="#2463a6"><title>{index*5}–{(index+1)*5}%: {count} siblings</title></rect>')
        if count:
            bars.append(f'<text x="{x+14}" y="{175-height:.2f}" text-anchor="middle" font-size="12">{count}</text>')
    tx = 45 + threshold * 600
    ticks = ''.join(f'<text x="{45+x*6}" y="201" text-anchor="middle" font-size="12">{x}%</text>' for x in range(0,101,10))
    return (f'<svg viewBox="0 0 700 235" role="img" aria-label="Edited-sibling local volume loss histogram">'
            f'{"".join(bars)}<line x1="45" y1="180" x2="645" y2="180" stroke="#444"/>'
            f'<line x1="{tx}" x2="{tx}" y1="20" y2="181" stroke="#c73535" stroke-width="2"/>'
            f'<text x="{tx+5}" y="15" fill="#a11">Threshold {threshold:.0%}</text>{ticks}'
            '<text x="350" y="228" text-anchor="middle" font-size="13">Local target volume loss from the CoA step</text></svg>')


def write_html(path: Path, document: dict) -> None:
    summary, config, rows = document["summary"], document["definition"], document["samples"]
    esc = lambda x: html.escape(str(x))
    table = lambda headers, values: '<table><thead><tr>' + ''.join(f'<th>{esc(h)}</th>' for h in headers) + '</tr></thead><tbody>' + ''.join('<tr>' + ''.join(f'<td>{esc(v)}</td>' for v in row) + '</tr>' for row in values) + '</tbody></table>'
    cases = table(["Case", "Positive", "Below-threshold edited", "Reference", "Review band", "Unknown"],
                  [[case, c["benchmark_positive"], c["below_threshold_edited"], c["original_reference"], c["review_near_threshold"], c["unknown"]] for case,c in summary["per_case"].items()])
    sensitivity = table(["Volume-loss threshold", "Positive", "Negative", "Unknown"],
                        [[f'{r["threshold"]:.0%}', r["benchmark_positive"], r["benchmark_negative"], r["unknown"]] for r in summary["threshold_sensitivity"]])
    sample_table = table(["Sample", "Loss", "Classification", "Review band", "CoA removed/pre-step voxels", "Errors"],
                         [[r["sample_id"], f'{r[SCORE_NAME]:.2%}' if r[SCORE_NAME] is not None else "unknown",
                           r["class_description"], "yes" if r["review_near_threshold"] else "",
                           f'{r["coa_removed_voxels"]}/{r["baseline_target_voxels"]}' if r["baseline_target_voxels"] is not None else "reference",
                           "; ".join(r["errors"])] for r in rows])
    limitations = ''.join(f'<li>{esc(v)}</li>' for v in document["limitations"])
    path.write_text(f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Draft CoA cohort label review</title><style>body{{font:16px system-ui,sans-serif;color:#1e293b;max-width:1120px;margin:35px auto;padding:0 20px;line-height:1.5}}h1,h2{{line-height:1.2}}.notice{{background:#fff0cf;border-left:5px solid #c88800;padding:15px}}table{{border-collapse:collapse;width:100%;margin:16px 0;font-size:14px}}td,th{{padding:8px;border-bottom:1px solid #d5dce4;text-align:left}}th{{background:#eaf0f7}}.scroll{{overflow:auto}}svg{{max-width:780px;width:100%}}code{{background:#eee;padding:2px 4px}}</style>
<h1>Draft CoA cohort label review</h1><p class="notice"><strong>Engineering proxy — review required before training.</strong> This threshold is not a clinical CoA criterion. Benchmark-negative edited siblings are not necessarily healthy.</p>
<p>{summary['samples']} samples from {summary['case_count']} phantoms: <strong>{summary['benchmark_positive']} benchmark-positive</strong>, {summary['benchmark_negative']} benchmark-negative ({summary['original_reference']} original references + {summary['below_threshold_edited']} below-threshold edited siblings), {summary['unknown']} unknown. {summary['review_near_threshold']} samples lie within the review band.</p>
<h2>Definition</h2><p>Positive if <code>removed target voxels / pre-CoA target voxels ≥ {config['positive_loss_fraction']:.0%}</code>. The denominator is the pre-step target in the active longitudinal support, after earlier edits. The full selected cross-section is counted before angular weighting. CoA must be a single final erosion with accepted safeguards and consistent recorded counts.</p>
<p>Review band: {config['positive_loss_fraction']-config['review_half_width']:.0%}–{config['positive_loss_fraction']+config['review_half_width']:.0%}. These samples retain their provisional class and are flagged for review; they are not silently excluded. Original references have score zero by the explicit study assumption.</p>
{histogram_svg(rows, config['positive_loss_fraction'])}<p>The histogram contains edited siblings only; original references are excluded.</p>
<h2>Threshold sensitivity</h2>{sensitivity}<h2>Per-phantom counts</h2><div class="scroll">{cases}</div>
<h2>Interpretation and limitations</h2><ul>{limitations}</ul>
<details><summary>All sample labels and audit status</summary><div class="scroll">{sample_table}</div></details>
<h2>Provenance</h2><p>Generated {esc(document['created_utc'])}. Source and configuration hashes, audited parent manifests, exact definitions and sample labels are in <a href="labels.json">labels.json</a>. Download <a href="labels.csv">CSV</a> or inspect the <a href="input.ini">input INI</a>. No frozen dataset or image payload was modified.</p></html>''')


def run(config: dict, validate_only=False) -> dict:
    output = Path(config["output_directory"])
    if output.exists() and not validate_only:
        raise ValueError(f"Output already exists; use a new versioned directory: {output}")
    rows, provenance = verified_rows(config)
    labels = [classify_sample(row, config) for row in rows]
    document = {"schema_version": SCHEMA, "status": "draft", "ready_for_training": False,
                "created_utc": datetime.now(timezone.utc).isoformat(),
                "definition": config, "score_name": SCORE_NAME,
                "summary": summarize(labels, config), "limitations": LIMITATIONS,
                "source_provenance": provenance,
                "code": {"path": str(Path(__file__).resolve()), "sha256": sha256(Path(__file__))},
                "samples": labels}
    if validate_only:
        return document
    output.mkdir(parents=True, exist_ok=False)
    marker = output / "INCOMPLETE"
    marker.write_text("Draft label export is being written.\n")
    shutil.copyfile(config["config_path"], output / "input.ini")
    json_path = output / "labels.json"
    json_path.write_text(json.dumps(document, indent=2, allow_nan=False) + "\n")
    fields = [k for k in labels[0] if k not in {"errors", "other_applied_steps", "scope_semantics", "reference_semantics"}]
    fields += ["other_applied_steps", "errors"]
    with (output / "labels.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for label in labels:
            values = {k: label[k] for k in fields}
            values["other_applied_steps"] = ";".join(values["other_applied_steps"])
            values["errors"] = ";".join(values["errors"])
            writer.writerow(values)
    write_html(output / "report.html", document)
    for name in ("labels.json", "labels.csv", "report.html", "input.ini"):
        path = output / name
        (output / (name + ".sha256")).write_text(sha256(path) + "\n")
    marker.unlink()
    return document


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--validate-only", action="store_true", help="Audit provenance/counts and print draft counts without writing files")
    args = parser.parse_args(argv)
    try:
        result = run(load_config(args.config), args.validate_only)
        print(json.dumps({"status": result["status"], "ready_for_training": False,
                          "summary": result["summary"],
                          "output_directory": None if args.validate_only else result["definition"]["output_directory"]}, indent=2))
        return 2 if result["summary"]["unknown"] else 0
    except (OSError, ValueError, KeyError, configparser.Error) as exc:
        print(f"error: {exc}", file=sys.stderr)
        return 2


if __name__ == "__main__":
    raise SystemExit(main())
