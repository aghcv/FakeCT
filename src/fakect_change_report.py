"""Saved evaluation of CT-only prediction of net original-to-sibling changes.

Classes are unchanged=0, added=1 and removed=2. Absent-target overlap scores
are undefined (None), with false positives reported separately. Reports never
fit a model, select an epoch or modify inputs, weights or saved measurements.
"""
from collections import defaultdict
import hashlib
from html import escape
import json
import math
from pathlib import Path

import numpy as np

CLASSES = {"unchanged": 0, "added": 1, "removed": 2}
_SCORES = ("dice", "iou", "precision", "recall")


def _grid(value, name, shape=None, binary=False):
    array = np.asarray(value)
    if array.ndim != 3 or not array.size or (shape is not None and array.shape != shape):
        raise ValueError(name + " must use the same nonempty three-dimensional grid")
    if not np.isfinite(array).all() or not np.isin(array, (0, 1) if binary else (0, 1, 2)).all():
        raise ValueError(name + " contains invalid " + ("binary validity" if binary else "class labels"))
    return array


def _overlap(tp, fp, fn, target):
    if target == 0:
        return dict.fromkeys(_SCORES)
    return {"dice": 2 * tp / (2 * tp + fp + fn),
            "iou": tp / (tp + fp + fn),
            "precision": tp / (tp + fp) if tp + fp else 0.,
            "recall": tp / target}


def change_metrics(labels, predicted, valid, spacing_ijk_mm):
    """Count three-class predictions on valid voxels only, in native mm units.

    Labels and predictions are categorical grids, not soft probabilities. No
    scores are defined when a class has no true voxels; empty-class false
    positives remain visible. A no-op is a valid crop with neither change class.
    """
    labels = _grid(labels, "labels")
    predicted = _grid(predicted, "predicted", labels.shape)
    valid = _grid(valid, "valid", labels.shape, binary=True).astype(bool)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("spacing_ijk_mm requires three finite positive values")
    volume = float(np.prod(spacing))
    count = int(valid.sum())
    classes = {}
    for name, label in CLASSES.items():
        truth, estimate = (labels == label) & valid, (predicted == label) & valid
        target, n_predicted = int(truth.sum()), int(estimate.sum())
        tp = int(np.count_nonzero(truth & estimate))
        fp, fn = n_predicted - tp, target - tp
        classes[name] = dict(label=label, target_voxels=target,
            predicted_voxels=n_predicted, tp=tp, fp=fp, fn=fn,
            target_empty=target == 0, target_mm3=target * volume,
            predicted_mm3=n_predicted * volume, false_positive_mm3=fp * volume,
            false_negative_mm3=fn * volume, **_overlap(tp, fp, fn, target))
    no_op = count > 0 and not (classes["added"]["target_voxels"] + classes["removed"]["target_voxels"])
    false_changes = int(np.count_nonzero((predicted != 0) & valid)) if no_op else None
    return dict(valid_voxels=count, spacing_ijk_mm=spacing.tolist(), voxel_volume_mm3=volume,
        classes=classes, no_op_control=no_op, no_op_false_positive_voxels=false_changes,
        no_op_false_positive_mm3=false_changes * volume if no_op else None)


def _mean(values):
    values = [float(v) for v in values if v is not None and math.isfinite(float(v))]
    return float(np.mean(values)) if values else None


def _macro(items):
    return {key: _mean(entry.get(key) for entry in items) for key in _SCORES}


def _group_macro(rows, name, key):
    groups = defaultdict(list)
    for row in rows:
        identity = row.get(key) or (row.get("family") if key == "anatomy_family" else None)
        if identity is None:
            identity = row.get("case_id") or row.get("sample_id")
        groups[str(identity)].append(row["metrics"]["classes"][name])
    group_scores = [_macro(members) for members in groups.values()]
    return _macro(group_scores), len(groups)


def summarize(records):
    """Descriptive sample/case/family summaries; siblings are not independent.

    Record contract: sample_id, optional case_id/anatomy_family/is_control and
    metrics from change_metrics. Family means first average siblings within a
    case, then cases within a family; empty-target scores are excluded.
    """
    rows = list(records)
    ids = [row.get("sample_id") for row in rows]
    identities = [(row.get("sample_id"), row.get("proposal_index", row.get("focus_index"))) for row in rows]
    if any(not isinstance(sid, str) or not sid for sid in ids) or len(set(identities)) != len(identities):
        raise ValueError("Records require unique sample_id/proposal_index identities")
    for row in rows:
        if not isinstance(row.get("metrics"), dict) or not all(
                name in row["metrics"].get("classes", {}) for name in CLASSES):
            raise ValueError("Every record requires change_metrics classes")
    result = dict(record_count=len(rows), scenario_count=len(set(ids)), proposal_count=len(rows),
        case_count=len({row.get("case_id", row["sample_id"]) for row in rows}),
        family_count=len({row.get("anatomy_family", row.get("family", row.get("case_id", row["sample_id"]))) for row in rows}),
        empty_target_policy="Undefined overlap scores; report false positive volume separately.",
        scope="Descriptive crop results; proposals can overlap and siblings are correlated. Pooled crop-voxel counts may count the same physical voxel twice; proposals are not independent patients.", classes={})
    for name in CLASSES:
        entries = [row["metrics"]["classes"][name] for row in rows]
        present = [entry for entry in entries if not entry["target_empty"]]
        absent = [entry for entry in entries if entry["target_empty"]]
        counts = {key: sum(int(entry[key]) for entry in entries)
                  for key in ("tp", "fp", "fn", "target_voxels", "predicted_voxels")}
        case_macro, _ = _group_macro(rows, name, "case_id")
        families = defaultdict(lambda: defaultdict(list))
        for row in rows:
            case = row.get("case_id", row["sample_id"])
            family = row.get("anatomy_family", row.get("family", case))
            families[str(family)][str(case)].append(row["metrics"]["classes"][name])
        family_scores = [_macro([_macro(members) for members in cases.values()]) for cases in families.values()]
        result["classes"][name] = dict(nonempty_target_records=len(present), empty_target_records=len(absent),
            nonempty_sample_macro=_macro(present), case_macro=case_macro,
            family_macro=_macro(family_scores), pooled_counts={**counts,
                **_overlap(counts["tp"], counts["fp"], counts["fn"], counts["target_voxels"]),
                "false_positive_mm3": sum(entry["false_positive_mm3"] for entry in entries)},
            empty_target=dict(records=len(absent), false_positive_voxels=sum(entry["fp"] for entry in absent),
                false_positive_mm3=sum(entry["false_positive_mm3"] for entry in absent),
                mean_false_positive_mm3=_mean(entry["false_positive_mm3"] for entry in absent)))
    controls = [row for row in rows if row.get("is_control", row.get("baseline", row["metrics"].get("no_op_control", False)))]
    control_volumes = [sum(row["metrics"]["classes"][name]["false_positive_mm3"]
                          for name in ("added", "removed")) for row in controls]
    result["no_op_controls"] = dict(records=len(controls), scenarios=len({row["sample_id"] for row in controls}), total_false_positive_mm3=sum(control_volumes),
        mean_false_positive_mm3=_mean(control_volumes),
        zero_false_positive_records=sum(value == 0 for value in control_volumes))
    return result


def _number(value):
    return "—" if value is None else f"{value:.3f}"


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _prediction_path(directory, row):
    value = row.get("prediction_file") or row.get("prediction_npz")
    if not isinstance(value, str) or not value:
        return None
    relative = Path(value)
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("Prediction paths must remain inside the run directory")
    path = (directory / relative).resolve()
    if directory.resolve() not in path.parents or not path.is_file():
        raise ValueError("Saved prediction path is unavailable or escapes the run")
    expected = row.get("prediction_sha256") or row.get("prediction_npz_sha256")
    if expected and _hash(path) != expected:
        raise ValueError("Saved prediction checksum changed")
    return path


def _scene(path, row):
    from fakect_two_stage_3d import build_scene
    with np.load(path, allow_pickle=False) as data:
        if "sample_id" in data and str(data["sample_id"].item()) != row["sample_id"]:
            raise ValueError("Saved prediction belongs to another sample")
        labels = _grid(data["labels"], "labels")
        predicted = _grid(data["predicted"], "predicted", labels.shape)
        valid = _grid(data["valid"], "valid", labels.shape, binary=True).astype(bool)
        spacing = data["spacing_ijk_mm"] if "spacing_ijk_mm" in data else row["metrics"]["spacing_ijk_mm"]
        masks = dict(added=(labels == 1) & valid, removed=(labels == 2) & valid,
                     predicted_added=(predicted == 1) & valid, predicted_removed=(predicted == 2) & valid)
        for key in ("before", "after"):
            if key in data:
                masks[key] = _grid(data[key], key, labels.shape, binary=True).astype(bool) & valid
        if "before" in masks and "after" in masks:
            if not np.array_equal(masks["added"], masks["after"] & ~masks["before"]) or not np.array_equal(
                    masks["removed"], masks["before"] & ~masks["after"]):
                raise ValueError("Change labels disagree with original and sibling masks")
    styles = {"before": ("Original baseline · truth", "#7187a3", False, .15),
              "after": ("Sibling target · truth", "#6f7785", True, .15),
              "added": ("Net enlargement · truth", "#e41a80", True, .8),
              "removed": ("Net narrowing · truth", "#147df5", True, .8),
              "predicted_added": ("Net enlargement · predicted", "#f4b400", True, .8),
              "predicted_removed": ("Net narrowing · predicted", "#18a572", True, .8)}
    layers = [dict(key=key, label=styles[key][0], color=styles[key][1], visible=styles[key][2], opacity=styles[key][3])
              for key in ("before", "after", "added", "removed", "predicted_added", "predicted_removed") if key in masks]
    display_id = row["sample_id"]
    if "proposal_index" in row:
        display_id += " · focus " + str(row["proposal_index"])
    return build_scene(masks, spacing, sample_id=display_id,
        modes=[dict(key="net_changes", label="Original-to-sibling net changes",
                    explanation="CT-only predictions and exact net truth; classes combine every recipe edit. Original and sibling masks are display truth, not model inputs.", layers=layers)],
        metadata=dict(case_id=row.get("case_id"), is_control=row.get("is_control", row.get("baseline", row["metrics"].get("no_op_control")))),
        show_metrics=False, footer_note="Exact saved valid-grid surfaces. No smoothing or pooling. Meshes over the display budget are explicitly omitted; saved scores retain all valid voxels. Changes refer to the original baseline and are not condition-specific or clinically validated.")


def _history_plot(directory, document):
    import matplotlib
    matplotlib.use("Agg")
    from matplotlib import pyplot as plt
    history = document.get("history")
    source = None
    if history is None:
        for candidate in (directory / "history.json", directory / "model" / "history.json"):
            if candidate.is_file():
                history, source = json.loads(candidate.read_text()), candidate
                break
    if isinstance(history, dict):
        history = history.get("epochs", history.get("records", history.get("history", [])))
    if not isinstance(history, list) or not history:
        return None, source
    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4))
    fields = ((axes[0], ("train_loss", "validation_loss", "loss", "val_loss")),
              (axes[1], ("validation_added_dice", "validation_removed_dice", "train_added_dice", "train_removed_dice", "dice_loss")))
    for ax, keys in fields:
        for key in keys:
            pairs = [(row.get("epoch", i + 1), row.get(key)) for i, row in enumerate(history) if isinstance(row, dict)]
            pairs = [(float(x), float(y)) for x, y in pairs if isinstance(x, (float, int)) and isinstance(y, (float, int)) and math.isfinite(x) and math.isfinite(y)]
            if pairs:
                ax.plot(*zip(*pairs), label=key.replace("_", " "))
        if ax.lines:
            ax.legend(frameon=False, fontsize=8)
        else:
            ax.text(.5, .5, "No saved values", ha="center", va="center", transform=ax.transAxes)
        ax.set_xlabel("Epoch"); ax.grid(alpha=.2)
    axes[0].set_ylabel("Training / validation loss")
    axes[1].set_ylabel("Change Dice / Dice loss"); axes[1].set_ylim(0, 1.02)
    fig.tight_layout()
    path = directory / "change-learning.png"
    fig.savefig(path, dpi=150, facecolor="white"); plt.close(fig)
    return path, source


def write_report(run_directory):
    """Render saved results.json without changing completed measurements.

    Top-level records (or evaluation.records) contain metrics and optional
    relative prediction_file/sha256. NPZ labels,predicted,valid and optional
    before/after use one shared grid; spacing_ijk_mm can be saved or in metrics.
    A three-class model predicts all net edits, without condition labels.
    """
    directory = Path(run_directory).resolve()
    source = directory / "results.json"
    document = json.loads(source.read_text())
    evaluation = document.get("evaluation", document)
    records = evaluation.get("records", evaluation.get("predictions", []))
    if not isinstance(records, list):
        raise ValueError("Saved evaluation records must be a list")
    summary = summarize(records)
    history, history_source = _history_plot(directory, document)
    paths = [(row, _prediction_path(directory, row)) for row in records]
    paths = [(row, path) for row, path in paths if path is not None]
    viewer = None
    if paths:
        from fakect_two_stage_3d import render_viewer
        viewer = render_viewer(directory / "report-3d", (_scene(path, row) for row, path in paths),
            title="CT-only net enlargement and narrowing · saved predictions", show_metrics=False,
            footer_note="True baseline and sibling masks are shown for evaluation only. Every layer has independent visibility and opacity.")
    rows = []
    for name in ("added", "removed", "unchanged"):
        entry, macro = summary["classes"][name], summary["classes"][name]["family_macro"]
        rows.append("<tr><td>" + escape(name) + "</td>" + "".join("<td>" + _number(macro[key]) + "</td>" for key in _SCORES)
                    + f"<td>{entry['nonempty_target_records']}</td><td>{entry['empty_target_records']}</td><td>{entry['empty_target']['false_positive_mm3']:.3f}</td></tr>")
    per_sample = []
    for row in records:
        display_id = row["sample_id"] + (" · focus " + str(row["proposal_index"]) if "proposal_index" in row else "")
        values = [escape(str(display_id)), escape(str(row.get("case_id", ""))), escape(str(row.get("anatomy_family", row.get("family", ""))))]
        values += [_number(row["metrics"]["classes"][name]["dice"]) for name in ("added", "removed")]
        values.append(_number(row["metrics"].get("no_op_false_positive_mm3")))
        per_sample.append("<tr>" + "".join("<td>" + value + "</td>" for value in values) + "</tr>")
    controls = summary["no_op_controls"]
    warnings = "".join("<p>" + escape(str(value)) + "</p>" for value in document.get("warnings", []))
    document_html = ('<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>CT-only net change prediction</title><style>body{font:16px/1.5 system-ui,sans-serif;background:#f5f7fa;color:#17212b;max-width:1250px;margin:2rem auto;padding:0 1.5rem}section{background:white;padding:1.2rem;margin:1rem 0;border:1px solid #d8dee4}table{border-collapse:collapse;width:100%;font-size:.88rem}th,td{padding:.5rem;border:1px solid #d8dee4;text-align:left}img{width:100%}iframe{width:100%;height:1100px;border:0}.table{overflow:auto}.notice{border-left:5px solid #d55e00}</style>'
        '<h1>CT-only prediction of net enlargement and narrowing</h1><section class="notice"><p>The model sees sibling attenuation and image validity. Original baseline and difference maps are training supervision and evaluation truth, never inference inputs. Targets include all final-versus-original edits, without condition-specific labels.</p>'
        '<p>Absent-target Dice, IoU, precision and recall are undefined and excluded from overlap means. Their false positive volume is reported separately. Family averages first average crop measurements within each case; siblings are correlated. Overlapping focus crops can count a physical voxel twice in pooled crop counts and are not independent patients. These are synthetic development results.</p>' + warnings + '</section>'
        f'<section><h2>Saved evaluation</h2><p>Status: {escape(str(document.get("status", "not recorded")))}. {summary["scenario_count"]} scenarios · {summary["proposal_count"]} focus crops · {summary["case_count"]} cases · {summary["family_count"]} anatomy families.</p>'
        '<div class="table"><table><tr><th>Class</th><th>Family Dice</th><th>IoU</th><th>Precision</th><th>Recall</th><th>Nonempty</th><th>Empty</th><th>Empty-target FP mm³</th></tr>' + ''.join(rows) + '</table></div>'
        f'<p>No-op controls: {controls["scenarios"]} scenarios, {controls["records"]} focus crops; mean false change volume {_number(controls["mean_false_positive_mm3"])} mm³ per crop; {controls["zero_false_positive_records"]} crops with zero false changes.</p><p>Saved gate: {escape(json.dumps(document.get("gate"), allow_nan=False))}. Classifier started: {escape(str(document.get("classifier_started", False)))}.</p></section>'
        + ('<section><h2>Learning history</h2><img src="change-learning.png" alt="Saved training and validation history"><p>Only recorded training and validation values are plotted. Test results do not select an epoch.</p></section>' if history else '')
        + ('<section><h2>Interactive 3D review</h2><p>Rotate and zoom; independently toggle original, sibling, true changes and predicted changes at 15%, 45% or 80% opacity. Keep report-3d beside this report when sharing.</p><a href="report-3d/index.html">Open 3D viewer</a><iframe src="report-3d/index.html" title="Registered change surfaces"></iframe></section>' if viewer else '<section><p>Saved prediction arrays were unavailable; this report shows recorded metrics.</p></section>')
        + '<section><h2>Per-scenario measurements</h2><div class="table"><table><tr><th>Sample</th><th>Case</th><th>Family</th><th>Enlargement Dice</th><th>Narrowing Dice</th><th>No-op FP mm³</th></tr>' + ''.join(per_sample) + '</table></div></section></html>')
    report = directory / "report.html"
    report.write_text(document_html, encoding="utf-8")
    summary_path = directory / "change-summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, allow_nan=False) + "\n")
    inputs = {"results.json": _hash(source), **{str(path.relative_to(directory)): _hash(path) for _, path in paths}}
    if history_source:
        inputs[str(history_source.relative_to(directory))] = _hash(history_source)
    provenance = dict(schema_version="fakect.change-report-provenance/1", inputs_sha256=inputs,
                      reporter_sha256=_hash(__file__), output_summary_sha256=_hash(summary_path),
                      output_report_sha256=_hash(report), interactive_3d=viewer)
    (directory / "change-report-provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
    return dict(report=str(report), summary=str(summary_path), learning_plot=str(history) if history else None,
                interactive_3d=viewer, records=len(records))
