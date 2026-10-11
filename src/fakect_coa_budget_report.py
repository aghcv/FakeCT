"""Immutable, metadata-only reports of the preregistered 200-epoch control.

The baseline and control share the exact five development family panels. Only
the final classifier budget changes; the severity head and its loss remain.
No model, checkpoint, image, reserved prediction or intermediate outer prediction
is opened. ROC labels use exact empirical areas, not sampled mean-curve areas.
"""
from collections import defaultdict
import copy
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import fakect_coa_rotation_report as rotation


SCHEMA = "fakect.coa-budget-report/1"
BASELINE = "baseline"
CONTROL = "epoch200"
LOADED_MODULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
METRICS = (*rotation.METRICS, "severity_mae_percentage_points")
COLORS = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                     allow_nan=False).encode()).hexdigest()


def _json(path, fingerprints):
    """Require a checksum sidecar, and track both source files for export."""
    path = Path(path).resolve()
    sidecar = Path(str(path) + ".sha256")
    digest = _hash(path)
    if sidecar.read_text().strip() != digest:
        raise ValueError(f"Source checksum differs: {path}")
    fingerprints[str(path)] = digest
    fingerprints[str(sidecar)] = _hash(sidecar)
    return json.loads(path.read_text())


def corrected_mean_roc(per_family, protocol, points=1001):
    """Average complete ordered ROC coordinates; seeds first, then families.

    Retain repeated FPR coordinates before interpolation: deleting the lower
    endpoints of vertical jumps creates optimistic diagonal shortcuts. The
    finite-grid integral is descriptive and may differ from exact macro AUROC.
    """
    if type(points) is not int or points < 2:
        raise ValueError("ROC grid needs at least two points")
    grid = np.linspace(0., 1., points)
    groups = defaultdict(list)
    for row in per_family:
        if row["protocol"] == protocol and row.get("roc") is not None:
            groups[row["family"]].append(row)
    if not groups:
        return None
    curves, areas = [], []
    for family in sorted(groups):
        rows = groups[family]
        curves.append(np.mean([np.interp(grid, r["roc"]["fpr"], r["roc"]["tpr"])
                               for r in rows], axis=0))
        areas.append(math.fsum(r["auroc"] for r in rows) / len(rows))
    mean, sd = np.mean(curves, axis=0), np.std(curves, axis=0)
    return {"fpr_grid": grid.tolist(), "mean_tpr": mean.tolist(),
            "descriptive_family_sd_tpr": sd.tolist(), "families": len(groups),
            "exact_mean_family_auroc": math.fsum(areas) / len(areas),
            "finite_grid_integral": float(np.sum(np.diff(grid) * (mean[1:] + mean[:-1]) / 2)),
            "confidence_interval": None,
            "method": "Full ordered empirical coordinates, 1001-point linear interpolation; seeds averaged within each family, then families equally."}


def analyze_records(baseline, control, manifest):
    """Validate identical planned panels; compare only after all controls finish."""
    manifest = rotation.validate_manifest(manifest)
    paired_manifest = copy.deepcopy(manifest)
    paired_manifest["folds"] = [dict(f, protocol=protocol, selection_stages=[])
                               for protocol in (BASELINE, CONTROL) for f in manifest["folds"]]
    paired_manifest["stage_catalog"] = []
    records = [dict(row, protocol=protocol) for protocol, rows in
               ((BASELINE, baseline), (CONTROL, control)) for row in rows]
    analysis = rotation.aggregate_records(records, paired_manifest)
    expected = sum(len(f["evaluation_sample_ids"]) for f in manifest["folds"])
    if len(baseline) != expected:
        raise ValueError("The frozen baseline must contain every planned sample")
    for protocol in (BASELINE, CONTROL):
        summary = analysis["protocols"][protocol]
        summary["mean_roc"] = corrected_mean_roc(analysis["per_family"], protocol)
        # The frozen helper reports classification macro means. Add the retained
        # severity head using the same seed-within-family weighting.
        family_groups = defaultdict(list)
        for row in analysis["per_family"]:
            if row["protocol"] == protocol and row["severity_mae_percentage_points"] is not None:
                family_groups[row["family"]].append(row["severity_mae_percentage_points"])
        values = [math.fsum(v) / len(v) for v in family_groups.values()]
        summary["equal_family_metrics"]["severity_mae_percentage_points"] = {
            "mean": math.fsum(values) / len(values) if values else None,
            "descriptive_sd": float(np.std(values)) if values else None,
            "units": len(values), "confidence_interval": None}
    complete = analysis["protocols"][CONTROL]["completed_fold_runs"] == len(manifest["folds"])
    if complete:
        comparison = rotation.paired_comparison(analysis, BASELINE, CONTROL)
        left = {(r["seed"], r["family"]): r for r in analysis["per_family"] if r["protocol"] == BASELINE}
        severity_deltas = []
        for row in analysis["per_family"]:
            if row["protocol"] == CONTROL:
                old = left[(row["seed"], row["family"])]["severity_mae_percentage_points"]
                new = row["severity_mae_percentage_points"]
                severity_deltas.append({"seed": row["seed"], "family": row["family"],
                                        "delta_percentage_points": new - old if old is not None and new is not None else None})
        comparison["severity_proxy_family_deltas"] = severity_deltas
        analysis["paired_comparisons"] = [comparison]
    else:
        analysis["paired_comparisons"] = [{"status": "pending", "reason": "All five identical outer panels must complete before the full paired comparison.",
                                            "left_protocol": BASELINE, "right_protocol": CONTROL}]
    analysis.update(schema_version=SCHEMA, status="complete" if complete else "pending" if not control else "partial",
                    checkpoint_selection="Preregistered final epoch 200; no held-out checkpoint selection",
                    severity_scope="Descriptive synthetic local-volume-loss proxy; not clinically validated severity",
                    independent_observation_ci=None)
    return analysis


def checked_history(history, receipt, stage):
    """Accept training-only epoch evidence, never an outer selection history."""
    rows = history.get("epochs")
    if not isinstance(rows, list) or len(rows) != 200:
        raise ValueError("Completed control requires all 200 training epochs")
    train_ids = set(stage["train_sample_ids"])
    if (receipt.get("selected_epoch") != 200 or receipt.get("epochs_completed") != 200 or
            receipt.get("validation_sample_ids") != [] or receipt.get("validation") is not None or
            set(receipt.get("training_sample_ids", [])) != train_ids or
            history.get("selected_epoch") != 200 or
            history.get("checkpoint_selection") != "final fixed-budget epoch"):
        raise ValueError("Control receipt violates the train-only final-epoch contract")
    if len(receipt.get("training_sample_ids", [])) != len(train_ids):
        raise ValueError("Repeated classifier training sample IDs")
    output = []
    pairs_per_epoch = len(train_ids) // 2
    for epoch, record in enumerate(rows, 1):
        if (record.get("epoch") != epoch or record.get("validation") is not None or
                record.get("selection_bce") is not None or
                record.get("optimizer_updates") != epoch * pairs_per_epoch):
            raise ValueError("History contains selection metrics or an inconsistent epoch/update count")
        flattened = [v for pair in record.get("training_pair_order", []) for v in pair]
        if len(flattened) != len(train_ids) or set(flattened) != train_ids:
            raise ValueError("Training history exposes a held-out or missing sample")
        train = record.get("training", {})
        exported = {"stage_id": stage["id"], "family": stage["outer_fold_id"], "epoch": epoch,
                    "bce": train.get("bce"), "accuracy": train.get("accuracy"),
                    "severity_mae_percentage_points": 100 * train["score_mae"] if train.get("score_mae") is not None else None,
                    "optimizer_updates": record["optimizer_updates"]}
        for field in ("bce", "accuracy", "severity_mae_percentage_points"):
            value = exported[field]
            if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value) or value < 0:
                raise ValueError("Training history contains invalid metric values")
        if exported["accuracy"] > 1 or exported["severity_mae_percentage_points"] > 100:
            raise ValueError("Training history metric is outside its permitted range")
        output.append(exported)
    if receipt.get("optimizer_updates") != 200 * pairs_per_epoch:
        raise ValueError("Control optimizer-update count differs from its training panel")
    return output


def _completed(row, stage, lock_hash):
    if (row.get("status") != "completed" or row.get("stage_id") != stage["id"] or
            row.get("protocol") != "family_holdout" or row.get("source_lock_sha256") != lock_hash or
            row.get("final_test_evaluated") is not True or row.get("reserved_test_evaluated") is not False):
        raise ValueError("Completed result identity or evaluation contract differs from its frozen lock")
    predictions = row.get("final_predictions")
    if not isinstance(predictions, list):
        raise ValueError("Completed result requires its exact final prediction panel")
    for pred in predictions:
        if (pred.get("protocol") != stage["protocol"] or pred.get("fold_id") != stage["outer_fold_id"] or
                pred.get("seed") != stage["seed"] or pred.get("score_prediction") is None):
            raise ValueError("Prediction identifies another stage or seed")
    return predictions


def source_bundle(output_root, lock):
    """Read only checksum-checked metadata, receipts, training metrics and OOF rows."""
    root = Path(output_root).resolve()
    if (root / "INCOMPLETE").exists():
        raise ValueError("The control experiment lock is incomplete")
    fingerprints = {}
    actual = _json(root / "experiment-lock.json", fingerprints)
    if actual != lock:
        raise ValueError("Passed lock differs from the frozen experiment-lock.json")
    own_hash = fingerprints[str(root / "experiment-lock.json")]
    parent = Path(lock["parent_directory"])
    if not parent.is_absolute() or parent.resolve() == root:
        raise ValueError("A separate absolute parent_directory is required")
    parent = parent.resolve()
    old = _json(parent / "experiment-lock.json", fingerprints)
    if fingerprints[str(parent / "experiment-lock.json")] != lock["parent_lock_sha256"]:
        raise ValueError("Parent lock differs from its pinned checksum")
    training = lock["config"]["training"]
    unchanged = lock.get("unchanged_training_settings", {})
    interval = training.get("checkpoint_interval")
    if (training.get("classifier_epochs") != 200 or type(interval) is not int or
            not 1 <= interval <= 200 or 200 % interval or
            unchanged.get("regression_weight") != 1.0 or unchanged != old["config"]["training"]):
        raise ValueError("This report requires the 200-epoch budget-only control with regression_weight=1")
    stages = lock["stages"]
    old_stages = {s["id"]: s for s in old["stages"] if s["stage"] == "outer_refit" and s["protocol"] == "family_holdout"}
    if len(stages) != 5 or {s["id"] for s in stages} != set(old_stages):
        raise ValueError("Exactly the five original primary outer-refit stages are required")
    for stage in stages:
        changed_policy = {"locator_policy", "classifier_policy", "checkpoint_policy", "parent_inner_stage_ids",
                          "expected_initial_weights_sha256"}
        if ({k: v for k, v in stage.items() if k not in changed_policy} !=
                {k: v for k, v in old_stages[stage["id"]].items() if k not in changed_policy} or
                stage.get("parent_inner_stage_ids") != []):
            raise ValueError("Control outer-stage metadata differs from the frozen baseline")
    if lock["reserved_case_ids"] != old["reserved_case_ids"]:
        raise ValueError("Reserved case identities changed")
    derived = rotation.rotation_manifest(old)
    derived["folds"] = [dict(f, selection_stages=[]) for f in derived["folds"] if f["protocol"] == "family_holdout"]
    derived["stage_catalog"] = [r for r in derived["stage_catalog"] if r["stage_id"] in old_stages]
    manifest = lock.get("fold_manifest", derived)
    # The fallback and supplied manifest must describe identical primary panels.
    checked = rotation.validate_manifest(manifest)
    expected_folds = {f["stage_id"]: f for f in derived["folds"]}
    if len(checked["folds"]) != 5 or {f.get("stage_id") for f in checked["folds"]} != set(expected_folds):
        raise ValueError("The report manifest does not contain the five original folds")
    for fold in checked["folds"]:
        if fold != expected_folds[fold["stage_id"]]:
            raise ValueError("Report fold panels differ from the original primary design")
    if (checked["family_by_case"] != derived["family_by_case"] or
            checked["sample_targets"] != derived["sample_targets"] or
            checked["class_threshold"] != derived["class_threshold"] or
            checked["reserved_case_ids"] != sorted(derived["reserved_case_ids"])):
        raise ValueError("Report labels, categories or threshold differ from the baseline")
    for fold in checked["folds"]:
        stage = next(s for s in stages if s["id"] == fold["stage_id"])
        if (fold["seed"] != 42 or len(fold["evaluation_case_ids"]) != 2 or
                len(fold["evaluation_sample_ids"]) != 12 or len(stage["train_sample_ids"]) != 48):
            raise ValueError("Budget report requires the original five seed-42 48/12 panels")
        for case in fold["evaluation_case_ids"]:
            targets = [checked["sample_targets"][sid] for sid in fold["evaluation_sample_ids"]
                       if checked["sample_targets"][sid]["case_id"] == case]
            if len(targets) != 6 or sum(r["label"] for r in targets) != 3:
                raise ValueError("Each evaluation case requires the same six balanced scenarios")
    baseline, control, histories, status = [], [], [], []
    for stage in stages:
        old_folder = parent / "stages" / stage["id"]
        if (old_folder / "INCOMPLETE").exists() or (old_folder / "failure.json").exists():
            raise ValueError("The pinned parent outer stage is incomplete or failed")
        old_path = old_folder / "results.json"
        old_result = _json(old_path, fingerprints)
        pinned = lock.get("dependency_sha256", {}).get(str(old_path))
        if pinned is None or pinned != fingerprints[str(old_path.resolve())]:
            raise ValueError("Parent result differs from the new lock's dependency pin")
        baseline.extend(_completed(old_result, stage, lock["parent_lock_sha256"]))
        if stage.get("expected_initial_weights_sha256") != old_result["classifier"].get("fresh_initial_weights_sha256"):
            raise ValueError("Control initialization differs from its baseline pin")
        folder = root / "stages" / stage["id"]
        failure = folder / "failure.json"
        result = folder / "results.json"
        if failure.exists():
            receipt = _json(failure, fingerprints)
            if result.exists() and not (folder / "INCOMPLETE").exists():
                raise ValueError("A completed control has a contradictory failure receipt")
            status.append({"stage_id": stage["id"], "family": stage["outer_fold_id"],
                           "status": "failed", "reason": receipt.get("message", receipt.get("reason", "See failure receipt"))})
        elif result.exists() and not (folder / "INCOMPLETE").exists():
            row = _json(result, fingerprints)
            control.extend(_completed(row, stage, own_hash))
            receipt = _json(folder / "classifier" / "receipt.json", fingerprints)
            if receipt != row["classifier"]:
                raise ValueError("Classifier receipt differs from the completed result")
            if (receipt.get("fresh_initial_weights_sha256") != stage["expected_initial_weights_sha256"] or
                    row.get("classifier_selected_epoch") != 200 or row.get("parent_lock_sha256") != lock["parent_lock_sha256"] or
                    row.get("locator_retrained") is not False or row.get("locator_proposals_changed") is not False):
                raise ValueError("Control changed initialization, locator proposals or its fixed budget")
            final = _json(folder / "final-test" / "predictions.json", fingerprints)
            if (final.get("predictions") != row["final_predictions"] or
                    final.get("checkpoint_sha256") != receipt.get("checkpoint_sha256") or
                    final.get("source_lock_sha256") != own_hash):
                raise ValueError("Final prediction artifact differs from the completed result")
            history = _json(folder / "classifier" / "history.json", fingerprints)
            histories.extend(checked_history(history, row["classifier"], stage))
            status.append({"stage_id": stage["id"], "family": stage["outer_fold_id"], "status": "completed"})
        else:
            marker = folder / "INCOMPLETE"
            if marker.exists():
                fingerprints[str(marker.resolve())] = _hash(marker)
            status.append({"stage_id": stage["id"], "family": stage["outer_fold_id"],
                           "status": "running" if folder.exists() else "pending"})
    analysis = analyze_records(baseline, control, checked)
    analysis["stage_status"] = status
    analysis["training_history"] = histories
    if any(s["status"] == "failed" for s in status):
        analysis["status"] = "partial_with_failed_stages" if control else "blocked_by_failed_stage"
    return {"analysis": analysis, "input_sha256": fingerprints, "source_lock_sha256": own_hash,
            "parent_lock_sha256": lock["parent_lock_sha256"], "parent_directory": str(parent),
            "reserved_case_ids": list(lock["reserved_case_ids"])}


def _write_json(path, value):
    path.write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def _csv(path, rows, fields):
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _figures(folder, analysis):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fakect-budget-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 13, "axes.titlesize": 15, "axes.labelsize": 14,
                         "legend.fontsize": 11, "svg.hashsalt": SCHEMA, "pdf.fonttype": 42,
                         "svg.fonttype": "none", "savefig.facecolor": "white"})
    figures = []

    def save(fig, stem, title):
        for suffix in ("pdf", "svg", "png"):
            metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else None
            fig.savefig(folder / (stem + "." + suffix), dpi=300, metadata=metadata)
        plt.close(fig)
        figures.append({"stem": stem, "title": title})

    labels = {BASELINE: "Baseline (8–29 epochs)", CONTROL: "Fixed 200 epochs"}
    histories = analysis["training_history"]
    if histories:
        fig, axes = plt.subplots(1, 3, figsize=(14, 4.9))
        fig.subplots_adjust(bottom=.26, top=.8, left=.07, right=.98, wspace=.3)
        for i, family in enumerate(sorted({r["family"] for r in histories})):
            rows = sorted((r for r in histories if r["family"] == family), key=lambda r: r["epoch"])
            for ax, metric in zip(axes, ("bce", "accuracy", "severity_mae_percentage_points")):
                ax.plot([r["epoch"] for r in rows], [r[metric] for r in rows], color=COLORS[i], label=family.replace("xcat_reference_", ""))
        for ax, title in zip(axes, ("Training BCE", "Training accuracy", "Training severity proxy MAE (pp)")):
            ax.set(xlabel="Epoch", title=title, xlim=(1, 200))
            ax.grid(alpha=.2)
        axes[0].set_yscale("log")
        axes[1].set_ylim(0, 1.03)
        axes[0].legend(title="Outer family", loc="best")
        fig.suptitle("Preregistered final epoch 200 · training curves only", fontsize=17)
        fig.text(.07, .1, "No held-out predictions select an epoch; milestones are archived without scoring.", fontsize=12)
        save(fig, "budget01-training", "Training-only classifier progression")

    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.1))
    fig.subplots_adjust(bottom=.29, top=.83, left=.075, right=.97, wspace=.29)
    mean_points = []
    for protocol, color in ((BASELINE, COLORS[0]), (CONTROL, COLORS[1])):
        summary = analysis["protocols"][protocol]
        metrics = summary["pooled_scenario_metrics"]
        if metrics is None:
            continue
        roc = metrics["roc"]
        axes[0].plot(roc["fpr"], roc["tpr"], color=color, lw=2.4,
                     label=f"{labels[protocol]}\nAUC {metrics['auroc']:.4f}; n={metrics['samples']}")
        mean = summary["mean_roc"]
        grid, curve, sd = (np.array(mean[f]) for f in ("fpr_grid", "mean_tpr", "descriptive_family_sd_tpr"))
        axes[1].plot(grid, curve, color=color, lw=2.4,
                     label=f"{labels[protocol]}\nMean exact AUC {mean['exact_mean_family_auroc']:.4f}; families={mean['families']}")
        axes[1].fill_between(grid, np.maximum(0, curve-sd), np.minimum(1, curve+sd), alpha=.12, color=color)
        mean_points.extend(dict(protocol=protocol, fpr=x, mean_tpr=y, descriptive_family_sd_tpr=z)
                           for x, y, z in zip(grid, curve, sd))
    for ax, title in zip(axes, ("Pooled empirical ROC", "Equal-family mean ROC")):
        ax.plot([0, 1], [0, 1], "--", color=".65", lw=1.2)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate", title=title)
        ax.grid(alpha=.2)
        ax.legend(loc="lower right")
    fig.suptitle("CoA development family cross-validation", fontsize=17)
    fig.text(.075, .14, "Band = descriptive family SD; siblings and seeds are correlated. No independent-patient CI.", fontsize=11.5)
    fig.text(.075, .085, "Mean labels use exact empirical areas. The 1001-point curve integral can differ slightly.", fontsize=11.5)
    if analysis["status"] != "complete":
        fig.text(.075, .035, "Control incomplete: available-fold curves describe a different panel; full paired comparison is pending.", fontsize=11)
    save(fig, "budget02-roc", "Exact pooled and corrected equal-family ROC")
    _csv(folder / "mean-roc-points.csv", mean_points, ["protocol", "fpr", "mean_tpr", "descriptive_family_sd_tpr"])

    if analysis["paired_comparisons"][0].get("status") != "pending":
        family_rows = analysis["per_family"]
        families = sorted({r["family"] for r in family_rows})
        fig, axes = plt.subplots(1, 3, figsize=(14, 6.1))
        fig.subplots_adjust(bottom=.3, top=.82, left=.07, right=.98, wspace=.32)
        for ax, metric, title in zip(axes, ("auroc", "balanced_accuracy", "severity_mae_percentage_points"),
                                     ("Family AUROC", "Balanced accuracy (threshold 0.5)", "Severity proxy MAE (pp)")):
            for i, family in enumerate(families):
                values = [next(r[metric] for r in family_rows if r["family"] == family and r["protocol"] == p) for p in (BASELINE, CONTROL)]
                ax.plot([0, 1], values, "o-", color=COLORS[i], lw=1.8, label=family.replace("xcat_reference_", ""))
            ax.set(xticks=[0, 1], xticklabels=["Baseline", "Epoch 200"], xlim=(-.15, 1.15), title=title)
            ax.grid(alpha=.2)
        axes[0].set_ylim(0, 1.03)
        axes[1].set_ylim(0, 1.03)
        axes[0].legend(title="Family", loc="best")
        fig.suptitle("Paired budgets · identical 60 scenarios / five families", fontsize=17)
        fig.text(.07, .12, "Development comparison at a fixed epoch; families are provisional anatomy groupings.", fontsize=12)
        fig.text(.07, .065, "The retained severity head predicts a synthetic volume-loss proxy; clinical severity is unvalidated.", fontsize=12)
        save(fig, "budget03-paired-families", "Matched family performance and severity proxy")
    return figures


def write_report(output_root, lock):
    """Publish a fresh immutable review-rN, then update the root HTML redirect."""
    root = Path(output_root).resolve()
    bundle = source_bundle(root, lock)
    analysis = bundle["analysis"]
    source_paths = (Path(__file__), Path(rotation.__file__), Path(rotation.empirical_roc.__code__.co_filename))
    expected_loaded = (LOADED_MODULE_SHA256, rotation.LOADED_MODULE_SHA256, rotation.POSTER_HELPER_LOADED_SHA256)
    generator_hashes = {str(path.resolve()): _hash(path) for path in source_paths}
    if any(_hash(path) != expected for path, expected in zip(source_paths, expected_loaded)):
        raise ValueError("Report source changed after import; restart the reporter")
    revision = 1
    while (root / f"review-r{revision}").exists():
        revision += 1
    output = root / f"review-r{revision}"
    temporary = Path(tempfile.mkdtemp(prefix=".budget-review-", dir=root))
    try:
        figures = _figures(temporary, analysis)
        _write_json(temporary / "analysis.json", analysis)
        _write_json(temporary / "fold-manifest.json", analysis["manifest"])
        _csv(temporary / "predictions.csv", analysis["records"],
             ["protocol", "fold_id", "seed", "sample_id", "case_id", "family", "label", "probability", "score_label", "score_prediction", "locator_coverage_passed", "pipeline_failure"])
        _csv(temporary / "family-metrics.csv", analysis["per_family"],
             ["protocol", "family", "seed", "samples", *METRICS, "pipeline_failure_samples", "locator_coverage_pass_rate"])
        _csv(temporary / "training-history.csv", analysis["training_history"],
             ["stage_id", "family", "epoch", "bce", "accuracy", "severity_mae_percentage_points", "optimizer_updates"])
        statuses = "".join(f"<tr><td>{html.escape(r['family'])}</td><td>{html.escape(r['status'])}</td><td>{html.escape(str(r.get('reason','')))}</td></tr>" for r in analysis["stage_status"])
        def value(v):
            return "pending" if v is None else f"{v:.6f}"
        metric_rows = []
        for protocol in (BASELINE, CONTROL):
            summary = analysis["protocols"][protocol]
            pooled = summary["pooled_scenario_metrics"]
            macro = summary["equal_family_metrics"]
            metric_rows.append(f"<tr><td>{html.escape(protocol)}</td><td>{summary['completed_fold_runs']}/5</td><td>{value(pooled['auroc'] if pooled else None)}</td><td>{value(macro['auroc']['mean'])}</td><td>{value(macro['balanced_accuracy']['mean'])}</td><td>{value(macro['severity_mae_percentage_points']['mean'])}</td></tr>")
        cards = "".join(f"<article><h2>{html.escape(f['title'])}</h2><p><a href=\"{f['stem']}.pdf\">PDF</a> · <a href=\"{f['stem']}.svg\">SVG</a> · <a href=\"{f['stem']}.png\">300 dpi PNG</a></p><img src=\"{f['stem']}.svg\" alt=\"{html.escape(f['title'])}\"></article>" for f in figures)
        complete = analysis["protocols"][CONTROL]["completed_fold_runs"]
        status_text = f"{complete}/5 controls complete. " + ("Full paired comparison available." if complete == 5 else "Full paired comparison pending; pending jobs are not failures.")
        document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoA fixed classifier budget</title><style>body{{font:17px/1.5 system-ui,sans-serif;max-width:1250px;margin:2rem auto;padding:0 1.5rem;background:#f6f8fa;color:#17212b}}article{{background:white;padding:1.4rem;margin:1.5rem 0;border:1px solid #d8dee4;border-radius:8px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;margin:1rem 0}}td,th{{border:1px solid #d8dee4;padding:.55rem .8rem}}a{{color:#0072b2}}.scope{{padding:1rem;background:#fff0e8;border-left:5px solid #d55e00}}</style><h1>CoA fixed classifier budget: 200 epochs</h1><p class="scope">{html.escape(status_text)}</p><p>Five development family folds, seed 42, the same 60 scenarios. Only the classifier budget changes from the baseline's inner-selected 8–29 epochs to a preregistered final epoch 200. Locator, input proposals, architecture, classification and severity heads, regression weight 1.0 and learning rates remain unchanged. Archived milestones do not select checkpoints; no intermediate held-out predictions are used.</p><p>This is a development cross-validation control using previously reviewed families, not an unopened external test. Reserved infant images and predictions are not read. Sibling/seed observations are correlated; the five family averages and SD bands are descriptive, without independent-patient confidence intervals. Severity MAE measures the synthetic local-volume-loss proxy and is not a clinically validated severity measure.</p><table><tr><th>Budget</th><th>Folds</th><th>Pooled AUROC</th><th>Mean family AUROC</th><th>Mean family BA</th><th>Mean severity MAE (pp)</th></tr>{''.join(metric_rows)}</table><table><tr><th>Family</th><th>Control state</th><th>Reason</th></tr>{statuses}</table><p>Exact exports: <a href="analysis.json">analysis</a> · <a href="predictions.csv">prediction CSV</a> · <a href="family-metrics.csv">family metrics</a> · <a href="training-history.csv">training-only history</a> · <a href="mean-roc-points.csv">mean ROC coordinates</a> · <a href="fold-manifest.json">fold manifest</a> · <a href="provenance.json">provenance</a>.</p>{cards}</html>'''
        (temporary / "report.html").write_text(document)
        all_inputs = {**bundle["input_sha256"], **generator_hashes}
        if any(_hash(path) != expected for path, expected in all_inputs.items()):
            raise ValueError("Report input or generator changed during export")
        proof = {"schema_version": SCHEMA, "source_lock_sha256": bundle["source_lock_sha256"],
                 "parent_directory": bundle["parent_directory"], "parent_lock_sha256": bundle["parent_lock_sha256"],
                 "input_sha256": bundle["input_sha256"], "generator_sha256": generator_hashes,
                 "records_canonical_sha256": _canonical(analysis["records"]),
                 "artifact_sha256": {p.name: _hash(p) for p in sorted(temporary.iterdir())},
                 "image_or_checkpoint_payloads_read": False, "reserved_test_predictions_read": False,
                 "intermediate_outer_predictions_read": False, "clinical_severity_validated": False,
                 "independent_observation_ci": None, "figure_count": len(figures)}
        _write_json(temporary / "provenance.json", proof)
        os.rename(temporary, output)
        target = output.name + "/report.html"
        redirect = root / ".budget-report.html.tmp"
        redirect.write_text(f'<!doctype html><meta http-equiv="refresh" content="0;url={target}"><a href="{target}">Open fixed-budget review</a>')
        os.replace(redirect, root / "report.html")
        return {"output_directory": str(output), "status": analysis["status"], "figure_count": len(figures),
                "records": len(analysis["records"]), "completed_fold_runs": complete}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
