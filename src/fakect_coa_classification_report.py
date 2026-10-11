"""Metadata-only, immutable review of the matched classification-loss control.

Both protocols fit the same graph for 200 epochs on the same five family folds.
The control disables the auxiliary severity loss; its untrained head is never
interpreted as a severity prediction. Reporting cannot select outer checkpoints.
"""
from collections import defaultdict
import copy
import hashlib
import html
import math
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

import fakect_coa_budget_report as budget
import fakect_coa_rotation_report as rotation

SCHEMA = "fakect.coa-classification-report/1"
LOCK_SCHEMA = "fakect.coa-classification-control/1"
BASELINE = "joint200"
CONTROL = "classification200"
LABELS = {BASELINE: "Joint loss · 200 epochs", CONTROL: "Classification only · 200 epochs"}
COLORS = budget.COLORS
LOADED_MODULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
METRICS = rotation.METRICS


def _classification_only(value):
    """Remove severity metrics produced internally by the shared ROC helper."""
    if isinstance(value, dict):
        return {k: _classification_only(v) for k, v in value.items()
                if k not in {"severity_mae_percentage_points", "severity_prediction_samples"}}
    if isinstance(value, list):
        return [_classification_only(v) for v in value]
    return value


def analyze_records(baseline, control, manifest):
    manifest = rotation.validate_manifest(manifest)
    paired = copy.deepcopy(manifest)
    paired["folds"] = [dict(f, protocol=p, selection_stages=[])
                       for p in (BASELINE, CONTROL) for f in manifest["folds"]]
    paired["stage_catalog"] = []
    if any(r.get("score_prediction") is not None for r in control):
        raise ValueError("An untrained severity head cannot enter control predictions")
    records = [dict(r, protocol=p, score_prediction=None)
               for p, rows in ((BASELINE, baseline), (CONTROL, control)) for r in rows]
    analysis = rotation.aggregate_records(records, paired)
    if len(baseline) != sum(len(f["evaluation_sample_ids"]) for f in manifest["folds"]):
        raise ValueError("The completed joint-loss baseline requires the entire planned panel")
    for p in (BASELINE, CONTROL):
        analysis["protocols"][p]["mean_roc"] = budget.corrected_mean_roc(analysis["per_family"], p)
    complete = analysis["protocols"][CONTROL]["completed_fold_runs"] == len(manifest["folds"])
    analysis["paired_comparisons"] = [rotation.paired_comparison(analysis, BASELINE, CONTROL)] if complete else [{
        "status": "pending", "left_protocol": BASELINE, "right_protocol": CONTROL,
        "reason": "All five identical outer panels must complete before the full paired comparison."}]
    analysis.update(schema_version=SCHEMA, status="complete" if complete else "partial" if control else "pending",
                    checkpoint_selection="Preregistered final epoch 200; no held-out checkpoint selection",
                    severity_trained=False, severity_metrics_exported=False,
                    protocol_labels=LABELS, independent_observation_ci=None)
    return _classification_only(analysis)


def checked_history(history, receipt, stage, *, regression_weight=0.0):
    """Export only classification metrics from a complete train-only history."""
    epochs = history.get("epochs")
    ids = set(stage["train_sample_ids"])
    if (not isinstance(epochs, list) or len(epochs) != 200 or
            receipt.get("selected_epoch") != 200 or receipt.get("epochs_completed") != 200 or
            receipt.get("validation") is not None or receipt.get("validation_sample_ids") != [] or
            set(receipt.get("training_sample_ids", [])) != ids or
            len(receipt.get("training_sample_ids", [])) != len(ids) or
            history.get("selected_epoch") != 200 or
            history.get("checkpoint_selection") != "final fixed-budget epoch"):
        raise ValueError("A completed classifier requires 200 train-only final-epoch records")
    if regression_weight == 0.0 and (receipt.get("regression_weight") != 0.0 or
                                     receipt.get("severity_trained") is not False):
        raise ValueError("Control receipt must identify its disabled severity loss")
    output = []
    updates = len(ids) // 2
    for epoch, row in enumerate(epochs, 1):
        pairs = row.get("training_pair_order", [])
        if not isinstance(pairs, list) or len(pairs) != updates or any(not isinstance(p, list) or len(p) != 2 for p in pairs):
            raise ValueError("Each training update must identify one two-image matched pair")
        flattened = [sid for pair in pairs for sid in pair]
        if (row.get("epoch") != epoch or row.get("validation") is not None or
                row.get("selection_bce") is not None or row.get("optimizer_updates") != epoch * updates or
                len(flattened) != len(ids) or set(flattened) != ids):
            raise ValueError("Training history contains selection metrics or a changed sample/update panel")
        train = row.get("training", {})
        if regression_weight == 0.0 and train.get("score_mae") is not None:
            raise ValueError("Untrained severity output cannot appear in training history")
        export = dict(stage_id=stage["id"], family=stage["outer_fold_id"], epoch=epoch,
                      bce=train.get("bce"), accuracy=train.get("accuracy"), optimizer_updates=row["optimizer_updates"])
        for field in ("bce", "accuracy"):
            v = export[field]
            if isinstance(v, bool) or not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0:
                raise ValueError("Training classification metrics must be finite and nonnegative")
        if export["accuracy"] > 1:
            raise ValueError("Training accuracy exceeds one")
        output.append(export)
    if receipt.get("optimizer_updates") != 200 * updates:
        raise ValueError("Final optimizer count differs from the 200-epoch training panel")
    if regression_weight == 0.0 and receipt.get("training", {}).get("score_mae") is not None:
        raise ValueError("Control receipt exposes an untrained severity metric")
    return output


def _completed(row, stage, lock_hash, *, control=False):
    if (row.get("status") != "completed" or row.get("stage_id") != stage["id"] or
            row.get("protocol") != "family_holdout" or row.get("source_lock_sha256") != lock_hash or
            row.get("final_test_evaluated") is not True or row.get("reserved_test_evaluated") is not False or
            row.get("classifier_selected_epoch") != 200 or row.get("locator_retrained") is not False or
            row.get("locator_proposals_changed") is not False):
        raise ValueError("Completed result violates its locked identity or final-epoch contract")
    if control and (row.get("regression_weight") != 0.0 or row.get("severity_trained") is not False):
        raise ValueError("Completed control must identify its disabled severity objective")
    predictions = row.get("final_predictions")
    if not isinstance(predictions, list):
        raise ValueError("Completed result requires its full final prediction panel")
    for r in predictions:
        if (r.get("protocol") != stage["protocol"] or r.get("fold_id") != stage["outer_fold_id"] or
                r.get("seed") != stage["seed"] or control and r.get("score_prediction") is not None):
            raise ValueError("Final prediction identity or untrained severity output is invalid")
    return predictions


def _initialization(left, right, stage):
    """Compare training-only initial predictions without opening checkpoint data."""
    if (left.get("weights_sha256") != stage["expected_initial_weights_sha256"] or
            right.get("weights_sha256") != left.get("weights_sha256") or
            left.get("seed") != stage["seed"] or right.get("seed") != stage["seed"] or
            left.get("fresh_weights") is not True or right.get("fresh_weights") is not True):
        raise ValueError("Fresh graph initialization differs from the joint-loss baseline")
    panels = []
    for value in (left, right):
        rows = value.get("predictions", [])
        if len(rows) != len(stage["train_sample_ids"]) or {r["sample_id"] for r in rows} != set(stage["train_sample_ids"]):
            raise ValueError("Initial prediction panel contains a missing, repeated or held-out sample")
        if any(r.get("split") != "train" for r in rows):
            raise ValueError("Initialization parity may use only training predictions")
        panels.append({r["sample_id"]: r for r in rows})
    for sid, old in panels[0].items():
        new = panels[1][sid]
        for field in ("case_id", "family", "label", "score_label"):
            if old.get(field) != new.get(field):
                raise ValueError("Initial training identities or targets changed")
        for field in ("logit", "probability"):
            a, b = old.get(field), new.get(field)
            if (isinstance(a, bool) or isinstance(b, bool) or not isinstance(a, (int, float)) or
                    not isinstance(b, (int, float)) or not math.isfinite(a) or not math.isfinite(b) or
                    not math.isclose(a, b, rel_tol=1e-5, abs_tol=1e-6)):
                raise ValueError("Initial classification predictions differ from baseline tolerance")


def source_bundle(output_root, lock):
    """Read checksum-checked JSON and sidecars; never read images/checkpoints."""
    root = Path(output_root).resolve()
    if (root / "INCOMPLETE").exists():
        raise ValueError("Classification experiment lock is incomplete")
    fingerprints = {}
    read = lambda path: budget._json(path, fingerprints)
    if read(root / "experiment-lock.json") != lock or lock.get("schema_version") != LOCK_SCHEMA:
        raise ValueError("Passed classification lock differs from its frozen schema or content")
    own_hash = fingerprints[str(root / "experiment-lock.json")]
    locations = [Path(lock[k]) for k in ("parent_directory", "comparison_directory")]
    if any(not p.is_absolute() for p in locations) or len({root, *(p.resolve() for p in locations)}) != 3:
        raise ValueError("Original, comparison and control require three distinct absolute directories")
    parent, comparison = (p.resolve() for p in locations)
    if any((p / "INCOMPLETE").exists() for p in (parent, comparison)):
        raise ValueError("A frozen prerequisite lock is incomplete")
    old, prior = read(parent / "experiment-lock.json"), read(comparison / "experiment-lock.json")
    if (fingerprints[str(parent / "experiment-lock.json")] != lock["parent_lock_sha256"] or
            fingerprints[str(comparison / "experiment-lock.json")] != lock["comparison_lock_sha256"] or
            prior.get("parent_lock_sha256") != lock["parent_lock_sha256"] or
            Path(prior["parent_directory"]).resolve() != parent):
        raise ValueError("Original or completed comparison lock differs from its pinned dependency")
    training = lock["config"]["training"]
    expected = dict(old["config"]["training"], classifier_epochs=200, regression_weight=0.0)
    interval = training.get("checkpoint_interval")
    if (lock.get("unchanged_training_settings") != old["config"]["training"] or
            old["config"]["training"].get("regression_weight") != 1.0 or
            lock.get("intended_training_settings") != expected or
            training.get("regression_weight") != 0.0 or training.get("classifier_epochs") != 200 or
            lock.get("classifier_epochs") != 200 or type(interval) is not int or not 1 <= interval <= 200 or 200 % interval or
            prior["config"]["training"].get("classifier_epochs") != 200 or
            prior.get("unchanged_training_settings") != old["config"]["training"]):
        raise ValueError("Only the auxiliary regression weight may change between these 200-epoch fits")
    stages = lock["stages"]
    old_stages = {s["id"]: s for s in old["stages"] if s["stage"] == "outer_refit" and s["protocol"] == "family_holdout"}
    prior_stages = {s["id"]: s for s in prior["stages"]}
    if len(stages) != 5 or {s["id"] for s in stages} != set(old_stages) or set(prior_stages) != set(old_stages):
        raise ValueError("Exactly the same five primary outer family folds are required")
    policies = {"locator_policy", "classifier_policy", "checkpoint_policy", "parent_inner_stage_ids", "expected_initial_weights_sha256"}
    for s in stages:
        core = {k: v for k, v in s.items() if k not in policies}
        if any(core != {k: v for k, v in d.items() if k not in policies} for d in (old_stages[s["id"]], prior_stages[s["id"]])) or s.get("parent_inner_stage_ids") != []:
            raise ValueError("Classification control changed a frozen fold or selected an outer checkpoint")
        if s.get("expected_initial_weights_sha256") != prior_stages[s["id"]].get("expected_initial_weights_sha256"):
            raise ValueError("Classification control changed its initialization pin")
    if lock["reserved_case_ids"] != old["reserved_case_ids"] or lock["reserved_case_ids"] != prior["reserved_case_ids"]:
        raise ValueError("Reserved anatomy identities changed")
    derived = rotation.rotation_manifest(old)
    derived["folds"] = [dict(f, selection_stages=[]) for f in derived["folds"] if f["protocol"] == "family_holdout"]
    derived["stage_catalog"] = [r for r in derived["stage_catalog"] if r["stage_id"] in old_stages]
    manifest = rotation.validate_manifest(lock.get("fold_manifest", derived))
    if manifest != rotation.validate_manifest(derived) or manifest != rotation.validate_manifest(prior["fold_manifest"]):
        raise ValueError("Classification report panels, labels or threshold differ from the frozen design")
    for fold in manifest["folds"]:
        s = next(s for s in stages if s["id"] == fold["stage_id"])
        if fold["seed"] != 42 or len(fold["evaluation_case_ids"]) != 2 or len(fold["evaluation_sample_ids"]) != 12 or len(s["train_sample_ids"]) != 48:
            raise ValueError("This report requires the original seed-42 48/12 panels")
        for case in fold["evaluation_case_ids"]:
            targets = [manifest["sample_targets"][sid] for sid in fold["evaluation_sample_ids"] if manifest["sample_targets"][sid]["case_id"] == case]
            if len(targets) != 6 or sum(r["label"] for r in targets) != 3:
                raise ValueError("Every held-out phantom requires the same six balanced scenarios")
    dependencies = lock.get("dependency_sha256", {})
    def pinned(path):
        value = read(path)
        if dependencies.get(str(Path(path).resolve())) != fingerprints[str(Path(path).resolve())]:
            raise ValueError("Comparison metadata differs from the control lock's dependency pin")
        return value
    baseline, control, histories, statuses = [], [], [], []
    for s in stages:
        prior_folder = comparison / "stages" / s["id"]
        if (prior_folder / "INCOMPLETE").exists() or (prior_folder / "failure.json").exists():
            raise ValueError("The joint-loss comparison stage is incomplete or failed")
        previous = pinned(prior_folder / "results.json")
        baseline.extend(_completed(previous, s, lock["comparison_lock_sha256"]))
        old_history = pinned(prior_folder / "classifier/history.json")
        histories.extend(dict(r, protocol=BASELINE) for r in checked_history(old_history, previous["classifier"], s, regression_weight=1.0))
        old_initial = pinned(prior_folder / "classifier/initialization.json")
        if previous["classifier"].get("fresh_initial_weights_sha256") != s["expected_initial_weights_sha256"]:
            raise ValueError("Baseline initialization changed")
        old_proposals = {}
        for name, role, ids in (("proposal-audit.json", "train", s["train_sample_ids"]),
                                ("final-test/proposal-audit.json", "final_test", s["final_test_sample_ids"])):
            original = pinned(parent / "stages" / s["id"] / name)
            audit = pinned(prior_folder / name)
            rows = original.get("samples", [])
            if (audit.get("samples") != rows or original.get("selection_allowed") is not False or audit.get("selection_allowed") is not False or
                    len(rows) != len(ids) or {r["sample_id"] for r in rows} != set(ids) or any(r["split"] != role for r in rows)):
                raise ValueError("Comparison crop proposals differ from the reviewed parent panels")
            old_proposals[name] = rows
        folder = root / "stages" / s["id"]
        failure, result = folder / "failure.json", folder / "results.json"
        if failure.exists():
            failed = read(failure)
            if result.exists() and not (folder / "INCOMPLETE").exists():
                raise ValueError("Completed control has a contradictory failure receipt")
            statuses.append(dict(stage_id=s["id"], family=s["outer_fold_id"], status="failed", reason=failed.get("message", failed.get("reason", "See failure receipt"))))
        elif result.exists() and not (folder / "INCOMPLETE").exists():
            row = read(result)
            control.extend(_completed(row, s, own_hash, control=True))
            receipt = read(folder / "classifier/receipt.json")
            if (row["classifier"] != receipt or row.get("parent_lock_sha256") != lock["parent_lock_sha256"] or
                    row.get("comparison_lock_sha256") != lock["comparison_lock_sha256"] or
                    receipt.get("fresh_initial_weights_sha256") != s["expected_initial_weights_sha256"]):
                raise ValueError("Control receipt or prerequisite initialization identity changed")
            final = read(folder / "final-test/predictions.json")
            if (final.get("predictions") != row["final_predictions"] or final.get("source_lock_sha256") != own_hash or
                    final.get("checkpoint_sha256") != receipt.get("checkpoint_sha256")):
                raise ValueError("Final prediction artifact differs from the completed control result")
            history = read(folder / "classifier/history.json")
            histories.extend(dict(r, protocol=CONTROL) for r in checked_history(history, receipt, s))
            if [r["training_pair_order"] for r in history["epochs"]] != [r["training_pair_order"] for r in old_history["epochs"]]:
                raise ValueError("Training pair order changed between the matched 200-epoch fits")
            _initialization(old_initial, read(folder / "classifier/initialization.json"), s)
            for name, expected_rows in old_proposals.items():
                audit = read(folder / name)
                if audit.get("samples") != expected_rows or audit.get("selection_allowed") is not False:
                    raise ValueError("Classification control changed its input crop proposals")
            statuses.append(dict(stage_id=s["id"], family=s["outer_fold_id"], status="completed"))
        else:
            marker = folder / "INCOMPLETE"
            if marker.exists():
                fingerprints[str(marker.resolve())] = budget._hash(marker)
            statuses.append(dict(stage_id=s["id"], family=s["outer_fold_id"], status="running" if folder.exists() else "pending"))
    analysis = analyze_records(baseline, control, manifest)
    analysis.update(stage_status=statuses, training_history=histories)
    if any(s["status"] == "failed" for s in statuses):
        analysis["status"] = "partial_with_failed_stages" if control else "blocked_by_failed_stage"
    return dict(analysis=analysis, input_sha256=fingerprints, source_lock_sha256=own_hash,
                parent_directory=str(parent), parent_lock_sha256=lock["parent_lock_sha256"],
                comparison_directory=str(comparison), comparison_lock_sha256=lock["comparison_lock_sha256"],
                reserved_case_ids=list(lock["reserved_case_ids"]))


def _figures(folder, analysis):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fakect-classification-mpl")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 12, "axes.titlesize": 14, "legend.fontsize": 10,
                         "svg.hashsalt": SCHEMA, "svg.fonttype": "none", "pdf.fonttype": 42})
    figures = []
    def save(fig, stem, title):
        for suffix in ("pdf", "svg", "png"):
            metadata = {"CreationDate": None, "ModDate": None} if suffix == "pdf" else {"Date": None} if suffix == "svg" else None
            fig.savefig(folder / (stem + "." + suffix), dpi=300, metadata=metadata)
        plt.close(fig)
        figures.append(dict(stem=stem, title=title))
    histories = analysis["training_history"]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2))
    fig.subplots_adjust(bottom=.26, top=.82, left=.08, right=.98, wspace=.25)
    families = sorted({r["family"] for r in histories})
    for i, family in enumerate(families):
        for protocol, style in ((BASELINE, "--"), (CONTROL, "-")):
            rows = [r for r in histories if r["family"] == family and r["protocol"] == protocol]
            for ax, metric in zip(axes, ("bce", "accuracy")):
                if rows:
                    ax.plot([r["epoch"] for r in rows], [r[metric] for r in rows], style, color=COLORS[i], alpha=.85,
                            label=family.replace("xcat_reference_", "") if protocol == BASELINE else None)
    for ax, title in zip(axes, ("Training BCE", "Training accuracy")):
        ax.set(xlabel="Epoch", title=title, xlim=(1, 200)); ax.grid(alpha=.2)
    axes[0].set_yscale("log"); axes[1].set_ylim(0, 1.03)
    axes[0].legend(title="Outer family", loc="best")
    fig.suptitle("Training-only progression · fixed final epoch 200", fontsize=16)
    fig.text(.08, .12, "Dashed: joint loss (weight 1). Solid: classification only (weight 0).", fontsize=11)
    fig.text(.08, .065, "No outer checkpoint selection. Untrained severity output is not evaluated.", fontsize=11)
    save(fig, "classification01-training", "Training-only classification loss and accuracy")
    fig, axes = plt.subplots(1, 2, figsize=(12.8, 6.1))
    fig.subplots_adjust(bottom=.29, top=.83, left=.075, right=.97, wspace=.29)
    mean_points = []
    for protocol, color in ((BASELINE, COLORS[0]), (CONTROL, COLORS[1])):
        summary = analysis["protocols"][protocol]; metrics = summary["pooled_scenario_metrics"]
        if metrics is None:
            continue
        roc = metrics["roc"]
        axes[0].plot(roc["fpr"], roc["tpr"], color=color, lw=2.4, label=f"{LABELS[protocol]}\nAUC {metrics['auroc']:.4f}; n={metrics['samples']}")
        mean = summary["mean_roc"]
        grid, curve, sd = (np.array(mean[k]) for k in ("fpr_grid", "mean_tpr", "descriptive_family_sd_tpr"))
        axes[1].plot(grid, curve, color=color, lw=2.4, label=f"{LABELS[protocol]}\nMean exact AUC {mean['exact_mean_family_auroc']:.4f}; families={mean['families']}")
        axes[1].fill_between(grid, np.maximum(0, curve-sd), np.minimum(1, curve+sd), color=color, alpha=.12)
        mean_points.extend(dict(protocol=protocol, fpr=x, mean_tpr=y, descriptive_family_sd_tpr=z) for x, y, z in zip(grid, curve, sd))
    for ax, title in zip(axes, ("Pooled empirical ROC", "Equal-family mean ROC")):
        ax.plot([0, 1], [0, 1], "--", color=".65"); ax.grid(alpha=.2)
        ax.set(xlim=(0, 1), ylim=(0, 1.02), xlabel="False positive rate", ylabel="True positive rate", title=title)
        ax.legend(loc="lower right")
    fig.suptitle("Matched classification-loss control · development cross-validation", fontsize=16)
    fig.text(.075, .14, "Band = descriptive family SD. Correlated siblings are not independent patients.", fontsize=11)
    fig.text(.075, .085, "Mean labels use exact empirical areas; sampled mean-curve integrals may differ slightly.", fontsize=11)
    if analysis["status"] != "complete":
        fig.text(.075, .035, "Control incomplete: available-fold curves use a different panel. Full comparison pending.", fontsize=11)
    save(fig, "classification02-roc", "Exact pooled and equal-family ROC")
    budget._csv(folder / "mean-roc-points.csv", mean_points, ["protocol", "fpr", "mean_tpr", "descriptive_family_sd_tpr"])
    if analysis["paired_comparisons"][0].get("status") != "pending":
        rows = analysis["per_family"]
        fig, axes = plt.subplots(1, 3, figsize=(14, 5.6))
        fig.subplots_adjust(bottom=.26, top=.81, left=.07, right=.98, wspace=.32)
        for ax, metric, title in zip(axes, ("auroc", "balanced_accuracy", "bce"), ("Family AUROC", "Balanced accuracy (threshold 0.5)", "Held-out BCE")):
            for i, family in enumerate(sorted({r["family"] for r in rows})):
                values = [next(r[metric] for r in rows if r["family"] == family and r["protocol"] == p) for p in (BASELINE, CONTROL)]
                ax.plot([0, 1], values, "o-", color=COLORS[i], label=family.replace("xcat_reference_", ""))
            ax.set(xticks=[0, 1], xticklabels=["Joint loss", "Classification"], xlim=(-.15, 1.15), title=title); ax.grid(alpha=.2)
        axes[0].set_ylim(0, 1.03); axes[1].set_ylim(0, 1.03); axes[0].legend(title="Family", loc="best")
        fig.suptitle("Matched 60 scenarios · five provisional anatomy families", fontsize=16)
        fig.text(.07, .1, "Both models train for 200 epochs. Only auxiliary severity loss changes; no severity metrics.", fontsize=11)
        save(fig, "classification03-paired-families", "Paired family classification performance")
    return figures


def write_report(output_root, lock):
    """Publish a new immutable revision and atomically update its root redirect."""
    root = Path(output_root).resolve(); bundle = source_bundle(root, lock); analysis = bundle["analysis"]
    paths = (Path(__file__), Path(budget.__file__), Path(rotation.__file__), Path(rotation.empirical_roc.__code__.co_filename))
    loaded = (LOADED_MODULE_SHA256, budget.LOADED_MODULE_SHA256, rotation.LOADED_MODULE_SHA256, rotation.POSTER_HELPER_LOADED_SHA256)
    generators = {str(p.resolve()): budget._hash(p) for p in paths}
    if any(budget._hash(p) != expected for p, expected in zip(paths, loaded)):
        raise ValueError("Report implementation changed after import; restart the reporter")
    pinned = lock.get("code_sha256", {}).get("src/fakect_coa_classification_report.py")
    if pinned is not None and pinned != LOADED_MODULE_SHA256:
        raise ValueError("Classification reporter differs from the frozen source pin")
    revision = 1
    while (root / f"review-r{revision}").exists():
        revision += 1
    output = root / f"review-r{revision}"
    temporary = Path(tempfile.mkdtemp(prefix=".classification-review-", dir=root))
    try:
        figures = _figures(temporary, analysis)
        budget._write_json(temporary / "analysis.json", analysis)
        budget._write_json(temporary / "fold-manifest.json", analysis["manifest"])
        budget._csv(temporary / "predictions.csv", analysis["records"], ["protocol", "fold_id", "seed", "sample_id", "case_id", "family", "label", "probability", "locator_coverage_passed", "pipeline_failure"])
        budget._csv(temporary / "family-metrics.csv", analysis["per_family"], ["protocol", "family", "seed", "samples", *METRICS, "accuracy", "pipeline_failure_samples", "locator_coverage_pass_rate"])
        budget._csv(temporary / "training-history.csv", analysis["training_history"], ["protocol", "stage_id", "family", "epoch", "bce", "accuracy", "optimizer_updates"])
        complete = analysis["protocols"][CONTROL]["completed_fold_runs"]
        status = f"{complete}/5 controls complete. " + ("Full matched comparison available." if complete == 5 else "Full matched comparison pending; pending jobs are not failures.")
        def value(v):
            return "pending" if v is None else f"{v:.6f}"
        metric_rows = []
        for protocol in (BASELINE, CONTROL):
            summary = analysis["protocols"][protocol]; pooled = summary["pooled_scenario_metrics"]; macro = summary["equal_family_metrics"]
            metric_rows.append(f"<tr><td>{html.escape(LABELS[protocol])}</td><td>{summary['completed_fold_runs']}/5</td><td>{value(pooled['auroc'] if pooled else None)}</td><td>{value(macro['auroc']['mean'])}</td><td>{value(macro['balanced_accuracy']['mean'])}</td><td>{value(macro['sensitivity']['mean'])}</td><td>{value(macro['specificity']['mean'])}</td><td>{value(pooled['bce'] if pooled else None)}</td></tr>")
        statuses = "".join(f"<tr><td>{html.escape(r['family'])}</td><td>{html.escape(r['status'])}</td><td>{html.escape(str(r.get('reason', '')))}</td></tr>" for r in analysis["stage_status"])
        cards = "".join(f'<article><h2>{html.escape(f["title"])}</h2><p><a href="{f["stem"]}.pdf">PDF</a> · <a href="{f["stem"]}.svg">SVG</a> · <a href="{f["stem"]}.png">300 dpi PNG</a></p><img src="{f["stem"]}.svg" alt="{html.escape(f["title"])}"></article>' for f in figures)
        document = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoA classification-loss control</title><style>body{{font:17px/1.5 system-ui,sans-serif;max-width:1250px;margin:2rem auto;padding:0 1.5rem;background:#f6f8fa;color:#17212b}}article{{background:white;padding:1.4rem;margin:1.5rem 0;border:1px solid #d8dee4;border-radius:8px}}img{{width:100%;height:auto}}table{{border-collapse:collapse;margin:1rem 0}}td,th{{border:1px solid #d8dee4;padding:.55rem .8rem}}a{{color:#0072b2}}.scope{{padding:1rem;background:#fff0e8;border-left:5px solid #d55e00}}</style><h1>CoA classification-loss control</h1><p class="scope">{html.escape(status)}</p><p>Both protocols train for 200 epochs on the same five development family folds, seed 42, and the same 60 held-out edited scenarios. Only the auxiliary regression weight changes from 1.0 to 0.0. Graph, initial weights, locator, crop proposals, training pair order, optimizer configuration and learning rate remain fixed. The retained severity output is untrained in the control and is not evaluated or interpreted.</p><p>The final epoch was declared before fitting; neither held-out predictions nor intermediate milestones select checkpoints. This is development cross-validation using previously reviewed families, not an unopened external test. Reserved infant predictions and image/checkpoint payloads are not read. Siblings and repeated seeds are correlated. Family means and SD bands are descriptive; no independent-patient confidence interval is claimed. BCE uses the same float32-epsilon clipping in both protocols.</p><table><tr><th>Objective</th><th>Folds</th><th>Pooled AUROC</th><th>Mean family AUROC</th><th>Mean family BA</th><th>Mean sensitivity</th><th>Mean specificity</th><th>Pooled BCE</th></tr>{''.join(metric_rows)}</table><table><tr><th>Family</th><th>Control state</th><th>Reason</th></tr>{statuses}</table><p><a href="analysis.json">Analysis</a> · <a href="predictions.csv">Predictions</a> · <a href="family-metrics.csv">Family metrics</a> · <a href="training-history.csv">Training-only history</a> · <a href="mean-roc-points.csv">Mean ROC coordinates</a> · <a href="fold-manifest.json">Fold manifest</a> · <a href="provenance.json">Provenance</a></p>{cards}</html>'''
        (temporary / "report.html").write_text(document)
        inputs = {**bundle["input_sha256"], **generators}
        if any(budget._hash(p) != expected for p, expected in inputs.items()):
            raise ValueError("Report input or generator changed during export")
        proof = {"schema_version": SCHEMA, **{k: v for k, v in bundle.items() if k != "analysis"},
                 "generator_sha256": generators, "records_canonical_sha256": budget._canonical(analysis["records"]),
                 "artifact_sha256": {p.name: budget._hash(p) for p in sorted(temporary.iterdir())},
                 "image_or_checkpoint_payloads_read": False, "reserved_test_predictions_read": False,
                 "intermediate_outer_predictions_read": False, "severity_metrics_exported": False,
                 "independent_observation_ci": None, "figure_count": len(figures)}
        budget._write_json(temporary / "provenance.json", proof)
        os.rename(temporary, output)
        target = output.name + "/report.html"; redirect = root / ".classification-report.html.tmp"
        redirect.write_text(f'<!doctype html><meta http-equiv="refresh" content="0;url={target}"><a href="{target}">Open classification-loss review</a>')
        os.replace(redirect, root / "report.html")
        return dict(output_directory=str(output), status=analysis["status"], figure_count=len(figures), records=len(analysis["records"]), completed_fold_runs=complete)
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
