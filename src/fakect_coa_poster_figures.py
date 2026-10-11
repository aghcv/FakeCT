"""Reproducible poster figures from immutable, completed CoA metadata.

This module never imports TensorFlow or reads imaging/checkpoint payloads.
Training predictions describe apparent performance; validation is development
evaluation of a single anatomy family. No test performance or uncertainty
interval is inferred from those eight validation scenarios.
"""
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


SCHEMA = "fakect.coa-poster-figures/1"
LOADED_MODULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
TRAIN = "#0072B2"
VALIDATION = "#D55E00"
GREY = "#636A73"
ROLES = ("original", "edited_negative", "lower_positive", "higher_positive")


def fingerprint(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _finite(value, name, lower=None, upper=None):
    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError(f"{name} must be finite numeric data")
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(f"{name} outside [{lower}, {upper}]")
    return float(value)


def empirical_roc(labels, probabilities):
    """Exact observed-threshold ROC; tied scores move as a single group.

    The first threshold is None, representing a threshold above all scores.
    Linear interpolation of tied-score groups gives the standard trapezoidal
    AUROC, equivalent to positive-negative concordance with half credit for ties.
    """
    labels, probabilities = list(labels), list(probabilities)
    if len(labels) != len(probabilities) or not labels:
        raise ValueError("ROC requires equally sized, nonempty labels and scores")
    if any(type(x) not in (int, np.int64, np.int32) or x not in (0, 1) for x in labels):
        raise ValueError("ROC labels must be binary integers")
    scores = np.array([_finite(x, "probability", 0, 1) for x in probabilities])
    target = np.asarray(labels)
    positive, negative = int(np.sum(target)), int(len(target) - np.sum(target))
    if not positive or not negative:
        raise ValueError("ROC is undefined for a single class")
    order = np.argsort(-scores, kind="stable")
    score, target = scores[order], target[order]
    ends = np.r_[np.flatnonzero(np.diff(score)), len(score) - 1]
    tp = np.cumsum(target)[ends]
    fp = (ends + 1) - tp
    tpr = np.r_[0., tp / positive]
    fpr = np.r_[0., fp / negative]
    auc = float(np.sum(np.diff(fpr) * (tpr[:-1] + tpr[1:]) / 2))
    return {"fpr": fpr.tolist(), "tpr": tpr.tolist(),
            "thresholds": [None] + score[ends].tolist(), "auroc": auc,
            "positive": positive, "negative": negative}


def checked_predictions(predictions):
    if not isinstance(predictions, list) or not predictions:
        raise ValueError("Predictions must be a nonempty list")
    seen, output = set(), []
    for row in predictions:
        if not isinstance(row, dict):
            raise ValueError("Prediction rows must be objects")
        identity = row.get("sample_id")
        if not isinstance(identity, str) or not identity or identity in seen:
            raise ValueError("Prediction sample IDs must be unique nonempty strings")
        seen.add(identity)
        if row.get("split") not in ("train", "validation"):
            raise ValueError("Only training and development validation predictions are permitted")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("Prediction labels must be binary integers")
        if row.get("role") not in ROLES:
            raise ValueError("Unknown development scenario role")
        for key in ("case_id", "family"):
            if not isinstance(row.get(key), str) or not row[key]:
                raise ValueError(f"Prediction {key} must be a nonempty string")
        p = _finite(row.get("probability"), "probability", 0, 1)
        _finite(row.get("score_label"), "score_label", 0, 1)
        _finite(row.get("score_prediction"), "score_prediction", 0, 1)
        if "logit" in row:
            logit = _finite(row["logit"], "logit")
            expected = 1 / (1 + math.exp(-logit)) if logit >= 0 else math.exp(logit) / (1 + math.exp(logit))
            if not math.isclose(p, expected, rel_tol=1e-10, abs_tol=1e-12):
                raise ValueError("Probability does not match the recorded logit")
        output.append(dict(row))
    return output


def confusion_counts(predictions, threshold=0.5):
    _finite(threshold, "decision threshold", 0, 1)
    counts = dict(TP=0, TN=0, FP=0, FN=0)
    for row in predictions:
        predicted = row["probability"] >= threshold
        counts[("T" if predicted == bool(row["label"]) else "F") + ("P" if predicted else "N")] += 1
    return counts


def _metrics(predictions, threshold):
    labels = np.array([p["label"] for p in predictions])
    scores = np.array([p["probability"] for p in predictions])
    epsilon = 2.0 ** -23  # Frozen classifier-report float32 clipping contract.
    def clipped(value):
        return min(1 - epsilon, max(epsilon, value))
    bce = math.fsum(-r["label"] * math.log(clipped(r["probability"]))
                   - (1 - r["label"]) * math.log(clipped(1 - r["probability"]))
                   for r in predictions) / len(predictions)
    counts = confusion_counts(predictions, threshold)
    mae = float(np.mean([abs(p["score_prediction"] - p["score_label"]) for p in predictions]))
    return {"samples": len(predictions), "bce": bce,
            "accuracy": (counts["TP"] + counts["TN"]) / len(predictions),
            "sensitivity": counts["TP"] / int(np.sum(labels)),
            "specificity": counts["TN"] / int(np.sum(1 - labels)),
            "auroc": empirical_roc(labels.tolist(), scores.tolist())["auroc"],
            "score_mae": mae, "confusion_counts": counts}


def _check_metrics(recorded, computed):
    for key, value in computed.items():
        if isinstance(value, dict):
            if recorded.get(key) != value:
                raise ValueError(f"Recorded {key} does not match prediction counts")
        elif not math.isclose(float(recorded.get(key, float("nan"))), value, rel_tol=1e-8, abs_tol=1e-10):
            raise ValueError(f"Recorded {key} does not match predictions")


def source_bundle(experiment):
    """Load and cross-check only frozen JSON, sidecars, INI and model source."""
    root = Path(experiment).resolve()
    for folder in (root, root / "pilot"):
        for name in ("INCOMPLETE", "failure.json"):
            if (folder / name).exists():
                raise ValueError(f"Experiment has an incomplete or failed stage marker: {folder / name}")
    relative = ("experiment-lock.json", "pilot/experiment-lock.json", "pilot/results.json",
                "pilot/classifier-history.json", "pilot/classifier-predictions.json")
    parsed, hashes = {}, {}
    for name in relative:
        path = root / name
        digest = fingerprint(path)
        sidecar = Path(str(path) + ".sha256")
        if sidecar.read_text().strip() != digest:
            raise ValueError(f"Checksum mismatch: {name}")
        parsed[name], hashes[name] = json.loads(path.read_text()), digest
        hashes[str(sidecar.relative_to(root))] = fingerprint(sidecar)
    lock, results = parsed[relative[0]], parsed[relative[2]]
    history, artifact = parsed[relative[3]], parsed[relative[4]]
    if hashes[relative[0]] != hashes[relative[1]]:
        raise ValueError("Stage and root source locks differ")
    if not results.get("classifier_trained") or results.get("test_evaluated") is not False or artifact.get("test_evaluated") is not False:
        raise ValueError("Only a completed development run with no test evaluation is supported")
    for document in (results, artifact):
        if document.get("source_lock_sha256") != hashes[relative[0]]:
            raise ValueError("Results do not identify this frozen source lock")
    checkpoints = results.get("checkpoint_sha256", {})
    if not isinstance(checkpoints, dict) or checkpoints.get("classifier.keras") != artifact.get("checkpoint_sha256"):
        raise ValueError("Prediction and result checkpoints differ")
    selected = results.get("selected_epoch")
    if selected != history.get("selected_epoch") or selected != artifact.get("selected_epoch"):
        raise ValueError("Selected epochs differ across artifacts")
    epochs = history.get("epochs", [])
    if not epochs or [r.get("epoch") for r in epochs] != list(range(1, results.get("epochs_completed", 0) + 1)):
        raise ValueError("History must contain all completed epochs in order")
    chosen = min(epochs, key=lambda row: _finite(row["validation"]["bce"], "validation BCE", 0))
    if chosen["epoch"] != selected:
        raise ValueError("Selected checkpoint is not minimum validation BCE")
    predictions = checked_predictions(artifact.get("predictions"))
    if predictions != results.get("predictions") or predictions != chosen.get("predictions"):
        raise ValueError("Selected predictions disagree across artifacts")
    samples = {r["sample_id"]: r for r in lock.get("samples", [])}
    if set(samples) != {r["sample_id"] for r in predictions}:
        raise ValueError("Predictions do not cover exactly the frozen development sample panel")
    reserved = set(lock.get("reserved_test_sample_ids", []))
    if reserved.intersection(samples):
        raise ValueError("Reserved test samples appear in development predictions")
    family_splits = {}
    cutoff = _finite(lock["annotation_acceptance"]["threshold"], "engineering cutoff", 0, 1)
    for row in predictions:
        sample = samples[row["sample_id"]]
        if (row["split"], row["case_id"], row["family"], row["role"]) != (sample["split"], sample["case_id"], sample["anatomy_family"], sample["pilot_role"]):
            raise ValueError("Prediction identity differs from the frozen panel")
        annotation = sample["annotation"]
        if row["score_label"] != annotation["coa_local_target_volume_loss_fraction"] or row["label"] != int(annotation["coa_benchmark_positive"]):
            raise ValueError("Prediction targets differ from the frozen annotations")
        if row["label"] != int(row["score_label"] >= cutoff):
            raise ValueError("Class does not match the frozen achieved-loss cutoff")
        previous = family_splits.setdefault(row["family"], row["split"])
        if previous != row["split"]:
            raise ValueError("Anatomy family crosses development partitions")
    threshold = _finite(results["classification_decision_threshold"], "decision threshold", 0, 1)
    for split, key in (("train", "training"), ("validation", "validation")):
        rows = [r for r in predictions if r["split"] == split]
        if len(rows) != lock["pilot_counts"][split]["samples"]:
            raise ValueError("Prediction counts differ from the frozen pilot")
        _check_metrics(results[key], _metrics(rows, threshold))
        if results[key] != chosen[key]:
            raise ValueError("Selected metrics differ from history")
    for row in epochs:
        current = checked_predictions(row["predictions"])
        if {r["sample_id"] for r in current} != set(samples):
            raise ValueError("History sample panel changed during training")
        identities = {r["sample_id"]: r for r in predictions}
        for prediction in current:
            original = identities[prediction["sample_id"]]
            if any(prediction[field] != original[field] for field in ("case_id", "family", "split", "role", "label", "score_label")):
                raise ValueError("History prediction targets or identities changed")
        for split, key in (("train", "training"), ("validation", "validation")):
            _check_metrics(row[key], _metrics([r for r in current if r["split"] == split], threshold))
            for metric in ("bce", "accuracy", "auroc", "score_mae"):
                _finite(row[key][metric], f"history {metric}", 0)
    model_source = "source/src/fakect_coa_focused_network.py"
    hashes[model_source] = fingerprint(root / model_source)
    if hashes[model_source] != lock["code_sha256"]["src/fakect_coa_focused_network.py"]:
        raise ValueError("Frozen model source differs from the experiment source lock")
    hashes["input.ini"] = fingerprint(root / "input.ini")
    if hashes["input.ini"] != lock["input_sha256"]:
        raise ValueError("Frozen input differs from its source lock")
    return {"experiment": str(root), "lock": lock, "results": results, "history": history,
            "predictions": predictions, "source_hashes": hashes,
            "decision_threshold": threshold, "engineering_cutoff": cutoff}


def verify_sources(bundle):
    root = Path(bundle["experiment"])
    for name, expected in bundle["source_hashes"].items():
        if fingerprint(root / name) != expected:
            raise ValueError(f"Source changed during figure generation: {name}")


def _style():
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fakect-coa-poster-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13,
                         "axes.titlesize": 15, "axes.labelsize": 13,
                         "legend.fontsize": 11, "xtick.labelsize": 11, "ytick.labelsize": 11,
                         "axes.spines.top": False, "axes.spines.right": False,
                         "axes.grid": True, "grid.alpha": .18, "grid.linewidth": .7,
                         "pdf.fonttype": 42, "ps.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": SCHEMA,
                         "savefig.facecolor": "white", "figure.facecolor": "white"})
    return plt


def _save(fig, stem, output, plt):
    for extension in ("pdf", "svg", "png"):
        options = {"dpi": 300} if extension == "png" else {}
        if extension == "pdf":
            options["metadata"] = {"CreationDate": None, "ModDate": None, "Creator": SCHEMA}
        elif extension == "svg":
            options["metadata"] = {"Date": None, "Creator": SCHEMA}
        fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight", **options)
    plt.close(fig)


def _curves(bundle, output, plt):
    records, selected = bundle["history"]["epochs"], bundle["results"]["selected_epoch"]
    fig, axes = plt.subplots(2, 2, figsize=(12, 8), constrained_layout=True)
    metrics = (("bce", "A  Binary cross entropy", 1), ("auroc", "B  ROC AUC", 1),
               ("accuracy", "C  Accuracy at P(CoA) ≥ 0.5", 100),
               ("score_mae", "D  Severity-proxy error", 100))
    for ax, (key, title, multiplier) in zip(axes.flat, metrics):
        for split, label, color in (("training", "Training (32)", TRAIN), ("validation", "Validation (8)", VALIDATION)):
            ax.plot([r["epoch"] for r in records], [r[split][key] * multiplier for r in records], color=color, lw=2, label=label)
        ax.axvline(selected, color=GREY, ls="--", lw=1.4, label=f"Selected epoch {selected}")
        ax.set(title=title, xlabel="Epoch", xlim=(1, len(records)))
        if key == "bce":
            ax.set_yscale("log")
            ax.set_ylabel("BCE (log scale)")
        elif key == "score_mae":
            ax.set_ylabel("MAE (percentage points)")
        elif key == "accuracy":
            ax.set(ylim=(0, 103), ylabel="Accuracy (%)")
        else:
            ax.set(ylim=(0, 1.03), ylabel="ROC AUC")
    axes[0, 0].legend(loc="upper left", frameon=True)
    fig.suptitle("Development pilot • checkpoint selected by validation BCE", fontsize=17)
    _save(fig, "fig01-training-validation", output, plt)


def _roc(bundle, output, plt):
    fig, ax = plt.subplots(figsize=(7, 6.4), constrained_layout=True)
    roc_data = {}
    for split, label, color in (("train", "Training apparent", TRAIN), ("validation", "Validation development", VALIDATION)):
        rows = [r for r in bundle["predictions"] if r["split"] == split]
        roc = empirical_roc([r["label"] for r in rows], [r["probability"] for r in rows])
        roc_data[split] = roc
        ax.plot(roc["fpr"], roc["tpr"], marker="o", ms=5, lw=2.5, color=color,
                label=f"{label}\nAUC {roc['auroc']:.3f} • n = {len(rows)}")
        counts = confusion_counts(rows, bundle["decision_threshold"])
        ax.scatter(counts["FP"] / roc["negative"], counts["TP"] / roc["positive"],
                   s=115, marker="s", facecolors="none", edgecolors=color, linewidths=2, zorder=5)
    ax.plot([0, 1], [0, 1], color=GREY, lw=1.2, ls="--", label="Chance")
    ax.set(xlabel="False-positive rate", ylabel="True-positive rate", xlim=(-.02, 1.02), ylim=(-.02, 1.02),
           title=f"Observed ROC • selected epoch {bundle['results']['selected_epoch']}")
    ax.set_aspect("equal")
    ax.legend(loc="lower right", framealpha=.94)
    ax.text(.03, .05, f"□ decision threshold {bundle['decision_threshold']:g}", transform=ax.transAxes, color=GREY, fontsize=11)
    _save(fig, "fig02-empirical-roc", output, plt)
    return roc_data


def _validation(bundle, output, plt):
    rows = [r for r in bundle["predictions"] if r["split"] == "validation"]
    counts = confusion_counts(rows, bundle["decision_threshold"])
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.7), constrained_layout=True, gridspec_kw={"width_ratios": [1, 1.65]})
    matrix = np.array([[counts["TN"], counts["FP"]], [counts["FN"], counts["TP"]]])
    axes[0].imshow(matrix, cmap="Blues", vmin=0, vmax=max(4, matrix.max()))
    names = (("TN", "FP"), ("FN", "TP"))
    for i in range(2):
        for j in range(2):
            axes[0].text(j, i, f"{names[i][j]}\n{matrix[i, j]}", ha="center", va="center", fontsize=22,
                         color="white" if matrix[i, j] >= 3 else "#17212B", weight="bold")
    axes[0].set(xticks=[0, 1], yticks=[0, 1], xticklabels=["Non-CoA", "CoA"], yticklabels=["Non-CoA", "CoA"],
                xlabel="Predicted", ylabel="Reference", title="A  Validation counts")
    axes[0].grid(False)
    ax = axes[1]
    cases = sorted({r["case_id"] for r in rows})
    short_roles = ["Original", "Edited −", "Lower +", "Higher +"]
    labels = []
    for case_no, case in enumerate(cases):
        for role_no, role in enumerate(ROLES):
            row = next(r for r in rows if r["case_id"] == case and r["role"] == role)
            x = case_no * 5 + role_no
            color = VALIDATION if row["label"] else TRAIN
            ax.scatter(x, row["probability"], color=color, s=90, marker="o" if case_no == 0 else "D", zorder=4)
            precision = 4 if abs(row["probability"] - bundle["decision_threshold"]) < .001 else 3
            ax.text(x, row["probability"] + .025, f"{row['probability']:.{precision}f}", ha="center", fontsize=11)
            labels.append(short_roles[role_no])
    ax.axhline(bundle["decision_threshold"], color=GREY, ls="--", lw=1.4)
    ax.axvline(4, color=GREY, lw=.7, alpha=.4)
    ax.set(xticks=[0, 1, 2, 3, 5, 6, 7, 8], xticklabels=labels, ylabel="Predicted P(CoA)",
           ylim=(0, .75), title="B  Eight adult validation scenarios")
    ax.tick_params(axis="x", rotation=35)
    for x, case in zip((1.5, 6.5), cases):
        ax.text(x, .7, f"Phantom {case}", ha="center", fontsize=12, weight="bold")
    ax.text(.015, .025, "Blue: negative reference • orange: positive reference", transform=ax.transAxes, fontsize=11)
    sensitivity = counts["TP"] / (counts["TP"] + counts["FN"])
    specificity = counts["TN"] / (counts["TN"] + counts["FP"])
    fig.suptitle(f"Development validation • {sensitivity:.0%} sensitivity / {specificity:.0%} specificity", fontsize=17)
    _save(fig, "fig03-validation-errors", output, plt)


def _severity(bundle, output, plt):
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.1), constrained_layout=True)
    for split, label, color, marker in (("train", "Training", TRAIN, "o"), ("validation", "Validation", VALIDATION, "^")):
        rows = [r for r in bundle["predictions"] if r["split"] == split]
        truth = [r["score_label"] * 100 for r in rows]
        axes[0].scatter(truth, [r["probability"] for r in rows], c=color, marker=marker, s=65 if split == "train" else 105,
                        label=f"{label} ({len(rows)})", edgecolors="white", lw=.6, alpha=.8, zorder=4)
        axes[1].scatter(truth, [r["score_prediction"] * 100 for r in rows], c=color, marker=marker,
                        s=65 if split == "train" else 105, label=label, edgecolors="white", lw=.6, alpha=.8, zorder=4)
    limit = math.ceil(max(r["score_label"] for r in bundle["predictions"]) * 10) * 10
    axes[0].axvline(bundle["engineering_cutoff"] * 100, color=GREY, ls="--", lw=1.2)
    axes[0].axhline(bundle["decision_threshold"], color=GREY, ls=":", lw=1.2)
    axes[0].set(xlabel="Achieved local target-volume loss (%)", ylabel="Predicted P(CoA)",
                xlim=(-2, limit + 2), ylim=(-.03, 1.04), title="A  Detection versus induced change")
    axes[0].legend(loc="lower right")
    axes[1].plot([0, limit], [0, limit], color=GREY, ls="--", lw=1.2, label="Parity")
    axes[1].set(xlabel="Achieved local target-volume loss (%)", ylabel="Predicted local target-volume loss (%)",
                xlim=(-2, limit + 2), ylim=(-2, limit + 2), title="B  Severity head collapsed near zero")
    axes[1].legend(loc="upper left")
    axes[1].text(.04, .68, f"Validation MAE\n{bundle['results']['validation']['score_mae'] * 100:.1f} percentage points", transform=axes[1].transAxes, fontsize=14)
    fig.suptitle("Engineering severity proxy • not a clinical grading threshold", fontsize=17)
    _save(fig, "fig04-detection-and-severity", output, plt)


def _partition(bundle, output, plt):
    from matplotlib.patches import FancyBboxPatch
    design = bundle["lock"]["design"]
    assignment = design["folds"][0]["family_assignment"]
    families = ["xcat_reference_1yr", "xcat_reference_5yr", "xcat_reference_10yr", "xcat_reference_15yr", "xcat_reference_adult", "xcat_reference_infant"]
    names = ["1 year", "5 years", "10 years", "15 years", "Adult", "Infant"]
    colors = {"train": TRAIN, "validation": VALIDATION, "test": GREY}
    fig, ax = plt.subplots(figsize=(10.5, 7), constrained_layout=True)
    table = []
    for i, (family, name) in enumerate(zip(families, names)):
        split, cases, y = assignment[family], sorted(design["families"][family]), 5 - i
        ax.text(-.15, y + .3, name, ha="right", va="center", fontsize=15, weight="bold")
        for j, case in enumerate(cases):
            ax.add_patch(FancyBboxPatch((j * 1.35, y), 1.2, .6, boxstyle="round,pad=.025", facecolor=colors[split], edgecolor="none"))
            ax.text(j * 1.35 + .6, y + .3, case, color="white", ha="center", va="center", fontsize=15, weight="bold")
            table.append({"family": family, "case_id": case, "split": split,
                          "development_scenarios": 0 if split == "test" else len([r for r in bundle["predictions"] if r["case_id"] == case]),
                          "test_prediction_read": False})
        text = "Training • 8 scenarios" if split == "train" else "Validation • 8 scenarios" if split == "validation" else "Reserved test • not evaluated"
        ax.text(2.9, y + .3, text, va="center", color=colors[split], fontsize=14)
    ax.text(-.15, -1, "32 training scenarios / 8 validation scenarios\n12 baselines grouped into 6 provisional anatomy families", fontsize=16, va="center")
    ax.text(-.15, -1.9, "Each development phantom: original + edited negative + lower positive + higher positive", fontsize=12)
    ax.set(xlim=(-1.2, 6.4), ylim=(-2.25, 6), title="Family-separated development split")
    ax.axis("off")
    _save(fig, "fig05-cohort-partition", output, plt)
    return table


def _architecture(bundle, output, plt):
    from matplotlib.patches import FancyBboxPatch
    fig, ax = plt.subplots(figsize=(13, 4.8), constrained_layout=True)
    boxes = [(0, 1.6, 1.9, .95, "Whole-body\nattenuation"),
             (2.35, 1.6, 2, .95, "Frozen locator\nlow-resolution context"),
             (4.8, 1.6, 2, .95, "Two fine CT blocks\n216 × 192 × 192"),
             (7.25, 1.6, 2.25, .95, "Shared 3D encoder\nstride 4 · 2.4 mm"),
             (9.95, 1.6, 2.45, .95, "Local evidence\nordered 6³ spatial bins"),
             (9.95, .15, 2.45, .85, "CoA probability\n+ volume-loss proxy")]
    for i, (x, y, width, height, text) in enumerate(boxes):
        color = GREY if i < 2 else TRAIN if i < 5 else VALIDATION
        ax.add_patch(FancyBboxPatch((x, y), width, height, boxstyle="round,pad=.04", facecolor=color, edgecolor="none"))
        ax.text(x + width / 2, y + height / 2, text, ha="center", va="center", color="white", fontsize=12)
        if 0 < i < 5:
            previous = boxes[i - 1]
            ax.annotate("", xy=(x - .07, y + height / 2), xytext=(previous[0] + previous[2] + .08, y + height / 2), arrowprops={"arrowstyle": "->", "lw": 1.7, "color": GREY})
    ax.annotate("", xy=(11.18, 1.05), xytext=(11.18, 1.54), arrowprops={"arrowstyle": "->", "lw": 1.7, "color": GREY})
    ax.text(0, .55, "Predictors: attenuation + acquisition validity\nAnatomy masks and edit parameters supply training targets only", fontsize=13, va="center")
    ax.text(0, -.1, "Balanced same-phantom pairs train together; inference takes one scan independently", fontsize=13)
    ax.set(xlim=(-.2, 12.7), ylim=(-.6, 3.2), title="Image-only focused CoA classifier")
    ax.axis("off")
    _save(fig, "fig06-model-architecture", output, plt)


def _write_csv(path, rows, keys=None):
    if not rows:
        raise ValueError("Cannot export empty figure data")
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=keys or list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build_figures(experiment, output):
    """Create a new immutable export revision; existing destinations are refused."""
    target = Path(output).resolve()
    generator_root = Path(__file__).resolve().parent.parent
    generators = [Path(__file__).resolve(), generator_root / "scripts/research/evaluation/build_coa_poster_figures.py"]
    generator_hashes = {str(p.relative_to(generator_root)): fingerprint(p) for p in generators}
    if fingerprint(__file__) != LOADED_MODULE_SHA256:
        raise ValueError("Loaded figure generator source changed before export")
    bundle = source_bundle(experiment)
    if target.exists():
        raise FileExistsError(f"Output exists; choose a new poster revision: {target}")
    if target == Path(bundle["experiment"]) or Path(bundle["experiment"]) in target.parents:
        raise ValueError("Poster exports must be outside the frozen experiment")
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{target.name}-building-", dir=target.parent))
    try:
        plt = _style()
        _curves(bundle, temporary, plt)
        roc = _roc(bundle, temporary, plt)
        _validation(bundle, temporary, plt)
        _severity(bundle, temporary, plt)
        partitions = _partition(bundle, temporary, plt)
        _architecture(bundle, temporary, plt)
        history_rows = [{"epoch": r["epoch"], "split": split, "bce": r[key]["bce"],
                         "auroc": r[key]["auroc"], "accuracy": r[key]["accuracy"],
                         "severity_mae_percentage_points": r[key]["score_mae"] * 100,
                         "selected_checkpoint": r["epoch"] == bundle["results"]["selected_epoch"]}
                        for r in bundle["history"]["epochs"] for split, key in (("train", "training"), ("validation", "validation"))]
        roc_rows = [{"split": split, "threshold": threshold, "fpr": fpr, "tpr": tpr}
                    for split, data in roc.items() for threshold, fpr, tpr in zip(data["thresholds"], data["fpr"], data["tpr"])]
        _write_csv(temporary / "training-curves.csv", history_rows)
        _write_csv(temporary / "roc-points.csv", roc_rows)
        _write_csv(temporary / "selected-predictions.csv", bundle["predictions"])
        _write_csv(temporary / "cohort-partitions.csv", partitions)
        data = {"schema_version": SCHEMA, "selected_epoch": bundle["results"]["selected_epoch"],
                "decision_threshold": bundle["decision_threshold"], "engineering_cutoff": bundle["engineering_cutoff"],
                "roc": roc, "training_curves": history_rows, "selected_predictions": bundle["predictions"],
                "cohort_partitions": partitions, "training_metrics": bundle["results"]["training"],
                "validation_metrics": bundle["results"]["validation"]}
        (temporary / "figure-data.json").write_text(json.dumps(data, indent=2, allow_nan=False) + "\n")
        titles = {"fig01-training-validation": "Training and validation progression",
                  "fig02-empirical-roc": "Exact empirical ROC",
                  "fig03-validation-errors": "Validation confusion and individual scenarios",
                  "fig04-detection-and-severity": "Detection and severity-proxy behavior",
                  "fig05-cohort-partition": "Family partition and reserved test cohort",
                  "fig06-model-architecture": "Image-only model architecture"}
        counts = confusion_counts([r for r in bundle["predictions"] if r["split"] == "validation"], bundle["decision_threshold"])
        validation_samples = bundle["results"]["validation"]["samples"]
        captions = {
            "fig01-training-validation": f"All {bundle['results']['epochs_completed']} completed epochs. Epoch {data['selected_epoch']} minimizes development validation BCE; subsequent instability remains visible. Training estimates are apparent performance.",
            "fig02-empirical-roc": f"Observed thresholds only, with no smoothing. Training AUROC {roc['train']['auroc']:.3f}; development validation AUROC {roc['validation']['auroc']:.3f} from {validation_samples} scenarios in one adult family. Squares show threshold {bundle['decision_threshold']:g}. No confidence interval is claimed from one validation family.",
            "fig03-validation-errors": f"At threshold {bundle['decision_threshold']:g}: TP {counts['TP']}, TN {counts['TN']}, FP {counts['FP']}, FN {counts['FN']}. Near-threshold probabilities use four decimal places; decisions use the exact recorded scores. Lower/higher describe achieved induced change, not clinical grades.",
            "fig04-detection-and-severity": f"The positive label uses at least {bundle['engineering_cutoff']:.0%} achieved local target-volume loss. Validation severity MAE: {bundle['results']['validation']['score_mae'] * 100:.2f} percentage points. The volume-loss proxy is an engineering label, not clinical severity.",
            "fig05-cohort-partition": "Phantoms and all related siblings remain in their anatomy-family partition. Family labels are provisional. Reserved infant metadata is depicted without reading test images, checkpoints or test predictions.",
            "fig06-model-architecture": "Schematic transcribed from the frozen network source. The frozen image locator supplies two 0.6-mm detail crops; CT attenuation and acquisition validity are the classifier inputs. Masks and edit labels are targets only."}
        cards = "\n".join(f'<article><h2>{html.escape(title)}</h2><p><a href="{stem}.pdf">PDF</a> · <a href="{stem}.svg">SVG</a> · <a href="{stem}.png">300 dpi PNG</a></p><img src="{stem}.svg" alt="{html.escape(title)}"><p>{html.escape(captions[stem])}</p></article>' for stem, title in titles.items())
        report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoA poster figures — development pilot</title><style>body{{font:17px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1.5rem;color:#17212b;background:#f6f8fa}}article{{background:white;padding:1.5rem;margin:1.5rem 0;border:1px solid #d8dee4;border-radius:8px}}img{{width:100%;height:auto}}a{{color:{TRAIN}}}h1,h2{{line-height:1.2}}.scope{{padding:1rem;background:#fff0e8;border-left:5px solid {VALIDATION}}}</style><h1>CoA development pilot: poster figures</h1><p class="scope">Measured development results from epoch {data['selected_epoch']}. Training: 32 scenarios / 4 anatomy families. Validation: 8 scenarios / 1 adult family. Reserved tests have not been evaluated. These results do not establish clinical generalization or a benefit of cohort size/diversity.</p><p>Portable vector PDF/SVG and 300 dpi PNG. Exact underlying data: <a href="figure-data.json">JSON</a> · <a href="roc-points.csv">ROC CSV</a> · <a href="training-curves.csv">learning CSV</a> · <a href="selected-predictions.csv">predictions CSV</a> · <a href="cohort-partitions.csv">partitions CSV</a>. <a href="provenance.json">Provenance and checksums</a>.</p>{cards}<p>Future fold averages, population-size curves and diversity comparisons require additional completed experiments. No synthetic performance curves have been drawn.</p></html>'''
        (temporary / "report.html").write_text(report)
        verify_sources(bundle)
        if any(fingerprint(generator_root / name) != expected for name, expected in generator_hashes.items()):
            raise ValueError("Figure generator source changed during export")
        files = {p.name: fingerprint(p) for p in sorted(temporary.iterdir()) if p.is_file()}
        provenance = {"schema_version": SCHEMA, "experiment_directory": bundle["experiment"],
                      "source_sha256": bundle["source_hashes"], "generator_sha256": generator_hashes,
                      "artifact_sha256": files, "source_verification_before_and_after": True,
                      "checkpoint_and_image_payloads_read": False, "test_predictions_read": False,
                      "tensorflow_imported": False, "figure_count": len(titles), "exports_per_figure": ["pdf", "svg", "png"],
                      "roc_method": "Exact unique observed thresholds, tied scores grouped, trapezoidal AUROC; no smoothing.",
                      "uncertainty": "No confidence intervals: one validation anatomy family.",
                      "status": "measured_development_results_only"}
        (temporary / "provenance.json").write_text(json.dumps(provenance, indent=2, allow_nan=False) + "\n")
        os.rename(temporary, target)
        return {"output": str(target), "figure_count": len(titles), "figure_exports": 3 * len(titles),
                "selected_epoch": data["selected_epoch"], "training_auroc": roc["train"]["auroc"],
                "validation_auroc": roc["validation"]["auroc"], "validation_confusion": confusion_counts([r for r in bundle["predictions"] if r["split"] == "validation"], bundle["decision_threshold"])}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
