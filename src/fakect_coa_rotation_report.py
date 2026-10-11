"""Audit and report development-only out-of-fold CoA predictions.

Inputs are prediction dictionaries and fold metadata, never imaging, masks,
models or reserved-test predictions. Seeds repeat training; siblings and seeds
are not independent patients. Family averages and ROC bands are descriptive,
with no independent-observation confidence intervals.
"""
from collections import defaultdict
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

from fakect_coa_poster_figures import empirical_roc, LOADED_MODULE_SHA256 as POSTER_HELPER_LOADED_SHA256


SCHEMA = "fakect.coa-rotation-report/1"
LOADED_MODULE_SHA256 = hashlib.sha256(Path(__file__).read_bytes()).hexdigest()
METRICS = ("auroc", "sensitivity", "specificity", "balanced_accuracy", "bce")
PALETTE = ("#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00")


def _number(value, name, lower=0., upper=1.):
    if isinstance(value, bool) or not isinstance(value, (int, float, np.number)) or not math.isfinite(value):
        raise ValueError(f"{name} must be a finite number")
    if lower is not None and value < lower or upper is not None and value > upper:
        raise ValueError(f"{name} is outside the permitted range")
    return float(value)


def _identity(row):
    return row["protocol"], row["fold_id"], row["seed"]


def _text(value, field):
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a nonempty string")
    return value


def _string_list(row, field):
    value = row.get(field, [])
    if not isinstance(value, list) or any(not isinstance(x, str) or not x for x in value) or len(value) != len(set(value)):
        raise ValueError(f"{field} must contain unique string IDs")
    return value


def rotation_manifest(design):
    """Adapt the metadata-only rotation planner's stage catalog to OOF panels.

    Inner-selection stages supply receipt metadata only. Their predictions never
    become performance records. Baseline locator references and unselected
    siblings likewise remain outside the six-scenario classification panel.
    """
    if not isinstance(design, dict) or not isinstance(design.get("stages"), list):
        raise ValueError("A rotation design requires a stage catalog")
    family_map = {r["case_id"]: r["anatomy_family"] for r in design["inventory"]}
    stages = {r["id"]: r for r in design["stages"]}
    if len(stages) != len(design["stages"]):
        raise ValueError("Duplicate rotation stage IDs")
    folds = []
    for stage in stages.values():
        if stage["stage"] == "inner_selection":
            continue
        if stage["stage"] not in ("outer_refit", "secondary_selection"):
            raise ValueError("Unknown rotation evaluation stage type")
        cases = stage["final_test_case_ids"]
        families = {family_map[c] for c in cases}
        if len(families) != 1:
            raise ValueError("Each planned outer fold must have one evaluation anatomy family")
        grouping = "family" if stage["stage"] == "outer_refit" else "case"
        if grouping == "case" and len(cases) != 1:
            raise ValueError("A secondary stage evaluates exactly one phantom")
        selection_ids = stage.get("parent_inner_stage_ids", []) if grouping == "family" else [stage["id"]]
        selection = []
        for identity in selection_ids:
            if identity not in stages:
                raise ValueError("Refit identifies an unknown inner-selection stage")
            selected = stages[identity]
            if (selected["protocol"], selected["seed"], selected["outer_fold_id"]) != (stage["protocol"], stage["seed"], stage["outer_fold_id"]):
                raise ValueError("Inner selection stage belongs to a different outer fold")
            if set(selected["validation_case_ids"]) & set(cases) or set(selected["train_case_ids"]) & set(cases):
                raise ValueError("Inner model selection exposed outer-evaluation cases")
            selection.append({"stage_id": identity, "validation_case_ids": selected["validation_case_ids"],
                              "training_case_ids": selected["train_case_ids"]})
        folds.append({"protocol": stage["protocol"], "fold_id": stage["outer_fold_id"], "seed": stage["seed"],
                      "outer_case": None if grouping == "family" else cases[0], "outer_family": next(iter(families)),
                      "grouping": grouping, "stage_id": stage["id"], "evaluation_case_ids": list(cases),
                      "evaluation_sample_ids": list(stage["final_test_sample_ids"]),
                      "training_case_ids": list(stage["train_case_ids"]),
                      "validation_case_ids": list(stage["validation_case_ids"]), "selection_stages": selection})
    targets = {}
    for sample in design.get("samples", []):
        annotation = sample["annotation"]
        targets[sample["sample_id"]] = {"case_id": sample["case_id"],
                                        "label": int(annotation["coa_benchmark_positive"]),
                                        "score_label": annotation["coa_local_target_volume_loss_fraction"]}
    return {"folds": folds, "family_by_case": family_map,
            "reserved_case_ids": list(design["reserved_case_ids"]),
            "class_threshold": design["prediction_threshold"], "sample_targets": targets,
            "stage_catalog": [{"stage_id": r["id"], "protocol": r["protocol"],
                               "fold_id": r["outer_fold_id"], "seed": r["seed"], "stage": r["stage"],
                               "validation_case_ids": r["validation_case_ids"]} for r in stages.values()],
            "source_schema_version": design.get("schema_version"),
            "annotation_definition": design.get("annotation_definition")}


def validate_manifest(fold_manifest):
    """Canonical manifest uses explicit train/inner-validation/evaluation panels.

    ``folds`` entries contain protocol, fold_id, seed, outer_case (or None),
    outer_family, grouping ('case'/'family'), evaluation_case_ids,
    evaluation_sample_ids, training_case_ids and validation_case_ids.
    The manifest supplies family_by_case, reserved_case_ids and class_threshold.
    """
    if isinstance(fold_manifest, dict) and "stages" in fold_manifest:
        fold_manifest = rotation_manifest(fold_manifest)
    if not isinstance(fold_manifest, dict) or not isinstance(fold_manifest.get("folds"), list):
        raise ValueError("fold_manifest requires a folds list")
    families = fold_manifest.get("family_by_case")
    if not isinstance(families, dict) or not families or any(not isinstance(k, str) or not isinstance(v, str) or not k or not v for k, v in families.items()):
        raise ValueError("family_by_case must identify every development phantom")
    reserved = set(_string_list(fold_manifest, "reserved_case_ids"))
    threshold = _number(fold_manifest.get("class_threshold", .5), "class_threshold")
    folds, keys, exposure = [], set(), {}
    for source in fold_manifest["folds"]:
        if not isinstance(source, dict):
            raise ValueError("Fold entries must be dictionaries")
        row = dict(source)
        for field in ("protocol", "fold_id", "outer_family"):
            _text(row.get(field), field)
        if type(row.get("seed")) is not int or row["seed"] < 0:
            raise ValueError("Fold seed must be a nonnegative integer")
        row.setdefault("outer_case", None)
        row.setdefault("grouping", "case" if row["outer_case"] else "family")
        if row["grouping"] not in ("case", "family"):
            raise ValueError("Fold grouping must be case or family")
        if row["outer_case"] is not None:
            _text(row["outer_case"], "outer_case")
        key = _identity(row)
        if key in keys:
            raise ValueError("Duplicate protocol/fold/seed manifest entry")
        keys.add(key)
        panels = {name: set(_string_list(row, name)) for name in ("evaluation_case_ids", "training_case_ids", "validation_case_ids")}
        if not panels["evaluation_case_ids"] or not _string_list(row, "evaluation_sample_ids"):
            raise ValueError("Each fold requires explicit evaluation cases and sample IDs")
        if any(panels[a] & panels[b] for a, b in (("evaluation_case_ids", "training_case_ids"), ("evaluation_case_ids", "validation_case_ids"), ("training_case_ids", "validation_case_ids"))):
            raise ValueError("Cases cross train, inner-validation or outer-evaluation panels")
        all_cases = set.union(*panels.values())
        if all_cases & reserved:
            raise ValueError("Reserved test cases occur in a development fold")
        if not all_cases <= set(families):
            raise ValueError("Fold includes a case absent from family_by_case")
        evaluation_families = {families[case] for case in panels["evaluation_case_ids"]}
        if evaluation_families != {row["outer_family"]}:
            raise ValueError("Fold evaluation cases differ from its outer family")
        if row["grouping"] == "case":
            if panels["evaluation_case_ids"] != {row["outer_case"]}:
                raise ValueError("Case fold must evaluate exactly its outer case")
        elif row["outer_case"] is not None:
            raise ValueError("Family fold outer_case must be None")
        if row["grouping"] == "family":
            if any(families[c] in evaluation_families for c in panels["training_case_ids"] | panels["validation_case_ids"]):
                raise ValueError("Held-out family appears in training or model selection")
            expected_family_cases = {c for c, f in families.items() if f == row["outer_family"] and c not in reserved}
            if panels["evaluation_case_ids"] != expected_family_cases:
                raise ValueError("A family fold must evaluate the complete development family")
        for selection in row.get("selection_stages", []):
            selection_cases = set(selection["training_case_ids"]) | set(selection["validation_case_ids"])
            if set(selection["training_case_ids"]) & set(selection["validation_case_ids"]) or not selection_cases <= set(families):
                raise ValueError("Inner stage cases cross roles or lack known family metadata")
            if selection_cases & (reserved | panels["evaluation_case_ids"]):
                raise ValueError("An inner selection stage uses reserved or outer cases")
            if row["grouping"] == "family" and any(families[c] in evaluation_families for c in selection_cases):
                raise ValueError("An inner selection stage uses the held-out family")
        for case in panels["evaluation_case_ids"]:
            previous = exposure.setdefault((row["protocol"], row["seed"], case), row["fold_id"])
            if previous != row["fold_id"]:
                raise ValueError("A case is outer-evaluated twice in one protocol and seed")
        folds.append(row)
    if not folds:
        raise ValueError("At least one planned fold is required")
    return {**fold_manifest, "folds": folds, "class_threshold": threshold,
            "reserved_case_ids": sorted(reserved), "family_by_case": dict(families)}


def _metrics(rows, threshold):
    counts = dict(TP=0, TN=0, FP=0, FN=0)
    for row in rows:
        predicted = row["probability"] >= threshold
        counts[("T" if predicted == bool(row["label"]) else "F") + ("P" if predicted else "N")] += 1
    positive, negative = counts["TP"] + counts["FN"], counts["TN"] + counts["FP"]
    sensitivity = counts["TP"] / positive if positive else None
    specificity = counts["TN"] / negative if negative else None
    epsilon = 2.0 ** -23
    clipped = lambda value: min(1 - epsilon, max(epsilon, value))
    bce = math.fsum(-r["label"] * math.log(clipped(r["probability"]))
                   - (1 - r["label"]) * math.log(clipped(1 - r["probability"])) for r in rows) / len(rows)
    roc = empirical_roc([r["label"] for r in rows], [r["probability"] for r in rows]) if positive and negative else None
    severity = [abs(r["score_prediction"] - r["score_label"]) for r in rows if r.get("score_prediction") is not None]
    coverage = [r["locator_coverage_passed"] for r in rows if "locator_coverage_passed" in r]
    return {"samples": len(rows), "cases": len({r["case_id"] for r in rows}), "positive": positive,
            "negative": negative, "auroc": roc["auroc"] if roc else None, "roc": roc,
            "sensitivity": sensitivity, "specificity": specificity,
            "balanced_accuracy": (sensitivity + specificity) / 2 if sensitivity is not None and specificity is not None else None,
            "accuracy": (counts["TP"] + counts["TN"]) / len(rows), "bce": bce,
            "confusion_counts": counts,
            "severity_mae_percentage_points": 100 * math.fsum(severity) / len(severity) if severity else None,
            "severity_prediction_samples": len(severity),
            "pipeline_failure_samples": sum(r.get("pipeline_failure", False) for r in rows),
            "locator_coverage_audited_samples": len(coverage),
            "locator_coverage_pass_rate": math.fsum(coverage) / len(coverage) if coverage else None}


def _mean(values):
    values = [v for v in values if v is not None]
    return math.fsum(values) / len(values) if values else None


def _summary(values):
    values = [v for v in values if v is not None]
    return {"mean": _mean(values), "descriptive_sd": float(np.std(values)) if values else None,
            "minimum": min(values) if values else None, "maximum": max(values) if values else None,
            "units": len(values), "confidence_interval": None}


def _roc_grid(roc, grid):
    # Use the upper TPR at repeated FPRs: the attainable ROC at each grid point.
    points = defaultdict(list)
    for fpr, tpr in zip(roc["fpr"], roc["tpr"]):
        points[fpr].append(tpr)
    xs = sorted(points)
    return np.interp(grid, xs, [max(points[x]) for x in xs])


def _receipts(receipts, folds, manifest):
    output, seen = [], set()
    catalog = {r["stage_id"]: r for r in manifest.get("stage_catalog", [])}
    for source in receipts:
        row = dict(source)
        stage_id = row.get("stage_id")
        if stage_id is not None:
            if stage_id not in catalog:
                raise ValueError("Receipt identifies an unknown planned stage")
            stage = catalog[stage_id]
            for field in ("protocol", "fold_id", "seed"):
                if field in row and row[field] != stage[field]:
                    raise ValueError("Receipt identity differs from its planned stage")
                row[field] = stage[field]
            row.setdefault("validation_case_ids", stage["validation_case_ids"])
            row.setdefault("stage", stage["stage"])
        key = _identity(row)
        receipt_key = key + (stage_id,)
        if key not in folds or receipt_key in seen:
            raise ValueError("Validation receipt has an unknown or duplicated fold identity")
        seen.add(receipt_key)
        fold = folds[key]
        selected_cases = set(_string_list(row, "validation_case_ids"))
        allowed_cases = set(fold["validation_case_ids"])
        for selection in fold.get("selection_stages", []):
            if stage_id == selection["stage_id"]:
                allowed_cases |= set(selection["validation_case_ids"])
        if not selected_cases <= allowed_cases:
            raise ValueError("Validation receipt includes a case outside the inner-validation panel")
        if selected_cases & set(fold["evaluation_case_ids"]):
            raise ValueError("Outer cases were used for checkpoint or threshold selection")
        row["performance_aggregation_use"] = "excluded_model_selection_metadata_only"
        output.append(row)
    return output


def aggregate_records(records, fold_manifest, *, validation_receipts=(), comparisons=()):
    """Audit complete OOF panels; partial reports may omit whole pending folds.

    Comparisons are dictionaries with ``left_protocol`` and ``right_protocol``.
    They require identical (seed, sample_id) panels and identical class/severity
    targets. Performance means first average seeds within each family, then give
    every evaluated family equal weight. Family spread is descriptive only.
    """
    manifest = validate_manifest(fold_manifest)
    folds = {_identity(f): f for f in manifest["folds"]}
    receipts = _receipts(validation_receipts, folds, manifest)
    rows, seen, grouped, sample_targets = [], set(), defaultdict(list), {}
    for source in list(records or []):
        if not isinstance(source, dict):
            raise ValueError("OOF records must be dictionaries")
        row = dict(source)
        for field in ("protocol", "fold_id", "sample_id", "case_id", "outer_family"):
            _text(row.get(field), field)
        if type(row.get("seed")) is not int or row["seed"] < 0:
            raise ValueError("Prediction seed must be a nonnegative integer")
        key = _identity(row)
        if key not in folds:
            raise ValueError("Prediction does not identify a planned outer fold")
        fold = folds[key]
        identity = key + (row["sample_id"],)
        if identity in seen:
            raise ValueError("Duplicate OOF sample within a fold and seed")
        seen.add(identity)
        if row.get("outer_case") != fold["outer_case"] or row["outer_family"] != fold["outer_family"]:
            raise ValueError("Prediction outer case/family differs from its planned fold")
        if row["case_id"] not in fold["evaluation_case_ids"] or row["sample_id"] not in fold["evaluation_sample_ids"]:
            raise ValueError("Prediction is outside its exact outer evaluation panel")
        if row["case_id"] in manifest["reserved_case_ids"]:
            raise ValueError("Reserved test predictions cannot enter a development report")
        if type(row.get("label")) is not int or row["label"] not in (0, 1):
            raise ValueError("OOF labels must be binary integers")
        row["probability"] = _number(row.get("probability"), "probability")
        row["score_label"] = _number(row.get("score_label"), "score_label")
        if row.get("score_prediction") is not None:
            row["score_prediction"] = _number(row["score_prediction"], "score_prediction")
        family = manifest["family_by_case"][row["case_id"]]
        if row.get("family", family) != family or family != row["outer_family"]:
            raise ValueError("Prediction phantom family differs from the canonical map")
        row["family"] = family
        if row.get("split", "outer_evaluation") not in ("outer_evaluation", "evaluation", "out_of_fold", "final_test"):
            raise ValueError("Training or inner-validation predictions are not OOF performance")
        for field in ("pipeline_failure", "locator_coverage_passed"):
            if field in row and type(row[field]) is not bool:
                raise ValueError(f"{field} must be a Boolean audit flag")
        if "pipeline_failure" in row and "locator_coverage_passed" in row and row["pipeline_failure"] != (not row["locator_coverage_passed"]):
            raise ValueError("Pipeline failure and locator coverage flags contradict")
        stable_target = (row["case_id"], family, row["label"], row["score_label"])
        previous = sample_targets.setdefault(row["sample_id"], stable_target)
        if previous != stable_target:
            raise ValueError("An OOF sample changes identity or target across protocols/seeds")
        expected = manifest.get("sample_targets", {}).get(row["sample_id"])
        if expected is not None and (row["case_id"] != expected["case_id"] or row["label"] != expected["label"] or row["score_label"] != expected["score_label"]):
            raise ValueError("Prediction disagrees with a frozen sample target")
        rows.append(row)
        grouped[key].append(row)
    for key, panel in grouped.items():
        fold = folds[key]
        if {r["sample_id"] for r in panel} != set(fold["evaluation_sample_ids"]) or {r["case_id"] for r in panel} != set(fold["evaluation_case_ids"]):
            raise ValueError("A completed fold has an incomplete or mismatched sample panel")
    rows.sort(key=lambda r: (r["protocol"], r["seed"], r["fold_id"], r["case_id"], r["sample_id"]))
    per_fold, per_case, per_family = [], [], []
    for key, panel in sorted(grouped.items()):
        per_fold.append({"protocol": key[0], "fold_id": key[1], "seed": key[2], **_metrics(panel, manifest["class_threshold"])})
    for fields, output in ((("protocol", "seed", "case_id"), per_case), (("protocol", "seed", "family"), per_family)):
        groups = defaultdict(list)
        for row in rows:
            groups[tuple(row[f] for f in fields)].append(row)
        for key, panel in sorted(groups.items()):
            output.append({**dict(zip(fields, key)), **_metrics(panel, manifest["class_threshold"])})
    protocols = {}
    failed_stages = [r for r in receipts if r.get("status") in ("failed", "failure", "blocked")]
    grid = np.linspace(0., 1., 101)
    for protocol in sorted({f["protocol"] for f in manifest["folds"]}):
        family_seed = [r for r in per_family if r["protocol"] == protocol]
        family_mean, family_curves = [], []
        for family in sorted({r["family"] for r in family_seed}):
            seeds = [r for r in family_seed if r["family"] == family]
            family_mean.append({"family": family, "seeds": [r["seed"] for r in seeds],
                                **{metric: _mean([r[metric] for r in seeds]) for metric in METRICS}})
            seed_curves = [_roc_grid(r["roc"], grid) for r in seeds if r["roc"]]
            if seed_curves:
                family_curves.append(np.mean(seed_curves, axis=0))
        panel = [r for r in rows if r["protocol"] == protocol]
        if family_curves:
            curves = np.stack(family_curves)
            mean_roc = {"fpr_grid": grid.tolist(), "mean_tpr": np.mean(curves, axis=0).tolist(),
                        "descriptive_family_sd_tpr": np.std(curves, axis=0).tolist(),
                        "families": len(family_curves), "confidence_interval": None,
                        "method": "Average seeds within family, then average families equally; upper attainable TPR on a fixed 101-point FPR grid."}
        else:
            mean_roc = None
        expected = sum(f["protocol"] == protocol for f in folds.values())
        completed = sum(k[0] == protocol for k in grouped)
        failures = [r for r in failed_stages if r["protocol"] == protocol]
        status = "blocked_by_failed_stage" if failures and not completed else "partial_with_failed_stages" if failures else "pending" if not completed else "complete" if completed == expected else "partial"
        protocols[protocol] = {"planned_fold_runs": expected, "completed_fold_runs": completed,
                               "failed_stage_receipts": len(failures), "status": status,
                               "pooled_scenario_metrics": _metrics(panel, manifest["class_threshold"]) if panel else None,
                               "family_means": family_mean, "equal_family_metrics": {metric: _summary([r[metric] for r in family_mean]) for metric in METRICS},
                               "mean_roc": mean_roc, "grouping": sorted({f["grouping"] for f in folds.values() if f["protocol"] == protocol})}
    overall_status = "blocked_by_failed_stage" if failed_stages and not rows else "partial_with_failed_stages" if failed_stages else "pending" if not rows else "complete" if len(grouped) == len(folds) else "partial"
    analysis = {"schema_version": SCHEMA, "status": overall_status,
                "manifest": manifest, "records": rows, "validation_receipts": receipts,
                "failed_stages": failed_stages,
                "receipt_performance_use": "none", "per_fold": per_fold, "per_case": per_case, "per_family": per_family,
                "protocols": protocols, "paired_comparisons": [],
                "scope": "Development outer-fold evaluation only; reserved test data not loaded. Locator failures remain in the evaluated scenario panel.",
                "independent_observation_ci": None,
                "statistical_units": "Repeated seeds and siblings are correlated; family means/spread are descriptive."}
    for comparison in comparisons:
        if not rows:
            analysis["paired_comparisons"].append({**comparison, "status": "pending", "performance": None})
        else:
            analysis["paired_comparisons"].append(paired_comparison(analysis, comparison["left_protocol"], comparison["right_protocol"]))
    return analysis


def paired_comparison(analysis, left_protocol, right_protocol):
    """Compare identical OOF sample/seed panels, with right-minus-left deltas."""
    panels = {}
    for protocol in (left_protocol, right_protocol):
        panel = {(r["seed"], r["sample_id"]): r for r in analysis["records"] if r["protocol"] == protocol}
        if not panel:
            raise ValueError("Paired comparisons require measured predictions from both protocols")
        panels[protocol] = panel
    left, right = panels[left_protocol], panels[right_protocol]
    if set(left) != set(right):
        raise ValueError("Paired protocols require identical seed and sample panels")
    for key in left:
        if any(left[key][f] != right[key][f] for f in ("case_id", "family", "label", "score_label")):
            raise ValueError("Paired protocols have inconsistent sample identities or targets")
    family_rows, case_rows = [], []
    for field, output in (("family", family_rows), ("case_id", case_rows)):
        for unit in sorted({r[field] for r in left.values()}):
            seed_deltas = []
            for seed in sorted({key[0] for key in left}):
                l = [r for r in left.values() if r[field] == unit and r["seed"] == seed]
                r = [r for r in right.values() if r[field] == unit and r["seed"] == seed]
                if not l:
                    continue
                a, b = _metrics(l, analysis["manifest"]["class_threshold"]), _metrics(r, analysis["manifest"]["class_threshold"])
                seed_deltas.append({m: b[m] - a[m] if a[m] is not None and b[m] is not None else None for m in METRICS})
            output.append({field: unit, **{m: _mean([r[m] for r in seed_deltas]) for m in METRICS}})
    return {"left_protocol": left_protocol, "right_protocol": right_protocol, "status": "measured_identical_panel",
            "delta_direction": "right_minus_left", "matched_sample_seed_pairs": len(left),
            "per_case_deltas": case_rows, "per_family_deltas": family_rows,
            "equal_family_delta": {m: _summary([r[m] for r in family_rows]) for m in METRICS},
            "confidence_interval": None}


def _hash(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def _canonical_hash(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _csv(path, rows, fields):
    with Path(path).open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _plots(analysis, output):
    os.environ.setdefault("MPLCONFIGDIR", str(Path(tempfile.gettempdir()) / "fakect-coa-rotation-mpl"))
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcdefaults()
    plt.rcParams.update({"font.family": "DejaVu Sans", "font.size": 13, "axes.titlesize": 15,
                         "axes.spines.top": False, "axes.spines.right": False, "axes.grid": True,
                         "grid.alpha": .16, "pdf.fonttype": 42, "svg.fonttype": "none", "svg.hashsalt": SCHEMA})
    exports = []
    def save(fig, stem, title):
        for extension in ("pdf", "svg", "png"):
            options = {"dpi": 300} if extension == "png" else {"metadata": {"CreationDate": None, "ModDate": None}} if extension == "pdf" else {"metadata": {"Date": None}}
            fig.savefig(output / f"{stem}.{extension}", bbox_inches="tight", **options)
        plt.close(fig)
        exports.append({"stem": stem, "title": title})
    measured = [(k, v) for k, v in analysis["protocols"].items() if v["completed_fold_runs"]]
    fig, axes = plt.subplots(1, 2, figsize=(12, 5.2), constrained_layout=True)
    for i, (protocol, summary) in enumerate(measured):
        color = PALETTE[i % len(PALETTE)]
        roc = summary["pooled_scenario_metrics"]["roc"]
        if roc:
            axes[0].plot(roc["fpr"], roc["tpr"], color=color, lw=2, marker="o", ms=3,
                         label=f"{protocol}\nPooled AUC {roc['auroc']:.3f}")
        mean = summary["mean_roc"]
        if mean:
            grid, curve, spread = np.asarray(mean["fpr_grid"]), np.asarray(mean["mean_tpr"]), np.asarray(mean["descriptive_family_sd_tpr"])
            axes[1].plot(grid, curve, color=color, lw=2, label=f"{protocol} • {mean['families']} families")
            axes[1].fill_between(grid, np.clip(curve - spread, 0, 1), np.clip(curve + spread, 0, 1), color=color, alpha=.15)
    for ax, title in zip(axes, ("A  Exact pooled out-of-fold ROC", "B  Equal-family mean ROC ± descriptive SD")):
        ax.plot([0, 1], [0, 1], ls="--", color="#737980", lw=1)
        ax.set(xlim=(0, 1), ylim=(0, 1), xlabel="False-positive rate", ylabel="True-positive rate", title=title)
        ax.legend(loc="lower right", fontsize=10)
    save(fig, "rotation01-roc", "Out-of-fold ROC and descriptive family spread")
    fig, axes = plt.subplots(2, 2, figsize=(11, 7.5), constrained_layout=True)
    for ax, metric in zip(axes.flat, ("auroc", "balanced_accuracy", "sensitivity", "specificity")):
        for i, (protocol, summary) in enumerate(measured):
            values = [r[metric] for r in summary["family_means"] if r[metric] is not None]
            offsets = np.linspace(-.14, .14, len(values)) if len(values) > 1 else [0]
            ax.scatter(np.array(offsets) + i, values, color=PALETTE[i % len(PALETTE)], s=50)
            if values:
                ax.plot([i - .2, i + .2], [_mean(values)] * 2, color="#17212b", lw=2.5)
        ax.set(xticks=list(range(len(measured))), xticklabels=[k for k, _ in measured], ylim=(-.03, 1.03),
               title=metric.replace("_", " ").title(), ylabel="Family mean (seeds averaged)")
        ax.tick_params(axis="x", rotation=20)
    fig.suptitle("Each point is one anatomy family • black line is equal-family mean", fontsize=15)
    save(fig, "rotation02-family-metrics", "Performance by anatomy family")
    fig, axes = plt.subplots(1, len(measured), figsize=(max(7, len(measured) * 5.5), 5.2), squeeze=False, constrained_layout=True)
    behavior = []
    for i, (protocol, _) in enumerate(measured):
        groups = defaultdict(list)
        for row in analysis["records"]:
            if row["protocol"] == protocol:
                groups[row["sample_id"]].append(row)
        case_points = defaultdict(list)
        for identity, rows in sorted(groups.items()):
            point = {"protocol": protocol, "sample_id": identity, "case_id": rows[0]["case_id"],
                     "score_label": rows[0]["score_label"], "probability_mean": _mean([r["probability"] for r in rows]),
                     "descriptive_seed_sd": float(np.std([r["probability"] for r in rows])), "seeds": len(rows)}
            behavior.append(point)
            case_points[point["case_id"]].append(point)
        ax = axes[0, i]
        for j, (case, points) in enumerate(sorted(case_points.items())):
            points.sort(key=lambda r: r["score_label"])
            ax.errorbar([r["score_label"] * 100 for r in points], [r["probability_mean"] for r in points],
                        yerr=[r["descriptive_seed_sd"] for r in points], fmt="o", ms=5,
                        color=PALETTE[j % len(PALETTE)], alpha=.75, label=case)
        ax.axhline(analysis["manifest"]["class_threshold"], color="#737980", ls="--", lw=1)
        ax.set(title=protocol, xlabel="Achieved local target-volume loss (%)", ylabel="Out-of-fold P(CoA)", ylim=(-.03, 1.03))
    fig.suptitle("Scenario means across seeds • bars show descriptive seed SD", fontsize=15)
    save(fig, "rotation03-induced-change", "Probability versus achieved induced change")
    comparisons = [r for r in analysis["paired_comparisons"] if r.get("status") == "measured_identical_panel"]
    if comparisons:
        fig, axes = plt.subplots(1, len(comparisons), figsize=(max(7, len(comparisons) * 6), 5), squeeze=False, constrained_layout=True)
        for i, comparison in enumerate(comparisons):
            ax = axes[0, i]
            for j, metric in enumerate(("auroc", "balanced_accuracy", "sensitivity", "specificity")):
                values = [r[metric] for r in comparison["per_family_deltas"] if r[metric] is not None]
                offsets = np.linspace(-.13, .13, len(values)) if len(values) > 1 else [0]
                ax.scatter(np.array(offsets) + j, values, color=PALETTE[j], s=55)
                if values:
                    ax.plot([j - .2, j + .2], [_mean(values)] * 2, color="#17212b", lw=2)
            ax.axhline(0, color="#737980", ls="--", lw=1)
            ax.set(xticks=list(range(4)), xticklabels=["AUC", "Balanced\naccuracy", "Sensitivity", "Specificity"],
                   ylabel="Right − left (equal-family units)", title=f"{comparison['right_protocol']} − {comparison['left_protocol']}")
        save(fig, "rotation04-paired-comparison", "Paired protocol comparisons on identical scenario panels")
    return exports, behavior


def write_rotation_report(output_directory, records=None, fold_manifest=None, *, validation_receipts=(), comparisons=(), provenance=None):
    """Write a new immutable report revision; pending plans contain no plots."""
    helper = Path(__file__).parent / "fakect_coa_poster_figures.py"
    generator_hash, helper_hash = _hash(__file__), _hash(helper)
    if generator_hash != LOADED_MODULE_SHA256 or helper_hash != POSTER_HELPER_LOADED_SHA256:
        raise ValueError("Loaded report source changed before export")
    analysis = aggregate_records(records, fold_manifest, validation_receipts=validation_receipts, comparisons=comparisons)
    output = Path(output_directory).resolve()
    if output.exists():
        raise FileExistsError("Rotation report exists; select a new report revision")
    output.parent.mkdir(parents=True, exist_ok=True)
    temporary = Path(tempfile.mkdtemp(prefix=f".{output.name}-building-", dir=output.parent))
    try:
        figures, behavior = _plots(analysis, temporary) if analysis["records"] else ([], [])
        for name, value in (("analysis.json", analysis), ("fold-manifest.json", analysis["manifest"]),
                            ("validation-receipts.json", analysis["validation_receipts"]), ("scenario-behavior.json", behavior)):
            (temporary / name).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")
        _csv(temporary / "out-of-fold-predictions.csv", analysis["records"],
             ["protocol", "fold_id", "outer_case", "outer_family", "seed", "sample_id", "case_id", "family", "label", "probability", "score_label", "score_prediction", "locator_coverage_passed", "pipeline_failure"])
        metrics = []
        for scope, rows in (("fold", analysis["per_fold"]), ("case", analysis["per_case"]), ("family", analysis["per_family"])):
            metrics.extend({"scope": scope, **r} for r in rows)
        _csv(temporary / "out-of-fold-metrics.csv", metrics, ["scope", "protocol", "fold_id", "seed", "case_id", "family", "samples", "auroc", "sensitivity", "specificity", "balanced_accuracy", "bce", "severity_mae_percentage_points", "pipeline_failure_samples", "locator_coverage_audited_samples", "locator_coverage_pass_rate"])
        _csv(temporary / "scenario-behavior.csv", behavior, ["protocol", "sample_id", "case_id", "score_label", "probability_mean", "descriptive_seed_sd", "seeds"])
        cards = "\n".join(f'<article><h2>{html.escape(r["title"])}</h2><p><a href="{r["stem"]}.pdf">PDF</a> · <a href="{r["stem"]}.svg">SVG</a> · <a href="{r["stem"]}.png">300 dpi PNG</a></p><img src="{r["stem"]}.svg" alt="{html.escape(r["title"])}"></article>' for r in figures)
        status = "No rotating evaluation results are available yet. This report records the planned folds only; it contains no performance curves." if not analysis["records"] else f"{len(analysis['per_fold'])} of {len(analysis['manifest']['folds'])} planned fold/seed runs have measured predictions. Status: {analysis['status']}."
        if analysis["failed_stages"]:
            status += f" {len(analysis['failed_stages'])} stages failed or are blocked; their folds have no completed performance result."
        failures = "".join(f"<li>{html.escape(str(r.get('stage_id',r['fold_id'])))}: {html.escape(str(r.get('status')))} — {html.escape(str(r.get('error',r.get('reason','See stage receipt'))))}</li>" for r in analysis["failed_stages"])
        details = "".join(f"<tr><td>{html.escape(name)}</td><td>{r['completed_fold_runs']} / {r['planned_fold_runs']}</td><td>{html.escape(r['status'])}</td><td>{html.escape(', '.join(r['grouping']))}</td></tr>" for name, r in analysis["protocols"].items())
        report = f'''<!doctype html><html lang="en"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>CoA rotating development evaluation</title><style>body{{font:17px/1.5 system-ui,sans-serif;max-width:1200px;margin:2rem auto;padding:0 1.5rem;background:#f6f8fa;color:#17212b}}article{{background:white;padding:1.5rem;border:1px solid #d8dee4;border-radius:8px;margin:1.5rem 0}}img{{width:100%;height:auto}}table{{border-collapse:collapse}}td,th{{padding:.5rem 1rem;border:1px solid #d8dee4}}a{{color:#0072b2}}.scope{{background:#fff0e8;border-left:5px solid #d55e00;padding:1rem}}</style><h1>CoA rotating development evaluation</h1><p class="scope">{html.escape(status)}</p><ul>{failures}</ul><p>Outer predictions measure development performance. Inner-validation receipts are selection metadata only and are excluded from every performance aggregate. Reserved infant/test image and prediction payloads are outside this reporter.</p><p>Related siblings remain together. Case-grouped rotation may expose another case from the same anatomy family during training; family-grouped rotation forbids that exposure. Seeds and siblings are correlated. Means and spread are descriptive; no independent-patient confidence intervals are reported. Locator coverage failures remain in the exact evaluated scenario panel and are counted in the metric exports.</p><table><tr><th>Protocol</th><th>Completed fold runs</th><th>Status</th><th>Grouping</th></tr>{details}</table><p>Exact data: <a href="analysis.json">analysis JSON</a> · <a href="fold-manifest.json">fold manifest</a> · <a href="out-of-fold-predictions.csv">predictions CSV</a> · <a href="out-of-fold-metrics.csv">metrics CSV</a> · <a href="scenario-behavior.csv">scenario behavior CSV</a> · <a href="validation-receipts.json">selection receipts</a> · <a href="provenance.json">provenance</a>.</p>{cards}<p>Clinical generalization and a benefit of population size or diversity require their corresponding completed comparisons.</p></html>'''
        (temporary / "report.html").write_text(report)
        if _hash(__file__) != generator_hash or _hash(helper) != helper_hash:
            raise ValueError("Report generator source changed during export")
        sources = {"records_canonical_sha256": _canonical_hash(analysis["records"]),
                   "fold_manifest_canonical_sha256": _canonical_hash(analysis["manifest"]),
                   "validation_receipts_canonical_sha256": _canonical_hash(analysis["validation_receipts"]),
                   "generator_sha256": generator_hash, "poster_roc_helper_sha256": helper_hash}
        proof = {"schema_version": SCHEMA, "status": analysis["status"], "source_fingerprints": sources,
                 "caller_provenance": provenance or {}, "image_or_checkpoint_payloads_read": False,
                 "reserved_test_predictions_read": False, "validation_receipts_used_for_performance": False,
                 "independent_observation_ci": None, "figure_count": len(figures),
                 "artifact_sha256": {p.name: _hash(p) for p in sorted(temporary.iterdir())}}
        (temporary / "provenance.json").write_text(json.dumps(proof, indent=2, allow_nan=False) + "\n")
        os.rename(temporary, output)
        return {"output_directory": str(output), "status": analysis["status"], "figure_count": len(figures),
                "records": len(analysis["records"]), "completed_fold_runs": len(analysis["per_fold"])}
    except BaseException:
        shutil.rmtree(temporary, ignore_errors=True)
        raise
