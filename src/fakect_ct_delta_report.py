"""Offline PCA review of baseline-relative CT differences in synthetic siblings.

This reporter never fits an embedding or clusters, and never reads a training
checkpoint. Anatomical mask counts, if provided, are posthoc annotations only.
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

SCHEMA = "fakect.ct-delta-report/1"
COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00",
          "#56B4E9", "#332288", "#882255", "#44AA99", "#999933"]
METRIC_ALIASES = {
    "ct_relative_l1": ("ct_relative_l1", "relative_l1", "ct_l1_relative", "relative_ct_l1"),
    "ct_positive_l1": ("ct_positive_l1", "positive_l1", "positive_relative_l1", "relative_positive_l1", "normalized_positive_ct_energy"),
    "ct_negative_l1": ("ct_negative_l1", "negative_l1", "negative_relative_l1", "relative_negative_l1", "normalized_negative_ct_energy"),
}


def _jsonable(value):
    if isinstance(value, np.ndarray):
        return [_jsonable(v) for v in value.tolist()]
    if isinstance(value, np.generic):
        return _jsonable(value.item())
    if isinstance(value, dict):
        return {str(k): _jsonable(v) for k, v in value.items()}
    if isinstance(value, (list, tuple)):
        return [_jsonable(v) for v in value]
    if isinstance(value, float) and not math.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _metric(row, name):
    for key in METRIC_ALIASES.get(name, (name,)):
        value = row.get(key)
        if value is not None:
            try:
                value = float(value)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                return value
    if name == "ct_relative_l1":
        positive, negative = _metric(row, "ct_positive_l1"), _metric(row, "ct_negative_l1")
        if math.isfinite(positive) and math.isfinite(negative):
            return positive + negative
    return math.nan


def _validated(analysis, records, scores, labels, baseline_records, baseline_scores):
    records = [dict(r) for r in records]
    scores = np.asarray(scores, dtype=float)
    labels = np.asarray(labels)
    if not records or scores.ndim != 2 or scores.shape[0] != len(records) or scores.shape[1] < 2:
        raise ValueError("Every sibling requires at least two matching PCA coordinates")
    if not np.isfinite(scores).all():
        raise ValueError("PCA coordinates must be finite")
    if labels.shape != (len(records),) or labels.dtype.kind not in "iu" or np.any(labels < 0):
        raise ValueError("Every sibling requires an integer cluster assignment")
    if len({r.get("sample_id") for r in records}) != len(records) or any(
            not r.get("sample_id") or not r.get("case_id") for r in records):
        raise ValueError("Sibling sample IDs must be unique and case IDs present")
    baselines = [dict(r) for r in (baseline_records or [])]
    baseline_scores = np.empty((0, scores.shape[1])) if baseline_scores is None else np.asarray(baseline_scores, dtype=float)
    if baseline_scores.ndim != 2 or len(baseline_scores) != len(baselines) or (
            len(baselines) and (baseline_scores.shape[1] < 2 or not np.isfinite(baseline_scores).all())):
        raise ValueError("Baseline markers require matching finite PCA coordinates")
    pca = analysis.get("pca", {})
    explained = analysis.get("explained_variance_ratio", pca.get("explained_variance_ratio", []))
    if len(explained) < 2 or any(not math.isfinite(float(v)) or not 0 <= float(v) <= 1 for v in explained[:2]):
        raise ValueError("The PCA explained variance ratios are required")
    return records, scores, labels.astype(int), baselines, baseline_scores, np.asarray(explained, dtype=float)


def cluster_summary(records, labels):
    """Summarize CT-derived extent; anatomical counts never define clusters."""
    output = []
    for label in sorted(set(int(v) for v in labels)):
        rows = [r for r, lab in zip(records, labels) if lab == label]
        item = dict(cluster=label + 1, cluster_index=label, n=len(rows), case_counts={})
        for r in rows:
            case = str(r["case_id"])
            item["case_counts"][case] = item["case_counts"].get(case, 0) + 1
        for field in METRIC_ALIASES:
            values = [_metric(r, field) for r in rows]
            values = np.asarray([v for v in values if math.isfinite(v)])
            item[field] = None if not len(values) else dict(
                median=float(np.median(values)), q25=float(np.quantile(values, .25)),
                q75=float(np.quantile(values, .75)), maximum=float(np.max(values)))
        output.append(item)
    return output


def _csv(path, rows, fields):
    with path.open("w", newline="", encoding="utf-8") as stream:
        writer = csv.DictWriter(stream, fieldnames=fields)
        writer.writeheader()
        for row in rows:
            writer.writerow({key: json.dumps(_jsonable(row.get(key)), sort_keys=True)
                             if isinstance(row.get(key), (dict, list, tuple)) else row.get(key) for key in fields})


def _axis(ax, explained):
    ax.set_xlabel("PC1 ({:.1f}% variance)".format(100 * explained[0]))
    ax.set_ylabel("PC2 ({:.1f}% variance)".format(100 * explained[1]))
    ax.grid(alpha=.15, linewidth=.6)
    ax.spines[["top", "right"]].set_visible(False)


def _save(fig, directory, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(directory / (stem + "." + extension), dpi=300, bbox_inches="tight")


def _figures(directory, analysis, records, scores, labels, baselines, baseline_scores, explained):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fakect-ct-delta-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"font.size": 10, "svg.fonttype": "none", "pdf.fonttype": 42})
    unique = sorted(set(labels))
    colors = {label: COLORS[i % len(COLORS)] for i, label in enumerate(unique)}
    fig, ax = plt.subplots(figsize=(8.2, 6.1))
    for label in unique:
        mask = labels == label
        ax.scatter(scores[mask, 0], scores[mask, 1], s=11, alpha=.6, linewidths=0,
                   color=colors[label], label="Cluster {} (n={})".format(label + 1, sum(mask)))
    if len(baselines):
        ax.scatter(baseline_scores[:, 0], baseline_scores[:, 1], s=115, marker="*", c="#555555",
                   edgecolors="white", linewidths=.6, zorder=5,
                   label="{} unchanged baselines".format(len(baselines)))
    _axis(ax, explained)
    ax.set_title("Baseline-relative CT differences · {} siblings".format(len(records)))
    ax.legend(loc="best", fontsize=8, framealpha=.9, markerscale=1.5)
    fig.tight_layout()
    _save(fig, directory, "pca-clusters")
    plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.3))
    cases = sorted({str(r["case_id"]) for r in records})
    for i, case in enumerate(cases):
        mask = np.array([str(r["case_id"]) == case for r in records])
        axes[0].scatter(scores[mask, 0], scores[mask, 1], s=10, alpha=.6, linewidths=0,
                        color=COLORS[i % len(COLORS)], label=case)
    axes[0].legend(title="Baseline phantom", fontsize=7, ncol=2)
    axes[0].set_title("Same coordinates, colored by phantom")
    extents = np.array([_metric(r, "ct_relative_l1") for r in records])
    good = np.isfinite(extents) & (extents >= 0)
    if good.any():
        colorscale = np.log10(1 + 100 * extents[good])
        artist = axes[1].scatter(scores[good, 0], scores[good, 1], c=colorscale,
                                 s=12, alpha=.7, linewidths=0, cmap="viridis")
        fig.colorbar(artist, ax=axes[1], label="log₁₀(1 + 100 × relative CT L1)", shrink=.8)
    axes[1].set_title("Same coordinates, colored by CT-change extent")
    for ax in axes:
        _axis(ax, explained)
    fig.tight_layout()
    _save(fig, directory, "pca-phantom-extent")
    plt.close(fig)

    clustering = analysis.get("clustering", analysis)
    candidates = clustering.get("candidates", clustering.get("k_candidates", []))
    valid = [r for r in candidates if isinstance(r, dict) and r.get("silhouette") is not None]
    if valid:
        fig, ax = plt.subplots(figsize=(6.4, 4.2))
        x = [int(r["k"]) for r in valid]
        y = [float(r["silhouette"]) for r in valid]
        ax.plot(x, y, "o-", color=COLORS[0])
        selected = clustering.get("candidate_best_k", clustering.get("selected_k", len(unique)))
        ax.axvline(selected, color=COLORS[1], ls="--", label="Best exploratory K={}".format(selected))
        ax.set(xlabel="Number of clusters K", ylabel="Silhouette score",
               title="Exploratory cluster separation in retained PCA space")
        ax.set_xticks(x)
        ax.grid(alpha=.2)
        ax.legend()
        fig.tight_layout()
        _save(fig, directory, "cluster-selection")
        plt.close(fig)


def _interactive(records, scores, labels, baselines, baseline_scores, explained):
    import plotly.graph_objects as go
    fig = go.Figure()
    cases = sorted({str(r["case_id"]) for r in records})
    case_colors = {case: COLORS[i % len(COLORS)] for i, case in enumerate(cases)}
    labels_unique = sorted(set(labels))
    hover = np.asarray([[str(r["sample_id"]), str(r["case_id"]), int(label) + 1,
                         _metric(r, "ct_relative_l1"), _metric(r, "ct_positive_l1"),
                         _metric(r, "ct_negative_l1")] for r, label in zip(records, labels)], dtype=object)
    template = ("%{customdata[0]}<br>Phantom: %{customdata[1]}<br>Cluster: %{customdata[2]}"
                "<br>CT relative L1: %{customdata[3]:.5g}<br>CT increase: %{customdata[4]:.5g}"
                "<br>CT decrease: %{customdata[5]:.5g}<br>PC1: %{x:.5g}<br>PC2: %{y:.5g}<extra></extra>")
    cluster_trace_count = len(labels_unique)
    for i, label in enumerate(labels_unique):
        mask = labels == label
        fig.add_trace(go.Scattergl(x=scores[mask, 0], y=scores[mask, 1], mode="markers",
                      name="Cluster {} (n={})".format(label + 1, int(sum(mask))), customdata=hover[mask],
                      hovertemplate=template, marker=dict(size=6, opacity=.65, color=COLORS[i % len(COLORS)])))
    for case in cases:
        mask = np.array([str(r["case_id"]) == case for r in records])
        fig.add_trace(go.Scattergl(x=scores[mask, 0], y=scores[mask, 1], mode="markers", visible=False,
                      name="Phantom " + case, customdata=hover[mask], hovertemplate=template,
                      marker=dict(size=6, opacity=.65, color=case_colors[case])))
    extents = np.array([_metric(r, "ct_relative_l1") for r in records])
    extents = np.where(np.isfinite(extents) & (extents >= 0), extents, 0)
    fig.add_trace(go.Scattergl(x=scores[:, 0], y=scores[:, 1], mode="markers", visible=False,
                  name="Relative CT-change extent", customdata=hover, hovertemplate=template,
                  marker=dict(size=6, opacity=.7, color=np.log10(1 + 100 * extents), colorscale="Viridis",
                              showscale=True, colorbar=dict(title="log10(1+100L1)"))))
    has_baseline = bool(len(baselines))
    if has_baseline:
        fig.add_trace(go.Scatter(x=baseline_scores[:, 0], y=baseline_scores[:, 1], mode="markers",
                      name="Unchanged baseline", customdata=[[r.get("case_id")] for r in baselines],
                      hovertemplate="Baseline phantom %{customdata[0]}<extra></extra>",
                      marker=dict(symbol="star", size=13, color="#555555", line=dict(color="white", width=1))))
    n = cluster_trace_count + len(cases) + 1
    masks = []
    masks.append([i < cluster_trace_count for i in range(n)] + ([True] if has_baseline else []))
    masks.append([cluster_trace_count <= i < cluster_trace_count + len(cases) for i in range(n)] + ([True] if has_baseline else []))
    masks.append([i == n - 1 for i in range(n)] + ([True] if has_baseline else []))
    fig.update_layout(template="plotly_white", height=690,
                      xaxis_title="PC1 ({:.1f}% variance)".format(100 * explained[0]),
                      yaxis_title="PC2 ({:.1f}% variance)".format(100 * explained[1]),
                      title="Each dot is one sibling; zoom, hover, or hide a group using its legend",
                      updatemenus=[dict(type="buttons", direction="right", x=0, y=1.12,
                        buttons=[dict(label=name, method="update", args=[{"visible": mask}])
                                 for name, mask in zip(("Clusters", "Phantoms", "CT-change extent"), masks)])])
    return fig.to_html(full_html=False, include_plotlyjs=True,
                       config=dict(displaylogo=False, toImageButtonOptions=dict(format="svg")))


def write_report(output_directory, analysis, records, scores, labels,
                 baseline_records=None, baseline_scores=None):
    """Write a new offline review directory; refuse to overwrite existing data."""
    records, scores, labels, baselines, baseline_scores, explained = _validated(
        analysis, records, scores, labels, baseline_records, baseline_scores)
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError("A PCA review directory must be new: " + str(destination))
    destination.parent.mkdir(parents=True, exist_ok=True)
    directory = Path(tempfile.mkdtemp(prefix="." + destination.name + "-", dir=destination.parent))
    try:
        summaries = cluster_summary(records, labels)
        payload = _jsonable(dict(analysis, report_schema_version=SCHEMA, n_siblings=len(records),
                                n_baseline_markers=len(baselines), cluster_summary=summaries))
        (directory / "analysis.json").write_text(json.dumps(payload, indent=2, sort_keys=True, allow_nan=False) + "\n")
        augmented = [dict(row, pc1=float(point[0]), pc2=float(point[1]), cluster=int(label) + 1,
                          cluster_index=int(label),
                          **{name: _metric(row, name) for name in METRIC_ALIASES})
                     for row, point, label in zip(records, scores, labels)]
        fields = ["sample_id", "case_id", "cluster", "pc1", "pc2", *METRIC_ALIASES]
        fields += sorted(set().union(*(set(r) for r in augmented)) - set(fields))
        _csv(directory / "siblings.csv", augmented, fields)
        _csv(directory / "cluster-summary.csv", summaries,
             ["cluster", "cluster_index", "n", "case_counts", *METRIC_ALIASES])
        if baselines:
            _csv(directory / "baselines.csv", [dict(r, pc1=float(s[0]), pc2=float(s[1]))
                                               for r, s in zip(baselines, baseline_scores)],
                 ["case_id", "pc1", "pc2"])
        _figures(directory, payload, records, scores, labels, baselines, baseline_scores, explained)
        interactive = _interactive(records, scores, labels, baselines, baseline_scores, explained)
        table = "".join("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            s["cluster"], s["n"], ", ".join("{}: {}".format(k, v) for k, v in sorted(s["case_counts"].items())),
            "—" if s["ct_positive_l1"] is None else "{:.5g}".format(s["ct_positive_l1"]["median"]),
            "—" if s["ct_negative_l1"] is None else "{:.5g}".format(s["ct_negative_l1"]["median"])) for s in summaries)
        description = analysis.get("representation_description", analysis.get("representation", "Baseline-relative CT difference representation."))
        if not isinstance(description, str):
            description = json.dumps(_jsonable(description), sort_keys=True)
        clustering = analysis.get("clustering", {})
        suggestion = clustering.get("suggested_k", len(summaries))
        candidate = clustering.get("candidate_best_k", len(summaries))
        support = ("The partition passed the prespecified separation, restart stability and size checks."
                   if clustering.get("evidence_passed") is True else
                   "The partition did not establish discrete classes; a continuous distribution remains possible.")
        cluster_note = ("Best exploratory partition: K={}; diagnostic suggested K={}. {}".format(
            candidate, suggestion, support))
        if clustering.get("fallback_reasons"):
            cluster_note += " Flags: " + ", ".join(clustering["fallback_reasons"]) + "."
        pca = analysis.get("pca", {})
        if pca.get("target_reached") is False:
            cluster_note += " The retained components did not reach the requested variance target."
        html_text = """<!doctype html><html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Baseline-relative CT differences</title>
<style>body{font:16px/1.5 system-ui,sans-serif;max-width:1400px;margin:30px auto;padding:0 24px;color:#183144}
h1,h2{line-height:1.2}a{color:#0072b2}table{border-collapse:collapse;width:100%%}th,td{text-align:left;padding:9px;border-bottom:1px solid #dde4e8}img{max-width:100%%}.note{padding:14px;background:#eef4f6;border-left:4px solid #0072b2}.small{font-size:14px;color:#455b69}</style></head><body>
<h1>Baseline-relative CT differences</h1><p>%s siblings from %s baseline phantoms. Every sibling is retained, including zero CT differences.</p>
<p>%s</p><p class="note">Clusters are exploratory groups of CT-proxy differences, not validated clinical severity classes.
CT increases and decreases do not necessarily mean anatomical addition and removal. Mask counts are posthoc annotations only.
PC1 and PC2 show %.1f%% of total representation variance; clustering may use additional retained components.</p>
<p>%s</p>
%s<h2>Cluster summary</h2><table><thead><tr><th>Cluster</th><th>Siblings</th><th>Phantom composition</th><th>Median CT increase</th><th>Median CT decrease</th></tr></thead><tbody>%s</tbody></table>
<h2>Anatomy and extent checks</h2><p>Compare phantom colors with cluster colors to assess residual anatomy effects. Multiple synthetic siblings share the same baseline and are not independent patients.</p>
<img src="pca-phantom-extent.png" alt="PCA colored by phantom and relative CT-change extent">
<h2>Exports and provenance</h2><p><a href="pca-clusters.png">PCA PNG</a> · <a href="pca-clusters.pdf">PDF</a> · <a href="pca-clusters.svg">SVG</a> · <a href="siblings.csv">All sibling coordinates and annotations CSV</a> · <a href="cluster-summary.csv">Cluster summary CSV</a> · <a href="analysis.json">Analysis JSON</a> · <a href="report-provenance.json">Report provenance</a></p>
<p class="small">No model training or classifier fitting is performed by this report. Unchanged baselines are shown only as transformed references and are excluded from cluster fitting.</p></body></html>""" % (
            len(records), len({r["case_id"] for r in records}), html.escape(description),
            100 * sum(explained[:2]), html.escape(cluster_note), interactive, table)
        (directory / "report.html").write_text(html_text, encoding="utf-8")
        outputs = {p.name: hashlib.sha256(p.read_bytes()).hexdigest() for p in sorted(directory.iterdir())}
        provenance = dict(schema_version=SCHEMA, reporter_sha256=hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
                          outputs_sha256=outputs, input_provenance=analysis.get("provenance", {}),
                          baseline_markers_excluded_from_fitting=True, anatomical_masks_used_as_features=False)
        (directory / "report-provenance.json").write_text(json.dumps(_jsonable(provenance), indent=2, sort_keys=True) + "\n")
        directory.rename(destination)
        return dict(directory=str(destination), report=str(destination / "report.html"),
                    n_siblings=len(records), n_clusters=len(summaries), exports=list(outputs))
    except Exception:
        shutil.rmtree(directory, ignore_errors=True)
        raise
