#!/usr/bin/env python3
"""Present saved pretrained and sibling-trained predictions as two models.

This creates a new report from immutable measurements and regenerated CT
presentation assets. It performs no model fitting, inference or metric replay.
"""
import argparse
import copy
import csv
import hashlib
from html import escape
import json
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "src"))
import fakect_pretrained_comparison_report as comparison

ARMS = ("pretrained", "siblings")
LABELS = {"pretrained": "Pretrained", "siblings": "Fine-tuned"}
COLORS = {"pretrained": "#00a8c6", "siblings": "#d000b8"}
PANELS = (("whole_thoracic_core", "dice", "Whole-aorta Dice"),
          ("lesion_local", "dice", "Change-local Dice"),
          ("lesion_local", "recall", "Change-local recall"),
          ("lesion_local", "surface_dice", "Change-local surface Dice · 1 mm"))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def verify(directory):
    path = directory / "provenance.json"
    data = json.loads(path.read_text())
    hashes = data.get("output_sha256", data.get("artifacts_sha256"))
    if not hashes:
        raise ValueError("Source report requires published file checksums")
    for relative, expected in hashes.items():
        if sha(directory / relative) != expected:
            raise ValueError("Source artifact changed: " + str(directory / relative))
    return data


def save(figure, output, stem):
    for axis in figure.axes:
        for line in axis.lines:
            line.set_linewidth(2.5)
        for spine in axis.spines.values():
            spine.set_linewidth(1.1)
    for extension in ("png", "pdf", "svg"):
        figure.savefig(output / (stem + "." + extension), dpi=220, bbox_inches="tight")


def mean(rows, arm, region, metric):
    return comparison._balanced_mean(rows, [comparison._metric(r, arm, region, metric) for r in rows])


def overview(rows, extents, output, histogram_scope="Virtual population"):
    plt = comparison._plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    for axis, quantity, word, color in zip(axes, ("removed_voxels", "added_voxels"),
            ("Removed", "Added"), (COLORS["siblings"], COLORS["pretrained"])):
        values = [r[quantity] for r in extents if not r["baseline"] and r[quantity] > 0]
        counts = [sum(v > lower and (upper is None or v <= upper) for v in values)
            for lower, upper in zip((0, *comparison.UPPER_COUNTS), (*comparison.UPPER_COUNTS, None))]
        bins = ("1–100", "101–500", "501–1k", "1k–2k", "2k–5k", "5k–10k", "10k–20k", "20k–50k", ">50k")
        axis.bar(np.arange(len(bins)), counts, color=color, edgecolor="#333", linewidth=1.1)
        axis.set_xticks(np.arange(len(bins)), bins, rotation=40, ha="right")
        axis.set(xlabel=word + " native aorta-core voxels", ylabel="Number of siblings",
                 title=word + " burden · {} siblings".format(len(values)))
        axis.grid(axis="y", alpha=.15)
    fig.suptitle(histogram_scope + " · exact edit counts · mixed changes appear in both histograms")
    fig.tight_layout(); save(fig, output, "cohort-native-counts"); plt.close(fig)

    profiles = {}
    for quantity, word in (("removed_voxels", "Removed"), ("added_voxels", "Added")):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
        profiles[quantity] = {}
        for axis, (region, metric, title) in zip(axes.ravel(), PANELS):
            bins = comparison.burden_bins(rows, quantity, region=region, metric=metric)
            profiles[quantity][region + "/" + metric] = bins
            for arm in ARMS:
                for mixed in (False, True):
                    points = [(r[quantity], comparison._metric(r, arm, region, metric)) for r in rows
                        if not r["baseline"] and r[quantity] > 0 and (r["change_kind"] == "mixed") == mixed]
                    points = [(x, y) for x, y in points if y is not None]
                    if points:
                        xy = np.asarray(points)
                        axis.scatter(xy[:, 0], xy[:, 1], s=18, alpha=.35,
                            marker="x" if mixed else "o", color=COLORS[arm], linewidths=.6)
                usable = [b for b in bins if b["means"][arm] is not None]
                axis.plot([b["x_native_voxels"] for b in usable], [b["means"][arm] for b in usable],
                    "o-", color=COLORS[arm], label=LABELS[arm])
            axis.set(xlabel=word + " native aorta-core voxel count", ylabel=title, title=title, ylim=(-.03, 1.03))
            axis.set_xscale("symlog", linthresh=100); axis.set_xlim(left=0); axis.grid(alpha=.15)
            if metric == "dice" and region == "lesion_local":
                for index, b in enumerate(bins):
                    axis.annotate("n={}\n{} cases".format(b["siblings"], b["cases"]),
                        (b["x_native_voxels"], .045 + (index % 2) * .10), fontsize=7,
                        ha="right" if index == len(bins) - 1 else "center")
        axes[0, 0].legend(fontsize=9)
        fig.suptitle(word + " burden · dots: siblings, crosses: mixed changes · lines: case/family-balanced means")
        fig.tight_layout(); save(fig, output, "performance-" + quantity.replace("_voxels", "")); plt.close(fig)

    fig, axis = plt.subplots(figsize=(5.8, 5.4))
    for original, marker in ((True, "*"), (False, "o")):
        points = [(comparison._metric(r, "pretrained", "whole_thoracic_core", "dice"),
                   comparison._metric(r, "siblings", "whole_thoracic_core", "dice"))
                  for r in rows if r["baseline"] == original]
        xy = np.asarray([(x, y) for x, y in points if x is not None and y is not None])
        if len(xy):
            axis.scatter(xy[:, 0], xy[:, 1], s=65 if original else 22, marker=marker,
                color=COLORS["siblings"], alpha=.65, label="Original phantoms" if original else "Digital siblings")
    axis.plot([0, 1], [0, 1], "--", color="#777")
    axis.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel="Pretrained · Dice", ylabel="Fine-tuned · Dice")
    axis.set_aspect("equal"); axis.grid(alpha=.15); axis.legend(fontsize=9)
    fig.suptitle("Same-scan comparison · whole-aorta Dice")
    fig.tight_layout(); save(fig, output, "paired-performance"); plt.close(fig)

    cases = sorted({r["case_id"] for r in rows})
    for quantity, word in (("removed_voxels", "Removed"), ("added_voxels", "Added")):
        columns = min(3, len(cases)); nrows = int(np.ceil(len(cases) / columns))
        fig, axes = plt.subplots(nrows, columns, figsize=(4.1 * columns, 3.3 * nrows), squeeze=False)
        for axis, case in zip(axes.ravel(), cases):
            for arm in ARMS:
                points = [(r[quantity], comparison._metric(r, arm, "whole_thoracic_core", "dice"))
                    for r in rows if r["case_id"] == case and (r["baseline"] or r[quantity] > 0)]
                xy = np.asarray([(x, y) for x, y in points if y is not None])
                if len(xy): axis.scatter(xy[:, 0], xy[:, 1], s=22, alpha=.75, color=COLORS[arm], label=LABELS[arm])
            axis.set(title="Phantom " + case, xlabel=word + " native aorta-core voxels", ylabel="Whole-aorta Dice", ylim=(-.03, 1.03))
            axis.set_xscale("symlog", linthresh=100); axis.set_xlim(left=0); axis.grid(alpha=.15)
        for axis in axes.ravel()[len(cases):]: axis.set_visible(False)
        axes.ravel()[0].legend(fontsize=8)
        fig.tight_layout(); save(fig, output, "case-facets-" + quantity.replace("_voxels", "")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for axis, quantity, word in zip(axes, ("removed_mm3", "added_mm3"), ("Removed", "Added")):
        for arm in ARMS:
            points = [(r[quantity], comparison._metric(r, arm, "lesion_local", "dice"))
                for r in rows if not r["baseline"] and r[quantity] > 0]
            xy = np.asarray([(x, y) for x, y in points if y is not None])
            if len(xy): axis.scatter(xy[:, 0], xy[:, 1], s=22, alpha=.6, color=COLORS[arm], label=LABELS[arm])
        axis.set(xlabel=word + " native aorta-core volume · mm³", ylabel="Change-local Dice", ylim=(-.03, 1.03))
        axis.set_xscale("symlog", linthresh=10); axis.set_xlim(left=0); axis.grid(alpha=.15)
    axes[0].legend(fontsize=9)
    fig.tight_layout(); save(fig, output, "physical-volume-adjunct"); plt.close(fig)

    families = sorted({r["anatomy_family"] for r in rows}, key=lambda f:
        (999 if f.endswith("adult") else int(f.rsplit("_", 1)[-1].replace("yr", ""))))
    fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
    for axis, (region, metric, title) in zip(axes.ravel(), PANELS):
        for index, arm in enumerate(ARMS):
            values = [mean([r for r in rows if r["anatomy_family"] == family], arm, region, metric) for family in families]
            axis.bar(np.arange(len(families)) + (index - .5) * .34, values, width=.34,
                color=COLORS[arm], edgecolor="#333", linewidth=1.1, label=LABELS[arm])
        axis.set_xticks(np.arange(len(families)), [f.replace("xcat_reference_", "") for f in families])
        axis.set(title=title, ylabel=title, ylim=(0, 1.05)); axis.grid(axis="y", alpha=.15)
    axes[0, 0].legend(fontsize=9)
    fig.suptitle("Five held-out anatomical families · two phantoms per family")
    fig.tight_layout(); save(fig, output, "family-performance"); plt.close(fig)
    return profiles


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--ct-assets", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    source, assets, destination = [p.resolve() for p in (args.source_report, args.ct_assets, args.output_directory)]
    if destination.exists(): raise FileExistsError("Use a new presentation revision")
    verify(source)
    original = json.loads((source / "analysis.json").read_text())
    if original["status"] != "complete" or original["paired_samples"] != 130:
        raise ValueError("Requires the complete fixed 130-scan scientific comparison")
    analysis = copy.deepcopy(original)
    for row in analysis["samples"]:
        row["predictions"] = {arm: row["predictions"][arm] for arm in ARMS}
    analysis["arms"] = {arm: analysis["arms"][arm] for arm in ARMS}
    analysis["arm_labels"] = LABELS
    analysis["source_protocol"]["comparators"] = "Displayed models: pretrained weights and fine-tuning using original phantoms plus digital siblings; identical preprocessing and inference."
    comparison.ARMS = ARMS
    comparison.LABELS = LABELS
    comparison.COLORS = COLORS
    manifest_path = source.parent / "comparison-review-comparison-r1.json"
    extents = json.loads(manifest_path.read_text())["cohort_extents"]
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + destination.name + "-", dir=destination.parent))
    try:
        analysis["burden_profiles"] = overview(analysis["samples"], extents, staging,original.get('histogram_scope','Virtual population'))
        ct = assets / "prespecified"
        ct_provenance = verify(ct)
        examples = ct_provenance["examples"]
        if [r["sample_id"] for r in examples] != analysis["example_sample_ids"]:
            raise ValueError("CT examples must retain all frozen identities and their order")
        for example in examples:
            stem = Path(example["figure"]).stem
            for extension in ("png", "pdf", "svg"):
                shutil.copy2(ct / (stem + "." + extension), staging / (stem + "." + extension))
        shutil.copytree(ct / "report-3d", staging / "report-3d")
        with (source / "metrics.csv").open(newline="") as stream:
            reader = csv.DictReader(stream)
            fields = ["model", *reader.fieldnames]
            csv_rows = [dict(model=LABELS[row["arm"]], **row) for row in reader if row["arm"] in ARMS]
        if len(csv_rows) != 260: raise ValueError("Two models must supply exactly 260 arm/scan rows")
        with (staging / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader(); writer.writerows(csv_rows)
        analysis["examples"] = examples
        analysis["viewer"] = dict(viewer=str(destination / "report-3d/index.html"), offline=True, models=LABELS)
        counts = {}
        for arm in ARMS:
            values = [row["predictions"][arm]["metrics"]["lesion_local"] for row in analysis["samples"]
                if row["predictions"][arm]["metrics"]["lesion_local"] is not None]
            counts[arm] = dict(total=len(values), hd95_valid=sum(v["surfaces"]["hd95_mm"] is not None for v in values))
        analysis["presentation"] = dict(displayed_models=LABELS,
            fine_tuned_definition="Fine-tuning using original phantoms plus digital siblings",
            reference_color="#00ffff", prediction_color="#ff00ff", contour_width_pt=2.2,
            arm_colors=COLORS, surface_distance_counts=counts,
            source_analysis_sha256=sha(source / "analysis.json"), metrics_recomputed=False)
        (staging / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
        summary_rows = [("Whole-aorta Dice · 130 scans", "whole_thoracic_core", "dice"),
                        ("Change-local Dice · 120 edited scans", "lesion_local", "dice"),
                        ("Change-local recall", "lesion_local", "recall"),
                        ("Change-local surface Dice · 1 mm", "lesion_local", "surface_dice")]
        table = ''.join('<tr><td>' + label + '</td>' + ''.join('<td>{:.3f}</td>'.format(mean(analysis["samples"], arm, region, metric)) for arm in ARMS) + '</tr>' for label, region, metric in summary_rows)
        ct_html = ''.join('<section><h2>' + escape(e["sample_id"]) + '</h2><p><a href="' + e["figure"].replace('.png', '.pdf') + '">PDF</a> · <a href="' + e["figure"].replace('.png', '.svg') + '">SVG</a></p><img src="' + e["figure"] + '" alt="Reference, Pretrained and Fine-tuned CT panels"></section>' for e in examples)
        document = '''<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pretrained and Fine-tuned aorta comparison</title><style>body{font:16px/1.5 system-ui;max-width:1400px;margin:auto;padding:24px;color:#183144}img{max-width:100%}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}.note{padding:16px;background:#eef4f6}</style></head><body><h1>Pretrained versus Fine-tuned</h1><p class="note">Fine-tuned means the model adapted using original phantoms plus digital siblings. Both models have completed predictions on the same 130 held-out scans. This is a presentation of saved results; model predictions and measurements are unchanged.</p><p>''' + escape(comparison.SCOPE) + '''</p><table><thead><tr><th>Mean metric</th><th>Pretrained</th><th>Fine-tuned</th></tr></thead><tbody>''' + table + '''</tbody></table><h2>Virtual population</h2><p>The histograms cover all 2,560 generated siblings; mixed changes appear in both distributions. Thirty-nine siblings have no net aorta-core change. Performance curves use the fixed 130-scan evaluation panel, including 120 edited siblings and ten original phantoms.</p><img src="cohort-native-counts.png" alt="Added and removed voxel-count histograms"><h2>Performance across removal burden</h2><img src="performance-removed.png" alt="Pretrained and Fine-tuned performance versus removed voxels"><h2>Performance across addition burden</h2><img src="performance-added.png" alt="Pretrained and Fine-tuned performance versus added voxels"><p>Lines average within each source phantom and then anatomical family. Bin labels show n siblings and source-case counts. Digital siblings are correlated; these are descriptive profiles, with no independent-patient confidence intervals.</p><h2>Held-out anatomical families</h2><img src="family-performance.png" alt="Pretrained and Fine-tuned family-specific performance"><h2>Same-scan paired performance</h2><img src="paired-performance.png" alt="Pretrained and Fine-tuned Dice for identical scans"><h2>Per-phantom profiles</h2><img src="case-facets-removed.png" alt="Removal burden per phantom"><img src="case-facets-added.png" alt="Addition burden per phantom"><h2>Physical-volume adjunct</h2><p>Native voxel sizes differ between phantoms; the companion volume view provides physical context.</p><img src="physical-volume-adjunct.png" alt="Change-local Dice versus physical edited volume"><h2>Zoomed CT comparisons</h2><p>Three columns: Reference, Pretrained, Fine-tuned. Every row uses the same current CT, planes, HU window and zoom. Cyan is the current scan's atlas reference; magenta is the prediction. Contours are 2.2 points wide. These three example identities were frozen before inference.</p>''' + ct_html + '''<h2>Interactive 3D</h2><p>Switch between Pretrained and Fine-tuned, rotate and zoom, and control reference, prediction and known-change visibility and opacity independently.</p><a href="report-3d/index.html">Open standalone 3D viewer</a><iframe src="report-3d/index.html" style="width:100%;height:950px;border:0" title="Pretrained and Fine-tuned native surfaces"></iframe><p class="note">Local HD95 is undefined for empty prediction surfaces: valid counts are 112/120 for Pretrained and 117/120 for Fine-tuned. Dice, recall and surface Dice retain these failures. A distance average over defined surfaces excludes failures.</p><h2>Exports</h2><p><a href="metrics.csv">260 model/scan metrics rows</a> · <a href="analysis.json">Full displayed analysis</a> · <a href="provenance.json">Provenance</a>. All figures have PNG/PDF/SVG versions.</p></body></html>'''
        modified=sum(not r['baseline'] for r in extents)
        unchanged=sum(not r['baseline'] and r['added_voxels']==0 and r['removed_voxels']==0 for r in extents)
        document=document.replace('The histograms cover all 2,560 generated siblings; mixed changes appear in both distributions. Thirty-nine siblings have no net aorta-core change.',
            'Histogram scope: '+escape(original.get('histogram_scope','Frozen population'))+'. It contains '+str(modified)+' edited siblings, with '+str(unchanged)+' net-unchanged targets; mixed changes appear in both distributions.')
        document=document.replace(escape(comparison.SCOPE),escape(original['scope']))
        document=document.replace('valid counts are 112/120 for Pretrained and 117/120 for Fine-tuned.',
            'valid counts are {}/{} for Pretrained and {}/{} for Fine-tuned.'.format(counts['pretrained']['hd95_valid'],counts['pretrained']['total'],counts['siblings']['hd95_valid'],counts['siblings']['total']))
        if original.get('target_policy'):
            component_rows=[]
            for arm in ARMS:
                cells=[]
                for name in ('aorta_core','connected_branches'):
                    values=[row['predictions'][arm].get('component_recalls',{}).get(name,{}).get('recall') for row in analysis['samples']]
                    values=[v for v in values if v is not None]
                    cells.append('{:.3f}'.format(float(np.mean(values))) if values else 'Unavailable')
                component_rows.append('<tr><td>'+LABELS[arm]+'</td>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
            component_table='<h2>Core and branch coverage</h2><p>Reference-positive voxel recalls; false positives are assessed by the combined-target metrics over the common physical domain.</p><table><tr><th>Model</th><th>Core recall</th><th>Connected-branch recall</th></tr>'+''.join(component_rows)+'</table>'
            document=document.replace('<h2>Exports</h2>',component_table+'<h2>Exports</h2>')
        (staging / "report.html").write_text(document, encoding="utf-8")
        shutil.copy2(Path(__file__), staging / "presentation-generator.py")
        provenance = dict(schema_version="fakect.two-model-presentation/1", complete=True,
            source_report=str(source), source_provenance_sha256=sha(source / "provenance.json"),
            ct_assets=str(ct), ct_provenance_sha256=sha(ct / "provenance.json"),
            generator_sha256=sha(Path(__file__)), models_run=False, metrics_recomputed=False,
            displayed_models=LABELS,
            output_sha256={str(p.relative_to(staging)): sha(p) for p in staging.rglob('*') if p.is_file()})
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    print(json.dumps(dict(report=str(destination / "report.html"), complete=True, scans=130, models=LABELS)))


if __name__ == "__main__":
    main()
