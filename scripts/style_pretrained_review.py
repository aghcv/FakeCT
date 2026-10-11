#!/usr/bin/env python3
"""Publish a high-contrast presentation of a completed scientific review.

Uses saved measurements and prespecified figures. It never trains, infers or
recalculates segmentation metrics, and leaves the frozen source review intact.
"""
import argparse
import hashlib
import json
from pathlib import Path
import re
import shutil
import sys
import tempfile

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import fakect_pretrained_comparison_report as comparison


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_report(directory, provenance_name, hash_key):
    provenance_path = directory / provenance_name
    provenance = json.loads(provenance_path.read_text())
    for relative, expected in provenance[hash_key].items():
        if digest(directory / relative) != expected:
            raise ValueError("Published source changed: " + str(directory / relative))
    return provenance, digest(provenance_path)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-report", required=True, type=Path)
    parser.add_argument("--examples-report", required=True, type=Path)
    parser.add_argument("--output-directory", required=True, type=Path)
    args = parser.parse_args()
    source, examples, destination = [p.resolve() for p in
        (args.source_report, args.examples_report, args.output_directory)]
    if destination.exists():
        raise FileExistsError("Use a new presentation revision")
    source_provenance, source_hash = checked_report(source, "provenance.json", "output_sha256")
    example_provenance, example_hash = checked_report(examples, "provenance.json", "output_sha256")
    analysis = json.loads((source / "analysis.json").read_text())
    if analysis["status"] != "complete" or analysis["paired_samples"] != analysis["sample_count"]:
        raise ValueError("The full scientific comparison must finish before presentation")
    if [e["sample_id"] for e in example_provenance["examples"]] != analysis["example_sample_ids"]:
        raise ValueError("Presentation examples must exactly match the frozen identities")
    if example_provenance["presentation_style"] != dict(
            reference_color="#00ffff", prediction_color="#ff00ff", contour_width_pt=2.2):
        raise ValueError("Prespecified figures must use the requested cyan/magenta style")
    manifest_path = source.parent / "comparison-review-comparison-r1.json"
    manifest = json.loads(manifest_path.read_text())
    comparison.COLORS.update(pretrained="#555555", baseline_only="#00a8c6", siblings="#d000b8")
    original_save = comparison._save

    def styled_save(figure, directory, stem):
        for axis_index, axis in enumerate(figure.axes):
            if stem != "cohort-native-counts" and "native CORE" in axis.get_xlabel():
                axis.set_xlim(left=0)
            annotations = [text for text in axis.texts if re.fullmatch(
                r"\d+ cases / \d+ siblings", text.get_text())]
            for index, text in enumerate(annotations):
                cases, siblings = re.findall(r"\d+", text.get_text())
                text.set_text("n={}\n{} cases".format(siblings, cases))
                text.set_rotation(0)
                text.set_fontsize(7)
                text.set_ha("right" if index == len(annotations) - 1 else "center")
                text.set_position((text.xy[0], .045 + (index % 2) * .10))
            for line in axis.lines:
                line.set_linewidth(2.5)
            for spine in axis.spines.values():
                spine.set_linewidth(1.1)
            for patch in axis.patches:
                patch.set_edgecolor("#333333")
                patch.set_linewidth(1.1)
                if stem == "cohort-native-counts":
                    patch.set_facecolor(("#d000b8", "#00a8c6")[axis_index])
        original_save(figure, directory, stem)

    comparison._save = styled_save
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + destination.name + "-", dir=destination.parent))
    try:
        profiles = comparison._overview_plots(analysis["samples"], staging,
            analysis["arm_labels"], manifest["cohort_extents"])
        for example in example_provenance["examples"]:
            stem = Path(example["figure"]).stem
            for suffix in ("png", "pdf", "svg"):
                shutil.copy2(examples / (stem + "." + suffix), staging / (stem + "." + suffix))
        shutil.copytree(examples / "report-3d", staging / "report-3d")
        shutil.copy2(source / "metrics.csv", staging / "metrics.csv")
        document = (source / "report.html").read_text()
        document = document.replace("Blue is that scan’s atlas reference; orange is the relevant prediction.",
            "Cyan is that scan’s atlas reference; magenta is the relevant prediction. Contours use 2.2-point lines.")
        document = document.replace("Bin case/sibling counts are annotated and recorded in analysis.json.",
            "Bin labels show n (siblings) and the number of source phantoms; exact counts are recorded in analysis.json.")
        distance_counts = {}
        for arm in comparison.ARMS:
            distance_counts[arm] = {}
            for region in ("whole_thoracic_core", "lesion_local"):
                values = [row["predictions"][arm]["metrics"][region] for row in analysis["samples"]
                    if row["predictions"][arm]["metrics"][region] is not None]
                distance_counts[arm][region] = dict(total=len(values),
                    hd95_valid=sum(value["surfaces"]["hd95_mm"] is not None for value in values))
        counts_text = "; ".join(analysis["arm_labels"][arm] + ": "
            + str(distance_counts[arm]["lesion_local"]["hd95_valid"]) + "/"
            + str(distance_counts[arm]["lesion_local"]["total"]) for arm in comparison.ARMS)
        distance_note = '<p class="note">Local HD95 is undefined when a reference or prediction surface is empty. Valid local distance counts: ' + counts_text + '. A mean over defined distances excludes such failures and must not be interpreted as an unconditional improvement. Dice, recall and surface Dice retain the empty-prediction failures.</p>'
        document = document.replace('<h2>Exports</h2>', distance_note + '<h2>Exports</h2>')
        (staging / "report.html").write_text(document, encoding="utf-8")
        analysis["burden_profiles"] = profiles
        analysis["examples"] = example_provenance["examples"]
        analysis["viewer"] = dict(example_provenance["viewer"], viewer=str(destination / "report-3d/index.html"))
        analysis["presentation"] = dict(style=example_provenance["presentation_style"],
            arm_colors=dict(comparison.COLORS), performance_linewidth_pt=2.5,
            nonnegative_burden_axes=True,
            surface_distance_counts=distance_counts,
            source_analysis_sha256=digest(source / "analysis.json"), metrics_recomputed=False)
        (staging / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
        shutil.copy2(Path(__file__), staging / "presentation-generator.py")
        provenance = dict(schema_version="fakect.high-contrast-review/1",
            source_report=str(source), source_provenance_sha256=source_hash,
            examples_report=str(examples), examples_provenance_sha256=example_hash,
            manifest_sha256=digest(manifest_path), generator_sha256=digest(Path(__file__)),
            complete=True, models_run=False, metrics_recomputed=False,
            output_sha256={str(p.relative_to(staging)): digest(p) for p in staging.rglob("*") if p.is_file()})
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    print(json.dumps(dict(report=str(destination / "report.html"), paired_samples=analysis["paired_samples"], complete=True)))


if __name__ == "__main__":
    main()
