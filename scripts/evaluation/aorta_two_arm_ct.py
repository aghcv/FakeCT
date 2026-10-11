#!/usr/bin/env python3
"""Derive simple CT panels and offline meshes from a completed frozen review.

This presentation changes no predictions, measurements, example identities,
native slice coordinates, HU window or reference-defined display zoom.
"""
import argparse
import copy
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile

import numpy as np

ROOT = next(p for p in Path(__file__).resolve().parents
            if (p / "src/fakect_pretrained_comparison_report.py").is_file())
sys.path.insert(0, str(ROOT / "src"))
os.environ.setdefault("MPLCONFIGDIR", "/tmp/fakect-two-arm-ct-matplotlib")
import fakect_pretrained_comparison_report as comparison
from fakect_two_stage_3d import render_viewer

ARMS = ("pretrained", "siblings")
LABELS = {"pretrained": "Pretrained", "siblings": "Fine-tuned"}
STYLE = dict(reference_color="#00ffff", prediction_color="#ff00ff", contour_width_pt=2.2)


def digest(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def checked_document(path, inputs):
    inputs[str(path)] = digest(path)
    return json.loads(path.read_text())


def check_receipt(directory, inputs):
    path = directory / "provenance.json"
    receipt = checked_document(path, inputs)
    for name, expected in receipt.get("output_sha256", receipt.get("artifacts_sha256", {})).items():
        if digest(directory / name) != expected:
            raise ValueError("Published artifact changed: " + str(directory / name))


def example_figure(image, masks, predictions, row, out, index, expected, scope):
    """The original reporter's native geometry, with only three columns."""
    plt = comparison._plotting()
    truth, domain, changed = (masks[k] for k in ("truth_nifti", "domain_nifti", "changed_nifti"))
    truth, changed = truth & domain, changed & domain
    hu_window = tuple(expected["hu_window"])
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    origin = np.asarray(row.get("export", {}).get("origin_ijk", row.get("export_origin_ijk", [0, 0, 0])), int)
    focus = changed if changed.any() else truth
    box = comparison._bbox(focus, np.ceil(12. / spacing[::-1]).astype(int))
    if box is None:
        box = tuple(slice(0, n) for n in truth.shape)
    positions = [int(np.argmax(focus.sum(axis=tuple(i for i in range(3) if i != a)))) for a in range(3)]
    fig, axes = plt.subplots(3, 3, figsize=(11.8, 11.6))
    metadata = dict(sample_id=row["sample_id"], baseline=bool(row.get("baseline")), hu_window=list(hu_window),
        display_zoom_source="Native reference and known changes only; fixed before inspecting predictions",
        low_export_kji=[s.start for s in box], high_export_kji_exclusive=[s.stop for s in box],
        source_origin_ijk=origin.tolist(), orientations=[])
    for axis, name in enumerate(("Axial", "Coronal", "Sagittal")):
        fixed = positions[axis]
        slices = list(box); slices[axis] = fixed
        plane = np.asarray(image.dataobj[tuple(slices[::-1])], dtype=np.float32).T
        vertical, horizontal = [a for a in range(3) if a != axis]
        horizontal_origin, vertical_origin = origin[2 - horizontal], origin[2 - vertical]
        extent = [box[horizontal].start + horizontal_origin - .5, box[horizontal].stop + horizontal_origin - .5,
                  box[vertical].start + vertical_origin - .5, box[vertical].stop + vertical_origin - .5]
        native_fixed = fixed + origin[2 - axis]
        centre_export_ijk = np.array([(s.start + s.stop - 1) / 2 for s in box][::-1], dtype=float)
        centre_export_ijk[2 - axis] = fixed
        centre_ras = (np.asarray(image.affine) @ np.r_[centre_export_ijk, 1.])[:3]
        metadata["orientations"].append(dict(name=name, fixed_export_index=fixed,
            fixed_source_index=int(native_fixed), fixed_native_axis="kji"[axis],
            native_spacing_ijk_mm=spacing.tolist(), slice_centre_ras_mm=centre_ras.tolist(),
            ct_plane_sha256=hashlib.sha256(np.ascontiguousarray(plane).tobytes()).hexdigest(),
            identical_current_ct_across_columns=True))
        for col, arm in enumerate((None, *ARMS)):
            ax = axes[axis, col]
            ax.imshow(plane, origin="lower", cmap="gray", vmin=hu_window[0], vmax=hu_window[1], extent=extent,
                interpolation="nearest", aspect=spacing[::-1][vertical] / spacing[::-1][horizontal])
            for mask, color in ((truth, STYLE["reference_color"]),
                    (None if arm is None else predictions[arm] & domain, STYLE["prediction_color"])):
                if mask is None:
                    continue
                local = mask[tuple(slices)]
                if local.any() and not local.all():
                    ax.contour(np.arange(box[horizontal].start, box[horizontal].stop) + horizontal_origin,
                        np.arange(box[vertical].start, box[vertical].stop) + vertical_origin,
                        local.astype(float), levels=[.5], colors=[color], linewidths=STYLE["contour_width_pt"])
            if axis == 0:
                ax.set_title("Reference" if arm is None else LABELS[arm], fontsize=12)
            ax.set_xlabel("ijk"[2 - horizontal] + " · source voxel index")
            ax.set_ylabel("ijk"[2 - vertical] + " · source voxel index")
            if col == 0:
                coordinate = "{}={:.1f} mm RAS".format("xyz"[2 - axis], centre_ras[2 - axis])
                ax.text(.02, .02, "{} · {}={}\n{} · {:.3g} mm/voxel".format(
                    name, "kji"[axis], native_fixed, coordinate, spacing[2 - axis]),
                    transform=ax.transAxes, color="white", fontsize=8, backgroundcolor="#444")
    for key, value in metadata.items():
        if value != expected[key]:
            raise ValueError("Native CT panel geometry changed: " + row["sample_id"] + " / " + key)
    tag = "Unmodified original CT" if row.get("baseline") else "Current modified sibling CT"
    fig.suptitle(scope + "\n" + row["sample_id"] + " · " + tag +
        "\nSame CT, planes, window and zoom · cyan reference / magenta prediction", fontsize=11)
    fig.tight_layout(rect=(0, 0, 1, .94))
    stem = "example-{:02d}".format(index)
    comparison._save(fig, out, stem)
    plt.close(fig)
    return dict(metadata, figure=stem + ".png", columns=["Reference", "Pretrained", "Fine-tuned"],
        added_voxels=row["added_voxels"], removed_voxels=row["removed_voxels"],
        source_geometry_verified=True)


def simplify_scenes(source, inputs):
    original_manifest = checked_document(source / "manifest.json", inputs)
    scenes = []
    for record in original_manifest["scenes"]:
        path = source / record["file"]
        if digest(path) != record["sha256"]:
            raise ValueError("Published scene checksum mismatch")
        inputs[str(path)] = digest(path)
        content = path.read_text()
        scene = json.loads(content[len("window.fakeCTReceiveScene("):-3])
        derived = copy.deepcopy(scene)
        derived["surfaces"] = {key: mesh for key, mesh in derived["surfaces"].items()
                               if key in ("truth", "known_changes", *ARMS)}
        derived["modes"] = [mode for mode in derived["modes"] if mode["key"] in ARMS]
        if {mode["key"] for mode in derived["modes"]} != set(ARMS):
            raise ValueError("Each derived scene needs the two completed models")
        for mode in derived["modes"]:
            mode["label"] = LABELS[mode["key"]]
            mode["explanation"] = "Same current CT and native grid; the reference is evaluation only. Fine-tuned was trained using originals plus digital siblings."
            for layer in mode["layers"]:
                if layer["key"] == "truth":
                    layer.update(label="Reference", color=STYLE["reference_color"])
                elif layer["key"] in ARMS:
                    layer.update(label=LABELS[layer["key"]], color=STYLE["prediction_color"])
        derived["triangles"] = sum(mesh["triangles"] for mesh in derived["surfaces"].values())
        derived["warnings"] = [warning for warning in derived["warnings"] if not warning.startswith("baseline_only:")]
        for key in ("shape_kji", "spacing_ijk_mm", "bounds_ijk_mm", "metadata"):
            if derived[key] != scene[key]:
                raise ValueError("Derived scene changed native geometry")
        if any(mesh != scene["surfaces"][key] for key, mesh in derived["surfaces"].items()):
            raise ValueError("Derived scene changed exact mesh coordinates")
        scenes.append(derived)
    return scenes, original_manifest


def html_page(examples, scope):
    sections = "".join('<section><h2>' + escape(example["sample_id"]) + '</h2><p>' +
        str(example["added_voxels"]) + ' added / ' + str(example["removed_voxels"]) +
        ' removed native core voxels · <a href="' + example["figure"].replace(".png", ".pdf") +
        '">PDF</a> · <a href="' + example["figure"].replace(".png", ".svg") + '">SVG</a></p>' +
        '<img src="' + example["figure"] + '" alt="Same current CT with reference, pretrained and fine-tuned contours"></section>'
        for example in examples)
    return ('<!doctype html><html lang="en"><meta charset="utf-8"><title>' + escape(scope) + '</title>' +
        '<style>body{font:16px system-ui,sans-serif;max-width:1250px;margin:25px auto;padding:0 20px;color:#17202a}img{width:100%;height:auto}</style>' +
        '<h1>' + escape(scope) + '</h1><p>Fine-tuned denotes training using originals plus digital siblings. Reference denotes the atlas label for this current CT, not its paired original image. Cyan marks reference; magenta marks prediction.</p>' +
        '<p>CT planes, coordinates, HU window and zoom match the original report exactly. Synthetic voxel burden is not clinical severity.</p>' +
        sections + '<p><a href="analysis.json">Coordinate metadata and saved metrics</a> · <a href="../provenance.json">Provenance</a></p></html>')


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-directory", type=Path,
        default=ROOT / "outputs/experiments/aorta/pretrained-finetune-r1/two-arm-ct-r1")
    parser.add_argument("--source-review", type=Path,
        help="Completed scientific review; use its frozen examples without legacy supplemental figures")
    args = parser.parse_args()
    destination = args.output_directory.resolve()
    if destination.exists():
        raise FileExistsError("Use a new presentation revision")
    base = args.source_review.resolve().parent if args.source_review else ROOT / "outputs/experiments/aorta/pretrained-finetune-r1"
    inputs = {}
    review_source = args.source_review.resolve() if args.source_review else base / "review-presentation-r4"
    names=(review_source.name,) if args.source_review else ("review-comparison-r1", "review-presentation-r4", "young-child-diagnostic-r1", "stress-removal-example-r1")
    for name in names:
        check_receipt(base / name, inputs)
    manifest = checked_document(base / "comparison-review-comparison-r1.json", inputs)
    review = checked_document((args.source_review.resolve() if args.source_review else base / "review-comparison-r1") / "analysis.json", inputs)
    presentation = checked_document(review_source / "analysis.json", inputs)
    checked_document(base / "study-lock.json", inputs)
    if review["status"] != "complete" or review["paired_samples"] != 130:
        raise ValueError("Final paired review is incomplete")
    rows = {row["sample_id"]: row for row in manifest["samples"]}
    saved_rows = {row["sample_id"]: row for row in review["samples"]}
    prespecified = [next(example for example in presentation["examples"] if example["sample_id"] == sid)
                   for sid in review["example_sample_ids"]]
    groups = [("prespecified", "Prespecified CT examples", prespecified, "Frozen before inference", None)]
    extras=() if args.source_review else (("stress", "stress-removal-example-r1", "Post-hoc removal stress illustration"),
            ("young-child", "young-child-diagnostic-r1", "Post-hoc young-child diagnostic"))
    for group, dirname, scope in extras:
        saved = checked_document(base / dirname / "analysis.json", inputs)
        groups.append((group, scope, [saved["figure"]], saved["selection_scope"], saved))
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + destination.name + "-", dir=destination.parent))
    try:
        group_reports = {}
        for group, scope, original_examples, selection, original_analysis in groups:
            out = staging / group
            out.mkdir()
            examples, saved_metrics = [], {}
            for index, expected in enumerate(original_examples):
                row = rows[expected["sample_id"]]
                image, masks, spacing = comparison._common(row, inputs)
                predictions = {}
                for arm in ARMS:
                    result = row["predictions"][arm]
                    if result["status"] != "completed":
                        raise ValueError("A displayed model is incomplete")
                    inferred = comparison._read_image(result["prediction_nifti"], result["prediction_nifti_sha256"], inputs)
                    comparison._grid(image, inferred, arm)
                    predictions[arm] = comparison._mask(np.asanyarray(inferred.dataobj).transpose(2, 1, 0))
                examples.append(example_figure(image, masks, predictions, row, out, index, expected, scope))
                saved_metrics[row["sample_id"]] = {arm: saved_rows[row["sample_id"]]["predictions"][arm]["metrics"] for arm in ARMS}
            analysis = dict(schema_version="fakect.two-arm-ct/1", complete=True, arm_labels=LABELS,
                display_arms=list(ARMS), examples=examples, metrics=saved_metrics, selection_scope=selection,
                style=STYLE, models_run=False, metrics_recomputed=False)
            if original_analysis is not None:
                preserved = copy.deepcopy(original_analysis)
                preserved.update(analysis)
                preserved["figure"] = examples[0]
                preserved["metrics"] = {arm: original_analysis["metrics"][arm] for arm in ARMS}
                analysis = preserved
            (out / "analysis.json").write_text(json.dumps(analysis, indent=2, allow_nan=False) + "\n")
            (out / "index.html").write_text(html_page(examples, scope))
            group_reports[group] = analysis
        scenes, original_viewer = simplify_scenes(review_source / "report-3d", inputs)
        viewer = render_viewer(staging / "prespecified/report-3d", scenes,
            title="Aorta predictions · Pretrained and Fine-tuned",
            coordinate_description=original_viewer["coordinates"],
            footer_note=original_viewer["display"]["footer_note"], show_metrics=False)
        viewer["viewer"] = str(destination / "prespecified/report-3d/index.html")
        for group in group_reports:
            out = staging / group
            receipt = dict(schema_version="fakect.two-arm-ct-group/1", complete=True,
                examples=group_reports[group]["examples"], arm_labels=LABELS, display_arms=list(ARMS),
                output_sha256={str(path.relative_to(out)): digest(path) for path in out.rglob("*") if path.is_file()},
                native_geometry_verified=True, metrics_recomputed=False, models_run=False)
            if group == "prespecified":
                receipt["viewer"] = viewer
            (out / "provenance.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (staging / "analysis.json").write_text(json.dumps(dict(complete=True, groups=group_reports,
            viewer=viewer, meshes_copied_exactly=True, native_geometry_verified=True), indent=2, allow_nan=False) + "\n")
        shutil.copy2(Path(__file__), staging / "aorta_two_arm_ct.py")
        receipt = dict(schema_version="fakect.two-arm-ct-presentation/1", complete=True,
            display_arms=list(ARMS), arm_labels=LABELS, inputs_sha256=inputs,
            output_sha256={str(path.relative_to(staging)): digest(path) for path in staging.rglob("*") if path.is_file()},
            native_geometry_verified=True, metrics_recomputed=False, models_run=False,
            frozen_sources_changed=False)
        (staging / "provenance.json").write_text(json.dumps(receipt, indent=2, sort_keys=True) + "\n")
        (staging / "provenance.json.sha256").write_text(digest(staging / "provenance.json") + "\n")
        staging.rename(destination)
    except Exception:
        shutil.rmtree(staging)
        raise
    print(json.dumps(dict(output_directory=str(destination), examples=sum(len(g['examples']) for g in group_reports.values()), scenes=len(scenes), complete=True)))


if __name__ == "__main__":
    main()
