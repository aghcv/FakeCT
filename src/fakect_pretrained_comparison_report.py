"""Matched native-grid review of pretrained and two aorta fine-tuning arms.

All four comparison columns use the SAME current CT, native slice, HU window
and reference-defined display zoom. The reference is the label for that scan,
not the paired original scan. Exact core edit counts are read from native truth
and its known net change mask; they never come from predicted masks.
"""
import csv
import hashlib
from html import escape
import json
import os
from pathlib import Path
import shutil
import tempfile

import numpy as np

from fakect_pretrained_report import evaluate_arrays, _mask, _bbox, _sha

SCHEMA = "fakect.pretrained-aorta-comparison/1"
ARMS = ("pretrained", "baseline_only", "siblings")
LABELS = {"pretrained": "Pretrained", "baseline_only": "Baseline-only fine-tuning",
          "siblings": "Sibling fine-tuning"}
COLORS = {"pretrained": "#0072b2", "baseline_only": "#d55e00", "siblings": "#009e73"}
GT_COLOR, PRED_COLOR = "#0072b2", "#e69f00"
UPPER_COUNTS = (100, 500, 1000, 2000, 5000, 10000, 20000, 50000)
SCOPE = ("Matched known-atlas-region experiment: the same fixed anatomical box and thoracic Z extent, "
         "native CT, preprocessing, prediction grid and class extraction are used for all arms. "
         "The CT-only locator is not evaluated here. Native voxel sizes differ between phantoms; "
         "voxel-count curves describe sampling burden, not equivalent clinical severity.")


def _safe_json(value):
    if isinstance(value, np.ndarray):
        return _safe_json(value.tolist())
    if isinstance(value, np.generic):
        return _safe_json(value.item())
    if isinstance(value, dict):
        return {str(k): _safe_json(v) for k, v in value.items()}
    if isinstance(value, (tuple, list)):
        return [_safe_json(v) for v in value]
    if isinstance(value, float) and not np.isfinite(value):
        return None
    if isinstance(value, Path):
        return str(value)
    return value


def _metric(record, arm, region, name):
    result = record.get("predictions", {}).get(arm, {})
    if result.get("status") != "completed":
        return None
    data = result.get("metrics", {}).get(region)
    if data is None:
        return None
    value = data.get("surfaces", {}).get(name) if name in ("surface_dice", "hd95_mm") else data.get(name)
    return value if value is not None and np.isfinite(value) else None


def comparison_status(records):
    counts = {arm: {state: 0 for state in ("completed", "pending", "failed")} for arm in ARMS}
    paired = []
    for row in records:
        good = True
        for arm in ARMS:
            state = row.get("predictions", {}).get(arm, {}).get("status", "pending")
            state = state if state in counts[arm] else "pending"
            counts[arm][state] += 1
            good &= state == "completed"
        if good:
            paired.append(row["sample_id"])
    done = sum(c["completed"] for c in counts.values())
    status = "complete" if done == len(records) * len(ARMS) else "partial" if done else "pending"
    return dict(status=status, arms=counts, sample_count=len(records), paired_samples=len(paired),
                paired_sample_ids=paired, paired_case_ids=sorted({r["case_id"] for r in records if r["sample_id"] in paired}))


def select_examples(records):
    """Choose one original and two edited examples without prediction scores.

    Call and freeze the returned IDs BEFORE model inference. Removal and
    addition examples use the upper quartile burden and favor different cases.
    """
    if not records:
        return []
    chosen = []
    originals = sorted((r for r in records if r.get("baseline")), key=lambda r: (r["case_id"], r["sample_id"]))
    if originals:
        chosen.append(originals[0])
    for quantity in ("removed_voxels", "added_voxels"):
        candidates = [r for r in records if not r.get("baseline") and int(r.get(quantity, 0)) > 0 and
                      r["sample_id"] not in {c["sample_id"] for c in chosen}]
        different = [r for r in candidates if r["case_id"] not in {c["case_id"] for c in chosen}]
        candidates = different or candidates
        if candidates:
            values = np.array([int(r[quantity]) for r in candidates])
            target = np.quantile(values, .75)
            chosen.append(min(candidates, key=lambda r: (abs(int(r[quantity]) - target), r["case_id"], r["sample_id"])))
    if len(chosen) < 3:
        remaining = sorted((r for r in records if r["sample_id"] not in {c["sample_id"] for c in chosen}),
                           key=lambda r: (r["case_id"], r["sample_id"]))
        chosen.extend(remaining[:3 - len(chosen)])
    return [r["sample_id"] for r in chosen]


def _balanced_mean(rows, values):
    """Equal case weighting within family, then equal weighting of families."""
    case_means, families = {}, {}
    for row, value in zip(rows, values):
        if value is None:
            continue
        case_means.setdefault(row["case_id"], []).append(float(value))
        families[row["case_id"]] = row.get("anatomy_family") or row["case_id"]
    grouped = {}
    for case, points in case_means.items():
        grouped.setdefault(families[case], []).append(float(np.mean(points)))
    return None if not grouped else float(np.mean([np.mean(v) for v in grouped.values()]))


def burden_bins(records, quantity, *, region="lesion_local", metric="dice"):
    """Matched-panel mean curves; never derive independent-patient sibling CIs."""
    if quantity not in ("added_voxels", "removed_voxels"):
        raise ValueError("Burden quantity must be an exact native core addition/removal count")
    matched = [r for r in records if all(r.get("predictions", {}).get(a, {}).get("status") == "completed" for a in ARMS)]
    result = []
    lowers, uppers = (0, *UPPER_COUNTS), (*UPPER_COUNTS, None)
    for lower, upper in zip(lowers, uppers):
        rows = [r for r in matched if not r.get("baseline") and int(r.get(quantity, 0)) > lower and
                (upper is None or int(r.get(quantity, 0)) <= upper)]
        if not rows:
            continue
        item = dict(lower_exclusive=lower, upper_inclusive=upper, siblings=len(rows),
                    cases=len({r["case_id"] for r in rows}),
                    families=len({r.get("anatomy_family") or r["case_id"] for r in rows}),
                    mixed_siblings=sum(r.get("change_kind") == "mixed" for r in rows),
                    x_native_voxels=_balanced_mean(rows, [r[quantity] for r in rows]),
                    means={})
        for arm in ARMS:
            values = [_metric(r, arm, region, metric) for r in rows]
            item["means"][arm] = _balanced_mean(rows, values)
            item.setdefault("metric_samples", {})[arm] = sum(v is not None for v in values)
        result.append(item)
    return result


def _read_image(path, expected, provenance):
    import nibabel as nib
    path = Path(path).resolve()
    digest = _sha(path)
    if expected and digest != expected:
        raise ValueError("Frozen comparison input changed: " + str(path))
    provenance[str(path)] = digest
    return nib.load(str(path))


def _grid(reference, other, name):
    if len(other.shape) != 3 or other.shape != reference.shape or not np.allclose(other.affine, reference.affine, atol=1e-5, rtol=0):
        raise ValueError("Comparison requires an identical native grid and affine: " + name)


def _common(row, provenance):
    keys = ("input_nifti", "truth_nifti", "domain_nifti", "changed_nifti")
    if any(not row.get(k) for k in keys):
        raise ValueError("Every exported comparison scan requires CT, core truth, thoracic domain and exact known changes")
    images = {k: _read_image(row[k], row.get(k + "_sha256"), provenance) for k in keys}
    image = images["input_nifti"]
    for key, value in images.items():
        _grid(image, value, key)
    matrix = image.affine[:3, :3]
    spacing = np.linalg.norm(matrix, axis=0)
    if not np.allclose(matrix.T @ matrix, np.diag(spacing ** 2), atol=1e-5, rtol=0):
        raise ValueError("Native comparison grid must have orthogonal physical axes")
    masks = {key: _mask(np.asanyarray(value.dataobj).transpose(2, 1, 0)) for key, value in images.items() if key != "input_nifti"}
    return image, masks, spacing


def _extent(row, masks, spacing):
    truth, domain, changed = (masks[k] for k in ("truth_nifti", "domain_nifti", "changed_nifti"))
    added = int(np.count_nonzero(changed & truth & domain))
    removed = int(np.count_nonzero(changed & ~truth & domain))
    for name, actual in (("added_voxels", added), ("removed_voxels", removed)):
        stated = row.get("core_" + name, row.get(name))
        if stated is not None and int(stated) != actual:
            raise ValueError("Declared exact native CORE burden disagrees with exported truth/change mask: " + name)
    volume = float(np.prod(spacing))
    return dict(added_voxels=added, removed_voxels=removed, changed_voxels=added + removed,
                current_core_voxels=int(np.count_nonzero(truth & domain)),
                baseline_core_voxels=int(np.count_nonzero(truth & domain)) - added + removed,
                added_mm3=added * volume, removed_mm3=removed * volume, changed_mm3=(added + removed) * volume,
                voxel_volume_mm3=volume,
                change_kind="mixed" if added and removed else "addition_only" if added else "removal_only" if removed else "none")


def _save(fig, out, stem):
    for extension in ("png", "pdf", "svg"):
        fig.savefig(out / (stem + "." + extension), dpi=220, bbox_inches="tight")


def _plotting():
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fakect-pretrained-comparison-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    plt.rcParams.update({"svg.fonttype": "none", "pdf.fonttype": 42, "font.size": 9})
    return plt


def _overview_plots(records, out, arm_labels, cohort_extents=None, histogram_title=None):
    plt = _plotting()
    fig, axes = plt.subplots(1, 2, figsize=(11.8, 4.2))
    for ax, quantity, word in zip(axes, ("removed_voxels", "added_voxels"), ("Removed", "Added")):
        values = [r[quantity] for r in (cohort_extents if cohort_extents is not None else records)
                  if not r.get("baseline") and r.get(quantity, 0) > 0]
        histogram = []
        for lower, upper in zip((0, *UPPER_COUNTS), (*UPPER_COUNTS, None)):
            histogram.append(sum(v > lower and (upper is None or v <= upper) for v in values))
        labels = ["1–100", "101–500", "501–1k", "1k–2k", "2k–5k", "5k–10k", "10k–20k", "20k–50k", ">50k"]
        ax.bar(np.arange(len(labels)), histogram, color=COLORS["pretrained"], alpha=.85)
        ax.set_xticks(np.arange(len(labels)), labels, rotation=40, ha="right")
        ax.set(xlabel=word + " native CORE voxels", ylabel="Number of siblings", title=word + " burden · {} siblings".format(len(values)))
        ax.grid(axis="y", alpha=.15)
    fig.suptitle((histogram_title or ("Frozen population" if cohort_extents is not None else "Fixed evaluation panel")) +
                 " · exact native edit counts · mixed edits appear in both histograms")
    fig.tight_layout(); _save(fig, out, "cohort-native-counts"); plt.close(fig)

    profiles = {}
    panels = (("whole_thoracic_core", "dice", "Whole-core Dice"), ("lesion_local", "dice", "Change-local Dice"),
              ("lesion_local", "recall", "Change-local recall"), ("lesion_local", "surface_dice", "Change-local surface Dice · 1 mm"))
    for quantity, word in (("removed_voxels", "Removed"), ("added_voxels", "Added")):
        fig, axes = plt.subplots(2, 2, figsize=(11.5, 8.4))
        profile = {}
        for ax, (region, metric, title) in zip(axes.ravel(), panels):
            bins = burden_bins(records, quantity, region=region, metric=metric)
            profile[region + "/" + metric] = bins
            for arm in ARMS:
                for mixed in (False, True):
                    points = [(r[quantity], _metric(r, arm, region, metric)) for r in records if not r.get("baseline") and
                              r.get(quantity, 0) > 0 and (r.get("change_kind") == "mixed") == mixed]
                    points = [(x, y) for x, y in points if y is not None]
                    if points:
                        xy = np.asarray(points)
                        ax.scatter(xy[:, 0], xy[:, 1], s=15, alpha=.3, marker="x" if mixed else "o", color=COLORS[arm], linewidths=.5)
                usable = [b for b in bins if b["means"][arm] is not None]
                ax.plot([b["x_native_voxels"] for b in usable], [b["means"][arm] for b in usable], "o-",
                        color=COLORS[arm], lw=1.5, label=arm_labels[arm])
            ax.set(xlabel=word + " native CORE voxel count", ylabel=title, ylim=(-.03, 1.03), title=title)
            ax.set_xscale("symlog", linthresh=100)
            ax.grid(alpha=.15)
            if not bins:
                ax.text(.5, .5, "Matched three-arm results pending", transform=ax.transAxes, ha="center", color="#566573")
            if metric == "dice" and region == "lesion_local":
                for b in bins:
                    ax.annotate("{} cases / {} siblings".format(b["cases"], b["siblings"]),
                                (b["x_native_voxels"], .03), fontsize=7, rotation=25)
        axes[0, 0].legend(fontsize=8)
        fig.suptitle(word + " burden · points are siblings, lines are family/case-balanced matched-bin means")
        fig.tight_layout(); _save(fig, out, "performance-" + quantity.replace("_voxels", "")); plt.close(fig)
        profiles[quantity] = profile

    fig, axes = plt.subplots(1, 2, figsize=(10.4, 4.8))
    for ax, (left, right) in zip(axes, (("pretrained", "baseline_only"), ("baseline_only", "siblings"))):
        for baseline, marker in ((True, "*"), (False, "o")):
            points = [(_metric(r, left, "whole_thoracic_core", "dice"), _metric(r, right, "whole_thoracic_core", "dice"))
                      for r in records if bool(r.get("baseline")) == baseline]
            points = [(x, y) for x, y in points if x is not None and y is not None]
            if points:
                xy = np.array(points); ax.scatter(xy[:, 0], xy[:, 1], s=55 if baseline else 17, marker=marker,
                                                 color=COLORS[right], alpha=.65, label="Original controls" if baseline else "Siblings")
        ax.plot([0, 1], [0, 1], "--", color="#888", lw=1)
        ax.set(xlim=(-.03, 1.03), ylim=(-.03, 1.03), xlabel=arm_labels[left] + " · Dice", ylabel=arm_labels[right] + " · Dice")
        ax.set_aspect("equal"); ax.grid(alpha=.15)
        if ax.get_legend_handles_labels()[0]: ax.legend(fontsize=8)
    fig.suptitle("Paired predictions for the identical current scan · whole-core Dice")
    fig.tight_layout(); _save(fig, out, "paired-performance"); plt.close(fig)

    cases = sorted({r["case_id"] for r in records})
    for quantity, word in (("removed_voxels", "Removed"), ("added_voxels", "Added")):
        cols = min(3, len(cases)); rows = int(np.ceil(len(cases) / cols))
        fig, axes = plt.subplots(rows, cols, figsize=(4.1 * cols, 3.3 * rows), squeeze=False)
        for ax, case in zip(axes.ravel(), cases):
            for arm in ARMS:
                points = [(r.get(quantity, 0), _metric(r, arm, "whole_thoracic_core", "dice")) for r in records
                          if r["case_id"] == case and (r.get("baseline") or r.get(quantity, 0) > 0)]
                points = [(x, y) for x, y in points if y is not None]
                if points:
                    xy = np.asarray(points); ax.scatter(xy[:, 0], xy[:, 1], s=15, alpha=.7, color=COLORS[arm], label=arm_labels[arm])
            ax.set(title="Phantom " + case, xlabel=word + " native CORE voxels", ylabel="Whole-core Dice", ylim=(-.03, 1.03))
            ax.set_xscale("symlog", linthresh=100); ax.grid(alpha=.15)
        for ax in axes.ravel()[len(cases):]: ax.set_visible(False)
        if axes.ravel()[0].get_legend_handles_labels()[0]: axes.ravel()[0].legend(fontsize=7)
        fig.tight_layout(); _save(fig, out, "case-facets-" + quantity.replace("_voxels", "")); plt.close(fig)

    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.2))
    for ax, quantity, word in zip(axes, ("removed_mm3", "added_mm3"), ("Removed", "Added")):
        for arm in ARMS:
            points = [(r.get(quantity, 0), _metric(r, arm, "lesion_local", "dice")) for r in records if not r.get("baseline") and r.get(quantity, 0) > 0]
            points = [(x, y) for x, y in points if y is not None]
            if points:
                xy = np.asarray(points); ax.scatter(xy[:, 0], xy[:, 1], s=17, alpha=.5, color=COLORS[arm], label=arm_labels[arm])
        ax.set(xlabel=word + " native CORE volume · mm³", ylabel="Change-local Dice", ylim=(-.03, 1.03))
        ax.set_xscale("symlog", linthresh=10); ax.grid(alpha=.15)
    if axes[0].get_legend_handles_labels()[0]: axes[0].legend(fontsize=8)
    fig.tight_layout(); _save(fig, out, "physical-volume-adjunct"); plt.close(fig)
    return profiles


def _example_figure(image, masks, predictions, row, out, index, arm_labels, *, hu_window=(-200, 500), zoom_context_mm=12.):
    """One current image per orientation, copied unchanged across all columns."""
    plt = _plotting()
    truth, domain, changed = (masks[k] for k in ("truth_nifti", "domain_nifti", "changed_nifti"))
    truth = truth & domain; changed = changed & domain
    spacing = np.linalg.norm(image.affine[:3, :3], axis=0)
    origin = np.asarray(row.get("export", {}).get("origin_ijk", row.get("export_origin_ijk", [0, 0, 0])), int)
    focus = changed if changed.any() else truth
    box = _bbox(focus, np.ceil(zoom_context_mm / spacing[::-1]).astype(int))
    if box is None:
        box = tuple(slice(0, n) for n in truth.shape)
    positions = [int(np.argmax(focus.sum(axis=tuple(i for i in range(3) if i != a)))) for a in range(3)]
    fig, axes = plt.subplots(3, 4, figsize=(15.6, 11.6))
    metadata = dict(sample_id=row["sample_id"], baseline=bool(row.get("baseline")), hu_window=list(hu_window),
                    display_zoom_source="Native reference and known changes only; fixed before inspecting predictions",
                    low_export_kji=[s.start for s in box], high_export_kji_exclusive=[s.stop for s in box],
                    source_origin_ijk=origin.tolist(), orientations=[])
    for axis, name in enumerate(("Axial", "Coronal", "Sagittal")):
        fixed = positions[axis]
        slices = list(box); slices[axis] = fixed
        ijk_slices = tuple(slices[::-1])
        plane = np.asarray(image.dataobj[ijk_slices], dtype=np.float32).T
        dimensions = [a for a in range(3) if a != axis]
        vertical, horizontal = dimensions
        horizontal_origin, vertical_origin = origin[2 - horizontal], origin[2 - vertical]
        extent = [box[horizontal].start + horizontal_origin - .5, box[horizontal].stop + horizontal_origin - .5,
                  box[vertical].start + vertical_origin - .5, box[vertical].stop + vertical_origin - .5]
        native_fixed = fixed + origin[2 - axis]
        centre_export_ijk = np.array([(s.start + s.stop - 1) / 2 for s in box][::-1], dtype=float)
        centre_export_ijk[2 - axis] = fixed
        centre_ras = (np.asarray(image.affine) @ np.r_[centre_export_ijk, 1.])[:3]
        metadata["orientations"].append(dict(name=name, fixed_export_index=fixed, fixed_source_index=int(native_fixed),
            fixed_native_axis="kji"[axis], native_spacing_ijk_mm=spacing.tolist(),
            slice_centre_ras_mm=centre_ras.tolist(),
            ct_plane_sha256=hashlib.sha256(np.ascontiguousarray(plane).tobytes()).hexdigest(),
            identical_current_ct_across_columns=True))
        for col, arm in enumerate((None, *ARMS)):
            ax = axes[axis, col]
            ax.imshow(plane, origin="lower", cmap="gray", vmin=hu_window[0], vmax=hu_window[1], extent=extent,
                      interpolation="nearest", aspect=spacing[::-1][vertical] / spacing[::-1][horizontal])
            for mask, color, width in ((truth, GT_COLOR, .85),
                                      (None if arm is None or arm not in predictions else predictions[arm] & domain, PRED_COLOR, 1.25)):
                if mask is None: continue
                local = mask[tuple(slices)]
                if local.any() and not local.all():
                    ax.contour(np.arange(box[horizontal].start, box[horizontal].stop) + horizontal_origin,
                               np.arange(box[vertical].start, box[vertical].stop) + vertical_origin,
                               local.astype(float), levels=[.5], colors=[color], linewidths=width)
            if axis == 0:
                ax.set_title("Reference for THIS scan" if arm is None else arm_labels[arm], fontsize=10)
            if arm is not None and arm not in predictions:
                state = row.get("predictions", {}).get(arm, {}).get("status", "pending")
                ax.text(.5, .92, state.upper(), transform=ax.transAxes, ha="center", color="#b33", backgroundcolor="white")
            ax.set_xlabel("ijk"[2 - horizontal] + " · source voxel index")
            ax.set_ylabel("ijk"[2 - vertical] + " · source voxel index")
            if col == 0:
                coordinate = "{}={:.1f} mm RAS".format("xyz"[2 - axis], centre_ras[2 - axis])
                ax.text(.02, .02, "{} · {}={}\n{} · {:.3g} mm/voxel".format(
                        name, "kji"[axis], native_fixed, coordinate, spacing[2 - axis]),
                        transform=ax.transAxes, color="white", fontsize=8, backgroundcolor="#444")
    tag = "Unmodified original CT" if row.get("baseline") else "Current modified sibling CT"
    fig.suptitle(row["sample_id"] + " · " + tag + "\nSame current CT, slices, HU window and zoom · blue reference / orange prediction", fontsize=12)
    fig.tight_layout(); stem = "example-{:02d}".format(index); _save(fig, out, stem); plt.close(fig)
    return dict(metadata, figure=stem + ".png")


def _example_scene(masks, predictions, spacing, row, arm_labels):
    from fakect_two_stage_3d import build_scene
    domain = masks["domain_nifti"]
    truth = masks["truth_nifti"] & domain
    display = dict(truth=truth, known_changes=masks["changed_nifti"] & domain)
    display.update({arm: value & domain for arm, value in predictions.items()})
    union = np.logical_or.reduce(list(display.values()))
    box = _bbox(union, np.ceil(3 / spacing[::-1]).astype(int))
    if box is None:
        return None, "No native masks available for the comparison scene"
    shape = tuple(s.stop - s.start for s in box)
    if np.prod(shape) > 16_777_216 or min(shape) < 2:
        return None, "Exact native display crop exceeds the bounded viewer grid; quantitative metrics remain available"
    display = {key: value[box] for key, value in display.items()}
    modes = []
    truth_layer = dict(key="truth", label="Reference for THIS scan", color=GT_COLOR, visible=True, opacity=.45)
    for arm in ARMS:
        if arm not in predictions: continue
        modes.append(dict(key=arm, label=arm_labels[arm], explanation="Same current scan and exact native grid; reference is evaluation only.",
                          layers=[truth_layer, dict(key=arm, label=arm_labels[arm], color=PRED_COLOR, visible=True, opacity=.45),
                                  dict(key="known_changes", label="Known baseline differences · evaluation only", color="#cc79a7", visible=False, opacity=.8)]))
    if not modes:
        modes = [dict(key="reference", label="Reference · prediction arms pending", explanation="No completed prediction arm is available.", layers=[truth_layer])]
    scene = build_scene(display, spacing, sample_id=row["sample_id"], modes=modes,
                        metadata=dict(case_id=row["case_id"], is_control=bool(row.get("baseline")),
                                      low_export_kji=[s.start for s in box], high_export_kji_exclusive=[s.stop for s in box],
                                      export_origin_ijk=row.get("export", {}).get("origin_ijk", row.get("export_origin_ijk", [0, 0, 0]))),
                        show_metrics=False)
    return scene, None


def write_report(manifest, output_directory, *, settings=None):
    """Write a new matched review; unfinished arms remain explicitly pending."""
    provenance = {}
    if isinstance(manifest, (str, Path)):
        source = Path(manifest).resolve(); provenance[str(source)] = _sha(source); manifest = json.loads(source.read_text())
    samples = manifest.get("samples", [])
    if not samples or any(not r.get("sample_id") or not r.get("case_id") for r in samples) or len({r["sample_id"] for r in samples}) != len(samples):
        raise ValueError("Matched comparison requires a unique fixed sample panel")
    example_ids = manifest.get("example_sample_ids", [])
    if len(example_ids) > 3 or len(set(example_ids)) != len(example_ids) or not set(example_ids) <= {r["sample_id"] for r in samples}:
        raise ValueError("At most three preregistered example IDs must belong to the fixed panel")
    destination = Path(output_directory).resolve()
    if destination.exists(): raise FileExistsError("A matched review must use a new output directory")
    destination.parent.mkdir(parents=True, exist_ok=True)
    out = Path(tempfile.mkdtemp(prefix="." + destination.name + "-", dir=destination.parent))
    arm_labels = dict(LABELS, **manifest.get("arm_labels", {}))
    scope = manifest.get("display_scope", SCOPE)
    records, examples, scenes, scene_warnings = [], [], [], []
    try:
        for sample in samples:
            row = {k: sample.get(k) for k in ("sample_id", "case_id", "anatomy_family", "baseline", "benchmark_role", "split")}
            row.update({k:sample[k] for k in ('target_semantics','branch_extent','target_components',
                'added_aorta_core_voxels','removed_aorta_core_voxels') if k in sample})
            row["predictions"] = {}
            prepared = all(sample.get(k) and Path(sample[k]).is_file() for k in ("input_nifti", "truth_nifti", "domain_nifti", "changed_nifti"))
            if not prepared:
                row.update({k: sample[k] for k in ("added_voxels", "removed_voxels", "changed_voxels", "added_mm3", "removed_mm3", "change_kind") if k in sample})
                for arm in ARMS: row["predictions"][arm] = dict(status="pending", reason="Common native CT/reference export is pending")
                records.append(row); continue
            image, masks, spacing = _common(sample, provenance)
            row.update(_extent(sample, masks, spacing))
            components = {}
            for name, artifact in sample.get('target_components', {}).items():
                component = _read_image(artifact['path'], artifact['sha256'], provenance)
                _grid(image, component, name)
                components[name] = _mask(np.asanyarray(component.dataobj).transpose(2,1,0)) & masks['domain_nifti']
            if components:
                if set(components) != {'aorta_core','connected_branches'} or np.any(components['aorta_core'] & components['connected_branches']) or not np.array_equal(components['aorta_core'] | components['connected_branches'],masks['truth_nifti'] & masks['domain_nifti']):
                    raise ValueError('Core and connected-branch references must exactly partition the current target')
            predictions = {}
            for arm in ARMS:
                result = sample.get("predictions", {}).get(arm, {})
                state = result.get("status", "pending")
                if state != "completed":
                    row["predictions"][arm] = dict(status=state, failure=result.get("failure")); continue
                if not result.get("prediction_nifti"):
                    raise ValueError("A completed comparison arm requires its exact native prediction")
                inferred = _read_image(result["prediction_nifti"], result.get("prediction_nifti_sha256"), provenance)
                _grid(image, inferred, arm)
                prediction = _mask(np.asanyarray(inferred.dataobj).transpose(2, 1, 0))
                metrics = evaluate_arrays(masks["truth_nifti"], prediction, spacing, domain_mask=masks["domain_nifti"],
                                         changed_mask=masks["changed_nifti"], **(settings or {}))
                row["predictions"][arm] = dict(status="completed", metrics=metrics)
                if components:
                    row['predictions'][arm]['component_recalls'] = {name:dict(truth_voxels=int(mask.sum()),
                        recovered_voxels=int(np.count_nonzero(mask & prediction)),
                        recall=float(np.count_nonzero(mask & prediction)/mask.sum()) if mask.any() else None)
                        for name,mask in components.items()}
                if sample["sample_id"] in example_ids:
                    predictions[arm] = prediction
            records.append(row)
            if sample["sample_id"] in example_ids:
                display_row = dict(sample, **row)
                examples.append(_example_figure(image, masks, predictions, display_row, out, example_ids.index(sample["sample_id"]), arm_labels))
                scene, warning = _example_scene(masks, predictions, spacing, sample, arm_labels)
                if scene is not None: scenes.append(scene)
                if warning: scene_warnings.append(dict(sample_id=sample["sample_id"], warning=warning))
        state = comparison_status(records)
        cohort_extents = manifest.get("cohort_extents")
        if cohort_extents is not None:
            if not isinstance(cohort_extents, list) or any(not r.get("sample_id") or
                    any(not isinstance(r.get(k), int) or isinstance(r.get(k), bool) or r[k] < 0
                        for k in ("added_voxels", "removed_voxels")) for r in cohort_extents):
                raise ValueError("Population histogram requires exact nonnegative native CORE extent records")
            if len({r["sample_id"] for r in cohort_extents}) != len(cohort_extents):
                raise ValueError("Population extent identities must be unique")
        profiles = _overview_plots(records, out, arm_labels, cohort_extents,manifest.get('histogram_scope'))
        viewer = None
        if scenes:
            from fakect_two_stage_3d import render_viewer
            viewer = render_viewer(out / "report-3d", scenes, title="Matched native aorta predictions · prespecified examples",
                coordinate_description="Native-grid millimeters relative to each display crop; export/source origins are saved in analysis.json.",
                footer_note=scope, show_metrics=False)
            viewer["viewer"] = str(destination / "report-3d/index.html")
        payload = dict(schema_version=SCHEMA, **state, scope=scope, arm_labels=arm_labels, settings=settings or {},
                       source_protocol=manifest.get("protocol", {}), examples=examples, example_sample_ids=example_ids,
                       example_selection="Frozen before inference, by case and exact edit-count burden; never by improvement",
                       samples=records, burden_profiles=profiles, viewer=viewer, scene_warnings=scene_warnings,
                       original_controls=sum(bool(r.get("baseline")) for r in records),
                       unchanged_siblings=sum(not r.get("baseline") and r.get("changed_voxels") == 0 for r in records),
                       native_extent_samples=sum("added_voxels" in r and "removed_voxels" in r for r in records),
                       histogram_scope=manifest.get('histogram_scope') or ("Frozen population" if cohort_extents is not None else "Fixed evaluation panel"),
                       histogram_extent_samples=len(cohort_extents) if cohort_extents is not None else len(records),
                       independence="Digital siblings are correlated; descriptive family/case means, no independent-patient confidence intervals")
        if manifest.get('target_policy'):payload['target_policy']=manifest['target_policy']
        (out / "analysis.json").write_text(json.dumps(_safe_json(payload), indent=2, allow_nan=False) + "\n")
        csv_rows = []
        for row in records:
            for arm in ARMS:
                flat = {k: row.get(k) for k in ("sample_id", "case_id", "anatomy_family", "baseline", "added_voxels", "removed_voxels", "added_mm3", "removed_mm3", "change_kind")}
                flat.update(arm=arm, status=row["predictions"][arm].get("status", "pending"))
                for name in ('aorta_core','connected_branches'):
                    flat[name+'_recall']=row['predictions'][arm].get('component_recalls',{}).get(name,{}).get('recall')
                for region in ("whole_thoracic_core", "lesion_local"):
                    for metric in ("dice", "precision", "recall", "surface_dice", "hd95_mm"):
                        flat[region + "_" + metric] = _metric(row, arm, region, metric)
                    data = row["predictions"][arm].get("metrics", {}).get(region)
                    for key in ("truth_components", "prediction_components"):
                        flat[region + "_" + key] = None if data is None else data[key]["components"]
                csv_rows.append(flat)
        with (out / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(csv_rows[0])); writer.writeheader(); writer.writerows(csv_rows)
        table = "".join("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(escape(arm_labels[a]),
                        state["arms"][a]["completed"], state["arms"][a]["pending"], state["arms"][a]["failed"]) for a in ARMS)
        figures = "".join('<section><h2>' + escape(e["sample_id"]) + '</h2><p>' + ('Unmodified original CT' if e["baseline"] else 'Current sibling CT, with its own reference label') + '</p><a href="' + e["figure"].replace(".png", ".pdf") + '">PDF</a> · <a href="' + e["figure"].replace(".png", ".svg") + '">SVG</a><img src="' + e["figure"] + '" alt="Same current CT compared across the fixed three prediction arms"></section>' for e in examples)
        document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Matched aorta fine-tuning comparison</title>'
            '<style>body{font:16px/1.5 system-ui;max-width:1450px;margin:auto;padding:24px;color:#183144}img{max-width:100%}table{border-collapse:collapse;width:100%}th,td{text-align:left;padding:8px;border-bottom:1px solid #ddd}.note{background:#eef4f6;padding:16px}h1,h2{line-height:1.2}</style></head><body>'
            '<h1>Matched pretrained aorta comparison · ' + escape(state["status"].upper()) + '</h1><p class="note">' + escape(scope) + '</p>'
            '<table><thead><tr><th>Arm</th><th>Completed scans</th><th>Pending scans</th><th>Failed scans</th></tr></thead><tbody>' + table + '</tbody></table>'
            '<p>' + str(state["paired_samples"]) + ' of ' + str(len(samples)) + ' scans have completed all three arms. Missing predictions are marked pending; they are not zero-valued scores. Curves use only the matched three-arm panel. Points show available individual scans.</p>'
            '<h2>Exact native CORE edits</h2><p>Counts are derived from this scan’s native reference and its known net difference from the original core. Mixed changes appear in both addition and removal plots. Original controls and unchanged siblings have zero counts; no global percentage-change labels are used.</p><img src="cohort-native-counts.png" alt="Sibling counts by native removed and added core voxel bins">'
            '<h2>Performance across removal burden</h2><img src="performance-removed.png" alt="Matched performance as a function of exact native removed core voxels">'
            '<h2>Performance across addition burden</h2><img src="performance-added.png" alt="Matched performance as a function of exact native added core voxels">'
            '<p>Lines average siblings within each phantom, then cases within anatomical family, then families. These are descriptive means; siblings do not count as independent patients. Bin case/sibling counts are annotated and recorded in analysis.json. No sibling-based patient confidence intervals are calculated.</p>'
            '<h2>Paired performance</h2><img src="paired-performance.png" alt="Paired predictions on identical current scans">'
            '<h2>Phantom-specific native counts</h2><img src="case-facets-removed.png" alt="Removal performance separately for each phantom"><img src="case-facets-added.png" alt="Addition performance separately for each phantom">'
            '<h2>Physical-volume adjunct</h2><p>Native voxel counts differ in physical volume between phantoms. This companion view uses mm³ without replacing the exact-count plots.</p><img src="physical-volume-adjunct.png" alt="Physical-volume adjunct performance view">'
            '<h2>Prespecified native CT examples</h2><p>Each row shows the same current scan, native slice, HU window and display zoom in all four columns. Blue is that scan’s atlas reference; orange is the relevant prediction. Display zooms and slice locations are selected from reference/known changes, not prediction performance. These display selections never alter inference or quantitative domains.</p>'
            + figures + ('<p>Example IDs have not been frozen; no examples were selected from observed improvements.</p>' if not example_ids else '')
            + ('<h2>Interactive native surfaces</h2><p>Switch the comparison arm, rotate and zoom, and toggle each reference/prediction/change surface independently at 15%, 45% or 80% opacity.</p><a href="report-3d/index.html">Open 3D comparison</a><iframe src="report-3d/index.html" title="Matched native prediction surfaces" style="width:100%;height:950px;border:0"></iframe>' if viewer else '')
            + '<h2>Exports</h2><p><a href="metrics.csv">Native metrics CSV</a> · <a href="analysis.json">Full metrics, bin counts and coordinate metadata</a> · <a href="provenance.json">Source checksums</a></p><p>All figures are available as PNG, PDF and SVG beside this report. Whole-core and change-local metrics include failed-status records explicitly; artificial crop/domain surface caps are excluded according to the recorded fixed settings.</p></body></html>')
        (out / "report.html").write_text(document, encoding="utf-8")
        if manifest.get('target_policy'):
            component_rows=[]
            for arm in ARMS:
                values={name:[r['predictions'][arm].get('component_recalls',{}).get(name,{}).get('recall') for r in records] for name in ('aorta_core','connected_branches')}
                cells=[]
                for name in ('aorta_core','connected_branches'):
                    defined=[v for v in values[name] if v is not None]
                    cells.append('{:.3f}'.format(float(np.mean(defined))) if defined else 'Pending')
                component_rows.append('<tr><td>'+escape(arm_labels[arm])+'</td>'+''.join('<td>'+v+'</td>' for v in cells)+'</tr>')
            component_table='<h2>Coverage of core and connected branches</h2><p>These recalls measure recovery of reference-positive voxels; they do not assign false positives to a named branch. Predictions remain unmasked by the target; combined-target metrics assess false positives over the common spatial domain.</p><table><tr><th>Model</th><th>Core recall</th><th>Connected-branch recall</th></tr>'+''.join(component_rows)+'</table>'
            document=document.replace('<h2>Exports</h2>',component_table+'<h2>Exports</h2>')
            document=document.replace('native CORE','native target').replace('Exact native CORE','Exact native target')
            (out/'report.html').write_text(document,encoding='utf-8')
        output_hashes = {str(p.relative_to(out)): _sha(p) for p in out.rglob("*") if p.is_file()}
        (out / "provenance.json").write_text(json.dumps(dict(schema_version=SCHEMA, input_sha256=provenance,
            output_sha256=output_hashes, reporter_sha256=_sha(Path(__file__)), models_trained_by_report=False,
            inference_recentered_on_truth=False), indent=2) + "\n")
        out.rename(destination)
        return dict(report=str(destination / "report.html"), status=state["status"], paired_samples=state["paired_samples"], samples=len(samples))
    except Exception:
        shutil.rmtree(out, ignore_errors=True)
        raise
