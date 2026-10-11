"""Native-grid review of a fixed, region-assisted pretrained aorta benchmark.

No model is fitted or tuned here. Exported image, truth, domain, change mask and
prediction must share a native grid and affine. Surface distances use physical,
area-weighted voxel-face centres; artificial domain/crop caps are excluded.
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
from scipy.ndimage import distance_transform_edt, generate_binary_structure, label
from scipy.spatial import cKDTree

SCHEMA = "fakect.pretrained-aorta-review/1"
FACE_LIMIT = 3_000_000
STRUCTURE = generate_binary_structure(3, 1)
FOOTER = ("Fixed known atlas thoracic region; aorta core excludes proximal branches. "
          "This is a region-assisted synthetic CT benchmark, not automatic thorax detection or clinical validation. "
          "Scores use exact native grids. Display surfaces may have crop closures; quantitative surface metrics exclude domain caps.")


def _mask(value, shape=None):
    value = np.asarray(value)
    if value.ndim != 3 or (shape is not None and value.shape != shape):
        raise ValueError("Masks must share a three-dimensional native grid")
    if value.dtype != np.bool_ and not np.isin(value, (0, 1)).all():
        raise ValueError("Native evaluation masks must be binary")
    return value.astype(bool, copy=False)


def _bbox(mask, halo=None):
    if not np.any(mask):
        return None
    lo, hi = [], []
    for axis in range(3):
        occupied = np.flatnonzero(mask.any(axis=tuple(i for i in range(3) if i != axis)))
        padding = 0 if halo is None else int(halo[axis])
        lo.append(max(0, int(occupied[0]) - padding))
        hi.append(min(mask.shape[axis], int(occupied[-1]) + 1 + padding))
    return tuple(slice(a, b) for a, b in zip(lo, hi))


def _bounds(slices):
    return {"low_kji": [s.start for s in slices], "high_kji_exclusive": [s.stop for s in slices]}


def _inside(domain, spacing_kji, margin):
    # Padding is essential: scipy's all-true EDT otherwise sees an implicit
    # background only on one side rather than all six crop faces.
    distance = distance_transform_edt(np.pad(domain, 1, constant_values=False), sampling=spacing_kji)[1:-1, 1:-1, 1:-1]
    return domain & (distance > margin)


def _surface_faces(mask, domain, spacing_kji, margin):
    admissible = _inside(domain, spacing_kji, margin)
    padded = np.pad(mask, 1, constant_values=False)
    padded_domain = np.pad(domain, 1, constant_values=False)
    centre = tuple(slice(1, n + 1) for n in mask.shape)
    points, weights = [], []
    total = 0
    for axis in range(3):
        area = float(np.prod(np.delete(spacing_kji, axis)))
        for sign in (-1, 1):
            neighbour = list(centre)
            neighbour[axis] = slice(1 + sign, mask.shape[axis] + 1 + sign)
            face = mask & ~padded[tuple(neighbour)] & admissible & padded_domain[tuple(neighbour)]
            indices = np.argwhere(face)
            total += len(indices)
            if total > FACE_LIMIT:
                return None, None
            xyz = indices.astype(np.float64) * spacing_kji
            xyz[:, axis] += sign * .5 * spacing_kji[axis]
            points.append(xyz)
            weights.append(np.full(len(indices), area))
    return np.concatenate(points, axis=0), np.concatenate(weights)


def _weighted_quantile(values, weights, quantile):
    order = np.argsort(values)
    cumulative = np.cumsum(weights[order])
    position = min(len(order) - 1, int(np.searchsorted(cumulative, quantile * cumulative[-1], side="left")))
    return float(values[order[position]])


def surface_metrics(truth, prediction, domain, spacing_kji, *, tolerance_mm=1., boundary_margin_mm=2.):
    """Symmetric face-area-weighted distances on the native voxelized surface."""
    points_t, weights_t = _surface_faces(truth, domain, spacing_kji, boundary_margin_mm)
    points_p, weights_p = _surface_faces(prediction, domain, spacing_kji, boundary_margin_mm)
    output = dict(surface_method="native voxel-face centres, face-area weights, symmetric pooled weighted HD95",
                  tolerance_mm=float(tolerance_mm), boundary_margin_mm=float(boundary_margin_mm),
                  surface_dice=None, hd95_mm=None, mean_surface_distance_mm=None)
    if points_t is None or points_p is None:
        return dict(output, status="surface_budget_exceeded", face_limit=FACE_LIMIT)
    output.update(truth_surface_area_mm2=float(weights_t.sum()), prediction_surface_area_mm2=float(weights_p.sum()),
                  truth_surface_faces=len(points_t), prediction_surface_faces=len(points_p))
    if not len(points_t) or not len(points_p):
        status = "no_interior_surfaces" if not len(points_t) and not len(points_p) else "empty_prediction_surface" if not len(points_p) else "empty_truth_surface"
        return dict(output, status=status, surface_dice=0. if bool(len(points_t)) != bool(len(points_p)) else None)
    distances_t = cKDTree(points_p).query(points_t, workers=1)[0]
    distances_p = cKDTree(points_t).query(points_p, workers=1)[0]
    values, weights = np.r_[distances_t, distances_p], np.r_[weights_t, weights_p]
    covered = weights_t[distances_t <= tolerance_mm].sum() + weights_p[distances_p <= tolerance_mm].sum()
    return dict(output, status="complete", surface_dice=float(covered / weights.sum()),
                hd95_mm=_weighted_quantile(values, weights, .95),
                mean_surface_distance_mm=float(np.sum(values * weights) / weights.sum()))


def component_metrics(mask, domain, spacing_kji, *, boundary_margin_mm=2.):
    """Report fragmentation without treating boundary-cut components as islands."""
    components, count = label(mask & domain, STRUCTURE)
    sizes = np.bincount(components.ravel())[1:]
    boundary = domain & ~_inside(domain, spacing_kji, boundary_margin_mm)
    touching = set(int(v) for v in np.unique(components[boundary]) if v)
    interior_ids = [i for i in range(1, count + 1) if i not in touching]
    return dict(connectivity=6, components=int(count), total_voxels=int(sizes.sum()),
                largest_component_voxels=int(sizes.max()) if count else 0,
                largest_component_fraction=float(sizes.max() / sizes.sum()) if count else None,
                boundary_touching_components=len(touching), interior_components=len(interior_ids),
                interior_component_voxels=int(sum(sizes[i - 1] for i in interior_ids)),
                interpretation="Boundary-touching components may be cut by the evaluation domain; this is not a physiological lumen-continuity test.")


def _region_metrics(truth, prediction, domain, spacing_kji, tolerance, margin):
    truth, prediction = truth & domain, prediction & domain
    tp = int(np.count_nonzero(truth & prediction))
    t, p = int(truth.sum()), int(prediction.sum())
    empty = t + p == 0
    return dict(truth_voxels=t, prediction_voxels=p, true_positive_voxels=tp,
                false_positive_voxels=p - tp, false_negative_voxels=t - tp,
                evaluation_voxels=int(domain.sum()),
                dice=1. if empty else 2 * tp / (t + p),
                precision=tp / p if p else (1. if not t else 0.),
                recall=tp / t if t else (1. if not p else 0.),
                truth_volume_mm3=t * float(np.prod(spacing_kji)),
                prediction_volume_mm3=p * float(np.prod(spacing_kji)),
                surfaces=surface_metrics(truth, prediction, domain, spacing_kji,
                                         tolerance_mm=tolerance, boundary_margin_mm=margin),
                truth_components=component_metrics(truth, domain, spacing_kji, boundary_margin_mm=margin),
                prediction_components=component_metrics(prediction, domain, spacing_kji, boundary_margin_mm=margin))


def evaluate_arrays(truth_kji, prediction_kji, spacing_ijk_mm, *, domain_mask=None,
                    changed_mask=None, local_context_mm=5., boundary_margin_mm=2.,
                    surface_tolerance_mm=1., lesion_group_gap_mm=2., max_local_regions=16):
    """Native overlap, cap-excluded surface distances and lesion-local overlap.

    Lesion regions are grouped by physical proximity of known net baseline
    changes, then evaluated in their bounding boxes plus fixed context. This
    uses truth only for evaluation, never to recenter model input/predictions.
    """
    truth = _mask(truth_kji)
    prediction = _mask(prediction_kji, truth.shape)
    domain = None if domain_mask is None else _mask(domain_mask, truth.shape)
    changed = None if changed_mask is None else _mask(changed_mask, truth.shape)
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    if spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0):
        raise ValueError("Spacing must contain three finite positive native dimensions")
    settings = [local_context_mm, boundary_margin_mm, surface_tolerance_mm, lesion_group_gap_mm]
    if any(not np.isfinite(v) or v < 0 for v in settings) or local_context_mm <= boundary_margin_mm:
        raise ValueError("Use nonnegative finite settings and local context larger than boundary margin")
    if isinstance(max_local_regions, bool) or not isinstance(max_local_regions, int) or max_local_regions < 1:
        raise ValueError("Local region report budget must be a positive integer")
    spacing_kji = spacing[::-1]
    halo = np.ceil((max(local_context_mm + lesion_group_gap_mm, boundary_margin_mm) + 2 * spacing.max()) / spacing_kji).astype(int)
    union = truth | prediction
    if changed is not None:
        union |= changed
    if domain is not None:
        union &= domain
    crop = _bbox(union, halo)
    if crop is None:
        # Keep a tiny but valid physical grid for an explicitly empty panel.
        crop = tuple(slice(0, min(3, n)) for n in truth.shape)
    t, p = truth[crop], prediction[crop]
    d = np.ones(t.shape, bool) if domain is None else domain[crop]
    output = dict(schema_version=SCHEMA, array_order="kji", native_shape_kji=list(truth.shape),
                  spacing_ijk_mm=spacing.tolist(), computation_crop=_bounds(crop),
                  settings=dict(local_context_mm=float(local_context_mm), boundary_margin_mm=float(boundary_margin_mm),
                                surface_tolerance_mm=float(surface_tolerance_mm), lesion_group_gap_mm=float(lesion_group_gap_mm)),
                  whole_thoracic_core=_region_metrics(t, p, d, spacing_kji, surface_tolerance_mm, boundary_margin_mm),
                  lesion_local=None, local_regions=[])
    output["whole_thoracic_core"]["computation_domain_voxels"] = output["whole_thoracic_core"]["evaluation_voxels"]
    output["whole_thoracic_core"]["evaluation_voxels"] = truth.size if domain is None else int(domain.sum())
    output["prediction_voxels_outside_thoracic_domain"] = 0 if domain is None else int(np.count_nonzero(prediction & ~domain))
    output["truth_voxels_outside_thoracic_domain"] = 0 if domain is None else int(np.count_nonzero(truth & ~domain))
    if changed is None:
        output["local_status"] = "no_change_mask_supplied"
        return output
    c = changed[crop] & _inside(d, spacing_kji, boundary_margin_mm)
    output["known_changed_voxels_in_domain"] = int(np.count_nonzero(changed[crop] & d))
    output["known_changed_voxels_away_from_caps"] = int(c.sum())
    if not c.any():
        output["local_status"] = "unchanged_baseline_or_no_interior_changes"
        return output
    grouped = distance_transform_edt(~c, sampling=spacing_kji) <= lesion_group_gap_mm
    groups, count = label(grouped & d, STRUCTURE)
    sizes = np.bincount(groups[c].ravel(), minlength=count + 1)
    context = np.ceil(local_context_mm / spacing_kji).astype(int)
    local_union = np.zeros(c.shape, bool)
    regions = []
    for group in np.argsort(sizes[1:])[::-1] + 1:
        if not sizes[group]:
            continue
        box = _bbox(c & (groups == group), context)
        local_union[box] = True
        regions.append((int(group), box, int(sizes[group])))
    local_domain = local_union & _inside(d, spacing_kji, boundary_margin_mm)
    output["lesion_local"] = _region_metrics(t, p, local_domain, spacing_kji, surface_tolerance_mm, boundary_margin_mm)
    output["local_status"] = "complete"
    output["local_region_count"] = len(regions)
    output["local_regions_omitted_from_individual_table"] = max(0, len(regions) - max_local_regions)
    for group, box, size in regions[:max_local_regions]:
        region_domain = np.zeros(c.shape, bool)
        region_domain[box] = True
        region_domain &= _inside(d, spacing_kji, boundary_margin_mm)
        # Volumetric local overlap is more useful than dozens of repeated EDT
        # and mesh-distance calculations; aggregate local surfaces are above.
        actual, inferred = t & region_domain, p & region_domain
        tp, nt, npred = int((actual & inferred).sum()), int(actual.sum()), int(inferred.sum())
        global_slices = tuple(slice(s.start + outer.start, s.stop + outer.start) for s, outer in zip(box, crop))
        output["local_regions"].append(dict(group=group, changed_voxels=size, bounds_native=_bounds(global_slices),
            truth_voxels=nt, prediction_voxels=npred, dice=2 * tp / (nt + npred) if nt + npred else 1.,
            precision=tp / npred if npred else (1. if not nt else 0.), recall=tp / nt if nt else (1. if not npred else 0.)))
    return output


def _sha(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _load_sample(row, provenance):
    import nibabel as nib
    required = ("input_nifti", "truth_nifti", "prediction_nifti", "domain_nifti")
    if any(not row.get(key) for key in required):
        raise ValueError("Every completed benchmark sample requires input, truth, prediction and domain NIfTI")
    images = {}
    for key in (*required, "changed_nifti"):
        if not row.get(key):
            continue
        path = Path(row[key]).resolve()
        digest = _sha(path)
        if row.get(key + "_sha256") and digest != row[key + "_sha256"]:
            raise ValueError("Frozen benchmark input changed: " + key)
        provenance[str(path)] = digest
        images[key] = nib.load(str(path))
    reference = images["input_nifti"]
    for name, image in images.items():
        if len(image.shape) != 3 or image.shape != reference.shape or not np.allclose(image.affine, reference.affine, atol=1e-5, rtol=0):
            raise ValueError("Native prediction/export grid or affine mismatch: " + name)
    linear = np.asarray(reference.affine[:3, :3])
    spacing = np.linalg.norm(linear, axis=0)
    if not np.allclose(linear.T @ linear, np.diag(spacing ** 2), atol=1e-5, rtol=0):
        raise ValueError("Metric grid must have orthogonal axes; no implicit resampling is allowed")
    masks = {key: _mask(np.asanyarray(image.dataobj).transpose(2, 1, 0)) for key, image in images.items() if key != "input_nifti"}
    return reference, masks, spacing


def _slices(image, truth, prediction, changed, spacing, path, title, origin_ijk):
    os.environ.setdefault("MPLCONFIGDIR", "/tmp/fakect-pretrained-matplotlib")
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    profile = (changed if changed is not None and changed.any() else truth).sum(axis=(1, 2))
    positions = np.flatnonzero(profile)
    indices = [int(positions[min(len(positions) - 1, int(q * (len(positions) - 1)))]) for q in (.2, .5, .8)] if len(positions) else [0, truth.shape[0] // 2, truth.shape[0] - 1]
    box = _bbox(truth | prediction, np.ceil(12 / spacing[::-1]).astype(int))
    if box is None:
        box = tuple(slice(0, n) for n in truth.shape)
    fig, axes = plt.subplots(1, 3, figsize=(14, 5), constrained_layout=True)
    origin_ijk = np.asarray(origin_ijk, dtype=int)
    for ax, k in zip(axes, indices):
        plane = np.asarray(image.dataobj[box[2], box[1], k], dtype=np.float32).T
        ax.imshow(plane, origin="lower", cmap="gray", vmin=-200, vmax=500, aspect=spacing[1] / spacing[0],
                  extent=[box[2].start + origin_ijk[0] - .5, box[2].stop + origin_ijk[0] - .5,
                          box[1].start + origin_ijk[1] - .5, box[1].stop + origin_ijk[1] - .5])
        for mask, color in ((truth, "#0072b2"), (prediction, "#e69f00")):
            local = mask[k, box[1], box[2]]
            if local.any() and not local.all():
                ax.contour(np.arange(box[2].start, box[2].stop) + origin_ijk[0],
                           np.arange(box[1].start, box[1].stop) + origin_ijk[1], local.astype(float),
                           levels=[.5], colors=[color], linewidths=1.2)
        ax.set(title="Native k = {} · export k = {}".format(k + origin_ijk[2], k),
               xlabel="i · native voxel index", ylabel="j · native voxel index")
    fig.suptitle(title + " · blue truth / orange pretrained aorta", fontsize=12)
    fig.savefig(path, dpi=160)
    plt.close(fig)
    return indices


def write_report(manifest, output_directory, *, settings=None):
    """Evaluate an exported manifest and write a new portable native-grid review."""
    if isinstance(manifest, (str, Path)):
        manifest_path = Path(manifest).resolve()
        manifest = json.loads(manifest_path.read_text())
        input_provenance = {str(manifest_path): _sha(manifest_path)}
    else:
        input_provenance = {}
    rows = manifest.get("samples", [])
    if not rows or len({r.get("sample_id") for r in rows}) != len(rows):
        raise ValueError("The benchmark requires unique sample identities")
    destination = Path(output_directory).resolve()
    if destination.exists():
        raise FileExistsError("Benchmark review directory must be new")
    destination.parent.mkdir(parents=True, exist_ok=True)
    staging = Path(tempfile.mkdtemp(prefix="." + destination.name + "-", dir=destination.parent))
    records = []
    from fakect_two_stage_3d import build_scene, render_viewer
    try:
        def scene_iterator():
            for index, row in enumerate(rows):
                if row.get("status", "completed") != "completed":
                    records.append(dict(sample_id=row["sample_id"], case_id=row.get("case_id"), status=row["status"], failure=row.get("failure")))
                    continue
                image, masks, spacing = _load_sample(row, input_provenance)
                truth, prediction, domain = (masks[k] for k in ("truth_nifti", "prediction_nifti", "domain_nifti"))
                changed = masks.get("changed_nifti")
                if not row.get("baseline", False) and changed is None:
                    raise ValueError("Edited siblings require a known net core change mask for lesion-local evaluation")
                metrics = evaluate_arrays(truth, prediction, spacing, domain_mask=domain, changed_mask=changed, **(settings or {}))
                image_name = "slices-{:04d}.png".format(index)
                export_origin = row.get("export", {}).get("origin_ijk", row.get("export_origin_ijk", [0, 0, 0]))
                if len(export_origin) != 3 or any(int(v) != v for v in export_origin):
                    raise ValueError("Source-to-export native origin requires three integer indices")
                planes = _slices(image, truth & domain, prediction & domain, changed, spacing,
                                 staging / image_name, row["sample_id"], export_origin)
                record = dict(sample_id=row["sample_id"], case_id=row.get("case_id"), baseline=bool(row.get("baseline")),
                              status="completed", metrics=metrics, slice_image=image_name,
                              slice_export_k=planes, slice_native_k=[k + export_origin[2] for k in planes],
                              export_origin_ijk=list(export_origin),
                              metric_index_scope="Bounds index the exported native-grid NIfTI; add export_origin_ijk for whole-source indices",
                              native_affine_xyz=image.affine.tolist())
                records.append(record)
                display_union = truth | prediction
                if changed is not None:
                    display_union |= changed
                crop = _bbox(display_union & domain, np.ceil(3 / spacing[::-1]).astype(int))
                if crop is None:
                    crop = tuple(slice(0, min(3, n)) for n in truth.shape)
                display = {"truth": (truth & domain)[crop], "prediction": (prediction & domain)[crop],
                           "false_positive": (prediction & ~truth & domain)[crop], "false_negative": (truth & ~prediction & domain)[crop]}
                if changed is not None:
                    display["known_changes"] = (changed & domain)[crop]
                if np.prod(display["truth"].shape) > 16_777_216 or min(display["truth"].shape) < 2:
                    record["viewer_status"] = "display_crop_exceeds_exact_surface_budget"
                    continue
                layers = [dict(key="truth", label="Atlas thoracic aorta core · truth", color="#0072b2", visible=True, opacity=.45),
                          dict(key="prediction", label="Pretrained aorta · prediction", color="#e69f00", visible=True, opacity=.45),
                          dict(key="false_positive", label="False positive", color="#cc79a7", visible=False, opacity=.8),
                          dict(key="false_negative", label="False negative", color="#009e73", visible=False, opacity=.8)]
                if changed is not None:
                    layers.append(dict(key="known_changes", label="Known baseline difference · evaluation only", color="#b22222", visible=False, opacity=.8))
                scene = build_scene(display, spacing, sample_id=row["sample_id"], modes=[dict(key="benchmark", label="Truth and pretrained prediction", explanation=FOOTER, layers=layers)],
                                    metadata=dict(case_id=row.get("case_id"), is_control=bool(row.get("baseline")), native_display_crop=_bounds(crop)),
                                    show_metrics=False)
                record["viewer_status"] = "complete" if not scene["warnings"] else "one_or_more_exact_surfaces_over_budget"
                yield scene
        viewer = render_viewer(staging / "report-3d", scene_iterator(), title="Pretrained aorta · native-grid evaluation",
                               coordinate_description="Native-grid millimeters relative to each display crop; NIfTI affine and native crop bounds are recorded in metrics.json.",
                               footer_note=FOOTER, show_metrics=False)
        viewer["viewer"] = str(destination / "report-3d/index.html")
        payload = dict(schema_version=SCHEMA, scope=FOOTER, predictor=manifest.get("predictor", {}), settings=settings or {},
                       samples=records, completed_samples=sum(r["status"] == "completed" for r in records),
                       failed_samples=sum(r["status"] != "completed" for r in records), viewer=viewer)
        (staging / "metrics.json").write_text(json.dumps(payload, indent=2, allow_nan=False) + "\n")
        flat = []
        for r in records:
            f = {k: r.get(k) for k in ("sample_id", "case_id", "baseline", "status")}
            for region in ("whole_thoracic_core", "lesion_local"):
                m = r.get("metrics", {}).get(region)
                for key in ("dice", "precision", "recall", "truth_voxels", "prediction_voxels", "false_positive_voxels", "false_negative_voxels"):
                    f[region + "_" + key] = None if m is None else m[key]
                for key in ("surface_dice", "hd95_mm"):
                    f[region + "_" + key] = None if m is None else m["surfaces"][key]
            flat.append(f)
        with (staging / "metrics.csv").open("w", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(flat[0])); writer.writeheader(); writer.writerows(flat)
        table = "".join("<tr><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td><td>{}</td></tr>".format(
            escape(str(r["sample_id"])), escape(str(r.get("case_id", ""))), r.get("baseline", ""),
            "—" if r["whole_thoracic_core_dice"] is None else "{:.3f}".format(r["whole_thoracic_core_dice"]),
            "—" if r["lesion_local_dice"] is None else "{:.3f}".format(r["lesion_local_dice"]),
            "—" if r["whole_thoracic_core_hd95_mm"] is None else "{:.2f}".format(r["whole_thoracic_core_hd95_mm"])) for r in flat)
        cards = "".join('<details><summary>' + escape(r["sample_id"]) + '</summary><img src="' + r["slice_image"] + '" alt="Native axial CT slices with truth and prediction contours"></details>' for r in records if r["status"] == "completed")
        document = ('<!doctype html><html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Pretrained aorta benchmark</title>'
                    '<style>body{font:16px/1.5 system-ui;max-width:1450px;padding:24px;margin:auto;color:#183144}table{border-collapse:collapse;width:100%}td,th{padding:8px;text-align:left;border-bottom:1px solid #ddd}iframe{width:100%;height:1000px;border:0}img{max-width:100%}.note{padding:15px;background:#eef4f6}</style></head><body>'
                    '<h1>Pretrained aorta · native-grid benchmark</h1><p class="note">' + escape(FOOTER) + '</p>'
                    '<p>No training or threshold optimization was performed. Whole-core scores include every predicted voxel inside the common thoracic domain. Lesion-local boxes use known net baseline changes plus fixed physical context, and are used only for evaluation. Surface scores exclude a 2 mm boundary margin by default and use area-weighted native voxel-face centres.</p>'
                    '<table><thead><tr><th>Sample</th><th>Phantom</th><th>Baseline</th><th>Whole-core Dice</th><th>Lesion-local Dice</th><th>Whole-core HD95 · mm</th></tr></thead><tbody>' + table + '</tbody></table>'
                    '<h2>Interactive 3D review</h2><p>Rotate and zoom; independently hide each surface and set its opacity to 15%, 45% or 80%. Exact display surfaces exceeding the budget are explicitly omitted; native metrics remain available.</p><a href="report-3d/index.html">Open viewer</a><iframe src="report-3d/index.html" title="Truth and pretrained aorta surfaces"></iframe>'
                    '<h2>Three native axial views per scan</h2>' + cards + '<h2>Data and provenance</h2><p><a href="metrics.csv">All metrics CSV</a> · <a href="metrics.json">Metrics and coordinate definitions JSON</a> · <a href="provenance.json">Input and output checksums</a></p></body></html>')
        (staging / "report.html").write_text(document, encoding="utf-8")
        outputs = {str(p.relative_to(staging)): _sha(p) for p in staging.rglob("*") if p.is_file()}
        provenance = dict(schema_version=SCHEMA, input_sha256=input_provenance, outputs_sha256=outputs,
                          reporter_sha256=_sha(Path(__file__)), models_trained=False, inference_recentered_on_truth=False)
        (staging / "provenance.json").write_text(json.dumps(provenance, indent=2) + "\n")
        staging.rename(destination)
        return dict(report=str(destination / "report.html"), completed_samples=payload["completed_samples"], failed_samples=payload["failed_samples"])
    except Exception:
        shutil.rmtree(staging, ignore_errors=True)
        raise
