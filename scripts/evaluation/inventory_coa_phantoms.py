#!/usr/bin/env python3
"""Inventory XCAT templates and propose native-grid aortic ROI review seeds.

Sources are read only. Localization uses a named artery label internally, never
surface ordinals or an assumed mesh-to-volume affine. Output INIs should select
the artery category without an original-ID restriction. All proposed coordinates
require review; body-size scaling is an initialization heuristic, not a lesion
severity calibration or evidence of independent patient ancestry.
"""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import sys
import time

import numpy as np
from scipy import ndimage
from skimage.graph import route_through_array

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT / "src"))
from fakect_roi import read_crop, tube_crop_bounds
from fakect_tissues import read_organ_table
from locate_aorta import scan_planes, surface_evidence

CASES = ["260602"] + [str(v) for v in range(260611, 260622)]


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def parameters(path):
    return {k: v.strip() for k, v in re.findall(r"^\s*(\w+)\s*=\s*([^#\n]+)", path.read_text(), re.M)}


def largest_plane_components(mask, k):
    groups, n = ndimage.label(mask[k])
    rows = []
    for group in range(1, n + 1):
        points = np.argwhere(groups == group)
        if len(points) >= 5:
            rows.append((len(points), float(points[:, 0].mean()), group))
    return groups, sorted(rows, reverse=True)[:2]


def interior_endpoint(groups, group, k, spacing):
    inside = groups == group
    edt = ndimage.distance_transform_edt(inside, sampling=spacing[1::-1])
    j, i = np.unravel_index(int(np.argmax(edt)), edt.shape)
    return (int(k), int(j), int(i))


def localize(case, entry, catalog, args):
    path = Path(entry["frame_1"]["act"]["path"])
    before = path.stat()
    shape = tuple(entry["shape_kji"])
    spacing = np.asarray(entry["spacing_ijk_mm"])
    matches = [r for r in catalog["records"] if r["original_name"] == "dias_aorta"
               and r["classification"]["tissue_name"] == "artery"]
    if len(matches) != 1:
        raise ValueError("Catalog must identify a unique dias_aorta artery label")
    label = int(matches[0]["original_id"])
    dictionary = read_organ_table(args.dictionary)
    if dictionary.get(label) != "dias_aorta":
        raise ValueError("Dictionary disagrees with catalog aorta identity")
    started = time.monotonic()
    levels = list(range(0, shape[0], args.coarse_stride))
    points, rows, plane_hash = scan_planes(path, shape, levels, label)
    if not len(points):
        raise ValueError("Aorta absent from coarse planes; reduce stride before proceeding")
    margin = np.array([16, 16, args.coarse_stride])
    low = np.maximum(0, points.min(axis=0)[::-1] - margin)
    high = np.minimum(np.asarray(shape[::-1]), points.max(axis=0)[::-1] + margin + 1)
    if np.prod(high - low) > 8_000_000:
        raise ValueError("Localization crop exceeds 8 million voxels")
    labels, crop_source = read_crop(path, shape, tuple(map(int, low)), tuple(map(int, high)))
    target = labels == label
    if any(np.any(np.take(target, index, axis=axis)) for axis in range(3) for index in (0, -1)):
        raise ValueError("Target touches localization crop boundary; enlarge bounded localization margin")
    groups, _ = ndimage.label(target, structure=ndimage.generate_binary_structure(3, 1))
    sizes = np.bincount(groups.ravel())
    sizes[0] = 0
    mask = groups == int(np.argmax(sizes))
    locations = np.argwhere(mask)
    upper = int(locations[:, 0].max())
    # The adult reference starts 72 mm below its superior cap and ends 108 mm
    # below it. Size scaling only proposes endpoints; each case remains unreviewed.
    scale = entry["body_height_mm"] / args.reference_height_mm
    desired_start = upper - 72 * scale / spacing[2]
    desired_end = upper - 108 * scale / spacing[2]
    candidates = []
    for k in range(max(0, upper - int(np.ceil(130 * scale / spacing[2]))), upper):
        plane_groups, components = largest_plane_components(mask, k)
        if len(components) == 2:
            components.sort(key=lambda row: row[1])
            if (components[1][1] - components[0][1]) * spacing[1] >= 15 * scale:
                candidates.append((abs(k - desired_start), k, components))
    if not candidates:
        raise ValueError("No two-limb aortic slice found; manual endpoints needed")
    _, start_k, components = min(candidates)
    plane_groups, _ = largest_plane_components(mask, start_k)
    start = interior_endpoint(plane_groups, components[0][2], start_k, spacing)
    end_k = int(np.clip(round(desired_end), locations[:, 0].min(), upper - 1))
    end_groups, end_components = largest_plane_components(mask, end_k)
    if not end_components:
        raise ValueError("No descending limb at proposed end level")
    end_group = max(end_components, key=lambda row: row[1])[2]
    end = interior_endpoint(end_groups, end_group, end_k, spacing)
    edt = ndimage.distance_transform_edt(mask, sampling=spacing[::-1])
    costs = np.where(mask, 1 / np.maximum(edt, .1 * scale) ** 2, np.inf)
    route, cost = route_through_array(costs, start, end, fully_connected=True, geometric=True)
    if not np.isfinite(cost):
        raise ValueError("No connected interior aortic path between proposed endpoints")
    route = np.asarray(route, dtype=int)
    native = route[:, ::-1] + low
    arc = np.r_[0., np.cumsum(np.linalg.norm(np.diff(native, axis=0) * spacing, axis=1))]
    sample = np.unique(np.r_[0, np.searchsorted(arc, np.arange(15 * scale, arc[-1], 15 * scale)), len(route) - 1])
    nodes = native[sample]
    radii = np.ceil(np.maximum(10 * scale, edt[tuple(route[sample].T)] + 3 * scale) * 10) / 10
    # Account for one 9*scale dilation plus the 3-mm reassignment search and
    # native-grid guards. Generation settings still need full recipe validation.
    padding = float(np.ceil((radii.max() + 9 * scale + 3 + 3 * spacing.max()) * 10) / 10)
    crop_low, crop_high = tube_crop_bounds(nodes, padding, shape, spacing)
    dimensions = np.asarray(crop_high) - crop_low
    stride = 2
    while np.prod(np.ceil(dimensions / stride).astype(int) + 2) * 4 > 650_000:
        stride += 1
    after = path.stat()
    if (before.st_size, before.st_mtime_ns) != (after.st_size, after.st_mtime_ns):
        raise ValueError("Source changed during localization")
    return {
        "schema_version": "fakect.aorta-localization/2", "case_id": case, "frame": 1,
        "status": "provisional_unreviewed", "created_utc": datetime.now(timezone.utc).isoformat(),
        "source": {"path": str(path), "bytes": before.st_size, "mtime_ns": before.st_mtime_ns,
                   "shape_kji": list(shape), "spacing_ijk_mm": spacing.tolist(), "dtype": "<f4"},
        "identity": {"original_id": label, "original_name": "dias_aorta", "tissue": "artery",
                     "catalog_sha256": sha256(args.catalog), "dictionary_sha256": sha256(args.dictionary),
                     "usage": "Internal localization only; final ROI selection uses artery category and blank source_ids."},
        "coarse_scan": {"k_stride": args.coarse_stride, "planes_read": len(levels),
                        "plane_payload_sha256": plane_hash, "planes_with_target": rows},
        "source_crop_provenance": crop_source,
        "observed_target": {"voxels": int(target.sum()), "largest_component_voxels": int(mask.sum()),
                            "components_6": int(len(sizes) - 1),
                            "bbox_low_ijk": (np.argwhere(target).min(axis=0)[::-1] + low).tolist(),
                            "bbox_high_ijk_exclusive": (np.argwhere(target).max(axis=0)[::-1] + low + 1).tolist()},
        "provisional_arch_roi": {"shape": "tube", "tissue": "artery", "source_ids": [],
                                 "center_ijk": nodes.tolist(), "radius_mm": radii.tolist(),
                                 "crop_half_width_mm": padding, "crop_low_ijk": list(crop_low),
                                 "crop_high_ijk_exclusive": list(crop_high),
                                 "crop_voxels": int(np.prod(dimensions)), "volume_stride": stride,
                                 "slice_ijk": nodes[len(nodes) // 2].tolist(),
                                 "coordinate_reviewed": False, "path_length_mm": float(arc[-1]),
                                 "path_order": "low-j limb ascending to superior arch to high-j limb descending; anatomical interpretation requires review",
                                 "path_method": "26-neighbor minimum inverse-square interior-distance path, sampled approximately every 15*body_scale mm",
                                 "radius_method": "max(10*body_scale, interior EDT+3*body_scale) mm rounded up to 0.1 mm; selection envelope only"},
        "scale_initialization": {"reference_body_height_mm": args.reference_height_mm,
                                 "body_height_ratio": scale,
                                 "dilation_distance_mm": [round(v * scale, 3) for v in (3, 6, 9)],
                                 "coa_distance_mm": [round(v * scale, 3) for v in (2, 4, 6, 8, 10)],
                                 "inner_arch_distance_mm": round(7 * scale, 3),
                                 "note": "Initialization only; comparable body scale is not equivalent vessel size or disease severity."},
        "surface": surface_evidence(args.base / case / f"{case}_1_heart.raw", "dias_aorta"),
        "limitations": ["Unreviewed coordinates: inspect Global/Local reports before freezing or preparing.",
                        "Coarse scan plus bounded crop does not rule out small disconnected remote label pieces.",
                        "Native low-j/high-j limb ordering follows the adult reference; anatomical orientation must be confirmed.",
                        "Named aorta label is used only for the seed. Category-based ROI may include nearby non-aortic arteries.",
                        "No native-volume affine is inferred from raw surface coordinates."],
        "script_sha256": sha256(__file__), "elapsed_seconds": time.monotonic() - started,
    }


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base", type=Path, default=Path("/home/aghorban/slurm/xcat"))
    parser.add_argument("--cases", nargs="+", default=CASES)
    parser.add_argument("--audit", type=Path, default=ROOT / "docs/integration/2026-09-16/xcat-results.json")
    parser.add_argument("--catalog", type=Path, default=ROOT / "outputs/tissues/atlas-v1/label-catalog.json")
    parser.add_argument("--dictionary", type=Path, default=ROOT / "references/atlas/xcat/organ_ids.primary.txt")
    parser.add_argument("--output", type=Path, default=ROOT / "docs/integration/2026-09-18/multiphantom-inventory.json")
    parser.add_argument("--localization-directory", type=Path, default=ROOT / "outputs/cohorts/coa/localization")
    parser.add_argument("--coarse-stride", type=int, default=32)
    parser.add_argument("--reference-height-mm", type=float, default=1752.3262)
    parser.add_argument("--inventory-only", action="store_true")
    args = parser.parse_args()
    if args.output.exists():
        parser.error("Inventory already exists; choose a new output path")
    audit = json.loads(args.audit.read_text())
    prior = {c["case_id"]: c for c in audit["cases"]}
    catalog = json.loads(args.catalog.read_text())
    baseline = parameters(args.base / "260602.par")
    report = {"schema_version": "fakect.multiphantom-inventory/1",
              "created_utc": datetime.now(timezone.utc).isoformat(),
              "source_root": str(args.base), "audit_path": str(args.audit), "audit_sha256": sha256(args.audit),
              "catalog_path": str(args.catalog), "catalog_sha256": sha256(args.catalog), "cases": [],
              "family_policy": "Provisional conservative grouping by age template family across sex. File names identify model templates, not independently sampled patients. All frames/edits of a source stay in one family.",
              "source_mutation": False,
              "scope": "Metadata/stat checks cover all channels. Optional localization reads sparse complete axial planes plus one bounded native crop, never full volumes."}
    for case in args.cases:
        p = args.base / f"{case}.par"
        par = parameters(p)
        old = prior[case]
        shape = (int(par["endslice"]) - int(par["startslice"]) + 1,
                 int(par["y_array_size"]), int(par["x_array_size"]))
        spacing = [float(par["pixel_width"]) * 10] * 2 + [float(par["slice_width"]) * 10]
        expected = int(np.prod(shape)) * 4
        directory = args.base / case
        files = list(directory.iterdir())
        bins = [f for f in files if f.suffix == ".bin"]
        raws = [f for f in files if f.suffix == ".raw"]
        frames = set(range(1, int(par["out_frames"]) + 1))
        absent = [f"{ch}_{fr}" for ch in ("act", "atn") for fr in sorted(frames)
                  if not (directory / f"{case}_{ch}_{fr}.bin").exists()]
        bad = [str(f) for f in bins if f.stat().st_size != expected]
        age_match = re.search(r"_(infant|\d+yr)_", par["organ_file"])
        family = "xcat_reference_" + (age_match[1] if age_match else "adult")
        entry = {"case_id": case, "directory": str(directory), "par_path": str(p), "par_sha256": sha256(p),
                 "log_path": str(directory / f"{case}_log"), "log_sha256": sha256(directory / f"{case}_log"),
                 "organ_file": par["organ_file"], "heart_base": par["heart_base"],
                 "sex_parameter": par["gender"], "provisional_anatomy_family": family,
                 "family_reviewed": False, "shape_kji": list(shape), "spacing_ijk_mm": spacing,
                 "body_height_mm": old["log_metadata"]["body_height_mm"],
                 "parameter_differences_from_260602": {k: v for k, v in par.items() if baseline.get(k) != v},
                 "source_template": {"path": str(args.base / par["organ_file"]),
                                     "bytes": (args.base / par["organ_file"]).stat().st_size,
                                     "sha256": sha256(args.base / par["organ_file"])},
                 "heart_template": {"path": str(args.base / par["heart_base"]),
                                    "sha256": sha256(args.base / par["heart_base"])},
                 "binary_count": len(bins), "raw_count": len(raws),
                 "instantaneous_frames": sorted(frames), "expected_volume_bytes": expected,
                 "missing_frame_channels": absent, "binary_size_mismatches": bad,
                 "averages_present": all((directory / f"{case}_{c}_av.bin").exists() for c in ("act", "atn")),
                 "raw_frame_1_groups": sorted(f.name for f in raws if f.name.startswith(f"{case}_1_")),
                 "frame_1": {ch: {"path": str(directory / f"{case}_{ch}_1.bin"),
                                  "bytes": (directory / f"{case}_{ch}_1.bin").stat().st_size,
                                  "mtime_ns": (directory / f"{case}_{ch}_1.bin").stat().st_mtime_ns}
                             for ch in ("act", "atn")},
                 "prior_audit_par_log_mismatches": old["par_log_mismatches"],
                 "prior_sampled_unknown_ids": old["samples"]["act_1"]["unknown_ids"],
                 "review_status": "reviewed_reference_roi" if case == "260602" else "awaiting_roi_review"}
        if list(shape) != old["shape_kji"] or not np.allclose(spacing, old["spacing_ijk_mm"]):
            raise ValueError(f"Current geometry differs from prior source audit for {case}")
        if absent or bad:
            raise ValueError(f"Incomplete or mismatched binary channels for {case}")
        if not args.inventory_only:
            out = args.localization_directory / f"{case}.json"
            if out.exists():
                candidate = json.loads(out.read_text())
                if (candidate["source"]["mtime_ns"] != entry["frame_1"]["act"]["mtime_ns"]
                        or candidate["script_sha256"] != sha256(__file__)):
                    raise ValueError(f"Existing localization provenance differs: {out}")
            else:
                print(f"Localizing {case}: {par['organ_file']}", flush=True)
                candidate = localize(case, entry, catalog, args)
                out.parent.mkdir(parents=True, exist_ok=True)
                out.write_text(json.dumps(candidate, indent=2) + "\n")
            entry["localization_path"] = str(out)
            entry["localization_sha256"] = sha256(out)
            entry["provisional_arch_roi"] = candidate["provisional_arch_roi"]
            entry["scale_initialization"] = candidate["scale_initialization"]
            print(f"{case}: {len(candidate['provisional_arch_roi']['center_ijk'])} nodes, "
                  f"length {candidate['provisional_arch_roi']['path_length_mm']:.1f} mm", flush=True)
        report["cases"].append(entry)
        # Write incremental progress separately so readers never mistake it for
        # the complete immutable inventory requested by --output.
        progress = args.output.with_suffix(".incomplete.json")
        progress.parent.mkdir(parents=True, exist_ok=True)
        progress.write_text(json.dumps(report, indent=2) + "\n")
    report["validation"] = {"case_count": len(report["cases"]),
                            "all_binary_sizes_match": True, "all_frame_channels_present": True,
                            "family_assignments_reviewed": False,
                            "roi_review_still_required": [c for c in args.cases if c != "260602"]}
    args.output.write_text(json.dumps(report, indent=2) + "\n")
    progress.unlink()
    print(str(args.output), flush=True)


if __name__ == "__main__":
    main()
