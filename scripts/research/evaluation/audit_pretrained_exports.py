#!/usr/bin/env python3
"""Independently audit frozen pretrained exports with nibabel and native ATN.

Reads existing images and masks only. Writes a new, checksum-bound JSON audit;
it never rewrites an export, frozen manifest, source volume, or existing audit.
Run in the isolated TotalSegmentator environment that supplies nibabel.
"""
import argparse
from datetime import datetime, timezone
import gzip
import hashlib
import json
from pathlib import Path
import sys

import nibabel as nib
import numpy as np


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def bound_file(path, expected=None):
    path = Path(path).resolve()
    before = path.stat()
    digest = sha256(path)
    after = path.stat()
    require((before.st_size, before.st_mtime_ns) ==
            (after.st_size, after.st_mtime_ns), f"File changed while hashing: {path}")
    if expected is not None:
        require(digest == expected, f"SHA-256 mismatch: {path}")
    return {"path": str(path), "bytes": after.st_size,
            "mtime_ns": after.st_mtime_ns, "sha256": digest}


def bound_json(path, dependencies, expected=None, sidecar=False):
    path = Path(path).resolve()
    if sidecar:
        checksum_path = path.with_name(path.name + ".sha256")
        expected = checksum_path.read_text().strip().split()[0]
        dependencies.append(bound_file(checksum_path))
    dependencies.append(bound_file(path, expected))
    return json.loads(path.read_text())


def count_binary(path, image):
    total = voxels = 0
    opener = gzip.open if str(path).endswith(".gz") else open
    with opener(path, "rb") as stream:
        stream.read(int(image.dataobj.offset))
        for block in iter(lambda: stream.read(8 * 1024 * 1024), b""):
            values = np.frombuffer(block, np.uint8)
            require(bool(np.all(values <= 1)), f"Nonbinary mask: {path}")
            total += int(values.sum())
            voxels += values.size
    require(voxels == int(np.prod(image.shape)), f"Wrong mask payload size: {path}")
    return total


def audit(experiment):
    dependencies = []
    lock = bound_json(experiment / "benchmark-lock.json", dependencies, sidecar=True)
    rows = {row["sample_id"]: row for row in lock["samples"]}
    summary = {"schema_version": "fakect.pretrained-export-audit/1",
               "created_utc": datetime.now(timezone.utc).isoformat(),
               "experiment_directory": str(experiment),
               "software": {"python": sys.version, "nibabel": nib.__version__,
                            "numpy": np.__version__},
               "script": bound_file(__file__), "dependencies": dependencies,
               "cases": [], "artifacts": [], "headers_checked": 0,
               "artifacts_sha256_checked": 0, "source_samples_checked": 0,
               "full_hu_arrays_allocated": 0, "sources_modified": False,
               "limitations": [
                   "Anatomical boundaries remain an engineering proxy.",
                   "Eight baseline HU samples per case are checked, not every image voxel.",
                   "Native ATN files are bound by recorded size and mtime; no whole-source hash is claimed.",
                   "Baseline branch exclusion is checked against the frozen branch mask; edited masks are checked against recorded counts.",
                   "RAS encoding is checked against the prior generator/landmark evidence, not an independent registration."]}
    seen_samples = set()
    for case in lock["config"]["benchmark"]["cases"]:
        prep_path = experiment / "cases" / case / "prepared.json"
        prep = bound_json(prep_path, dependencies, sidecar=True)
        require(prep["complete"], f"Incomplete preparation: {case}")
        require(len(prep["samples"]) == len(lock["config"]["benchmark"]["panel"]),
                f"Unexpected panel size: {case}")
        result = {"case_id": case, "roles": [], "voxel_samples": []}
        baseline = None
        for sample in prep["samples"]:
            require(sample["sample_id"] not in seen_samples, "Duplicate prepared sample")
            seen_samples.add(sample["sample_id"])
            row = rows[sample["sample_id"]]
            ex = sample["export"]
            images, counts = {}, {}
            for kind, key in (("hu", "input_nifti"), ("target", "truth_nifti"),
                              ("domain", "domain_nifti"), ("changed", "changed_nifti")):
                path = Path(sample[key])
                record = bound_file(path, sample[key + "_sha256"])
                image = nib.load(str(path))
                images[kind] = image
                require(tuple(image.shape) == tuple(ex["shape_ijk"]), f"Shape mismatch: {path}")
                require(np.allclose(image.affine, ex["affine_ras_ijk"], rtol=0, atol=5e-5), f"Affine mismatch: {path}")
                require(np.allclose(image.get_qform(), image.get_sform(), rtol=0, atol=5e-5), f"qform/sform mismatch: {path}")
                require(int(image.header["qform_code"]) == int(image.header["sform_code"]) == 1, f"Wrong form codes: {path}")
                require(nib.aff2axcodes(image.affine) == ("L", "P", "S"), f"Wrong orientation: {path}")
                require(np.allclose(image.header.get_zooms(), ex["spacing_ijk_mm"], rtol=0, atol=1e-7), f"Spacing mismatch: {path}")
                require(image.header.get_xyzt_units()[0] == "mm", f"Wrong spatial units: {path}")
                require(image.get_data_dtype() == np.dtype("float32" if kind == "hu" else "uint8"), f"Wrong dtype: {path}")
                record.update({"case_id": case, "sample_id": sample["sample_id"],
                    "role": sample["benchmark_role"], "kind": kind,
                    "shape_ijk": list(image.shape), "affine_ras_ijk": image.affine.tolist(),
                    "qform_ras_ijk": image.get_qform().tolist(),
                    "sform_ras_ijk": image.get_sform().tolist(),
                    "qform_code": int(image.header["qform_code"]),
                    "sform_code": int(image.header["sform_code"]),
                    "axis_directions": list(nib.aff2axcodes(image.affine)),
                    "spacing_ijk_mm": [float(x) for x in image.header.get_zooms()],
                    "spatial_units": image.header.get_xyzt_units()[0],
                    "dtype": str(image.get_data_dtype()), "header_checks_passed": True})
                if kind != "hu":
                    counts[kind] = count_binary(path, image)
                    require(counts[kind] == ex["statistics"]["mask_voxel_counts"][kind], f"Mask count mismatch: {path}")
                    record.update({"binary": True, "mask_voxel_count": counts[kind],
                                   "expected_mask_voxel_count": ex["statistics"]["mask_voxel_counts"][kind]})
                summary["artifacts"].append(record)
                summary["headers_checked"] += 1
                summary["artifacts_sha256_checked"] += 1
            require(ex["image_units"] == "HU" and ex["mu_water_cm_inverse"] == .1614, f"Unexpected HU contract: {case}")
            require(ex["target_metadata"]["semantics"] == "atlas thoracic aorta core; proximal branches excluded", f"Unexpected target contract: {case}")
            result["roles"].append({"role": sample["benchmark_role"],
                "sample_id": sample["sample_id"], "mask_voxel_counts": counts,
                "broad_reference_after_voxels": sample["mask_counts"]["after"]})
            if row["baseline"]:
                require(baseline is None and counts["changed"] == 0, f"Invalid baseline: {case}")
                baseline = sample, row, images
        require(baseline is not None, f"Missing baseline: {case}")
        sample, row, images = baseline
        ex = sample["export"]
        selection_path = Path(row["selection_manifest"])
        selection = bound_json(selection_path, dependencies, row["selection_manifest_sha256"])
        mask_path = Path(selection["mask_npz"])
        if not mask_path.is_absolute():
            mask_path = selection_path.parent / mask_path
        dependencies.append(bound_file(mask_path, selection["mask_npz_sha256"]))
        with np.load(mask_path, allow_pickle=False) as masks:
            core = np.asarray(masks["core_mask"], np.uint8)
            origin = np.asarray(selection["geometry"]["crop_low_ijk"], int)
            local = origin - np.asarray(ex["origin_ijk"], int)
            target = np.asarray(images["target"].dataobj[
                local[0]:local[0] + core.shape[2], local[1]:local[1] + core.shape[1],
                local[2]:local[2] + core.shape[0]], dtype=np.uint8).transpose(2, 1, 0)
            require(np.array_equal(target, core), f"Baseline core differs: {case}")
            require(int(core.sum()) == ex["statistics"]["mask_voxel_counts"]["target"] == selection["counts"]["core_voxels"], f"Baseline core count differs: {case}")
            branch = np.asarray(masks["branch_mask"], bool)
            require(not np.any(target[branch]), f"Baseline includes branches: {case}")
            branch_count = int(branch.sum())
            require(branch_count == selection["counts"]["branch_voxels"], f"Branch count differs: {case}")
            core_indices = np.argwhere(core)
            core_point = core_indices[len(core_indices) // 2][::-1] + origin
        result.update({"baseline_core_exact": True, "baseline_changed_voxels": 0,
                       "proximal_branch_voxels_excluded": branch_count})
        source_path = Path(row["source_atn"]["path"])
        source_stat = source_path.stat()
        require((source_stat.st_size, source_stat.st_mtime_ns) ==
                (row["source_atn"]["bytes"], row["source_atn"]["mtime_ns"]), f"ATN source changed: {case}")
        result["source_atn"] = dict(row["source_atn"])
        source = np.memmap(source_path, mode="r", dtype="<f4", shape=tuple(row["source_shape_kji"]))
        crop_origin = np.asarray(ex["origin_ijk"], int)
        shape = np.asarray(ex["shape_ijk"], int)
        points = [np.array([0, 0, crop_origin[2]], int),
                  np.array([shape[0]-1, shape[1]-1, crop_origin[2]+shape[2]-1], int), core_point]
        k = int(crop_origin[2] + shape[2] // 2)
        plane = np.asarray(source[k])
        values = np.unique(plane)
        values = values[values > 0]
        require(len(values) >= 5, f"Insufficient distinct tissue sample values: {case}")
        for index in np.linspace(0, len(values)-1, 5).astype(int):
            j, i = np.argwhere(plane == values[index])[0]
            points.append(np.array([i, j, k], int))
        for global_ijk in points:
            local = global_ijk - crop_origin
            raw = float(source[tuple(global_ijk[::-1])])
            observed = float(images["hu"].dataobj[tuple(local)])
            expected = 1000 * (raw * ex["calibration_evidence"]["source_atn_to_cm_inverse_scale"] / .1614 - 1)
            error = abs(observed - expected)
            require(error < .001, f"ATN/HU mismatch: {case} {global_ijk.tolist()}")
            result["voxel_samples"].append({"global_ijk": global_ijk.tolist(),
                "export_local_ijk": local.tolist(), "source_atn_per_pixel": raw,
                "export_hu": observed, "expected_hu": expected, "abs_error_hu": error})
            summary["source_samples_checked"] += 1
        after = source_path.stat()
        require((after.st_size, after.st_mtime_ns) == (source_stat.st_size, source_stat.st_mtime_ns), f"ATN source changed during audit: {case}")
        result.update({"shape_ijk": ex["shape_ijk"], "affine_ras_ijk": ex["affine_ras_ijk"],
                       "spacing_ijk_mm": ex["spacing_ijk_mm"],
                       "calibration_evidence": ex["calibration_evidence"]})
        summary["cases"].append(result)
        print(f"Passed case {case}: four exports, eight native ATN/HU samples", flush=True)
    require(seen_samples == set(rows), "Prepared sample set differs from frozen lock")
    require(summary["headers_checked"] == 4 * len(rows), "Missing artifact headers")
    for record in dependencies + summary["artifacts"] + [summary["script"]]:
        stat = Path(record["path"]).stat()
        require((stat.st_size, stat.st_mtime_ns) == (record["bytes"], record["mtime_ns"]),
                f"Audit dependency changed during audit: {record['path']}")
    summary["max_abs_error_hu"] = max(s["abs_error_hu"] for c in summary["cases"] for s in c["voxel_samples"])
    summary["passed"] = True
    return summary


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--experiment", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True, help="New audit directory; existing paths are refused")
    args = parser.parse_args()
    experiment, output = args.experiment.resolve(), args.output.resolve()
    require(not output.exists(), f"Refusing to overwrite audit directory: {output}")
    result = audit(experiment)
    result["command"] = [sys.executable, str(Path(__file__).resolve()),
                         "--experiment", str(experiment), "--output", str(output)]
    output.mkdir(parents=True, exist_ok=False)
    receipt = output / "receipt.json"
    with receipt.open("x") as stream:
        json.dump(result, stream, indent=2, sort_keys=True)
        stream.write("\n")
    with (output / "receipt.json.sha256").open("x") as stream:
        stream.write(sha256(receipt) + "\n")
    print(f"Audit passed: {result['headers_checked']} headers/hashes, "
          f"{result['source_samples_checked']} HU samples; receipt {receipt}")


if __name__ == "__main__":
    main()
