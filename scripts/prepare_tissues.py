#!/usr/bin/env python3
"""Build reversible anatomical tissue views and profile measured XCAT attenuation.

The profile command reuses the bounded paired samples from audit_xcat.py. It
reports observed levels/ranges, not voxel-weighted statistics or inferred bins.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import json
import os
from pathlib import Path
import re
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from fakect_tissues import AtlasClassifier, build_catalog, coarse_labels, read_organ_table


def read_json(path):
    return json.loads(Path(path).read_text())


def digest(path):
    hasher = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            hasher.update(chunk)
    return hasher.hexdigest()


def write_json(path, value):
    Path(path).write_text(json.dumps(value, indent=2, allow_nan=False) + "\n")


def new_output(path):
    path = Path(path)
    if path.exists() and any(path.iterdir()):
        raise ValueError(f"Output directory is not empty: {path}; use a new versioned location")
    path.mkdir(parents=True, exist_ok=True)
    return path


def merge_dictionaries(paths):
    names, sources = {}, defaultdict(list)
    for path in paths:
        for original_id, name in read_organ_table(path).items():
            if original_id in names and names[original_id] != name:
                raise ValueError(f"Conflicting dictionary names for {original_id}: {names[original_id]!r}, {name!r}")
            names[original_id] = name
            sources[original_id].append(str(path))
    return names, sources


def scan_surfaces(case_dir, frame):
    """Stream six .raw groups without loading or interpreting triangle geometry."""
    case_dir = Path(case_dir)
    files = []
    for group in ("body", "bones", "heart", "muscle", "organs", "vessels"):
        path = case_dir / f"{case_dir.name}_{frame}_{group}.raw"
        if not path.is_file():
            raise FileNotFoundError(path)
        surfaces = []
        hasher = hashlib.sha256()
        with path.open("rb") as stream:
            for line_number, raw in enumerate(stream, 1):
                hasher.update(raw)
                text = raw.strip()
                if not text:
                    continue
                # Triangle lines begin with a number. Check exceptional NaN/Inf
                # tokens too rather than inventing a surface named "nan".
                try:
                    float(text.split(None, 1)[0])
                    continue
                except ValueError:
                    pass
                surfaces.append({"name": text.decode("ascii"), "line": line_number,
                                 "surface_index_in_file": len(surfaces)})
        files.append({"path": str(path), "group": group, "bytes": path.stat().st_size,
                      "sha256": hasher.hexdigest(), "complete": True, "surfaces": surfaces})
    return {"schema": "fakect.surface-headers.v1", "case_id": case_dir.name,
            "frame": frame, "files": files,
            "index_semantics": "local occurrence only; never an act ID or DPI global block index"}


def catalog_command(args):
    paths = [args.organ_table, *args.supplemental_organ_table]
    names, dictionary_sources = merge_dictionaries(paths)
    classifier = AtlasClassifier(args.atlas, args.hierarchy, args.policy)
    catalog = build_catalog(classifier, names)
    catalog["dictionaries"] = [{"path": str(p), "sha256": digest(p)} for p in paths]
    catalog["reversibility"] = "Original per-voxel signed IDs remain authoritative; categories alone cannot recover them."
    by_id = {}
    for record in catalog["records"]:
        record["dictionary_sources"] = dictionary_sources.get(record["original_id"], [])
        by_id[record["original_id"]] = record
    surfaces = []
    inventory = None
    if args.surface_inventory:
        inventory = read_json(args.surface_inventory)
    elif args.raw_case_dir:
        inventory = scan_surfaces(args.raw_case_dir, args.frame)
    name_ids = defaultdict(list)
    for original_id, name in names.items():
        name_ids[name].append(original_id)
    if inventory:
        for file in inventory["files"]:
            if file.get("complete") is not True:
                raise ValueError(f"Surface inventory is incomplete: {file['path']}")
            for surface in file["surfaces"]:
                name = surface["name"]
                ids = sorted(name_ids.get(name, []))
                # Raw per-file ordinals are not DPI's global block indices.
                classification = classifier.classify(name, source_file=file["path"])
                surfaces.append({"source_file": file["path"], "group": file["group"],
                                 "local_surface_index": surface["surface_index_in_file"],
                                 "line": surface["line"], "original_name": name,
                                 "candidate_original_ids": ids,
                                 "id_join_status": "unique_name_match" if len(ids) == 1 else "ambiguous_name" if ids else "unmapped_name",
                                 "classification": classification})
    catalog["surface_sources"] = [{k: v for k, v in f.items() if k != "surfaces"}
                                  for f in (inventory or {}).get("files", [])]
    out = new_output(args.out)
    write_json(out / "label-catalog.json", catalog)
    write_json(out / "surface-catalog.json", {"schema_version": "fakect.surface-tissue-view.v1", "records": surfaces})
    with (out / "label-membership.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["original_id", "original_name", "tissue_id", "tissue_name", "status", "hierarchy_path"])
        for record in catalog["records"]:
            c = record["classification"]
            writer.writerow([record["original_id"], record["original_name"], c["tissue_id"], c["tissue_name"], c["status"], c.get("hierarchy_path", "")])
    summary = {"schema_version": "fakect.tissue-catalog-summary.v1", "catalog_sha256": digest(out / "label-catalog.json"),
               "original_ids": len(catalog["records"]), "categories": catalog["categories"],
               "labels_by_category": dict(Counter(r["classification"]["tissue_name"] for r in catalog["records"])),
               "label_status_counts": dict(Counter(r["classification"]["status"] for r in catalog["records"])),
               "surface_occurrences": len(surfaces), "surface_names": len({s["original_name"] for s in surfaces}),
               "surface_join_counts": dict(Counter(s["id_join_status"] for s in surfaces)),
               "warnings": catalog.get("warnings", []), "dictionaries": catalog["dictionaries"]}
    write_json(out / "summary.json", summary)
    print(json.dumps({k: summary[k] for k in ("original_ids", "labels_by_category", "surface_occurrences", "surface_join_counts")}, indent=2))


def material_table(log_path):
    text = Path(log_path).read_text()
    start = text.find("Linear Attenuation Coefficients (1/cm):")
    end = text.find("Linear Attenuation Coefficients (1/pixel):", start)
    if start < 0 or end < 0:
        raise ValueError(f"Missing explicit 1/cm material table: {log_path}")
    records = []
    for line in text[start:end].splitlines():
        match = re.fullmatch(r"\s*(.+?)\s*=\s*([+-]?[\d.]+)\s*", line)
        if match:
            records.append({"name": match[1].strip(), "attenuation_cm_inverse": float(match[2])})
    return records


def profile_command(args):
    catalog, audit = read_json(args.catalog), read_json(args.audit)
    unknown_category = next(c for c in catalog["categories"] if c["name"] == "unknown")
    if args.cases and set(args.cases) - {c["case_id"] for c in audit["cases"]}:
        raise ValueError("Requested case is absent from the paired sample audit")
    by_id = {r["original_id"]: r for r in catalog["records"]}
    records, materials, unknown = [], {}, set()
    group_levels = defaultdict(set)
    observed_ids = set()
    for case in audit["cases"]:
        if args.cases and case["case_id"] not in args.cases:
            continue
        expected_hash = audit.get("source_metadata_sha256", {}).get(case["log_path"])
        if expected_hash and expected_hash != digest(case["log_path"]):
            raise ValueError(f"Source log changed since sampling: {case['log_path']}")
        table = material_table(case["log_path"])
        materials[case["case_id"]] = {"log_path": case["log_path"], "sha256": digest(case["log_path"]), "table": table}
        levels = defaultdict(set)
        for row in case["scalar_to_organ_ids_sample"]:
            for original_id in row["organ_ids"]:
                levels[original_id].add(row["attenuation_cm_inverse"])
        for original_id, values in sorted(levels.items()):
            observed_ids.add(original_id)
            record = by_id.get(original_id)
            c = record["classification"] if record else {"tissue_id": unknown_category["id"], "tissue_name": "unknown", "status": "missing_dictionary_id"}
            if record is None:
                unknown.add(original_id)
            candidates = [{"attenuation_cm_inverse": value,
                           "material_candidates": [m["name"] for m in table if abs(m["attenuation_cm_inverse"] - value) <= 0.0000501]}
                          for value in sorted(values)]
            records.append({"case_id": case["case_id"], "original_id": original_id,
                            "original_name": record["original_name"] if record else None,
                            "tissue_id": c["tissue_id"], "tissue_name": c["tissue_name"], "mapping_status": c["status"],
                            "minimum_cm_inverse": min(values), "maximum_cm_inverse": max(values), "observed_levels": candidates})
            group_levels[c["tissue_name"]].update(values)
    if not records:
        raise ValueError("No paired samples selected")
    overlaps = []
    keys = sorted(group_levels)
    for index, left in enumerate(keys):
        for right in keys[index + 1:]:
            shared = sorted(v for v in group_levels[left] if any(abs(v - other) < 1e-6 for other in group_levels[right]))
            if shared:
                overlaps.append({"left": left, "right": right, "shared_levels_cm_inverse": shared})
    out = new_output(args.out)
    report = {"schema_version": "fakect.attenuation-profiles.v1", "catalog_sha256": digest(args.catalog),
              "audit_sha256": digest(args.audit), "sample_method": audit["method"],
              "scope": "Existing paired sample level sets only; no voxel counts, quantiles, full-volume extrema or HU inference.",
              "material_match_semantics": "Candidates within half a 4-decimal log rounding unit; not unique tissue identities.",
              "anatomy_and_material_are_separate": True, "range_is_not_classifier": True,
              "unknown_original_ids": sorted(unknown), "observed_original_ids": len(observed_ids),
              "unresolved_group_original_ids": sorted({r["original_id"] for r in records if r["tissue_id"] == unknown_category["id"]}),
              "material_candidates_by_group": {
                  group: sorted({name for r in records if r["tissue_name"] == group
                                 for level in r["observed_levels"] for name in level["material_candidates"]})
                  for group in keys},
              "materials": materials, "records": records, "shared_attenuation_levels": overlaps,
              "group_levels": {k: sorted(v) for k, v in group_levels.items()}}
    write_json(out / "attenuation-profiles.json", report)
    with (out / "attenuation-profiles.csv").open("w", newline="") as stream:
        writer = csv.writer(stream)
        writer.writerow(["case_id", "original_id", "original_name", "tissue_name", "mapping_status", "minimum_1_cm", "maximum_1_cm", "material_candidates"])
        for r in records:
            writer.writerow([r[k] for k in ("case_id", "original_id", "original_name", "tissue_name", "mapping_status", "minimum_cm_inverse", "maximum_cm_inverse")] +
                            [";".join(sorted({m for level in r["observed_levels"] for m in level["material_candidates"]}))])
    if not args.no_plot:
        os.environ.setdefault("MPLCONFIGDIR", "/tmp/fakect-mpl-cache")
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        fig, ax = plt.subplots(figsize=(11, 6), constrained_layout=True)
        for i, key in enumerate(keys):
            vals = sorted(group_levels[key])
            ax.scatter(vals, [i] * len(vals), s=14)
        ax.set(yticks=range(len(keys)), yticklabels=keys, xlabel="Observed attenuation (1/cm)",
               title="Atlas-derived anatomical groups: observed XCAT attenuation levels\nOverlapping levels cannot identify anatomy; original organ IDs are retained")
        ax.grid(axis="x", alpha=.25)
        fig.savefig(out / "attenuation-levels.png", dpi=160)
        plt.close(fig)
    print(json.dumps({"observed_original_ids": len(observed_ids), "unknown_original_ids": sorted(unknown),
                      "case_label_records": len(records), "groups_sharing_levels": len(overlaps)}, indent=2))


def reduce_command(args):
    import numpy as np
    catalog = read_json(args.catalog)
    original = np.load(args.labels, mmap_mode="r", allow_pickle=False)
    if not isinstance(original, np.ndarray):
        raise ValueError("--labels must be one .npy array")
    if args.max_voxels < 1 or original.size > args.max_voxels:
        raise ValueError("Label array exceeds --max-voxels; use a bounded crop or explicitly increase the limit")
    geometry = read_json(args.geometry)
    if not isinstance(geometry, dict) or not geometry:
        raise ValueError("Geometry must be a nonempty JSON object describing the source array")
    if original.ndim != 3 or geometry.get("array_order") != "zyx" or geometry.get("shape_zyx") != list(original.shape):
        raise ValueError("Geometry must declare array_order=zyx and shape_zyx matching the 3D label array")
    spacing = np.asarray(geometry.get("spacing_ijk_mm", []), dtype=float)
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError("Geometry spacing_ijk_mm must contain three finite positive values")
    if "crop_origin_ijk" in geometry:
        origin = np.asarray(geometry["crop_origin_ijk"])
        if origin.shape != (3,) or origin.dtype.kind not in "iu" or np.any(origin < 0):
            raise ValueError("crop_origin_ijk must contain three nonnegative integer indices")
        if "source_shape_zyx" in geometry:
            shape = np.asarray(geometry["source_shape_zyx"])
            if shape.shape != (3,) or shape.dtype.kind not in "iu" or np.any(shape <= 0) or np.any(origin[::-1] + original.shape > shape):
                raise ValueError("Crop bounds exceed source_shape_zyx")
    reduced = coarse_labels(original, catalog, strict=args.strict)
    if Path(args.out).exists():
        raise ValueError(f"Refusing to overwrite {args.out}")
    metadata = {"schema_version": "fakect.reversible-label-view.v1", "catalog_sha256": digest(args.catalog),
                "source_labels": str(args.labels), "source_sha256": digest(args.labels),
                "geometry": geometry, "catalog": catalog,
                "semantics": "Grouping view only; original_labels are authoritative. Edited categories need explicit new organ ownership."}
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    with Path(args.out).open("xb") as stream:
        np.savez_compressed(stream, original_labels=original, tissue_labels=reduced,
                            metadata_json=np.array(json.dumps(metadata, allow_nan=False)))
    print(f"Saved original IDs, tissue view and provenance to {args.out}")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("catalog", help="Join the DPI atlas to explicit organ dictionaries and raw surface names")
    p.add_argument("--atlas", type=Path, default=ROOT / "references/atlas/dpi/anatomy_atlas.csv")
    p.add_argument("--hierarchy", type=Path, default=ROOT / "references/atlas/dpi/xcat_anatomy_hierarchy.yml")
    p.add_argument("--policy", type=Path, default=ROOT / "configs/tissues/tissue-policy.v1.json")
    p.add_argument("--organ-table", type=Path, default=ROOT / "references/atlas/xcat/organ_ids.primary.txt")
    p.add_argument("--supplemental-organ-table", type=Path, action="append", default=[])
    group = p.add_mutually_exclusive_group()
    group.add_argument("--raw-case-dir", type=Path)
    group.add_argument("--surface-inventory", type=Path)
    p.add_argument("--frame", type=int, default=1)
    p.add_argument("--out", type=Path, required=True)
    p.set_defaults(func=catalog_command)
    p = sub.add_parser("profile", help="Measure tissue-group attenuation ranges from existing paired audit samples")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--audit", type=Path, required=True)
    p.add_argument("--cases", nargs="+")
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--no-plot", action="store_true")
    p.set_defaults(func=profile_command)
    p = sub.add_parser("reduce", help="Create a reversible label view from a bounded 3D .npy label array")
    p.add_argument("--catalog", type=Path, required=True)
    p.add_argument("--labels", type=Path, required=True)
    p.add_argument("--geometry", type=Path, required=True)
    p.add_argument("--out", type=Path, required=True)
    p.add_argument("--strict", action="store_true")
    p.add_argument("--max-voxels", type=int, default=16_777_216, help="Explicit bound for this crop-oriented NPZ export")
    p.set_defaults(func=reduce_command)
    args = parser.parse_args()
    try:
        args.func(args)
    except (ValueError, FileNotFoundError) as exc:
        parser.exit(2, f"error: {exc}\n")


if __name__ == "__main__":
    main()
