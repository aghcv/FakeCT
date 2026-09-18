"""Strict named ROI recipes built from the existing bounded morphology contract.

Recipes intentionally have a separate schema: existing preview, edit and study
files keep their meanings. Loading this module or an INI never reads voxel data.
"""
from __future__ import annotations

import configparser
from pathlib import Path

from fakect_config import (REPOSITORY_ROOT, _EDIT_FIELDS, _FIELDS, _IDENTIFIER, _float,
                           _integer, _name, _parts, parse_preview_sections)


SCHEMA = "fakect.recipe/1"
RECIPE_FIELDS = {"steps", "overlap"}
ROI_FIELDS = {"shape", "center_ijk", "radius_mm", "coordinate_reviewed"}
EDIT_FIELDS = _EDIT_FIELDS["edit"] | {"roi", "iterations"}
EDIT_SELECTORS = {"point_range", "path_percent"}
BASE_SECTIONS = set(_FIELDS) | {"reassignment", "recipe"}
STIFFNESS_SECTIONS = {"stiffness", "stiffness.labels"}
STIFFNESS_REQUIRED = {"default", "bone", "skin"}
STIFFNESS_TISSUES = {"bone", "skin", "soft_tissue", "cartilage", "muscle", "artery", "vein",
                    "lung", "adipose", "nervous_tissue", "fluid", "unknown"}
MAX_ITERATIONS = 20


def _parser(sections):
    result = configparser.ConfigParser(interpolation=None, strict=True)
    result.optionxform = str
    result.read_dict(sections)
    return result


def _fields(parser, section, expected, optional=frozenset()):
    actual = set(parser[section])
    if actual - expected - optional or expected - actual:
        raise ValueError(f"[{section}] invalid settings: unknown={sorted(actual - expected - optional)}, "
                         f"missing={sorted(expected - actual)}")
    if any("\n" in value for value in parser[section].values()):
        raise ValueError(f"[{section}] settings must each be on one line")


def load_recipe_config(path, *, repo_root=None):
    """Read a recipe INI, retaining repository-relative paths and exact ordering."""
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",),
                                       strict=True, empty_lines_in_values=False)
    parser.optionxform = str
    try:
        with Path(path).expanduser().open(encoding="utf-8-sig") as stream:
            parser.read_file(stream)
    except configparser.Error as error:
        raise ValueError(f"Invalid recipe input: {error}") from error
    return parse_recipe_sections(parser, repo_root or REPOSITORY_ROOT)


def _stiffness(parser, mode):
    present = set(parser.sections()) & STIFFNESS_SECTIONS
    if mode != "stiffness":
        if present:
            raise ValueError("[stiffness] and [stiffness.labels] require reassignment.mode=stiffness")
        return None
    if "stiffness" not in present:
        raise ValueError("Missing [stiffness] section for reassignment.mode=stiffness")
    fields = set(parser["stiffness"])
    unknown, missing = fields - (STIFFNESS_TISSUES | {"default"}), STIFFNESS_REQUIRED - fields
    if unknown or missing:
        raise ValueError(f"[stiffness] invalid settings: unknown={sorted(unknown)}, missing={sorted(missing)}")
    _fields(parser, "stiffness", fields)
    values = {key: _float(value.strip(), f"stiffness.{key}", opacity=True)
              for key, value in parser["stiffness"].items()}
    labels = {}
    if "stiffness.labels" in present:
        _fields(parser, "stiffness.labels", set(parser["stiffness.labels"]))
        for raw_key, value in parser["stiffness.labels"].items():
            key = _integer(raw_key, f"stiffness.labels key {raw_key!r}", -(2**31), 2**31 - 1)
            if key in labels:
                raise ValueError(f"[stiffness.labels] contains duplicate original ID {key}")
            labels[key] = _float(value.strip(), f"stiffness.labels.{raw_key}", opacity=True)
    return {"default": values.pop("default"), "tissues": values, "labels": labels}


def _edit_selector(values, section, roi):
    """Validate optional tube intervals; physical path interpolation happens later."""
    present = set(values) & EDIT_SELECTORS
    if len(present) > 1:
        raise ValueError(f"[{section}] point_range and path_percent are mutually exclusive")
    if not present:
        return {}
    key = next(iter(present))
    label = f"{section}.{key}"
    if roi["shape"] != "tube":
        raise ValueError(f"{label} requires the referenced ROI to have shape=tube")
    parts = _parts(values[key].strip(), label)
    if len(parts) != 2:
        raise ValueError(f"{label} must contain exactly two values: start,end")
    if key == "point_range":
        node_count = len(roi["center_ijk"])
        interval = tuple(_integer(part, label, minimum=1, maximum=node_count) for part in parts)
        if interval[0] >= interval[1]:
            raise ValueError(f"{label} requires 1 <= start < end <= {node_count}; indices are 1-based and inclusive")
    else:
        interval = tuple(_float(part, label) for part in parts)
        if not 0 <= interval[0] < interval[1] <= 100:
            raise ValueError(f"{label} requires 0 <= start < end <= 100 percent of physical path length")
    return {key: interval}


def parse_recipe_sections(parser, root=REPOSITORY_ROOT):
    """Normalize common sections, named ROI definitions and finite ordered steps.

    The main ROI remains the display crop and outer editing boundary, and may
    be referenced directly as ``roi=main``. Optional named ROIs carry no
    independent crop settings; their masks are intersected with that boundary
    by the recipe engine. Tube edits may select a point or physical path interval.
    Sections may appear in any order: only ``recipe.steps`` specifies execution order.
    """
    if parser.defaults():
        raise ValueError("[DEFAULT] settings are unsupported")
    if not parser.has_section("study") or parser["study"].get("schema_version", "").strip() != SCHEMA:
        raise ValueError(f"study.schema_version must be {SCHEMA}")
    actual_sections = set(parser.sections())
    unknown = sorted(section for section in actual_sections - BASE_SECTIONS - STIFFNESS_SECTIONS
                     if not section.startswith(("roi.", "edit.")))
    missing = sorted(BASE_SECTIONS - actual_sections)
    if unknown or missing:
        raise ValueError(f"Invalid recipe sections: unknown={unknown}, missing={missing}")
    _fields(parser, "recipe", RECIPE_FIELDS)
    roi_sections = [section for section in parser.sections() if section.startswith("roi.")]
    edit_sections = [section for section in parser.sections() if section.startswith("edit.")]
    if not edit_sections:
        raise ValueError("A recipe requires at least one [edit.NAME]")
    if "roi.main" in roi_sections:
        raise ValueError("[roi.main] is reserved; use roi=main in an edit to reference the top-level [roi]")
    for section in roi_sections + edit_sections:
        _name(section.split(".", 1)[1], section, _IDENTIFIER)
        if section.startswith("roi."):
            _fields(parser, section, ROI_FIELDS)
        else:
            _fields(parser, section, EDIT_FIELDS, EDIT_SELECTORS)
    steps = tuple(_name(value, "recipe.steps", _IDENTIFIER)
                  for value in _parts(parser["recipe"]["steps"].strip(), "recipe.steps"))
    names = {section[5:] for section in edit_sections}
    if len(set(steps)) != len(steps) or set(steps) != names:
        raise ValueError("recipe.steps must list every defined edit exactly once, in execution order")
    overlap = parser["recipe"]["overlap"].strip()
    if overlap not in {"sequential", "error"}:
        raise ValueError("recipe.overlap must be sequential or error")
    reassignment_fields = _EDIT_FIELDS["reassignment"]
    if "mode" in parser["reassignment"]:
        reassignment_fields = reassignment_fields | {"mode"}
    _fields(parser, "reassignment", reassignment_fields)
    reassignment_mode = parser["reassignment"].get("mode", "allowlist").strip()
    if reassignment_mode not in {"allowlist", "permissive_except_bone_skin", "stiffness"}:
        raise ValueError("reassignment.mode must be allowlist, permissive_except_bone_skin, or stiffness")
    if reassignment_mode != "allowlist" and parser["reassignment"]["allowed_tissues"].strip():
        raise ValueError(f"reassignment.allowed_tissues must be blank for mode={reassignment_mode}")
    stiffness = _stiffness(parser, reassignment_mode)

    # Validate common sections using the established schema, without mutating
    # the caller's parser or allowing recipe sections into older contracts.
    common = {section: dict(parser[section]) for section in BASE_SECTIONS - {"recipe"}}
    common["reassignment"].pop("mode", None)
    common["study"]["schema_version"] = "fakect.edit/1"
    common["edit"] = {"operation": "none", "distance_mm": "0", "profile": "uniform",
                      "profile_axis": "k", "shape_k": "1", "shape_window": "0,1"}
    result = parse_preview_sections(_parser(common), root)
    result.pop("edit")
    result["reassignment"]["mode"] = reassignment_mode
    if stiffness is not None:
        result["reassignment"]["stiffness"] = stiffness
    result["study"]["schema_version"] = SCHEMA
    result["recipe"] = {"steps": steps, "overlap": overlap}
    result["rois"], result["edits"] = {}, {}

    def roi_parser(roi_name, edit=None):
        sections = {section: dict(values) for section, values in common.items()}
        if roi_name != "main":
            sections["roi"] = dict(parser[f"roi.{roi_name}"])
            # This synthetic width only lets the common parser validate a named
            # shape. The engine always uses the main ROI's actual crop and mask.
            radii = _parts(sections["roi"]["radius_mm"].strip(), f"roi.{roi_name}.radius_mm")
            largest = max((_float(value, f"roi.{roi_name}.radius_mm", positive=True)
                           for value in radii), default=1.0)
            sections["roi"]["crop_half_width_mm"] = str(largest)
        if edit is not None:
            sections["edit"] = {key: edit[key] for key in _EDIT_FIELDS["edit"]}
        return _parser(sections)

    for section in roi_sections:
        name = section[4:]
        try:
            roi = parse_preview_sections(roi_parser(name), root)["roi"]
        except ValueError as error:
            raise ValueError(f"[{section}]: {error}") from error
        roi.pop("crop_half_width_mm")
        result["rois"][name] = roi
    total_iterations = 0
    for section in edit_sections:
        name, values = section[5:], parser[section]
        roi_name = _name(values["roi"].strip(), f"{section}.roi", _IDENTIFIER)
        if roi_name != "main" and roi_name not in result["rois"]:
            raise ValueError(f"{section}.roi references undefined ROI {roi_name!r}")
        base_roi = result["roi"] if roi_name == "main" else result["rois"][roi_name]
        selector = _edit_selector(values, section, base_roi)
        iterations = _integer(values["iterations"].strip(), f"{section}.iterations", 1, 10)
        total_iterations += iterations
        try:
            edit = parse_preview_sections(roi_parser(roi_name, values), root)["edit"]
        except ValueError as error:
            raise ValueError(f"[{section}]: {error}") from error
        result["edits"][name] = {"roi": roi_name, "iterations": iterations, **edit, **selector}
    if total_iterations > MAX_ITERATIONS:
        raise ValueError(f"A recipe permits at most {MAX_ITERATIONS} total iterations, including operation=none")
    return result
