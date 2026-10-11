"""Strict named ROI recipes built from the existing bounded morphology contract.

Recipes intentionally have a separate schema: existing preview, edit and study
files keep their meanings. Loading this module or an INI never reads voxel data.
"""
from __future__ import annotations

import configparser
from pathlib import Path

from fakect_config import (REPOSITORY_ROOT, _EDIT_FIELDS, _OPTIONAL_EDIT_FIELDS, _FIELDS, _IDENTIFIER, _float,
                           _integer, _name, _parts, parse_preview_sections)


SCHEMA = "fakect.recipe/1"
RECIPE_FIELDS = {"steps", "overlap"}
ROI_FIELDS = {"shape", "center_ijk", "radius_mm", "coordinate_reviewed"}
EDIT_FIELDS = _EDIT_FIELDS["edit"] | {"roi", "iterations"}
EDIT_SELECTORS = {"point_range", "path_percent"}
EDIT_DIRECTION_FIELDS = {"direction", "angular_width_deg"}
CENTERLINE_DEFAULTS = {"smoothing_mm": 1., "sample_step_mm": 1., "min_curvature_per_mm": .002}
BASE_SECTIONS = set(_FIELDS) | {"reassignment", "recipe"}
STIFFNESS_SECTIONS = {"stiffness", "stiffness.labels"}
STIFFNESS_REQUIRED = {"default", "bone", "skin"}
STIFFNESS_TISSUES = {"bone", "skin", "soft_tissue", "cartilage", "muscle", "artery", "vein",
                    "lung", "adipose", "nervous_tissue", "fluid", "unknown"}
MAX_ITERATIONS = 20


def _parser(sections):
    """Build a strict case-sensitive parser from section dictionaries.

    Parameters
    ----------
    sections : dict of dict
        Section names and raw setting values suitable for ConfigParser.read_dict.

    Returns
    -------
    configparser.ConfigParser
        New parser with interpolation disabled and exact field capitalization.
    """
    result = configparser.ConfigParser(interpolation=None, strict=True)
    result.optionxform = str
    result.read_dict(sections)
    return result


def _fields(parser, section, expected, optional=frozenset()):
    """Validate required and optional recipe keys and reject multiline values.

    Parameters
    ----------
    parser : configparser.ConfigParser
        Already-read, case-sensitive configuration sections.
    section : str
        Existing section whose fields are checked.
    expected : set of str
        Required field names.
    optional : set of str, optional
        Additional permitted field names.

    Returns
    -------
    None
        Leaves the parser unchanged when validation succeeds.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    actual = set(parser[section])
    if actual - expected - optional or expected - actual:
        raise ValueError(f"[{section}] invalid settings: unknown={sorted(actual - expected - optional)}, "
                         f"missing={sorted(expected - actual)}")
    if any("\n" in value for value in parser[section].values()):
        raise ValueError(f"[{section}] settings must each be on one line")


def load_recipe_config(path, *, repo_root=None):
    """Read and normalize a named recipe INI with repository-relative paths.

    Parameters
    ----------
    path : str or pathlib.Path
        INI filename, resolved using the caller's working directory.
    repo_root : str or pathlib.Path, optional
        Root for paths inside the INI; defaults to the repository containing this module.

    Returns
    -------
    dict
        Validated common sections, named ROIs, ordered edits, optional centerline/stiffness
        settings and Path objects.

    Raises
    ------
    ValueError
        If syntax, schema or field validation fails.
    OSError
        If the INI cannot be read.
    """
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
    """Parse tissue and signed original-label resistance settings for stiffness mode.

    Parameters
    ----------
    parser : configparser.ConfigParser
        Already-read, case-sensitive configuration sections.
    mode : str
        Normalized reassignment mode.

    Returns
    -------
    dict or None
        Default factor, tissue factors and signed-ID overrides in [0, 1]; None for modes
        without stiffness sections.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
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
    """Parse an optional one-based control or physical arc-percentage interval.

    Parameters
    ----------
    values : mapping of str to str
        Raw edit fields.
    section : str
        Edit section name used in validation errors.
    roi : dict
        Normalized parent ROI defining shape and ordered controls.

    Returns
    -------
    dict
        One validated point_range/path_percent tuple, or an empty mapping when omitted.
        Interpolated geometry is resolved later.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
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


def _edit_direction(values, section, roi):
    """Parse curvature-relative direction and its permitted angular sector.

    Parameters
    ----------
    values : mapping of str to str
        Raw edit fields.
    section : str
        Edit section name used in validation errors.
    roi : dict
        Normalized parent ROI; inner/outer requires a tube with at least four controls.

    Returns
    -------
    dict
        Optional direction and full angular_width_deg, capped at 180 degrees; omitted all-
        direction defaults produce an empty mapping.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    direction = values.get("direction", "all").strip()
    if direction not in {"all", "inner", "outer"}:
        raise ValueError(f"{section}.direction must be all, inner, or outer")
    if direction == "all":
        if "angular_width_deg" in values:
            raise ValueError(f"{section}.angular_width_deg requires direction=inner or outer")
        return {"direction": direction} if "direction" in values else {}
    if roi["shape"] != "tube" or len(roi["center_ijk"]) < 4:
        raise ValueError(f"{section}.direction={direction} requires a referenced parent tube "
                         "with at least four center_ijk nodes for a cubic spline")
    width = _float(values.get("angular_width_deg", "180").strip(),
                   f"{section}.angular_width_deg", positive=True)
    if width > 180:
        raise ValueError(f"{section}.angular_width_deg must be <= 180")
    return {"direction": direction, "angular_width_deg": width}


def _centerline_settings(parser):
    """Normalize optional spline-frame sampling settings and their defaults.

    Parameters
    ----------
    parser : configparser.ConfigParser
        Already-read, case-sensitive configuration sections.

    Returns
    -------
    dict
        Smoothing residual scale and sampling step in millimetres, plus minimum curvature in
        inverse millimetres.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    values = parser["centerline"] if parser.has_section("centerline") else {}
    if parser.has_section("centerline"):
        _fields(parser, "centerline", set(), set(CENTERLINE_DEFAULTS))
    result = {}
    for name, default in CENTERLINE_DEFAULTS.items():
        value = _float(values.get(name, str(default)).strip(), f"centerline.{name}",
                       positive=name != "smoothing_mm")
        if name == "smoothing_mm" and value < 0:
            raise ValueError("centerline.smoothing_mm must be >= 0")
        result[name] = value
    return result


def parse_recipe_sections(parser, root=REPOSITORY_ROOT):
    """Normalize common sections, named ROI definitions and finite ordered edits.

    The main ROI defines the display crop and initial target selection, and may
    be referenced directly as ``roi=main``. It also bounds all edits by default;
    ``recipe.roi_role=selection`` separates seed selection from growth limits.
    Optional named ROIs carry no independent crop settings. Tube edits may
    select a point or physical path interval.
    Sections may appear in any order: only ``recipe.steps`` specifies execution order.

    Parameters
    ----------
    parser : configparser.ConfigParser
        Already-read, case-sensitive configuration sections.
    root : str or pathlib.Path
        Repository root used to resolve paths inside configuration sections.

    Returns
    -------
    dict
        Typed recipe configuration with paths, numeric/Boolean fields, explicit edit order,
        named geometry and optional centerline/stiffness settings. The input parser is
        unchanged.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    if parser.defaults():
        raise ValueError("[DEFAULT] settings are unsupported")
    if not parser.has_section("study") or parser["study"].get("schema_version", "").strip() != SCHEMA:
        raise ValueError(f"study.schema_version must be {SCHEMA}")
    actual_sections = set(parser.sections())
    unknown = sorted(section for section in actual_sections - BASE_SECTIONS - STIFFNESS_SECTIONS - {"centerline"}
                     if not section.startswith(("roi.", "edit.")))
    missing = sorted(BASE_SECTIONS - actual_sections)
    if unknown or missing:
        raise ValueError(f"Invalid recipe sections: unknown={unknown}, missing={missing}")
    _fields(parser, "recipe", RECIPE_FIELDS, {"roi_role"})
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
            _fields(parser, section, EDIT_FIELDS, EDIT_SELECTORS | EDIT_DIRECTION_FIELDS | _OPTIONAL_EDIT_FIELDS)
    steps = tuple(_name(value, "recipe.steps", _IDENTIFIER)
                  for value in _parts(parser["recipe"]["steps"].strip(), "recipe.steps"))
    names = {section[5:] for section in edit_sections}
    if len(set(steps)) != len(steps) or set(steps) != names:
        raise ValueError("recipe.steps must list every defined edit exactly once, in execution order")
    overlap = parser["recipe"]["overlap"].strip()
    if overlap not in {"sequential", "error"}:
        raise ValueError("recipe.overlap must be sequential or error")
    roi_role = parser["recipe"].get("roi_role")
    if roi_role is not None:
        roi_role = roi_role.strip()
        if roi_role not in {"boundary", "selection"}:
            raise ValueError("recipe.roi_role must be boundary (edits stay inside the ROI) "
                             "or selection (ROI selects initial targets)")
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
    if roi_role is not None:
        result["recipe"]["roi_role"] = roi_role
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
            sections["edit"] = {key: edit[key] for key in _EDIT_FIELDS["edit"] | _OPTIONAL_EDIT_FIELDS
                                if key in edit}
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
        direction = _edit_direction(values, section, base_roi)
        iterations = _integer(values["iterations"].strip(), f"{section}.iterations", 1, 10)
        total_iterations += iterations
        try:
            edit = parse_preview_sections(roi_parser(roi_name, values), root)["edit"]
        except ValueError as error:
            raise ValueError(f"[{section}]: {error}") from error
        result["edits"][name] = {"roi": roi_name, "iterations": iterations, **edit, **selector, **direction}
    if total_iterations > MAX_ITERATIONS:
        raise ValueError(f"A recipe permits at most {MAX_ITERATIONS} total iterations, including operation=none")
    if parser.has_section("centerline") or any(edit.get("direction") in {"inner", "outer"}
                                               for edit in result["edits"].values()):
        result["centerline"] = _centerline_settings(parser)
    return result
