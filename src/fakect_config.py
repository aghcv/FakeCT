"""Strict, human-editable inputs for the XCAT ROI preview and edit workbench.

Preview schemas remain preview-only; ``fakect.edit/1`` adds bounded morphology
and tissue reassignment. Paths in an INI file are resolved against this
repository's root, even when the command runs from another directory.
Anatomical names, source IDs and volume bounds are checked against the selected
catalog and source geometry by the preview renderer.
"""
from __future__ import annotations

import configparser
import math
from pathlib import Path
import re
from typing import Any, Dict, Optional, Union


REPOSITORY_ROOT = Path(__file__).resolve().parents[1]
SCHEMA_VERSION = "fakect.preview/2"
EDIT_SCHEMA_VERSION = "fakect.edit/1"
SUPPORTED_SCHEMAS = {"fakect.preview/1", SCHEMA_VERSION, EDIT_SCHEMA_VERSION}
_FIELDS = {
    "study": {"schema_version", "name"},
    "input": {"root", "case_id", "frame", "audit", "catalog"},
    "selection": {"tissue", "source_ids"},
    "roi": {"center_ijk", "radius_mm", "crop_half_width_mm", "coordinate_reviewed"},
    "preview": {"slice_ijk", "overlay_opacity", "volume_opacity", "context_tissues",
                "context_opacity", "volume_stride"},
    "output": {"directory"},
}
_EDIT_FIELDS = {
    "edit": {"operation", "distance_mm", "profile", "profile_axis", "shape_k", "shape_window"},
    "reassignment": {"allowed_tissues", "max_distance_mm", "unresolved"},
}
_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]*\Z")
_CASE_IDENTIFIER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_-]*\Z")
_TISSUE_NAME = re.compile(r"[a-z][a-z0-9_]*\Z")
_INTEGER = re.compile(r"[+-]?[0-9]+\Z")


def _integer(value: str, label: str, minimum: Optional[int] = None,
             maximum: Optional[int] = None) -> int:
    if not _INTEGER.fullmatch(value):
        raise ValueError(f"{label} must be an integer")
    result = int(value)
    if minimum is not None and result < minimum:
        raise ValueError(f"{label} must be >= {minimum}")
    if maximum is not None and result > maximum:
        raise ValueError(f"{label} must be <= {maximum}")
    return result


def _float(value: str, label: str, *, positive: bool = False,
           opacity: bool = False) -> float:
    try:
        result = float(value)
    except ValueError as error:
        raise ValueError(f"{label} must be a finite number") from error
    if not math.isfinite(result):
        raise ValueError(f"{label} must be a finite number")
    if positive and result <= 0:
        raise ValueError(f"{label} must be positive")
    if opacity and not 0 <= result <= 1:
        raise ValueError(f"{label} must be between 0 and 1")
    return result


def _parts(value: str, label: str):
    if not value:
        return ()
    parts = tuple(part.strip() for part in value.split(","))
    if any(not part for part in parts):
        raise ValueError(f"{label} has an empty item; remove doubled or trailing commas")
    return parts


def _indices(value: str, label: str):
    parts = _parts(value, label)
    if len(parts) != 3:
        raise ValueError(f"{label} must contain exactly three integers: i, j, k")
    return tuple(_integer(part, label, minimum=0) for part in parts)


def _coordinates(value: str, label: str):
    parts = _parts(value, label)
    if len(parts) != 3:
        raise ValueError(f"{label} must contain exactly three coordinates: i, j, k")
    coordinates = tuple(_float(part, label) for part in parts)
    if any(coordinate < 0 for coordinate in coordinates):
        raise ValueError(f"{label} coordinates must be nonnegative")
    return coordinates


def _name(value: str, label: str, pattern: re.Pattern) -> str:
    if not pattern.fullmatch(value):
        raise ValueError(f"{label} contains an invalid name: {value!r}")
    return value


def _path(value: str, label: str, root: Path) -> Path:
    if not value:
        raise ValueError(f"{label} must contain a path")
    if "\x00" in value or "\n" in value:
        raise ValueError(f"{label} must be a single path")
    if "$" in value:
        raise ValueError(f"{label}: environment variables are not expanded; use a path or ~")
    path = Path(value).expanduser()
    return (path if path.is_absolute() else root / path).resolve()


def load_preview_config(path: Union[str, Path], *,
                        repo_root: Optional[Union[str, Path]] = None) -> Dict[str, Dict[str, Any]]:
    """Load a preview or bounded-edit INI into typed sections or raise ``ValueError``.

    ``path`` itself is resolved by the caller's usual working-directory rules.
    Paths *inside* the file use ``repo_root`` (the repository containing this
    module by default). Path values become ``Path`` objects, comma-separated
    fields become tuples, and ``slice_ijk = roi`` becomes ``None``. Blank
    ``source_ids``, ``context_tissues`` and ``allowed_tissues`` become empty tuples. No files other
    than the INI are read here and nothing is written.
    """
    path = Path(path).expanduser()
    root = Path(repo_root).expanduser().resolve() if repo_root is not None else REPOSITORY_ROOT
    parser = configparser.ConfigParser(interpolation=None, inline_comment_prefixes=("#",),
                                       strict=True, empty_lines_in_values=False)
    parser.optionxform = str  # Misspelled capitalization is an error, not a silent alias.
    try:
        with path.open(encoding="utf-8-sig") as handle:
            parser.read_file(handle)
    except configparser.Error as error:
        raise ValueError(f"Invalid preview input {path}: {error}") from error
    return parse_preview_sections(parser, root)


def parse_preview_sections(parser, root=REPOSITORY_ROOT):
    """Validate already-read workbench sections; shared by the study INI loader."""
    root = Path(root).expanduser().resolve()
    if parser.defaults():
        raise ValueError("[DEFAULT] settings are unsupported; put each key in its named section")
    actual_sections = set(parser.sections())
    if "study" not in actual_sections:
        raise ValueError("Invalid workbench sections: missing=['study']")
    version = parser["study"].get("schema_version", "").strip()
    if version not in SUPPORTED_SCHEMAS:
        raise ValueError(f"study.schema_version must be one of {sorted(SUPPORTED_SCHEMAS)!r}, received {version!r}")
    fields = _FIELDS | (_EDIT_FIELDS if version == EDIT_SCHEMA_VERSION else {})
    if actual_sections != set(fields):
        unknown = sorted(actual_sections - set(fields))
        missing = sorted(set(fields) - actual_sections)
        raise ValueError(f"Invalid workbench sections: unknown={unknown}, missing={missing}")
    for section, base_fields in fields.items():
        expected = base_fields | ({"shape"} if section == "roi" and version != "fakect.preview/1" else set())
        actual = set(parser[section])
        if actual != expected:
            unknown, missing = sorted(actual - expected), sorted(expected - actual)
            raise ValueError(f"[{section}] invalid settings: unknown={unknown}, missing={missing}")
        for key in expected:
            if "\n" in parser[section][key]:
                raise ValueError(f"{section}.{key} must be written on one line")
    read = lambda section, key: parser[section][key].strip()
    source_ids = tuple(_integer(part, "selection.source_ids", minimum=-(2 ** 31), maximum=2 ** 31 - 1)
                       for part in _parts(read("selection", "source_ids"), "selection.source_ids"))
    if len(set(source_ids)) != len(source_ids):
        raise ValueError("selection.source_ids must not contain duplicate IDs")
    contexts = tuple(_name(part, "preview.context_tissues", _TISSUE_NAME)
                     for part in _parts(read("preview", "context_tissues"), "preview.context_tissues"))
    if len(set(contexts)) != len(contexts):
        raise ValueError("preview.context_tissues must not contain duplicate names")
    reviewed = read("roi", "coordinate_reviewed")
    if reviewed not in {"true", "false"}:
        raise ValueError("roi.coordinate_reviewed must be true or false")
    shape = "sphere" if version == "fakect.preview/1" else read("roi", "shape")
    if shape not in {"sphere", "tube"}:
        raise ValueError("roi.shape must be sphere or tube")
    if shape == "sphere":
        coordinate_parser = _indices if version == "fakect.preview/1" else _coordinates
        center = coordinate_parser(read("roi", "center_ijk"), "roi.center_ijk")
        radius = _float(read("roi", "radius_mm"), "roi.radius_mm", positive=True)
        largest_radius = radius
    else:
        center = tuple(_coordinates(node.strip(), "roi.center_ijk")
                       for node in read("roi", "center_ijk").split(";"))
        radius = tuple(_float(part, "roi.radius_mm", positive=True)
                       for part in _parts(read("roi", "radius_mm"), "roi.radius_mm"))
        if len(center) < 2:
            raise ValueError("A tube requires at least two center_ijk nodes separated by semicolons")
        if len(radius) != len(center):
            raise ValueError("A tube requires one radius_mm value per center_ijk node")
        if any(first == second for first, second in zip(center, center[1:])):
            raise ValueError("Consecutive tube center_ijk nodes must be distinct")
        largest_radius = max(radius)
    half_width = _float(read("roi", "crop_half_width_mm"), "roi.crop_half_width_mm", positive=True)
    if half_width < largest_radius:
        raise ValueError("roi.crop_half_width_mm must be >= the largest roi.radius_mm to contain the ROI")
    slice_value = read("preview", "slice_ijk")
    result = {
        "study": {"schema_version": version,
                  "name": _name(read("study", "name"), "study.name", _IDENTIFIER)},
        "input": {"root": _path(read("input", "root"), "input.root", root),
                  "case_id": _name(read("input", "case_id"), "input.case_id", _CASE_IDENTIFIER),
                  "frame": _integer(read("input", "frame"), "input.frame", minimum=1),
                  "audit": _path(read("input", "audit"), "input.audit", root),
                  "catalog": _path(read("input", "catalog"), "input.catalog", root)},
        "selection": {"tissue": _name(read("selection", "tissue"), "selection.tissue", _TISSUE_NAME),
                      "source_ids": source_ids},
        "roi": {"shape": shape, "center_ijk": center,
                "radius_mm": radius, "crop_half_width_mm": half_width,
                "coordinate_reviewed": reviewed == "true"},
        "preview": {"slice_ijk": None if slice_value == "roi" else _indices(slice_value, "preview.slice_ijk"),
                    "overlay_opacity": _float(read("preview", "overlay_opacity"),
                                               "preview.overlay_opacity", opacity=True),
                    "volume_opacity": _float(read("preview", "volume_opacity"),
                                              "preview.volume_opacity", opacity=True),
                    "context_tissues": contexts,
                    "context_opacity": _float(read("preview", "context_opacity"),
                                               "preview.context_opacity", opacity=True),
                    "volume_stride": _integer(read("preview", "volume_stride"),
                                               "preview.volume_stride", minimum=1)},
        "output": {"directory": _path(read("output", "directory"), "output.directory", root)},
    }
    if version == EDIT_SCHEMA_VERSION:
        operation = read("edit", "operation")
        if operation not in {"none", "erosion", "dilation"}:
            raise ValueError("edit.operation must be none, erosion, or dilation")
        distance = _float(read("edit", "distance_mm"), "edit.distance_mm")
        if distance < 0 or (operation != "none" and distance == 0):
            raise ValueError("edit.distance_mm must be nonnegative, and positive for erosion or dilation")
        profile = read("edit", "profile")
        if profile not in {"uniform", "gaussian"}:
            raise ValueError("edit.profile must be uniform or gaussian")
        axis = read("edit", "profile_axis")
        if axis not in {"tube", "i", "j", "k"}:
            raise ValueError("edit.profile_axis must be tube, i, j, or k")
        if profile == "gaussian" and axis == "tube" and shape != "tube":
            raise ValueError("Gaussian edit.profile_axis=tube requires roi.shape=tube")
        window = tuple(_float(part, "edit.shape_window", opacity=True)
                       for part in _parts(read("edit", "shape_window"), "edit.shape_window"))
        if len(window) != 2 or window[0] >= window[1]:
            raise ValueError("edit.shape_window must contain exactly two values with 0 <= start < end <= 1")
        allowed = tuple(_name(part, "reassignment.allowed_tissues", _TISSUE_NAME)
                        for part in _parts(read("reassignment", "allowed_tissues"),
                                           "reassignment.allowed_tissues"))
        if len(set(allowed)) != len(allowed):
            raise ValueError("reassignment.allowed_tissues must not contain duplicate names")
        unresolved = read("reassignment", "unresolved")
        if unresolved not in {"preserve", "error"}:
            raise ValueError("reassignment.unresolved must be preserve or error")
        result["edit"] = {"operation": operation, "distance_mm": distance,
                          "profile": profile, "profile_axis": axis,
                          "shape_k": _float(read("edit", "shape_k"), "edit.shape_k", positive=True),
                          "shape_window": window}
        result["reassignment"] = {"allowed_tissues": allowed,
                                  "max_distance_mm": _float(read("reassignment", "max_distance_mm"),
                                                             "reassignment.max_distance_mm", positive=True),
                                  "unresolved": unresolved}
    return result
