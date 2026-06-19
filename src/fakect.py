#!/usr/bin/env python3
# ---------------------------------------------------------------------
# fakect.py
# Produces in/on/out masks (winding via python-igl when available) and optionally launches a Dash viewer.
#
# Usage examples:
#   python src/fakect.py --in data/cube.stl --n 8 --out outputs/cube_masks.npz
#   python src/fakect.py --in data/carotid.stl --n 9 --margin 0.10 --out outputs/carotid_masks.npz
#
# Requirements & installation
#  Recommended (conda - easiest, python-igl optional):
#    conda create -n fakect python=3.10 -y
#    conda activate fakect
#    conda install -c conda-forge trimesh scipy scikit-image plotly dash -y
#    # Optional, if available on your platform:
#    conda install -c conda-forge python-igl -y
#
#  Pip-only (virtualenv): python-igl often requires conda; use pip for the pure-Python deps
#    python -m venv .venv
#    source .venv/bin/activate
#    pip install --upgrade pip
#    pip install trimesh scipy scikit-image plotly dash
#    # If your platform supports it, you can try:
#    pip install igl
#    # But python-igl is best installed from conda-forge on many systems.
#
#  Quick note: to run without opening the Dash viewer (faster / headless), pass --no-show.
# ---------------------------------------------------------------------

from __future__ import annotations

import base64
import json
import re
import sys
import time
import numpy as np
from pathlib import Path
from typing import Any, Tuple
import zlib
from scipy.ndimage import binary_dilation
from skimage import measure

try:
    import plotly.graph_objects as go
except Exception:
    go = None

try:
    import trimesh
except Exception:
    trimesh = None

try:
    import dash
    from dash import dcc, html, Output, Input, State, callback_context
    from dash.exceptions import PreventUpdate
except Exception:
    dash = None
    dcc = html = Output = Input = State = callback_context = None
    class PreventUpdate(Exception):
        pass

# libigl for fast winding number
try:
    import igl  # pip/conda: python-igl
except Exception as e:
    igl = None

# ---------------------------------------------------------------------
# Console helpers
# ---------------------------------------------------------------------
def info(msg): print(f"\033[94m[INFO]\033[0m {msg}")
def ok(msg):   print(f"\033[92m[DONE]\033[0m {msg}")
def warn(msg): print(f"\033[93m[WARN]\033[0m {msg}")
def err(msg):  print(f"\033[91m[ERROR]\033[0m {msg}")

# ---------------------------------------------------------------------
# Visualization defaults
# ---------------------------------------------------------------------
OPACITY_LEVELS = {
    "low": 0.01,
    "mid": 0.12,
    "high": 0.24
}

SLICE_BORDER_WIDTH = 6

COLOR_SCHEME = {
    "slice_x": "#3B82F6",
    "slice_y": "#22C55E",
    "slice_z": "#F59E0B",
    "inside": "#E11D48",
    "on": "#22C55E",
    "out": "#F59E0B",
    "mesh": "#E11D48",
    "roi_p1": "#60A5FA",
    "roi_p2": "#F472B6",
    "roi_line": "#E5E7EB",
    "roi_sphere": "#A855F7"
}

LABEL_COLOR_PALETTE = [
    "#E11D48", "#22C55E", "#F59E0B", "#38BDF8", "#A855F7", "#F472B6",
    "#84CC16", "#F97316", "#14B8A6", "#EAB308", "#6366F1", "#EC4899",
    "#10B981", "#EF4444", "#06B6D4", "#8B5CF6", "#F43F5E", "#65A30D",
]

DEFAULT_TRANSFER_COLOR_SCHEME = "gray"

TRANSFER_COLOR_SCHEMES = {
    "gray": [
        [0.0, "#020617"],
        [1.0, "#F8FAFC"],
    ],
    "rainbow": [
        [0.0, "#4C1D95"],
        [0.18, "#2563EB"],
        [0.36, "#06B6D4"],
        [0.54, "#22C55E"],
        [0.72, "#F59E0B"],
        [1.0, "#EF4444"],
    ],
    "heat": [
        [0.0, "#050505"],
        [0.25, "#7F1D1D"],
        [0.52, "#DC2626"],
        [0.78, "#F59E0B"],
        [1.0, "#F8FAFC"],
    ],
}

TRANSFER_COLOR_OPTIONS = [
    {"label": "Gray", "value": "gray"},
    {"label": "Rainbow", "value": "rainbow"},
    {"label": "Heat", "value": "heat"},
]

VTK_DTYPE_MAP = {
    "Int8": np.int8,
    "UInt8": np.uint8,
    "Int16": np.int16,
    "UInt16": np.uint16,
    "Int32": np.int32,
    "UInt32": np.uint32,
    "Int64": np.int64,
    "UInt64": np.uint64,
    "Float32": np.float32,
    "Float64": np.float64,
}

VTK_HEADER_DTYPE_MAP = {
    "UInt32": np.uint32,
    "UInt64": np.uint64,
}

ROI_DEFAULTS = {
    "p1_opacity": 0.9,
    "p2_opacity": 0.9,
    "line_opacity": 0.8,
    "sphere_opacity": 0.05
}

# ---------------------------------------------------------------------
# Grid + (optional) voxelization
# ---------------------------------------------------------------------
def next_pow2(n: int) -> int:
    return 1 << (n - 1).bit_length()

def _grid_extents_display(grid):
    """Return (extent_x, extent_y, extent_z) in mm from grid shape × spacing."""
    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    return (Nx * sx, Ny * sy, Nz * sz)

def _grid_bounds_xyz(grid, pad_fraction: float = 0.0):
    """Return physical cell bounds as ((xmin, xmax), (ymin, ymax), (zmin, zmax))."""
    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]
    bounds = [
        sorted((ox, ox + Nx * sx)),
        sorted((oy, oy + Ny * sy)),
        sorted((oz, oz + Nz * sz)),
    ]
    padded = []
    for low, high in bounds:
        span = high - low
        pad = max(span * float(pad_fraction), 1e-6)
        padded.append((low - pad, high + pad))
    return tuple(padded)

def _aspectratio_from_extents(extents_xyz):
    """Normalize extents so that the largest side = 1 (realistic proportions)."""
    mx = max(extents_xyz)
    if mx <= 0:
        return dict(x=1, y=1, z=1)
    x, y, z = (e / mx for e in extents_xyz)
    return dict(x=x, y=y, z=z)

def center_mesh(mesh: trimesh.Trimesh) -> trimesh.Trimesh:
    """Return a copy of the mesh centered at the origin."""
    m = mesh.copy()
    m.apply_translation(-m.centroid)
    return m

def make_cube_grid_from_mesh(
    mesh: trimesh.Trimesh,
    spacing: float | None = None,
    n: int | None = None,
    margin_frac: float = 0.10,
    min_margin_voxels: int = 3
) -> dict:
    """
    Build a cubic (or near-cubic) grid safely larger than the mesh AABB.
    Guarantees at least `min_margin_voxels` empty voxels between geometry
    and grid boundaries even when spacing is inferred from n.

    This preserves the physical aspect ratios for visualization.
    """
    bounds = mesh.bounds
    Lx, Ly, Lz = (bounds[1] - bounds[0]).astype(float)
    Lmax = float(max(Lx, Ly, Lz))

    # --- explicit voxel pitch (compute N as next pow2 of required voxels) ---
    if spacing is not None:
        margin_abs = max(Lmax * margin_frac, min_margin_voxels * spacing)
        raw = int(np.ceil((Lmax + 2 * margin_abs) / spacing))
        N = next_pow2(raw)
        s = float(spacing)

    else:
        if n is None:
            n = 7
        N = 2 ** int(n)

        # initial spacing estimate using fractional margin only
        s = (Lmax * (1.0 + 2.0 * margin_frac)) / N

        # ensure at least min_margin_voxels safety
        min_margin_mm = min_margin_voxels * s
        margin_abs = max(Lmax * margin_frac, min_margin_mm)

        # adjust spacing so total extent matches new margin
        s = (Lmax + 2 * margin_abs) / N

    extent = N * s
    origin = (-extent / 2.0, -extent / 2.0, -extent / 2.0)

    grid = {
        "shape": (N, N, N),
        "spacing": (s, s, s),
        "origin": origin,
        "extent_mm": (extent, extent, extent),
        "aabb_mm": (Lx, Ly, Lz),
        "margin_mm": margin_abs,
    }

    info(f"Grid spacing={s:.4f} mm | N={N} | extent={extent:.3f} mm | safe margin={margin_abs:.3f} mm")
    return grid

def _parse_numeric_tuple(value: str, dtype=float) -> tuple:
    return tuple(dtype(v) for v in value.replace(",", " ").split())

def _extent_to_dims_xyz(extent: tuple[int, ...], association: str) -> tuple[int, int, int]:
    x0, x1, y0, y1, z0, z1 = extent
    if association == "CellData":
        return max(0, x1 - x0), max(0, y1 - y0), max(0, z1 - z0)
    return max(0, x1 - x0 + 1), max(0, y1 - y0 + 1), max(0, z1 - z0 + 1)

def _vtk_endian(root: ET.Element) -> str:
    byte_order = root.attrib.get("byte_order", "LittleEndian")
    return "<" if byte_order != "BigEndian" else ">"

def _vtk_dtype(type_name: str, endian: str) -> np.dtype:
    if type_name not in VTK_DTYPE_MAP:
        raise ValueError(f"Unsupported VTI DataArray type: {type_name}")
    return np.dtype(VTK_DTYPE_MAP[type_name]).newbyteorder(endian)

def _vtk_header_dtype(header_type: str, endian: str) -> np.dtype:
    if header_type not in VTK_HEADER_DTYPE_MAP:
        raise ValueError(f"Unsupported VTI header_type: {header_type}")
    return np.dtype(VTK_HEADER_DTYPE_MAP[header_type]).newbyteorder(endian)

def _vti_xml_without_appended(text: str) -> str:
    if "<AppendedData" not in text:
        return text
    return text.split("<AppendedData", 1)[0] + "</VTKFile>"

def _vti_appended_payload_b64(text: str) -> str:
    match = re.search(r"<AppendedData[^>]*>\s*_(.*?)\s*</AppendedData>", text, re.S)
    if not match:
        raise ValueError("VTI file has no appended data payload")
    return "".join(match.group(1).split())

def _decode_vtk_base64_payload(payload: str) -> bytes:
    """
    VTK XML writers may concatenate independently padded base64 chunks.
    Python's base64 decoder stops at the first padding marker, so decode each
    padded chunk and concatenate the raw bytes.
    """
    payload = "".join(payload.split())
    if "=" not in payload:
        return base64.b64decode(payload)

    chunks: list[bytes] = []
    start = 0
    i = 0
    while i < len(payload):
        if payload[i] != "=":
            i += 1
            continue
        j = i
        while j < len(payload) and payload[j] == "=":
            j += 1
        chunk = payload[start:j]
        if chunk:
            chunks.append(base64.b64decode(chunk))
        start = j
        i = j
    if start < len(payload):
        chunks.append(base64.b64decode(payload[start:]))
    return b"".join(chunks)

def _vti_appended_bytes(text: str) -> bytes:
    return _decode_vtk_base64_payload(_vti_appended_payload_b64(text))

def _find_vti_data_array(
    piece: ET.Element,
    requested_name: str | None = None
) -> tuple[str, ET.Element]:
    candidates: list[tuple[str, ET.Element]] = []
    for association in ("CellData", "PointData"):
        parent = piece.find(association)
        if parent is None:
            continue
        preferred = parent.attrib.get("Scalars")
        arrays = list(parent.findall("DataArray"))
        if requested_name:
            for data_array in arrays:
                if data_array.attrib.get("Name") == requested_name:
                    return association, data_array
        elif preferred:
            for data_array in arrays:
                if data_array.attrib.get("Name") == preferred:
                    return association, data_array
        for data_array in arrays:
            candidates.append((association, data_array))

    if requested_name:
        raise ValueError(f"No VTI DataArray named {requested_name!r}")
    if not candidates:
        raise ValueError("No CellData or PointData arrays found in VTI file")
    return candidates[0]

def _iter_vtk_uncompressed_payload(
    raw: bytes,
    header_dtype: np.dtype,
    offset: int
):
    word = header_dtype.itemsize
    nbytes = int(np.frombuffer(raw, dtype=header_dtype, count=1, offset=offset)[0])
    start = offset + word
    yield raw[start:start + nbytes]

def _iter_vtk_zlib_payloads(
    raw: bytes,
    header_dtype: np.dtype,
    offset: int
):
    word = header_dtype.itemsize
    header = np.frombuffer(raw, dtype=header_dtype, count=3, offset=offset).astype(np.int64)
    n_blocks = int(header[0])
    if n_blocks <= 0:
        return
    sizes_offset = offset + 3 * word
    compressed_sizes = np.frombuffer(
        raw,
        dtype=header_dtype,
        count=n_blocks,
        offset=sizes_offset
    ).astype(np.int64)
    data_offset = sizes_offset + n_blocks * word
    cursor = data_offset
    for compressed_size in compressed_sizes:
        compressed_size = int(compressed_size)
        block = raw[cursor:cursor + compressed_size]
        cursor += compressed_size
        try:
            yield zlib.decompress(block)
        except zlib.error:
            # Some VTK writers emit raw DEFLATE streams without a zlib wrapper.
            yield zlib.decompress(block, -zlib.MAX_WBITS)

def _sample_flat_chunk(
    sampled_zyx: np.ndarray,
    values: np.ndarray,
    start_elem: int,
    dims_xyz: tuple[int, int, int],
    stride: int,
    sample_offset: int,
) -> None:
    if values.size == 0:
        return

    nx, ny, nz = dims_xyz
    total = nx * ny * nz
    idx = np.arange(start_elem, start_elem + values.size, dtype=np.int64)
    valid = idx < total
    if not np.any(valid):
        return

    idx = idx[valid]
    vals = values[valid]
    x = idx % nx
    y = (idx // nx) % ny
    z = idx // (nx * ny)
    selected = (
        (x >= sample_offset) & ((x - sample_offset) % stride == 0) &
        (y >= sample_offset) & ((y - sample_offset) % stride == 0) &
        (z >= sample_offset) & ((z - sample_offset) % stride == 0)
    )
    if not np.any(selected):
        return

    sx = ((x[selected] - sample_offset) // stride).astype(np.int64)
    sy = ((y[selected] - sample_offset) // stride).astype(np.int64)
    sz = ((z[selected] - sample_offset) // stride).astype(np.int64)
    sampled_zyx[sz, sy, sx] = vals[selected]

def _read_vti_sampled_array(
    vti_path: Path,
    array_name: str | None = None,
    max_dim: int = 160
) -> tuple[np.ndarray, dict, dict]:
    """
    Read a VTK XML ImageData array into a sampled (Z,Y,X) ndarray.

    Large VTI files can be much larger than browser-friendly voxel grids.  The
    reader samples every Nth voxel/cell directly from the compressed stream,
    so it does not need to materialize the full uncompressed VTI volume.
    """
    text = vti_path.read_text(errors="ignore")
    root = ET.fromstring(_vti_xml_without_appended(text))
    if root.attrib.get("type") != "ImageData":
        raise ValueError(f"Expected VTK ImageData, got {root.attrib.get('type')!r}")

    image = root.find("ImageData")
    if image is None:
        raise ValueError("VTI file is missing ImageData")
    piece = image.find("Piece")
    if piece is None:
        raise ValueError("VTI file is missing ImageData/Piece")

    association, data_array = _find_vti_data_array(piece, array_name)
    num_components = int(data_array.attrib.get("NumberOfComponents", "1"))
    if num_components != 1:
        raise ValueError("VTI label import expects scalar DataArrays with NumberOfComponents=1")
    endian = _vtk_endian(root)
    dtype = _vtk_dtype(data_array.attrib.get("type", ""), endian)
    header_dtype = _vtk_header_dtype(root.attrib.get("header_type", "UInt32"), endian)
    extent = _parse_numeric_tuple(piece.attrib.get("Extent", image.attrib.get("WholeExtent", "")), int)
    dims_xyz = _extent_to_dims_xyz(extent, association)
    if any(d <= 0 for d in dims_xyz):
        raise ValueError(f"Invalid VTI dimensions for {association}: {dims_xyz}")

    max_full_dim = max(dims_xyz)
    max_dim = int(max_dim or 0)
    stride = 1 if max_dim <= 0 else max(1, int(np.ceil(max_full_dim / max_dim)))
    sample_offset = 0 if stride == 1 else stride // 2
    sampled_dims_xyz = tuple(
        0 if dim <= sample_offset else ((dim - 1 - sample_offset) // stride) + 1
        for dim in dims_xyz
    )
    sampled_zyx = np.zeros(
        (sampled_dims_xyz[2], sampled_dims_xyz[1], sampled_dims_xyz[0]),
        dtype=dtype
    )

    fmt = data_array.attrib.get("format", "appended")
    start_elem = 0
    itemsize = dtype.itemsize
    carry = b""

    def consume_payload(payload_iter):
        nonlocal start_elem, carry
        for block in payload_iter:
            data = carry + block
            usable = (len(data) // itemsize) * itemsize
            if usable:
                values = np.frombuffer(data[:usable], dtype=dtype)
                _sample_flat_chunk(
                    sampled_zyx,
                    values,
                    start_elem,
                    dims_xyz,
                    stride,
                    sample_offset
                )
                start_elem += int(values.size)
            carry = data[usable:]
        if carry:
            raise ValueError("VTI payload ended with an incomplete scalar value")

    if fmt == "ascii":
        values = np.fromstring(data_array.text or "", sep=" ", dtype=dtype)
        _sample_flat_chunk(sampled_zyx, values, 0, dims_xyz, stride, sample_offset)
    elif fmt == "binary":
        raw = _decode_vtk_base64_payload(data_array.text or "")
        if root.attrib.get("compressor") == "vtkZLibDataCompressor":
            consume_payload(_iter_vtk_zlib_payloads(raw, header_dtype, 0))
        else:
            consume_payload(_iter_vtk_uncompressed_payload(raw, header_dtype, 0))
    elif fmt == "appended":
        offset = int(data_array.attrib.get("offset", "0"))
        payload_b64 = _vti_appended_payload_b64(text)
        decoded = _decode_vtk_base64_payload(payload_b64)
        compressor = root.attrib.get("compressor")
        attempt_errors: list[str] = []

        def _consume_attempt(raw_bytes: bytes, byte_offset: int) -> None:
            nonlocal start_elem, carry
            start_elem = 0
            carry = b""
            sampled_zyx.fill(0)
            if compressor == "vtkZLibDataCompressor":
                consume_payload(_iter_vtk_zlib_payloads(raw_bytes, header_dtype, byte_offset))
            else:
                consume_payload(_iter_vtk_uncompressed_payload(raw_bytes, header_dtype, byte_offset))

        # Most files use decoded-byte offsets; some use encoded-base64 offsets.
        # Try both to maximize compatibility with VTK writers.
        try:
            _consume_attempt(decoded, offset)
        except Exception as e:
            attempt_errors.append(f"decoded-offset failed: {e}")
            try:
                if offset < 0 or offset >= len(payload_b64):
                    raise ValueError("encoded offset is out of payload bounds")
                decoded_from_encoded_offset = _decode_vtk_base64_payload(payload_b64[offset:])
                _consume_attempt(decoded_from_encoded_offset, 0)
            except Exception as e2:
                attempt_errors.append(f"encoded-offset failed: {e2}")
                raise ValueError("; ".join(attempt_errors)) from e2
    else:
        raise ValueError(f"Unsupported VTI DataArray format: {fmt!r}")

    origin_xyz = _parse_numeric_tuple(image.attrib.get("Origin", "0 0 0"), float)
    spacing_xyz = _parse_numeric_tuple(image.attrib.get("Spacing", "1 1 1"), float)
    sx, sy, sz = spacing_xyz
    ox, oy, oz = origin_xyz
    coarse_spacing_xyz = (sx * stride, sy * stride, sz * stride)
    coarse_origin_xyz = (
        ox + (sample_offset + 0.5) * sx - 0.5 * coarse_spacing_xyz[0],
        oy + (sample_offset + 0.5) * sy - 0.5 * coarse_spacing_xyz[1],
        oz + (sample_offset + 0.5) * sz - 0.5 * coarse_spacing_xyz[2],
    )

    nz, ny, nx = sampled_zyx.shape
    grid = {
        "shape": sampled_zyx.shape,
        "spacing": (coarse_spacing_xyz[2], coarse_spacing_xyz[1], coarse_spacing_xyz[0]),
        "origin": coarse_origin_xyz,
        "extent_mm": (nx * coarse_spacing_xyz[0], ny * coarse_spacing_xyz[1], nz * coarse_spacing_xyz[2]),
        "aabb_mm": (dims_xyz[0] * sx, dims_xyz[1] * sy, dims_xyz[2] * sz),
        "margin_mm": 0.0,
        "sample_stride": stride,
        "sample_offset": sample_offset,
    }
    metadata = {
        "array_name": data_array.attrib.get("Name", "values"),
        "association": association,
        "dtype": str(np.dtype(dtype).newbyteorder("=")),
        "full_dims_xyz": dims_xyz,
        "sampled_dims_xyz": sampled_dims_xyz,
        "sample_stride": stride,
        "sample_offset": sample_offset,
    }
    info(
        f"Loaded VTI array '{metadata['array_name']}' ({association}) "
        f"full={dims_xyz}, sampled={sampled_zyx.shape}, stride={stride}"
    )
    return sampled_zyx, grid, metadata

def _safe_layer_id(prefix: str, value: Any) -> str:
    raw = f"{prefix}_{value}".lower()
    raw = re.sub(r"[^a-z0-9_]+", "_", raw)
    return raw.strip("_") or "label"

def _label_display_value(value: Any) -> str:
    if isinstance(value, np.generic):
        value = value.item()
    if isinstance(value, float):
        return f"{value:g}"
    return str(value)

def _make_mask_layer(
    layer_id: str,
    name: str,
    mask: np.ndarray,
    color: str,
    opacity: float = OPACITY_LEVELS["mid"],
    editable: bool = True,
    label_value: Any = None,
    exclusive_group: str | None = None,
) -> dict:
    return {
        "id": layer_id,
        "name": name,
        "mask": (mask > 0).astype(np.uint8),
        "color": color,
        "opacity": opacity,
        "editable": bool(editable),
        "label_value": label_value,
        "exclusive_group": exclusive_group,
    }

def _is_discrete_label_array(values: np.ndarray, unique_values: np.ndarray, max_labels: int) -> bool:
    if unique_values.size > max_labels + 1:
        return False
    finite = unique_values[np.isfinite(unique_values)]
    if finite.size != unique_values.size:
        return False
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        return True
    return bool(np.allclose(finite, np.round(finite), atol=1e-6))

def _finite_range(values: np.ndarray) -> tuple[float, float]:
    finite = values[np.isfinite(values)]
    if finite.size == 0:
        return 0.0, 1.0
    vmin = float(np.min(finite))
    vmax = float(np.max(finite))
    if np.isclose(vmin, vmax):
        vmax = vmin + 1.0
    return vmin, vmax

def _non_background_unique_values(unique_values: np.ndarray, background: float, eps: float, dtype: np.dtype) -> list:
    if np.issubdtype(dtype, np.floating):
        return [v for v in unique_values if not np.isclose(float(v), float(background), atol=eps)]
    bg = np.asarray(background, dtype=dtype)
    return [v for v in unique_values if v != bg]

def _attach_scalar_volume(layer: dict, data: np.ndarray, scalar_range: tuple[float, float]) -> dict:
    layer["volume_kind"] = "scalar"
    layer["scalar_data"] = data.astype(np.float32, copy=False)
    layer["scalar_range"] = tuple(float(v) for v in scalar_range)
    layer["opacity"] = 1.0
    return layer

def _read_vti_header_info(vti_path: Path) -> dict:
    """
    Read just the XML header of a VTI file.
    Returns a dict with keys: path, origin, spacing, extent, association, array_name, dtype_str.
    """
    text = vti_path.read_text(errors="ignore")
    root = ET.fromstring(_vti_xml_without_appended(text))
    if root.attrib.get("type") != "ImageData":
        raise ValueError(f"Expected VTK ImageData, got {root.attrib.get('type')!r}")
    image = root.find("ImageData")
    if image is None:
        raise ValueError(f"VTI file {vti_path.name} is missing ImageData")
    piece = image.find("Piece")
    origin_xyz = _parse_numeric_tuple(image.attrib.get("Origin", "0 0 0"), float)
    spacing_xyz = _parse_numeric_tuple(image.attrib.get("Spacing", "1 1 1"), float)
    whole_extent_str = image.attrib.get("WholeExtent", "")
    piece_extent_str = piece.attrib.get("Extent", whole_extent_str) if piece is not None else whole_extent_str
    extent = _parse_numeric_tuple(piece_extent_str, int)
    array_name = None
    association = "CellData"
    dtype_str = "Float32"
    if piece is not None:
        try:
            assoc, da = _find_vti_data_array(piece, None)
            array_name = da.attrib.get("Name", "values")
            association = assoc
            dtype_str = da.attrib.get("type", "Float32")
        except ValueError:
            pass
    return {
        "path": vti_path,
        "origin": origin_xyz,
        "spacing": spacing_xyz,
        "extent": extent,
        "association": association,
        "array_name": array_name or "values",
        "dtype_str": dtype_str,
    }


def _assemble_vti_tiles(
    vti_dir: Path,
    array_name: str | None = None,
    max_dim: int = 160,
) -> tuple[np.ndarray, dict, dict]:
    """
    Load all VTI tiles in a directory and assemble them into a single sampled volume.

    Tiles must share the same spacing and use their Origin attribute to locate themselves
    in a common integer coordinate system.  Each tile's data is read at full resolution
    then placed into the appropriate region of the global sampled output.

    Returns (data_zyx, grid, metadata) with the same format as _read_vti_sampled_array.
    """
    vti_paths = sorted(vti_dir.glob("*.vti"))
    if not vti_paths:
        raise ValueError(f"No VTI files found in {vti_dir}")
    info(f"Assembling {len(vti_paths)} VTI tile(s) from {vti_dir.name}/")

    # --- Phase 1: read headers to find global bounding box ---
    tile_infos = []
    for p in vti_paths:
        try:
            tile_infos.append(_read_vti_header_info(p))
        except Exception as e:
            warn(f"Skipping {p.name}: {e}")

    if not tile_infos:
        raise ValueError(f"No valid VTI files could be read from {vti_dir}")

    # Use first tile's spacing as reference (assume uniform)
    ref_spacing = np.array(tile_infos[0]["spacing"], dtype=float)
    sx_ref, sy_ref, sz_ref = ref_spacing

    # Compute global integer cell extents
    gx_min = gx_max = gy_min = gy_max = gz_min = gz_max = None
    for ti in tile_infos:
        x0, x1, y0, y1, z0, z1 = ti["extent"]
        assoc = ti["association"]
        ox, oy, oz = int(round(ti["origin"][0] / sx_ref)), int(round(ti["origin"][1] / sy_ref)), int(round(ti["origin"][2] / sz_ref))
        # For CellData: tile covers global cells [ox+x0, ox+x1)
        if assoc == "CellData":
            tx_min, tx_max = ox + x0, ox + x1
            ty_min, ty_max = oy + y0, oy + y1
            tz_min, tz_max = oz + z0, oz + z1
        else:
            tx_min, tx_max = ox + x0, ox + x1 + 1
            ty_min, ty_max = oy + y0, oy + y1 + 1
            tz_min, tz_max = oz + z0, oz + z1 + 1
        gx_min = tx_min if gx_min is None else min(gx_min, tx_min)
        gx_max = tx_max if gx_max is None else max(gx_max, tx_max)
        gy_min = ty_min if gy_min is None else min(gy_min, ty_min)
        gy_max = ty_max if gy_max is None else max(gy_max, ty_max)
        gz_min = tz_min if gz_min is None else min(gz_min, tz_min)
        gz_max = tz_max if gz_max is None else max(gz_max, tz_max)
        ti["_gx_min"] = tx_min; ti["_gx_max"] = tx_max
        ti["_gy_min"] = ty_min; ti["_gy_max"] = ty_max
        ti["_gz_min"] = tz_min; ti["_gz_max"] = tz_max

    global_dims_xyz = (gx_max - gx_min, gy_max - gy_min, gz_max - gz_min)
    max_full_dim = max(global_dims_xyz)
    stride = max(1, int(np.ceil(max_full_dim / max_dim))) if max_dim > 0 else 1
    sample_offset = stride // 2 if stride > 1 else 0

    # Output dimensions (ZYX order)
    def sampled_len(full_len):
        return max(1, (full_len - 1 - sample_offset) // stride + 1) if full_len > sample_offset else 0

    out_nx = sampled_len(global_dims_xyz[0])
    out_ny = sampled_len(global_dims_xyz[1])
    out_nz = sampled_len(global_dims_xyz[2])

    # Determine output dtype from first tile
    ref_dtype_str = tile_infos[0]["dtype_str"]
    endian = "<"
    out_dtype = VTK_DTYPE_MAP.get(ref_dtype_str, np.float32)
    out = np.zeros((out_nz, out_ny, out_nx), dtype=out_dtype)

    # Global sample positions in each axis
    # out_cell_k → global cell g = sample_offset + k * stride + g_min
    out_gx = sample_offset + np.arange(out_nx) * stride + gx_min  # global x index of each output cell
    out_gy = sample_offset + np.arange(out_ny) * stride + gy_min
    out_gz = sample_offset + np.arange(out_nz) * stride + gz_min

    # --- Phase 2: read each tile and fill output ---
    ref_array_name = array_name or tile_infos[0]["array_name"]
    for ti in tile_infos:
        tx_min, tx_max = ti["_gx_min"], ti["_gx_max"]
        ty_min, ty_max = ti["_gy_min"], ti["_gy_max"]
        tz_min, tz_max = ti["_gz_min"], ti["_gz_max"]

        # Which output cells fall within this tile's global range?
        mask_x = (out_gx >= tx_min) & (out_gx < tx_max)
        mask_y = (out_gy >= ty_min) & (out_gy < ty_max)
        mask_z = (out_gz >= tz_min) & (out_gz < tz_max)
        if not (np.any(mask_x) and np.any(mask_y) and np.any(mask_z)):
            continue

        out_ix = np.where(mask_x)[0]
        out_iy = np.where(mask_y)[0]
        out_iz = np.where(mask_z)[0]

        # Local tile indices for those global sample positions
        local_x = (out_gx[mask_x] - tx_min).astype(np.int64)
        local_y = (out_gy[mask_y] - ty_min).astype(np.int64)
        local_z = (out_gz[mask_z] - tz_min).astype(np.int64)

        # Read tile at full resolution (max_dim=0)
        try:
            tile_data, _, _ = _read_vti_sampled_array(ti["path"], array_name=ref_array_name, max_dim=0)
        except Exception as e:
            warn(f"Could not read {ti['path'].name}: {e}")
            continue
        # tile_data shape: (tz, ty, tx)
        tz_size, ty_size, tx_size = tile_data.shape
        local_x = np.clip(local_x, 0, tx_size - 1)
        local_y = np.clip(local_y, 0, ty_size - 1)
        local_z = np.clip(local_z, 0, tz_size - 1)

        OZ, OY, OX = np.meshgrid(out_iz, out_iy, out_ix, indexing="ij")
        LZ, LY, LX = np.meshgrid(local_z, local_y, local_x, indexing="ij")
        out[OZ, OY, OX] = tile_data[LZ, LY, LX].astype(out_dtype)

    # Build grid metadata
    out_spacing_xyz = (sx_ref * stride, sy_ref * stride, sz_ref * stride)
    out_origin_xyz = (
        gx_min * sx_ref + (sample_offset + 0.5) * sx_ref - 0.5 * out_spacing_xyz[0],
        gy_min * sy_ref + (sample_offset + 0.5) * sy_ref - 0.5 * out_spacing_xyz[1],
        gz_min * sz_ref + (sample_offset + 0.5) * sz_ref - 0.5 * out_spacing_xyz[2],
    )
    grid = {
        "shape": out.shape,
        "spacing": (out_spacing_xyz[2], out_spacing_xyz[1], out_spacing_xyz[0]),
        "origin": out_origin_xyz,
        "extent_mm": (out_nx * out_spacing_xyz[0], out_ny * out_spacing_xyz[1], out_nz * out_spacing_xyz[2]),
        "aabb_mm": (global_dims_xyz[0] * sx_ref, global_dims_xyz[1] * sy_ref, global_dims_xyz[2] * sz_ref),
        "margin_mm": 0.0,
        "sample_stride": stride,
        "sample_offset": sample_offset,
    }
    metadata = {
        "array_name": ref_array_name,
        "association": tile_infos[0]["association"],
        "dtype": str(np.dtype(out_dtype).newbyteorder("=")),
        "full_dims_xyz": global_dims_xyz,
        "sampled_dims_xyz": (out_nx, out_ny, out_nz),
        "sample_stride": stride,
        "sample_offset": sample_offset,
        "tile_count": len(tile_infos),
        "source": "directory",
    }
    info(
        f"Assembly complete: {len(tile_infos)} tiles | global={global_dims_xyz} cells | "
        f"sampled={out.shape} (stride={stride})"
    )
    return out, grid, metadata


def load_vti_label_layers(
    vti_path: str,
    array_name: str | None = None,
    background: float = 0.0,
    background_eps: float = 0.0,
    max_dim: int = 160,
    max_labels: int = 64,
) -> tuple[list[dict], dict, dict]:
    data, grid, metadata = _read_vti_sampled_array(
        Path(vti_path).resolve(),
        array_name=array_name,
        max_dim=max_dim
    )
    unique_values = np.unique(data)
    array_label = metadata["array_name"]
    max_labels = max(1, int(max_labels))
    eps = max(0.0, float(background_eps))
    bg = float(background)
    scalar_range = _finite_range(data)

    if np.issubdtype(data.dtype, np.floating):
        background_mask = np.isclose(data, bg, atol=eps)
    else:
        background_mask = data == np.asarray(background, dtype=data.dtype)

    discrete = _is_discrete_label_array(data, unique_values, max_labels)
    label_values = _non_background_unique_values(unique_values, bg, eps, data.dtype)
    layers: list[dict] = []
    if discrete:
        if len(label_values) > max_labels:
            discrete = False
            warn(
                f"VTI array has {len(label_values)} non-background values; "
                f"treating it as one scalar occupancy layer. Increase --vti-max-labels to split it."
            )
        else:
            for idx, value in enumerate(label_values):
                if np.issubdtype(data.dtype, np.floating):
                    mask = np.isclose(data, value, atol=max(eps, 1e-6))
                else:
                    mask = data == value
                name = f"{array_label}={_label_display_value(value)}"
                layer = _make_mask_layer(
                    _safe_layer_id(array_label, _label_display_value(value)),
                    name,
                    mask,
                    LABEL_COLOR_PALETTE[idx % len(LABEL_COLOR_PALETTE)],
                    label_value=_label_display_value(value),
                    exclusive_group="vti-labels"
                )
                layer["volume_kind"] = "labels"
                layers.append(layer)

    if not discrete:
        occupancy = ~background_mask
        scalar_layer = _make_mask_layer(
            _safe_layer_id(array_label, "nonzero"),
            f"{array_label} non-background",
            occupancy,
            LABEL_COLOR_PALETTE[0],
            label_value="non-background",
            exclusive_group=None
        )
        layers.append(_attach_scalar_volume(scalar_layer, data, scalar_range))

    if not layers:
        warn("VTI did not contain any non-background voxels after sampling")
        layers.append(_make_mask_layer(
            _safe_layer_id(array_label, "empty"),
            f"{array_label} empty",
            np.zeros_like(data, dtype=np.uint8),
            LABEL_COLOR_PALETTE[0],
            label_value="empty",
            exclusive_group=None
        ))

    metadata.update({
        "background": background,
        "background_eps": background_eps,
        "discrete_labels": discrete,
        "volume_kind": "labels" if discrete else "scalar",
        "scalar_range": scalar_range,
        "unique_value_count": int(unique_values.size),
        "non_background_unique_value_count": int(len(label_values)),
        "labels": [layer["label_value"] for layer in layers],
    })
    info(f"Prepared {len(layers)} editable VTI layer(s): {', '.join(layer['name'] for layer in layers[:8])}")
    if len(layers) > 8:
        info(f"... plus {len(layers) - 8} more labels")
    return layers, grid, metadata


def load_vti_directory_label_layers(
    vti_dir: str,
    array_name: str | None = None,
    background: float = 0.0,
    background_eps: float = 0.0,
    max_dim: int = 160,
    max_labels: int = 64,
) -> tuple[list[dict], dict, dict]:
    """
    Load all VTI tiles in a directory, assemble them spatially, and return label layers.

    Each tile must be a VTK ImageData file with an Origin attribute that positions it in
    a shared coordinate system.  Tiles are stitched together into a single sampled volume
    and then split into discrete label layers exactly as load_vti_label_layers() does.
    """
    vti_dir_path = Path(vti_dir).resolve()
    if not vti_dir_path.is_dir():
        raise ValueError(f"Not a directory: {vti_dir}")
    data, grid, metadata = _assemble_vti_tiles(vti_dir_path, array_name=array_name, max_dim=max_dim)
    unique_values = np.unique(data)
    array_label = metadata["array_name"]
    max_labels = max(1, int(max_labels))
    eps = max(0.0, float(background_eps))
    bg = float(background)
    scalar_range = _finite_range(data)

    if np.issubdtype(data.dtype, np.floating):
        background_mask = np.isclose(data, bg, atol=eps)
    else:
        background_mask = data == np.asarray(background, dtype=data.dtype)

    discrete = _is_discrete_label_array(data, unique_values, max_labels)
    label_values = _non_background_unique_values(unique_values, bg, eps, data.dtype)
    layers: list[dict] = []
    if discrete:
        if len(label_values) > max_labels:
            discrete = False
            warn(
                f"Assembled VTI volume has {len(label_values)} non-background values; "
                f"treating it as a single scalar occupancy layer. Increase --vti-max-labels to split."
            )
        else:
            for idx, value in enumerate(label_values):
                if np.issubdtype(data.dtype, np.floating):
                    mask = np.isclose(data, value, atol=max(eps, 1e-6))
                else:
                    mask = data == value
                name = f"{array_label}={_label_display_value(value)}"
                layer = _make_mask_layer(
                    _safe_layer_id(array_label, _label_display_value(value)),
                    name,
                    mask,
                    LABEL_COLOR_PALETTE[idx % len(LABEL_COLOR_PALETTE)],
                    label_value=_label_display_value(value),
                    exclusive_group="vti-labels"
                )
                layer["volume_kind"] = "labels"
                layers.append(layer)

    if not discrete:
        occupancy = ~background_mask
        scalar_layer = _make_mask_layer(
            _safe_layer_id(array_label, "nonzero"),
            f"{array_label} non-background",
            occupancy,
            LABEL_COLOR_PALETTE[0],
            label_value="non-background",
            exclusive_group=None
        )
        layers.append(_attach_scalar_volume(scalar_layer, data, scalar_range))

    if not layers:
        warn("VTI directory assembly did not yield any non-background voxels")
        layers.append(_make_mask_layer(
            _safe_layer_id(array_label, "empty"),
            f"{array_label} empty",
            np.zeros_like(data, dtype=np.uint8),
            LABEL_COLOR_PALETTE[0],
            label_value="empty",
            exclusive_group=None
        ))

    metadata.update({
        "background": background,
        "background_eps": background_eps,
        "discrete_labels": discrete,
        "volume_kind": "labels" if discrete else "scalar",
        "scalar_range": scalar_range,
        "unique_value_count": int(unique_values.size),
        "non_background_unique_value_count": int(len(label_values)),
        "labels": [layer["label_value"] for layer in layers],
    })
    info(f"Prepared {len(layers)} editable layer(s) from directory: "
         f"{', '.join(layer['name'] for layer in layers[:8])}")
    if len(layers) > 8:
        info(f"... plus {len(layers) - 8} more labels")
    return layers, grid, metadata


def voxelize_mesh(mesh: trimesh.Trimesh, grid: dict) -> np.ndarray:
    """
    OPTIONAL: Voxelize mesh with trimesh.voxelized(pitch=s) and embed it in the cube.
    (Not required for winding classification, but kept for parity / debugging.)
    Returns a boolean (Z,Y,X) volume aligned with the grid.
    """
    s = grid["spacing"][0]
    N = grid["shape"][0]

    vg = mesh.voxelized(pitch=s)
    vol = vg.matrix.astype(bool)  # native (z,y,x) order from trimesh

    # center-embed into NxNxN cube
    cube = np.zeros(grid["shape"], dtype=bool)
    sz, sy, sx = vol.shape
    oz = max((N - sz) // 2, 0); oy = max((N - sy) // 2, 0); ox = max((N - sx) // 2, 0)
    z1, y1, x1 = oz + sz, oy + sy, ox + sx
    cube[oz:z1, oy:y1, ox:x1] = vol
    info(f"Voxelized (optional) → vol={vol.shape}, cube={cube.shape}, filled={cube.mean():.5f}")
    return cube

# ---------------------------------------------------------------------
# Winding-based in/on/out (original approach)
# ---------------------------------------------------------------------
def classify_by_winding(mesh: trimesh.Trimesh, grid: dict, band: float = 0.6
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use libigl’s fast winding number to classify inside / on / out.
    Returns uint8 masks of shape (N,N,N) in (Z,Y,X) order.
    """
    if igl is None:
        raise RuntimeError(
            "python-igl is required for winding classification. "
            "Install with: conda install -c conda-forge python-igl  (or pip install igl)"
        )

    N = grid["shape"][0]
    s = grid["spacing"][0]
    ox, oy, oz = grid["origin"]

    # Query points = centers of all voxels (Z,Y,X → world X,Y,Z)
    kji = np.indices((N, N, N), dtype=float)  # (3, N, N, N)
    xs = ox + (kji[2] + 0.5) * s  # i
    ys = oy + (kji[1] + 0.5) * s  # j
    zs = oz + (kji[0] + 0.5) * s  # k
    Q = np.stack((xs.ravel(), ys.ravel(), zs.ravel()), axis=1)  # (N^3, 3)

    V = mesh.vertices.copy()
    F = mesh.faces.copy()

    WN = igl.fast_winding_number_for_meshes(V, F, Q)  # type:ignore
    WN = np.asarray(WN).reshape((N, N, N))  # (k, j, i) == (Z,Y,X)

    # Inside = WN > 0.5
    inside = (WN > 0.5)

    # On = narrow band around surface: boundary voxels of 'inside'
    dil = binary_dilation(inside, structure=np.ones((3, 3, 3), dtype=bool))
    boundary = dil ^ inside
    on = boundary & ~inside

    out = ~(inside | on)

    return inside.astype(np.uint8), on.astype(np.uint8), out.astype(np.uint8)

def classify_by_trimesh_contains(
    mesh: trimesh.Trimesh,
    grid: dict,
    batch_size: int = 200_000
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Use trimesh.contains (ray parity) to classify inside / on / out.
    Slower than winding but avoids python-igl.
    """
    N = grid["shape"][0]
    s = grid["spacing"][0]
    ox, oy, oz = grid["origin"]

    total = N * N * N
    inside_flat = np.zeros(total, dtype=bool)

    for start in range(0, total, batch_size):
        end = min(start + batch_size, total)
        idx = np.arange(start, end, dtype=np.int64)

        k = idx // (N * N)
        rem = idx % (N * N)
        j = rem // N
        i = rem % N

        xs = ox + (i + 0.5) * s
        ys = oy + (j + 0.5) * s
        zs = oz + (k + 0.5) * s
        Q = np.stack((xs, ys, zs), axis=1)

        try:
            inside_flat[start:end] = mesh.contains(Q)
        except Exception as exc:
            raise RuntimeError(
                "trimesh.contains failed; consider installing python-igl or using a different mesh"
            ) from exc

    inside = inside_flat.reshape((N, N, N))

    dil = binary_dilation(inside, structure=np.ones((3, 3, 3), dtype=bool))
    boundary = dil ^ inside
    on = boundary & ~inside
    out = ~(inside | on)

    return inside.astype(np.uint8), on.astype(np.uint8), out.astype(np.uint8)

# ---------------------------------------------------------------------
# Viewer utilities
# ---------------------------------------------------------------------
def _slice_frames(grid, i_idx, j_idx, k_idx, color_map=None, line_width=SLICE_BORDER_WIDTH):
    """
    Draw wireframe rectangles showing slice locations (X/Y/Z).
    Colors correspond to sliders: X=blue, Y=green, Z=orange.
    """
    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]
    color_map = color_map or {
        "x": COLOR_SCHEME["slice_x"],
        "y": COLOR_SCHEME["slice_y"],
        "z": COLOR_SCHEME["slice_z"]
    }

    extent_x = Nx * sx
    extent_y = Ny * sy
    extent_z = Nz * sz

    Xr = [ox, ox + extent_x]
    Yr = [oy, oy + extent_y]
    Zr = [oz, oz + extent_z]

    # convert index → voxel center position
    Xpos = ox + (i_idx + 0.5) * sx
    Ypos = oy + (j_idx + 0.5) * sy
    Zpos = oz + (k_idx + 0.5) * sz

    def rect(x, y, z, color, name):
        return go.Scatter3d(
            x=x + [x[0]], y=y + [y[0]], z=z + [z[0]],
            mode="lines", line=dict(color=color, width=line_width),
            name=name, showlegend=False
        )

    frames = []
    # X-slice (YZ plane)
    frames.append(rect(
        x=[Xpos]*4,
        y=[Yr[0], Yr[0], Yr[1], Yr[1]],
        z=[Zr[0], Zr[1], Zr[1], Zr[0]],
        color=color_map["x"], name="x-slice"
    ))
    # Y-slice (XZ plane)
    frames.append(rect(
        x=[Xr[0], Xr[1], Xr[1], Xr[0]],
        y=[Ypos]*4,
        z=[Zr[0], Zr[0], Zr[1], Zr[1]],
        color=color_map["y"], name="y-slice"
    ))
    # Z-slice (XY plane)
    frames.append(rect(
        x=[Xr[0], Xr[1], Xr[1], Xr[0]],
        y=[Yr[0], Yr[0], Yr[1], Yr[1]],
        z=[Zpos]*4,
        color=color_map["z"], name="z-slice"
    ))
    return frames

def _roi_frame(
    grid,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    r_vox: int,
    *,
    n_lat: int = 12,
    n_lon: int = 12,
    color: str = "#E11D48",
    line_width: int = 1,
):
    """
    Build a spherical ROI wireframe centered at voxel indices (i,j,k)
    with radius specified in voxels (r_vox). Returns a list of Scatter3d traces
    similar to _slice_frames(), intended to be directly added to the 3D figure.

    Notes:
    - Indices (i,j,k) are interpreted consistently with _slice_frames: position
      = origin + index * spacing (voxel-aligned). This mirrors the slice planes.
    - Radius is converted to millimeters using grid spacing.
    - The wireframe consists of latitude and longitude rings for clarity and speed.
    """
    if r_vox is None or r_vox <= 0:
        return []

    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]

    # Center in world coordinates (voxel-aligned like slice frames)
    Xc = ox + i_idx * sx
    Yc = oy + j_idx * sy
    Zc = oz + k_idx * sz

    # Radius in mm (use isotropic spacing assumption in this grid)
    s = sx  # == sy == sz
    r = float(r_vox) * s
    if r <= 0:
        return []

    # Sampling resolution for wireframe rings
    n_lat = max(3, int(n_lat))   # latitude (phi in [0, pi])
    n_lon = max(4, int(n_lon))   # longitude (theta in [0, 2pi))
    n_circle_pts = 60            # points per ring

    traces = []

    # Latitude rings (parallel to XY planes): phi from epsilon..pi-epsilon to avoid degeneracy
    phis = np.linspace(1e-6, np.pi - 1e-6, n_lat)
    thetas_ring = np.linspace(0, 2 * np.pi, n_circle_pts)
    for phi in phis:
        sinp = np.sin(phi)
        cosp = np.cos(phi)
        xs = Xc + r * sinp * np.cos(thetas_ring)
        ys = Yc + r * sinp * np.sin(thetas_ring)
        zs = Zc + r * np.full_like(thetas_ring, cosp)
        traces.append(
            go.Scatter3d(
                x=np.r_[xs, xs[0]], y=np.r_[ys, ys[0]], z=np.r_[zs, zs[0]],
                mode="lines",
                line=dict(color=color, width=line_width),
                name="roi-sphere-lat",
                showlegend=False,
            )
        )

    # Longitude rings (half-meridians closing into lines): theta grid
    thetas = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    phis_mer = np.linspace(0, np.pi, n_circle_pts)
    for th in thetas:
        cost = np.cos(th)
        sint = np.sin(th)
        sinp = np.sin(phis_mer)
        cosp = np.cos(phis_mer)
        xs = Xc + r * sinp * cost
        ys = Yc + r * sinp * sint
        zs = Zc + r * cosp
        traces.append(
            go.Scatter3d(
                x=xs, y=ys, z=zs,
                mode="lines",
                line=dict(color=color, width=line_width),
                name="roi-sphere-lon",
                showlegend=False,
            )
        )

    return traces

def mask_to_trace(mask_u8, grid, color, name, opacity=0.4):
    total = int(np.sum(mask_u8))
    info(f"[{name}] mask sum={total}")
    if total == 0:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(mask_u8, level=0.5)
        info(f"[{name}] marching cubes → verts={len(verts)}, faces={len(faces)}")
    except Exception as e:
        err(f"[{name}] marching cubes failed: {e}")
        return None

    sz, sy, sx = grid["spacing"]
    origin = np.array(grid["origin"])
    # marching_cubes returns verts in (z,y,x) index space → map to world (x,y,z)
    coords = origin + verts[:, [2, 1, 0]] * np.array([sx, sy, sz])

    i, j, k = faces.T
    return go.Mesh3d(
        x=coords[:, 0], y=coords[:, 1], z=coords[:, 2],
        i=i, j=j, k=k,
        name=name, color=color, opacity=opacity,
        flatshading=True, showscale=False
    )

def mask_to_volume_trace(mask_u8: np.ndarray, grid: dict, color: str, name: str, opacity: float):
    total = int(np.sum(mask_u8))
    info(f"[{name}] volume mask sum={total}")
    if total == 0:
        return None

    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]
    x_coords = ox + (np.arange(Nx) + 0.5) * sx
    y_coords = oy + (np.arange(Ny) + 0.5) * sy
    z_coords = oz + (np.arange(Nz) + 0.5) * sz
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

    return go.Volume(
        x=xx.ravel(),
        y=yy.ravel(),
        z=zz.ravel(),
        value=mask_u8.ravel(),
        isomin=0.5,
        isomax=1.5,
        surface_count=2,
        opacity=max(0.01, min(1.0, float(opacity))),
        colorscale=[[0.0, "rgba(0,0,0,0)"], [1.0, color]],
        showscale=False,
        caps=dict(x_show=False, y_show=False, z_show=False),
        name=name,
        hoverinfo="skip"
    )

def scalar_to_volume_trace(
    values_zyx: np.ndarray,
    grid: dict,
    transfer_points: list[dict],
    scalar_range: tuple[float, float],
    opacity_scale: float,
    color_scheme: str = DEFAULT_TRANSFER_COLOR_SCHEME,
    name: str = "scalar-volume",
):
    if values_zyx is None or values_zyx.size == 0:
        return None
    vmin, vmax = scalar_range
    if np.isclose(vmin, vmax):
        return None

    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]
    x_coords = ox + (np.arange(Nx) + 0.5) * sx
    y_coords = oy + (np.arange(Ny) + 0.5) * sy
    z_coords = oz + (np.arange(Nz) + 0.5) * sz
    zz, yy, xx = np.meshgrid(z_coords, y_coords, x_coords, indexing="ij")

    return go.Volume(
        x=xx.ravel(),
        y=yy.ravel(),
        z=zz.ravel(),
        value=values_zyx.ravel(),
        isomin=vmin,
        isomax=vmax,
        surface_count=24,
        opacity=1.0,
        opacityscale=_transfer_opacityscale(transfer_points, vmin, vmax, opacity_scale),
        colorscale=_transfer_colorscale(color_scheme),
        showscale=True,
        caps=dict(x_show=False, y_show=False, z_show=False),
        colorbar=dict(
            title="Signal",
            thickness=10,
            len=0.50,
            x=0.985,
            xanchor="right",
            xpad=0,
            y=0.50,
            yanchor="middle",
            ypad=0,
            bgcolor="rgba(15, 19, 26, 0.72)",
            bordercolor="rgba(226, 232, 240, 0.35)",
            borderwidth=1,
        ),
        name=name,
        hovertemplate="x=%{x:.2f}<br>y=%{y:.2f}<br>z=%{z:.2f}<br>value=%{value:.3g}<extra></extra>",
    )

def mask_to_trimesh(mask_u8: np.ndarray, grid: dict, name: str | None = None) -> trimesh.Trimesh | None:
    """Convert a binary mask to a surface mesh using marching cubes."""
    if trimesh is None:
        raise RuntimeError("trimesh is required for STL export. Install with: pip install trimesh")
    if mask_u8 is None or int(np.sum(mask_u8)) == 0:
        return None
    try:
        verts, faces, _, _ = measure.marching_cubes(mask_u8, level=0.5)
    except Exception as e:
        err(f"[{name or 'mask'}] marching cubes failed: {e}")
        return None

    sz, sy, sx = grid["spacing"]
    origin = np.array(grid["origin"])
    coords = origin + verts[:, [2, 1, 0]] * np.array([sx, sy, sz])
    return trimesh.Trimesh(vertices=coords, faces=faces, process=False)

def save_mask_stl(mask_u8: np.ndarray, grid: dict, out_path: Path, name: str) -> bool:
    """Save a binary mask as STL (returns True when written)."""
    mesh = mask_to_trimesh(mask_u8, grid, name=name)
    if mesh is None:
        return False
    out_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        mesh.export(out_path)
    except Exception as e:
        err(f"[{name}] STL export failed: {e}")
        return False
    return True

def _resolve_output_paths(in_mesh_path: str, out_dir: str) -> tuple[Path, Path]:
    """Return validated output directory and derived NPZ path from input name."""
    input_path = Path(in_mesh_path).resolve()
    out_root = Path(out_dir)
    if out_root.suffix:
        raise ValueError("--out must be a directory path, not a filename")
    if out_root.exists() and not out_root.is_dir():
        raise ValueError(f"--out points to a file, expected a directory: {out_root}")

    out_root.mkdir(parents=True, exist_ok=True)
    input_token = input_path.name if input_path.is_dir() else input_path.stem
    npz_path = out_root / f"{input_token}_masks.npz"
    return out_root, npz_path

def _binary_colorscale(on_color="#ffffff"):
    return [[0.0, "black"], [1.0, on_color]]

def _slice_colorscale(color):
    """Compatibility helper for slice overlays."""
    return _binary_colorscale(color)

def _slice_plane_colorscale(slice_color, roi_color, roi_alpha):
    """Compatibility helper for slice planes in 3-D view."""
    _ = roi_color, roi_alpha
    return _binary_colorscale(slice_color)

def _slice_fig(z2d, title, border_color="#ffffff", colorscale=None, zmax=1):
    if colorscale is None:
        colorscale = _binary_colorscale(border_color)
    x_vals = np.arange(z2d.shape[1])
    y_vals = np.arange(z2d.shape[0])
    fig = go.Figure(
        data=[go.Heatmap(
            z=z2d,
            x=x_vals,
            y=y_vals,
            zmin=zmin,
            zmax=zmax,
            colorscale=colorscale,
            showscale=showscale,
            hovertemplate="x=%{x}<br>y=%{y}<br>value=%{z}<extra></extra>"
        )],
        layout=go.Layout(
            title=dict(text=title, font=dict(color=border_color, size=14)),
            margin=dict(l=2, r=2, t=28, b=2),
            xaxis=dict(showticklabels=False), yaxis=dict(autorange="reversed", showticklabels=False),
            plot_bgcolor="black", paper_bgcolor="black"
        )
    )
    fig.add_shape(
        type="rect", x0=0, y0=0, x1=1, y1=1,
        xref="paper", yref="paper",
        line=dict(color=border_color, width=3),
        fillcolor="rgba(0,0,0,0)"
    )
    return fig

def _empty_slice(grid, axis):
    Nz, Ny, Nx = grid["shape"]
    if axis == "x":
        return np.zeros((Nz, Ny), dtype=np.uint8)
    if axis == "y":
        return np.zeros((Nz, Nx), dtype=np.uint8)
    if axis == "z":
        return np.zeros((Ny, Nx), dtype=np.uint8)
    raise ValueError("axis must be 'x', 'y', or 'z'")

def _roi_slice_mask(grid, axis, idx, center, radius):
    if radius <= 0:
        return _empty_slice(grid, axis)

    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]

    x_coords = ox + (np.arange(Nx) + 0.5) * sx
    y_coords = oy + (np.arange(Ny) + 0.5) * sy
    z_coords = oz + (np.arange(Nz) + 0.5) * sz
    cx, cy, cz = center
    r2 = radius * radius

    if axis == "x":
        x = ox + (idx + 0.5) * sx
        Y, Z = np.meshgrid(y_coords, z_coords)
        dist2 = (x - cx) ** 2 + (Y - cy) ** 2 + (Z - cz) ** 2
        return (dist2 <= r2).astype(np.uint8)
    if axis == "y":
        y = oy + (idx + 0.5) * sy
        X, Z = np.meshgrid(x_coords, z_coords)
        dist2 = (X - cx) ** 2 + (y - cy) ** 2 + (Z - cz) ** 2
        return (dist2 <= r2).astype(np.uint8)
    if axis == "z":
        z = oz + (idx + 0.5) * sz
        X, Y = np.meshgrid(x_coords, y_coords)
        dist2 = (X - cx) ** 2 + (Y - cy) ** 2 + (z - cz) ** 2
        return (dist2 <= r2).astype(np.uint8)
    raise ValueError("axis must be 'x', 'y', or 'z'")


def compose_slice(masks, axis, idx):
    """Combine active masks along a slice (arrays are (Z,Y,X))."""
    if len(masks) == 0:
        return np.zeros((1, 1), dtype=np.uint8)
    if axis == 'x':   # i: keep Z,Y
        slices = [m[:, :, idx] for m in masks]
    elif axis == 'y': # j: keep Z,X
        slices = [m[:, idx, :] for m in masks]
    elif axis == 'z': # k: keep Y,X
        slices = [m[idx, :, :] for m in masks]
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")
    out = np.zeros_like(slices[0], dtype=np.uint8)
    for s in slices:
        out |= (s > 0).astype(np.uint8)
    return out

# ---------------------------------------------------------------------
# ROI-based masking utilities
# ---------------------------------------------------------------------
def filter_inside_by_roi(
        inside_u8: np.ndarray,
        i_idx: int,
        j_idx: int,
        k_idx: int,
        r_vox: int,
) -> Tuple[np.ndarray, np.ndarray]:
        """
        Intersect the inside_u8 mask with a spherical ROI centered at voxel
        indices (i_idx, j_idx, k_idx) with radius r_vox (in voxels).

        Inputs
        - inside_u8: uint8 mask array of shape (Z, Y, X); nonzero denotes inside
        - i_idx, j_idx, k_idx: ROI center in voxel indices matching the UI sliders
          and _slice_frames convention (i=X, j=Y, k=Z)
        - r_vox: radius of the sphere in voxels

        Returns (masked_in_roi, masked_outside_roi)
        - masked_in_roi:  uint8 mask (Z, Y, X) of inside_u8 limited to the ROI
        - masked_outside_roi: uint8 mask (Z, Y, X) of inside_u8 outside the ROI

        Notes
        - Computation happens in index space (voxel units). Grid spacing is not
          required since r_vox is already in voxels.
        """
        if inside_u8 is None:
                raise ValueError("inside_u8 must not be None")
        if r_vox is None or r_vox <= 0:
                # Nothing selected → all inside remains outside-ROI part
                outside_only = (inside_u8 > 0).astype(np.uint8)
                return np.zeros_like(inside_u8, dtype=np.uint8), outside_only

        Nz, Ny, Nx = inside_u8.shape
        # Distance in voxel units from (i_idx, j_idx, k_idx)
        ii = np.arange(Nx) - int(i_idx)
        jj = np.arange(Ny) - int(j_idx)
        kk = np.arange(Nz) - int(k_idx)

        rr2 = (ii * ii)[None, None, :] + (jj * jj)[None, :, None] + (kk * kk)[:, None, None]
        roi = rr2 <= (int(r_vox) * int(r_vox))

        inside_bool = (inside_u8 > 0)
        masked_in_roi = (inside_bool & roi).astype(np.uint8)
        masked_outside_roi = (inside_bool & (~roi)).astype(np.uint8)
        return masked_in_roi, masked_outside_roi

def _bitwise_dilate6(mask_bool: np.ndarray) -> np.ndarray:
    """Single-iteration 3D dilation using 6-neighborhood via bitwise shifts.
    mask_bool is shape (Z, Y, X), dtype=bool.
    """
    zpos = np.pad(mask_bool[:-1, :, :], ((1,0),(0,0),(0,0)), mode='constant', constant_values=False)
    zneg = np.pad(mask_bool[1:,  :, :], ((0,1),(0,0),(0,0)), mode='constant', constant_values=False)
    ypos = np.pad(mask_bool[:, :-1, :], ((0,0),(1,0),(0,0)), mode='constant', constant_values=False)
    yneg = np.pad(mask_bool[:, 1:,  :], ((0,0),(0,1),(0,0)), mode='constant', constant_values=False)
    xpos = np.pad(mask_bool[:, :, :-1], ((0,0),(0,0),(1,0)), mode='constant', constant_values=False)
    xneg = np.pad(mask_bool[:, :, 1: ], ((0,0),(0,0),(0,1)), mode='constant', constant_values=False)
    out = mask_bool | zpos | zneg | ypos | yneg | xpos | xneg
    return out

def _bitwise_erode6(mask_bool: np.ndarray) -> np.ndarray:
    """Single-iteration 3D erosion using 6-neighborhood via bitwise shifts.
    mask_bool is shape (Z, Y, X), dtype=bool.
    """
    zpos = np.pad(mask_bool[:-1, :, :], ((1,0),(0,0),(0,0)), mode='constant', constant_values=False)
    zneg = np.pad(mask_bool[1:,  :, :], ((0,1),(0,0),(0,0)), mode='constant', constant_values=False)
    ypos = np.pad(mask_bool[:, :-1, :], ((0,0),(1,0),(0,0)), mode='constant', constant_values=False)
    yneg = np.pad(mask_bool[:, 1:,  :], ((0,0),(0,1),(0,0)), mode='constant', constant_values=False)
    xpos = np.pad(mask_bool[:, :, :-1], ((0,0),(0,0),(1,0)), mode='constant', constant_values=False)
    xneg = np.pad(mask_bool[:, :, 1: ], ((0,0),(0,0),(0,1)), mode='constant', constant_values=False)
    out = mask_bool & zpos & zneg & ypos & yneg & xpos & xneg
    return out

def proximity_bridge(mask_u8: np.ndarray, max_dist: int = 2) -> np.ndarray:
    """
    Bridge disconnected components that are within `max_dist` voxels of each other.
    - mask_u8: uint8 mask (Z,Y,X)
    - max_dist: Chebyshev distance threshold (voxels)

    Strategy:
        1. Dilate mask by max_dist
        2. Compute potential bridging voxels that lie between disconnected components
        3. Activate these voxels in the output mask
    """
    mask = (mask_u8 > 0)
    if max_dist <= 0:
        return mask_u8

    # Step 1: Chebyshev dilation (cube of size (2*max_dist+1))
    size = 2 * max_dist + 1
    kernel = np.ones((size, size, size), dtype=bool)
    dil = binary_dilation(mask, structure=kernel)

    # Step 2: bridging candidates = dilated minus original
    bridge = dil & (~mask)

    # Step 3: activate only voxels that lie between ≥2 different components
    # Identify connected components
    labeled, num = measure.label(mask.astype(np.uint8), return_num=True, connectivity=1)

    if num <= 1:
        # nothing to connect
        out = mask.copy()
        return out.astype(np.uint8)

    # For each bridging voxel, check if neighboring labels belong to >=2 components
    Z, Y, X = np.where(bridge)
    out = mask.copy()

    for z, y, x in zip(Z, Y, X):
        # Neighborhood (6-connectivity)
        neigh = []
        if z > 0:     neigh.append(labeled[z-1, y, x])
        if z < labeled.shape[0]-1: neigh.append(labeled[z+1, y, x])
        if y > 0:     neigh.append(labeled[z, y-1, x])
        if y < labeled.shape[1]-1: neigh.append(labeled[z, y+1, x])
        if x > 0:     neigh.append(labeled[z, y, x-1])
        if x < labeled.shape[2]-1: neigh.append(labeled[z, y, x+1])

        neigh = [n for n in neigh if n != 0]

        if len(set(neigh)) >= 2:
            out[z, y, x] = True

    return out.astype(np.uint8)

def apply_roi_morphology(
    inside_u8: np.ndarray,
    i_idx: int,
    j_idx: int,
    k_idx: int,
    r_vox: int,
    iterations: int = 1,
    mode: str = "dilate",
) -> np.ndarray:
    """
    Apply iterative 3D dilation or shrink (erosion) confined to a spherical ROI.

    Steps
    - Use filter_inside_by_roi to split the global inside mask into:
      inside_in_roi, inside_outside_roi.
    - Apply `iterations` rounds of bitwise morphology (6-neighborhood) to
      inside_in_roi only, clamping to the ROI so changes don't leak outside.
    - Recombine with inside_outside_roi to form the new global inside mask.

    Parameters
    - inside_u8: uint8 global inside mask (Z,Y,X)
    - i_idx, j_idx, k_idx: center of ROI in voxel indices (i=X, j=Y, k=Z)
    - r_vox: ROI radius in voxels
    - iterations: number of iterative morphology steps
    - mode: 'dilate' to expand, 'shrink' to erode

    Returns
    - uint8 global inside mask after local morphology within ROI.
    """
    if iterations is None or iterations <= 0:
        return (inside_u8 > 0).astype(np.uint8)

    roi_inside_u8, outside_roi_inside_u8 = filter_inside_by_roi(inside_u8, i_idx, j_idx, k_idx, r_vox)

    # Build ROI boolean from geometry (always clamp to the full spherical ROI)
    Nz, Ny, Nx = inside_u8.shape
    ii = np.arange(Nx) - int(i_idx)
    jj = np.arange(Ny) - int(j_idx)
    kk = np.arange(Nz) - int(k_idx)
    rr2 = (ii * ii)[None, None, :] + (jj * jj)[None, :, None] + (kk * kk)[:, None, None]
    roi_bool = rr2 <= (int(r_vox) * int(r_vox))

    work = roi_inside_u8.astype(bool)
    for _ in range(int(iterations)):
        if mode == "dilate":
            work = _bitwise_dilate6(work)
        elif mode in ("shrink", "erode", "erosion"):
            work = _bitwise_erode6(work)
        else:
            raise ValueError("mode must be 'dilate' or 'shrink' (alias: 'erode','erosion')")
        # Clamp to ROI to prevent leaking outside
        work = work & roi_bool

    # Recompose: updated ROI portion + untouched outside-ROI portion
    updated_roi_u8 = work.astype(np.uint8)
    new_inside = ((outside_roi_inside_u8 > 0) | (updated_roi_u8 > 0)).astype(np.uint8)
    # Bridge any ROI→non-ROI pairs within 2 voxels
    new_inside = proximity_bridge(new_inside, max_dist=2)
    
    return new_inside

def _sigmoid_iteration_schedule(num_steps: int, total_iters: int) -> np.ndarray:
    """
    Distribute `total_iters` across `num_steps` with a slow-fast-slow profile.
    Uses a Hann window to produce low weights at the ends and high in the middle.

    Returns an int array of length `num_steps` with at least 1 per step
    and approximately summing to `total_iters`.
    """
    if num_steps <= 0 or total_iters <= 0:
        return np.array([], dtype=int)

    t = np.linspace(0.0, 1.0, num_steps)
    weights = 0.5 - 0.5 * np.cos(2.0 * np.pi * t)  # Hann window: 0 at ends, peak mid
    weights = weights + 1e-6  # avoid exact zeros
    weights = weights / np.sum(weights) * float(total_iters)

    iters = np.maximum(1, np.round(weights).astype(int))
    diff = int(total_iters - int(np.sum(iters)))
    if diff != 0:
        # Adjust to match total_iters by tweaking largest/smallest weights
        order = np.argsort(-weights) if diff > 0 else np.argsort(weights)
        for idx in order[:abs(diff)]:
            iters[idx] += 1 if diff > 0 else -1
            if iters[idx] < 1:
                iters[idx] = 1
    return iters

def _sigmoid_sparse_apply_schedule(total_ticks: int, num_applies: int) -> np.ndarray:
    """
    Build a boolean schedule of length `total_ticks` marking which ticks should
    apply a single-iteration morphology step. The placement follows a
    slow–fast–slow (sigmoidal) density using a Hann window: denser in the
    middle, sparser at the ends. Ensures exactly `num_applies` True entries
    (clamped to `total_ticks`).

    This keeps each morphology change to a single-voxel iteration so
    `proximity_bridge` can reliably handle gaps.
    """
    if total_ticks <= 0 or num_applies <= 0:
        return np.zeros(max(0, total_ticks), dtype=bool)
    num_applies = min(num_applies, total_ticks)

    # Hann window weights across ticks (exclude endpoint to avoid duplicate zero)
    t = np.linspace(0.0, 1.0, total_ticks, endpoint=False)
    weights = 0.5 - 0.5 * np.cos(2.0 * np.pi * t)

    order = np.argsort(-weights)  # descending by weight (middle first)
    chosen = order[:num_applies]
    schedule = np.zeros(total_ticks, dtype=bool)
    schedule[chosen] = True
    return schedule

def show_viewer_dash(
    mesh,
    inside_u8=None,
    on_u8=None,
    out_u8=None,
    grid=None,
    port=8050,
    mask_layers: list[dict] | None = None,
    viewer_kind: str = "surface",
):
    """Interactive viewer: X/Y/Z slices + 3-D, with mask/label toggles."""
    if dash is None or go is None:
        raise RuntimeError("Dash and Plotly are required for the browser viewer. Install with: pip install dash plotly")
    if mask_layers is None:
        mask_layers = [
            _make_mask_layer("inside", "Inside", inside_u8, COLOR_SCHEME["inside"], OPACITY_LEVELS["mid"], True),
            _make_mask_layer("on", "On", on_u8, COLOR_SCHEME["on"], OPACITY_LEVELS["low"], False),
            _make_mask_layer("out", "Out", out_u8, COLOR_SCHEME["out"], OPACITY_LEVELS["low"], False),
        ]
    if not mask_layers:
        raise ValueError("show_viewer_dash needs at least one mask layer")
    if grid is None:
        raise ValueError("show_viewer_dash needs a grid")

    viewer_kind = "volume" if viewer_kind == "volume" else "surface"
    is_vti_viewer = viewer_kind == "volume"
    scalar_layers = [
        layer for layer in mask_layers
        if layer.get("volume_kind") == "scalar" and layer.get("scalar_data") is not None
    ]
    is_scalar_volume = is_vti_viewer and bool(scalar_layers)
    scalar_layer_id = scalar_layers[0]["id"] if is_scalar_volume else None
    scalar_data = (
        scalar_layers[0]["scalar_data"].astype(np.float32, copy=False)
        if is_scalar_volume else None
    )
    scalar_range = (
        tuple(float(v) for v in scalar_layers[0].get("scalar_range", _finite_range(scalar_data)))
        if is_scalar_volume else (0.0, 1.0)
    )
    scalar_min, scalar_max = scalar_range
    default_transfer_points = _default_transfer_points(scalar_min, scalar_max, scalar_data)

    Nz, Ny, Nx = mask_layers[0]["mask"].shape
    x_mid, y_mid, z_mid = Nx // 2, Ny // 2, Nz // 2

    # Mesh (as loaded, centered at origin)
    mesh_trace = go.Mesh3d(
        x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
        i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
        name="mesh", color=COLOR_SCHEME["mesh"], opacity=OPACITY_LEVELS["low"]
    )
    # We will recompute the inside trace dynamically to reflect morphology changes
    # inside_trace = mask_to_trace(inside_u8, grid, "#3B82F6", "inside", 0.35)
    on_trace     = mask_to_trace(on_u8,     grid, "#22C55E", "on",     0.55)
    out_trace    = mask_to_trace(out_u8,    grid, "#F59E0B", "out",    0.08)

    # Mutable state for inside mask so we can apply morphology and reset
    inside_state = {"inside": inside_u8.copy(), "original": inside_u8.copy()}

    app = dash.Dash(__name__)
    panel_card = {
        "backgroundColor": "#0f131a",
        "border": "1px solid #232838",
        "borderRadius": "8px",
        "padding": "10px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "8px"
    }
    control_row = {
        "display": "flex",
        "alignItems": "center",
        "gap": "8px"
    }
    color_chip = {
        "width": "32px",
        "height": "24px",
        "padding": "0",
        "border": "1px solid #2b3347",
        "borderRadius": "4px",
        "background": "#111522"
    }
    slider_compact = {
        "flex": "1 1 auto",
        "minWidth": "140px"
    }
    dropdown_compact = {
        "minHeight": "34px",
        "fontSize": "12px"
    }
    button_compact = {
        "height": "30px",
        "padding": "4px 10px",
        "fontSize": "12px"
    }
    input_dark = {
        "width": "100%",
        "backgroundColor": "#0f1115",
        "color": "#e6e6e6",
        "border": "1px solid #2a2f3a",
        "borderRadius": "6px",
        "padding": "4px 6px",
        "fontSize": "11px",
    }

    def _resolved_active_transfer_id(points: list[dict], active_id: str | None = None) -> str | None:
        clean_points = _sanitize_transfer_points(points, scalar_min, scalar_max)
        ids = [point["id"] for point in clean_points]
        if active_id in ids:
            return active_id
        return ids[0] if ids else None

    def make_transfer_figure(
        points: list[dict],
        active_id: str | None = None,
        color_scheme: str = DEFAULT_TRANSFER_COLOR_SCHEME,
    ):
        clean_points = _sanitize_transfer_points(points, scalar_min, scalar_max)
        active_id = _resolved_active_transfer_id(clean_points, active_id)
        xs = [point["value"] for point in clean_points]
        ys = [point["opacity"] for point in clean_points]
        colorscale = _transfer_colorscale(color_scheme)
        marker_positions = [_tf_position(point["value"], scalar_min, scalar_max) for point in clean_points]
        ids = [point["id"] for point in clean_points]
        sizes = [14 if point["id"] == active_id else 8 for point in clean_points]
        line_colors = ["#F8FAFC" if point["id"] == active_id else "#111827" for point in clean_points]
        line_widths = [3 if point["id"] == active_id else 1 for point in clean_points]
        strip_x = np.linspace(scalar_min, scalar_max, 160)
        strip_y = np.linspace(0.0, 1.0, 24)
        strip_z = np.tile(np.linspace(0.0, 1.0, strip_x.size), (strip_y.size, 1))
        fig = go.Figure()
        fig.add_trace(go.Heatmap(
            z=strip_z,
            x=strip_x,
            y=strip_y,
            colorscale=colorscale,
            showscale=False,
            opacity=0.30,
            hoverinfo="skip",
        ))
        fig.add_trace(go.Scatter(
            x=xs,
            y=ys,
            mode="lines+markers",
            line=dict(color="#E5E7EB", width=2),
            marker=dict(
                size=sizes,
                color=marker_positions,
                cmin=0.0,
                cmax=1.0,
                colorscale=colorscale,
                showscale=False,
                line=dict(color=line_colors, width=line_widths),
            ),
            customdata=ids,
            hovertemplate="value=%{x:.3g}<br>opacity=%{y:.2f}<extra></extra>",
            showlegend=False,
        ))
        fig.update_layout(
            height=132,
            margin=dict(l=26, r=8, t=4, b=24),
            paper_bgcolor="#0f131a",
            plot_bgcolor="#0f131a",
            clickmode="event+select",
            dragmode=False,
            hovermode="closest",
            uirevision="transfer-map-fixed",
            xaxis=dict(
                color="#CBD5E1",
                showgrid=False,
                zeroline=False,
                range=[scalar_min, scalar_max],
                autorange=False,
                fixedrange=True,
            ),
            yaxis=dict(
                color="#CBD5E1",
                showgrid=True,
                range=[0.0, 1.0],
                autorange=False,
                fixedrange=True,
                title=None,
            ),
        )
        return fig

    def make_transfer_rows(points: list[dict], active_id: str | None = None):
        clean_points = _sanitize_transfer_points(points, scalar_min, scalar_max)
        active_id = _resolved_active_transfer_id(clean_points, active_id)
        rows = []
        for point in clean_points:
            pid = point["id"]
            is_active = pid == active_id
            rows.append(
                html.Div(
                    id={"type": "tf-row", "index": pid},
                    n_clicks=0,
                    style={
                        "display": "grid",
                        "gridTemplateColumns": "1fr 78px 26px",
                        "gap": "6px",
                        "alignItems": "center",
                        "padding": "4px",
                        "borderRadius": "6px",
                        "border": "1px solid #38BDF8" if is_active else "1px solid transparent",
                        "backgroundColor": "rgba(56, 189, 248, 0.12)" if is_active else "transparent",
                        "opacity": "1.0" if is_active else "0.55",
                        "cursor": "pointer",
                    },
                    children=[
                        html.Div(
                            f"{point['value']:.4g}",
                            title=f"value {point['value']:.6g}",
                            style={
                                "fontSize": "11px",
                                "color": "#CBD5E1",
                                "overflow": "hidden",
                                "textOverflow": "ellipsis",
                                "whiteSpace": "nowrap",
                            },
                        ),
                        dcc.Input(
                            id={"type": "tf-opacity", "index": pid},
                            type="number",
                            min=0.0,
                            max=1.0,
                            step=0.01,
                            value=point["opacity"],
                            debounce=True,
                            disabled=not is_active,
                            style=input_dark,
                        ),
                        html.Button(
                            "X",
                            id={"type": "tf-remove", "index": pid},
                            n_clicks=0,
                            disabled=(len(clean_points) <= 2 or not is_active),
                            title="Remove point",
                            style={
                                "height": "24px",
                                "padding": "0",
                                "fontSize": "11px",
                                "background": "#1f2937",
                                "color": "#e5e7eb",
                                "border": "1px solid #2b3347",
                                "borderRadius": "4px",
                                "cursor": "pointer",
                            },
                        ),
                    ],
                )
            )
        return rows

    def make_active_transfer_opacity_control(points: list[dict], active_id: str | None = None):
        clean_points = _sanitize_transfer_points(points, scalar_min, scalar_max)
        active_id = _resolved_active_transfer_id(clean_points, active_id)
        active_point = next(
            (point for point in clean_points if point["id"] == active_id),
            clean_points[0] if clean_points else {"opacity": 0.0},
        )
        active_opacity = float(np.clip(active_point.get("opacity", 0.0), 0.0, 1.0))
        return html.Div(
            style={"display": "flex", "flexDirection": "column", "gap": "4px"},
            children=[
                html.Div(
                    style={"display": "flex", "justifyContent": "space-between",
                           "alignItems": "center", "fontSize": "11px", "opacity": "0.85"},
                    children=[
                        html.Span("Point opacity"),
                        html.Span(f"{active_opacity:.2f}"),
                    ],
                ),
                dcc.Slider(
                    id="active-transfer-opacity",
                    min=0.0,
                    max=1.0,
                    step=0.01,
                    value=active_opacity,
                    updatemode="drag",
                    marks=None,
                    tooltip={"placement": "bottom", "always_visible": False},
                ),
            ],
        )

    def make_transfer_editor(points: list[dict], active_id: str | None = None):
        return [
            *make_transfer_rows(points, active_id),
            make_active_transfer_opacity_control(points, active_id),
        ]

    transfer_controls = html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "6px",
            **({} if is_scalar_volume else {"display": "none"})
        },
        children=[
            html.Div(
                style={"display": "flex", "alignItems": "center", "justifyContent": "space-between"},
                children=[
                    html.Div("Transfer map", style={"fontSize": "12px", "fontWeight": "600"}),
                    html.Button(
                        "Add point",
                        id="tf-add-point",
                        n_clicks=0,
                        style={"background": "#0EA5E9", "color": "white",
                               "border": "none", "borderRadius": "6px", **button_compact},
                    ),
                ],
            ),
            html.Div(
                f"{scalar_min:.3g} to {scalar_max:.3g}",
                style={"fontSize": "11px", "opacity": "0.7"},
            ),
            dcc.Graph(
                id="transfer-map",
                figure=make_transfer_figure(
                    default_transfer_points,
                    default_transfer_points[0]["id"],
                    DEFAULT_TRANSFER_COLOR_SCHEME,
                ),
                config={"displayModeBar": False},
                className="transfer-map-host",
                style={"height": "132px"},
            ),
            html.Div("Color scheme", style={"fontSize": "11px", "opacity": "0.7"}),
            dcc.Dropdown(
                id="transfer-color-scheme",
                options=TRANSFER_COLOR_OPTIONS,
                value=DEFAULT_TRANSFER_COLOR_SCHEME,
                clearable=False,
                searchable=False,
                style={"fontSize": "12px", "color": "#0f172a"},
            ),
            html.Div(
                style={
                    "display": "grid",
                    "gridTemplateColumns": "1fr 78px 26px",
                    "gap": "6px",
                    "fontSize": "10px",
                    "opacity": "0.7",
                },
                children=["value", "opacity", ""],
            ),
            html.Div(
                id="transfer-editor",
                style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                children=make_transfer_editor(default_transfer_points, default_transfer_points[0]["id"]),
            ),
        ],
    )
    default_volume_opacity = 0.65 if is_scalar_volume else (0.45 if is_vti_viewer else 0.12)

    volume_opacity_control = html.Div(
        style={"display": "none"} if is_scalar_volume else {
            "display": "flex",
            "flexDirection": "column",
            "gap": "6px",
        },
        children=[
            html.Div("Volume opacity", style={"fontSize": "12px", "opacity": "0.85"}),
            dcc.Slider(
                id="volume-opacity",
                min=0.01,
                max=0.9,
                step=0.01,
                value=default_volume_opacity,
                updatemode="drag",
                marks=None
            ),
        ],
    )

    render_controls = html.Div(
        style={
            "display": "flex",
            "flexDirection": "column",
            "gap": "6px",
            **({} if is_vti_viewer else {"display": "none"})
        },
        children=[
            volume_opacity_control,
            transfer_controls,
        ],
    )
    layer_controls_header = [] if is_scalar_volume else [
        html.Div("Labels" if is_vti_viewer else "Masks", style={"fontWeight": "600"})
    ]
    layer_list_style = {
        "maxHeight": "260px",
        "overflowY": "auto",
        "display": "flex",
        "flexDirection": "column",
        "gap": "4px",
        "paddingRight": "4px",
    }
    if is_scalar_volume:
        layer_list_style["display"] = "none"
    layer_control_rows = [
        html.Div(
            style={"display": "flex", "alignItems": "center", "gap": "5px",
                   "minWidth": "0"},
            children=[
                dcc.Checklist(
                    id={"type": "label-vis", "index": lid},
                    options=[{"label": "", "value": "on"}],
                    value=["on"] if lid in default_visible else [],
                    style={"flexShrink": "0"}
                ),
                html.Div(
                    style={"width": "12px", "height": "12px",
                           "borderRadius": "2px", "flexShrink": "0",
                           "backgroundColor": layer_states[lid]["color"],
                           "border": "1px solid #374151"}
                ),
                html.Span(
                    layer_states[lid]["name"],
                    title=layer_states[lid]["name"],
                    style={"fontSize": "11px", "flex": "1",
                           "overflow": "hidden", "textOverflow": "ellipsis",
                           "whiteSpace": "nowrap", "minWidth": "0"}
                ),
                html.Div(
                    dcc.Slider(
                        id={"type": "label-opacity", "index": lid},
                        min=0.0, max=1.0, step=0.05,
                        value=layer_states[lid]["opacity"],
                        updatemode="drag", marks=None,
                        tooltip={"placement": "bottom", "always_visible": False}
                    ),
                    style={"width": "70px", "flexShrink": "0"}
                ),
            ]
        )
        for lid in layer_states
    ]
    slider_css = f"""
    .slider-x .rc-slider-rail {{ background-color: #1f2937; }}
    .slider-y .rc-slider-rail {{ background-color: #1f2937; }}
    .slider-z .rc-slider-rail {{ background-color: #1f2937; }}

    .slider-x .rc-slider-track {{ background-color: {COLOR_SCHEME['slice_x']}; }}
    .slider-y .rc-slider-track {{ background-color: {COLOR_SCHEME['slice_y']}; }}
    .slider-z .rc-slider-track {{ background-color: {COLOR_SCHEME['slice_z']}; }}

    .slider-x .rc-slider-handle {{ border-color: {COLOR_SCHEME['slice_x']}; background-color: {COLOR_SCHEME['slice_x']}; }}
    .slider-y .rc-slider-handle {{ border-color: {COLOR_SCHEME['slice_y']}; background-color: {COLOR_SCHEME['slice_y']}; }}
    .slider-z .rc-slider-handle {{ border-color: {COLOR_SCHEME['slice_z']}; background-color: {COLOR_SCHEME['slice_z']}; }}

    .slider-x .rc-slider-handle:focus,
    .slider-x .rc-slider-handle:hover,
    .slider-x .rc-slider-handle:active {{ box-shadow: 0 0 0 2px rgba(59, 130, 246, 0.35); }}

    .slider-y .rc-slider-handle:focus,
    .slider-y .rc-slider-handle:hover,
    .slider-y .rc-slider-handle:active {{ box-shadow: 0 0 0 2px rgba(34, 197, 94, 0.35); }}

    .slider-z .rc-slider-handle:focus,
    .slider-z .rc-slider-handle:hover,
    .slider-z .rc-slider-handle:active {{ box-shadow: 0 0 0 2px rgba(245, 158, 11, 0.35); }}

    button,
    input[type="color"],
    .rc-slider-handle,
    .transfer-map-host .scatterlayer .points path {{ cursor: pointer; }}

    input[type="number"],
    input[type="text"] {{ cursor: text; }}
    """
    wheel_js = """
    (function() {
        var transferHoverPoint = null;
        var transferDragging = null;
        var transferLastEmit = 0;
        function attachWheel(id, axis) {
            var el = document.getElementById(id);
            if (!el || el.__wheelBound) return;
            el.__wheelBound = true;
            el.addEventListener("wheel", function(e) {
                if (!window.dash_clientside || !window.dash_clientside.set_props) return;
                e.preventDefault();
                var delta = e.deltaY > 0 ? 1 : -1;
                window.dash_clientside.set_props("wheel-store", {data: {axis: axis, delta: delta, ts: Date.now()}});
            }, {passive: false});
        }
        function init() {
            attachWheel("x-view", "x");
            attachWheel("y-view", "y");
            attachWheel("z-view", "z");
            attachTransferCursor();
        }
        function setCursor(el, cursor) {
            if (!el) return;
            el.style.cursor = cursor;
            var svg = el.querySelector(".main-svg");
            if (svg) svg.style.cursor = cursor;
        }
        function activeTransferId(el) {
            var trace = el && el.data && el.data[1];
            var ids = trace && trace.customdata;
            var sizes = trace && trace.marker && trace.marker.size;
            if (!ids || !sizes || !ids.length) return null;
            var active = null;
            var maxSize = -Infinity;
            for (var i = 0; i < ids.length; i += 1) {
                var size = Number(Array.isArray(sizes) ? sizes[i] : sizes);
                if (size > maxSize) {
                    maxSize = size;
                    active = ids[i];
                }
            }
            return active;
        }
        function transferIndexById(el, id) {
            var ids = el && el.data && el.data[1] && el.data[1].customdata;
            if (!ids) return -1;
            for (var i = 0; i < ids.length; i += 1) {
                if (ids[i] === id) return i;
            }
            return -1;
        }
        function isMovableTransferPoint(id) {
            return id && id !== "tf_min" && id !== "tf_max";
        }
        function hoveredTransferPoint(data) {
            var point = data && data.points && data.points[0];
            if (!point || point.curveNumber !== 1) return null;
            return {
                id: point.customdata,
                index: Number(point.pointIndex != null ? point.pointIndex : point.pointNumber)
            };
        }
        function axisRange(axis) {
            var range = (axis && axis.range) || [0, 1];
            var lo = Number(range[0]);
            var hi = Number(range[1]);
            if (!isFinite(lo) || !isFinite(hi) || lo === hi) return [0, 1];
            return lo < hi ? [lo, hi] : [hi, lo];
        }
        function pixelToTransferX(el, clientX) {
            var axis = el && el._fullLayout && el._fullLayout.xaxis;
            if (!axis) return null;
            var rect = el.getBoundingClientRect();
            var px = clientX - rect.left - (axis._offset || 0);
            var value = null;
            if (typeof axis.p2d === "function") {
                value = Number(axis.p2d(px));
            } else if (typeof axis.p2l === "function") {
                value = Number(axis.p2l(px));
            }
            if (!isFinite(value)) {
                var range = axisRange(axis);
                var width = Number(axis._length || 1);
                value = range[0] + (px / width) * (range[1] - range[0]);
            }
            var bounds = axisRange(axis);
            return Math.max(bounds[0], Math.min(bounds[1], value));
        }
        function currentTransferPoints(el, movingIndex, movingValue) {
            var trace = el && el.data && el.data[1];
            var ids = trace && trace.customdata;
            var xs = trace && trace.x;
            var ys = trace && trace.y;
            if (!ids || !xs || !ys) return [];
            var bounds = axisRange(el._fullLayout && el._fullLayout.xaxis);
            var points = [];
            for (var i = 0; i < ids.length; i += 1) {
                var id = ids[i];
                var value = Number(i === movingIndex ? movingValue : xs[i]);
                if (id === "tf_min") value = bounds[0];
                if (id === "tf_max") value = bounds[1];
                points.push({
                    id: String(id),
                    value: value,
                    opacity: Number(ys[i])
                });
            }
            return points;
        }
        function previewTransferPoint(el, movingIndex, movingValue) {
            var trace = el && el.data && el.data[1];
            if (!trace || !trace.x || !window.Plotly || !window.Plotly.restyle) return;
            var xs = Array.prototype.slice.call(trace.x);
            xs[movingIndex] = movingValue;
            var bounds = axisRange(el._fullLayout && el._fullLayout.xaxis);
            var span = Math.max(1e-12, bounds[1] - bounds[0]);
            var markerColors = xs.map(function(x) {
                return Math.max(0, Math.min(1, (Number(x) - bounds[0]) / span));
            });
            window.Plotly.restyle(el, {x: [xs], "marker.color": [markerColors]}, [1]);
        }
        function emitTransferPoints(el, movingIndex, movingValue, force) {
            if (!window.dash_clientside || !window.dash_clientside.set_props) return;
            var now = Date.now();
            if (!force && now - transferLastEmit < 45) return;
            transferLastEmit = now;
            window.dash_clientside.set_props("transfer-store", {
                data: currentTransferPoints(el, movingIndex, movingValue)
            });
        }
        function attachTransferCursor() {
            var el = document.getElementById("transfer-map");
            if (!el || el.__transferCursorBound || !el.on) return;
            el.__transferCursorBound = true;
            el.on("plotly_hover", function(data) {
                transferHoverPoint = hoveredTransferPoint(data);
                if (transferHoverPoint) {
                    var activeId = activeTransferId(el);
                    var canDrag = transferHoverPoint.id === activeId && isMovableTransferPoint(activeId);
                    setCursor(el, canDrag ? "ew-resize" : "pointer");
                }
            });
            el.on("plotly_unhover", function() {
                transferHoverPoint = null;
                if (!transferDragging) setCursor(el, "default");
            });
            el.addEventListener("mousedown", function(e) {
                if (e.button !== 0 || !transferHoverPoint) return;
                var activeId = activeTransferId(el);
                if (transferHoverPoint.id !== activeId || !isMovableTransferPoint(activeId)) return;
                transferDragging = {id: activeId};
                setCursor(el, "ew-resize");
                if (window.dash_clientside && window.dash_clientside.set_props) {
                    window.dash_clientside.set_props("active-transfer-point", {data: activeId});
                }
                e.preventDefault();
                e.stopPropagation();
            });
        }
        document.addEventListener("mousemove", function(e) {
            if (!transferDragging) return;
            var el = document.getElementById("transfer-map");
            var idx = transferIndexById(el, transferDragging.id);
            var value = pixelToTransferX(el, e.clientX);
            if (!el || idx < 0 || value == null) {
                transferDragging = null;
                return;
            }
            previewTransferPoint(el, idx, value);
            emitTransferPoints(el, idx, value, false);
            e.preventDefault();
        });
        document.addEventListener("mouseup", function(e) {
            if (!transferDragging) return;
            var el = document.getElementById("transfer-map");
            var idx = transferIndexById(el, transferDragging.id);
            var value = pixelToTransferX(el, e.clientX);
            if (el && idx >= 0 && value != null) {
                previewTransferPoint(el, idx, value);
                emitTransferPoints(el, idx, value, true);
            }
            transferDragging = null;
            setCursor(el, "default");
        });
        document.addEventListener("DOMContentLoaded", init);
        setInterval(init, 1000);
    })();
    """
    app.index_string = f"""
    <!DOCTYPE html>
    <html>
        <head>
            {{%metas%}}
            <title>{{%title%}}</title>
            {{%favicon%}}
            {{%css%}}
            <style>
            {slider_css}
            </style>
        </head>
        <body>
            {{%app_entry%}}
            <footer>
                {{%config%}}
                {{%scripts%}}
                {{%renderer%}}
                <script>
                {wheel_js}
                </script>
            </footer>
        </body>
    </html>
    """

    app_shell_base = {
        "display": "grid",
        "gridTemplateColumns": "300px 1fr",
        "gap": "10px",
        "height": "100vh",
        "backgroundColor": "#0f1115",
        "color": "#e6e6e6",
        "padding": "10px",
        "position": "relative",
        "overflow": "hidden",
    }
    sidebar_base = {
        "backgroundColor": "#151922",
        "borderRadius": "10px",
        "padding": "12px",
        "display": "flex",
        "flexDirection": "column",
        "gap": "10px",
        "minHeight": "0",
        "overflowY": "auto",
    }
    floating_toggle_base = {
        "position": "absolute",
        "top": "14px",
        "left": "14px",
        "zIndex": "12",
        "height": "32px",
        "padding": "4px 10px",
        "fontSize": "12px",
        "background": "#0EA5E9",
        "color": "white",
        "border": "none",
        "borderRadius": "6px",
        "cursor": "pointer",
        "display": "none",
    }

    app.layout = html.Div(
        id="app-shell",
        style=app_shell_base,
        children=[
            html.Div(
                id="sidebar-panel",
                style=sidebar_base,
                children=[
                    html.Div(
                        style={
                            "display": "grid",
                            "gridTemplateColumns": "auto 1fr",
                            "gap": "8px",
                            "alignItems": "center",
                        },
                        children=[
                            html.Button(
                                "Hide panel",
                                id="sidebar-toggle",
                                n_clicks=0,
                                style={
                                    "background": "#1f2937",
                                    "color": "#e5e7eb",
                                    "border": "1px solid #2b3347",
                                    "borderRadius": "6px",
                                    **button_compact,
                                },
                            ),
                            dcc.Checklist(
                                id="sidebar-pin",
                                options=[{"label": " Pin", "value": "on"}],
                                value=["on"],
                                inputStyle={"marginRight": "6px"},
                                style={"fontSize": "12px", "justifySelf": "end"},
                            ),
                            html.Div("Width", style={"fontSize": "11px", "opacity": "0.7"}),
                            dcc.Slider(
                                id="sidebar-width",
                                min=260,
                                max=560,
                                step=10,
                                value=300,
                                marks=None,
                                updatemode="drag",
                            ),
                        ],
                    ),
                    html.Div(
                        style=panel_card,
                        children=[
                            *layer_controls_header,
                            render_controls,
                            html.Div(
                                id="label-list",
                                style=layer_list_style,
                                children=layer_control_rows
                            ),
                            dcc.Store(
                                id="vis-store",
                                data={lid: lid in default_visible for lid in layer_states}
                            ),
                            dcc.Store(
                                id="opacity-store",
                                data={lid: layer_states[lid]["opacity"] for lid in layer_states}
                            ),
                            dcc.Store(
                                id="transfer-store",
                                data=default_transfer_points
                            ),
                            dcc.Store(
                                id="active-transfer-point",
                                data=default_transfer_points[0]["id"]
                            ),
                        ]
                    ),
                    html.Div(
                        style=panel_card,
                        children=[
                            html.Div("Slices", style={"fontWeight": "600"}),
                            html.Label("X-slice (i)"),
                            dcc.Slider(id="x-slider", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag", marks=None, className="slider-x"),
                            html.Label("Y-slice (j)"),
                            dcc.Slider(id="y-slider", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag", marks=None, className="slider-y"),
                            html.Label("Z-slice (k)"),
                            dcc.Slider(id="z-slider", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag", marks=None, className="slider-z"),
                        ]
                    ),
                    html.Hr(),
                    dcc.Tabs(id="tools-tabs", value="roi", children=[
                        dcc.Tab(label="ROI", value="roi", children=[
                            html.Div([
                                html.Div("ROI (sphere)", style={"fontWeight": 600, "opacity": 0.9}),
                                dcc.Tabs(id="roi-mode-tabs", value="center", children=[
                                    dcc.Tab(label="Center + Radius", value="center", children=[
                                        html.Label("ROI X (i)"),
                                        dcc.Slider(id="roi-x", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag"),
                                        html.Label("ROI Y (j)"),
                                        dcc.Slider(id="roi-y", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                                        html.Label("ROI Z (k)"),
                                        dcc.Slider(id="roi-z", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
                                        html.Label("ROI Radius (voxels)"),
                                        dcc.Slider(id="roi-r", min=0, max=int(max(Nx, Ny, Nz)//2), step=1, value=1, updatemode="drag",
                                                   tooltip={"always_visible": False, "placement": "bottom"}),
                                    ]),
                                    dcc.Tab(label="Two Points", value="twopoints", children=[
                                        html.Div("Point A", style={"fontWeight": 600, "opacity": 0.9, "marginTop": "6px"}),
                                        html.Label("A: i"),
                                        dcc.Slider(id="pA-i", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag"),
                                        html.Label("A: j"),
                                        dcc.Slider(id="pA-j", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                                        html.Label("A: k"),
                                        dcc.Slider(id="pA-k", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
                                        html.Div("Point B", style={"fontWeight": 600, "opacity": 0.9, "marginTop": "6px"}),
                                        html.Label("B: i"),
                                        dcc.Slider(id="pB-i", min=0, max=Nx-1, step=1, value=x_mid+1 if x_mid+1 < Nx else x_mid, updatemode="drag"),
                                        html.Label("B: j"),
                                        dcc.Slider(id="pB-j", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag"),
                                        html.Label("B: k"),
                                        dcc.Slider(id="pB-k", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag"),
                                        html.Div("Note: Sphere center = midpoint(A,B); radius = half distance(A,B)",
                                                 style={"fontSize": "12px", "opacity": 0.8, "marginTop": "6px"}),
                                    ])
                                ])
                            ], style={"padding": "6px"})
                        ]),
                        dcc.Tab(label="Operations", value="ops", children=[
                            html.Div([
                                html.Div("Morphology Mode", style={"fontWeight": 600, "opacity": 0.9}),
                                dcc.RadioItems(
                                    id="morph-ui-mode",
                                    options=[
                                        {"label": " Manual", "value": "manual"},
                                        {"label": " Automated (percent diameter stenosis)", "value": "auto"},
                                    ],
                                    value="manual",
                                    labelStyle={"display": "block", "marginBottom": "4px"}
                                ),
                                html.Div(id="ops-manual", children=[
                                    dcc.RadioItems(
                                        id="morph-mode",
                                        options=[
                                            {"label": " Dilate", "value": "dilate"},
                                            {"label": " Erode (shrink)", "value": "shrink"},
                                        ],
                                        value="dilate",
                                        labelStyle={"display": "block", "marginBottom": "4px"}
                                    ),
                                    html.Label("Iterations"),
                                    dcc.Input(id="morph-iters", type="number", min=1, step=1, value=1,
                                              style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                     "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                    html.Button("Apply Morphology", id="apply-morph",
                                                style={"background": "#16A34A", "color": "white",
                                                       "border": "none", "padding": "6px 10px",
                                                       "borderRadius": "6px", "marginTop": "6px"}),
                                ], style={"marginTop": "8px"}),
                                html.Div(id="ops-auto", children=[
                                    html.Label("Percent Diameter Stenosis (%)"),
                                    dcc.Input(id="stenosis-pct", type="number", min=1, max=99, step=1, value=50,
                                              style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                     "border": "1px solid #2a2f3a", "borderRadius": "6px", "padding": "4px 6px"}),
                                    html.Button("Run Automated Stenosis", id="apply-auto",
                                                style={"background": "#F59E0B", "color": "black",
                                                       "border": "none", "padding": "6px 10px",
                                                       "borderRadius": "6px", "marginTop": "6px"}),
                                ], style={"marginTop": "8px"}),
                            ], style={"padding": "6px"})
                        ]),
                    ]),
                    html.Div(id="status", style={"fontSize": "12px", "marginTop": "8px"})
                ]
            ),
            html.Div(
                id="plot-panel",
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                       "gridTemplateRows": "1fr 1fr", "gap": "10px"},
                children=[
                    dcc.Graph(id="x-view"), dcc.Graph(id="y-view"),
                    dcc.Graph(id="z-view"), dcc.Graph(id="threeD-view"),
                    dcc.Store(id="camera-store"),
                    dcc.Store(id="wheel-store"),
                    dcc.Store(id="hover-store"),
                    dcc.Store(id="catch-store"),
                    dcc.Store(id="fakect-store")
                ]
            ),
            html.Button(
                "Tools",
                id="sidebar-toggle-floating",
                n_clicks=0,
                style=floating_toggle_base,
            )
        ]
    )

    def refresh_layer_trace(layer_id: str) -> None:
        state = layer_states[layer_id]
        state["trace"] = mask_to_trace(
            state["current"],
            grid,
            state["color"],
            state["name"],
            state["opacity"]
        )

    def apply_exclusive_conflicts(target_id: str) -> None:
        target_state = layer_states.get(target_id)
        if not target_state:
            return
        group = target_state.get("exclusive_group")
        if not group:
            return
        occupied = target_state["current"] > 0
        for layer_id, state in layer_states.items():
            if layer_id == target_id or state.get("exclusive_group") != group:
                continue
            if np.any(state["current"][occupied]):
                state["current"][occupied] = 0
                refresh_layer_trace(layer_id)

    def batch_layer_snapshot_with_target(
        target_id: str,
        center_idx: tuple[int, int, int],
        radius_vox: int,
        scale_factor: float,
        shape_k_value: float,
        shape_window_value: tuple[float, float],
    ) -> dict[str, np.ndarray]:
        snapshot = {
            layer_id: state["original"].copy()
            for layer_id, state in layer_states.items()
        }
        if target_id not in snapshot:
            return snapshot
        snapshot[target_id] = apply_roi_scale(
            snapshot[target_id],
            center_idx,
            radius_vox,
            scale_factor,
            shape_k=shape_k_value,
            shape_window=shape_window_value
        )
        group = layer_states[target_id].get("exclusive_group")
        if group:
            occupied = snapshot[target_id] > 0
            for layer_id, state in layer_states.items():
                if layer_id != target_id and state.get("exclusive_group") == group:
                    snapshot[layer_id][occupied] = 0
        return snapshot

    @app.callback(
        Output("x-slider", "value"),
        Output("y-slider", "value"),
        Output("z-slider", "value"),
        Input("wheel-store", "data"),
        State("x-slider", "value"),
        State("y-slider", "value"),
        State("z-slider", "value"),
        prevent_initial_call=True
    )
    def update_sliders_from_wheel(data, x_idx, y_idx, z_idx):
        if not data:
            raise PreventUpdate
        axis = data.get("axis")
        try:
            delta = int(data.get("delta", 0))
        except (TypeError, ValueError):
            raise PreventUpdate
        if delta == 0:
            raise PreventUpdate

        if axis == "x":
            x_idx = int(np.clip(x_idx + delta, 0, Nx - 1))
        elif axis == "y":
            y_idx = int(np.clip(y_idx + delta, 0, Ny - 1))
        elif axis == "z":
            z_idx = int(np.clip(z_idx + delta, 0, Nz - 1))
        else:
            raise PreventUpdate

        return x_idx, y_idx, z_idx

    # ---- Per-label visibility and opacity stores ----

    try:
        from dash import ALL as _ALL
    except ImportError:
        _ALL = "ALL"

    @app.callback(
        Output("app-shell", "style"),
        Output("sidebar-panel", "style"),
        Output("sidebar-toggle-floating", "style"),
        Output("sidebar-toggle", "children"),
        Input("sidebar-toggle", "n_clicks"),
        Input("sidebar-toggle-floating", "n_clicks"),
        Input("sidebar-width", "value"),
        Input("sidebar-pin", "value"),
        prevent_initial_call=False
    )
    def update_sidebar_layout(toggle_clicks, floating_clicks, sidebar_width, pin_values):
        try:
            width = int(sidebar_width)
        except (TypeError, ValueError):
            width = 300
        width = int(np.clip(width, 260, 560))
        toggle_clicks = int(toggle_clicks or 0)
        floating_clicks = int(floating_clicks or 0)
        is_open = toggle_clicks <= floating_clicks
        is_pinned = "on" in (pin_values or [])

        shell_style = dict(app_shell_base)
        sidebar_style = dict(sidebar_base)
        floating_style = dict(floating_toggle_base)

        if not is_open:
            shell_style["gridTemplateColumns"] = "0px 1fr"
            sidebar_style["display"] = "none"
            floating_style["display"] = "block"
            return shell_style, sidebar_style, floating_style, "Show panel"

        floating_style["display"] = "none"
        if is_pinned:
            shell_style["gridTemplateColumns"] = f"{width}px 1fr"
            sidebar_style["width"] = "auto"
            sidebar_style["position"] = "relative"
        else:
            shell_style["gridTemplateColumns"] = "0px 1fr"
            sidebar_style.update({
                "position": "absolute",
                "top": "10px",
                "left": "10px",
                "bottom": "10px",
                "width": f"{width}px",
                "zIndex": "10",
                "boxShadow": "0 16px 40px rgba(0, 0, 0, 0.42)",
            })
        return shell_style, sidebar_style, floating_style, "Hide panel"

    @app.callback(
        Output("vis-store", "data"),
        Input({"type": "label-vis", "index": _ALL}, "value"),
        State({"type": "label-vis", "index": _ALL}, "id"),
        State("vis-store", "data"),
        prevent_initial_call=True
    )
    def update_vis_store(values, ids, current_vis):
        triggered = callback_context.triggered or []
        if not triggered:
            raise PreventUpdate
        result = dict(current_vis or {})
        for id_dict, val in zip(ids or [], values or []):
            lid = id_dict["index"]
            result[lid] = "on" in (val or [])
        return result

    @app.callback(
        Output("opacity-store", "data"),
        Input({"type": "label-opacity", "index": _ALL}, "value"),
        State({"type": "label-opacity", "index": _ALL}, "id"),
        State("opacity-store", "data"),
        prevent_initial_call=True
    )
    def update_opacity_store(values, ids, current_opacity):
        result = dict(current_opacity or {})
        for id_dict, val in zip(ids or [], values or []):
            lid = id_dict["index"]
            if val is not None:
                result[lid] = float(np.clip(val, 0.0, 1.0))
        return result

    @app.callback(
        Output("active-transfer-point", "data"),
        Input("transfer-map", "clickData"),
        Input({"type": "tf-row", "index": _ALL}, "n_clicks"),
        State("transfer-store", "data"),
        State({"type": "tf-row", "index": _ALL}, "id"),
        State("active-transfer-point", "data"),
        prevent_initial_call=True
    )
    def update_active_transfer_point(click_data, row_clicks, transfer_points, row_ids, current_active):
        if not is_scalar_volume:
            raise PreventUpdate
        points = _sanitize_transfer_points(transfer_points, scalar_min, scalar_max)
        current_active = _resolved_active_transfer_id(points, current_active)
        triggered = callback_context.triggered or []
        if not triggered:
            raise PreventUpdate
        prop = triggered[0]["prop_id"]

        if prop.startswith("transfer-map"):
            clicked_points = (click_data or {}).get("points") or []
            if not clicked_points:
                raise PreventUpdate
            point = clicked_points[0]
            if point.get("curveNumber") != 1:
                raise PreventUpdate
            custom = point.get("customdata")
            if custom:
                resolved = _resolved_active_transfer_id(points, custom)
                if resolved == current_active:
                    raise PreventUpdate
                return resolved
            try:
                point_index = int(point.get("pointIndex", point.get("pointNumber")))
            except (TypeError, ValueError):
                raise PreventUpdate
            if 0 <= point_index < len(points):
                resolved = points[point_index]["id"]
                if resolved == current_active:
                    raise PreventUpdate
                return resolved
            raise PreventUpdate

        if "\"type\":\"tf-row\"" in prop or "'type': 'tf-row'" in prop:
            raw_id = prop.rsplit(".", 1)[0]
            try:
                clicked_id = json.loads(raw_id).get("index")
            except Exception:
                clicked_id = None
            if clicked_id:
                resolved = _resolved_active_transfer_id(points, clicked_id)
                if resolved == current_active:
                    raise PreventUpdate
                return resolved
            raise PreventUpdate

        raise PreventUpdate

    @app.callback(
        Output("transfer-store", "data"),
        Input("tf-add-point", "n_clicks"),
        Input({"type": "tf-opacity", "index": _ALL}, "value"),
        Input("active-transfer-opacity", "value"),
        Input({"type": "tf-remove", "index": _ALL}, "n_clicks"),
        State({"type": "tf-opacity", "index": _ALL}, "id"),
        State("active-transfer-point", "data"),
        State({"type": "tf-remove", "index": _ALL}, "id"),
        State("transfer-store", "data"),
        prevent_initial_call=True
    )
    def update_transfer_store(add_clicks, opacities, active_opacity, remove_clicks,
                              opacity_ids, active_id, remove_ids, current_points):
        if not is_scalar_volume:
            raise PreventUpdate

        triggered = callback_context.triggered or []
        if not triggered:
            raise PreventUpdate
        prop = triggered[0]["prop_id"]

        points = _sanitize_transfer_points(current_points, scalar_min, scalar_max)
        by_id = {point["id"]: dict(point) for point in points}
        changed = False

        for id_dict, opacity in zip(opacity_ids or [], opacities or []):
            pid = id_dict.get("index")
            if pid is None:
                continue
            point = by_id.get(pid, {
                "id": pid,
                "value": scalar_min,
                "opacity": 0.1,
            })
            try:
                next_opacity = float(np.clip(float(opacity), 0.0, 1.0))
            except (TypeError, ValueError):
                next_opacity = float(np.clip(point.get("opacity", 0.1), 0.0, 1.0))
            if not np.isclose(float(point.get("opacity", 0.1)), next_opacity):
                changed = True
            point["opacity"] = next_opacity
            by_id[pid] = point

        if prop.startswith("active-transfer-opacity"):
            active_id = _resolved_active_transfer_id(points, active_id)
            if active_id in by_id:
                try:
                    next_opacity = float(np.clip(float(active_opacity), 0.0, 1.0))
                except (TypeError, ValueError):
                    next_opacity = float(np.clip(by_id[active_id].get("opacity", 0.1), 0.0, 1.0))
                if not np.isclose(float(by_id[active_id].get("opacity", 0.1)), next_opacity):
                    changed = True
                by_id[active_id]["opacity"] = next_opacity

        points = _sanitize_transfer_points(list(by_id.values()), scalar_min, scalar_max)

        if prop.startswith("tf-add-point"):
            sorted_points = sorted(points, key=lambda p: p["value"])
            gaps = [
                (sorted_points[i + 1]["value"] - sorted_points[i]["value"], i)
                for i in range(len(sorted_points) - 1)
            ]
            if gaps:
                _, gap_idx = max(gaps, key=lambda item: item[0])
                left = sorted_points[gap_idx]
                right = sorted_points[gap_idx + 1]
                new_value = 0.5 * (left["value"] + right["value"])
                new_opacity = 0.5 * (left["opacity"] + right["opacity"])
            else:
                new_value = 0.5 * (scalar_min + scalar_max)
                new_opacity = 0.1
            seed = int(add_clicks or 0)
            new_id = f"tf_user_{seed}_{len(points)}"
            existing = {point["id"] for point in points}
            while new_id in existing:
                seed += 1
                new_id = f"tf_user_{seed}_{len(points)}"
            points.append({
                "id": new_id,
                "value": float(np.clip(new_value, scalar_min, scalar_max)),
                "opacity": float(np.clip(new_opacity, 0.0, 1.0)),
            })
            return _sanitize_transfer_points(points, scalar_min, scalar_max)

        if "\"type\":\"tf-remove\"" in prop or "'type': 'tf-remove'" in prop:
            raw_id = prop.rsplit(".", 1)[0]
            remove_id = None
            try:
                remove_id = json.loads(raw_id).get("index")
            except Exception:
                for id_dict, clicks in zip(remove_ids or [], remove_clicks or []):
                    if clicks:
                        remove_id = id_dict.get("index")
                        break
            if remove_id and len(points) > 2:
                points = [point for point in points if point["id"] != remove_id]
            return _sanitize_transfer_points(points, scalar_min, scalar_max)

        if not changed:
            raise PreventUpdate
        return _sanitize_transfer_points(points, scalar_min, scalar_max)

    @app.callback(
        Output("transfer-map", "figure"),
        Output("transfer-editor", "children"),
        Input("transfer-store", "data"),
        Input("active-transfer-point", "data"),
        Input("transfer-color-scheme", "value"),
        prevent_initial_call=False
    )
    def render_transfer_editor(points, active_id, color_scheme):
        clean_points = _sanitize_transfer_points(points, scalar_min, scalar_max)
        active_id = _resolved_active_transfer_id(clean_points, active_id)
        return (
            make_transfer_figure(clean_points, active_id, color_scheme),
            make_transfer_editor(clean_points, active_id),
        )

    @app.callback(
        Output("roi-scale-pos", "value"),
        Output("roi-scale-input", "value"),
        Output("roi-scale-display", "children"),
        Input("roi-scale-pos", "value"),
        Input("roi-scale-input", "value"),
        prevent_initial_call=False
    )
    def sync_roi_scale(scale_pos, scale_input):
        triggered = callback_context.triggered or []
        if not triggered:
            raise PreventUpdate
        src = triggered[0]["prop_id"].split(".")[0]

        if src == "roi-scale-input":
            try:
                val = float(scale_input)
            except (TypeError, ValueError):
                val = _pos_to_scale(scale_pos if scale_pos is not None else 0.5)
            val = max(0.001, min(99.999, val))
            display_val = val
            return float(_scale_to_pos(val)), float(val), f"Current: {display_val:.3f}"

        pos = scale_pos if scale_pos is not None else 0.5
        display_val = _pos_to_scale(pos)
        return float(pos), float(min(99.999, max(0.001, display_val))), f"Current: {display_val:.3f}"

    @app.callback(
        Output("shape-window", "value"),
        Input("shape-window", "value"),
        Input("shape-symmetric", "value"),
        prevent_initial_call=True
    )
    def enforce_symmetric_window(window_vals, symmetric_vals):
        if not window_vals or len(window_vals) != 2:
            raise PreventUpdate
        symmetric = "on" in (symmetric_vals or [])
        if not symmetric:
            return window_vals
        left, right = window_vals
        left = max(0.0, min(1.0, float(left)))
        right = max(0.0, min(1.0, float(right)))
        half_width = abs(0.5 - left)
        new_left = max(0.0, 0.5 - half_width)
        new_right = min(1.0, 0.5 + half_width)
        return [new_left, new_right]

    @app.callback(
        Output("roi-batch-status", "children"),
        Input("roi-batch-export", "n_clicks"),
        State("p1-x", "value"),
        State("p1-y", "value"),
        State("p1-z", "value"),
        State("p2-x", "value"),
        State("p2-y", "value"),
        State("p2-z", "value"),
        State("roi-batch-scale-min", "value"),
        State("roi-batch-scale-max", "value"),
        State("roi-batch-scale-count", "value"),
        State("roi-batch-k-min", "value"),
        State("roi-batch-k-max", "value"),
        State("roi-batch-k-count", "value"),
        State("shape-window", "value"),
        State("roi-batch-out-dir", "value"),
        State("roi-batch-stl", "value"),
        State("target-label", "value"),
        prevent_initial_call=True
    )
    def export_batch_masks(n_clicks,
                           p1_x, p1_y, p1_z,
                           p2_x, p2_y, p2_z,
                           scale_min, scale_max, scale_count,
                           k_min, k_max, k_count,
                           shape_window, out_dir, export_stl_flags,
                           target_label):
        if not n_clicks:
            raise PreventUpdate

        center_idx = (
            int(np.clip(round((p1_x + p2_x) / 2.0), 0, Nx - 1)),
            int(np.clip(round((p1_y + p2_y) / 2.0), 0, Ny - 1)),
            int(np.clip(round((p1_z + p2_z) / 2.0), 0, Nz - 1))
        )
        radius_vox = int(round(0.5 * float(np.linalg.norm(
            np.array([p2_x - p1_x, p2_y - p1_y, p2_z - p1_z], dtype=float)
        ))))

        scales = _linspace_safe(scale_min, scale_max, scale_count)
        ks = _linspace_safe(k_min, k_max, k_count)
        w0, w1 = (0.25, 0.75)
        if shape_window and len(shape_window) == 2:
            try:
                w0, w1 = float(shape_window[0]), float(shape_window[1])
            except (TypeError, ValueError):
                w0, w1 = (0.25, 0.75)
        w0 = max(0.0, min(1.0, w0))
        w1 = max(0.0, min(1.0, w1))
        if w1 <= w0:
            w0, w1 = (0.25, 0.75)

        out_dir = out_dir or "outputs/stenosis_batch"
        out_path = Path(out_dir)
        out_path.mkdir(parents=True, exist_ok=True)

        total = 0
        stl_total = 0
        target_label = target_label if target_label in layer_states else target_default
        target_state = layer_states[target_label]
        for sf in scales:
            sf = max(0.001, min(99.999, float(sf)))
            for k in ks:
                k = max(0.1, float(k))
                snapshot = batch_layer_snapshot_with_target(
                    target_label,
                    center_idx,
                    radius_vox,
                    sf,
                    k,
                    (w0, w1)
                )
                base_name = (
                    f"{target_label}_sf{_fmt_param(sf)}_k{_fmt_param(k)}"
                    f"_w{_fmt_param(w0, 2)}-{_fmt_param(w1, 2)}.npz"
                )
                export_arrays = {
                    f"mask_{layer_id}": mask.astype(np.uint8)
                    for layer_id, mask in snapshot.items()
                }
                np.savez_compressed(
                    out_path / base_name,
                    **export_arrays,
                    spacing=np.array(grid["spacing"]),
                    origin=np.array(grid["origin"]),
                    layer_ids=np.array(list(snapshot.keys())),
                    layer_names=np.array([layer_states[layer_id]["name"] for layer_id in snapshot.keys()]),
                    target_label=np.array(target_label),
                    scale_factor=np.array(sf),
                    shape_k=np.array(k),
                    shape_window=np.array([w0, w1]),
                    roi_center_ijk=np.array(center_idx),
                    roi_radius_vox=np.array(radius_vox)
                )
                export_stl = "on" in (export_stl_flags or [])
                if export_stl:
                    stl_name = base_name.replace(".npz", ".stl")
                    if save_mask_stl(snapshot[target_label], grid, out_path / stl_name, name=target_state["name"]):
                        stl_total += 1
                total += 1

        if stl_total > 0:
            return (
                f"Export done. Wrote {total} label stacks and {stl_total} target STL files to {out_path} "
                f"(scales={len(scales)}, k={len(ks)})."
            )
        return (
            f"Export done. Wrote {total} label stacks to {out_path} "
            f"(scales={len(scales)}, k={len(ks)})."
        )

    @app.callback(
        Output("fakect-status", "children"),
        Output("fakect-store", "data"),
        Input("fakect-apply", "n_clicks"),
        State("fakect-model", "value"),
        State("fakect-model-path", "value"),
        State("fakect-mask-source", "value"),
        State("fakect-axis", "value"),
        State("fakect-context-step", "value"),
        prevent_initial_call=True
    )
    def apply_fakect(n_clicks, model_dropdown, model_path, mask_source, axis, context_step):
        if not n_clicks:
            raise PreventUpdate

        model_path = (model_path or "").strip() or (model_dropdown or "").strip()
        if not model_path:
            return "Select a fakenoise model path.", None

        path = Path(model_path).expanduser().resolve()
        if not path.exists():
            return f"Model not found: {path}", None

        try:
            import tensorflow as tf
        except Exception:
            return "TensorFlow is not available. Install tensorflow to run FakeCT.", None

        try:
            if fakect_state["model_path"] != str(path):
                fakect_state["model"] = tf.keras.models.load_model(str(path))
                fakect_state["model_path"] = str(path)
            model = fakect_state["model"]
            th, tw, c = _infer_keras_input(model)
        except Exception as e:
            return f"Failed to load model: {e}", None

        if mask_source == "on":
            mask_vol = on_u8
        elif mask_source == "out":
            mask_vol = out_u8
        else:
            mask_vol = inside_state["inside"]

        try:
            fakect_state["target_hw"] = (th, tw)
            fakect_state["input_channels"] = c
            fakect_state["context_step"] = int(context_step or 1)
            fakect_state["mask_source"] = mask_source or "inside"
            fakect_state["volume"] = _predict_fake_volume(
                mask_vol,
                model,
                (th, tw),
                c,
                axis=axis or "i",
                context_step=fakect_state["context_step"]
            )
        except Exception as e:
            return f"FakeCT inference failed: {e}", None

        channel_note = ""
        if c > 1 and (c - 1) % 2 != 0:
            channel_note = " (non-symmetric channel count)"
        axis_label = (axis or "i").lower()
        msg = f"FakeCT ready: {path.name} | axis={axis_label} | input={th}x{tw}x{c}{channel_note}"
        return msg, {"ts": time.time()}

    @app.callback(
        Output("x-view", "figure"),
        Output("y-view", "figure"),
        Output("z-view", "figure"),
        Output("threeD-view", "figure"),
        Output("camera-store", "data"),
        Input("vis-store", "data"),
        Input("opacity-store", "data"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("roi-mode-tabs", "value"),
        Input("roi-x", "value"),
        Input("roi-y", "value"),
        Input("roi-z", "value"),
        Input("roi-r", "value"),
        Input("pA-i", "value"),
        Input("pA-j", "value"),
        Input("pA-k", "value"),
        Input("pB-i", "value"),
        Input("pB-j", "value"),
        Input("pB-k", "value"),
        Input("tools-tabs", "value"),
        Input("morph-ui-mode", "value"),
        Input("morph-mode", "value"),
        Input("morph-iters", "value"),
        Input("apply-morph", "n_clicks"),
        Input("stenosis-pct", "value"),
        Input("apply-auto", "n_clicks"),
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=False
    )
    def update(mask_values, x_idx, y_idx, z_idx, roi_mode,
               roi_x, roi_y, roi_z, roi_r,
               pA_i, pA_j, pA_k, pB_i, pB_j, pB_k,
               tabs_value, ui_mode, morph_mode, morph_iters, apply_clicks,
               stenosis_pct, apply_auto_clicks, n_clicks):
        camera_state = None
        uirev = "viewer"
        sphere_color = "#E11D48"
        sphere_opacity = float(OPACITY_LEVELS.get("sphere_opacity", 0.05))
        fakect_show = []
        fakect_alpha = 0.6

        triggered = [t["prop_id"] for t in (callback_context.triggered or [])]
        if "reset-btn.n_clicks" in triggered:
            mask_values = ["inside", "on", "out"]
            x_idx, y_idx, z_idx = x_mid, y_mid, z_mid
            roi_x, roi_y, roi_z, roi_r = x_mid, y_mid, z_mid, 1
            pA_i, pA_j, pA_k = x_mid, y_mid, z_mid
            pB_i, pB_j, pB_k = min(x_mid+1, Nx-1), y_mid, z_mid
            # Reset the working inside mask
            inside_state["inside"] = inside_state["original"].copy()

        # Apply morphology when requested
        if "apply-morph.n_clicks" in triggered and (ui_mode == "manual"):
            try:
                iters = int(morph_iters) if morph_iters is not None else 1
                iters = max(1, iters)
            except Exception:
                iters = 1
            mode = morph_mode if morph_mode in ("dilate", "shrink") else "dilate"
            # Update only the inside portion within ROI, keep others untouched, then recombine
            inside_state["inside"] = apply_roi_morphology(
                inside_state["inside"], roi_x, roi_y, roi_z, roi_r, iterations=iters, mode=mode
            )

        # Determine effective ROI center+radius depending on ROI mode
        if roi_mode == "twopoints":
            # Midpoint center and half-distance radius (in voxels)
            cx = int(np.round((int(pA_i) + int(pB_i)) / 2))
            cy = int(np.round((int(pA_j) + int(pB_j)) / 2))
            cz = int(np.round((int(pA_k) + int(pB_k)) / 2))
            dx = int(pA_i) - int(pB_i)
            dy = int(pA_j) - int(pB_j)
            dz = int(pA_k) - int(pB_k)
            dist = int(np.round(np.sqrt(dx*dx + dy*dy + dz*dz)))
            eff_roi_x, eff_roi_y, eff_roi_z = cx, cy, cz
            eff_roi_r = max(1, dist // 2)
        else:
            eff_roi_x, eff_roi_y, eff_roi_z, eff_roi_r = int(roi_x), int(roi_y), int(roi_z), int(roi_r)

        # Automated stenosis: shrink ROI radius using a sparse sigmoidal tick schedule
        if "apply-auto.n_clicks" in triggered and (ui_mode == "auto"):
            try:
                pct = float(stenosis_pct)
            except Exception:
                pct = 50.0
            pct = max(1.0, min(99.0, pct))

            target_r = int(max(0, np.floor(eff_roi_r * (pct / 100.0))))
            start_r  = int(eff_roi_r)
            steps = max(0, start_r - target_r)  # number of single-iteration erosions needed
            auto_status = ""
            # Safety threshold: do not erode below 10% of the initial in-ROI voxels
            safety_frac = 0.10
            base_roi_u8, _ = filter_inside_by_roi(inside_state["inside"], eff_roi_x, eff_roi_y, eff_roi_z, start_r)
            base_count = int(np.sum(base_roi_u8))
            safe_min = max(1, int(np.floor(safety_frac * base_count)))

            if steps > 0:
                # Total time ticks: default to 2×steps (keeps overall duration similar)
                total_ticks = 2 * steps
                if total_ticks < steps:
                    total_ticks = steps
                apply_ticks = _sigmoid_sparse_apply_schedule(total_ticks, steps)

                current_r = start_r
                safety_triggered = False
                applied = 0
                for tick in range(total_ticks):
                    if current_r <= target_r:
                        break
                    if not apply_ticks[tick]:
                        continue  # skip this tick to throttle rate (no morphology this tick)

                    # Apply a single-iteration shrink confined to current ROI radius
                    inside_state["inside"] = apply_roi_morphology(
                        inside_state["inside"], eff_roi_x, eff_roi_y, eff_roi_z, current_r,
                        iterations=1, mode="shrink"
                    )
                    applied += 1

                    # Safety check: count current in-ROI voxels at this radius
                    cur_roi_u8, _ = filter_inside_by_roi(inside_state["inside"], eff_roi_x, eff_roi_y, eff_roi_z, current_r)
                    cur_count = int(np.sum(cur_roi_u8))
                    if cur_count < safe_min:
                        safety_triggered = True
                        auto_status = (
                            f" | Auto stopped early: r={current_r}, vox={cur_count} < {safe_min}, "
                            f"applied={applied}/{steps}, ticks={total_ticks}"
                        )
                        break

                    current_r -= 1

                if not safety_triggered:
                    auto_status = f" | Auto done: steps={applied}/{steps}, ticks={total_ticks}"
            # UI slider for roi-r remains user-controlled; internal loop uses shrinking radii

        active = []
        show_inside = "inside" in mask_values
        show_on     = "on"     in mask_values
        show_out    = "out"    in mask_values
        if show_inside:
            active.append(inside_state["inside"])
        if show_on:
            active.append(on_u8)
        if show_out:
            active.append(out_u8)

        s = float(grid["spacing"][0])
        ox, oy, oz = grid["origin"]
        center = (
            float(ox + (int(eff_roi_x) + 0.5) * s),
            float(oy + (int(eff_roi_y) + 0.5) * s),
            float(oz + (int(eff_roi_z) + 0.5) * s),
        )
        radius = float(max(0, int(eff_roi_r)) * s)

        transfer_points = _sanitize_transfer_points(transfer_data, scalar_min, scalar_max)
        scalar_colorscale = _transfer_colorscale(transfer_color_scheme) if is_scalar_volume else None
        scalar_visible = bool(is_scalar_volume and scalar_layer_id in visible_ids and scalar_data is not None)

        if scalar_visible:
            x_slice = scalar_data[:, :, x_idx]
            y_slice = scalar_data[:, y_idx, :]
            z_slice = scalar_data[z_idx, :, :]
            x_colors = y_colors = z_colors = []
        elif is_scalar_volume:
            x_slice = np.full((Nz, Ny), np.nan, dtype=np.float32)
            y_slice = np.full((Nz, Nx), np.nan, dtype=np.float32)
            z_slice = np.full((Ny, Nx), np.nan, dtype=np.float32)
            x_colors = y_colors = z_colors = []
        else:
            scalar_colorscale = None
            x_slice, x_colors = compose_layer_slice(layer_states, visible_ids, "x", x_idx)
            y_slice, y_colors = compose_layer_slice(layer_states, visible_ids, "y", y_idx)
            z_slice, z_colors = compose_layer_slice(layer_states, visible_ids, "z", z_idx)

        roi_x = _roi_slice_mask(grid, "x", x_idx, center, radius)
        roi_y = _roi_slice_mask(grid, "y", y_idx, center, radius)
        roi_z = _roi_slice_mask(grid, "z", z_idx, center, radius)

        x_fig = _slice_fig(
            x_slice,
            f"X-slice (i={x_idx})",
            COLOR_SCHEME["slice_x"],
            colorscale=_slice_colorscale(COLOR_SCHEME["slice_x"]),
            zmax=1
        )
        y_fig = _slice_fig(
            y_slice,
            f"Y-slice (j={y_idx})",
            COLOR_SCHEME["slice_y"],
            colorscale=_slice_colorscale(COLOR_SCHEME["slice_y"]),
            zmax=1
        )
        z_fig = _slice_fig(
            z_slice,
            f"Z-slice (k={z_idx})",
            COLOR_SCHEME["slice_z"],
            colorscale=_slice_colorscale(COLOR_SCHEME["slice_z"]),
            zmax=1
        )

        show_fakect = "on" in (fakect_show or [])
        fake_vol = fakect_state.get("volume")
        if show_fakect and isinstance(fake_vol, np.ndarray) and fake_vol.shape == inside_state["inside"].shape:
            alpha = max(0.0, min(1.0, float(fakect_alpha if fakect_alpha is not None else 0.6)))
            fake_x = fake_vol[:, :, x_idx]
            fake_y = fake_vol[:, y_idx, :]
            fake_z = fake_vol[z_idx, :, :]
            overlay = dict(colorscale=_binary_colorscale("#ffffff"), showscale=False, opacity=alpha, zmin=0, zmax=1, hoverinfo="skip")
            x_fig.add_trace(go.Heatmap(z=fake_x, **overlay))
            y_fig.add_trace(go.Heatmap(z=fake_y, **overlay))
            z_fig.add_trace(go.Heatmap(z=fake_z, **overlay))

        x_fig.add_trace(_roi_overlay_trace(roi_x, sphere_color, sphere_opacity))
        y_fig.add_trace(_roi_overlay_trace(roi_y, sphere_color, sphere_opacity))
        z_fig.add_trace(_roi_overlay_trace(roi_z, sphere_color, sphere_opacity))

        # Rebuild inside surface dynamically from current inside state
        dyn_inside_trace = mask_to_trace(inside_state["inside"], grid, "#3B82F6", "inside", 0.35) if show_inside else None

        valid_traces = [t for t in [mesh_trace,
                                    dyn_inside_trace,
                                    on_trace     if show_on     else None,
                                    out_trace    if show_out    else None] if t is not None]
        extent_x, extent_y, extent_z = grid["extent_mm"]
        aspectratio = _aspectratio_from_extents((extent_x, extent_y, extent_z))

        fig3d = go.Figure(data=valid_traces)
        plane_traces = [
            _slice_plane_surface(
                x_slice,
                f"X-slice (i={x_idx})",
                COLOR_SCHEME["slice_x"],
                colorscale=scalar_colorscale,
                zmin=scalar_min,
                zmax=scalar_max,
                showscale=False
            )
            y_fig = _slice_fig(
                y_slice,
                f"Y-slice (j={y_idx})",
                COLOR_SCHEME["slice_y"],
                colorscale=scalar_colorscale,
                zmin=scalar_min,
                zmax=scalar_max,
                showscale=False
            )
            z_fig = _slice_fig(
                z_slice,
                f"Z-slice (k={z_idx})",
                COLOR_SCHEME["slice_z"],
                colorscale=scalar_colorscale,
                zmin=scalar_min,
                zmax=scalar_max,
                showscale=False
            )
        else:
            x_fig = _slice_fig(
                x_slice,
                f"X-slice (i={x_idx})",
                COLOR_SCHEME["slice_x"],
                colorscale=_discrete_label_colorscale(x_colors),
                zmax=max(1, len(x_colors))
            )
            y_fig = _slice_fig(
                y_slice,
                f"Y-slice (j={y_idx})",
                COLOR_SCHEME["slice_y"],
                colorscale=_discrete_label_colorscale(y_colors),
                zmax=max(1, len(y_colors))
            )
            z_fig = _slice_fig(
                z_slice,
                f"Z-slice (k={z_idx})",
                COLOR_SCHEME["slice_z"],
                colorscale=_discrete_label_colorscale(z_colors),
                zmax=max(1, len(z_colors))
            )

        x_fig.add_trace(_roi_overlay_trace(roi_x, sphere_color, sphere_opacity))
        y_fig.add_trace(_roi_overlay_trace(roi_y, sphere_color, sphere_opacity))
        z_fig.add_trace(_roi_overlay_trace(roi_z, sphere_color, sphere_opacity))

        x_range, y_range, z_range = _grid_bounds_xyz(grid, pad_fraction=0.02)
        extent_x = x_range[1] - x_range[0]
        extent_y = y_range[1] - y_range[0]
        extent_z = z_range[1] - z_range[0]
        aspectratio = _aspectratio_from_extents((extent_x, extent_y, extent_z))

        fig3d = go.Figure(data=[mesh_trace] if mesh_trace is not None else [])
        if render_mode == "volume":
            try:
                volume_opacity = float(volume_opacity)
            except (TypeError, ValueError):
                volume_opacity = 0.12
            volume_opacity = max(0.01, min(0.9, volume_opacity))
            if scalar_visible:
                scalar_layer_opacity = float(np.clip(
                    opacity_data.get(scalar_layer_id, layer_states[scalar_layer_id]["opacity"]),
                    0.0,
                    1.0,
                ))
                trace = scalar_to_volume_trace(
                    scalar_data,
                    grid,
                    transfer_points,
                    scalar_range,
                    volume_opacity * scalar_layer_opacity,
                    transfer_color_scheme,
                    layer_states[scalar_layer_id]["name"]
                )
                if trace is not None:
                    fig3d.add_trace(trace)
            elif not is_scalar_volume:
                for layer_id in visible_ids:
                    trace = mask_to_volume_trace(
                        layer_states[layer_id]["current"],
                        grid,
                        layer_states[layer_id]["color"],
                        layer_states[layer_id]["name"],
                        float(np.clip(opacity_data.get(layer_id, layer_states[layer_id]["opacity"]), 0.0, 1.0)) * volume_opacity
                    )
                    if trace is not None:
                        fig3d.add_trace(trace)
        else:
            for layer_id in visible_ids:
                trace = layer_states[layer_id]["trace"]
                if trace is None:
                    continue
                # Apply current opacity from store (may differ from cached trace)
                current_opacity = opacity_data.get(layer_id, layer_states[layer_id]["opacity"])
                # Plotly Mesh3d / Scatter3d support opacity update via update_traces
                fig3d.add_trace(trace)
                fig3d.update_traces(
                    opacity=float(np.clip(current_opacity, 0.0, 1.0)),
                    selector=dict(name=layer_states[layer_id]["name"])
                )
            if is_scalar_volume:
                plane_traces = [
                    _slice_plane_surface(
                        x_slice,
                        grid,
                        "x",
                        x_idx,
                        COLOR_SCHEME["slice_x"],
                        colorscale=scalar_colorscale,
                        cmin=scalar_min,
                        cmax=scalar_max
                    ),
                    _slice_plane_surface(
                        y_slice,
                        grid,
                        "y",
                        y_idx,
                        COLOR_SCHEME["slice_y"],
                        colorscale=scalar_colorscale,
                        cmin=scalar_min,
                        cmax=scalar_max
                    ),
                    _slice_plane_surface(
                        z_slice,
                        grid,
                        "z",
                        z_idx,
                        COLOR_SCHEME["slice_z"],
                        colorscale=scalar_colorscale,
                        cmin=scalar_min,
                        cmax=scalar_max
                    )
                ]
            else:
                plane_traces = [
                    _slice_plane_surface(
                        x_slice,
                        grid,
                        "x",
                        x_idx,
                        COLOR_SCHEME["slice_x"],
                        colorscale=_discrete_label_plane_colorscale(x_colors),
                        cmax=max(1, len(x_colors))
                    ),
                    _slice_plane_surface(
                        y_slice,
                        grid,
                        "y",
                        y_idx,
                        COLOR_SCHEME["slice_y"],
                        colorscale=_discrete_label_plane_colorscale(y_colors),
                        cmax=max(1, len(y_colors))
                    ),
                    _slice_plane_surface(
                        z_slice,
                        grid,
                        "z",
                        z_idx,
                        COLOR_SCHEME["slice_z"],
                        colorscale=_discrete_label_plane_colorscale(z_colors),
                        cmax=max(1, len(z_colors))
                    )
                ]
            for plane in plane_traces:
                if plane is not None:
                    fig3d.add_trace(plane)
        for frame in _slice_frames(grid, x_idx, y_idx, z_idx):
            fig3d.add_trace(frame)

        # ROI sphere (wireframe)
        roi_traces = _roi_frame(grid, eff_roi_x, eff_roi_y, eff_roi_z, eff_roi_r,
                                n_lat=6, n_lon=8, color="#E11D48", line_width=3)
        for t in roi_traces:
            fig3d.add_trace(t)
        fig3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            uirevision=uirev,
            scene=dict(
                uirevision=uirev,
                bgcolor="#000000",
                aspectmode="manual",
                aspectratio=aspectratio,
                xaxis=dict(title="X (mm)", range=list(x_range), showgrid=False, showspikes=False),
                yaxis=dict(title="Y (mm)", range=list(y_range), showgrid=False, showspikes=False),
                zaxis=dict(title="Z (mm)", range=list(z_range), showgrid=False, showspikes=False),
            )
        )
        if camera_state:
            fig3d.update_layout(scene_camera=camera_state)

        return x_fig, y_fig, z_fig, fig3d, camera_state

    @app.callback(
        Output("hover-store", "data"),
        Input("x-view", "hoverData"),
        Input("y-view", "hoverData"),
        Input("z-view", "hoverData"),
        State("x-slider", "value"),
        State("y-slider", "value"),
        State("z-slider", "value"),
        prevent_initial_call=True
    )
    def update_hover_store(x_hover, y_hover, z_hover, x_idx, y_idx, z_idx):
        triggered = callback_context.triggered or []
        if not triggered:
            raise PreventUpdate
        prop = triggered[0]["prop_id"]
        data = None
        if prop.startswith("x-view"):
            data = x_hover
            axis = "x"
        elif prop.startswith("y-view"):
            data = y_hover
            axis = "y"
        elif prop.startswith("z-view"):
            data = z_hover
            axis = "z"
        else:
            raise PreventUpdate

        if not data or "points" not in data or not data["points"]:
            raise PreventUpdate

        pt = data["points"][0]
        try:
            xv = int(round(pt.get("x", 0)))
            yv = int(round(pt.get("y", 0)))
        except (TypeError, ValueError):
            raise PreventUpdate

        if axis == "x":
            i_idx = x_idx
            j_idx = int(np.clip(xv, 0, Ny - 1))
            k_idx = int(np.clip(yv, 0, Nz - 1))
        elif axis == "y":
            i_idx = int(np.clip(xv, 0, Nx - 1))
            j_idx = y_idx
            k_idx = int(np.clip(yv, 0, Nz - 1))
        else:
            i_idx = int(np.clip(xv, 0, Nx - 1))
            j_idx = int(np.clip(yv, 0, Ny - 1))
            k_idx = z_idx

        return {"axis": axis, "i": i_idx, "j": j_idx, "k": k_idx, "ts": data.get("event", {}).get("timeStamp", None)}

    @app.callback(
        Output("catch-store", "data"),
        Input("x-view", "clickData"),
        Input("y-view", "clickData"),
        Input("z-view", "clickData"),
        State("x-slider", "value"),
        State("y-slider", "value"),
        State("z-slider", "value"),
        prevent_initial_call=True
    )
    def update_catch_store(x_click, y_click, z_click, x_idx, y_idx, z_idx):
        triggered = callback_context.triggered or []
        if not triggered:
            raise PreventUpdate
        prop = triggered[0]["prop_id"]
        data = None
        if prop.startswith("x-view"):
            data = x_click
            axis = "x"
        elif prop.startswith("y-view"):
            data = y_click
            axis = "y"
        elif prop.startswith("z-view"):
            data = z_click
            axis = "z"
        else:
            raise PreventUpdate

        if not data or "points" not in data or not data["points"]:
            raise PreventUpdate

        pt = data["points"][0]
        try:
            xv = int(round(pt.get("x", 0)))
            yv = int(round(pt.get("y", 0)))
        except (TypeError, ValueError):
            raise PreventUpdate

        if axis == "x":
            i_idx = x_idx
            j_idx = int(np.clip(xv, 0, Ny - 1))
            k_idx = int(np.clip(yv, 0, Nz - 1))
        elif axis == "y":
            i_idx = int(np.clip(xv, 0, Nx - 1))
            j_idx = y_idx
            k_idx = int(np.clip(yv, 0, Nz - 1))
        else:
            i_idx = int(np.clip(xv, 0, Nx - 1))
            j_idx = int(np.clip(yv, 0, Ny - 1))
            k_idx = z_idx

        return {"axis": axis, "i": i_idx, "j": j_idx, "k": k_idx}

    @app.callback(
        Output("p1-x", "value"),
        Output("p1-y", "value"),
        Output("p1-z", "value"),
        Input("p1-set-btn", "n_clicks"),
        State("catch-store", "data"),
        State("p1-x", "value"),
        State("p1-y", "value"),
        State("p1-z", "value"),
        prevent_initial_call=True
    )
    def set_p1_from_catch(n_clicks, catch_data, p1_x, p1_y, p1_z):
        if not n_clicks or not catch_data:
            raise PreventUpdate
        return catch_data.get("i", p1_x), catch_data.get("j", p1_y), catch_data.get("k", p1_z)

        status = (
            f"Active masks: {', '.join(mask_values)} | X={x_idx}, Y={y_idx}, Z={z_idx} "
            f"| ROI: (i,j,k)=({eff_roi_x},{eff_roi_y},{eff_roi_z}), r={eff_roi_r} vox"
        )
        try:
            if auto_status:
                status += auto_status
        except Exception:
            pass
        return x_fig, y_fig, z_fig, fig3d, status

    # Open browser on a fresh port each run to avoid caching
    import threading, webbrowser, random
    port = random.randint(8050, 9000)
    threading.Timer(0.5, lambda: webbrowser.open(f"http://127.0.0.1:{port}")).start()
    app.run(debug=False, port=port)

# ---------------------------------------------------------------------
# Pipeline
# ---------------------------------------------------------------------
def run_pipeline(
    in_mesh_path: str,
    spacing: float | None = None,
    n: int | None = None,
    margin_frac: float = 0.10,
    out_dir: str = "outputs",
    show: bool = True,
    mc_map: str = "xyz",
    viewer: str = "dash",
    port: int = 8050,
    method: str = "auto",
    export_stl: bool = False,
    stl_dir: str | None = None,
):
    input_path = Path(in_mesh_path).resolve()
    if not input_path.exists():
        err(f"Input not found: {in_mesh_path}")
        raise FileNotFoundError(in_mesh_path)

    out_root, out_npz = _resolve_output_paths(in_mesh_path, out_dir)

    suffix = input_path.suffix.lower()

    # --- VTI directory: assemble spatial tiles then extract labels ---
    if input_path.is_dir():
        layers, grid, metadata = load_vti_directory_label_layers(
            str(input_path),
            array_name=vti_array,
            background=vti_background,
            background_eps=vti_background_eps,
            max_dim=vti_max_dim,
            max_labels=vti_max_labels,
        )
        layer_arrays = {
            f"mask_{layer['id']}": layer["mask"].astype(np.uint8)
            for layer in layers
        }
        scalar_layers = [
            layer for layer in layers
            if layer.get("volume_kind") == "scalar" and layer.get("scalar_data") is not None
        ]
        if scalar_layers:
            layer_arrays["scalar_values"] = scalar_layers[0]["scalar_data"].astype(np.float32, copy=False)
        np.savez_compressed(
            str(out_npz),
            **layer_arrays,
            spacing=np.array(grid["spacing"]),
            origin=np.array(grid["origin"]),
            layer_ids=np.array([layer["id"] for layer in layers]),
            layer_names=np.array([layer["name"] for layer in layers]),
            layer_values=np.array([str(layer.get("label_value", "")) for layer in layers]),
            full_dims_xyz=np.array(metadata["full_dims_xyz"]),
            sampled_dims_xyz=np.array(metadata["sampled_dims_xyz"]),
            sample_stride=np.array(metadata["sample_stride"]),
            tile_count=np.array(metadata.get("tile_count", 0)),
            source_type=np.array("vti_directory"),
            source_path=np.array(str(input_path)),
            vti_array=np.array(metadata["array_name"]),
            vti_discrete_labels=np.array(metadata["discrete_labels"]),
            vti_volume_kind=np.array(metadata.get("volume_kind", "labels")),
            vti_scalar_range=np.array(metadata.get("scalar_range", (0.0, 1.0))),
            vti_unique_value_count=np.array(metadata.get("unique_value_count", 0)),
            vti_non_background_unique_value_count=np.array(metadata.get("non_background_unique_value_count", 0)),
        )
        save_desc = "scalar volume + occupancy mask" if metadata.get("volume_kind") == "scalar" else "label masks"
        ok(f"VTI directory {save_desc} saved → {out_npz}")

        if export_stl:
            stl_root = Path(stl_dir) if stl_dir else out_root
            stl_root.mkdir(parents=True, exist_ok=True)
            stl_total = 0
            for layer in layers:
                if save_mask_stl(layer["mask"], grid, stl_root / f"{layer['id']}.stl", name=layer["name"]):
                    stl_total += 1
            ok(f"STL export complete → {stl_root} ({stl_total} label surface(s))")

        if show:
            show_viewer_dash(
                None,
                grid=grid,
                mask_layers=layers,
                port=port,
                viewer_kind="volume",
            )
        return

    if suffix == ".vti":
        layers, grid, metadata = load_vti_label_layers(
            str(input_path),
            array_name=vti_array,
            background=vti_background,
            background_eps=vti_background_eps,
            max_dim=vti_max_dim,
            max_labels=vti_max_labels,
        )
        layer_arrays = {
            f"mask_{layer['id']}": layer["mask"].astype(np.uint8)
            for layer in layers
        }
        scalar_layers = [
            layer for layer in layers
            if layer.get("volume_kind") == "scalar" and layer.get("scalar_data") is not None
        ]
        if scalar_layers:
            layer_arrays["scalar_values"] = scalar_layers[0]["scalar_data"].astype(np.float32, copy=False)
        np.savez_compressed(
            str(out_npz),
            **layer_arrays,
            spacing=np.array(grid["spacing"]),
            origin=np.array(grid["origin"]),
            layer_ids=np.array([layer["id"] for layer in layers]),
            layer_names=np.array([layer["name"] for layer in layers]),
            layer_values=np.array([str(layer.get("label_value", "")) for layer in layers]),
            full_dims_xyz=np.array(metadata["full_dims_xyz"]),
            sampled_dims_xyz=np.array(metadata["sampled_dims_xyz"]),
            sample_stride=np.array(metadata["sample_stride"]),
            source_type=np.array("vti"),
            source_path=np.array(str(input_path)),
            vti_array=np.array(metadata["array_name"]),
            vti_discrete_labels=np.array(metadata["discrete_labels"]),
            vti_volume_kind=np.array(metadata.get("volume_kind", "labels")),
            vti_scalar_range=np.array(metadata.get("scalar_range", (0.0, 1.0))),
            vti_unique_value_count=np.array(metadata.get("unique_value_count", 0)),
            vti_non_background_unique_value_count=np.array(metadata.get("non_background_unique_value_count", 0)),
        )
        save_desc = "scalar volume + occupancy mask" if metadata.get("volume_kind") == "scalar" else "label masks"
        ok(f"VTI {save_desc} saved → {out_npz}")

        if export_stl:
            stl_root = Path(stl_dir) if stl_dir else out_root
            stl_root.mkdir(parents=True, exist_ok=True)
            stl_total = 0
            for layer in layers:
                if save_mask_stl(layer["mask"], grid, stl_root / f"{layer['id']}.stl", name=layer["name"]):
                    stl_total += 1
            ok(f"STL export complete → {stl_root} ({stl_total} label surface(s))")

        if show:
            show_viewer_dash(
                None,
                grid=grid,
                mask_layers=layers,
                port=port,
                viewer_kind="volume",
            )
        return

    if trimesh is None:
        raise RuntimeError("trimesh is required for mesh inputs. Install with: pip install trimesh")
    if igl is None:
        raise RuntimeError(
            "python-igl is required for mesh winding classification. "
            "Install with: conda install -c conda-forge python-igl  (or pip install igl)"
        )

    mesh = trimesh.load(input_path, force="mesh")
    info(f"Loaded mesh: {input_path.name} | watertight={getattr(mesh, 'is_watertight', 'unknown')}")

    # Center mesh at origin
    mesh = center_mesh(mesh) # type: ignore

    # Grid (power-of-two cubic grid)
    grid = make_cube_grid_from_mesh(mesh, spacing=spacing, n=n, margin_frac=margin_frac)

    # In/on/out masks
    if method == "auto":
        if igl is not None:
            inside_u8, on_u8, out_u8 = classify_by_winding(mesh, grid, band=0.6)
        else:
            warn("python-igl not available; falling back to trimesh.contains (slower).")
            inside_u8, on_u8, out_u8 = classify_by_trimesh_contains(mesh, grid)
    elif method == "winding":
        if igl is None:
            raise RuntimeError("python-igl is required for method='winding'.")
        inside_u8, on_u8, out_u8 = classify_by_winding(mesh, grid, band=0.6)
    elif method == "trimesh":
        inside_u8, on_u8, out_u8 = classify_by_trimesh_contains(mesh, grid)
    else:
        raise ValueError(f"Unknown method: {method}")

    # Save masks in a single compressed NPZ with spacing/origin like core.run_pipeline
    np.savez_compressed(str(out_npz), inside=inside_u8, on=on_u8, out=out_u8,
                        spacing=np.array(grid["spacing"]), origin=np.array(grid["origin"]))
    ok(f"Masks saved (uint8) → {out_npz}")

    if export_stl:
        stl_root = Path(stl_dir) if stl_dir else out_root
        stl_root.mkdir(parents=True, exist_ok=True)
        save_mask_stl(inside_u8, grid, stl_root / "inside.stl", name="inside")
        save_mask_stl(on_u8, grid, stl_root / "on.stl", name="on")
        save_mask_stl(out_u8, grid, stl_root / "out.stl", name="out")
        ok(f"STL export complete → {stl_root}")

    # Viewer
    if show:
        if viewer == "dash":
            show_viewer_dash(mesh, inside_u8, on_u8, out_u8, grid, port=port, viewer_kind="surface")
        else:
            # fallback: use the 3-D html viewer (not implemented separately here)
            show_viewer_dash(mesh, inside_u8, on_u8, out_u8, grid, port=port, viewer_kind="surface")

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Standalone FakeCT pipeline (winding or trimesh)")
    ap.add_argument("--in", dest="in_mesh", required=True,
                    help="Input mesh (.stl/.obj/.ply) or voxel image (.vti)")
    ap.add_argument("--spacing", type=float, default=None,
                    help="Voxel edge length in mm (if provided, N will be computed as next power-of-two)")
    ap.add_argument("--n", type=int, default=7,
                    help="Grid exponent (2^n per side). Used if --spacing is None.")
    ap.add_argument("--margin", type=float, default=0.10,
                    help="Extra margin fraction around the mesh AABB (default 0.10 = 10%%)")
    ap.add_argument("--mc-map", type=str, default="zyx",
                    choices=["zyx", "xyz", "xzy", "yxz", "yzx", "zxy"],
                    help="Axis mapping from marching-cubes (z,y,x) → (X,Y,Z).")
    ap.add_argument("--out", default="outputs/masks_demo.npz",
                    help="Output compressed npz file")
    ap.add_argument("--method", choices=["auto", "winding", "trimesh"], default="auto",
                    help="Inside/outside method: auto (default), winding (python-igl), or trimesh")
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open viewer")
    ap.add_argument("--viewer", choices=["dash","html"], default="dash",
                    help="Viewer type: 'dash' or 'html'.")
    ap.add_argument("--port", type=int, default=8050,
                    help="Port for Dash viewer (default 8050).")
    ap.add_argument("--export-stl", action="store_true",
                    help="Export inside/on/out masks as STL files.")
    ap.add_argument("--stl-dir", default=None,
                    help="Directory for STL export (defaults to --out directory).")
    ap.add_argument("--vti-array", default=None,
                    help="VTI DataArray name to read. Defaults to the VTI Scalars array or first array.")
    ap.add_argument("--vti-background", type=float, default=0.0,
                    help="Background scalar/label value for VTI volume extraction.")
    ap.add_argument("--vti-background-eps", type=float, default=0.0,
                    help="Tolerance for matching floating-point VTI background values.")
    ap.add_argument("--vti-max-dim", type=int, default=160,
                    help="Maximum sampled VTI dimension for browser interaction. Use 0 for full resolution.")
    ap.add_argument("--vti-max-labels", type=int, default=64,
                    help="Maximum number of discrete non-background VTI values to split into label layers.")

    args = ap.parse_args()

    if args.method == "winding" and igl is None:
        err("python-igl is required for --method winding. Install via conda-forge:\n"
            "  conda install -c conda-forge python-igl\n"
            "or pip:\n"
            "  pip install igl")
        sys.exit(1)

    spacing = None if (args.spacing is None or args.spacing <= 0) else float(args.spacing)

    run_pipeline(
        in_mesh_path=args.in_mesh,
        spacing=spacing,
        n=args.n,
        margin_frac=args.margin,
        out_npz=args.out,
        show=not args.no_show,
        mc_map=args.mc_map,
        viewer=args.viewer,
        port=args.port,
        method=args.method,
        export_stl=args.export_stl,
        stl_dir=args.stl_dir,
    )

if __name__ == "__main__":
    main()
