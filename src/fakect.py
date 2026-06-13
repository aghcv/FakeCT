#!/usr/bin/env python3
# ---------------------------------------------------------------------
# fakect.py
# Produces winding-based in/on/out masks and (optionally) launches a Dash viewer.
#
# Usage examples:
#   python fakect.py --in cube.stl --n 8 --out cube_masks.npz
#   python fakect.py --in carotid.stl --n 9 --margin 0.10 --out examples/outputs/carotid_masks.npz
#
# Requirements & installation
#  Recommended (conda - easiest, includes python-igl):
#    conda create -n fakect python=3.10 -y
#    conda activate fakect
#    conda install -c conda-forge python-igl trimesh scipy scikit-image plotly dash -y
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
import re
import sys
import xml.etree.ElementTree as ET
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
    match = re.search(r"<AppendedData[^>]*>\s*_(.*?)\s*</AppendedData>", text, re.S)
    if not match:
        raise ValueError("VTI file has no appended data payload")
    return _decode_vtk_base64_payload(match.group(1))

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
        yield zlib.decompress(block)

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
        raw = _vti_appended_bytes(text)
        offset = int(data_array.attrib.get("offset", "0"))
        if root.attrib.get("compressor") == "vtkZLibDataCompressor":
            consume_payload(_iter_vtk_zlib_payloads(raw, header_dtype, offset))
        else:
            consume_payload(_iter_vtk_uncompressed_payload(raw, header_dtype, offset))
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
    if np.issubdtype(values.dtype, np.integer) or np.issubdtype(values.dtype, np.bool_):
        return True
    if unique_values.size > max_labels + 1:
        return False
    finite = unique_values[np.isfinite(unique_values)]
    if finite.size != unique_values.size:
        return False
    return bool(np.allclose(finite, np.round(finite), atol=1e-6))

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

    if np.issubdtype(data.dtype, np.floating):
        background_mask = np.isclose(data, bg, atol=eps)
    else:
        background_mask = data == np.asarray(background, dtype=data.dtype)

    discrete = _is_discrete_label_array(data, unique_values, max_labels)
    layers: list[dict] = []
    if discrete:
        label_values = [v for v in unique_values if not np.isclose(float(v), bg, atol=eps)]
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
                layers.append(_make_mask_layer(
                    _safe_layer_id(array_label, _label_display_value(value)),
                    name,
                    mask,
                    LABEL_COLOR_PALETTE[idx % len(LABEL_COLOR_PALETTE)],
                    label_value=_label_display_value(value),
                    exclusive_group="vti-labels"
                ))

    if not discrete:
        occupancy = ~background_mask
        layers.append(_make_mask_layer(
            _safe_layer_id(array_label, "nonzero"),
            f"{array_label} non-background",
            occupancy,
            LABEL_COLOR_PALETTE[0],
            label_value="non-background",
            exclusive_group=None
        ))

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
        "labels": [layer["label_value"] for layer in layers],
    })
    info(f"Prepared {len(layers)} editable VTI layer(s): {', '.join(layer['name'] for layer in layers[:8])}")
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

    WN = igl.fast_winding_number_for_meshes(V, F, Q)  # (N^3,)
    WN = np.asarray(WN).reshape((N, N, N))  # (k, j, i) == (Z,Y,X)

    # Inside = WN > 0.5
    inside = (WN > 0.5)

    # On = narrow band around surface: boundary voxels of 'inside'
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

def _hex_to_rgba(hex_color: str, alpha: float) -> str:
    hex_color = hex_color.lstrip("#")
    if len(hex_color) != 6:
        return f"rgba(255,255,255,{alpha})"
    r = int(hex_color[0:2], 16)
    g = int(hex_color[2:4], 16)
    b = int(hex_color[4:6], 16)
    return f"rgba({r},{g},{b},{alpha})"

def _slice_colorscale(mask_color: str) -> list:
    return [[0.0, "black"], [1.0, mask_color]]

def _slice_plane_colorscale(mask_color: str, roi_color: str, alpha: float) -> list:
    return [
        [0.0, "rgba(0,0,0,0)"],
        [0.5, _hex_to_rgba(mask_color, alpha)],
        [1.0, _hex_to_rgba(roi_color, alpha)]
    ]

def _roi_overlay_trace(roi_mask: np.ndarray, color: str, opacity: float) -> go.Heatmap:
    z = np.where(roi_mask > 0, 1.0, np.nan)
    return go.Heatmap(
        z=z,
        zmin=0,
        zmax=1,
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        hoverinfo="skip",
        opacity=max(0.0, min(1.0, float(opacity)))
    )

def _voxel_center(grid, i_idx, j_idx, k_idx):
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]
    x = ox + (i_idx + 0.5) * sx
    y = oy + (j_idx + 0.5) * sy
    z = oz + (k_idx + 0.5) * sz
    return x, y, z

def _sphere_surface(center, radius, color, opacity, resolution=24):
    if radius <= 0:
        return None
    u = np.linspace(0, 2 * np.pi, resolution)
    v = np.linspace(0, np.pi, resolution)
    x = center[0] + radius * np.outer(np.cos(u), np.sin(v))
    y = center[1] + radius * np.outer(np.sin(u), np.sin(v))
    z = center[2] + radius * np.outer(np.ones_like(u), np.cos(v))
    return go.Surface(
        x=x, y=y, z=z,
        surfacecolor=np.zeros_like(x),
        colorscale=[[0.0, color], [1.0, color]],
        showscale=False,
        opacity=opacity,
        name="ROI sphere",
        hoverinfo="skip"
    )

def _slice_plane_surface(
    slice2d,
    grid,
    axis,
    idx,
    color,
    alpha=OPACITY_LEVELS["high"],
    colorscale=None,
    cmax=1
):
    """
    Add a semi-transparent slice plane colored by the same mask as the 2D view.
    slice2d shapes (Z,Y) for x, (Z,X) for y, (Y,X) for z.
    """
    if slice2d.size == 0:
        return None

    Nz, Ny, Nx = grid["shape"]
    sz, sy, sx = grid["spacing"]
    ox, oy, oz = grid["origin"]

    x_coords = ox + (np.arange(Nx) + 0.5) * sx
    y_coords = oy + (np.arange(Ny) + 0.5) * sy
    z_coords = oz + (np.arange(Nz) + 0.5) * sz

    if axis == "x":
        Y, Z = np.meshgrid(y_coords, z_coords)
        X = np.full_like(Y, ox + (idx + 0.5) * sx)
    elif axis == "y":
        X, Z = np.meshgrid(x_coords, z_coords)
        Y = np.full_like(X, oy + (idx + 0.5) * sy)
    elif axis == "z":
        X, Y = np.meshgrid(x_coords, y_coords)
        Z = np.full_like(X, oz + (idx + 0.5) * sz)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    if colorscale is None:
        colorscale = [
            [0.0, "rgba(0,0,0,0)"],
            [1.0, _hex_to_rgba(color, alpha)]
        ]

    return go.Surface(
        x=X, y=Y, z=Z,
        surfacecolor=slice2d,
        cmin=0, cmax=cmax,
        colorscale=colorscale,
        showscale=False,
        name=f"{axis}-slice-plane",
        hoverinfo="skip",
        opacity=1.0
    )

def mask_to_trace(mask_u8, grid, color, name, opacity=OPACITY_LEVELS["high"]):
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

def _binary_colorscale(on_color="#ffffff"):
    return [[0.0, "black"], [1.0, on_color]]

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
            zmin=0,
            zmax=zmax,
            colorscale=colorscale,
            showscale=False,
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

def _discrete_label_colorscale(colors: list[str]) -> list:
    if not colors:
        return [[0.0, "black"], [1.0, "black"]]
    n = len(colors)
    colorscale = [[0.0, "black"]]
    for i, color in enumerate(colors, start=1):
        pos = i / max(1, n)
        colorscale.append([pos, color])
    return colorscale

def _discrete_label_plane_colorscale(colors: list[str], alpha: float = OPACITY_LEVELS["high"]) -> list:
    if not colors:
        return [[0.0, "rgba(0,0,0,0)"], [1.0, "rgba(0,0,0,0)"]]
    n = len(colors)
    colorscale = [[0.0, "rgba(0,0,0,0)"]]
    for i, color in enumerate(colors, start=1):
        pos = i / max(1, n)
        colorscale.append([pos, _hex_to_rgba(color, alpha)])
    return colorscale

def compose_layer_slice(layer_states: dict, visible_ids: list[str], axis: str, idx: int) -> tuple[np.ndarray, list[str]]:
    """Compose visible binary layers into integer-coded slice values."""
    active_ids = [lid for lid in visible_ids if lid in layer_states]
    if len(active_ids) == 0:
        first = next(iter(layer_states.values()), None)
        if first is None:
            return np.zeros((1, 1), dtype=np.uint8), []
        return _empty_slice({"shape": first["current"].shape}, axis), []

    first_mask = layer_states[active_ids[0]]["current"]
    if axis == "x":
        out = np.zeros(first_mask[:, :, idx].shape, dtype=np.uint16)
    elif axis == "y":
        out = np.zeros(first_mask[:, idx, :].shape, dtype=np.uint16)
    elif axis == "z":
        out = np.zeros(first_mask[idx, :, :].shape, dtype=np.uint16)
    else:
        raise ValueError("axis must be 'x', 'y', or 'z'")

    colors = []
    for code, layer_id in enumerate(active_ids, start=1):
        mask = layer_states[layer_id]["current"]
        if axis == "x":
            sl = mask[:, :, idx]
        elif axis == "y":
            sl = mask[:, idx, :]
        else:
            sl = mask[idx, :, :]
        out[sl > 0] = code
        colors.append(layer_states[layer_id]["color"])
    return out, colors

def _scale_to_pos(scale_val: float) -> float:
    """Map scale value in [0.001, 99.999] to slider position in [0, 1] with 1.0 at 0.5."""
    v = max(0.001, min(99.999, float(scale_val)))
    if v <= 1.0:
        return 0.5 * (np.log10(v) - np.log10(0.001)) / (np.log10(1.0) - np.log10(0.001))
    return 0.5 + 0.5 * (np.log10(v) - np.log10(1.0)) / (np.log10(99.999) - np.log10(1.0))

def _pos_to_scale(pos: float) -> float:
    """Map slider position in [0, 1] to scale value in [0.001, 99.999] with 1.0 at 0.5."""
    p = max(0.0, min(1.0, float(pos)))
    if p <= 0.5:
        t = p / 0.5
        return float(10 ** (np.log10(0.001) + t * (np.log10(1.0) - np.log10(0.001))))
    t = (p - 0.5) / 0.5
    return float(10 ** (np.log10(1.0) + t * (np.log10(99.999) - np.log10(1.0))))

def _linspace_safe(vmin: float, vmax: float, count: int) -> list:
    """Generate a safe list of values for batch export (count >= 1)."""
    try:
        vmin = float(vmin)
        vmax = float(vmax)
    except (TypeError, ValueError):
        return [1.0]
    try:
        count = int(count)
    except (TypeError, ValueError):
        count = 1
    count = max(1, count)
    if count == 1:
        return [vmin]
    return list(np.linspace(vmin, vmax, count))

def _fmt_param(val: float, decimals: int = 3) -> str:
    """Format a float for filenames (avoid dots)."""
    try:
        v = float(val)
    except (TypeError, ValueError):
        v = 0.0
    return f"{v:.{decimals}f}".replace(".", "p")

def _gaussian_shape(x: float, k: float) -> float:
    """Normalized Gaussian shape with S(0)=0, S(0.5)=1, S(1)=0."""
    k = max(1e-6, float(k))
    s = np.exp(-k * (x - 0.5) ** 2)
    s0 = np.exp(-k * 0.25)
    denom = max(1e-8, 1.0 - s0)
    val = (s - s0) / denom
    return float(max(0.0, min(1.0, val)))

def _roi_mask_vox(shape_zyx, center_ijk, r_vox):
    """Return a boolean spherical ROI mask in voxel index space."""
    if r_vox is None or r_vox <= 0:
        return np.zeros(shape_zyx, dtype=bool)
    Nz, Ny, Nx = shape_zyx
    i0, j0, k0 = center_ijk
    ii = np.arange(Nx) - int(i0)
    jj = np.arange(Ny) - int(j0)
    kk = np.arange(Nz) - int(k0)
    rr2 = (ii * ii)[None, None, :] + (jj * jj)[None, :, None] + (kk * kk)[:, None, None]
    return rr2 <= (int(r_vox) * int(r_vox))

def _bitwise_dilate6(mask_bool: np.ndarray) -> np.ndarray:
    """Single-iteration 3D dilation using 6-neighborhood via bitwise shifts."""
    zpos = np.pad(mask_bool[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=False)
    zneg = np.pad(mask_bool[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=False)
    ypos = np.pad(mask_bool[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=False)
    yneg = np.pad(mask_bool[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=False)
    xpos = np.pad(mask_bool[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=False)
    xneg = np.pad(mask_bool[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=False)
    return mask_bool | zpos | zneg | ypos | yneg | xpos | xneg

def _bitwise_erode6(mask_bool: np.ndarray) -> np.ndarray:
    """Single-iteration 3D erosion using 6-neighborhood via bitwise shifts."""
    zpos = np.pad(mask_bool[:-1, :, :], ((1, 0), (0, 0), (0, 0)), mode="constant", constant_values=False)
    zneg = np.pad(mask_bool[1:, :, :], ((0, 1), (0, 0), (0, 0)), mode="constant", constant_values=False)
    ypos = np.pad(mask_bool[:, :-1, :], ((0, 0), (1, 0), (0, 0)), mode="constant", constant_values=False)
    yneg = np.pad(mask_bool[:, 1:, :], ((0, 0), (0, 1), (0, 0)), mode="constant", constant_values=False)
    xpos = np.pad(mask_bool[:, :, :-1], ((0, 0), (0, 0), (1, 0)), mode="constant", constant_values=False)
    xneg = np.pad(mask_bool[:, :, 1:], ((0, 0), (0, 0), (0, 1)), mode="constant", constant_values=False)
    return mask_bool & zpos & zneg & ypos & yneg & xpos & xneg

def proximity_bridge(mask_u8: np.ndarray, max_dist: int = 8) -> np.ndarray:
    """Bridge disconnected components within a small Chebyshev distance."""
    mask = (mask_u8 > 0)
    if max_dist <= 0:
        return mask_u8

    size = 2 * max_dist + 1
    kernel = np.ones((size, size, size), dtype=bool)
    dil = binary_dilation(mask, structure=kernel)
    bridge = dil & (~mask)

    labeled, num = measure.label(mask.astype(np.uint8), return_num=True, connectivity=1)
    if num <= 1:
        return mask.astype(np.uint8)

    Z, Y, X = np.where(bridge)
    out = mask.copy()
    for z, y, x in zip(Z, Y, X):
        neigh = []
        if z > 0:
            neigh.append(labeled[z - 1, y, x])
        if z < labeled.shape[0] - 1:
            neigh.append(labeled[z + 1, y, x])
        if y > 0:
            neigh.append(labeled[z, y - 1, x])
        if y < labeled.shape[1] - 1:
            neigh.append(labeled[z, y + 1, x])
        if x > 0:
            neigh.append(labeled[z, y, x - 1])
        if x < labeled.shape[2] - 1:
            neigh.append(labeled[z, y, x + 1])

        neigh = [n for n in neigh if n != 0]
        if len(set(neigh)) >= 2:
            out[z, y, x] = True

    return out.astype(np.uint8)

def apply_roi_scale(
    inside_u8: np.ndarray,
    center_ijk: tuple[int, int, int],
    r_vox: int,
    scale_factor: float,
    bridge_dist: int = 2,
    shape_k: float = 10.0,
    shape_window: tuple[float, float] = (0.25, 0.75),
) -> np.ndarray:
    """Scale vessel diameter within a spherical ROI using bitwise morphology."""
    if inside_u8 is None:
        return inside_u8
    try:
        sf = float(scale_factor)
    except (TypeError, ValueError):
        return inside_u8

    r0 = int(round(r_vox))
    if r0 <= 0 or sf == 1.0:
        return inside_u8

    # Convert scale factor to voxel diameter space for 1-voxel iterations.
    base_diam = max(1, int(round(2 * r0)))
    target_diam = max(0, int(round(base_diam * sf)))
    r1 = int(round(target_diam / 2.0))
    if r1 == r0:
        return inside_u8

    inside_bool = (inside_u8 > 0)
    if r1 <= 0:
        r1 = 1

    # Ensure target ROI contains at least one voxel of volume.
    roi_target = _roi_mask_vox(inside_u8.shape, center_ijk, r1)
    if np.sum(roi_target) < 1:
        r1 = 1
        roi_target = _roi_mask_vox(inside_u8.shape, center_ijk, r1)

    # Step ROI size toward target, applying one morphology iteration per step.
    mode = "dilate" if r1 > r0 else "erode"
    step = 1 if r1 > r0 else -1
    r_current = r0
    current_inside = inside_bool.copy()
    last_valid_inside = current_inside.copy()
    min_roi_fraction = 0.05
    initial_roi = _roi_mask_vox(current_inside.shape, center_ijk, r_current)
    initial_count = int(np.sum(current_inside & initial_roi))
    min_inside = max(1, int(round(initial_count * min_roi_fraction)))
    total_steps = abs(r1 - r0)
    step_idx = 0
    accum = 0.0

    while r_current != r1:
        roi_current = _roi_mask_vox(current_inside.shape, center_ijk, r_current)
        inside_in_roi = current_inside & roi_current
        outside = current_inside & (~roi_current)

        x = 0.5
        if total_steps > 1:
            x = step_idx / float(total_steps - 1)

        w0, w1 = shape_window
        w0 = max(0.0, min(1.0, float(w0)))
        w1 = max(0.0, min(1.0, float(w1)))
        if w1 <= w0:
            w0, w1 = 0.25, 0.75
        if x < w0 or x > w1:
            s = 0.0
        else:
            x = (x - w0) / (w1 - w0)
            s = _gaussian_shape(x, shape_k)

        accum += s
        apply_step = accum >= 1.0
        if apply_step:
            accum -= 1.0

        work = inside_in_roi
        if apply_step:
            if mode == "dilate":
                work = _bitwise_dilate6(inside_in_roi)
            else:
                work = _bitwise_erode6(inside_in_roi)

        # Clamp to current ROI for this iteration.
        work = work & roi_current
        merged = (outside | work).astype(np.uint8)
        if bridge_dist and bridge_dist > 0:
            merged = proximity_bridge(merged, max_dist=bridge_dist)
        current_inside = (merged > 0)

        # Safety: never erase the ROI completely during erosion.
        if mode == "erode":
            roi_after = _roi_mask_vox(current_inside.shape, center_ijk, r_current)
            if int(np.sum(current_inside & roi_after)) < min_inside:
                current_inside = last_valid_inside
                break
            last_valid_inside = current_inside.copy()

        r_current += step
        step_idx += 1

    return current_inside.astype(np.uint8)

def show_viewer_dash(
    mesh,
    inside_u8=None,
    on_u8=None,
    out_u8=None,
    grid=None,
    port=8050,
    mask_layers: list[dict] | None = None,
    dataset_name: str = "FakeCT",
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

    Nz, Ny, Nx = mask_layers[0]["mask"].shape
    x_mid, y_mid, z_mid = Nx // 2, Ny // 2, Nz // 2

    mesh_trace = None
    if mesh is not None:
        mesh_trace = go.Mesh3d(
            x=mesh.vertices[:, 0], y=mesh.vertices[:, 1], z=mesh.vertices[:, 2],
            i=mesh.faces[:, 0], j=mesh.faces[:, 1], k=mesh.faces[:, 2],
            name="mesh", color=COLOR_SCHEME["mesh"], opacity=OPACITY_LEVELS["low"]
        )

    layer_states: dict[str, dict] = {}
    for idx, layer in enumerate(mask_layers):
        layer_id = layer["id"]
        color = layer.get("color", LABEL_COLOR_PALETTE[idx % len(LABEL_COLOR_PALETTE)])
        opacity = layer.get("opacity", OPACITY_LEVELS["mid"])
        mask = layer["mask"].astype(np.uint8)
        layer_states[layer_id] = {
            "id": layer_id,
            "name": layer.get("name", layer_id),
            "current": mask.copy(),
            "original": mask.copy(),
            "trace": mask_to_trace(mask, grid, color, layer.get("name", layer_id), opacity),
            "color": color,
            "opacity": opacity,
            "editable": bool(layer.get("editable", True)),
            "exclusive_group": layer.get("exclusive_group"),
        }

    layer_options = [
        {"label": state["name"], "value": layer_id}
        for layer_id, state in layer_states.items()
    ]
    editable_options = [
        {"label": state["name"], "value": layer_id}
        for layer_id, state in layer_states.items()
        if state["editable"]
    ]
    if not editable_options:
        editable_options = layer_options
    target_default = editable_options[0]["value"]
    nonempty_ids = [
        layer_id for layer_id, state in layer_states.items()
        if int(np.sum(state["current"])) > 0
    ]
    default_visible = nonempty_ids[:8] if nonempty_ids else [target_default]

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
    """
    wheel_js = """
    (function() {
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
        }
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

    app.layout = html.Div(
        style={"display": "grid", "gridTemplateColumns": "300px 1fr", "gap": "10px",
               "height": "100vh", "backgroundColor": "#0f1115", "color": "#e6e6e6",
               "padding": "10px"},
        children=[
            html.Div(
                style={"backgroundColor": "#151922", "borderRadius": "10px",
                       "padding": "12px", "display": "flex", "flexDirection": "column", "gap": "10px"},
                children=[
                    html.H3(dataset_name, style={"margin": "0 0 6px 0"}),
                    html.Div(
                        style=panel_card,
                        children=[
                            html.Div("Labels", style={"fontWeight": "600"}),
                            dcc.Dropdown(
                                id="mask-check",
                                options=layer_options,
                                value=default_visible,
                                multi=True,
                                placeholder="Select visible labels",
                                clearable=False,
                                style=dropdown_compact
                            ),
                            html.Button("Reset", id="reset-btn",
                                        style={"background": "#2563EB", "color": "white",
                                               "border": "none", "borderRadius": "6px",
                                               **button_compact}),
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
                    html.Div(
                        style=panel_card,
                        children=[
                            dcc.Tabs(
                                id="tool-tabs",
                                value="roi",
                                children=[
                                    dcc.Tab(
                                        label="ROI",
                                        value="roi",
                                        children=[
                                            html.Div(
                                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "padding": "4px 0"},
                                                children=[
                                                    html.Div("ROI", style={"fontWeight": "600"}),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            html.Label("P1 (i, j, k)", style={"flex": "1 1 auto"}),
                                                            html.Button("Set from view", id="p1-set-btn",
                                                                        style={"background": "#1f2937", "color": "#e5e7eb",
                                                                               "border": "1px solid #2b3347", "borderRadius": "6px",
                                                                               **button_compact})
                                                        ]
                                                    ),
                                                    dcc.Slider(id="p1-x", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag", marks=None, className="slider-x"),
                                                    dcc.Slider(id="p1-y", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag", marks=None, className="slider-y"),
                                                    dcc.Slider(id="p1-z", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag", marks=None, className="slider-z"),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            dcc.Input(id="p1-color", type="color", value=COLOR_SCHEME["roi_p1"], style=color_chip),
                                                            html.Div("Opacity", style={"fontSize": "11px", "minWidth": "54px"}),
                                                            html.Div(
                                                                dcc.Slider(id="p1-opacity", min=0.05, max=1.0, step=0.05, value=ROI_DEFAULTS["p1_opacity"], updatemode="drag", marks=None),
                                                                style=slider_compact
                                                            )
                                                        ]
                                                    ),
                                                    html.Div(
                                                        style={**control_row, "marginTop": "6px"},
                                                        children=[
                                                            html.Label("P2 (i, j, k)", style={"flex": "1 1 auto"}),
                                                            html.Button("Set from view", id="p2-set-btn",
                                                                        style={"background": "#1f2937", "color": "#e5e7eb",
                                                                               "border": "1px solid #2b3347", "borderRadius": "6px",
                                                                               **button_compact})
                                                        ]
                                                    ),
                                                    dcc.Slider(id="p2-x", min=0, max=Nx-1, step=1, value=x_mid, updatemode="drag", marks=None, className="slider-x"),
                                                    dcc.Slider(id="p2-y", min=0, max=Ny-1, step=1, value=y_mid, updatemode="drag", marks=None, className="slider-y"),
                                                    dcc.Slider(id="p2-z", min=0, max=Nz-1, step=1, value=z_mid, updatemode="drag", marks=None, className="slider-z"),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            dcc.Input(id="p2-color", type="color", value=COLOR_SCHEME["roi_p2"], style=color_chip),
                                                            html.Div("Opacity", style={"fontSize": "11px", "minWidth": "54px"}),
                                                            html.Div(
                                                                dcc.Slider(id="p2-opacity", min=0.05, max=1.0, step=0.05, value=ROI_DEFAULTS["p2_opacity"], updatemode="drag", marks=None),
                                                                style=slider_compact
                                                            )
                                                        ]
                                                    ),
                                                    html.Label("ROI line", style={"marginTop": "6px"}),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            dcc.Input(id="roi-line-color", type="color", value=COLOR_SCHEME["roi_line"], style=color_chip),
                                                            html.Div("Opacity", style={"fontSize": "11px", "minWidth": "54px"}),
                                                            html.Div(
                                                                dcc.Slider(id="roi-line-opacity", min=0.05, max=1.0, step=0.05, value=ROI_DEFAULTS["line_opacity"], updatemode="drag", marks=None),
                                                                style=slider_compact
                                                            )
                                                        ]
                                                    ),
                                                    html.Label("ROI sphere", style={"marginTop": "6px"}),
                                                    html.Div(
                                                        style=control_row,
                                                        children=[
                                                            dcc.Input(id="roi-sphere-color", type="color", value=COLOR_SCHEME["roi_sphere"], style=color_chip),
                                                            html.Div("Opacity", style={"fontSize": "11px", "minWidth": "54px"}),
                                                            html.Div(
                                                                dcc.Slider(id="roi-sphere-opacity", min=0.05, max=1.0, step=0.05, value=ROI_DEFAULTS["sphere_opacity"], updatemode="drag", marks=None),
                                                                style=slider_compact
                                                            )
                                                        ]
                                                    )
                                                ]
                                            )
                                        ]
                                    ),
                                    dcc.Tab(
                                        label="Stenosis",
                                        value="stenosis",
                                        children=[
                                            html.Div(
                                                style={"display": "flex", "flexDirection": "column", "gap": "8px", "padding": "4px 0"},
                                                children=[
                                                    html.Div("ROI Morphology", style={"fontWeight": "600"}),
                                                    html.Div("Target label", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    dcc.Dropdown(
                                                        id="target-label",
                                                        options=editable_options,
                                                        value=target_default,
                                                        clearable=False,
                                                        style=dropdown_compact
                                                    ),
                                                    html.Div("Scale factor (diameter)", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    html.Div(id="roi-scale-display", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    dcc.Slider(
                                                        id="roi-scale-pos",
                                                        min=0.0,
                                                        max=1.0,
                                                        step=0.001,
                                                        value=0.5,
                                                        updatemode="drag",
                                                        marks=None
                                                    ),
                                                    dcc.Input(
                                                        id="roi-scale-input",
                                                        type="number",
                                                        min=0.001,
                                                        max=99.999,
                                                        step=0.001,
                                                        value=1.000,
                                                        style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                               "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                               "padding": "4px 6px"}
                                                    ),
                                                    html.Div("Gaussian profile", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    html.Div(
                                                        style={"display": "flex", "flexDirection": "column", "gap": "6px"},
                                                        children=[
                                                            html.Div(
                                                                style={"display": "flex", "alignItems": "center", "gap": "6px"},
                                                                children=[
                                                                    html.Div("Steepness k", style={"fontSize": "12px", "opacity": "0.85"}),
                                                                    html.Span("i", title="Steepness for Gaussian profile (larger k = sharper).",
                                                                              style={"fontSize": "11px", "opacity": "0.7", "cursor": "help"})
                                                                ]
                                                            ),
                                                            dcc.Input(
                                                                id="shape-k",
                                                                type="number",
                                                                min=0.1,
                                                                max=100.0,
                                                                step=0.1,
                                                                value=10.0,
                                                                style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                       "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                       "padding": "4px 6px"}
                                                            )
                                                        ]
                                                    ),
                                                    html.Div("Transition window", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    dcc.RangeSlider(
                                                        id="shape-window",
                                                        min=0.0,
                                                        max=1.0,
                                                        step=0.01,
                                                        value=[0.25, 0.75],
                                                        allowCross=False,
                                                        marks=None
                                                    ),
                                                    dcc.Checklist(
                                                        id="shape-symmetric",
                                                        options=[{"label": " Symmetric window", "value": "on"}],
                                                        value=["on"],
                                                        inputStyle={"marginRight": "6px"}
                                                    ),
                                                    html.Div("< 1 erosion, > 1 dilation", style={"fontSize": "11px", "opacity": "0.7"}),
                                                    html.Button("Apply", id="roi-scale-apply",
                                                                style={"background": "#16A34A", "color": "white",
                                                                       "border": "none", "borderRadius": "6px",
                                                                       **button_compact})
                                                    ,
                                                    html.Div("Batch export", style={"fontWeight": "600", "marginTop": "8px"}),
                                                    html.Div("Output folder", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    dcc.Input(
                                                        id="roi-batch-out-dir",
                                                        type="text",
                                                        value="outputs/stenosis_batch",
                                                        style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                               "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                               "padding": "4px 6px"}
                                                    ),
                                                    html.Div("Scale factor range", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    html.Div(
                                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "6px",
                                                               "fontSize": "11px", "opacity": "0.7"},
                                                        children=["min", "max", "count"]
                                                    ),
                                                    html.Div(
                                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "6px"},
                                                        children=[
                                                            dcc.Input(id="roi-batch-scale-min", type="number", value=0.8, step=0.01,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"}),
                                                            dcc.Input(id="roi-batch-scale-max", type="number", value=1.2, step=0.01,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"}),
                                                            dcc.Input(id="roi-batch-scale-count", type="number", value=5, step=1,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"})
                                                        ]
                                                    ),
                                                    html.Div("Gaussian k range", style={"fontSize": "12px", "opacity": "0.85"}),
                                                    html.Div(
                                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "6px",
                                                               "fontSize": "11px", "opacity": "0.7"},
                                                        children=["min", "max", "count"]
                                                    ),
                                                    html.Div(
                                                        style={"display": "grid", "gridTemplateColumns": "1fr 1fr 1fr", "gap": "6px"},
                                                        children=[
                                                            dcc.Input(id="roi-batch-k-min", type="number", value=6.0, step=0.1,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"}),
                                                            dcc.Input(id="roi-batch-k-max", type="number", value=14.0, step=0.1,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"}),
                                                            dcc.Input(id="roi-batch-k-count", type="number", value=5, step=1,
                                                                      style={"width": "100%", "backgroundColor": "#0f1115", "color": "#e6e6e6",
                                                                             "border": "1px solid #2a2f3a", "borderRadius": "6px",
                                                                             "padding": "4px 6px"})
                                                        ]
                                                    ),
                                                    html.Button("Export batch", id="roi-batch-export",
                                                                style={"background": "#0EA5E9", "color": "white",
                                                                       "border": "none", "borderRadius": "6px",
                                                                       **button_compact}),
                                                    dcc.Checklist(
                                                        id="roi-batch-stl",
                                                        options=[{"label": " Export STL", "value": "on"}],
                                                        value=["on"],
                                                        inputStyle={"marginRight": "6px"}
                                                    ),
                                                    html.Div(id="roi-batch-status", style={"fontSize": "12px", "opacity": "0.85"})
                                                ]
                                            )
                                        ]
                                    )
                                ]
                            )
                        ]
                    ),
                    html.Div(id="status", style={"fontSize": "12px", "marginTop": "4px"})
                ]
            ),
            html.Div(
                style={"display": "grid", "gridTemplateColumns": "1fr 1fr",
                       "gridTemplateRows": "1fr 1fr", "gap": "10px"},
                children=[
                    dcc.Graph(id="x-view"), dcc.Graph(id="y-view"),
                    dcc.Graph(id="z-view"), dcc.Graph(id="threeD-view"),
                    dcc.Store(id="camera-store"),
                    dcc.Store(id="wheel-store"),
                    dcc.Store(id="hover-store"),
                    dcc.Store(id="catch-store")
                ]
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

    def reset_all_layers() -> None:
        for layer_id, state in layer_states.items():
            state["current"] = state["original"].copy()
            refresh_layer_trace(layer_id)

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
        Output("x-view", "figure"),
        Output("y-view", "figure"),
        Output("z-view", "figure"),
        Output("threeD-view", "figure"),
        Output("camera-store", "data"),
        Input("mask-check", "value"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("p1-x", "value"),
        Input("p1-y", "value"),
        Input("p1-z", "value"),
        Input("p2-x", "value"),
        Input("p2-y", "value"),
        Input("p2-z", "value"),
        Input("p1-color", "value"),
        Input("p2-color", "value"),
        Input("p1-opacity", "value"),
        Input("p2-opacity", "value"),
        Input("roi-line-color", "value"),
        Input("roi-line-opacity", "value"),
         Input("roi-sphere-color", "value"),
         Input("roi-sphere-opacity", "value"),
        Input("roi-scale-pos", "value"),
        Input("roi-scale-apply", "n_clicks"),
        Input("shape-k", "value"),
        Input("shape-window", "value"),
        Input("target-label", "value"),
        Input("threeD-view", "relayoutData"),
        Input("reset-btn", "n_clicks"),
        prevent_initial_call=False
    )
    def update(mask_values, x_idx, y_idx, z_idx,
               p1_x, p1_y, p1_z,
               p2_x, p2_y, p2_z,
               p1_color, p2_color,
               p1_opacity, p2_opacity,
               line_color, line_opacity,
             sphere_color, sphere_opacity,
             scale_pos, scale_apply_clicks,
             shape_k,
             shape_window,
             target_label,
             relayout_data, n_clicks):
        triggered = [t["prop_id"] for t in (callback_context.triggered or [])]
        reset_triggered = "reset-btn.n_clicks" in triggered
        if reset_triggered:
            reset_all_layers()

        center_idx = (
            int(np.clip(round((p1_x + p2_x) / 2.0), 0, Nx - 1)),
            int(np.clip(round((p1_y + p2_y) / 2.0), 0, Ny - 1)),
            int(np.clip(round((p1_z + p2_z) / 2.0), 0, Nz - 1))
        )
        radius_vox = int(round(0.5 * float(np.linalg.norm(
            np.array([p2_x - p1_x, p2_y - p1_y, p2_z - p1_z], dtype=float)
        ))))

        if (not reset_triggered) and ("roi-scale-apply.n_clicks" in triggered):
            sf = _pos_to_scale(scale_pos if scale_pos is not None else 0.5)
            sf = max(0.001, min(99.999, sf))
            target_label = target_label if target_label in layer_states else target_default
            layer_states[target_label]["current"] = apply_roi_scale(
                layer_states[target_label]["current"],
                center_idx,
                radius_vox,
                sf,
                shape_k=10.0 if shape_k is None else float(shape_k),
                shape_window=(shape_window[0], shape_window[1]) if shape_window else (0.25, 0.75)
            )
            apply_exclusive_conflicts(target_label)
            refresh_layer_trace(target_label)

        mask_values = mask_values or []
        visible_ids = [layer_id for layer_id in mask_values if layer_id in layer_states]

        p1 = _voxel_center(grid, p1_x, p1_y, p1_z)
        p2 = _voxel_center(grid, p2_x, p2_y, p2_z)
        center = ((p1[0] + p2[0]) / 2.0, (p1[1] + p2[1]) / 2.0, (p1[2] + p2[2]) / 2.0)
        radius = 0.5 * float(np.linalg.norm(np.array(p2) - np.array(p1)))

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

        valid_traces = [mesh_trace] if mesh_trace is not None else []
        valid_traces.extend(
            layer_states[layer_id]["trace"]
            for layer_id in visible_ids
            if layer_states[layer_id]["trace"] is not None
        )
        extent_x, extent_y, extent_z = grid["extent_mm"]
        aspectratio = _aspectratio_from_extents((extent_x, extent_y, extent_z))

        fig3d = go.Figure(data=valid_traces)
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
        fig3d.add_trace(go.Scatter3d(
            x=[p1[0]], y=[p1[1]], z=[p1[2]],
            mode="markers",
            marker=dict(size=6, color=p1_color, opacity=p1_opacity),
            name="P1",
            showlegend=False,
            hoverinfo="skip"
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[p2[0]], y=[p2[1]], z=[p2[2]],
            mode="markers",
            marker=dict(size=6, color=p2_color, opacity=p2_opacity),
            name="P2",
            showlegend=False,
            hoverinfo="skip"
        ))
        fig3d.add_trace(go.Scatter3d(
            x=[p1[0], p2[0]], y=[p1[1], p2[1]], z=[p1[2], p2[2]],
            mode="lines",
            line=dict(color=line_color, width=4),
            opacity=line_opacity,
            name="ROI line",
            showlegend=False,
            hoverinfo="skip"
        ))
        sphere = _sphere_surface(center, radius, sphere_color, sphere_opacity)
        if sphere is not None:
            fig3d.add_trace(sphere)
        uirev = 0 if n_clicks is None else n_clicks
        camera_state = None
        if isinstance(relayout_data, dict):
            if "scene.camera" in relayout_data:
                camera_state = relayout_data.get("scene.camera")
            elif any(k.startswith("scene.camera") for k in relayout_data.keys()):
                camera_state = {
                    k.split("scene.camera.", 1)[-1]: v
                    for k, v in relayout_data.items()
                    if k.startswith("scene.camera.")
                }
        fig3d.update_layout(
            margin=dict(l=0, r=0, b=0, t=0),
            uirevision=uirev,
            scene=dict(
                uirevision=uirev,
                bgcolor="#000000",
                aspectmode="manual",
                aspectratio=aspectratio,
                xaxis=dict(title="X (mm)", range=[-extent_x/2, extent_x/2], showgrid=False, showspikes=False),
                yaxis=dict(title="Y (mm)", range=[-extent_y/2, extent_y/2], showgrid=False, showspikes=False),
                zaxis=dict(title="Z (mm)", range=[-extent_z/2, extent_z/2], showgrid=False, showspikes=False),
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

    @app.callback(
        Output("p2-x", "value"),
        Output("p2-y", "value"),
        Output("p2-z", "value"),
        Input("p2-set-btn", "n_clicks"),
        State("catch-store", "data"),
        State("p2-x", "value"),
        State("p2-y", "value"),
        State("p2-z", "value"),
        prevent_initial_call=True
    )
    def set_p2_from_catch(n_clicks, catch_data, p2_x, p2_y, p2_z):
        if not n_clicks or not catch_data:
            raise PreventUpdate
        return catch_data.get("i", p2_x), catch_data.get("j", p2_y), catch_data.get("k", p2_z)

    @app.callback(
        Output("status", "children"),
        Input("mask-check", "value"),
        Input("x-slider", "value"),
        Input("y-slider", "value"),
        Input("z-slider", "value"),
        Input("p1-x", "value"),
        Input("p1-y", "value"),
        Input("p1-z", "value"),
        Input("p2-x", "value"),
        Input("p2-y", "value"),
        Input("p2-z", "value"),
        Input("hover-store", "data"),
        Input("catch-store", "data"),
        Input("target-label", "value"),
        prevent_initial_call=False
    )
    def update_status(mask_values, x_idx, y_idx, z_idx,
                      p1_x, p1_y, p1_z, p2_x, p2_y, p2_z,
                      hover_data, catch_data, target_label):
        hover_txt = "Hover: -"
        if isinstance(hover_data, dict):
            hover_txt = f"Hover: ({hover_data.get('i')},{hover_data.get('j')},{hover_data.get('k')})"
        catch_txt = "Catch: -"
        if isinstance(catch_data, dict):
            catch_txt = f"Catch: ({catch_data.get('i')},{catch_data.get('j')},{catch_data.get('k')})"

        active_names = [
            layer_states[layer_id]["name"]
            for layer_id in (mask_values or [])
            if layer_id in layer_states
        ]
        target_name = layer_states.get(target_label, layer_states[target_default])["name"]

        return (
            f"Active labels: {', '.join(active_names) if active_names else '-'} | Target: {target_name}"
            f" | X={x_idx}, Y={y_idx}, Z={z_idx}"
            f" | P1=({p1_x},{p1_y},{p1_z}) P2=({p2_x},{p2_y},{p2_z})"
            f" | {hover_txt} | {catch_txt}"
        )

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
    out_npz: str = "outputs/masks_demo.npz",
    show: bool = True,
    mc_map: str = "xyz",
    viewer: str = "dash",
    port: int = 8050,
    export_stl: bool = False,
    stl_dir: str | None = None,
    vti_array: str | None = None,
    vti_background: float = 0.0,
    vti_background_eps: float = 0.0,
    vti_max_dim: int = 160,
    vti_max_labels: int = 64,
):
    input_path = Path(in_mesh_path).resolve()
    if not input_path.exists():
        err(f"Input not found: {in_mesh_path}")
        raise FileNotFoundError(in_mesh_path)

    suffix = input_path.suffix.lower()
    if suffix == ".vti":
        layers, grid, metadata = load_vti_label_layers(
            str(input_path),
            array_name=vti_array,
            background=vti_background,
            background_eps=vti_background_eps,
            max_dim=vti_max_dim,
            max_labels=vti_max_labels,
        )
        Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
        layer_arrays = {
            f"mask_{layer['id']}": layer["mask"].astype(np.uint8)
            for layer in layers
        }
        np.savez_compressed(
            out_npz,
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
        )
        ok(f"VTI label masks saved (uint8) → {out_npz}")

        if export_stl:
            stl_root = Path(stl_dir) if stl_dir else Path(out_npz).parent
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
                dataset_name=input_path.name,
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
    mesh = center_mesh(mesh)

    # Grid (power-of-two cubic grid)
    grid = make_cube_grid_from_mesh(mesh, spacing=spacing, n=n, margin_frac=margin_frac)

    # Winding-based masks
    inside_u8, on_u8, out_u8 = classify_by_winding(mesh, grid, band=0.6)

    # Save masks in a single compressed NPZ with spacing/origin like core.run_pipeline
    Path(out_npz).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(out_npz, inside=inside_u8, on=on_u8, out=out_u8,
                        spacing=np.array(grid["spacing"]), origin=np.array(grid["origin"]))
    ok(f"Masks saved (uint8) → {out_npz}")

    if export_stl:
        stl_root = Path(stl_dir) if stl_dir else Path(out_npz).parent
        stl_root.mkdir(parents=True, exist_ok=True)
        save_mask_stl(inside_u8, grid, stl_root / "inside.stl", name="inside")
        save_mask_stl(on_u8, grid, stl_root / "on.stl", name="on")
        save_mask_stl(out_u8, grid, stl_root / "out.stl", name="out")
        ok(f"STL export complete → {stl_root}")

    # Viewer
    if show:
        if viewer == "dash":
            show_viewer_dash(mesh, inside_u8, on_u8, out_u8, grid, port=port)
        else:
            # fallback: use the 3-D html viewer (not implemented separately here)
            show_viewer_dash(mesh, inside_u8, on_u8, out_u8, grid, port=port)

# ---------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------
def main():
    import argparse

    ap = argparse.ArgumentParser(description="Standalone FakeCT pipeline for STL meshes or VTI voxel labels")
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
    ap.add_argument("--no-show", action="store_true",
                    help="Do not open viewer")
    ap.add_argument("--viewer", choices=["dash","html"], default="dash",
                    help="Viewer type: 'dash' or 'html'.")
    ap.add_argument("--port", type=int, default=8050,
                    help="Port for Dash viewer (default 8050).")
    ap.add_argument("--export-stl", action="store_true",
                    help="Export inside/on/out masks as STL files.")
    ap.add_argument("--stl-dir", default=None,
                    help="Directory for STL export (defaults to --out folder).")
    ap.add_argument("--vti-array", default=None,
                    help="VTI DataArray name to read. Defaults to the VTI Scalars array or first array.")
    ap.add_argument("--vti-background", type=float, default=0.0,
                    help="Background scalar/label value for VTI label extraction.")
    ap.add_argument("--vti-background-eps", type=float, default=0.0,
                    help="Tolerance for matching floating-point VTI background values.")
    ap.add_argument("--vti-max-dim", type=int, default=160,
                    help="Maximum sampled VTI dimension for browser interaction. Use 0 for full resolution.")
    ap.add_argument("--vti-max-labels", type=int, default=64,
                    help="Maximum number of discrete non-background VTI labels to split into layers.")

    args = ap.parse_args()

    spacing = None if (args.spacing is None or args.spacing <= 0) else float(args.spacing)

    try:
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
            export_stl=bool(args.export_stl),
            stl_dir=args.stl_dir,
            vti_array=args.vti_array,
            vti_background=args.vti_background,
            vti_background_eps=args.vti_background_eps,
            vti_max_dim=args.vti_max_dim,
            vti_max_labels=args.vti_max_labels,
        )
    except (RuntimeError, ValueError) as e:
        err(str(e))
        sys.exit(1)

if __name__ == "__main__":
    main()
