#!/usr/bin/env python3
# ------------------------------------------------------------
# stenosis.py — Local morphological expand/shrink (OpenVDB-style)
#
# Example:
#   python stenosis.py --in in.npz --operation shrink \
#       --kernel sphere --radius 5 --center 80,60,45 --iterations 3
#
#   python stenosis.py --in in.npz --operation expand \
#       --kernel cube --radius 2 --iterations 1
# ------------------------------------------------------------

import numpy as np
import argparse
from scipy.ndimage import binary_dilation, binary_erosion, binary_fill_holes
from scipy.ndimage import generate_binary_structure


# ------------------------------------------------------------
# Utilities
# ------------------------------------------------------------
def make_kernel(shape: str, radius: int):
    """
    Create a 3D structuring element (kernel) for morphological operations.
    shape ∈ {"cross", "cube", "sphere"}
    """
    if shape == "cross":
        # 6-connected
        kernel = np.zeros((3, 3, 3), dtype=bool)
        kernel[1, 1, 0] = kernel[1, 1, 2] = True
        kernel[1, 0, 1] = kernel[1, 2, 1] = True
        kernel[0, 1, 1] = kernel[2, 1, 1] = True
        return kernel
    elif shape == "cube":
        # Full 26-connected cube
        size = 2 * radius + 1
        return np.ones((size, size, size), dtype=bool)
    elif shape == "sphere":
        # Euclidean sphere kernel
        grid = np.indices((2 * radius + 1,) * 3) - radius
        return (np.sqrt((grid ** 2).sum(0)) <= radius)
    else:
        raise ValueError(f"Unknown kernel shape: {shape}")


def make_roi_mask(shape, center=None, radius=None):
    """
    Create ROI mask (boolean 3D array). If center/radius not given → full volume.
    """
    Nz, Ny, Nx = shape
    if center is None or radius is None:
        return np.ones(shape, dtype=bool)

    cx, cy, cz = center
    zz, yy, xx = np.indices(shape)
    roi = (xx - cx) ** 2 + (yy - cy) ** 2 + (zz - cz) ** 2 <= radius ** 2
    return roi


def apply_operation(inside, operation, kernel, iterations, roi_mask):
    """
    Perform iterative expand/shrink within ROI.
    """
    result = inside.copy().astype(bool)
    op_fn = binary_dilation if operation == "expand" else binary_erosion

    for _ in range(iterations):
        updated = op_fn(result, structure=kernel)
        result = np.where(roi_mask, updated, result)

    # Clean up small voids/holes if any
    result = binary_fill_holes(result)
    return result.astype(np.uint8)


# ------------------------------------------------------------
# Main CLI
# ------------------------------------------------------------
def main():
    parser = argparse.ArgumentParser(
        description="Apply localized morphological expand/shrink to a 3D binary mask (.npz)"
    )
    parser.add_argument("--in", dest="infile", required=True,
                        help="Input .npz file containing 'inside' or 'inside_u8' array")
    parser.add_argument("--out", dest="outfile", default="in_updated.npz",
                        help="Output .npz file (default: in_updated.npz)")
    parser.add_argument("--operation", choices=["expand", "shrink"], required=True,
                        help="Morphological operation: expand (dilate) or shrink (erode)")
    parser.add_argument("--kernel", choices=["cross", "cube", "sphere"], default="sphere",
                        help="Kernel shape (default: sphere)")
    parser.add_argument("--radius", type=int, default=1,
                        help="Radius of kernel (and ROI if --center given)")
    parser.add_argument("--iterations", type=int, default=1,
                        help="Number of iterative passes")
    parser.add_argument("--center", type=str, default=None,
                        help="Optional voxel-space center (x,y,z) for local operation. "
                             "If omitted, applies globally.")

    args = parser.parse_args()

    # ------------------------------------------------------------
    # Load mask
    # ------------------------------------------------------------
    data = np.load(args.infile)
    inside_key = "inside" if "inside" in data else "inside_u8"
    inside = data[inside_key].astype(bool)
    print(f"[stenosis] Loaded {args.infile} | shape={inside.shape}")

    # ------------------------------------------------------------
    # Build kernel & ROI
    # ------------------------------------------------------------
    kernel = make_kernel(args.kernel, args.radius)
    center = None
    if args.center:
        try:
            cx, cy, cz = [int(v) for v in args.center.split(",")]
            center = (cx, cy, cz)
        except Exception:
            raise ValueError("--center must be formatted as x,y,z (integers)")

    roi_mask = make_roi_mask(inside.shape, center=center, radius=args.radius if center else None)

    # ------------------------------------------------------------
    # Perform operation
    # ------------------------------------------------------------
    print(f"[stenosis] Operation={args.operation} | kernel={args.kernel} | "
          f"radius={args.radius} | iterations={args.iterations} | "
          f"center={center or 'global'}")

    updated = apply_operation(
        inside=inside,
        operation=args.operation,
        kernel=kernel,
        iterations=args.iterations,
        roi_mask=roi_mask
    )

    # ------------------------------------------------------------
    # Save
    # ------------------------------------------------------------
    np.savez_compressed(args.outfile, inside_u8=updated.astype(np.uint8))
    print(f"[stenosis] Saved updated mask → {args.outfile}")


if __name__ == "__main__":
    main()
