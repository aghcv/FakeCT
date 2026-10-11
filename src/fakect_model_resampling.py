"""Explicit model-time grid conversion; immutable native pairs are untouched."""
import hashlib
from importlib.metadata import version
import math
from pathlib import Path
import numpy as np


def spacing(values):
    """Validate positive physical voxel spacing for model-time conversion.

    Parameters
    ----------
    values : list or tuple
        Three numeric spacing values; Boolean values are invalid.

    Returns
    -------
    tuple of float
        Finite positive voxel distances in i,j,k order, in millimetres.

    Raises
    ------
    ValueError
        The spacing shape, type, finiteness or positivity is invalid.
    """
    if not isinstance(values, (list, tuple)) or len(values) != 3:
        raise ValueError('target_spacing_mm must contain three positive i,j,k distances')
    if any(isinstance(value, (bool, np.bool_)) for value in values):
        raise ValueError('target_spacing_mm must contain distances, not Boolean values')
    try:
        result = tuple(float(value) for value in values)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('target_spacing_mm must contain three positive finite distances') from exc
    if any(not math.isfinite(value) or value <= 0 for value in result):
        raise ValueError('target_spacing_mm must contain three positive finite distances')
    return result


def output_shape(shape, native_spacing, target_spacing):
    """Compute a bounded output grid aligned at the lower voxel face.

    Parameters
    ----------
    shape : list or tuple of int
        Positive source volume dimensions in k,j,i order.
    native_spacing : sequence of float
        Positive source voxel spacing in i,j,k order, in millimetres.
    target_spacing : sequence of float
        Positive target voxel spacing in i,j,k order, in millimetres.

    Returns
    -------
    tuple of int
        Resampled dimensions in k,j,i order, each at least one voxel.

    Raises
    ------
    ValueError
        Dimensions/spacing are invalid or the target exceeds the 16,777,216-voxel
        budget.
    """
    if (not isinstance(shape, (list, tuple)) or len(shape) != 3 or
            any(isinstance(n, (bool, np.bool_)) or not isinstance(n, (int, np.integer)) or n <= 0
                for n in shape)):
        raise ValueError('Resampling requires three positive integer k,j,i dimensions')
    native, target = spacing(native_spacing)[::-1], spacing(target_spacing)[::-1]
    if any(not math.isfinite(a/b) or not math.isfinite(b/a) or a/b <= 0 or b/a <= 0
           for a, b in zip(native, target)):
        raise ValueError('Native and target spacing ratios must be finite and positive')
    extents = [int(n)*a/b for n, a, b in zip(shape, native, target)]
    if any(not math.isfinite(value) or value > 16_777_216 for value in extents):
        raise ValueError('Model resampling exceeds the 16,777,216 voxel crop limit; increase target_spacing_mm')
    result = tuple(max(1, math.ceil(value-1e-9)) for value in extents)
    if math.prod(result) > 16_777_216:
        raise ValueError('Model resampling exceeds the 16,777,216 voxel crop limit; increase target_spacing_mm')
    return result


def resampling_contract(target_spacing):
    """Record the model-grid convention and its consumer implementation.

    Parameters
    ----------
    target_spacing : sequence of float
        Positive target voxel spacing in i,j,k order, in millimetres.

    Returns
    -------
    dict
        Target spacing, shape/alignment/interpolation rules, implementation hashes and
        dependency versions.

    Notes
    -----
    Pin the grid convention and its consumer without binding generator code.
    """
    source = Path(__file__).resolve()
    return {'schema_version': 'fakect.model-resampling/1',
            'target_spacing_mm': list(spacing(target_spacing)),
            'array_order': 'kji', 'grid_alignment': 'lower_voxel_face',
            'shape_rule': 'ceil(native_count * native_spacing / target_spacing - 1e-9), minimum 1',
            'image_interpolation': 'linear', 'mask_interpolation': 'nearest',
            'boundary_mode': 'nearest', 'prefilter': False,
            'implementation_sha256': {path.name: hashlib.sha256(path.read_bytes()).hexdigest()
                                      for path in (source, source.with_name('fakect_segmentation.py'))},
            'dependency_versions': {'numpy': np.__version__, 'scipy': version('scipy')}}


def resample_pair(image, mask, native_spacing, target_spacing):
    """Convert an immutable native image/mask pair to a physical model grid.

    Parameters
    ----------
    image : numpy.ndarray of float32, shape (K, J, I)
        Finite attenuation volume.
    mask : numpy.ndarray of uint8, shape (K, J, I)
        Matching binary target volume.
    native_spacing : sequence of float
        Positive source voxel spacing in i,j,k order, in millimetres.
    target_spacing : sequence of float
        Positive target voxel spacing in i,j,k order, in millimetres.

    Returns
    -------
    tuple of numpy.ndarray
        float32 linearly interpolated image and uint8 nearest-neighbor mask in k,j,i
        order; identical grids return the input arrays.

    Raises
    ------
    ValueError
        Input dtypes/shapes/values, spacing or output budget is invalid.

    Notes
    -----
    The native arrays are not modified. Lower voxel-face alignment and nearest boundary handling retain partially covered final cells.

    Lower voxel-face aligned grids; linear image, nearest label and boundary.

    A final partially covered output cell uses the nearest source boundary.
    Distances are physical mm; no anatomy-derived normalization or statistics.
    """
    from scipy.ndimage import affine_transform
    if (not isinstance(image, np.ndarray) or not isinstance(mask, np.ndarray) or
            image.ndim != 3 or image.shape != mask.shape or image.dtype != np.float32 or
            mask.dtype != np.uint8 or not np.isfinite(image).all() or
            not np.all((mask == 0) | (mask == 1))):
        raise ValueError('Resampling needs matching finite float32 image and binary uint8 mask volumes')
    native, target = np.asarray(spacing(native_spacing))[::-1], np.asarray(spacing(target_spacing))[::-1]
    shape = output_shape(image.shape, native_spacing, target_spacing)
    if np.array_equal(native, target) and tuple(image.shape) == shape:
        return image, mask
    scale = target/native
    kwargs = dict(matrix=np.diag(scale), offset=(scale-1)/2, output_shape=shape,
                  mode='nearest', prefilter=False)
    return (affine_transform(image, order=1, output=np.float32, **kwargs),
            affine_transform(mask, order=0, output=np.uint8, **kwargs))
