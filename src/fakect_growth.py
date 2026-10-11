"""Bounded edit regions and original-voxel ancestry for selection-only ROIs."""
import numpy as np
from scipy import ndimage
from fakect_limits import MAX_NATIVE_EDIT_VOXELS


MAX_VOXELS = MAX_NATIVE_EDIT_VOXELS
LINEAGE_SEMANTICS = (
    'Each current target voxel stores its original target ancestor as a zero-based '
    'C-order flat index in the native k,j,i crop; -1 means non-target. Each edit '
    'selects ancestors inside its fixed original ROI, including their surviving '
    'offspring outside that ROI. Dilation inherits the winning input-state seed '
    'ancestor; erosion clears ancestry. Other target voxels never become new seeds '
    'merely by entering an edit region or touching the selected structure.')


def selection_mode(config):
    """Determine whether ROIs select original ancestors or bound edit geometry.

    Parameters
    ----------
    config : dict
        Normalized configuration sections; paths are resolved Path objects.

    Returns
    -------
    bool
        True for recipe.roi_role=selection; False for the default boundary role.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    role = config.get('recipe', {}).get('roi_role', 'boundary')
    if role not in ('boundary', 'selection'):
        raise ValueError('recipe.roi_role must be boundary or selection')
    return role == 'selection'


def _mask(value, name):
    """Validate a bounded native Boolean crop without copying its values.

    Parameters
    ----------
    value : array_like
        Three-dimensional Boolean crop no larger than the native edit budget.
    name : str
        Field name used in validation errors.

    Returns
    -------
    ndarray of bool
        Validated k,j,i view or array; storage may be shared with value.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    value = np.asarray(value)
    if value.dtype != np.bool_ or value.ndim != 3 or not 0 < value.size <= MAX_VOXELS:
        raise ValueError(f'{name} must be a bounded native Boolean crop')
    return value


def initial_origins(candidates):
    """Assign each original target voxel its own native crop ancestry index.

    Parameters
    ----------
    candidates : ndarray of bool
        Bounded native k,j,i target membership.

    Returns
    -------
    ndarray of int32
        C-order crop-flat ancestor index for each target voxel; -1 elsewhere.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    candidates = _mask(candidates, 'candidates')
    return np.where(candidates, np.arange(candidates.size, dtype=np.int32).reshape(candidates.shape), -1)


def lineage_selection(origins, original_selected):
    """Select surviving target voxels whose original ancestors belong to a fixed ROI.

    Parameters
    ----------
    origins : ndarray of signed int
        Native crop-flat original ancestor index per current target voxel; -1 denotes non-
        target.
    original_selected : ndarray of bool
        Fixed original target selection in the same native crop.

    Returns
    -------
    ndarray of bool
        Current target membership inherited from the fixed original selection.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    selected = _mask(original_selected, 'original_selected')
    origins = np.asarray(origins)
    if (origins.shape != selected.shape or origins.dtype.kind != 'i'
            or np.any(origins < -1) or np.any(origins >= selected.size)):
        raise ValueError('Target ancestry must contain matching native crop indices or -1')
    return (origins >= 0) & selected.ravel()[np.maximum(origins, 0)]


def expanded_region(selected, distance_mm, spacing_ijk_mm):
    """Construct a physical Euclidean growth envelope around tracked target voxels.

    This footprint is computed from tracked target voxels, not all same-tissue
    voxels in a padded ROI. It adds room; it does not grant the distance budget,
    override tissue resistance, or change the profile and directional factors.

    Parameters
    ----------
    selected : ndarray of bool
        Current tracked target voxels in a bounded native k,j,i crop.
    distance_mm : float
        Finite nonnegative envelope expansion in millimetres.
    spacing_ijk_mm : sequence of float
        Positive physical voxel spacing in i,j,k order, in millimetres.

    Returns
    -------
    ndarray of bool
        New same-shape envelope; a zero distance or empty selection returns a copy of
        selected.

    Raises
    ------
    ValueError
        If the supplied values violate the documented configuration or geometry constraints.
    """
    selected = _mask(selected, 'selected')
    spacing = np.asarray(spacing_ijk_mm, dtype=float)
    distance = float(distance_mm)
    if (spacing.shape != (3,) or not np.isfinite(spacing).all() or np.any(spacing <= 0)
            or not np.isfinite(distance) or distance < 0):
        raise ValueError('Growth envelope requires positive finite spacing and nonnegative finite distance')
    if not selected.any() or distance == 0:
        return selected.copy()
    return ndimage.distance_transform_edt(~selected, sampling=spacing[::-1]) <= distance + 1e-9


def inherit_origins(origins, result, selected, candidates_after):
    """Transfer original target ancestry through accepted dilation and erosion changes.

    Parameters
    ----------
    origins : ndarray of signed int
        Input-state original ancestor indices, with -1 for non-target.
    result : dict
        Morphology result containing accepted added/removed masks and winning
        target_seed_index.
    selected : ndarray of bool
        Tracked input-state target voxels permitted as dilation seeds.
    candidates_after : ndarray of bool
        Full target membership after the accepted pass.

    Returns
    -------
    ndarray of signed int
        Updated copy of origins; additions inherit their winning seed ancestor and removals
        receive -1.

    Raises
    ------
    ValueError
        If native shapes, mask types or ancestry indices are invalid.
    RuntimeError
        If additions use an unselected seed or ancestry disagrees with final target
        membership.
    """
    selected = _mask(selected, 'selected')
    after = _mask(candidates_after, 'candidates_after')
    origins = np.asarray(origins)
    # This also checks index bounds before indexing the input ancestor array.
    lineage_selection(origins, selected)
    added, removed = result['added_mask'], result['removed_mask']
    seeds = np.asarray(result['target_seed_index'])
    if seeds.shape != selected.shape or seeds.dtype.kind != 'i':
        raise ValueError('Dilation seed ancestry must match the native crop')
    winners = seeds[added]
    if (np.any(winners < 0) or np.any(winners >= selected.size)
            or not np.all(selected.ravel()[winners])):
        raise RuntimeError('Dilation ancestry refers to an unselected or invalid target seed')
    updated = origins.copy()
    updated[removed] = -1
    updated[added] = origins.ravel()[winners]
    if after.shape != updated.shape or not np.array_equal(updated >= 0, after):
        raise RuntimeError('Target ancestry disagrees with current target membership')
    return updated
