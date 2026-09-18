"""Physical spline frames with explicitly qualified inner/outer bend directions.

The fitting parameter is the ORIGINAL polyline's normalized physical arc
length. Spline arc length controls sampling only; it does not redefine ROI
percentages. Frenet N follows the curvature vector and is never sign-flipped
for visual continuity. Low-curvature samples carry a transported display frame
and are not reliable inner/outer directions.

Numerical references (also recorded in metadata): SciPy 1.13.1 ``splprep`` and
``BSpline`` documentation, and Gallier's differential geometry chapter 19.
"""
import numpy as np
from scipy.integrate import cumulative_trapezoid
from scipy.interpolate import BSpline, splprep


MAX_INPUT_NODES = 512
MAX_FRAME_SAMPLES = 4096
MAX_ARC_GRID = 65_537
REFERENCES = (
    'https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.interpolate.splprep.html',
    'https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.interpolate.BSpline.html',
    'https://www.cis.upenn.edu/~cis6100/gma-v2-chap19.pdf',
)


def _norm(value):
    return np.hypot(np.hypot(value[..., 0], value[..., 1]), value[..., 2])


def _positive_number(value, name, *, allow_zero=False):
    if isinstance(value, (bool, np.bool_)):
        raise ValueError(f'{name} must be a finite number')
    try:
        result = float(value)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError(f'{name} must be a finite number') from exc
    if not np.isfinite(result) or (result < 0 if allow_zero else result <= 0):
        sign = 'nonnegative' if allow_zero else 'positive'
        raise ValueError(f'{name} must be finite and {sign}')
    return result


def _perpendicular(tangent):
    axis = np.eye(3)[int(np.argmin(np.abs(tangent)))]
    normal = axis - np.dot(axis, tangent) * tangent
    return normal / _norm(normal)


def _transport(normal, first_tangent, last_tangent):
    """Minimum rotation between neighboring tangents, followed by reprojection."""
    cross = np.cross(first_tangent, last_tangent)
    cosine = float(np.clip(np.dot(first_tangent, last_tangent), -1., 1.))
    reversal = cosine < -1. + 1e-10
    if reversal:
        # The minimum-rotation axis is ambiguous at 180 degrees. Rotating about
        # the preceding normal leaves it unchanged; the ambiguity is flagged.
        result = normal.copy()
    else:
        result = normal + np.cross(cross, normal) + np.cross(cross, np.cross(cross, normal)) / (1. + cosine)
    result -= np.dot(result, last_tangent) * last_tangent
    length = float(_norm(result))
    return (_perpendicular(last_tangent) if length < 1e-12 else result / length), reversal


def _reference_frame(tangent, frenet_normal, bending):
    reference = np.empty_like(tangent)
    available = np.flatnonzero(bending)
    anchor = int(available[0]) if len(available) else 0
    reference[anchor] = frenet_normal[anchor] if len(available) else _perpendicular(tangent[anchor])
    ambiguous = np.zeros(len(tangent), dtype=bool)
    for indexes in (range(anchor + 1, len(tangent)), range(anchor - 1, -1, -1)):
        for index in indexes:
            previous = index - 1 if index > anchor else index + 1
            reference[index], reversal = _transport(reference[previous], tangent[previous], tangent[index])
            if reversal:
                ambiguous[[previous, index]] = True
    return reference, np.cross(tangent, reference), ambiguous


def _arc_samples(spline, parent_fraction, parent_length, step, maximum):
    """Invert a bounded, dense cumulative spline-speed integral."""
    # At least 32 samples per original segment, plus a step-based refinement.
    # This table is numerical quadrature, not another geometric smoothing step.
    estimated = parent_length / step
    if not np.isfinite(estimated) or estimated > MAX_ARC_GRID:
        raise ValueError('Requested frame spacing exceeds the bounded arc integration budget; increase sample_step_mm')
    grid_count = max(1025, 32 * len(parent_fraction) + 1, int(np.ceil(estimated)) * 8 + 1)
    grid_count = min(grid_count, MAX_ARC_GRID - len(parent_fraction))
    grid = np.unique(np.r_[np.linspace(0., 1., grid_count), parent_fraction])
    speed = _norm(spline(grid, nu=1))
    if not np.all(np.isfinite(speed)):
        raise ValueError('Spline derivatives are not finite')
    cumulative = np.r_[0., cumulative_trapezoid(speed, grid)]
    total = float(cumulative[-1])
    if not np.isfinite(total) or total <= 0:
        raise ValueError('The fitted spline has no finite positive arc length')
    requested_intervals = total / step
    if not np.isfinite(requested_intervals) or requested_intervals > maximum:
        raise ValueError(f'Centerline frame exceeds max_samples={maximum}; increase sample_step_mm')
    intervals = int(np.ceil(requested_intervals))
    count = max(3, intervals + 1)
    if count > maximum:
        raise ValueError(f'Centerline frame requires {count} samples, above max_samples={maximum}; increase sample_step_mm')
    distances = np.linspace(0., total, count)
    # A stationary interval has no arc length and therefore no unique inverse;
    # retain its first parameter value, while explicitly retaining u=1 at end.
    distinct = np.r_[True, np.diff(cumulative) > 0]
    fractions = np.interp(distances, cumulative[distinct], grid[distinct])
    fractions[0], fractions[-1] = 0., 1.
    return fractions, distances, len(grid)


def build_centerline_frame(nodes_ijk, spacing_ijk_mm, *, smoothing_mm=1., sample_step_mm=1.,
                           min_curvature_per_mm=.002, max_samples=MAX_FRAME_SAMPLES):
    """Fit and sample a physical centerline reference, preserving parent u.

    ``smoothing_mm`` is an RMS residual budget at the supplied controls, using
    FITPACK's sum of squared 3D residuals ``s = n * smoothing_mm**2``. It is not
    a guaranteed displacement at every point. At least four controls produce
    a cubic spline; two or three use degree one or two respectively.

    ``normal`` points along true principal curvature where curvature exceeds
    the requested threshold; otherwise it is transported solely for display.
    Consumers MUST gate inner/outer semantics using ``reliable``. Endpoints,
    stationary tangents, and sampled inflection/normal-turn candidates are
    excluded. ``reference_normal`` is a separate transported display frame.
    """
    smoothing = _positive_number(smoothing_mm, 'smoothing_mm', allow_zero=True)
    step = _positive_number(sample_step_mm, 'sample_step_mm')
    threshold = _positive_number(min_curvature_per_mm, 'min_curvature_per_mm', allow_zero=True)
    if (isinstance(max_samples, (bool, np.bool_)) or not isinstance(max_samples, (int, np.integer))
            or not 3 <= max_samples <= MAX_FRAME_SAMPLES):
        raise ValueError(f'max_samples must be an integer between 3 and {MAX_FRAME_SAMPLES}')
    try:
        nodes, spacing = np.asarray(nodes_ijk, dtype=float), np.asarray(spacing_ijk_mm, dtype=float)
    except (ValueError, TypeError, OverflowError) as exc:
        raise ValueError('Centerline coordinates and spacing must be finite numbers') from exc
    if (nodes.ndim != 2 or nodes.shape[1:] != (3,) or not 2 <= len(nodes) <= MAX_INPUT_NODES
            or not np.all(np.isfinite(nodes)) or np.any(nodes < 0)):
        raise ValueError(f'Centerline requires 2..{MAX_INPUT_NODES} finite nonnegative i,j,k controls')
    if spacing.shape != (3,) or not np.all(np.isfinite(spacing)) or np.any(spacing <= 0):
        raise ValueError('Spacing must contain three finite positive i,j,k values')
    with np.errstate(over='ignore', invalid='ignore'):
        xyz = nodes * spacing
        segment_lengths = _norm(np.diff(nodes, axis=0) * spacing)
        parent_distances = np.r_[0., np.cumsum(segment_lengths)]
        budget = float(np.float64(len(nodes)) * np.float64(smoothing) * np.float64(smoothing))
    if (not np.all(np.isfinite(xyz)) or not np.all(np.isfinite(parent_distances))
            or np.any(np.diff(parent_distances) <= 0) or not np.isfinite(budget)):
        raise ValueError('Physical coordinates, distinct segment lengths, and smoothing budget must remain finite')
    parent_length = float(parent_distances[-1])
    parent_fraction = parent_distances / parent_length
    origin = xyz[0].copy()
    degree = min(3, len(nodes) - 1)
    try:
        (tck, _), residual, fit_status, fit_message = splprep(
            (xyz - origin).T, u=parent_fraction, k=degree, s=budget, full_output=1)
    except (TypeError, ValueError, RuntimeError) as exc:
        raise ValueError(f'Centerline smoothing spline failed: {exc}') from exc
    if fit_status > 0:
        raise ValueError(f'Centerline smoothing spline did not converge: {fit_message}')
    spline = BSpline(tck[0], np.asarray(tck[1]).T, tck[2], extrapolate=False)
    fraction, arc_distance, quadrature_count = _arc_samples(spline, parent_fraction, parent_length, step, int(max_samples))
    positions = spline(fraction) + origin
    first = spline(fraction, nu=1)
    second = spline(fraction, nu=2) if degree >= 2 else np.zeros_like(first)
    speed = _norm(first)
    # Use the parent scale as well as sampled speeds. A coarse resampling can
    # happen to land only on stationary points, making all observed speeds
    # roundoff-sized and a purely relative threshold incorrectly trust them.
    speed_floor = max(max(float(speed.max()), parent_length) * 1e-12, np.finfo(float).tiny)
    degenerate = speed <= speed_floor
    tangent = np.empty_like(first)
    tangent[~degenerate] = first[~degenerate] / speed[~degenerate, None]
    # A stationary spline derivative has no tangent. A nearby valid tangent is
    # a finite display fallback only; the sample remains semantically invalid.
    valid = np.flatnonzero(~degenerate)
    for index in np.flatnonzero(degenerate):
        if len(valid):
            nearest = valid[np.argmin(np.abs(fraction[valid] - fraction[index]))]
            tangent[index] = tangent[nearest]
        else:
            segment = min(len(nodes) - 2, max(0, int(np.searchsorted(parent_fraction, fraction[index], side='right')) - 1))
            tangent[index] = (xyz[segment + 1] - xyz[segment]) / segment_lengths[segment]
    acceleration_normal = second - np.sum(second * tangent, axis=1)[:, None] * tangent
    curvature_vector = np.zeros_like(first)
    curvature_vector[~degenerate] = acceleration_normal[~degenerate] / speed[~degenerate, None] ** 2
    curvature = _norm(curvature_vector)
    numerical_floor = 1e-10 / max(parent_length, 1.)
    frenet_defined = (~degenerate) & (curvature > numerical_floor)
    frenet = np.zeros_like(first)
    frenet[frenet_defined] = curvature_vector[frenet_defined] / curvature[frenet_defined, None]
    bending = frenet_defined & (curvature >= threshold)
    reference_normal, reference_binormal, ambiguous_transport = _reference_frame(tangent, frenet, bending)
    inflection = np.zeros(len(fraction), dtype=bool)
    bend_indexes = np.flatnonzero(bending)
    for previous, current in zip(bend_indexes[:-1], bend_indexes[1:]):
        transported, reversal = _transport(frenet[previous], tangent[previous], tangent[current])
        if reversal or np.dot(transported, frenet[current]) < 0:
            # This is an ambiguity flag, not an analytic proof of an inflection:
            # a large unresolved normal turn can produce the same observation.
            inflection[previous:current + 1] = True
    endpoint = np.zeros(len(fraction), dtype=bool)
    endpoint[[0, -1]] = True
    reliable = bending & ~degenerate & ~endpoint & ~inflection & ~ambiguous_transport
    normal = reference_normal.copy()
    normal[bending] = frenet[bending]  # Never sign-align true curvature to the reference frame.
    binormal = np.cross(tangent, normal)
    statuses = np.full(len(fraction), 'transported_low_curvature', dtype='<U36')
    statuses[bending] = 'frenet'
    statuses[bending & endpoint] = 'frenet_endpoint'
    statuses[bending & inflection] = 'frenet_inflection_candidate'
    statuses[degenerate] = 'transported_degenerate_tangent'
    fitted_controls = spline(parent_fraction) + origin
    displacements = _norm(fitted_controls - xyz)
    rms = float(np.sqrt(np.mean(displacements ** 2)))
    output = {
        'xyz_mm': positions, 'parent_fraction': fraction,
        'parent_distance_mm': fraction * parent_length, 'spline_distance_mm': arc_distance,
        'tangent': tangent, 'normal': normal, 'binormal': binormal,
        'curvature_vector_per_mm': curvature_vector, 'curvature_per_mm': curvature,
        'frenet_normal': frenet, 'frenet_defined': frenet_defined, 'reliable': reliable,
        'normal_status': statuses, 'reference_normal': reference_normal,
        'reference_binormal': reference_binormal, 'endpoint': endpoint,
        'inflection_candidate': inflection, 'degenerate_tangent': degenerate,
        'ambiguous_transport': ambiguous_transport,
    }
    if any(not np.all(np.isfinite(value)) for value in output.values() if value.dtype.kind in 'fiu'):
        raise ValueError('Centerline frame contains nonfinite numerical results')
    output['metadata'] = {
        'schema_version': 'fakect-centerline-frame-1', 'input_node_count': len(nodes),
        'sample_count': len(fraction), 'spline_degree': degree, 'smoothing_mm': smoothing,
        'smoothing_residual_budget_mm2': budget, 'fit_residual_sum_mm2': float(residual),
        'fit_rms_displacement_mm': rms, 'fit_max_displacement_mm': float(displacements.max()),
        'parent_total_length_mm': parent_length, 'sampled_spline_length_mm': float(arc_distance[-1]),
        'sample_step_mm': step, 'actual_sample_step_mm': float(arc_distance[-1] / (len(fraction) - 1)),
        'arc_quadrature_samples': quadrature_count,
        'min_curvature_per_mm': threshold, 'numerical_curvature_floor_per_mm': numerical_floor,
        'stationary_speed_floor_mm_per_parent_fraction': speed_floor,
        'reliable_count': int(reliable.sum()), 'reliable_fraction': float(reliable.mean()),
        'low_curvature_count': int((~bending).sum()), 'endpoint_count': int(endpoint.sum()),
        'degenerate_tangent_count': int(degenerate.sum()),
        'inflection_candidate_count': int(inflection.sum()),
        'ambiguous_transport_count': int(ambiguous_transport.sum()),
        'parent_node_fraction': parent_fraction.tolist(),
        'spacing_ijk_mm': spacing.tolist(), 'original_nodes_ijk': nodes.tolist(),
        'fitted_controls_xyz_mm': fitted_controls.tolist(),
        'parameterization': 'Spline parameter u is original polyline physical arc fraction; arc-length resampling does not redefine u',
        'smoothing_semantics': 'RMS 3D residual budget at supplied controls; FITPACK s=n*smoothing_mm^2; no per-point displacement guarantee',
        'sampling_semantics': 'Uniform fitted-spline arc distance from bounded dense trapezoidal speed quadrature and monotone linear inversion',
        'frame_semantics': 'T follows supplied order; N follows principal curvature toward inner bend; B=T cross N; outer=-N',
        'reliability_semantics': 'Inner/outer valid only where reliable: sufficient curvature and no endpoint, stationary tangent, or sampled normal-turn ambiguity',
        'transport_semantics': 'Minimum-rotation reference normals are display-only in low curvature; never substitute them for valid inner/outer directions',
        'stationary_tangent_semantics': 'Nearest valid fitted tangent is display-only fallback; if all samples are stationary, original parent-segment direction is used',
        'inflection_semantics': 'Opposing transported principal normals flag an inflection or unresolved normal-turn candidate; true Frenet normals are not flipped',
        'endpoint_policy': 'First and last fitted samples excluded from reliable inner/outer semantics',
        'limits': {'input_nodes': MAX_INPUT_NODES, 'samples': int(max_samples), 'arc_grid': MAX_ARC_GRID},
        'references': list(REFERENCES), 'source_arrays_modified': False,
    }
    return output
