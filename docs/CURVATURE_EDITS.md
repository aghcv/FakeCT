# Curvature-relative aorta edits

The [curvature INI](../configs/studies/thoracic-aorta-curvature.ini) reuses the
reviewed parent tube saved with recipe-v6 and now writes a
[recipe-v10 report](../outputs/studies/thoracic-aorta/recipe-v10/report.html).
The user's current trial applies 9 mm outward dilation over parent 3–20% with
local Gaussian window 0.1–0.8, then 7 mm inward erosion over 25–35% with window
0.3–0.9. Selection uses the
`artery` category with blank `source_ids`, so mapped artery branches within the
ROI participate without requiring XCAT ID knowledge. No per-ID stiffness
overrides are used. Both steps use one pass and the existing resistance factors.
The erosion now uses `assign_surrounding_tissue=false` to expose released
voxels without inferring a surrounding material; see [diagnostic erosion](DIAGNOSTIC_EROSION.md).
With `roi_role=selection`, offspring may grow beyond the original selector;
see [selection and growth regions](SELECTION_GROWTH.md).
Edit profiles now use physical distance along the original ROI centerline;
see [centerline profiles](CENTERLINE_PROFILES.md) for measurement details.

```bash
cd /home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-curvature.ini --validate-only
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-curvature.ini
```

After changing the file, choose a fresh `[output] directory`, such as
`outputs/studies/thoracic-aorta/recipe-v11`. Each invocation starts from the
original phantom. The two steps within a run consume each other's results.
The curvature INI retains the user's edit parameters. Other recipe and study
INIs retain their previous state.

The earlier [v7 evaluation](integration/2026-09-17/CURVATURE_EDITS.md) records the
previous 3 mm, 25–65% trial. In particular, the current skin mapping of
interior-adjacent `chest_surface` limits erosion, and the broad artery category
also includes pulmonary labels in this ROI. Review those findings with the views.

## Editable controls

These optional recipe settings define the shared directional reference:

```ini
[centerline]
smoothing_mm = 1.0
sample_step_mm = 1.0
min_curvature_per_mm = 0.002
```

Add these lines inside an existing `[edit.NAME]` section to select the outside
of a bend:

```ini
roi = main
path_percent = 25,65
direction = outer
angular_width_deg = 180
```

Use `direction = inner` to select the inside. `operation` independently chooses
dilation or erosion, so either operation can act on either side. Width is the
**full sector angle**, greater than zero and at most 180 degrees: 180 covers at
most one transverse half, 90 a quarter. The factor tapers smoothly from one in
the requested direction to zero at the sector boundaries. To restore ordinary
circumferential editing, omit both directional settings, or use `direction=all`
and remove `angular_width_deg`. Legacy recipes without these fields keep their
existing behavior.

`path_percent` still measures the **original polyline's physical length** from
its first listed node. `point_range` remains an alternative. The smoothed
reference does not alter ROI coordinates, masks, radii, crop, or CT sampling.
Each edit range uses the full original parent path to obtain its directions.
Directional edits require at least four parent points; a selected short range
may contain only two. Named parent tubes remain supported.

The example's numbered INI notes 24–26 explain each new setting; the earlier
notes cover ROI, distance, iteration, profile and tissue controls.

## Geometry and reliability

Coordinates are converted to millimeters using native i,j,k spacing. A cubic
parametric smoothing spline is fitted using original cumulative physical arc
fraction as its parameter. The FITPACK residual budget is
`s = number_of_controls * smoothing_mm**2`. Thus `smoothing_mm` is an RMS
residual budget at the controls, not a maximum displacement of every control
or a smoothing kernel radius. Zero interpolates controls. The report records
RMS and maximum displacement. See [SciPy's spline fitting documentation](https://docs.scipy.org/doc/scipy-1.13.1/reference/generated/scipy.interpolate.splprep.html).

The fitted curve is resampled approximately uniformly by spline arc length,
using bounded numerical integration and inversion of its speed. This retains
the original parent fraction associated with every sample. Analytic spline
derivatives give the tangent T, curvature vector, principal normal N, and
binormal B = T × N. N points toward the local center of curvature; the outer
direction is -N. Reversing point order reverses T and B, while geometric
inner/outer remains the same. These are local curve directions, not patient
left/right axes. The definitions follow [Gallier's differential geometry text,
chapter 19](https://www.cis.upenn.edu/~cis6100/gma-v2-chap19.pdf).

Inner/outer is usable only where curvature meets `min_curvature_per_mm`, with
no endpoint, stationary tangent, or sampled normal-turn ambiguity. The default
0.002/mm corresponds to a 500 mm radius. Low-curvature sections carry a separate
transported frame for possible display/reference use, but never use that
fallback to invent an inner/outer edit direction. True principal normals are
never sign-flipped for visual continuity. Opposing neighboring directions are
flagged as possible inflections or insufficiently resolved normal turns.
Sparse controls can miss bends; the numerical flags do not prove anatomical
validity or detect every possible unsampled inflection.

Voxels map through their closest position on the original parent polyline to
bracketing fitted samples. T and N are interpolated and re-orthogonalized;
unreliable or opposing brackets suppress the edit. Exact sample positions use
that sample's reliability. The offset from the fitted reference is projected
perpendicular to T. Offsets with no defined transverse direction are skipped.
This uses the ROI path as a reference; it does not estimate a lumen centerline
from the segmented vessel. Poorly centered paths can bias the selected side. Branches use the same parent
reference in this trial; inspect ostia and junctions for smoothness and continuity.
Category selection includes only labels mapped to artery and does not repair
missing or misclassified branch labels.

For transverse angle theta from N (inner) or -N (outer), and half the configured
sector width h, the angular weight is `(1 + cos(pi * theta / h)) / 2` for
`theta < h`, otherwise zero. The local edit budget is:

```text
distance_mm × longitudinal_profile × angular_weight × (1 − stiffness)
```

The existing weighted six-neighbor morphology, tissue eligibility and recipient
search then determine the achieved changes. In the current selection role, the
tube and arc range constrain original target ancestors while the growth region
provides space for offspring. In the legacy boundary role they constrain every
change. Small budgets can produce no changed voxels on
a 1 mm grid. Inner erosion removes target on the inner-facing boundary; it does
not move the whole vessel or prescribe a stenosis percentage. The scalar output
remains the existing attenuation-copy proxy.

## Report and saved evidence

The report's **Edits** tab retains the independently toggleable before/after
overlay and adds a centerline frame view. Orbit it and use the legend to toggle
the original target, original controls, fitted path, and T/N/-N/B arrows.
Arrow length is a display scale, not edit distance. Curvature is plotted against
original parent percentage. Static frame and curvature figures are embedded
as fallbacks. The context target is pooled for display; editing remains native.

Each parent has `centerline-<number>-<name>/frame.json`, containing full frame
arrays, control coordinates, physical spacing, sampling settings and reliability
flags. `frame.html`, `frame.png`, and `curvature.png` accompany it. The report
records hashes and fitting details. Each pass's `step.npz` saves
`direction_weight`, `direction_reliable_mask`, and `direction_cosine`, along
with the original and edited labels, effective budget and exact change masks.
Direction summary counts describe the effective ROI, not only vessel voxels.

The frame builder is bounded at 512 input controls, 4096 output samples and a
65,537-point integration grid. The existing two-million-voxel edit crop limit
still applies. Increase `sample_step_mm` if the frame sample budget is exceeded.
Arbitrary circumferential angles, patient-axis directions, mechanical tissue
deformation and anatomy-specific severity calibration remain separate work.
