# Terminal input contract and remaining cohort design

The executable preview authoring standard is now a commented **INI file**:
[`configs/examples/xcat-roi.ini`](../../../configs/examples/xcat-roi.ini).
Run it with `python3 scripts/preview_roi.py --config configs/examples/xcat-roi.ini`.
See the [ROI guide](../../ROI_PREVIEWS.md) for overlays, 3D views and iteration.
The INI uses one setting per line, short note references beside the parameters,
and detailed numbered NOTES at the bottom. Its strict typed loader rejects
unsupported keys. It covers current preview operations; the broader cohort
operations below remain a design proposal.

Historical design example: [`configs/examples/xcat-cohort.v1.draft.json`](../../../configs/examples/xcat-cohort.v1.draft.json).
The JSON file is a design proposal, **not an input accepted by the existing CLI**.
Null ROI coordinates, unreviewed orientation, and empty donor/recipient lists are
intentional: preprocessing can run, but geometry changes must fail validation until
these are resolved. The example left carotid ID comes from the supplied organ table; its
presence and position must be checked for the chosen case/frame.

Use the versioned INI document for human authoring and generated JSON for the
resolved machine representation and provenance. The preview loader implements
typed validation; the planned cohort schema still needs to cover generation,
reassignment and sweeps. Reject unknown keys, ambiguous
units, contradictory selectors, overlapping tissue bins, absent target IDs, and
unresolved source geometry. Absent target IDs block apply/run; discovery previews
must remain available so the user can select another label. Do not silently supply
anatomy or ROI choices.

## Workflow and proposed commands

These are **planned interfaces**, not runnable commands on this baseline:

```text
fakect validate --config run.ini
fakect preprocess --config run.ini
fakect preview --config run.ini
# edit bins, target label IDs, slice locations, ROI, donor/recipient rules; repeat
fakect apply --config run.ini --reviewed-config-hash HASH
fakect plan --config run.ini --out manifest.jsonl
fakect run --manifest manifest.jsonl --task-index N
fakect export-training --manifest accepted.jsonl
```

Preprocessing produces metadata, label tables, distributions, cached preview data,
and a resolved configuration. Preview produces linked histogram and slice views;
selecting a histogram category highlights it in each slice. Every slice shows its
fixed i/j/k index and physical coordinate, in-plane coordinate axes and crosshair,
original scalar, label ID/name, selected category, ROI boundary and protected labels.
The display maps decimated pixels back to **original** i/j/k indices. Z-level means
both k and source-relative z in mm; anatomical orientation names remain explicitly
unverified until landmarks establish them.

After a geometry trial, preview before/after/difference views, donor/recipient counts,
removed/added volumes and unresolved assignments. A review marker is the hash of
configuration + category map + source metadata, not a boolean that survives edits.
Changing any of these invalidates the review. Sweeps always start from the immutable
original, never from the previous sweep result. Cartesian vs random sampling, seed,
job count, output location and cost/memory estimates are printed before submission.

## Geometry and values

- Internal arrays: `[k, j, i] = [z, y, x]`. User index tuples: `[i, j, k]`.
  The notebook uses i=y and j=x in some previews: explicitly transpose that legacy
  convention at its adapter; never reuse its tuple unchanged.
  Store full shape, spacing in mm, origin, direction, index extent and source transform.
  XCAT C-order z/y/x reshape is an evaluation assumption until landmark validation;
  `.par`/log dimensions prove size, not patient orientation or handedness.
- Convert `.par` pixel/slice widths from cm to mm. Keep preview subsampling transforms;
  never turn a display stride or a VTI maximum dimension into training resolution.
- Per-frame color-coded `act` values are signed integer organ IDs encoded as float32.
  Preserve negative IDs in int32. Zero is background. Averaged `act_av` contains
  fractional mixtures and is not valid categorical ground truth.
- `atn` output logs specify linear attenuation in 1/pixel. Normalize using the case's
  pixel width in cm for comparable 1/cm values. Do not call these HU; HU conversion
  needs an explicit, validated calibration at the selected energy. Preserve original
  values/units too. XCAT phantoms are not automatically reconstructed/noisy CT scans.
- Prefer organ-ID→tissue-category mapping, while preserving both source IDs and category
  labels. The supplied dictionary is incomplete (41 unknown sampled IDs): discovery
  may display unknown IDs verbatim, but production edits must require a reconciled
  map or an explicit preserve-as-unknown policy. Never silently merge them. Names are display strings, not unique keys. For scalar-only sources, define
  finite, explicit half-open bins with units and a final inclusive upper bound, report
  ambiguity, and retain the mapping. Intensities alone do not identify every organ.
- Match label and scalar cases, frames, extent, spacing and transform exactly. Inputs
  can include STL/OBJ for mesh workflows, VTI/tiled VTI, or paired XCAT bins; adapters
  return the same volume record. Native-grid editing is preferred; if resampling is
  required it is explicit (nearest labels; defined scalar interpolation).

## Edit and reassignment semantics

The sample edits a single source organ ID. Donor/recipient/protected IDs always refer
to source organ IDs, even when a coarser category map is present. Multi-target/category
edits need an explicit ownership rule (e.g. nearest original organ with stable ID
tie-breaking) before assigning newly claimed voxels. `direction=from_scale` means
values below one erode, one is identity, and values above one dilate; a fixed direction
may not be swept across one. Component/side selection must be reviewed.

A geometry engine proposes a target mask inside an explicitly bounded domain. Use
physical distances for anisotropic data and define the radius/diameter/severity
convention. The legacy GUI's `scale_factor` is an ROI-radius/morphology control;
it is **not yet a calibrated vessel diameter ratio or clinical percent stenosis**.

Let O be the original target membership and N the accepted target membership:
borrowed = N AND NOT O; released = O AND NOT N. A single integer label field is the
source of truth; per-label masks are derived views, so no voxel can belong to two
labels. Dilation can replace only explicit allowed donors, never protected labels.
Erosion must validate that requested releases belong to the target, then find an
allowed local recipient under bounded, deterministic rules. An empty allowlist means
allow none; priority breaks ties within eligible recipients, not eligibility itself.
No eligible recipient means preserve-and-report or reject, chosen explicitly; no
silent fill with air or an arbitrary distant tissue. Protected barriers and whether
search can cross them must have a defined rule.

Record the original→new label transition table, operation order, changed indices,
protected/blocked/unresolved counts and per-label volume deltas. Outside-domain labels
and scalars stay identical. A bridge/smoothing step obeys the same boundary and label
rules as the edit. Composition of multiple ROIs needs a stable order and conflict rule.

Maintain scalar data separately. First implement deterministic material/recipient
assignment at changed voxels. TensorFlow recovery is an optional later scalar stage,
conditioned on retained multi-label anatomy and local context; it must not overwrite
categorical labels or untouched anatomy. Scalar writes are restricted to the accepted
change mask; any blend halo must be separately configured and audited. Existing binary mask-to-gray training is a
starting point, not a validated multi-label hole-filling implementation.

## Execution and provenance

Each planned job fixes case/frame, anatomy family, source file sizes/metadata hashes,
optional content checksums, config hash, label table hash, code commit, seed,
parameters, environment/container version and expected outputs. Cache keys include all
geometry/label dependencies. Write atomically and refuse overwrite by default; resume
only a matching manifest entry. Never infer a missing `.par` dimension or reuse a
previous case's resolution. Memory limits govern chunk/crop sizes and loaded arrays.
SLURM array index selects a manifest row; local execution uses that same row.

Export labels, scalar volume, transform metadata, change masks, QA and lineage.
Split by underlying anatomy family **before** frame selection, ROI variants or slice
extraction. Every descendant of one base anatomy stays in one train/validation/test
partition. Keep the old dataset as an immutable baseline and real-data validation
source while evaluating synthetic and mixed training cohorts.

## Implemented atlas grouping extension

The draft example now references the source atlas, hierarchy, explicit original-ID
dictionaries and tissue policy. `scripts/prepare_tissues.py` implements catalog building,
attenuation profiling from the audit and reversible crop export independently of the
planned full cohort CLI. Details are in [the tissue workbench](../../TISSUE_MAPPING.md).
Original per-voxel signed IDs must survive grouping; a lookup table alone cannot invert
a many-to-one volume. Material evidence is separate from anatomical hierarchy.
