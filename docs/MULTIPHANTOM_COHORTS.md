# Preparing and reusing a multi-phantom CoA cohort

The workflow has three independently versioned parts:

| Part | Editable input | Frozen result |
| --- | --- | --- |
| Phantom preparation | `configs/cohorts/coa/<case>.ini` | Reviewed input freeze and native paired crops |
| Dataset publication | Registry revision name and source-family provenance | Copied dataset object plus a checksum-pinned receipt |
| Model experiment | `configs/models/coa.ini` | Dataset selection, family splits, preprocessing and model lock |

Each case can be reviewed and regenerated without rerunning a model. A published
dataset can be reused across model experiments without opening raw XCAT files
or matching the current generator's code. Editing a draft does not alter the
published revision.

## Current cases and review entry point

The [project definition](../configs/cohorts/coa/project.json) lists 260602 and
260611–260621. This generated index points to the human-editable case INIs; the
JSON project index is not the parameter editor. Each case uses frame 1 initially.
The inventory found paired ACT/ATN channels for 51 motion frames per case plus
averages. Motion frames of a source anatomy are not independent subjects.

Refresh and open the [review dashboard](../outputs/cohorts/coa/index.html):

```bash
cd /home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16
python3 scripts/cohort_status.py --project configs/cohorts/coa/project.json --output outputs/cohorts/coa
```

The dashboard distinguishes a current preview from a stale one and a published
dataset from its editable draft. It links the global, local, slice and surface
views in each case's report. Unlike frozen data, the dashboard is a mutable
index and can be refreshed repeatedly.

The reviewed 260602 dataset from `cohort-pairs-v2` is registered as
`coa-260602-r1`. Its original files, 16 samples and historical scenario splits
are preserved. The new clean 260602 INI describes a future preparation revision;
it does not replace the already published data. A later model experiment
assigns the whole registered anatomy to one split.

The other eleven cases start with **unreviewed** aorta-localization candidates.
Localization uses the atlas/dictionary internally to seed the path; the editable
selection remains `tissue = artery` with blank `source_ids`. Surface ordinals
are not assumed to be voxel IDs, and unregistered raw-surface coordinates are
not used as a volume affine.

The seeds run from the low-j limb through the superior arch to the high-j limb.
That anatomical interpretation and all captured branches need visual review.
Initial distances were scaled from the adult example by body height, then
bounded where needed for the current native crop/halo limits. These are starting
values, not calibrated disease severity or equivalent vessel-size changes.
The [inventory](integration/2026-09-18/multiphantom-inventory.json) records source
geometry and provenance. The [bounds and preview audit](integration/2026-09-18/coa-draft-bounds.json)
records adjustments and per-step outcomes; the
[evaluation](integration/2026-09-18/MULTIPHANTOM_EVALUATION.md) summarizes the
twelve reports, zero-change steps, source-crop duplicates and verification.

## 1. Review, freeze and prepare each phantom

Open a case INI, for example [260611.ini](../configs/cohorts/coa/260611.ini).
Each parameter refers to a numbered note at the end of the file. The ordered
ROI points and radii select original artery voxels; named edit intervals reuse
that path. Growth may extend outside the original selector.

```bash
python3 scripts/prepare_cohort.py --config configs/cohorts/coa/260611.ini --stage preview
```

This generates the full interactive report for the base recipe and shows the
planned grid. It does not execute all grid combinations. For another preview,
edit the INI and choose a fresh `[output] directory`, such as `review-r2`.
Use the global view for initial placement and the local/surface views for detail.

After the ROI is acceptable, set `[roi] coordinate_reviewed = true`. Then review
the parameter ranges, for example:

```ini
[sweep.arch_outer_expand]
distance_mm = 3,6,9

[sweep.coa_narrow]
distance_mm = 2:10:2
```

These values illustrate the adult reference; use the actual per-case ranges
instead of copying them to pediatric cases. Sweeps form a Cartesian product;
every variant starts from the original phantom, and its named edits run in
recipe order. Unspecified values, tissue resistance and per-edit safeguards
retain the settings in that case INI.

```bash
# Metadata and bounds for every discrete combination; no voxel simulation
python3 scripts/prepare_cohort.py --config configs/cohorts/coa/260611.ini --validate-only

# Native execution of all combinations; inspect preflight.html and CSV
python3 scripts/prepare_cohort.py --config configs/cohorts/coa/260611.ini --stage preflight
```

Native preflight is necessary before freezing: metadata validation cannot prove
target presence, achieved severity, connectivity, recipient availability or
distinct geometries. Requested distances remain budgets; stiffness, unresolved
reassignment and manually enabled safeguards can reduce achieved changes.

After reviewing native outcomes, set `[train] parameters_reviewed = true`:

```bash
python3 scripts/prepare_cohort.py --config configs/cohorts/coa/260611.ini --stage freeze
python3 scripts/prepare_cohort.py --config configs/cohorts/coa/260611.ini --stage prepare
```

The freeze requires the reviewed ROI, reviewed parameters and a successful
matching preflight. It saves the exact INI, preflight and checksums. Preparation
requires that freeze, verifies the generation dependencies again, and publishes
a complete dataset manifest only after all native cases succeed. Failed work
keeps an `INCOMPLETE` marker and cannot be registered.

**Do not edit the INI after freezing**, including comments, paths or its `stage`
line. Use `--stage` overrides for freeze and prepare. For changed geometry or
ranges, create a new revision and fresh preflight/freeze/pair output directories.
Setting only `parameters_reviewed=true` after preflight records the review
decision; changing data-generating settings requires another preflight. Set
`coordinate_reviewed=true` before the final preflight because it is part of the
resolved ROI configuration.

New `fakect.recipe-cohort/1` inputs have no `[model]` or model split fractions.
Their prepared samples have `split = unassigned`. Successful preparation is
cohort readiness, not a claim that a model is ready to fit.

## 2. Publish reusable dataset revisions

After preparation, register the complete manifest with a new revision name:

```bash
python3 scripts/cohort_registry.py register \
  --registry data/cohorts/coa/registry \
  --dataset outputs/cohorts/coa/260611/pairs-r1/dataset-manifest.json \
  --name coa-260611-r1 --anatomy-family xcat_reference_adult

python3 scripts/cohort_registry.py verify \
  --registry data/cohorts/coa/registry --name coa-260611-r1

python3 scripts/cohort_registry.py list --registry data/cohorts/coa/registry
```

Registration validates all recorded artifact checksums and native array
contracts, then copies the complete dataset under `objects/<manifest-sha256>/`.
An immutable named receipt lives under `entries/`. Repeating the same name and
contents is idempotent; changed bytes or family policy require a new name.
Metadata aliases to the same object do not duplicate its arrays. The workflow
does not overwrite published files, and reuse detects changed bytes.

After publishing a newer revision, update that case's `registry_name` in
`configs/cohorts/coa/project.json` and refresh the dashboard. This index pins a
specific receipt; it does not automatically select the latest one. Update the
model INI's `datasets` list when that experiment should use the new revision.
For example, publishing the next 260602 draft as `coa-260602-r2` leaves `r1`
available to existing experiments.

Keep the complete registry on durable dataset storage and back it up independently
of Git. The repository ignores `*.npz` and `/data/cohorts/`, including registry
objects and receipts. Configs, code and documentation stay in Git. Ignore rules
prevent accidental addition of new files; they do not remove older committed
blobs or act as a data backup. Moving a registry requires preserving its internal
layout and planning a new model experiment against its new location.

The reference-family grouping is provisional. It groups male/female templates
of each age conservatively for review, but neither that grouping nor twelve
case IDs proves independent anatomy. Confirm template ancestry before using
`--family-verified`. To record revised/verified provenance, register a new receipt
name for the same object and use that name in a new model INI. Generation and
publication can proceed while this provenance question remains open.

## 3. Select datasets in a separate model INI

[configs/models/coa.ini](../configs/models/coa.ini) points only to registry names
and model settings. It does not reference ROI parameters, raw phantom paths or
the generation-stage INIs. Its current list reserves all twelve case revisions;
missing datasets and unverified families are reported together:

```bash
python3 scripts/model_experiment.py --config configs/models/coa.ini --validate-only
```

This model protocol requires at least three verified independent anatomy
families. A deterministic split assigns complete families to train, validation
or test. Variants, slices, motion frames and recipe revisions from a known shared
source stay together. The old 260602 scenario assignments are not reused for
this combined experiment.

Native voxel spacing ranges from 0.4 to 1.0 mm. The optional `[preprocess]`
`target_spacing_mm` makes model-time sampling explicit: linear interpolation
for attenuation and nearest-neighbor sampling for binary masks, without editing
stored native pairs. The starter's 1 mm setting is provisional and needs review
for small pediatric vessels. Omitting the section requires compatible native
spacing. Differing coordinate conventions, image units or target semantics are
still rejected instead of silently mixed.

The experiment lock records the resampling implementation and NumPy/SciPy
versions as well as its grid convention. Changing those requires a new model
experiment lock; the stored native datasets can still be reused.

Only after the cohort and provenance are ready:

```bash
# Pin dataset revisions, family assignments, preprocessing and model settings
python3 scripts/model_experiment.py --config configs/models/coa.ini --stage plan

# Separate explicit action; not run during cohort preparation
python3 scripts/model_experiment.py --config configs/models/coa.ini --stage fit
```

The lock verifies dataset receipts and manifests before reuse; training decodes
only training/validation payloads. The test partition remains reserved for a
later explicit evaluation workflow. Automated test evaluation and patient
inference are not part of this cohort-preparation milestone.

## Scope of the resulting data

These pairs are native cropped attenuation-copy proxies in cm^-1, not HU,
learned background recovery or physical CT reconstructions. Their binary target
tracks the reviewed ROI-selected artery ancestors and descendants, including
growth outside the selector. It may include nearby branches captured by the
ROI; it is not automatically an anatomically exclusive whole-aorta annotation.
Normal tissue reassignment can suppress erosion when a recipient cannot be
assigned. Keep reviewing that behavior and the chest-surface tissue mapping
before using larger cohorts to draw anatomy or patient-generalization claims.
