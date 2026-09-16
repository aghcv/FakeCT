# FakeCT integration evaluation and milestone plan — 2026-09-16

Branch: **`fakect.26.09.16`**. This is an evaluated integration baseline, not a
finished cohort generator. It assembles the phantom GUI/VTI implementation,
newer TensorFlow trainer and existing XCAT utilities; the unified configuration,
headless edits and population pipeline below remain development milestones.
The original checkout, user environment edits and source datasets are preserved.
Worktree: `/home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16`.
The local branch has no remote upstream yet.
No training job, XCAT generation job or SLURM submission was started.

## Baseline and reproducibility

- Base: `origin/phantom` at `c6a5c2c9c47ebcbee116c8ade18269eef4ac09ff`.
- Trainer/environment: local `stenosis` at `d3b7a62923959abd7f6f983526367829ffcf0c65`.
- XCAT job/pool/converter and requirements: local `main` at
  `183e0035b3e01d877b532aa4718db90d0802d0a4`.
- Original training launchers are captured under `scripts/slurm/reference/`.
  The parameterized GPU wrapper preserves the supplied TensorFlow 2.17 setup.
- Original notebook exports are retained as reference text, with hashes in
  [SOURCES.json](SOURCES.json); they are not imported as production Python.
- Dependencies are inherited and not yet locked. Cluster TensorFlow must come from
  its verified container; do not upgrade it via the generic requirements file.

Evidence: [XCAT audit](XCAT_AUDIT.md) and [indexed histogram/slice preview](xcat-preview.png), [training/branch audit](TRAINING_AUDIT.md),
[notebook audit](NOTEBOOK_AUDIT.md), [verification](VERIFICATION.md), and
[proposed input contract](INPUT_CONTRACT.md). See the adjacent JSON reports for
measurements and sampling scope. Reproduction commands are in `VERIFICATION.md`.

## Findings that determine the design

**Prefer paired organ labels and scalar volumes.** All 12 supplied XCAT cases are
present: 612 paired non-averaged motion frames, 1,248 binary volumes including
averages, and 3,672 mesh files. The `.raw` files are ASCII surface meshes; the `.bin` files are volumetric.
Per-frame `act` files with `color_code=1` contain signed organ IDs, including negative
IDs, and the organ-name table has repeated names. Sampling found **41 IDs absent from
the supplied organ table**, so explicit dictionary reconciliation is an M1 gate.
Preserve integer IDs and provide a
reviewable ID→tissue-category map. `act_av` is a motion average with fractional IDs
and must not be used as a categorical mask. Scalar-only intensity grouping remains a
fallback for other inputs, not the best anatomical selector for these paired cases.

**Keep units and geometry explicit.** .bin files are the verified sources in these case folders; no separate FakeCT
derivatives were identified there. Case resolutions differ; `.par`/output-log
geometry and byte lengths agree. Logs describe `atn` values as attenuation in 1/pixel,
not HU. Normalize by pixel width in cm before comparing attenuation across cases.
C-order `[k,j,i]` and little-endian float32 produce plausible sampled images/labels,
but anatomical orientation and origin are not established by file size. Validate
landmarks and cell/point conventions before saving a reusable ROI. Cases span adult
and pediatric reference anatomies; frames and edits do not add independent subjects.

**The GUI is a useful front end, but its edit controls need calibration.** The phantom
branch has multi-layer VTI import, interactive ROI and batch controls. Its terminal CLI
loads/exports volumes but exposes no equivalent ROI edit configuration. Bounded tests
found remote changes from the default global bridge, no-op scale ranges and
non-monotonic erosion with nominal scale. Define permitted support explicitly and
measure achieved target geometry; do not label the current scale as a measured vessel
stenosis percentage. GUI and terminal mode must call the same validated core.

**Notebook donor policies are useful, but incomplete.** Fourteen selected notebook test
functions and 60 randomized checks passed. Adversarial tests found self-donor counting
errors, malformed erosion masks that can overwrite protected tissue, recipients
selected across protected barriers, and 2D-only erosion. Example physical sphere
requests reached only 51–64% of their requested domain under the iteration heuristic.
The final VTI example inserts an artificial seed; production must modify the selected
existing anatomical label/component. Preserve accounting and report underachievement.

**The current trainer is not yet a multi-label recovery model.** It collapses masks
with `>0`, which would discard negative IDs and merge positive organs. Its NRRD slice
axis is named sagittal without checking the physical directions. Of 22,753 manifest
rows from 56 cases, shape filtering in the reproduced split retains only 2,708 slices
from 17 cases. All 14 retained training case IDs also occur in validation; test case IDs
are separate. This undermines validation and wastes available data. Make shape policy,
orientation, physical context depth across resolutions, eligible counts and anatomy-grouped splits explicit before generating more.

**Scale requires bounded memory and verified GPU use.** At the selected 512×666 shape,
context 15 means 31 mask channels; a full 256-sample expanded shuffle buffer alone is
about 10.4 GiB, exceeding the supplied job's 8 GiB allocation before volume caches. XCAT sources occupy about 4.30 TiB logically but 422 GiB
physically; preserve sparse/storage behavior and avoid duplicating full volumes per
variant. Network-storage orthogonal reads were slow, so previews need local scratch
or cached, chunked/downsampled access.
Existing logs show successful saved models but also CPU fallback messages; they do not
prove that the currently supplied GPU launcher used the GPU. The launcher is accepted
as the user's confirmed recipe; next training validation must record runtime versions,
visible GPU, actual placement and peak memory. Existing normalized-image metrics are
historical baselines, not evidence of accurate multi-label or HU recovery.

## Milestones and exit criteria

| Milestone | Deliverable | Required evidence to finish |
|---|---|---|
| **M0 — Integration baseline (this change)** | Versioned source provenance, existing GUI/trainer/XCAT tools together, launcher references, reproducible bounded audits, draft input contract and preview | CLI/import and VTI smoke checks pass; original data/checkouts preserved; observed defects recorded rather than hidden |
| **M1 — Volume and configuration foundation** | Strict versioned JSON schema and validator; common volume record; XCAT act/atn/.par/log adapter; VTI and mesh adapters; signed labels, units, transforms, category mapping | All 12 case headers and file sizes validate; matched frame geometry; crop read/write roundtrip; negative IDs preserved; nonfinite/unknown labels rejected; orientation landmarks and unknown-ID mapping reviewed; no full-volume allocation for metadata/preview |
| **M2 — Iterative preprocessing and preview** | `validate → preprocess → preview`; cached histograms/category tables; axial/coronal/sagittal slices with native i/j/k and relative mm; editable ROI/config; before/after/difference preview; GUI config import/export | ROI roundtrip between preview, GUI and config selects identical native voxels; decimation never shifts saved coordinates; missing target still permits discovery preview; config changes invalidate review hashes |
| **M3 — Bounded 3D morphology and reassignment** | Shared pure geometry engine; physical ROI/profile semantics; n-D donor rules; 3D recipient policy with explicit allowed/protected IDs, barriers and unresolved handling | Identity exact; no outside-domain/protected changes; single label per voxel; signed IDs safe; borrowed/released and transition counts balance; deterministic output; anisotropic and crop/full equivalence; achieved geometry monotonic or unmet targets rejected/reported |
| **M4 — Scalar reconstruction and one-case acceptance** | Deterministic scalar/material reassignment first; preserve original/scalar/labels/change mask separately; optional AI interface; one reviewed case/frame/ROI pilot | Labels and scalar align; untouched voxels identical; no unresolved reassignment; units/calibration explicit; target geometry and tissue transitions reviewed; output reopens in GUI; measured runtime/RSS fits chosen job allocation |
| **M5 — Virtual-population jobs** | Config sweeps → immutable JSONL task manifest; local executor and SLURM arrays use the same task; seeds, limits, dry-run, resume, atomic outputs, failure/QA ledger | Repeated task reproducible; output collisions impossible; failed tasks resume independently; every variant starts from original; job count/cost/memory estimate visible; small array pilot completes before 12-case rollout |
| **M6 — Multi-label training and fair evaluation** | Multi-label conditioning, orientation-aware crops/patches, bounded loading, metadata shuffle before expansion, anatomy-family splits; frozen real-data reference; optional AI scalar recovery | No anatomy/frame/variant leakage; all eligibility exclusions explicit; one TF2.17 GPU smoke run proves placement and memory; compare deterministic, real-only, synthetic-only and mixed baselines; report per-case/label/edited-region errors and unchanged-region preservation |
| **M7 — Controlled dataset transition** | QA-gated synthetic release, immutable cohort manifests, versioned training exports and model registry; expand one ROI → several settings → multiple anatomies → full cohort | Pilot acceptance carries across adult/pediatric resolutions; independent real-data evaluation meets predeclared goals; population diversity/counts reported by anatomy, not slices; release is reproducible and the previous dataset remains recoverable |

The critical path is **M1 → M2 → M3 → M4 → M5**. M6's training-data audit/fixes can
proceed alongside M2–M4, but AI recovery should not be required to validate categorical
label reassignment. The first real edit should use one small crop in case `260602`,
frame 1, after target presence/orientation/ROI review. The left carotid ID in the sample
configuration is a proposed selector, not a confirmed ROI.

## Cohort and training strategy

Start with paired label and scalar ground truth from a single non-averaged motion
frame. Preserve all source organ IDs even if training also uses a coarser finite tissue
vocabulary. Establish deterministic, auditable geometry and scalar reassignment before
asking a neural network to fill changed regions. For AI, compare a multi-label/context
model with deterministic material and local-recipient baselines; measure errors inside
the edit, along its boundary and outside it separately. Global MAE alone can hide a
bad small vessel edit in a large unchanged background.

The proposed virtual population includes anatomy/model lineage, age/size category,
motion phase, ROI location, geometry settings and reassignment policy. Treat repeated
frames and synthetic descendants as correlated. Split base anatomy families before
sample generation and keep the current dataset as a frozen reference/holdout wherever
appropriate. Promote a new versioned manifest after evaluation rather than deleting
`data/dataset` or immediately making a synthetic-only replacement.

## First implementation slice

Implement M1 and the read-only part of M2: load one matching act/atn pair via memmap,
validate `.par` against output log and byte lengths, preserve signed IDs, emit a category
table/histogram and indexed cross-sections, then persist a reviewed ROI in the draft
config. This delivers the requested edit–preview loop early and supplies an objective
fixture for M3 without starting a large generation or training run.
