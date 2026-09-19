# Multi-phantom CoA review and reusable dataset milestone

The generation-only inputs, dataset registry and model experiment are separated.
The [workflow guide](../../MULTIPHANTOM_COHORTS.md) contains the commands and review
sequence; the [dashboard](../../../outputs/cohorts/coa/index.html) links each
editable case input and interactive report.

## What was evaluated

- Twelve frame-1 phantoms have artery-category inputs with optional source IDs
  left blank. The original five study/ROI INIs were checked unchanged.
- Every case's full 16-combination metadata grid passed: 192 planned combinations
  in total. The same validator runs for `--validate-only` and preview. The
  [bounds audit](coa-draft-bounds.json) records standalone versus preview-embedded
  validation and exact input/report hashes.
- One native base recipe was executed per case and its full report generated.
  This is twelve native base recipes, not native execution of all 192 variants.
  All new cases still require ROI/range review and complete native preflight.
  The final audit verified 552 artifact hashes, matching INI snapshots and
  unchanged source-file statistics.
- Existing `cohort-pairs-v2` from 260602 was independently verified and registered
  as `coa-260602-r1`: 16 samples, manifest SHA256
  `5c3940f12a100d026d679ee1e1e3a6eab96351bf0fcec7e8a2efe74716c74c52`.
  Historical scenario splits remain in its native manifest; a later model
  experiment will assign its whole anatomy family together.
- The full suite passed 427 tests. After extending the actual native-generation
  fixture through registry publication and reload, its five focused tests also
  passed. These checks cover freeze invalidation, immutable publication,
  payload integrity, family isolation, resampling and withheld test data.
- The [model readiness check](model-readiness.json) exited 2 as expected: eleven
  planned dataset revisions are missing and source-family provenance is not
  verified. No model lock or fitted model was written.

## Native base-recipe observations

Counts below compare the final recipe with the original source, in native
voxels. Voxel sizes differ between cases; raw counts are not a cross-case
severity measure. A nonzero edit does not establish anatomical accuracy.

| Case report | XCAT template | Spacing i/j/k (mm) | Original selected voxels | Added | Removed | Active edits with zero changes |
| --- | --- | --- | ---: | ---: | ---: | --- |
| [260602](../../../outputs/cohorts/coa/260602/review-r1/report.html) | vmale50.nrb | 1/1/1 | 52,576 | 635 | 904 | none |
| [260611](../../../outputs/cohorts/coa/260611/review-r1/report.html) | vfemale50.nrb | 1/1/1 | 38,367 | 582 | 729 | arch_inner_narrow |
| [260612](../../../outputs/cohorts/coa/260612/review-r1/report.html) | male_infant_ref.nrb | 0.4/0.4/0.4 | 31,868 | 759 | 1,073 | none |
| [260613](../../../outputs/cohorts/coa/260613/review-r1/report.html) | male_1yr_ref.nrb | 0.5/0.5/0.5 | 45,610 | 2,873 | 1,373 | none |
| [260614](../../../outputs/cohorts/coa/260614/review-r1/report.html) | male_5yr_ref.nrb | 0.6/0.6/0.6 | 55,228 | 235 | 955 | none |
| [260615](../../../outputs/cohorts/coa/260615/review-r1/report.html) | male_10yr_ref.nrb | 0.8/0.8/0.8 | 54,829 | 1,657 | 814 | arch_inner_narrow |
| [260616](../../../outputs/cohorts/coa/260616/review-r1/report.html) | male_15yr_ref.nrb | 0.9/0.9/0.9 | 44,421 | 2,573 | 1,036 | none |
| [260617](../../../outputs/cohorts/coa/260617/review-r1/report.html) | female_infant_ref.nrb | 0.4/0.4/0.4 | 31,485 | 692 | 1,021 | none |
| [260618](../../../outputs/cohorts/coa/260618/review-r1/report.html) | female_1yr_ref.nrb | 0.5/0.5/0.5 | 45,610 | 2,873 | 1,373 | none |
| [260619](../../../outputs/cohorts/coa/260619/review-r1/report.html) | female_5yr_ref.nrb | 0.6/0.6/0.6 | 55,228 | 235 | 955 | none |
| [260620](../../../outputs/cohorts/coa/260620/review-r1/report.html) | female_10yr_ref.nrb | 0.8/0.8/0.8 | 54,829 | 1,657 | 814 | arch_inner_narrow |
| [260621](../../../outputs/cohorts/coa/260621/review-r1/report.html) | female_15yr_ref.nrb | 0.9/0.9/0.9 | 35,596 | 3,069 | 803 | none |

All eleven non-reference ROIs remain unreviewed; all twelve new generation
inputs keep `parameters_reviewed=false`. The reviewed 260602 source ROI is
preserved, and its new INI is a future draft distinct from the published r1
snapshot. No additional production dataset was frozen or prepared.

The 260614 and 260619 native source label and attenuation crops have identical
byte hashes, shape and spacing. The audit records this crop-level duplicate;
it does not establish equality of their whole-body phantoms. Keep these cases
in one source family and review duplicate crop weighting when selecting model
datasets. Distinct case IDs or sex parameters do not establish independence.

The initial non-reference distances use body-height scaling only. Some were
reduced further to fit the current native crop/halo budget, with coordinates,
radii, tissue stiffness and manual safeguards retained. In particular, the
260614/260619 provisional distance budgets are about 0.444 times their initial
body-scaled seeds. These adjustments are computational starting points, not
calibrated disease severity; inspect and revise them before preflight.

Case 260611's inner-arch erosion proposes 376 releases but cannot assign their
recipient labels within the configured search. `unresolved=preserve` retains
all 376 voxels, so that step achieves no erosion. This is a recipient-assignment
limitation rather than an enabled erosion safeguard. The same step in 260615
and 260620 retains 513 unresolved releases each. The bounds audit preserves
all per-step proposed, accepted, blocked and unresolved counts.

The initial 260602/260611/260612 reports were reused with identical current INI
bytes and verified artifact checksums. Their recorded segmentation module hash
predates the model-only resampling changes; the audit records this difference.
They remain historical preview snapshots. The dashboard labels matching INI
bytes explicitly and does not claim current generator identity. Freeze and
preparation require a fresh matching native preflight.

## Storage and next review

Commit `2112092` ignores `*.npz` and `/data/cohorts/` on the integration branch.
No NPZ or registry objects remain tracked. The old 61,145-byte `outputs.npz`
was removed from the index and retained on disk; prior Git history was not
rewritten. The 260602 registry snapshot occupies 59,305,240 bytes, of which
13,807,158 bytes are its 17 NPZ files. Keep the registry on backed-up dataset
storage independently of Git.

Begin with the next case's global/local ROI views, then check its base edit
and proposed ranges. Choose fresh output paths for revisions. After ROI review,
run the complete native preflight, review achieved changes, mark the parameters
reviewed, freeze, prepare, and publish a new receipt. Update the project receipt
name and model dataset selection when publishing a newer revision.

Native arrays retain their spacing (0.4–1.0 mm), attenuation units (cm^-1) and
ROI-selected artery-lineage target. The optional model-time spacing conversion
uses linear attenuation and nearest-neighbor masks; its provisional 1 mm grid
requires pediatric detail review. Template ancestry must be verified before
independent-family splits. Automated held-out evaluation and patient inference
remain later milestones.
