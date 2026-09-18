# Aorta erosion safeguards — 2026-09-18

The current user-edited aorta recipe now supports per-edit retained-volume and
connectivity checks with bounded distance retries. User ROI coordinates, artery
category selection, edit ranges, requested distances, Gaussian settings,
directions and stiffness values were preserved. The new controls were enabled
on both erosions with a provisional 50% retained-volume floor and six-face
connectivity preservation. The dilation has no erosion guard.

See [configuration and semantics](../../EROSION_SAFEGUARDS.md) and the full
[commented input](../../../configs/studies/thoracic-aorta-curvature.ini).

## Native phantom evaluation

Case 260602, frame 1; crop shape 158 × 116 × 77 in k,j,i order, 1 mm isotropic
spacing. Both evaluations used the same original source crop. The unrestricted
run removed only the four new guard settings; the guarded run retained them.

| Step / trial | Distance (mm) | Retained active target | Local retained volume | Affected component survivors | Decision |
| --- | ---: | ---: | ---: | ---: | --- |
| `arch_inner_narrow`, requested | 7 | 8,823 / 9,577 | 92.13% | 12 | Reject: split |
| `arch_inner_narrow`, retry | 3.5 | 9,341 / 9,577 | 97.54% | 6 | Reject: split |
| `arch_inner_narrow`, retry | 1.75 | 9,541 / 9,577 | 99.62% | 1 | Accept |
| `coa_narrow`, requested | 10 | 1,246 / 3,636 | 34.27% | 2 | Reject: volume floor and split |
| `coa_narrow`, retry | 5 | 2,218 / 3,636 | 61.00% | 1 | Accept |

Each affected component began as one connected component. The table counts
its surviving pieces, not anatomical vessels or XCAT source IDs. In particular,
the arch erosion demonstrates why high retained volume alone cannot protect
connectivity: its 7 mm trial retained over 92% yet produced 12 pieces.

For `coa_narrow`, `path_percent=65,75` and `shape_window=0.05,0.95` define active
longitudinal support at parent 65.5–74.5%. Its volume baseline contains 3,636
selected artery voxels, compared with 4,042 in the entire selected 65–75%
subpath and 52,576 in the original main target selection. Untouched distant
target cannot dilute this check.

`arch_outer_expand` remains at the requested 9 mm and adds 1,710 voxels. Accepted
erosions release 36 arch voxels and 1,418 coarctation-region voxels. Final net
counts are 1,710 additions, 1,454 removals and 3,164 changed label positions.
The 1,454 released voxels retain diagnostic labels and placeholder attenuation;
the original diagnostic training restriction remains in force.

Full-crop target component counts remain 27 → 27 throughout the guarded recipe.
The unrestricted recipe gives 27 → 38 → 39 through its erosion steps. Existing
separate components were not repaired, merged or silently discarded.

## Review artifacts

The new standalone [recipe-v11 report](../../../outputs/studies/thoracic-aorta/recipe-v11/report.html)
includes the current before/after overlay, tissue toggles, curvilinear profiles,
local slices, and per-edit retry tables. Per-pass profile captions distinguish
the requested INI distance from the accepted trial distance. Existing populated
reports were preserved; the next interactive run should choose a new output
directory such as `recipe-v12`.

- [Native evaluation results](erosion-safeguard-evaluation.json): both runs,
  input hashes, exact pass counts, retries, affected component sizes and source
  crop integrity metadata.
- [Artifact audit](erosion-safeguard-artifact-audit.json): final report/input/code
  hashes, mask/count checks and preserved unrelated user-input checks.
- `steps/.../step.npz`: accepted arrays, original per-pass inputs, the fixed
  `erosion_guard_scope_mask`, the retained mask and the complete retry audit.

Rejected trial arrays are never persisted or passed to later edits. Retry
diagnostics record their failures. The step reference is fixed across its
iterations; separate named edits capture separate references after preceding
steps, so overlapping names do not establish a whole-recipe cumulative floor.

Connectivity is a six-neighbor voxel check inside the loaded crop. It can
preserve a one-voxel bridge and does not enforce minimum lumen area or diameter.
The 50% floor is a debugging control, not a calibrated clinical severity.

## Validation

- Full suite: **353 tests passed** (`python3 -m unittest discover -s tests`).
- Guard cases cover localized Gaussian support, angular scope, pre-existing
  components, split/component-loss cancellation, fixed iteration references,
  exhausted and underflowing retries, unchanged disabled behavior, and rollback
  of diagnostic labels, attenuation and ancestry.
- Mixed standalone training sweeps preserve guards on erosion variants and
  remove them on generated baseline/dilation variants. Enabled guard settings
  participate in erosion scenario identities.
- All **49 artifact hashes** and **22 generating-code hashes** match the final
  report. Saved scope/retained-mask counts and accepted labels agree with the
  retry audit. The INI snapshot matches the edited working input.
- Source sizes/mtimes and crop hashes were recorded; full source-volume hashes
  were not computed. Unrelated user-edited INIs remain byte-for-byte unchanged.
- The accepted coarctation profile and static before/after overlay were visually
  inspected. Browser WebGL interaction was not exercised in this evaluation.
