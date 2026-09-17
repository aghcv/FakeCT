# Named aorta regions and sequential edits

Date: 2026-09-17. Development branch: `fakect.26.09.16`.

The user's third overall-ROI revision is the basis for a new
[recipe input](../../../configs/studies/thoracic-aorta-recipe.ini). Its 14 native
centers, radii and shared crop margin match the user's study input exactly.
The copied main ROI is recorded as reviewed, following the user's confirmation;
new ascending, descending and arch subdivisions remain unreviewed. The existing
user-edited INIs were preserved.

See the current [stiffness HTML report](../../../outputs/studies/thoracic-aorta/recipe-v2/report.html),
[stiffness results](aorta-recipe-stiffness-results.json), the earlier
[restrictive report](../../../outputs/studies/thoracic-aorta/recipe-v1/report.html),
[restrictive results](aorta-recipe-results.json), and
[editing instructions](../../EDIT_RECIPES.md).

## Implemented behavior

`fakect.recipe/1` introduces `[roi.NAME]`, `[edit.NAME]`, and `[recipe] steps`.
Every edit references a named sphere/tube ROI. The main `[roi]` fixes the common
crop and outer edit boundary; effective named regions are intersections with it.
Shared reassignment rules control tissue eligibility and resistance. The initial
allowlist trial protected other artery IDs; the current stiffness trial allows
them at the configured factor. An active region's full envelope plus edit/search
margin must fit the crop.

Execution uses finite nested `for` loops: ordered edit names, then each step's
`iterations`. Every pass consumes the current labels and scalar proxy, recomputes
target/tissue masks, and saves its own before/after evidence. The limit is 10
iterations per step and 20 total passes, including inactive `none` passes.
`overlap=sequential` permits order-dependent changes; `overlap=error` rejects
intersections between distinct active named regions before edits. Reusing the
same named region is intentional and permitted.

The report separates per-pass activity, unique voxels ever changed, and net
differences from the original. Scalar changes are tracked independently because
restoring a label need not restore the original copied attenuation. Final
overlays give net changes visual precedence over historical failed proposals.
Connectivity diagnostics explicitly distinguish ROI intersections from the
full-crop target; the engine does not promise topology preservation.

## Restrictive baseline: case 260602, frame 1

The shared crop is `146×104×65` in `k,j,i` order, with native `1×1×1` mm spacing.
Its lower index is `(351,338,1294)` and upper exclusive index is `(416,442,1440)`.
It contains 55,814 source-ID-2922 voxels; 50,485 are inside the main editable ROI.

| Pass | Named ROI | Operation | Requested distance | Applied additions | Applied removals |
| --- | --- | --- | ---: | ---: | ---: |
| 1 | ascending | dilation | 2 mm, Gaussian | 0 | 0 |
| 2 | descending | erosion | 2 mm, Gaussian | 0 | 1,134 |
| 3 | arch | none | inactive | 0 | 0 |

Ascending dilation proposed 768 additions, all blocked by encountered tissue:

| Source ID | Anatomical name | Current group | Blocked voxels |
| ---: | --- | --- | ---: |
| 2897 | dias_pericardium | unknown | 663 |
| 2912 | dias_rca1 | artery, outside target selection | 42 |
| 2914 | dias_lad1 | artery, outside target selection | 63 |

This result identified tissue eligibility as the growth blocker. Changing an
iteration count alone cannot convert excluded labels. The baseline retained
the original allowlist; the subsequent trial below implements the user's
requested relaxation through editable resistance factors.

Descending erosion requested 1,997 releases: 1,134 were reassigned and 863
remained unresolved and preserved. The result is therefore constrained by
recipient availability as well as the spatial profile. Final target counts are
54,680 across the crop and 49,351 inside the main ROI. Full-crop six-neighbor
component counts remained 2 before and after; this trial changed geometry
without changing that component count.

The two active regions do not overlap in this example. The arch region overlaps
neighboring subdivisions but is inactive. No named region is clipped by the
main ROI for these initial controls. All modifications occur within the active
region union. No real training cohort or model was generated.

## Editable stiffness trial

The user requested bone = 1, skin = 0.95, and softer surrounding tissues, with
all factors editable in the INI. The same source, ROI geometry, step order and
requested distances were rerun using `[reassignment] mode = stiffness`.
`[stiffness]` supplies category factors and the fallback; `[stiffness.labels]`
overrides individual original signed IDs. The example includes pericardium
ID 2897 = 0.05. These are dimensionless editing controls, not calibrated moduli.

| Pass | Restrictive baseline | Editable stiffness |
| --- | ---: | ---: |
| Ascending: accepted additions | 0 | 741 |
| Ascending: blocked proposals | 768 | 27 |
| Descending: applied removals | 1,134 | 1,500 |
| Descending: unresolved releases preserved | 863 | 435 |
| Arch: active changes | 0 | 0 |

Ascending growth borrows 636 pericardium voxels, 42 `dias_rca1` voxels and
63 `dias_lad1` voxels. Borrowing these nearby coronary labels is permitted by
the requested relaxed policy; their original IDs and transition counts remain
recorded for anatomical review. The remaining 27 pericardium proposals do not
meet the weighted accepted-path budget.

Descending erosion initially requests 1,997 releases. The artery factor of 0.05
suppresses 62 at the boundary of the spatial budget, leaving 1,935 eligible
release proposals. Recipient assignment resolves 1,500 and preserves 435.
Recipient choice minimizes its stiffness penalty plus physical path length,
rather than physical distance alone.

The final main-ROI target count is **49,726**; the full-crop target count is
**55,055**. There are 2,241 net label changes and 2,136 attenuation-value changes:
some neighboring arterial labels share the same scalar value. Full-crop
six-neighbor component counts remain **2 → 2**.

Protection was independently checked using final atlas hierarchy/structure
fields, without historical matched-rule strings:

| Anatomy | Original catalog IDs | Voxels in crop | Voxels in main ROI | Label/scalar changes |
| --- | ---: | ---: | ---: | ---: |
| Bone, including negative inner skeletal IDs | 616 | 110,136 | 1,519 | 0 / 0 |
| Skin-associated original labels | 32 | 82,734 | 9,442 | 0 / 0 |

Of the bone IDs, 210 negative inner skeletal labels currently belong to the
coarse unknown group; protecting only coarse bone would miss 133 voxels inside
this ROI. Skin IDs belong to coarse soft tissue. Their original hierarchy
therefore determines resistance. Whole skin surface-associated labels are used;
they do not isolate the dermal layer. The pinned DPI atlas files match their
current DPI source files.

Bone at 1 is rigid. Skin at 0.95 gets at most 0.1 mm donor distance and 0.15 mm
recipient reach under the current 2 mm edit and 3 mm search settings, below a
step on this 1 mm grid. Larger distances or smaller voxels can permit changes;
0.95 is not absolute protection. ID overrides can change either factor.

## Verification

After the stiffness implementation, the complete suite passed **182 tests**,
including the existing TensorFlow CPU fixture. Tests additionally cover weight
monotonicity, independent target/recipient resistance, signed-ID overrides,
fine skin/bone classification, unrecognized-ID protection, and shortest-path
recipient ownership when the nearest seed is stiffer than a more distant seed.
An independent review found no blocking defects.

Before stiffness, the full suite passed **165 tests**, including the existing TensorFlow CPU
fixture. After adding blocked-label diagnostics and refining overlay accounting,
**63 focused tests passed** across the recipe engine/configuration/report/CLI,
legacy morphology and HTML report. The added engine test brings the current
pre-stiffness suite to 166 tests; the focused rerun included it.

Tests cover order-dependent outcomes, repeated edits, overlap rejection,
unchanged sources, protected signed labels, common-crop bounds, exact input
capture, chained pass states, interrupted-run markers, and embedded figures.
One synthetic round trip restores all original labels while recording six
unique touched voxels, twelve pass changes and persistent scalar differences.

Both reports' artifact and generating-code hashes were verified when generated. Each
saved pass's input arrays match the previous output, final net masks match
original-versus-final arrays, and source size/mtime remain unchanged. The HTML
embeds all its visual assets. Static named-region 3D views and native edit
comparisons were visually inspected. Browser WebGL execution was not tested.
The original draft remains in `recipe-initial-draft`; `recipe-v1` retains the
allowlist comparison and `recipe-v2` is the current stiffness report. Its 28
artifacts include the saved input, native arrays and per-pass snapshots; the
standalone HTML embeds 15 PNGs and two interactive frames. The original crop
matches the restrictive trial exactly, and all values outside active named
regions remain unchanged. Current generating-code hashes are recorded in the
stiffness results; the older report keeps its original code hashes.

Recipe parameter sweeps are a later extension of training preparation. The
existing single-edit study schema remains unchanged, and recipe files do not
silently run the old sweep. Future cohort generation should restart each member
from the immutable phantom and run its complete recipe before making full-crop
image/mask pairs.
