# Named ROIs and ordered edits

Use [thoracic-aorta-recipe.ini](../configs/studies/thoracic-aorta-recipe.ini) to
try several regional changes on one anatomy. It copies the centerline and radii
from your third aorta preview iteration. The original study input remains your
single-edit/cohort input; this additional file uses `fakect.recipe/1` for combined
edit review.

Run from the integrated checkout:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-recipe.ini
```

Open [the stiffness recipe report](../outputs/studies/thoracic-aorta/recipe-v2/report.html).
For subsequent runs, set `[output] directory` to a fresh location such as
`outputs/studies/thoracic-aorta/recipe-v3`. Add `--validate-only` to check the
input and crop/search bounds without reading voxel payloads. Native overlap and
selection counts require the actual preview run.

## The INI structure

The existing `[roi]` describes the overall workbench region: it fixes the shared
crop and acts as an outer boundary for edits. Define local regions in named
sections such as `[roi.ascending]`, `[roi.arch]` and `[roi.descending]`. Each can
be a sphere or an ordered variable-radius tube. Coordinates use native `i,j,k`;
all lengths use millimetres.

Each `[edit.NAME]` references a region by name. The list in `[recipe] steps`
defines execution order, independently of where sections appear in the file:

```ini
[recipe]
steps = ascending_expand, descending_narrow, arch_refine
overlap = sequential

[edit.ascending_expand]
roi = ascending
operation = dilation
iterations = 1
distance_mm = 2
profile = gaussian
profile_axis = tube
shape_k = 6
shape_window = 0,1
```

The complete example includes all three region definitions and edit sections.
It starts with ascending dilation and descending erosion, each at 2 mm for one
pass. `arch_refine` has `operation = none`, so its region is visible and ready
to edit. These local regions are new proposals derived from the reviewed overall
path; inspect their locations before interpreting the anatomical names as final.

An effective edit region is:

```text
named ROI intersected with the overall [roi]
```

The global `[selection]` then limits the anatomy: the example edits source ID
2922 within the artery tissue group. A named ROI reaching outside the overall
ROI is clipped, and the report records that coverage. Increase or reposition
the overall ROI if the intended change needs more space. Each active named ROI's
full geometric envelope plus its edit/recipient search margin must fit the shared
crop, even where the overall ROI clips it. Voxels outside effective edit regions
remain unchanged. Inside them, the selected reassignment policy controls which
surrounding labels may be borrowed or receive released voxels.

## Editable tissue resistance

The starter uses low resistance for surrounding tissues, rigid bone and highly
resistant skin. Change the factors directly in the INI:

```ini
[reassignment]
mode = stiffness
allowed_tissues =
max_distance_mm = 3
unresolved = preserve

[stiffness]
default = 0.05
bone = 1.0
skin = 0.95
artery = 0.05
soft_tissue = 0.05

[stiffness.labels]
2897 = 0.05                 # original anatomical ID: dias_pericardium
```

The full example lists all coarse tissue categories. `default`, `bone`, and
`skin` are required in this mode; other categories use `default` if omitted.
The optional `[stiffness.labels]` section accepts catalogued signed original
IDs, including negative IDs. Precedence is **original-ID override → anatomical
bone/skin identity or coarse tissue factor → default**. Missing dictionary IDs
and background always remain protected. All supplied factors can be edited.

Factors range from **0 (no additional resistance) to 1 (rigid)**. These are
dimensionless editing weights, not measured elastic moduli. The implementation
uses physical six-neighbor path lengths:

- Dilation accepts a donor voxel within a local budget of
  `requested profile distance × (1 − donor factor)` along an accepted path.
- Erosion releases target voxels within
  `requested profile distance × (1 − target factor)`.
- Erosion recipient seeds begin with a cost of
  `max_distance_mm × recipient factor`; physical path length adds to that cost.
  The seed with the smallest total cost wins, up to `max_distance_mm`. Thus a
  softer recipient can win over a nearer stiff recipient. Unresolved releases
  preserve their input-state label and attenuation.

Skin at 0.95 retains only 5% of a distance budget: for a 2 mm request that is
0.1 mm, below a single step on this 1 mm grid. It is highly resistant, not
absolutely immutable; set a factor to 1 for complete rigidity. Softening the
selected artery factor also makes erosion easier. Repeated edits recompute
these rules on the current geometry.

Skin is identified using the applied anatomical hierarchy/structure metadata,
even when its coarse group is `soft_tissue`. Bone identification also includes
inner skeletal labels grouped as `unknown`. Skin surface-associated labels can
cover more anatomy than the dermal layer; the report lists the exact IDs and
effective factors. Original labels and atlas metadata stay preserved.

This mode permits borrowing catalogued unknown tissue and nearby arteries with
different IDs, as requested. Their factors are adjustable in the same file.
The older `allowlist` mode retains strict tissue exclusions and excludes other
arteries when the target is artery. A separate `permissive_except_bone_skin`
mode gives binary bone/skin protection. Those two modes reject unused stiffness
sections; `allowed_tissues` must be blank in the stiffness or permissive modes.

## Execution and repetition

The runner starts with one copy of the original label and attenuation crop.
It applies each listed step to the current result, recomputing tissue and target
masks after every pass:

```python
state = copy_of_original_crop()
for step_name in recipe.steps:
    step = edits[step_name]
    for iteration in range(step.iterations):
        state = apply_edit(state, rois[step.roi], step)
```

You do not need to write this loop. It is a bounded `for` loop, with 1–10
iterations per step and at most 20 total passes. A `while` loop until convergence
would need a separate stopping criterion and could erase a narrow structure;
the present recipe always executes the explicit finite count.

`distance_mm` is the maximum weighted six-neighbor propagation distance **per
pass**, modulated by the profile. Two passes at 1 mm need not equal one pass at
2 mm: eligibility, available recipients and the intermediate geometry can differ.
Neither setting specifies a diameter change or stenosis percentage. Gaussian
position is normalized along each named tube, not the overall aorta path.

For several operations on the same region, define separate edit names with the
same `roi` value and put them in the desired sequence. To inspect individual
effects, temporarily set the other operations to `none` while keeping their
definitions and ordering.

## Overlap and geometry changes

`overlap = sequential` allows overlapping regions. Later operations see the
earlier labels and copied attenuation values; an erosion may partly reverse a
prior dilation. An operation has no implicit priority based on its name.

`overlap = error` rejects overlap between distinct active named ROIs before
applying any edits. Reusing the same ROI name intentionally is still allowed.
Checks use effective native voxel masks. Regions used only by `operation = none`
do not create active conflicts. Even disjoint regions can interact through
nearby recipient context, so the runner always honors the declared order.

Erosion/dilation changes geometry and can also change connectivity, for example
by closing a gap or eliminating a thin connection. Topology changes are possible,
not guaranteed or prohibited. Per-pass and final component counts are diagnostics;
they do not establish anatomical validity. Inspect the native before/after
sections as well as the transparent 3D views.

## What to inspect in the report

The single HTML file includes colored named-region overlays and local views,
an execution table, before/after comparisons for every pass, and the final
combined result with 3D context. It distinguishes:

- **Per-pass activity:** what each operation added, released, blocked or could
  not reassign in its input state.
- **Net result:** how final labels differ from the original, after possible
  reversals by later steps.
- **Any-pass activity:** voxels touched at least once, including changes that
  were later reversed.

Each pass has a `steps/.../step.npz` snapshot, and `edit.npz` preserves the final
result with original data and recipe metadata. Attenuation is a copy proxy:
values follow the winning input-state target/recipient seed. A later pass may
copy a value already moved by an earlier pass. This is not new AI background
recovery or physical CT reconstruction, and reversing labels need not restore
the original scalar values. Separate masks record final label and scalar changes.

In the first, restrictive aorta trial, descending erosion reassigns 1,134 voxels. Ascending
dilation proposes 768 additions but accepts none: 663 candidates carry
`dias_pericardium` (ID 2897), currently in the unknown tissue group, and 105 carry
protected coronary-artery labels (42 ID 2912 and 63 ID 2914). The per-pass report
lists these encountered labels. The current stiffness example permits these
labels at factor 0.05, so they can participate in edits. The original
[restrictive report](../outputs/studies/thoracic-aorta/recipe-v1/report.html)
remains available for comparison; see the
[evaluation](integration/2026-09-17/RECIPE_EVALUATION.md) for measured results.

## Connection to training

This recipe interface currently reviews one combined anatomy. The existing
`fakect.study/1` train/prepare workflow remains a single-edit parameter sweep;
it does not silently interpret a recipe as that sweep. Once region placement,
order and ranges are settled, the cohort builder can use `apply_recipe` inside
its population loop. Each cohort member must restart from the original phantom;
only the operations within that member share evolving state. Full-crop binary
targets and source-label provenance remain the pairing contract.
