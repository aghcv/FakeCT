# Shared tube ranges and ordered edits

Use [thoracic-aorta-recipe.ini](../configs/studies/thoracic-aorta-recipe.ini) to
try several regional changes on one anatomy. Define its centerline and radii
once in `[roi]`, then select a physical path percentage in each edit. The original study input remains your
single-edit/cohort input; this additional file uses `fakect.recipe/1` for combined
edit review.

Anatomical IDs are optional. Blank or omit `[selection] source_ids` to select
the named tissue by ROI alone; omit `[stiffness.labels]` to use category factors.
Use the report's Global view to find a neighborhood and Local view to refine
its native masks. See [exploration without IDs](ROI_EXPLORATION.md).

Run from the integrated checkout:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-recipe.ini
```

Open [the recipe report with Global, Local, and before/after overlay views](../outputs/studies/thoracic-aorta/recipe-v6/report.html).
For subsequent runs, set `[output] directory` to a fresh location such as
`outputs/studies/thoracic-aorta/recipe-v7`. Add `--validate-only` to check the
input and crop/search bounds without reading voxel payloads. Native overlap and
selection counts require the actual preview run.

## When stronger edits need more crop context

The halo check requires `distance_mm + reassignment.max_distance_mm + largest
voxel spacing` beyond each active named ROI envelope. With a 10 mm edit,
3 mm reassignment search and 1 mm voxels, that is 14 mm. `crop_half_width_mm`
extends from the **main centerline bounds**, so it must also cover the named
ROI radius and placement. It is not the amount of empty space beyond the ROI.

For the current aorta paths, use these settings together:

```ini
[roi]
crop_half_width_mm = 28

[preview]
volume_stride = 3
```

These are entries to update in the existing sections, not duplicate sections to
append. A larger crop supplies editing context but also increases the size of
the 3D context display. Raising the stride reduces only that pooled display;
editing, native slices, and the before/after surface overlay retain source
resolution. Crop width does not enlarge the ROI or increase editing strength.

The errors now identify the failing edit/ROI and available halo, or recommend
the smallest display stride that fits the current crop. Context beyond the
actual source boundary requires moving/shortening the ROI or reducing the
edit/search distance; a larger crop cannot create missing source data.

After changing the paths, radii, or edit distances, choose a fresh output
directory and check settings before generating the full report:

```bash
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-recipe.ini --validate-only
python3 scripts/preview_roi.py --config configs/studies/thoracic-aorta-recipe.ini
```

## The INI structure

The main `[roi]` defines the shared centerline, radii, crop, and hard outer
editing boundary. Use `roi = main` in each edit to reuse it. A tube's first
listed point is 0% and its last is 100%; put the proximal end first if that is
the direction you want. Coordinates remain in supplied order and are never
automatically sorted or assigned an anatomical direction.

Each `[edit.NAME]` can select its own interval. This example edits only 30–75%
of the main tube's physical length:

```ini
[recipe]
steps = ascending_expand, descending_narrow, arch_refine
overlap = sequential

[edit.ascending_expand]
roi = main
path_percent = 30,75
operation = dilation
iterations = 1
distance_mm = 2
profile = gaussian
profile_axis = tube
shape_k = 6
shape_window = 0,1
```

`path_percent` measures cumulative distance along the tube in millimeters,
including anisotropic voxel spacing. Uneven point spacing does not change the
meaning of a percentage. Endpoint coordinates and radii interpolate on the
parent segments, so you maintain only one coordinate/radius list.

Alternatively replace the percentage line with `point_range = 2,8` to use
original points 2 through 8 exactly. Point numbers are **1-based and inclusive**;
the i,j,k coordinates themselves remain zero-based. The selectors are mutually
exclusive. Omit both to edit the entire referenced ROI. Selectors require tubes;
whole named spheres remain supported. The report's **Main tube point reference**
table lists each node's coordinates, radius, cumulative distance, and percentage.

The current aorta input uses percentages approximately matching points 2–8,
10–13, and 4–9 for its three edits. It retains the user's 10 mm ascending dilation
and descending erosion, each for one pass. `arch_refine` remains inactive. These
ranges are starting points to inspect and adjust, including the ascending
extension in the user's latest example.

With `profile_axis = tube`, the selected interval becomes a local coordinate:
`u=0` at the range start and `u=1` at its end. `shape_window = 0,1` tapers a
Gaussian across that entire range; `shape_window = 0.2,0.8` narrows it to the
middle 60% of the selected range. For a parent range of 30–75%, local u=0.5
corresponds to parent 52.5%. Uniform edits still obey the selected range, even
though they ignore Gaussian shape settings. Circumferential/eccentric asymmetry
controls remain future work; this coordinate currently controls longitudinal
localization and taper.

A selected tube interval is bounded by:

```text
subpath tube ∩ parent tube ∩ selected parent arc interval ∩ overall [roi]
```

The parent arc coordinate is the closest centerline position at each voxel;
equal-distance segments use the first in the supplied order. This prevents
rounded subpath caps from extending edits beyond an internal range boundary,
including uniform edits. At tight bends or self-approaches, nearest-point
partitions can be nonplanar; inspect the native range overlays.

Explicit `[roi.NAME]` sections are still available for independent paths, radii,
or spheres. Use `roi = NAME`, optionally with a range on that named tube. These
legacy whole named regions keep their previous rounded-tube mask semantics.
`main` is reserved for the top-level ROI, so `[roi.main]` is rejected.

The global `[selection]` then limits the anatomy. Regions reaching outside the
overall ROI are clipped, and the report records that coverage. Increase or reposition
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
        state = apply_edit(state, resolved_region_for(step), step)
```

You do not need to write this loop. It is a bounded `for` loop, with 1–10
iterations per step and at most 20 total passes. A `while` loop until convergence
would need a separate stopping criterion and could erase a narrow structure;
the present recipe always executes the explicit finite count.

`distance_mm` is the maximum weighted six-neighbor propagation distance **per
pass**, modulated by the profile. Two passes at 1 mm need not equal one pass at
2 mm: eligibility, available recipients and the intermediate geometry can differ.
Neither setting specifies a diameter change or stenosis percentage. Gaussian
position is normalized within the selected interval, or along the full referenced
tube if no interval was supplied.

For several operations on the same region, define separate edit names with the
same `roi` and range values and put them in the desired sequence. To inspect individual
effects, temporarily set the other operations to `none` while keeping their
definitions and ordering.

## Overlap and geometry changes

`overlap = sequential` allows overlapping regions. Later operations see the
earlier labels and copied attenuation values; an erosion may partly reverse a
prior dilation. An operation has no implicit priority based on its name.

`overlap = error` rejects overlap between distinct effective edit regions before
applying any edits. Two intervals on `main` are checked separately, even when
their input ROI names match. Legacy repeated use of the same whole named ROI
retains its previous behavior and is allowed.
Checks use effective native voxel masks. Regions used only by `operation = none`
do not create active conflicts. Even disjoint regions can interact through
nearby recipient context, so the runner always honors the declared order.

Erosion/dilation changes geometry and can also change connectivity, for example
by closing a gap or eliminating a thin connection. Topology changes are possible,
not guaranteed or prohibited. Per-pass and final component counts are diagnostics;
they do not establish anatomical validity. Inspect the native before/after
sections as well as the transparent 3D views.

## What to inspect in the report

The **Edits / recipe** tab starts with a **Before / after surface overlay**:
blue is the original anatomy and orange is the final result of all recipe
passes. Both surfaces occupy the same 3D coordinates. Each has its own Show
checkbox and **15%, 45%, 80%** opacity presets; changing one leaves the other
surface and camera alone. Drag to rotate, scroll to zoom, or use the legend to
toggle a surface. The static overlay is available in the expandable fallback.

This comparison uses native voxel surfaces, independently of the coarser
`preview.volume_stride` used in other 3D context views. It includes the full
selected anatomy inside the crop, with no additional clipping to the main or
named ROIs. Changes therefore appear against the unchanged surrounding target.
Counts describe native additions/removals; overlap colors are not a quantitative
change map. Browser controls do not change masks or the saved input.

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
