# Morphology illustrations

The **neighbor-search version** adds a split original view and two small worked examples below it. Open the [viewing page](../../../outputs/illustrations/bitwise-2d-neighbors/report.html), [PNG](../../../outputs/illustrations/bitwise-2d-neighbors/bitwise-morphology-2d-neighbors.png), [PDF](../../../outputs/illustrations/bitwise-2d-neighbors/bitwise-morphology-2d-neighbors.pdf), or [editable SVG](../../../outputs/illustrations/bitwise-2d-neighbors/bitwise-morphology-2d-neighbors.svg). Solid outlines track a resistant Label 4 pixel that dilation cannot borrow; arrows show permitted growth into other neighbors. Dashed outlines track a released pixel and its replacement by a neighboring Label 2. Cells have no letters or anatomical names.

The two upper results independently edit the full original mask; the vertical divider only associates each half with its example. One four-neighbor dilation proposes eight new pixels, accepts seven, and preserves the resistant pixel. Erosion releases and reassigns four pixels. Its inset isolates one decision using original neighbors: faint arrows indicate other eligible candidates, and the solid arrow indicates the selected seed. The retained target neighbor is excluded. Equal path costs use lower label ID and then original seed position, as in the native ownership primitive; this is not a majority vote. The blank center is an illustrative intermediate state, not an unassigned output label. The figure omits variable stiffness, ROI and safeguards. [Exact arrays and validation](../../../outputs/illustrations/bitwise-2d-neighbors/evidence.json) accompany it.

Regenerate with `python3 scripts/evaluation/illustrate_bitwise_2d_neighbors.py --overwrite`. Previous figures remain available below.

LaTeX algorithm tables are saved in the same folder: [dilation PDF](../../../outputs/illustrations/bitwise-2d-neighbors/dilation-label-reassignment-table.pdf), [erosion PDF](../../../outputs/illustrations/bitwise-2d-neighbors/erosion-label-reassignment-table.pdf), and [combined two-page PDF](../../../outputs/illustrations/bitwise-2d-neighbors/label-reassignment-tables.pdf). Each has a self-contained source: [dilation LaTeX](../../../outputs/illustrations/bitwise-2d-neighbors/dilation-label-reassignment-table.tex), [erosion LaTeX](../../../outputs/illustrations/bitwise-2d-neighbors/erosion-label-reassignment-table.tex), and [combined LaTeX](../../../outputs/illustrations/bitwise-2d-neighbors/label-reassignment-tables.tex). They summarize single-pass neighbor ownership, stiffness, deterministic ties, unresolved releases, and diagnostic release mode. Compile from that folder on Wahab with `module load texlive/2020` followed by `crun pdflatex -interaction=nonstopmode -halt-on-error label-reassignment-tables.tex` (or either individual source).

The **simplified, color-only 2D version** uses six **5×5 grids**: Boolean dilation/erosion above, label borrowing/reassignment below. Both rows branch from the central original. It has three generic labels and no cell letters, coordinates, anatomical names, or explanatory paragraphs in the figure. Open the [viewing page](../../../outputs/illustrations/bitwise-2d-simple/report.html), [PNG](../../../outputs/illustrations/bitwise-2d-simple/bitwise-morphology-2d-simple.png), [PDF](../../../outputs/illustrations/bitwise-2d-simple/bitwise-morphology-2d-simple.pdf), or [editable SVG](../../../outputs/illustrations/bitwise-2d-simple/bitwise-morphology-2d-simple.svg).

The original five-pixel cross dilates to 13 pixels or erodes to one. Borrowing changes eight surrounding pixels to Label 1. Erosion releases four pixels, which copy eligible face-neighbor labels using the native ownership primitive. All neighbors are eligible and there is no extra resistance in this deliberately minimal example; deterministic cost ties use the original label ID and seed coordinate. [Exact grids and checks](../../../outputs/illustrations/bitwise-2d-simple/evidence.json) are saved alongside it. Regenerate with `python3 scripts/evaluation/illustrate_bitwise_2d_simple.py --overwrite`.

The original generic 3D figure is [H-bitwise-neighborhood.png](../../../outputs/cohorts/coarctation-workflow-v2/panels/H-bitwise-neighborhood.png). Its [PDF](../../../outputs/cohorts/coarctation-workflow-v2/panels/H-bitwise-neighborhood.pdf), [editable SVG](../../../outputs/cohorts/coarctation-workflow-v2/panels/H-bitwise-neighborhood.svg), and [original report](../../../outputs/cohorts/coarctation-workflow-v2/report.html) are alongside it. The generator is `outputs/cohorts/coarctation-workflow-v2/render_bitwise.py`.

The earlier, detailed 2D example is available as a [viewing page](../../../outputs/illustrations/bitwise-2d/report.html), [PNG](../../../outputs/illustrations/bitwise-2d/bitwise-morphology-2d.png), [PDF](../../../outputs/illustrations/bitwise-2d/bitwise-morphology-2d.pdf), or [editable SVG](../../../outputs/illustrations/bitwise-2d/bitwise-morphology-2d.svg). [Evidence and exact displayed labels](../../../outputs/illustrations/bitwise-2d/evidence.json) accompany it.

The eight panels explain original labels/ROI, Boolean dilation proposals, accepted borrowing and blocked pixels, the resulting dilation label map, erosion release proposals, surrounding label seeds, weighted seed choice, and the final erosion labels. The two operations start independently from the same original image.

The toy grid uses the current `apply_morphology` kernel on an extruded 3D object, displaying its central plane. At unit spacing, the displayed native proposals are checked against two rounds of four-neighbor Boolean dilation/erosion. The full-volume implementation uses six face neighbors, physical path distances, and policy constraints; it is not simply OR/AND on integer tissue IDs.

| Displayed-plane outcome | Pixels |
| --- | ---: |
| Original artery | 37 |
| Dilation candidates | 44 |
| Successfully borrowed | 27 |
| Blocked by policy or reachable-path limits | 17 |
| Proposed erosion releases | 28 |
| Reassigned to surrounding tissue | 24 |
| Unresolved, preserved as artery | 4 |

The highlighted released pixel `P = (i=12,j=14,k=15)` chooses lung seed `Q = (11,15,15)`. Its cost is two physical pixel edges plus zero stiffness penalty: **2.0 mm**. The nearest soft-tissue seed costs one edge plus `3 mm × 0.4`: **2.2 mm**. Thus the current policy chooses the lowest admissible weighted path, not a majority vote or an unconstrained nearest pixel. Ties use the signed original label ID and then the original seed coordinate. Labels and attenuation-copy proxies come from the immutable input state; this is not physical CT reconstruction.

The stiffness values are intentionally illustrative: artery 0.05, soft tissue 0.4, lung 0, bone 1. The edit budget is 2.2 mm and the recipient search cap is 3 mm. Pixels without a reachable recipient retain their original artery label. With `assign_surrounding_tissue=false`, erosion instead writes a diagnostic released label with unassigned attenuation; this is distinct from paired-training preparation.

Reproduce from the integration checkout:

```bash
python3 scripts/evaluation/illustrate_bitwise_2d.py --overwrite
```

The generator checks Boolean/native proposal agreement, source and protected-label preservation, recipient ownership, and pixel accounting before saving the figures. Generated illustration files remain under ignored `outputs/`; the generator and this locator document are versionable. Frozen cohort inputs and generation modules are unchanged.
