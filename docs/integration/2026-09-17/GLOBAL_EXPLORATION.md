# Global-to-local ROI exploration without anatomical IDs

The report now has separate Global and Local tabs. Global navigation covers the
whole original phantom; Local view begins with native crop close-ups. Named edit
recipes add an Edits / recipe tab. Inputs and provenance remain accessible in
the same portable HTML file. Tabs stay visible while scrolling; keyboard arrows,
Home/End, and existing section links select the corresponding panel. Static
sections remain available when JavaScript is disabled or the report is printed.

`selection.source_ids` may be blank or omitted for preview, single-edit, and
recipe inputs. The selected mask is the requested tissue intersected with the
ROI. Optional `stiffness.labels` overrides may also be omitted. The original
fine IDs remain in the catalog and output provenance; users do not need to
enter them to place an ROI or perform tissue-based edits.

## Generated examples

- [Tissue-only exploration report](../../../outputs/studies/thoracic-aorta/explore-v1/report.html)
  and [input](../../../configs/studies/thoracic-aorta-explore.ini).
- [Existing aorta recipe with the new tabs](../../../outputs/studies/thoracic-aorta/recipe-v3/report.html)
  and [input](../../../configs/studies/thoracic-aorta-recipe.ini).
- [Usage instructions](../../ROI_EXPLORATION.md) and
  [machine-readable verification](global-exploration-results.json).

Both use case 260602, frame 1 and the reviewed overall ROI. The full source is
`1860×750×750` in k,j,i order. Global views sample `187×76×76` (1,080,112 voxels)
with nominal native strides of 10 on each axis. Both source edges are retained;
the final intervals are 9 voxels. Exact sampled coordinate axes and source
sample hashes are embedded. The reader retains only a bounded axial plane and
the sampled volume. Source files remain read-only.

Global controls browse three linked slices, switch attenuation/tissue displays,
show saved ROI overlays, move a temporary sphere guide, and copy native
coordinates. Browser checks confirm that image clicks and sliders update the
coordinate readout. Whole-body sampling is coarse; thin vessels can be missed.
The native local crop remains `146×104×65`, with 1 mm spacing.

The tissue-only example selects **52,576** arterial voxels in the ROI, compared
with **50,485** aorta-ID-filtered voxels. The additional **2,091** voxels belong
to neighboring arterial labels. There are seven six-connected components;
these are spatial diagnostics, not seven identified anatomical vessels. The
result demonstrates why a tissue-only tube must be refined geometrically if
nearby arteries enter it. A synthetic end-to-end fixture additionally confirms
that a tube isolates one of two nearby arteries even when they share one ID.

The updated recipe retains its optional aorta filter so the editing comparison
is unchanged. Its labels, attenuation proxy, changed mask and original arrays
are exactly equal to the previous stiffness trial: 741 additions and 1,500
removals. Browsing in Global view does not change this captured result.

## Verified behavior and limits

The complete suite passed **194 tests**. After final tab focus/layout refinements,
17 report tests and five recipe-report tests passed again. Firefox 140.12.0 ESR
headless checks exercised tab activation, keyboard focus, deep links, all three
global canvases, sliders, native-coordinate entry, clicks, tissue display,
guide radius and reset. Browser screenshots were inspected. This check did not
validate WebGL rendering of the existing local 3D viewer.

Artifact and generating-code hashes were checked for both published reports;
source stats remained unchanged. Source IDs were absent from the exploration
selection input, and its native candidate and selected masks were independently
checked against the artery tissue mask and ROI. Full-volume sampling tests cover
anisotropic coordinates, edge retention, bounded reads, source changes and
variable-radius ROI geometry.

The global guide is temporary. Copy coordinates to the INI, save it, select a
fresh output directory, and rerun to regenerate native local views or edits.
Training ground-truth targets are separate: the current anatomy-specific study
trainer still requires explicit `train.target_source_ids`. This change makes
ROI placement and editing independent of user-supplied anatomical IDs; it does
not silently redefine training targets.
