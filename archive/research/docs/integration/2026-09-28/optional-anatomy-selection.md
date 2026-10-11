# Optional anatomical selection for CoA cohorts

The existing selection remains available: `tissue = artery` with an ROI selects
all arterial label voxels inside that ROI. The optional anatomical step narrows
the original label IDs before applying the same ROI and editing pipeline. It
does not change existing configurations unless enabled.

The first supported automatic region is `thoracic_aorta`. It uses the pinned
atlas hierarchy, case/frame RAW surface names, and native ACT label contact to
qualify aortic and connected arterial branch IDs. Pulmonary hierarchy labels are
excluded before connectivity analysis, even when they touch the aorta on the
voxel grid.

Use the separate example `archive/research/configs/cohorts/coa/260602-anatomy.ini`. Its selection
section has this form:

```ini
[selection]
tissue = artery
source_ids =
anatomy = thoracic_aorta
anatomical_labels =
anatomy_file = outputs/anatomy/coa/260602-r3/selection.json
include_branches = true
connectivity = 6
protect_other_arteries = true
```

- `anatomy` enables automatic resolution. Omit the anatomy-related options for
  the existing tissue-only approach; an `anatomy_file` without `anatomy` is invalid.
- `source_ids` remains an optional numeric restriction. A blank value needs no
  knowledge of XCAT IDs.
- `anatomical_labels` optionally restricts original or atlas canonical names.
  Multiple comma-separated names form an allowed-name list; the list intersects
  the tissue, explicit IDs and automatic anatomical policy. Unknown names fail
  validation. Names do not override pulmonary exclusion or manufacture a label
  that the source dictionary does not contain.
- `anatomy_file` is the immutable, case-specific resolution artifact. Choose a
  fresh directory for each revised selector; the resolver refuses existing
  output directories.
- `include_branches = true` allows eligible systemic, coronary and
  cerebrovascular arterial labels that connect to an aortic anchor in the
  bounded crop and also intersect the ROI. `false` restricts eligibility to the
  atlas aortic hierarchy. It does not automatically assign specific branch
  names to generic artery labels.
- `connectivity = 6` requires face contact between native voxels. `26` also
  admits edge/corner contacts. This is voxel-grid contact, not independently
  verified physiological connectivity.
- `protect_other_arteries = true` keeps excluded arterial IDs out of the
  editing donor/recipient pool. This avoids expanding into a nearby excluded
  pulmonary or systemic artery even when ordinary tissue borrowing is permissive.
  `false` leaves those voxels subject to the existing reassignment and stiffness
  rules. Neither setting changes the selected-ID list.

## Resolve, preview and review

For a fresh selector revision, run from the integration worktree:

```bash
cd /home/aghorban/repo/FakeCT/.worktrees/fakect.26.09.16
python3 scripts/resolve_anatomy.py --config archive/research/configs/cohorts/coa/260602-anatomy.ini
python3 scripts/prepare_cohort.py --config archive/research/configs/cohorts/coa/260602-anatomy.ini --stage preview
```

The resolver reads a bounded native crop and inventories the source surfaces.
It writes `selection.json`, its SHA256 sidecar and the captured INI under
`outputs/anatomy/coa/260602-r3/`. It does not apply geometry edits or generate
paired datasets. The preview then uses that captured selection and runs the
configured base recipe; it does not prepare the parameter grid.

If the referenced selector artifact already exists and matches the inputs, skip
`resolve_anatomy.py` and run the preview command directly. Resolution is a
one-time step for each selector revision, not a prerequisite that must be rerun
before every preview. The example preview destination is
`outputs/cohorts/coa/260602/anatomy-review-r1/report.html`; after that directory
exists, choose a fresh `[output] directory` for another preview.

The preview report includes accepted and excluded original IDs, original and
canonical names, atlas paths, ROI/crop/contact counts, reasons and uncertainty.
RAW evidence includes verified centerline status and available ACT samples.
The report also embeds a downloadable `anatomy-selection.json` copy of the
resolved artifact. Global cyan overlays locate accepted IDs across the phantom;
the local native views show the actual ROI intersection.

After changing the main ROI, its crop bounds, anatomy settings, source frame,
catalog or selector implementation, select a fresh `anatomy_file` revision and
run the resolver again. The loader rejects stale dependencies. Edit-only changes
can reuse an existing artifact when they leave its recorded selection, source
and crop geometry unchanged. Choose a fresh preview output directory for each
trial. Changing the selected IDs changes the target; repeat review and native
preflight before freezing or preparing a new cohort revision. Existing paired
datasets and freezes remain unchanged.

## What the source channels establish

The ACT `.bin` contains original signed anatomical IDs, stored as integral
values in the native volume. The ATN `.bin` contains attenuation values; it
provides image context and attenuation summaries, not anatomical identity.
Use the same instantaneous case/frame for ACT, ATN and RAW evidence. Averaged
ACT values are not valid categorical labels.

The existing `[input] root`, `case_id`, `frame`, `audit` and `catalog` locate the
volume channels, case `.par` geometry, pinned atlas-derived catalog and RAW
surface groups. Automatic selection requires the expected RAW groups for that
case/frame. Missing or malformed surface inputs fail instead of silently
inventing identities.

RAW surface ordinals are local positions in a file, **never ACT voxel IDs**.
Exact original names join surface records to candidate dictionary IDs. Repeated
names can map to several IDs or surfaces, so this join may be ambiguous. The
audit preserves that ambiguity rather than treating the Nth surface as label N.

For supported coordinate contracts, RAW bounds supply a fast candidate screen.
Tubular centerlines are recovered only when welded topology verifies two poles
and intervening rings; ring centroids and radii then provide additional spatial
evidence. Existing ACT cells can be sampled along those centerlines. No surface
mesh is voxelized. Bounding-box or centerline-envelope overlap does not prove
surface intersection or a unique whole-surface ID assignment. Unsupported
registration/topology leaves the corresponding spatial evidence unavailable;
native labeled-volume contact determines connectivity.

## Scope and limits of the mask

The automatic resolver finds an atlas-qualified aortic anchor inside the main
ROI. Eligible labels connect through the **bounded native crop**, so a branch
whose only connecting path is outside that crop may be excluded. Pulmonary
labels cannot serve as connecting paths.

The output is a list of **whole original IDs**, not a component-only mask:

```text
initial target = main ROI ∩ voxels carrying any accepted original ID
```

Consequently, disconnected pieces that share an accepted ID remain selected
when they fall inside the ROI. The report records these extra voxels. Refine the
ROI when needed; this optional step does not replace anatomical review. The
atlas may describe an ID only as a generic systemic artery; it cannot safely be
renamed carotid or subclavian solely because it touches the aortic target.

Edits retain the existing lineage behavior: dilation can grow beyond the
original selector, and erosion removes selected descendants under the chosen
reassignment and safeguard policies. The anatomical filter qualifies the
initial target and optionally protects excluded arteries; it is not a new
anatomical segmentation model.

## Optional manual names without automatic resolution

For an explicit, simpler name filter, omit both `anatomy` and `anatomy_file`:

```ini
[selection]
tissue = artery
source_ids =
anatomical_labels = dias_aorta
protect_other_arteries = true
```

Run the normal preview command directly. This selects IDs whose original or
canonical names match the supplied names and intersects them with the ROI.
It requires no RAW discovery and performs no aortic-anchor connectivity test.
Adding a manual list to `anatomy = thoracic_aorta` instead narrows automatic
eligibility, so that combined mode still requires resolution and a valid
artifact.

## Verified example: case 260602

The new optional example has a completed review at
`outputs/cohorts/coa/260602/anatomy-review-r1/report.html`. The compact numerical
record is [260602-anatomy-evaluation.json](260602-anatomy-evaluation.json).

All 643 catalog-artery surface blocks recovered verified two-pole/ring
centerlines, within an inventory of 2,634 surfaces and 4,336,446 triangles.
No surface was voxelized. The scoped RAW/BIN coordinate contract passed.

| Native label | Original name | ROI voxels | Decision |
|---|---|---:|---|
| 2922 | dias_aorta | 50,485 | Include aortic anchor |
| 1186 | internal_carotid_right | 311 | Include connected branch |
| 1547 | arteries | 265 | Include generic connected branch |
| 1550 | arteries | 657 | Include generic connected branch |
| 2912 | dias_rca1 | 59 | Include connected coronary label |
| 2914 | dias_lad1 | 112 | Include connected coronary label |
| 1747 | right_pulmonary_arts | 76 | Exclude pulmonary hierarchy |
| 1969 | left_pulmonary_arts | 116 | Exclude pulmonary hierarchy |
| 2920 | dias_pul_art | 495 | Exclude pulmonary hierarchy |

The initial target decreased from 52,576 to 51,889 voxels by excluding 687
pulmonary-label voxels. No disconnected pieces sharing admitted IDs remained
inside this ROI. Native samples of vessel surface ordinals 2 and 5 corroborated
IDs 1547 and 1550 respectively; those ordinals are not the IDs themselves.

All three base recipe edits completed: 635 voxels added and 904 removed.
All 78,639 excluded arterial voxels in the crop retained their labels and
attenuation. Existing stiffness and recipient rules still apply: 2,240 proposed
removal voxels lacked acceptable recipients and were preserved. The preview
shows achieved changes, not an assumption that requested distance is attained.

All 16 planned combinations passed metadata validation, and 476 regression
tests passed. The complete native parameter sweep, freeze, paired generation
and model training were not run. Review this new selection and its edits before
marking the INI reviewed and performing native preflight. Other phantom INIs
and existing prepared datasets were not changed.
