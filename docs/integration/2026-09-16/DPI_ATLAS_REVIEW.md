# DPI atlas review for FakeCT tissue mapping

Read-only source review at DPI commit `c7532cfd3c0230cb6cf76f83afe924ed6d791edf`.

## What is reusable

`config/atlas/anatomy_atlas.csv` contains anatomical surface classifications and review overrides; `config/source/xcat_anatomy_hierarchy.yml` supplies aliases and prioritized fallback regular-expression rules. Together they can form the anatomical layer of FakeCT's tissue grouping. The cardiovascular 1.1 and vascular routing atlases assign downstream model roles, including endocardium/myocardium distinctions. They are different schemas and should not be read as the source anatomy table or as numeric voxel-label dictionaries.

Source fields preserve original name, canonical name, normalized name, system/subsystem, anatomical region, structure type, laterality, piece number, temporal phase, hierarchy path, classification confidence/rule, and review evidence. `original_block_index` is a surface block ordinal, **not** an `act` voxel label. `example_source_file` is review evidence, not automatically an active matcher. Neither atlas has the numeric organ IDs necessary to recover the 41 missing ID names. A separately discovered supplemental dictionary, `high_res/040/organ_ids.txt`, supplies names for 30 of these 41 IDs; 11 remain absent from the merged dictionaries.

## Audited counts

- 459 source-atlas rows, 453 distinct original names. Duplicate names are `arteries_lung` and `veins_lung`, each with four indexed rows.
- Eight indexed rows; all source_file fields empty. The example paths span adult and infant cases and must not silently become universal case geometry mappings.
- 421 `corrected` rows, 37 rows with blank review_action, one `needs_review` row. Native DPI accepts 458 rows.
- The 37 appended cardiac rows contain 23 fields against 22 header columns: an extra empty column after classification_rule shifts review metadata. Their core classification fields remain readable, but provenance should record the malformed record and any conservative exclusion.
- The organ dictionary has 3119 signed unique IDs but only 1134 distinct names. Multiple IDs sharing a name is normal. 921 IDs have an exact original-name atlas row before checking scope or applying fallback rules.
- Independently compiled the current native AnatomyClassifier.cpp in /tmp and classified all dictionary names with index=-1: 3117 classified, two `throat` entries unclassified. This is name-only native classification coverage, **not** validated tissue/material correctness.
- Reference native output: `/tmp/fakect-organ-native-classifications.csv`. Counts: `/tmp/fakect-dpi-atlas-counts.json`.

## Native matching behavior

1. Normalize raw name; apply raw alias. Extract temporal prefix, laterality prefix/suffix, trailing piece number, then laterality again if needed. Apply normalized alias.
2. Construct search text `normalized_name + ' ' + canonical_name + ' ' + normalized_raw`.
3. Sort regex rules by descending integer priority then ascending rule ID. Apply the first case-insensitive regex search. Rule fields overwrite nonempty classification fields; an earlier alias can preserve confidence.
4. Apply CSV override using exact `(source_file, original_block_index, original_name)`, then `(original_block_index, original_name)`, then original-name-only. Indexed rows are not inserted into the name-only map. Duplicate map keys are last-row-wins.
5. Override actions accepted natively are empty, corrected, override, manual_override, approved, accepted, active. Others, including needs_review, are skipped. If a CSV column exists, its value overwrites the field even when the cell is empty. Core derived fields are refreshed afterward.

The native config loader has a detail relevant to exact parity: automatic discovery loads YAML onto the base system-root config; an explicit YAML filename loads it on top of compiled defaults, appending rules. FakeCT should pin one declared snapshot and state its interpretation instead of depending on current working directory.

A conservative FakeCT loader may intentionally reject malformed/pending/ambiguous records or tied conflicting rules. Document these differences rather than calling it exact DPI runtime parity.

## Tissue/material caveats demonstrated by existing data

A mesh boundary's anatomical classification is **not** a guarantee of the material inside it. Prior XCAT sample results already demonstrate this:

- Case 260602 `head` (ID 1), `chest_surface` (ID 4), and many limb/digit IDs classified under Integumentary/Skin have sampled attenuation 0.14807545 cm^-1, matching the log's adipose material.
- Known negative `skull_inner` IDs such as -829 and -811 have sampled attenuation 0.16450932 cm^-1, matching red marrow, although the native hierarchy places skull_inner under Bone.
- Arteries and veins can share the same contrast-enhanced blood attenuation. Intensity cannot recover vessel identity.
- Name-only native classification puts `arteries_lung` and `veins_lung` under Lung; their reviewed indexed overrides identify internal thoracic vessels. Conservative unknown or tissue-consensus logic is better than asserting pulmonary identity without the matching mesh context.
- Brain ventricular labels have reviewed Nervous/Fluid overrides; the generic heart regex otherwise catches ventricle. Heart myocardium/papillary labels retain structure_type=heart, so type alone cannot identify blood versus muscle.
- `index` has type organ but Skin hierarchy. Use full hierarchy/canonical evidence, not only structure_type.

Recommended data model is original signed anatomical label + full per-label atlas metadata + a replaceable small anatomical-group code + independent sampled scalar/material evidence. Keep material assignment distinguishable from anatomical-group assignment. Do not infer missing names from ID adjacency, abs(ID), or mesh ordinals. Any geometry-based crosswalk needs an explicit surface-to-volume transform, interior sampling, overlap/ambiguity checks, and review of nested boundaries.

## Proposed finite grouping (design proposal, not atlas truth)

For an initial anatomical editing vocabulary, ten nonbackground groups could be soft tissue, muscle, bone, cartilage/connective tissue, arteries, veins, cardiac structures, lung/airway, nervous tissue, and adipose. Keep background and unresolved as separate reserved codes. Skin can initially sit under soft tissue and fluids under their anatomical parent while remaining selectable through retained metadata. If CT material physics is the intended grouping instead, add independent material classes for air, fluid, marrow, etc.; collapsing to ten should not force incorrect anatomical/material equivalence.

Before calling a group a validated tissue material, review per-ID attenuation distributions normalized by pixel_width_cm and compare them with the case log. Retain energy, contrast settings, case/frame, original units, source hashes, source paths, review state, selected mapping rule, and all original IDs. Range overlap is expected; report ranges for descriptive QA, not universal threshold segmentation.

## Unknown IDs

Atlas alone does not name any of the 41 previously sampled IDs absent from the primary organ_ids.txt:

`[-830, -828, -827, -825, -824, -822, -821, -820, -817, -816, -814, -812, -806, -805, -802, 430, 431, 583, 584, 586, 871, 872, 873, 874, 875, 876, 877, 878, 879, 887, 1014, 1016, 1017, 1018, 1019, 1020, 1021, 1022, 1023, 1024, 1027]`

The supplemental `/home/aghorban/slurm/xcat/high_res/040/organ_ids.txt` has 3129 entries; all 3083 shared IDs have exactly the same names as the primary dictionary. A conflict-checked merge has 3165 named IDs and resolves 30 of these 41 gaps explicitly. The resolved names include skull_inner, lmusc336, lmusc1663, tendons_lleg, cart, sinus, and male reproductive structures. Eleven sampled signed IDs remain absent after merging: `[-830, -827, -825, -824, -822, -821, -817, -816, -814, -806, -802]`. Preserve which dictionary supplied each name and both source hashes. The supplemental dictionary is useful explicit name evidence, but matching shared IDs alone does not prove all case/voxel geometry associations. Scalar values can suggest material candidates for the remaining 11, but cannot establish anatomical identity.

## Relevant source locations

- config/atlas/README.md: source vs downstream atlas roles and SHA256 provenance policy.
- src/source/adapters/xcat/AnatomyClassifier.cpp:813 classifier ordering, 825 classification, 1348 config loading, 1406 CSV loading, 1592 override precedence.
- tests/native/test_xcat_source_adapter.cpp:784 override test, 873 pending-atlas append test.
- src/source/adapters/xcat/RawSurfaceReader.cpp: ASCII token labels followed by triangle coordinates; geometry transforms are scale/translate, not an implicit voxel alignment.
- FakeCT docs/integration/2026-09-16/xcat-results.json: existing sampled scalar-to-signed-ID pairs.

## Independent FakeCT adapter comparison

Compared the initial `src/fakect_tissues.py` implementation with the separately compiled native DPI reference for every one of the 1134 distinct primary dictionary names and all 453 atlas names. The 22 name-level anatomy differences in the dictionary comparison are explained by intentionally empty default fields, tissue-only agreement across indexed vessel alternatives, and skipping the 37 malformed CSV rows in favor of YAML. No unexplained current-dataset anatomical classification mismatch was found. Examples correctly distinguish papillary/myocardial muscle from unresolved heart chambers, brain ventricular fluid from heart anatomy, and scoped arteries_lung/veins_lung tissue consensus from their unconfirmed fine anatomy.

The proposed policy intentionally excludes unresolved airway/cavity, generic cardiac chambers, and sensory organ/duct types rather than silently putting every unresolved structure in soft tissue. Broad aorta names such as ascending_aorta and descending_aorta are currently unknown even though their generic aorta subtype is clear; this is an optional policy completeness improvement, not loss of original information.

An independently demonstrated numeric-boundary defect was reported to the core implementation agent: float32 2^31 can pass an upper check written as `value > 2^31-1` because NumPy rounds that bound to 2^31 before comparison. The safe upper rejection is `value >= 2^31`. This validation protects the signed original-ID layer from a wrap to -2^31.

The float32 upper-bound fix was independently rerun after implementation: float32, float64, uint32 and int64 inputs equal to 2^31 now all raise ValueError. A subsequent conservative rule also detects numeric musc labels misclassified by the broad c1 vertebral regex (e.g. musc104), keeping them unknown until reviewed rather than repeating that native anatomical error.
