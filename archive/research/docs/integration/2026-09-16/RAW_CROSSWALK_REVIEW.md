# XCAT surface-to-label crosswalk review

Read-only inspection on 2026-09-16. Source repositories and phantom files were not modified. Inventory: `/tmp/fakect-raw-surface-inventory.json`.

**Current coverage: 30 of the earlier 41 dictionary gaps are resolved by explicit supplemental entries; 11 signed IDs remain unresolved.**

## What can be reused

DPI's `config/atlas/anatomy_atlas.csv` is a reviewed override table, not the complete ontology by itself. `config/source/xcat_anatomy_hierarchy.yml` supplies aliases and routing rules, and `src/source/adapters/xcat/AnatomyClassifier.cpp` supplies the classifier defaults. Apply the same classification logic before the reviewed overrides. Preserve original name, canonical name, hierarchy, structure type, laterality, temporal phase, source file, surface occurrence, rule, confidence, and source document checksums when deriving a finite tissue view.

The downstream `cardiovascular_atlas_v1_1.csv` further distinguishes blood-contact endocardium, myocardium, epicardium and papillary muscle. A generic cardiovascular hierarchy is therefore not sufficient to label all enclosed voxels arterial or venous blood. Surface role and volume material are distinct.

## Bounded surface inventory

Exactly six complete frame-1 files for case 260602 were streamed once, 433,118,866 bytes total. Numeric triangle lines were counted but geometry was not loaded. No other raw frame was scanned.

| Group | Surface occurrences | Base dictionary name matches | Exact atlas row matches |
|---|---:|---:|---:|
| body | 34 | 34 | 11 |
| bones | 376 | 376 | 28 |
| heart | 27 | 27 | 21 |
| muscle | 387 | 387 | 367 |
| organs | 481 | 468 | 94 |
| vessels | 1,329 | 1,329 | 400 |
| Total | 2,634 | 2,621 | 921 |

There are 912 distinct raw names. Repeated names are significant: `left_pulmonary_veins` occurs 228 times; `right_pulmonary_arts` 221; `arteries` 51; `skull` 32. The inventory preserves every occurrence and local zero-based position. Exact name joins identify a family of dictionary IDs, not a unique numeric ID for each repeated surface fragment.

The 13 names absent from the base dictionary are male reproductive structures: `penis`, `testicles`, `prostate`, `r_testis`, `l_testis`, `r_seminal_ves`, `l_seminal_ves`, `r_vas_def`, `l_vas_def`, `r_epididymus`, `l_epididymus`, `ejac_duct`, `penis_inner`. Every one is present in the alternate 040 dictionary; the union covers all 2,634 raw surface occurrences by exact name.

## Numeric IDs: a useful additional dictionary

`/home/aghorban/slurm/xcat/high_res/040/organ_ids.txt` differs from the primary dictionary and supplies authoritative text entries for 30 of the previously sampled 41 missing IDs:

- `-828, -820, -812, -805`: `skull_inner`
- `430`: `lmusc336`; `431`: `lmusc1663`
- `583, 584, 586`: `tendons_lleg`
- `871` through `879`: `cart`
- `887`: `sinus`
- `1014`: `penis`; `1016`: `testicles`; `1017`: `prostate`
- `1018`: `r_testis`; `1019`: `l_testis`
- `1020`: `r_seminal_ves`; `1021`: `l_seminal_ves`
- `1022`: `r_vas_def`; `1023`: `l_vas_def`
- `1024`: `r_epididymus`; `1027`: `penis_inner`

The two dictionaries share 3,083 signed IDs, with **zero naming disagreements**. There are 46 additional IDs in 040 and 36 only in the base. The 020 dictionary is byte-identical to the base. A provenance-bearing union may therefore be used for an explicit supplemental dictionary, with conflict detection retained and each source recorded. The lack of conflicts supports compatibility but is not a proof of per-case generator provenance.

SHA-256:
- base and 020: `e429081ac4412344f77a99ce1ed5ba0e4e3cc00fabe949153d9aa938988aa7e5`
- 040: `7bee1ae8717f94a4dd7bca7f4c3e8b2409a0fbae9227c2a1be5dfb3b382b4390`

Eleven IDs remain absent from every located dictionary: `-830, -827, -825, -824, -822, -821, -817, -816, -814, -806, -802`. Their positive counterparts are named `skull`, and some other explicitly declared negative skull IDs are `skull_inner`; this is evidence for review, **not an exact signed assignment**. Preserve them as unknown until a per-case generator map or validated spatial correspondence confirms them. No `skull_inner` mesh header occurs in the sampled case's six raw files, so these inner compartments cannot be joined directly by surface name in this scan.

## Why surface ordinals cannot fill dictionary gaps

`src/source/adapters/xcat/main.cpp:464-480` constructs each `originalIndex` as the number of already appended surfaces plus the local ordinal. `VtkSceneWriter.cpp:252-257` passes this ordinal to the classifier, and `VtkSceneWriter.cpp:74-116` exports it as `OriginalBlockIndex` alongside semantic metadata. This is not an act volume label. For example, the reviewed atlas's `musc1081` example index is 437 while `organ_ids.txt` assigns `musc1081 = 44`.

`RawSurfaceReader.cpp:172-225` parses only nonnumeric name tokens and groups of 9 numeric coordinates. `SurfaceTypes.h` stores group, label, originalLabel, sourceFile, originalIndex and geometry; no act ID appears. Searches of DPI's adapter/source config/docs found no act-ID crosswalk. The atlas CSV contains no numeric act-ID column. Therefore neither raw order nor `example_original_block_index` can serve as a fallback label number.

## Integration implication

Use two independent identity layers: immutable fine signed act labels plus the source dictionary, and named surface occurrences plus DPI classifications. Derive coarse tissue labels as a versioned lookup onto the fine labels. Sample paired atn values under known act IDs to estimate attenuation distributions, retaining unknown IDs separately. Surface geometry can validate correspondence and help resolve gaps after registration, overlap, closure and boundary tests. Do not infer artery versus vein from attenuation alone, or assume a surface's enclosing volume is homogeneously the surface's structure type.

## Independent first-implementation checks

The new `scripts/prepare_tissues.py` passed bounded independent checks against the current source at review time:

- Dictionary unions preserve sources and reject conflicting signed-ID names.
- Real six-file inventory yields 2,634 occurrences, 796 unique-name candidates and 1,838 explicitly ambiguous name-family joins. No ordinal is used as an act ID.
- A one-case profile yields 674 observed IDs. Only -830 and -817 lack dictionary entries in the case260602 sample; all measured attenuation levels matched at least one coefficient in that case's log using the stated rounding tolerance.
- Signed original arrays, the entire catalog, geometry JSON, and source checksums round-trip through NPZ. Strict unknown rejection, permissive unknown retention, and overwrite refusal passed.
- Explicit skull_inner labels remain marked unknown/material-review rather than inheriting the bone category.

The current Python adapter explicitly uses the exported YAML/CSV rules and does not claim full parity with all of DPI's built-in C++ rules. Missing coverage therefore remains visible. Following this review, the NPZ helper was extended to require matching shape, explicit zyx array order and positive finite spacing, and to check optional crop origins against source dimensions. It preserves additional geometry metadata but does not establish anatomical orientation or surface-to-volume registration; it remains a bounded archival view pending the full import contract. No critical defects were found in the reviewed bounded workflow.
