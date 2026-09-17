# DPI atlas currency check

The FakeCT copies already match the latest applicable files in the accessible local DPI checkout. They also match the frozen inputs and actual output provenance of DPI's latest accessible v6 cardiovascular campaign. **No newer source anatomy classifications were found to import.** The `1.0.0-proposed` label in the original preview is the version of FakeCT's editable ten-group tissue policy, not the age or version of the DPI atlas.

The check used DPI branch `generic-anatomy-model-s1`, HEAD `c7532cfd3c0230cb6cf76f83afe924ed6d791edf`. The relevant atlas, hierarchy, and native classifier files match that HEAD despite unrelated working-tree changes elsewhere in DPI. DPI was read without modification; no remote fetch was performed. Full hashes and paths are in [CURRENCY.json](../../../references/atlas/dpi/CURRENCY.json).

## Which atlas does what

| Document | Role | Version or content identity | Result |
|---|---|---|---|
| `anatomy_atlas.csv` | Reviewed source surface-name classifications and scoped overrides | SHA256 `59268c0e15f7...`; no release-version column | Already current |
| `xcat_anatomy_hierarchy.yml` | Source aliases and prioritized hierarchy rules | SHA256 `601355516163...`; no release-version field | Already current |
| `cardiovascular_atlas_v1_1.csv` | Downstream cardiovascular roles, including endocardium, myocardium, and papillary structures | `1.1.0`, SHA256 `e80e03c811ae...` | Already current and copied |
| `vascular_routing_atlas_v1.csv` | Downstream left/right circuit and territory routing | `1.0.0`, SHA256 `b13687d85a3e...` | Added as reference alongside the others |
| `tissue-policy.v1.json` | FakeCT's replaceable ten-group anatomical vocabulary | `1.0.0-proposed` | Independent of all DPI versions |

The added routing atlas is reference material; it is not fed into the tissue classifier. Routing and cardiovascular role tables have different schemas from source anatomy overrides and do not supply a numeric voxel-ID dictionary. Replacing the source atlas with one of those tables would change its meaning rather than update it.

The source hierarchy's last committed change was `3a75ffed7692edbf660d40eca568c86edde0bbbe` on 2026-08-18. The source anatomy CSV's last committed change was `1e70249f84388bcb184de3985505a7f07bac3ea4` on 2026-07-30. A newer pipeline run does not necessarily require a newer atlas file.

## Evidence from the latest run

The latest accessible frozen anatomy-instance campaign is:

```text
/scratch/aghorban/dpi/xcat-anatomy-instance-all-frames-four-quarter-v6-20260822-r1
```

All four classification/routing control documents in its `campaign_inputs/` directory match the current DPI files and FakeCT reference copies byte for byte. Their frozen checksum inventory also agrees. This check inspected the actual `cases/260602/frames/frame_001/topology/cardiovascular_selection.json` and `anatomy_instance_manifest.json`: the selection records cardiovascular atlas `1.1.0`, and the recorded source-atlas, hierarchy, cardiovascular-atlas, and routing-atlas hashes agree with the copies above.

DPI's v6 changes pulmonary component-root connectivity recovery. It is an execution-policy version, not source atlas version 6. The later 2026-08-23 S2 reference-model archives contain derived branch/ownership atlases and accepted S1 handoffs; those describe realized model geometry and connectivity rather than a replacement whole-body source classification CSV.

The native loader in `src/source/adapters/xcat/AnatomyClassifier.cpp` accepts an explicit hierarchy YAML or discovers `config/source/xcat_anatomy_hierarchy.yml`. The campaign scripts explicitly freeze and pass the source anatomy, cardiovascular, and routing atlases. FakeCT pins the two source classification files by content hash instead of relying on working-directory discovery.

## Implications for small structures

Using the current source atlas does not validate every voxel's material or every surface-to-volume association. The current source files still contain the issues documented in [DPI_ATLAS_REVIEW.md](DPI_ATLAS_REVIEW.md), including 37 malformed cardiac override rows, scoped vessel rows whose global surface indices cannot be assumed to equal voxel IDs, and broad fallback regex matches. FakeCT retains its conservative guards for the `bladder_inner`/`LAD` substring collision and numeric `musc` names matching vertebral `c1`; these remain review-required rather than silently acquiring a confident group.

The previous whole-body preview also decimates the long-axis context and in-plane samples. Thin structures can disappear or look discontinuous even when their original labels are correct. Full-resolution close-ups are the appropriate place to review those elements. Original signed IDs, surface names, detailed hierarchy, candidate alternatives, and matching evidence remain available underneath the small tissue vocabulary.

Future reviewed corrections should receive a new upstream snapshot/hash and an explicit FakeCT policy or override revision. They should be compared on affected original IDs and shown in native-resolution ROI previews before claiming improved small-structure classification. This currency check makes no tissue assignments and does not change previous numerical evaluations.

## Preview identification

The rendering script now labels the version explicitly as **FakeCT tissue policy** and separately displays the DPI source-atlas and hierarchy SHA256 prefixes. Its JSON provenance records the full classification-source hashes and policy version. The whole-body tissue preview has been regenerated with this clarification and unchanged measured group counts; previous images remain available in Git history. New ROI close-ups also identify the grouping policy and source atlas separately.
