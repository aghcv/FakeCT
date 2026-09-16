# Multi-label notebook audit — 2026-09-16

## Sources and execution scope

- Dilation (`D` below): `/home/aghorban/.codex/attachments/eee0320a-7c9b-44e9-bbbd-86eb116832ba/pasted-text.txt`, 4,789 lines.
- Erosion (`E` below): `/home/aghorban/.codex/attachments/11085a55-af1e-4f43-b350-69cd17fe56b4/pasted-text.txt`, 1,795 lines.
- Reproducible numerical harness: `scripts/evaluation/audit_notebooks.py`; machine-readable results: `notebook-results.json`.
- Ran selected pure function definitions extracted by Python AST, with installed NumPy 2.0.2 and SciPy. No notebook top-level code, downloads, installations, production data writes, or GPU work ran. Plot helpers in tests were replaced with a no-op.
- Dilation's full pasted Python export does **not parse**: `-import numpy as np` at D:3121. Harness fixes only that typo in an in-memory string for extraction. Erosion parses. Dilation contains 94 top-level function definitions, many repeated/redefined; extraction uses the final definition of each requested name. This is not a production module to import wholesale.

## Reusable functionality

1. Binary morphology/ROI composition: 4-neighbor Boolean morphology and `result=(original & ~roi)|(candidate & roi)` (D:391–445 / E:390–444). The array of integer label IDs remains authoritative; bitwise edits apply only to Boolean target masks.
2. Dilation donor policy (final 2D D:3509; 3D D:3683): copy labels; borrow only requested voxels from allowed, unprotected donor labels; report original donor counts and blocked mask. Protection takes precedence. This logic is dimension-independent and can become one validated n-D kernel.
3. 3D 6-neighbor morphology (D:4149), sphere ROI in physical spacing (D:4115), and sphere/ellipsoid/focal/tapered spatial domains (D:4161) provide useful starting geometry helpers.
4. Final VTI donor round (D:4229) uses an explicit per-band `borrowable`/`resistance` table and force threshold. Unlike the earlier borrow helpers, it explicitly excludes voxels already carrying the target label. Growth is propagated from **successfully claimed** target voxels (D:4463–4485), so blocked tissue cannot be tunneled through by a blindly expanding candidate mask.
5. Erosion recipient policy (E:566–697): all reads use the original label snapshot, yielding order-independent reassignment. Direct 4-neighbor majority wins; priority breaks ties; if no direct donor exists, search increasing Chebyshev square rings up to the configured pixel radius. If none exists, retain target label and mark unresolved. Preserve the unresolved mask and distinguish requested from achieved geometry.
6. Interactive discovery pattern (D:4259–4285 and D:4528–4715): histogram colors match category colors; manual `(z,i,j)` centers; multiple z-slice previews with i/j ticks; optional selected-label outlines; true sphere intersections projected as ellipses on slices; separate preview and edit flags. This closely matches the requested terminal `prepare`/`preview`/`generate` workflow.
7. Classification is explicitly separated from original scalar intensities (D:4103, D:4361–4369). Optional post-edit scalar visualization fills claimed voxels with the seed median (D:4487–4491); it is a display proxy, **not** CT recovery, XCAT physics, or AI reconstruction.

## Numerical evaluation

All 14 selected notebook test functions passed (10 dilation test functions and 4 erosion test functions, including the four-case recipient policy suite). Additional tests:

| Evaluation | Observed result | Meaning |
|---|---|---|
| 30 random 3D donor-policy cases, target excluded from donor set | All invariants passed | Outside-request labels and protected labels preserved; donor totals equal changed voxels |
| 30 random 3D six-neighbor masks vs SciPy reference | All exactly equal | Slice-shift morphology implementation agrees with 6-connectivity reference, including borders |
| Full protected 3D separating plane | 0 protected changes; 0 target voxels beyond plane; 324 target voxels on seed side | Final VTI frontier logic preserves barrier behavior |
| Target label accidentally included in allowed donors, 2D and 3D | 1 changed voxel but donor totals = 2 | Earlier borrow helper incorrectly counts self-borrowing; existing self-borrow tests exclude target from allowed donors and miss this |
| Erosion center inside closed protected 3×3 ring, recipient outside ring | Center assigned outer recipient; 0 protected voxels altered | Square-ring fallback can select tissue across a protected barrier; protected-label preservation does not imply anatomically local reassignment |
| Erosion priority `[2,3]`, neighboring label 7 | Label 7 accepted | `recipient_priority` is only a tie-break preference, **not** an allowed-recipient list; add an explicit whitelist |
| Invalid erosion request at protected center label 4 | Center changed from 4 to 2 | Helper trusts `released_mask`; it does not check every requested voxel is the target and unprotected |
| Erosion 3D input | `ValueError: too many values to unpack (expected 2)` | The supplied erosion algorithm is 2D only |
| Scalar classifier NaN / +Inf / -Inf | Assigned labels 7 / 7 / 1 | Nonfinite values are silently categorized; production preparation should reject or explicitly mask them |
| Preview auto-seed when target category absent | Raises `ValueError` | Preview can fail before histogram output even with edits disabled |

The default final-VTI growth-round heuristic (D:4449–4454), `ceil((target_radius-seed_radius)/min(spacing))`, does **not** achieve the requested Euclidean domain even when every donor is eligible:

| Spacing z/y/x (mm) | Seed → target radius (mm) | Rounds | Target sphere voxels | Achieved voxels | Domain fill |
|---|---:|---:|---:|---:|---:|
| 1 / 1 / 1 | 1 → 3 | 2 | 123 | 63 | 51.2% |
| 2 / 1 / 1 | 2 → 6 | 4 | 455 | 291 | 64.0% |
| 1 / 1 / 1 | 4 → 10 | 6 | 4,169 | 2,433 | 58.4% |

The six-neighbor growth metric and physical sphere radius have different semantics. Decide whether the configuration specifies iteration count or target physical geometry. For target geometry, grow to convergence inside the eligible connected domain or use a physical-distance/connected propagation rule, and audit requested-versus-achieved volume. Ellipsoid axis scales can increase the discrepancy further because the heuristic ignores them.

## Coordinate, I/O, and anatomy gaps

- 2D examples use `(y,x)` arrays and voxel/pixel radii, with no spacing. 3D examples use `(z,y,x)`. VTI output reverses XML `(x,y,z)` spacing/origin to `(z,y,x)` (D:4062–4064); plotting uses `i=y`, `j=x` (D:4682–4685). Standardize public schema coordinates and units, retain a documented conversion at I/O boundaries.
- VTI reader (D:3991–4064) supports a narrow fixture format: appended base64, assumed zlib blocks, scalar `CellData` named `activity` or the first CellData array, limited numeric types. It is not a general VTI adapter; it does not handle PointData, raw appended streams, uncompressed streams, multi-piece volumes, vector arrays, direction matrices, or full cell-center/extent metadata. Do not use this as the XCAT/raw ingestion layer.
- Native cell-center physical coordinates need origin + extent + half-cell offset and direction. Reader returns XML origin without accounting for cell center/nonzero extent. Preview downsampling adds `sample_offset = stride//2`; native index mapping exists (D:4597), but sampled physical origin is not updated (D:4362–4368). One authoritative affine/native-index map is required for cross-format ROI persistence.
- Sampling with stride is a preview approximation, not a topology-preserving resample. Small vessels can disappear and chosen coordinates move in sampled units. Store ROI in native physical/index coordinates and map it onto each preview resolution.
- Final VTI mode invents a spherical target seed label 8 inside a scalar band (D:4417–4427), with an interior-seed requirement. It does not identify/select the existing vessel's connected component and alter that original anatomy. Production needs selected label(s) plus ROI/component/centerline constraints rather than a demonstration seed.
- Bands are teaching categories, not validated tissue identities (D:4287–4310; E:345). Scalar values can be ambiguous across anatomy; the finite mapping must be versioned and case/source aware, with author-supplied category semantics. The classifier claims “exact zero” but uses `isclose(..., atol=1e-6)` (D:4109), so 5e-7 is also background. Expose actual interval closure and background tolerance explicitly.
- 3D erosion with physical-space nearest-neighbor or bounded geodesic/morphological propagation does not exist in these attachments. Specify eligible recipients, tie resolution, distance limits in mm, protected barriers, and unresolved behavior. Do not equate slice-by-slice 2D repair with a 3D algorithm.
- Resistance/force tables are arbitrary demonstration policy. No constitutive tissue mechanics, mass conservation, or CT intensity reconstruction is implemented. Record the policy in output provenance.
- Full-volume temporaries, copies, and repeated per-label scans in 3D will amplify memory use on XCAT cases. Restrict computation to ROI bounding boxes with an explicit halo, while preserving original full-volume storage. Crop-vs-full equivalence must be tested.
- Notebook summary text near E:1758 says only four direct neighbors, but the actual function at E:566 includes square-ring fallback. Use tested implementation as evidence, not the stale prose.

## Recommended extraction milestones / acceptance tests

1. **Configuration and preview contract:** schema version, input format/dtype/endian/shape/order/spacing/origin/direction, immutable source ID/checksum, finite label mapping with interval closure and background rule, operation list, target labels/component policy, ROI physical units, donor/recipient policies, deterministic seed, output root. Separate `validate`, `prepare`, `preview`, `generate`, `sweep`. Preview must work with absent target labels and without seeds. Emit histogram and axial/coronal/sagittal views showing native z/i/j and mm coordinates; preserve preview→native mapping in JSON.
2. **One validated morphology/reassignment core:** extract n-D donor helper and geometry functions; repair self-counting; reject wrong dtype/shape/target overflow; explicit protected override; reject malformed requests. Require exact no-change outside ROI/protected labels, exclusive labels, original-donor counts equal changed voxels, deterministic outputs and monotone target size for dilation/erosion when requests are accepted.
3. **3D erosion policy:** port majority/priority as one explicit option; add allowed recipients and physically bounded search that cannot jump across prohibited barriers. Test direct votes, ties, borders, empty masks, unknown labels, thick released shells, enclosed cavities, disconnected components, anisotropy and unresolved outcomes. Require requested = assigned + unresolved and target reduction = assigned count.
4. **Physical geometry validation:** test known spheres/cylinders and anisotropic grids against expected physical extent/volume; distinguish radius reduction, area stenosis, volume reduction and iteration count; plot and audit achieved versus requested geometry. Test barrier-limited underachievement as an expected, reported condition.
5. **Real-case pilot and export:** use a tiny crop and one actual XCAT case after metadata verification. Confirm axis orientation/dtype/value range; label mapping; ROI edit locality; reader/writer roundtrip; artifacts and metadata reproducibility; processing memory/time. Validate no source overwrite.
6. **Cohort/AI integration:** provenance ties source case→parameters→edited labels→original/modified scalar/reference volumes; case-level train/validation/test split before generating parameter variants to prevent leakage. Keep labels and AI-rendered intensity products separate; evaluate intensity recovery in changed regions and preservation outside edits before replacing current training data.
