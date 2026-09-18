# Diagnostic erosion and category overlay: recipe-v10

[Open the report](../../../outputs/studies/thoracic-aorta/recipe-v10/report.html).
The [current input](../../../configs/studies/thoracic-aorta-curvature.ini)
sets `assign_surrounding_tissue=false` only on `arch_inner_narrow`.
All ROI, distance, iteration, profile, direction and stiffness parameters match
recipe-v9. The source catalog and atlas remain unchanged.

| Measurement | recipe-v9 | recipe-v10 |
|---|---:|---:|
| Dilation additions | 1,710 | 1,710 |
| Erosion proposals after target stiffness | 35 | 35 |
| Accepted erosion releases | 0 | 35 |
| Unresolved releases retained as artery | 35 | 0 |
| Released diagnostic markers | 0 | 35 |

The 35 new markers exactly equal v9's unresolved erosion mask. Every other
final label matches v9, and the entire final attenuation array matches v9:
the released positions retain their input values as explicitly unassigned
placeholders. Their native before/after close-up focuses at i,j,k=378,373,1404.

The observed face-neighbor shell contains 44 `chest_surface` voxels and 68
remaining `dias_aorta` voxels. The table exposes the existing coarse soft-tissue
classification and fine skin resistance for `chest_surface`; diagnostic release
does not decide whether that material mapping is correct.

The registered overlay now has controls for ten present categories: released,
soft tissue, bone, cartilage, muscle, artery, vein, lung, nervous tissue and
unknown. Released starts visible at 80% and uses a native surface. Other
categories start hidden; each supports 15%, 45% and 80% opacity independently.
Selected before/after surfaces remain native. Context meshes use native or
stride-3 occupancy as recorded in the metadata.

Verification: 327 unit/integration tests passed. All 43 output-manifest hashes
and 21 generating-code hashes were verified, with conserved profile volumes,
unchanged edit proposals and correct label/attenuation masks. Five JavaScript
control checks ran in V8 with DOM/Plotly mocks, including hidden-state and
camera preservation. Native close-up/profile PNGs were visually inspected;
no browser WebGL pixel test was performed.

Detailed counts and checks are saved in
[diagnostic-erosion-evaluation.json](diagnostic-erosion-evaluation.json).
See [diagnostic erosion controls](../../DIAGNOSTIC_EROSION.md) for mask semantics
and rerun instructions.
