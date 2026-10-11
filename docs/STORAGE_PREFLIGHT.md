# Storage assessment before larger-context aorta segmentation

No deletion is needed before the next development step. No datasets, models,
reports or source files were removed during this audit.

The filesystem containing the integration checkout and original XCAT volumes
reports 10 TiB total and approximately 5.5 TiB available (the human-readable
`df -h` display rounds this to 5.6T). Free inodes are abundant. `quota -s` and
`quota -s -u aghorban` did not return a usage/limit record, so this is filesystem
availability, not independent confirmation of an additional per-user quota.
The timestamped byte counts and estimates are recorded in
[the machine-readable audit](../outputs/storage/preflight-2026-09-28.json).

| Location | Allocated space reported by `du` |
| --- | ---: |
| Original `/home/aghorban/slurm/xcat/` inputs, including surfaces and frames | About 429 GiB |
| Main checkout `data/` | About 9.8 GiB |
| Integration checkout `outputs/`, total | About 3.9 GiB |
| Of that total: cohort outputs | About 2.1 GiB |
| Of that total: study/recipe previews | About 1.0 GiB |
| Of that total: experiments, including saved predictions and reports | About 569 MiB |

Allocated space can differ from logical file sizes on this filesystem. The
above rows for output categories overlap the total and must not be added to it.
These values are a point-in-time assessment, not a reservation of capacity.

## Dependencies to preserve

The 192-sample dataset manifest under
`outputs/datasets/coa/two-stage-radius115-r1/` references the prepared source
and sibling NPZs in `outputs/cohorts/coa/radius115/<case>/pairs-r2/`; case
260617 uses `pairs-r4/`. The adapter directory alone is not a self-contained
dataset. Keep the referenced arrays, frozen metadata and hashes together.

Also retain the atlas catalog, anatomical selection inventories, original ACT
and ATN files, the completed `two-stage-r2` experiment, and the current broader
atlas-mask review. Some files under `outputs/` are inputs to other stages.

Old visualization meshes are possible future regeneration candidates. Failed
training attempts have no useful large checkpoints to reclaim; existing smoke
models are small. Removing those would save little and discard diagnostic
history. Historical preflight metadata also records manual adjustments and
should remain available. No cleanup candidate needs to be removed now.

## Storage design for the next implementation

Existing source geometry gives these **uncompressed float32 estimates**:

| Representation | Size |
| --- | ---: |
| One full attenuation volume for each of twelve cases | 39.948 GiB |
| A separate full attenuation volume for each of 192 siblings | 639.173 GiB |
| Separate full ACT and ATN volumes for every sibling | 1,278.346 GiB |
| Existing local attenuation crops for all 192 siblings | 1.447 GiB |

The next loader should read blocks from the existing whole-body ATN volume and
overlay the sibling's edited native crop. It must apply the same per-pixel to
inverse-centimetre conversion to base and edited values, preserve physical
coordinates and source identities, and leave the original volume immutable.
The full-body context is then available without copying the complete body for
every sibling. Atlas-derived targets remain separate compact masks.

Any downsampled context caches should have explicit byte budgets and source/
configuration keys. Predictions and rendered meshes should be generated only
for selected evaluations and review cases. Before a large export, estimate its
upper-bound bytes and check available space again. This avoids relying on the
current free-space snapshot as a permanent guarantee.
