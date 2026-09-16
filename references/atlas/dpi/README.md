# Versioned DPI Anatomy Atlases

The files in this directory are control documents for interpreting anatomy
already classified by DPI's native XCAT source adapter. They are ordinary CSV
files so that they:

- open directly in Excel;
- remain readable in Git diffs;
- can be reviewed and approved like a spreadsheet;
- travel with every derived DPI dataset.

`cardiovascular_atlas_v1.csv` is the released 1.0 interpretation.
`cardiovascular_atlas_v1_1.csv` is the 1.1 interpretation used by the
blood-flow model; it adds explicit endocardium, myocardium, epicardium, and
papillary-muscle roles. Both use schema `dpi.anatomy_atlas.v1`. Each row
defines either the top-level block selected by `vtkExtractBlock` or a model
role assigned to leaves beneath that block.

## Versioning policy

1. Do not modify a released atlas in place when a change alters interpretation.
2. Copy it to a new filename and increment `atlas_version`.
3. Keep all rows in one file on the same schema and version.
4. Review `required=true` roles before enabling `--strict-atlas`.
5. Supply `config/atlas/anatomy_atlas.csv` with `--source-atlas`.

The extraction manifest stores SHA-256 hashes and copies both documents under
the output `provenance/` directory. The DPI control atlas describes downstream
model roles; the XCAT source atlas records how raw XCAT labels were assigned
to the anatomical hierarchy.
