# Developing FakeCT

Use the [developer conventions](docs/developer/contributing.html) for extension
boundaries, function contracts, array order, units and immutable dataset handling.
The [workflow guide](docs/developer/index.html), [API reference](docs/developer/api.html)
and [testing guide](docs/developer/testing.html) are readable offline.

Keep reusable algorithms in `src/`, command dispatch in `scripts/`, editable
inputs in `configs/`, and generated artifacts in ignored `outputs/` or `scratch/`.
The CoA case study demonstrates the general pipeline; it does not define a
restriction on target anatomy or model architecture.

Every maintained function starts with a NumPy-style docstring describing its
purpose, parameters and return value. Include physical units, axis order,
mutation, file writes and meaningful failure conditions when relevant. The
maintained scope is declared in `configs/development/api.json`; add new workflow
modules to that scope. Editor defaults are provided in `.editorconfig`.

Install the development dependencies and check a change from the repository root:

```bash
python -m pip install -r requirements/dev.txt
python scripts/dev/build_api_docs.py
python scripts/dev/check_quality.py
python -m pytest
```

Use generated analytic fixtures for regression tests. Run the affected optional
backend or browser checks when changing those interfaces; their environments are
described in the testing guide. Report exact test results and skipped checks in
the change description.

Preserve original signed labels, native geometry and source bytes. Review and
freeze a new cohort revision when inputs or implementation hashes change;
registered datasets remain immutable. Keep siblings from the same source
anatomy together across model splits. Do not modify historical receipts or
snapshots to make changed code appear identical to an earlier release.
