# FakeCT — student framework for synthetic anatomy and AI data

Review an ROI, apply controlled dilation/erosion, freeze paired images and labels,
then use them in a separate model experiment. The student entry point uses small
mathematical phantoms; XCAT CoA inputs are available as a research case study.

Open [the offline student guide](docs/student/index.html) in a browser for the
stepwise workflow, INI reference, four-panel ROI interface and development
checklist. Use [the testing guide](docs/student/testing.html) for the regression
suite and optional TensorFlow checks.

## Where students work

```text
FakeCT/
  configs/students/          Editable teaching templates
  configs/examples/          Small ROI/edit input examples
  configs/case-studies/coa/   Reviewed CoA phantom recipes and project index
  configs/tissues/           Tissue resistance/grouping policy
  scripts/                   Eight student command entry points
  src/                       Shared geometry, editing, data and model modules
  tests/                     Regression contracts and optional backend checks
  docs/student/              Offline walkthrough and testing guide
  docs/reference/            Detailed method guides
  references/atlas/          Versioned anatomical evidence and dictionaries
  requirements/              Core, development and optional backend profiles
  data/                      Small public geometry fixtures; local inputs ignored
  outputs/                   Generated reports, pairs and models (ignored)
  scratch/                   Temporary student experiments (ignored)
  scripts/research/           Advanced studies, audits, figures and cluster jobs
  archive/                   Research history and legacy scratch assets
```

Keep new reusable inputs under `configs/`, results under `outputs/`, and
temporary work under `scratch/`. Outputs and scratch files need separate backups;
Git ignore rules do not protect them from deletion. Shared `src/` imports remain
stable so students can extend the pipeline without chasing reorganized modules.

| Command in `scripts/` | Purpose |
| --- | --- |
| `student_demo.py` | Initialize public phantoms, editable inputs and a model INI. |
| `prepare_cohort.py` | Review ROI, preflight recipes, freeze and export pairs. |
| `preview_roi.py` | Review a standalone ROI/edit input. |
| `prepare_tissues.py` | Build an atlas-derived tissue catalog. |
| `resolve_anatomy.py` | Optionally refine selection using anatomical evidence. |
| `cohort_registry.py` | Publish and verify immutable paired datasets. |
| `model_experiment.py` | Plan, fit, test and summarize a separate model experiment. |
| `train_study.py` | Compatibility interface for earlier combined study inputs. |

Use `prepare_cohort.py` and `model_experiment.py` for new work. The
[reference index](docs/reference/index.html) explains advanced methods; the
[research archive](archive/research/README.html) preserves prior studies and
provenance. The [development checklist](docs/student/index.html#protection)
defines the contracts student changes must preserve.

## Quick start

Use the `student-framework.26.10.11` checkout supplied for the course and run from
the repository root. Python 3.10 or 3.11 is a practical starting environment.

After this branch is published by the instructor, clone it with:

```bash
git clone --branch student-framework.26.10.11 https://github.com/aghcv/FakeCT.git
cd FakeCT
```

```bash
python3 -m venv .venv
source .venv/bin/activate
# Windows PowerShell: py -3 -m venv .venv; .venv\Scripts\Activate.ps1
python -m pip install -r requirements/core.txt

python scripts/student_demo.py --output outputs/student-demo
python scripts/prepare_cohort.py --config outputs/student-demo/cases/demo01/cohort.ini --stage preview --preview-mode roi-selection
```

Open `report.html` in the INI's `[output] directory`. Every ordered tube point has
native i/j/k sections and a normal cross-section for radius review. Edit
`center_ijk` and `radius_mm`, choose a fresh preview output directory and rerun.
The report supports attenuation/tissue views, point navigation and exact
coordinate/radius references; parameter edits happen in the INI.

The initializer creates three mathematical cases. Use the same per-case
`cohort.ini` from ROI selection through preparation; `model.ini` is separate.
No XCAT installation, licensed phantom, GPU or pretrained model is required for
ROI review or pair preparation. These examples demonstrate software behavior,
not clinical validation.

## Prepare data separately from models

After accepting ROI placement, set `coordinate_reviewed = true`. Inspect all
native preflight outcomes before setting `parameters_reviewed = true` and freezing.

```bash
python scripts/prepare_cohort.py --config outputs/student-demo/cases/demo01/cohort.ini --validate-only
python scripts/prepare_cohort.py --config outputs/student-demo/cases/demo01/cohort.ini --stage preflight
# Review preflight.html and set parameters_reviewed = true.
python scripts/prepare_cohort.py --config outputs/student-demo/cases/demo01/cohort.ini --stage freeze
python scripts/prepare_cohort.py --config outputs/student-demo/cases/demo01/cohort.ini --stage prepare
```

Repeat for `demo02` and `demo03`, then follow the student guide to register their
checksum-pinned datasets and run `scripts/model_experiment.py` using `model.ini`.
Install `requirements/tensorflow.txt` when fitting/prediction is needed. The core
pipeline retains attenuation in `cm^-1`, signed anatomical labels, binary masks
and explicit native geometry. Siblings of one source anatomy stay in one model
split.

Keep frozen inputs unchanged and use fresh output directories for revisions.
Generated arrays, registries, checkpoints and reports stay outside Git and need
their own dataset backup.

## Develop and check changes

```bash
python -m pip install -r requirements/dev.txt
python -m pytest
```

The suite protects input contracts, physical ROI geometry, serial edits,
reassignment, safeguards, immutable releases and model split/data boundaries.
Backend-specific checks use optional environments; see
[testing.html](docs/student/testing.html) before interpreting skipped tests.

## Case-study and advanced references

- [XCAT CoA preparation and registry](docs/reference/MULTIPHANTOM_COHORTS.md)
- [Reviewed CoA input example](configs/case-studies/coa/phantoms/260602.ini)
- [Tissue mapping and atlas provenance](docs/reference/TISSUE_MAPPING.md)
- [Edit recipes](docs/reference/EDIT_RECIPES.md), [curvature](docs/reference/CURVATURE_EDITS.md),
  [selection/growth](docs/reference/SELECTION_GROWTH.md) and
  [erosion safeguards](docs/reference/EROSION_SAFEGUARDS.md)
- [Grouped model evaluation](docs/reference/MODEL_EVALUATION.md)
- [Legacy mesh/VTI CLI](src/fakect.py) and [cluster wrappers](scripts/research/slurm/README.md)
- [Research studies and preserved provenance](archive/research/README.html)

Historical research documents retain earlier milestones and local source paths.
Use the student guide for the current entry point. FakeCT code is
[MIT licensed](LICENSE); third-party phantom assets and pretrained weights retain
their respective terms.
