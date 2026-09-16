# Verification scope and reproduction

These checks establish an integration starting point and reproduce defects in the
inherited implementations. They do not validate a completed cohort generator.

## Executed checks

- Both `python3 src/fakect.py --help` and `python3 src/fakenoise.py --help` completed.
- Python syntax/AST checks passed for the two imported source modules.
- Phantom VTI headless smoke command completed using the existing fixture:

  ```bash
  MPLCONFIGDIR=/tmp/fakect-mpl-cache python3 src/fakect.py \
    --in tests/activity_grid_000_001_005.vti --out /tmp/fakect-vti-smoke \
    --vti-max-dim 32 --no-show
  ```

  It produced a readable NPZ with 36 editable labels, including -893/-892/-891;
  scalar range -893 to 2263, sampled shape 31×19×24, and stored spacing/origin.
  This is a sampled fixture check, not proof of anatomical orientation or a
  production full-resolution roundtrip.
- The notebook audit reproduced exactly from the versioned reference copies:
  14 existing notebook test functions, 30 randomized donor-policy cases and
  30 morphology comparisons with SciPy passed. Adversarial evaluations exposed
  the documented defects; the full dilation export still has its original syntax
  error and is intentionally retained as text. No notebook top-level code ran.
- Training audit reproduced the 22,753-row/56-case manifest accounting, six stored
  model/metric records, historical log findings and 18 small morphology cases.
  It reads NRRD headers and Keras metadata, not image pixels or TensorFlow models.
  Memory values are analytical estimates, not newly measured process RSS.
- `bash -n` passed for the new GPU wrapper and inherited XCAT shell scripts.
  Eighteen mocked wrapper cases passed: quoting, script/submission/explicit roots,
  required arguments, equals syntax, rejection before launch, module failures and
  container exit statuses. Shell mocks replaced both `module` and `crun`.
- Notebook SHA-256 digests match the originals, and Slurm reference copies match
  their source files byte for byte.
- All 12 XCAT cases were evaluated with file-size/metadata checks and bounded
  memmap samples; see `XCAT_AUDIT.md` and `xcat-results.json` for the exact scope.
  The preview is a sampled discovery image, not a reviewed edit selection.
  Dictionary coverage failed: 41 sampled signed IDs are unmapped in 11 cases.
  Histograms and three slice planes were visually inspected. Metadata hashes
  were captured after sampling; multi-terabyte source binaries were not hashed.

## Reproduce from this branch checkout

The audited environment has NumPy, SciPy, scikit-image, pandas, pynrrd and Matplotlib.
The existing GUI CLI also needs its normal dependencies. No packages were installed.

```bash
python3 scripts/evaluation/audit_notebooks.py \
  --output outputs/evaluation/notebook-results.json
python3 scripts/evaluation/audit_training.py \
  --dataset-root /home/aghorban/repo/FakeCT/data/dataset \
  --slurm-dir /home/aghorban/slurm \
  --out-json outputs/evaluation/training-results.json
python3 scripts/evaluation/check_slurm_wrapper.py --repo . \
  --output outputs/evaluation/slurm-wrapper-results.json
MPLCONFIGDIR=/tmp/fakect-mpl-cache python3 scripts/evaluation/audit_xcat.py \
  --base /home/aghorban/slurm/xcat --output-prefix outputs/evaluation/xcat
```

The XCAT audit can take several minutes on shared storage. It stats every direct
case file and samples selected frames; it does not read every voxel. `--no-plots`
skips the orthogonal preview, which can trigger substantial page I/O on sparse
files despite small sampled arrays. `--cases 260602` limits a rerun to one case.
Script defaults never write to the XCAT or training source data directories.

## Not performed

No XCAT simulation, SLURM submission, GPU/container invocation, model training,
full-volume XCAT validation, real-case geometry edit, medical/clinical validation,
or dataset replacement was performed. Existing logs/metrics are historical evidence
and are explicitly separated from newly run bounded evaluations. The draft cohort
configuration is not executable by the inherited CLI; implementing it starts at M1.
