# Wahab training entry points

`reference/` contains byte-for-byte copies of the existing scripts in
`/home/aghorban/slurm` taken on 2026-09-16. The GPU script is user-confirmed;
the CPU scripts use different, older execution paths and are historical references,
not a validated CPU recipe for this branch.

`fakenoise_train_gpu.sh` retains the confirmed GPU resources, TensorFlow 2.17
container, environment prefix, and context defaults. It accepts dataset/output
paths so the integration checkout can be evaluated without overwriting old runs.
Create the log directory **before** submission. From this branch checkout:

```bash
mkdir -p logs
sbatch --export=ALL,FAKECT_ROOT="$PWD" scripts/slurm/fakenoise_train_gpu.sh \
  --csv /home/aghorban/repo/FakeCT/data/dataset/paired_datasets/pairs.csv \
  --out-dir /scratch/CHOOSE_A_NEW_RUN_DIRECTORY
```

The checked-in wrapper has been syntax-checked and command-tested with mocks;
a new GPU training job has not been submitted. The container and existing
`~/envs/fakect` prefix must already be available. Do not install the unpinned
`tensorflow` requirement over the cluster-provided runtime. Override
`FAKECT_ENV_PREFIX` only when using a separately prepared prefix.

The imported trainer comes from the newer local `stenosis` commit and writes
context-specific output filenames. Its known data/splitting/axis issues are listed
in the integration evaluation and must be resolved before cohort-scale training.
