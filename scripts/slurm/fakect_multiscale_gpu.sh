#!/bin/bash
#SBATCH -p gpu
#SBATCH --gres=gpu:1
#SBATCH --job-name=coa_multiscale
#SBATCH --output=logs/coa_multiscale.%j.out
#SBATCH --error=logs/coa_multiscale.%j.err
#SBATCH --time=04:00:00
#SBATCH --mem=16G
#SBATCH --cpus-per-task=4

set -euo pipefail
fakect_input=${1:?Pass the frozen pilot INI}
fakect_checkout=${FAKECT_ROOT:-${SLURM_SUBMIT_DIR:-$PWD}}
cd -- "$fakect_checkout"
module load container_env tensorflow-gpu/2.17
export PYTHONUNBUFFERED=1
export TF_NUM_INTRAOP_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export TF_NUM_INTEROP_THREADS=1
export OMP_NUM_THREADS="${SLURM_CPUS_PER_TASK:-4}"
export MPLCONFIGDIR="/tmp/fakect-multiscale-mpl-${SLURM_JOB_ID:-local}"
mkdir -p "$MPLCONFIGDIR"
crun -p "$HOME/envs/fakect" python - "$fakect_input" <<'PY'
import sys
import tempfile
from pathlib import Path
import numpy as np
sys.path.insert(0, str(Path.cwd()/'src'))
import fakect_multiscale_experiment as experiment
from fakect_multiscale_learning import configure_tensorflow
config = experiment.load_config(sys.argv[1])
tf, network = configure_tensorflow(config)
if not tf.config.list_physical_devices('GPU'):
    raise SystemExit('No GPU visible in the TensorFlow container')
overview = np.ones((1,16,16,16,2),np.float32)*.2
locator = network.compile_locator(network.build_locator((16,16,16),2))
locator.train_on_batch({'overview':overview}, np.array([[.5,.5,.5,.2,.2,.2]],np.float32))
classifier = network.compile_classifier(network.build_classifier((16,16,16),(16,16,16),
    base_filters=2,locator=locator))
inputs = {'overview':overview,'proposal_boxes':np.array([[[.5,.5,.5,.5,.5,.5]]*2],np.float32),
          'detail_0':overview,'detail_1':overview}
classifier.train_on_batch(inputs,{'coa_logit':np.array([[1]],np.float32),
                                 'local_volume_loss':np.array([[.3]],np.float32)})
before = classifier(inputs,training=False)
with tempfile.TemporaryDirectory(prefix='fakect-multiscale-gpu-') as directory:
    path = Path(directory)/'smoke.keras'
    classifier.save(path)
    after = network.load_model(path)(inputs,training=False)
    for key in before:
        np.testing.assert_allclose(before[key],after[key],rtol=1e-5,atol=1e-6)
print('GPU smoke passed:',tf.__version__,tf.config.list_logical_devices('GPU'),flush=True)
PY
crun -p "$HOME/envs/fakect" python scripts/train_multiscale_coa.py --config "$fakect_input" --stage all
