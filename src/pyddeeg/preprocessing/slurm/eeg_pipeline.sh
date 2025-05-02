#!/usr/bin/env bash
#SBATCH -J EEG_RQA_Array_%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128GB
#SBATCH --time=20:00:00
#SBATCH --constraint=amd
#SBATCH --error=rqa_%A_%a.err
#SBATCH --output=rqa_%A_%a.out
#SBATCH --array=0-30           # 31 electrodes
#SBATCH --mail-type=END,FAIL   # optional
#SBATCH --mail-user=<you@domain>

set -euo pipefail
echo "Job $SLURM_JOB_ID | task $SLURM_ARRAY_TASK_ID started $(date)"

# ---------------- Channel look-up ----------------
CHANNELS=(Fp1 Fp2 F7 F3 Fz F4 F8 FC5 FC1 FC2 FC6 T7 \
    C3 C4 T8 TP9 CP5 CP1 CP2 CP6 TP10 P7 P3 Pz \
P4 P8 PO9 O1 Oz O2 PO10 Cz)
CHANNEL=${CHANNELS[$SLURM_ARRAY_TASK_ID]}

# --------------- Software environment -----------
module load miniconda
source activate pyddeeg     # <- your conda env

# --------------- Flags --------------------------
PATH_ROOT=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/Dyslexia_EEG_characterization
STIM=2                     # stimulus to process
CONF=$PATH_ROOT/src/pyddeeg/preprocessing/cfgs/pipeline_config.yaml
SCRIPT=$PATH_ROOT/src/pyddeeg/preprocessing/pipelines/preprocessing.py
# run zero-lag only once (task 0) – others will skip because the
# sentinel *.npz already exists
ZL_FLAG=""
if [[ "$SLURM_ARRAY_TASK_ID" -eq 0 ]]; then
    ZL_FLAG="--do-zerolag"
fi

python $SCRIPT \
--config  "$CONF" \
--stim    "$STIM" \
--channel "$CHANNEL" \
$ZL_FLAG \
--do-rqa

echo "Task $SLURM_ARRAY_TASK_ID finished $(date)"
