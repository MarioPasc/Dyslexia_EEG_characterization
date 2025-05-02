#!/usr/bin/env bash
#SBATCH -J EEG_RQA_Win_%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=128gb
#SBATCH --time=20:00:00
#SBATCH --constraint=amd
#SBATCH --error=rqa_win_%a_%j.err
#SBATCH --output=rqa_win_%a_%j.out
#SBATCH --array=0-30

set -e
echo "Job started at $(date)"

# Map array index to channel name
CHANNELS=("Fp1" "Fp2" "F7" "F3" "Fz" "F4" "F8" "FC5" "FC1" "FC2" "FC6" "T7"
    "C3" "C4" "T8" "TP9" "CP5" "CP1" "CP2" "CP6" "TP10" "P7" "P3" "Pz"
"P4" "P8" "PO9" "O1" "Oz" "O2" "PO10" "Cz")

module load miniconda
source activate pyddeeg


CHANNEL=${CHANNELS[$SLURM_ARRAY_TASK_ID]}

python preprocessing.py \
--config /path/to/pipeline_config.yaml \
--stim 2 \
--channel "$CHANNEL" \
--do-rqa           # zerolag is auto-skipped if done
