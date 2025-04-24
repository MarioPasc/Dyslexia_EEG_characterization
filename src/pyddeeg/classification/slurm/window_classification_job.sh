#!/usr/bin/env bash
#
# === Job description =======================================================
#  One array-job per EEG electrode → hyper-parameter tuning  +  outer-CV
#  Everything is run from $LOCALSCRATCH to avoid hammering the shared FS.
# ===========================================================================

#SBATCH -J EEG_RQA_win          # job-name root
#SBATCH --ntasks=1              # a single MPI task
#SBATCH --cpus-per-task=48      # 48 logical cores (see comments below)
#SBATCH --mem=32gb
#SBATCH --time=20:00:00
#SBATCH --constraint=amd        # fast Zen-based nodes with large localscratch
#SBATCH --array=0-30            # 31 electrodes
#SBATCH --error=rqa_%A_%a.err
#SBATCH --output=rqa_%A_%a.out

set -euo pipefail
echo "Job  ${SLURM_ARRAY_JOB_ID}-${SLURM_ARRAY_TASK_ID} started at $(date)"

# --------------------------- electrode lookup ------------------------------
CHANNELS=( "Fp1" "Fp2" "F7" "F3" "Fz" "F4" "F8" "FC5" "FC1" "FC2" "FC6" "T7"
    "C3"  "C4"  "T8" "TP9" "CP5" "CP1" "CP2" "CP6" "TP10" "P7" "P3" "Pz"
"P4" "P8" "PO9" "O1" "Oz" "O2" "PO10" "Cz" )
ELEC=${CHANNELS[$SLURM_ARRAY_TASK_ID]}
echo "→ Electrode selected: ${ELEC}"
PROJ_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/Dyslexia_EEG_characterization

# --------------------------- localscratch set-up ---------------------------
MY_LSCR=${LOCALSCRATCH}/${USER}/${SLURM_ARRAY_JOB_ID}/${SLURM_ARRAY_TASK_ID}
mkdir -p "${MY_LSCR}"
echo "Local scratch  : ${MY_LSCR}"

# ---- copy “slow-changing” inputs that are tiny ----------------------------
cp  settings.yaml                     "${MY_LSCR}/"
cp  model_config.yaml                 "${MY_LSCR}/"
# (If your dataset is *huge* keep it on the shared FS; the pipeline only *reads*
#  it once.  If it’s < few-GB you can also rsync it here.)

cd "${MY_LSCR}"

# --------------------------- software environment --------------------------
module load miniconda
source activate pyddeeg

# --------------------------- run the pipeline ------------------------------
SCRIPT_PATH="${PROJ_DIR}"/src/pyddeeg/classification/tests/tune_hyperparams_electrode.py
python $SCRIPT_PATH             \
--settings  settings.yaml          \
--electrode "${ELEC}"              \
--threads   "${SLURM_CPUS_PER_TASK}"   # your script should honour this
echo "Pipeline finished at $(date)"

# --------------------------- persist results -------------------------------
RUN_DEST=$HOME/EEG_runs/${ELEC}/${SLURM_ARRAY_JOB_ID}_${SLURM_ARRAY_TASK_ID}
mkdir -p "${RUN_DEST}"
rsync -a --info=progress2  "${MY_LSCR}/"  "${RUN_DEST}/"

# --------------------------- clean-up --------------------------------------
cd "${LOCALSCRATCH}/${USER}" || exit  # guard against rm -rf /
rm -rf --one-file-system "${MY_LSCR}"

echo "All done at $(date)"
