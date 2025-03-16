#!/usr/bin/env bash
#SBATCH -J EEG_RQE_Processing
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=128  # Using AMD node with 128 cores
#SBATCH --mem=400gb          # Using ~400GB RAM (out of 439GB available)
#SBATCH --time=72:00:00      # 72 hours should be enough for large datasets
#SBATCH --constraint=amd     # Selecting AMD nodes for high core count
#SBATCH --error=eeg_rqe_job.%J.err
#SBATCH --output=eeg_rqe_job.%J.out
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mpascual@uma.es

# Load required modules
module load miniconda
source activate pyddeeg


# Create a temp directory in localscratch
# shellcheck disable=SC2153
MYLOCALSCRATCH=$LOCALSCRATCH$USER/$SLURM_JOB_ID
mkdir -p "$MYLOCALSCRATCH"

# Define paths
PROJ_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/Dyslexia_EEG_characterization
INPUT_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/EEG
OUTPUT_DIR=$MYLOCALSCRATCH/results
CONFIG_DIR=$MYLOCALSCRATCH/config
LOG_DIR=$MYLOCALSCRATCH/logs

# Create directories
mkdir -p "$OUTPUT_DIR" "$CONFIG_DIR" "$LOG_DIR"

# Copy necessary files to scratch
cp -r "$PROJ_DIR/src" "$MYLOCALSCRATCH/"

# Create/modify config file for HPC usage
cat > "$CONFIG_DIR"/rqe_config.yaml << EOF
# Configuration for EEG RQE Processing on HPC

# Dask configuration
dask:
  n_workers: 32             # Using 32 workers (optimal for task distribution)
  threads_per_worker: 4     # 4 threads per worker (32*4=128 total cores)
  memory_limit: "12GB"      # ~12GB per worker (32*12=384GB total)

# Directories
input_directory: "$INPUT_DIR"
output_directory: "$OUTPUT_DIR"

# Logging
logging:
  directory: "$LOG_DIR"
  level: "INFO"

# Datasets to process
datasets:
  CT_UP: "CT_UP_preprocess_2.npz"
  DD_UP: "DD_UP_preprocess_2.npz"
  CT_DOWN: "CT_DOWN_preprocess_2.npz"
  DD_DOWN: "DD_DOWN_preprocess_2.npz"

# RQA/RQE parameters
rqa_parameters:
  embedding_dim: 10
  radius: 0.8
  time_delay: 1
  raw_signal_window_size: 100
  rqa_space_window_size: 25
  min_diagonal_line: 5
  min_vertical_line: 1
  min_white_vertical_line: 1
  stride: 7
  metrics_to_use:
    - "RR"
    - "DET"
    - "L_max"
    - "L_mean"
    - "ENT"
    - "LAM"
    - "TT"
    - "V_max"
    - "V_mean"
    - "V_ENT"
    - "W_max"
    - "W_mean"
    - "W_ENT"
    - "CLEAR"
    - "PERM_ENT"

# Whether to normalize metrics before RQE computation
normalize_metrics: true
EOF

# Go to working directory
cd "$MYLOCALSCRATCH" || exit

# Execute script with timing
echo "Starting EEG RQE Processing at $(date)"
echo "Executing Python script with config $CONFIG_DIR/rqe_config.yaml"

time python $PROJ_DIR/src/pyddeeg/signal_processing/preprocessing/rqe_preproc_picasso.py \
--config "$CONFIG_DIR"/hpc_config.yaml \
--cores "$SLURM_CPUS_PER_TASK"

echo "Processing completed at $(date)"

# Copy results back to home directory
RESULTS_DIR=/mnt/home/users/tic_163_uma/mpascual/execs/RQE/rqe_run_$(date +%Y%m%d_%H%M%S)
mkdir -p "$RESULTS_DIR"
cp -rp "$OUTPUT_DIR"/* "$RESULTS_DIR"/
cp -rp "$LOG_DIR"/* "$RESULTS_DIR"/
cp -rp "$CONFIG_DIR"/* "$RESULTS_DIR"/

echo "Results copied to $RESULTS_DIR"

# Clean up scratch space
if cd "$LOCALSCRATCH"/"$USER"; then
    if [ -n "$MYLOCALSCRATCH" ]; then
        rm -rf --one-file-system "$MYLOCALSCRATCH"
        echo "Local scratch directory cleaned"
    fi
fi

echo "Job completed successfully"