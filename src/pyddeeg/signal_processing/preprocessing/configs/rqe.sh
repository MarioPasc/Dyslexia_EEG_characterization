#!/usr/bin/env bash
#SBATCH -J EEG_RQA_Ch_%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=30gb
#SBATCH --time=20:00:00
#SBATCH --constraint=amd
#SBATCH --error=rqa_ch_%a_%j.err
#SBATCH --output=rqa_ch_%a_%j.out
#SBATCH --array=0-30
#SBATCH --mail-type=BEGIN,END,FAIL
#SBATCH --mail-user=mpascual@uma.es

set -e
echo "Job started at $(date)"

# Map array index to channel name
CHANNELS=("Fp1" "Fp2" "F7" "F3" "Fz" "F4" "F8" "FC5" "FC1" "FC2" "FC6" "T7"
    "C3" "C4" "T8" "TP9" "CP5" "CP1" "CP2" "CP6" "TP10" "P7" "P3" "Pz"
"P4" "P8" "PO9" "O1" "Oz" "O2" "PO10" "Cz")

COMPUTE_RQE=false

# Get channel name for this array job
if [ "$SLURM_ARRAY_TASK_ID" -lt "${#CHANNELS[@]}" ]; then
    TARGET_CHANNEL=${CHANNELS[$SLURM_ARRAY_TASK_ID]}
else
    echo "Invalid array index: $SLURM_ARRAY_TASK_ID"
    exit 1
fi

# Reduced debug information - just essentials
echo "========================================="
echo "Processing channel: $TARGET_CHANNEL (index: $SLURM_ARRAY_TASK_ID)"
echo "Cores: $SLURM_CPUS_PER_TASK, Memory: $SLURM_MEM_PER_NODE"
echo "========================================="

# Load required modules
module load miniconda
source activate pyddeeg

# Create a temp directory in localscratch
MYLOCALSCRATCH="${LOCALSCRATCH}/${USER}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MYLOCALSCRATCH=$(echo "$MYLOCALSCRATCH" | sed 's|//|/|g')  # Remove double slashes
echo "Using local scratch: $MYLOCALSCRATCH"

# Clean up existing directory if it exists (safe for parallel array jobs as each has unique ID)
if [ -d "$MYLOCALSCRATCH" ]; then
    echo "Removing existing scratch directory"
    rm -rf --one-file-system "$MYLOCALSCRATCH"
fi

mkdir -p "$MYLOCALSCRATCH"

# Define paths
PROJ_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/Dyslexia_EEG_characterization
INPUT_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/EEG
OUTPUT_DIR=$MYLOCALSCRATCH/results
CONFIG_DIR=$MYLOCALSCRATCH/config
LOG_DIR=$MYLOCALSCRATCH/logs
STATUS_DIR=$MYLOCALSCRATCH/status

# Create directories
mkdir -p "$OUTPUT_DIR" "$CONFIG_DIR" "$LOG_DIR" "$STATUS_DIR"

# Copy necessary files to scratch (only what's needed)
echo "Copying project files to scratch..."
cp -r "$PROJ_DIR/src" "$MYLOCALSCRATCH/"
cp -r "$PROJ_DIR/pyproject.toml" "$MYLOCALSCRATCH/"

# Create config file for this channel
cat > "$CONFIG_DIR"/rqe_config.yaml << EOF
# Configuration for EEG RQA Processing on HPC - Channel: $TARGET_CHANNEL

# Dask configuration
dask:
  n_workers: 24
  threads_per_worker: 4
  memory_limit: "1.5GB"

# Directories
input_directory: "$INPUT_DIR"
output_directory: "$OUTPUT_DIR"

# Logging
logging:
  directory: "$LOG_DIR"
  level: "INFO"

# Channel to process
target_channel: "$TARGET_CHANNEL"

# Whether to compute RQE metrics (false = RQA metrics only)
return_rqe: ${COMPUTE_RQE}

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

# Go to working directory and set PYTHONPATH
cd "$MYLOCALSCRATCH" || { echo "Failed to change directory to $MYLOCALSCRATCH"; exit 1; }
export PYTHONPATH="$MYLOCALSCRATCH:$PYTHONPATH"

# Execute script with timing
echo "Starting EEG RQE Processing for channel $TARGET_CHANNEL at $(date)"

SCRIPT_PATH="$MYLOCALSCRATCH/src/pyddeeg/signal_processing/preprocessing/pipelines/rqe_preproc_picasso.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script file not found at $SCRIPT_PATH"
    find "$MYLOCALSCRATCH/src/pyddeeg" -type f -name "*.py" | sort | head -5
    exit 1
fi

# Run the processing with timing
time python "$SCRIPT_PATH" \
--config "$CONFIG_DIR/rqe_config.yaml" \
--cores "$SLURM_CPUS_PER_TASK" \
--channel "$TARGET_CHANNEL" \
--status-dir "$STATUS_DIR" \
--progress-interval 60  # Reduced progress output frequency

# Check script exit status
if ! mycmd; then
    echo "ERROR: Python script execution failed!"
    exit 1
fi

echo "Processing completed at $(date)"

# Copy results back with timestamp to avoid overwriting
RESULTS_DIR=/mnt/home/users/tic_163_uma/mpascual/execs/RQA/rqa_ch_${TARGET_CHANNEL}_$(date +%Y%m%d_%H%M%S)
echo "Copying results to $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
cp -rp "$OUTPUT_DIR"/* "$RESULTS_DIR"/ 2>/dev/null || echo "No output files"
cp -rp "$LOG_DIR"/* "$RESULTS_DIR"/ 2>/dev/null || echo "No log files"
cp -rp "$CONFIG_DIR"/* "$RESULTS_DIR"/ 2>/dev/null || echo "No config files"
cp -rp "$STATUS_DIR"/* "$RESULTS_DIR"/ 2>/dev/null || echo "No status files"

# Clean up scratch space
if [ -n "$MYLOCALSCRATCH" ] && [ -d "$MYLOCALSCRATCH" ]; then
    echo "Cleaning up scratch directory"
    rm -rf --one-file-system "$MYLOCALSCRATCH"
fi

echo "Job completed at $(date)"