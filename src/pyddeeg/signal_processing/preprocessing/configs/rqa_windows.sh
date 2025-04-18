#!/usr/bin/env bash
#SBATCH -J EEG_RQA_Win_%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32
#SBATCH --mem=16gb
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

TARGET_BANDWIDTH=4 # GAMMA

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
cat > "$CONFIG_DIR"/rqa_windows_config.yaml << EOF
# Configuration for EEG RQA Processing on HPC - Channel: $TARGET_CHANNEL

# Dask parallel processing configuration
dask:
  use_dask: true
  n_workers: 16  # Number of worker processes to spawn
  threads_per_worker: 2  # Number of threads per worker
  memory_limit: "8GB"  # Memory limit per worker
  
# Directories
input_directory: "$INPUT_DIR"
output_directory: "$OUTPUT_DIR"

# Datasets to process
datasets:
  CT_UP: "CT_UP_preprocess_20.npz"
  DD_UP: "DD_UP_preprocess_20.npz"
  CT_DOWN: "CT_DOWN_preprocess_20.npz"
  DD_DOWN: "DD_DOWN_preprocess_20.npz"

# Logging
logging:
  directory: "$LOG_DIR"
  filename: "rqa_windows.log"
  level: "INFO"

# Processing parameters
target_channel: "$TARGET_CHANNEL"
target_bandwidth: "$TARGET_BANDWIDTH" 
window_sizes: [50, 100, 150, 200]

# RQA/RQE parameters
rqa_parameters:
  embedding_dim: 10
  radius: 0.8
  time_delay: 1
  min_diagonal_line: 5
  min_vertical_line: 1
  min_white_vertical_line: 1
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
EOF

# Go to working directory and set PYTHONPATH
cd "$MYLOCALSCRATCH" || { echo "Failed to change directory to $MYLOCALSCRATCH"; exit 1; }
export PYTHONPATH="$MYLOCALSCRATCH:$PYTHONPATH"

# Execute script with timing
echo "Starting EEG RQE Processing for channel $TARGET_CHANNEL at $(date)"

SCRIPT_PATH="$MYLOCALSCRATCH/src/pyddeeg/signal_processing/preprocessing/pipelines/rqa_windows_picasso.py"
if [ ! -f "$SCRIPT_PATH" ]; then
    echo "ERROR: Script file not found at $SCRIPT_PATH"
    find "$MYLOCALSCRATCH/src/pyddeeg" -type f -name "*.py" | sort | head -5
    exit 1
fi

# Run the processing with timing
time python "$SCRIPT_PATH" \
--config "$CONFIG_DIR/rqa_windows_config.yaml" \

# Check script exit status
if [ $? -ne 0 ]; then
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

echo "Job completed at $(date)"
