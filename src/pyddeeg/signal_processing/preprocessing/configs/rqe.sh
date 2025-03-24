#!/usr/bin/env bash
#SBATCH -J EEG_RQA_Ch_%j              # Updated job name to reflect RQA focus
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=32            # Reduced from 64 to 32 cores
#SBATCH --mem=40gb                    # Reduced from 128GB to 40GB
#SBATCH --time=10:00:00               # Reduced from 24h to 8h
#SBATCH --constraint=amd              # Keep AMD nodes for high core count
#SBATCH --error=rqa_ch_%a_%j.err      # Updated output filename
#SBATCH --output=rqa_ch_%a_%j.out     # Updated output filename
#SBATCH --array=0-30                  # Keep processing all channels except Cz
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

# Debug information
echo "========================================="
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_ARRAY_TASK_ID: $SLURM_ARRAY_TASK_ID"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "Processing channel: $TARGET_CHANNEL (index: $SLURM_ARRAY_TASK_ID)"
echo "SLURM constraint: amd"
echo "========================================="

# Load required modules
echo "Loading miniconda..."
module load miniconda
echo "Activating conda environment..."
source activate pyddeeg

# Print Python information for debugging
which python
python --version
echo "Python path:"
python -c "import sys; print(sys.path)"
echo "Checking for required modules..."
python -c "import numpy; print('NumPy found, version:', numpy.__version__)"
python -c "import dask; print('Dask found, version:', dask.__version__)"

echo "========================================="

# Create a temp directory in localscratch
MYLOCALSCRATCH="${LOCALSCRATCH}/${USER}/${SLURM_JOB_ID}_${SLURM_ARRAY_TASK_ID}"
MYLOCALSCRATCH=$(echo "$MYLOCALSCRATCH" | sed 's|//|/|g')  # Remove double slashes
echo "Creating local scratch directory: $MYLOCALSCRATCH"
mkdir -p "$MYLOCALSCRATCH"

# Define paths
PROJ_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/repos/Dyslexia_EEG_characterization
INPUT_DIR=/mnt/home/users/tic_163_uma/mpascual/fscratch/datasets/EEG
OUTPUT_DIR=$MYLOCALSCRATCH/results
CONFIG_DIR=$MYLOCALSCRATCH/config
LOG_DIR=$MYLOCALSCRATCH/logs
STATUS_DIR=$MYLOCALSCRATCH/status

# Create directories
echo "Creating output directories..."
mkdir -p "$OUTPUT_DIR" "$CONFIG_DIR" "$LOG_DIR" "$STATUS_DIR"

# Copy necessary files to scratch
echo "Copying project files to scratch..."
cp -r "$PROJ_DIR/src" "$MYLOCALSCRATCH/"
cp -r "$PROJ_DIR/pyproject.toml" "$MYLOCALSCRATCH/" # Copy setup files

# Create/modify config file for this channel
echo "Creating configuration file for channel $TARGET_CHANNEL..."
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
return_rqe: "$COMPUTE_RQE"

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
echo "Changing to scratch directory..."
cd "$MYLOCALSCRATCH" || { echo "Failed to change directory to $MYLOCALSCRATCH"; exit 1; }

# Set PYTHONPATH to find modules
echo "Setting PYTHONPATH..."
export PYTHONPATH="$MYLOCALSCRATCH:$PYTHONPATH"
echo "PYTHONPATH: $PYTHONPATH"

# Print monitoring instructions
echo "To monitor real-time progress, use:"
echo "  tail -f $LOG_DIR/rqe_processing_${TARGET_CHANNEL,,}*.log"
echo ""
echo "To check current status:"
echo "  cat $STATUS_DIR/status.txt"

# Execute script with timing
echo "Starting EEG RQE Processing for channel $TARGET_CHANNEL at $(date)"
echo "Executing Python script with config $CONFIG_DIR/rqe_config.yaml"

SCRIPT_PATH="$MYLOCALSCRATCH/src/pyddeeg/signal_processing/preprocessing/pipelines/rqe_preproc_picasso.py"
echo "Checking if script exists at: $SCRIPT_PATH"
if [ -f "$SCRIPT_PATH" ]; then
    echo "Script exists."
else
    echo "ERROR: Script file not found at $SCRIPT_PATH"
    # List directory contents to debug
    echo "Directory contents:"
    find "$MYLOCALSCRATCH/src/pyddeeg" -type f -name "*.py" | sort
    exit 1
fi

# Option 1: Use direct path execution
echo "Running script using direct path execution..."
time python "$SCRIPT_PATH" \
--config "$CONFIG_DIR/rqe_config.yaml" \
--cores "$SLURM_CPUS_PER_TASK" \
--channel "$TARGET_CHANNEL" \
--status-dir "$STATUS_DIR" \
--progress-interval 30

# Check script exit status
if [ $? -ne 0 ]; then
    echo "ERROR: Python script execution failed!"
    exit 1
fi

echo "Processing completed at $(date)"

# Copy results back to home directory
RESULTS_DIR=/mnt/home/users/tic_163_uma/mpascual/execs/RQA/rqa_ch_${TARGET_CHANNEL}_$(date +%Y%m%d_%H%M%S)
echo "Copying results to $RESULTS_DIR"
mkdir -p "$RESULTS_DIR"
cp -rp "$OUTPUT_DIR"/* "$RESULTS_DIR"/ || echo "Warning: No output files to copy"
cp -rp "$LOG_DIR"/* "$RESULTS_DIR"/ || echo "Warning: No log files to copy"
cp -rp "$CONFIG_DIR"/* "$RESULTS_DIR"/ || echo "Warning: No config files to copy"
cp -rp "$STATUS_DIR"/* "$RESULTS_DIR"/ || echo "Warning: No status files to copy"

echo "Results copied to $RESULTS_DIR"

# Clean up scratch space
if cd "$LOCALSCRATCH/$USER"; then
    if [ -n "$MYLOCALSCRATCH" ] && [ -d "$MYLOCALSCRATCH" ]; then
        echo "Cleaning up scratch directory: $MYLOCALSCRATCH"
        rm -rf --one-file-system "$MYLOCALSCRATCH"
        echo "Local scratch directory cleaned"
    else
        echo "Warning: MYLOCALSCRATCH not found or not a directory"
    fi
else
    echo "Warning: Could not change to LOCALSCRATCH/$USER directory"
fi

echo "Job completed at $(date)"