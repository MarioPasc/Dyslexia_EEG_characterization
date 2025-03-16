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

echo "Job started at $(date)"

# Debug information
echo "SLURM_JOB_ID: $SLURM_JOB_ID"
echo "SLURM_CPUS_PER_TASK: $SLURM_CPUS_PER_TASK"
echo "SLURM_MEM_PER_NODE: $SLURM_MEM_PER_NODE"
echo "SLURM constraint: amd"

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

# Create a temp directory in localscratch - FIX: Added missing slash
MYLOCALSCRATCH=$LOCALSCRATCH/$USER/$SLURM_JOB_ID
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

# Create/modify config file for HPC usage
echo "Creating configuration file..."
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
echo "Changing to scratch directory..."
cd "$MYLOCALSCRATCH" || { echo "Failed to change directory to $MYLOCALSCRATCH"; exit 1; }

# Print monitoring instructions
echo "To monitor real-time progress, use:"
echo "  tail -f $LOG_DIR/rqe_processing_*.log"
echo ""
echo "To check current status:"
echo "  cat $STATUS_DIR/status.txt"
echo ""
echo "To see only compute progress:"
echo "  grep compute $STATUS_DIR/status.txt"
echo ""
echo "To check last error (if any):"
echo "  cat $STATUS_DIR/error.txt"

# Execute script with timing
echo "Starting EEG RQE Processing at $(date)"
echo "Executing Python script with config $CONFIG_DIR/rqe_config.yaml"

# Option 1: Use direct path execution
echo "Running script using direct path execution..."
time python "$MYLOCALSCRATCH/src/pyddeeg/signal_processing/preprocessing/pipelines/rqe_preproc_picasso.py" \
--config "$CONFIG_DIR/rqe_config.yaml" \
--cores "$SLURM_CPUS_PER_TASK" \
--status-dir "$STATUS_DIR" \
--progress-interval 30

# If Option 1 fails, try Option 2 with module syntax
if [ $? -ne 0 ]; then
    echo "Direct path execution failed, trying module syntax..."
    cd "$MYLOCALSCRATCH" || exit 1
    time python -m src.pyddeeg.signal_processing.preprocessing.pipelines.rqe_preproc_picasso \
    --config "$CONFIG_DIR/rqe_config.yaml" \
    --cores "$SLURM_CPUS_PER_TASK" \
    --status-dir "$STATUS_DIR" \
    --progress-interval 30
fi

# Check script exit status
if [ $? -ne 0 ]; then
    echo "Python script execution failed!"
    exit 1
fi

echo "Processing completed at $(date)"

# Copy results back to home directory
RESULTS_DIR=/mnt/home/users/tic_163_uma/mpascual/execs/RQE/rqe_run_$(date +%Y%m%d_%H%M%S)
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