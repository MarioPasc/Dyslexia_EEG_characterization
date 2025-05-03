#!/usr/bin/env bash
#SBATCH -J EEG_RQA_Reorg_%j
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=8GB
#SBATCH --time=02:00:00
#SBATCH --constraint=amd
#SBATCH --error=reorg_%j.err
#SBATCH --output=reorg_%j.out
#SBATCH --mail-type=END,FAIL
#SBATCH --mail-user=<you@domain>

set -euo pipefail
echo "Re-organisation job $SLURM_JOB_ID started $(date)"

module load miniconda
source activate pyddeeg

STIM=2
CONF=/path/to/pipeline_config.yaml

# The re-org stage does not look at the electrode, but the CLI still
# requires a channel token.  Use any valid name, e.g. "Cz".
python preprocessing.py \
--config "$CONF" \
--stim   "$STIM" \
--channel Cz \
--do-reorg

echo "Re-organisation finished $(date)"
