#!/bin/bash
#SBATCH --partition=amc
#SBATCH --qos=normal
#SBATCH --time=02:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=10
#SBATCH --output=prefetch.out
#SBATCH --error=prefetch.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=weishan.2.li@cuanschutz.edu

# ---- environment setup ----
set -euo pipefail

# Initialize conda (important for non-interactive shells)
eval "$(conda shell.bash hook)"

# Activate environment
conda activate dspy-env

# ---- run prefetch ----
python /projects/wli19@xsede.org/alpine_lm/prep.prefetch_models.py
