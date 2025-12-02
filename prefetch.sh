set -euo pipefail

eval "$(conda shell.bash hook)"
conda activate dspy-env

python prep.prefetch_models.py
