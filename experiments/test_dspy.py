from pathlib import Path
import os
import pathlib
import time
import uuid

import pandas as pd
import mlflow

from py_scripts.orchestrator import orchestrate_memless_pred, orchestrate_toolless_pred
from py_scripts.utils import deterministic_seeds


def get_env(name: str, default=None, required: bool = False):
    """Small helper to read env vars with optional default + required flag."""
    value = os.environ.get(name, default)
    if required and value is None:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value

DATA_PATH = Path(
    "/projects/wli19@xsede.org/PRISM_dataset/"
).resolve()

MASTER_SEED = 42

if not DATA_PATH.exists() or not DATA_PATH.is_dir():
    raise FileNotFoundError(
        f"The specified path does not exist or is not a directory: {DATA_PATH}"
    )

tissues = list(DATA_PATH.iterdir())


valid_tissues = []
valid_cells = []
for tissue in tissues:
    if not tissue.is_dir():
        print(f"Skipping non-directory item: {tissue}")
        continue
    valid_tissues.append(tissue)

    cells = list(tissue.iterdir())
    if not cells:
        print(f"No cell line files found in tissue directory: {tissue}")
        continue

    for file in cells:
        if file.suffix == ".csv":
            valid_cells.append(file)


if not valid_tissues:
    raise ValueError("No valid tissue directories found.")
if not valid_cells:
    raise ValueError("No valid cell line CSV files found.")


print(f"Total valid tissues found: {len(valid_tissues)}")
print(f"Total valid cells found: {len(valid_cells)}")

MODEL_ID = 'unsloth/Llama-3.2-3B-Instruct'
API_BASE = "http://127.0.0.1:8000/v1"
API_KEY = "local"
LM_MAX_TOKENS = 4096
LM_TEMPERATURE = 1.0
LM_CONFIG = {
    # DSPy expects "provider/model_name" here, where provider="openai"
    "model": f"openai/{MODEL_ID}",
    "api_base": API_BASE,
    "api_key": API_KEY,
    "temperature": LM_TEMPERATURE,
    "max_tokens": LM_MAX_TOKENS,
}


print("LM_CONFIG (from environment):")
for k, v in LM_CONFIG.items():
    print(f"  {k}: {v}")


sample_data = pd.read_csv(valid_cells[0])
print("Sample data head:")
print(sample_data.head())


TRACKING_URI = "/scratch/alpine/wli19@xsede.org/mlruns/"
EXPERIMENT_NAME = "test_dspy_scratch"
mlflow.set_tracking_uri(pathlib.Path(TRACKING_URI).resolve().as_uri())
try:
    exp_id = mlflow.create_experiment(EXPERIMENT_NAME)
    print(f"Created new MLflow experiment {EXPERIMENT_NAME} with ID: {exp_id}")
except Exception as e:
    if all(keyword in str(e).lower() for keyword in ["experiment", "already", "exists"]):
        exp = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
        exp_id = exp.experiment_id if exp else None
        print(f"MLflow experiment {EXPERIMENT_NAME} already exists with ID: {exp_id}")
    else:
        print(f"Error creating MLflow experiment: {e}")

seed = 42
cell_path = valid_cells[0]
session_id = str(uuid.uuid4())
with mlflow.start_run(run_name=f"{cell_path.stem}_{session_id}") as parent_run:

    completion_mark_file = cell_path.parent / f"{cell_path.stem}_{str(seed)}.complete"
    if completion_mark_file.exists():
        print(f"Skipping already completed run for cell {cell_path.stem} with seed {seed}")
        exit(0)

    mlflow.set_tags(
        {
            "session_id": session_id,
            "group": "testing",
            "mem": "less",
            "task": "provisional_prediction",
            "base_seed": str(seed),   # rename for clarity
            "model_id": MODEL_ID,
            "cell": cell_path.stem,
            "toolset": True,
            "memory_aug": False

        }
    )

    mlflow.dspy.autolog()
    mlflow.log_params(LM_CONFIG | {"base_seed": seed})

    _ = orchestrate_toolless_pred(
        dataset=pd.read_csv(cell_path),
        lm_config={**LM_CONFIG},
        drug_col="name",
        cell_col="ccle_name",
        target_col="ic50",
        base_seed=seed,
        enable_mlflow=True,
        enable_dspy_autolog=True,
        model_id=MODEL_ID,
        session_id=session_id,
        nested_runs=True,
    )

    completion_mark_file.touch()
    print(f"Marked completion of run for cell {cell_path.stem} with seed {seed}")
