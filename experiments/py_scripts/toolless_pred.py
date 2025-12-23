#!/usr/bin/env python
# coding: utf-8

from pathlib import Path
import os
import pathlib
import time
import uuid

import pandas as pd
import mlflow

from orchestrator import orchestrate_toolless_pred
from utils import deterministic_seeds
from dspy_litl_agentic_system.tools.tool_cache.cache_config import (
    set_default_cache_root, 
    set_cache_defaults
)


def get_env(name: str, default=None, required: bool = False):
    """Small helper to read env vars with optional default + required flag."""
    value = os.environ.get(name, default)
    if required and value is None:
        raise RuntimeError(f"Required environment variable {name} is not set")
    return value


# =========
# Config
# =========

# Allow these to be overridden by env, but keep your current values as defaults.
DATA_PATH = Path(
    get_env("PRISM_DATA_PATH", "/PATH/TO/PROCESSED/PRISM/DATA/")
).resolve()

N_REPLICATES = int(get_env("N_REPLICATES", "10"))
MASTER_SEED = int(get_env("MASTER_SEED", "42"))
TRACKING_URI = get_env("MLFLOW_TRACKING_URI", "/MLFLOW/TRACKING/URI/")
EXPERIMENT_NAME = get_env("MLFLOW_EXPERIMENT_NAME", "EXPERIMENT_NAME")
TOOL_CACHE_ROOT = get_env("TOOL_CACHE_ROOT", ".")
if TOOL_CACHE_ROOT:
    cache_root = Path(TOOL_CACHE_ROOT).expanduser().resolve()
    cache_root.mkdir(parents=True, exist_ok=True)

    set_default_cache_root(cache_root)
    set_cache_defaults(
        size_limit_bytes=2 * 10**12,    # 2 TB
        expire=None
    )
    
    print(f"Configured cache root: {cache_root}")

# =========
# Data validation and reading
# =========

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

# =========
# LM config (fully driven by env)
# =========

# MODEL_ID should match the served-model-name you used in vLLM
# e.g. "unsloth/gpt-oss-safeguard-20b-BF16"
MODEL_ID = get_env("MODEL_ID", required=True)
API_BASE = get_env("OPENAI_BASE_URL", "http://127.0.0.1:8000/v1")
API_KEY = get_env("OPENAI_API_KEY", "local")
LM_MAX_TOKENS = int(get_env("LM_MAX_TOKENS", "4096"))
LM_TEMPERATURE = float(get_env("LM_TEMPERATURE", "1.0"))

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

# =========
# Sample data sanity check
# =========

sample_data = pd.read_csv(valid_cells[0])
print("Sample data head:")
print(sample_data.head())

# =========
# Config tracking and run replicates
# =========

print(f"Setting MLflow tracking URI to: {TRACKING_URI}")
mlflow.set_tracking_uri(pathlib.Path(TRACKING_URI).resolve().as_uri())
print(f"MLflow tracking URI set to: {mlflow.get_tracking_uri()}")

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

seeds = deterministic_seeds(MASTER_SEED, n=N_REPLICATES)[:N_REPLICATES]

# run as many as possible within time limit
TIME_LIMIT_SECONDS = int(os.environ.get("RUN_TIME_LIMIT_SECONDS", str(3 * 3600 + 45 * 60)))
start_time = time.time()
print(f"Soft run time limit set to {TIME_LIMIT_SECONDS} seconds (~{TIME_LIMIT_SECONDS/3600:.2f} hours)")

for cell_path in valid_cells:
    
    elapsed = time.time() - start_time
    
    if elapsed > TIME_LIMIT_SECONDS:
        print(f"Reached time limit of {TIME_LIMIT_SECONDS} seconds. Stopping further runs.")
        break
    else:
        print(f"Elapsed time: {elapsed:.2f} seconds. Continuing run on cell: {cell_path.name}")

    for seed in seeds:

        completion_mark_file = cell_path.parent / f"{cell_path.stem}_{str(seed)}.complete"
        if completion_mark_file.exists():
            print(f"Skipping already completed run for cell {cell_path.stem} with seed {seed}")
            continue

        session_id = str(uuid.uuid4())
        with mlflow.start_run(run_name=f"{cell_path.stem}_{session_id}") as parent_run:
            
            mlflow.set_tags(
                {
                    "session_id": session_id,
                    "group": "experimenting",
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

        # Mark completion
        completion_mark_file.touch()
        print(f"Marked completion of run for cell {cell_path.stem} with seed {seed}")
