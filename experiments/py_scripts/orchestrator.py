# orchestrator.py

from __future__ import annotations

from dataclasses import dataclass, asdict
import hashlib
import json
import tempfile
from pathlib import Path
from typing import Any, Optional

import pandas as pd
import dspy
import mlflow
from tqdm import tqdm

try:
    import yaml  # pyyaml
except Exception:
    yaml = None

from dspy_litl_agentic_system.tools.chembl_tools.context_inject import build_drug_context
from dspy_litl_agentic_system.tools.pubchem_tools.context_inject import build_pubchem_context
from dspy_litl_agentic_system.tools.cellosaurus_tools.context_inject import build_cell_context
from dspy_litl_agentic_system.tools.chembl_tools.for_agents import (
    search_chembl_id,
    get_compound_properties,
    get_compound_activities,
    get_drug_approval_status,
    get_drug_moa,
    get_drug_indications,
    search_target_id,
    get_target_activities_summary,
)
from dspy_litl_agentic_system.tools.pubchem_tools.for_agents import (
    search_pubchem_cid,
    get_properties,
    get_assay_summary,
    get_safety_summary,
    get_drug_summary,
    find_similar_compounds,
    compute_tanimoto,
)
from dspy_litl_agentic_system.tools.cellosaurus_tools.for_agents import (
    search_cellosaurus_ac,
    get_cellosaurus_summary,
)
from dspy_litl_agentic_system.agent.signatures import PredictIC50DrugCell


def _stable_salt_int(drug: str, cell: str) -> int:
    """
    Deterministic cross-platform salt derived from (drug, cell).
    Uses sha256 -> first 8 bytes -> uint64 -> bounded to signed 32-bit range.
    """
    key = f"{drug}||{cell}".encode("utf-8")
    h = hashlib.sha256(key).digest()
    salt_u64 = int.from_bytes(h[:8], byteorder="big", signed=False)
    return int(salt_u64 % (2**31 - 1))  # avoid negative / overflow in downstream libs


def _task_seed(base_seed: int, drug: str, cell: str) -> int:
    # keep within 32-bit signed-ish range
    return int((base_seed + _stable_salt_int(drug, cell)) % (2**31 - 1))


@dataclass
class Prediction:
    drug: str
    cell_line: str
    target: float
    predicted: Optional[float]
    base_seed: int
    salted_seed: int
    # Optional provenance fields you may want to tag/log
    model_id: Optional[str] = None
    session_id: Optional[str] = None
    error: Optional[str] = None


def orchestrate_toolless_pred(
    dataset: pd.DataFrame,
    lm_config: dict,
    drug_col: str = "name",
    cell_col: str = "ccle_name",
    target_col: str = "ic50",
    max_retries_per_task: int = 3,
    experimental_description: str = """
    The chemical-perturbation viability screen is conducted in a 8-step,
    4-fold dilution, starting from 10μM.
    """,
    output_unit: str = "uM",
    _n=None,  # subsetting for testing

    # NEW: reproducibility + logging controls
    base_seed: int = 0,
    enable_mlflow: bool = True,
    enable_dspy_autolog: bool = True,
    model_id: Optional[str] = None,
    session_id: Optional[str] = None,
    nested_runs: bool = True,
) -> list[dict]:
    """
    Orchestrate the memless prediction for IC50 values, with per-task salted seeding,
    per-task MLflow tagging, and JSONL/YAML artifact dumps.
    """

    if not all(col in dataset.columns for col in [drug_col, cell_col, target_col]):
        raise ValueError("Dataset must contain specified drug, cell, and target columns.")

    if _n is not None:
        dataset = dataset.head(_n)

    # Always disable caching so each prediction is fresh (matches your current behavior)
    lm_config = dict(lm_config)  # avoid mutating caller dict
    lm_config.update({"cache": False})

    # Build agent ONCE (cheap) – LM will be reconfigured per task to apply per-task seed.
    agent = dspy.ReAct(
        PredictIC50DrugCell,
        tools=[],
    )

    # Enable DSPy autolog once per parent run context (safe to call multiple times, but avoid spamming)
    if enable_mlflow and enable_dspy_autolog:
        try:
            mlflow.dspy.autolog()
        except Exception:
            # If mlflow.dspy isn't installed/available on some nodes, don't hard-fail orchestrator.
            pass

    pbar = tqdm(
        dataset[[drug_col, cell_col, target_col]].itertuples(index=False),
        total=len(dataset),
        desc="Predicting IC50 values",
    )

    results = []

    for drug, cell, target in pbar:
        # normalize cell naming as in your current code :contentReference[oaicite:2]{index=2}
        cell_parts = str(cell).split("_")
        if len(cell_parts) > 1:
            cell = cell_parts[0]

        salted_seed = _task_seed(base_seed=base_seed, drug=str(drug), cell=str(cell))

        # Configure DSPy LM PER TASK so the salted seed is actually applied.
        # This is the key change vs configuring once for the whole dataset. :contentReference[oaicite:3]{index=3}
        task_lm_config = dict(lm_config)
        task_lm_config["seed"] = salted_seed
        dspy.configure(lm=dspy.LM(**task_lm_config))

        # Optionally create a nested MLflow run per (drug, cell)
        # NOTE: if you run millions of pairs, nested_runs=True can explode run counts.
        # In that case, set nested_runs=False and rely on JSONL artifacts + tags on parent run.
        run_cm = None
        if enable_mlflow and mlflow.active_run() is not None and nested_runs:
            run_name = f"{drug}__{cell}"
            run_cm = mlflow.start_run(run_name=run_name, nested=True)

        if run_cm is not None:
            run_cm.__enter__()
            mlflow.set_tags(
                {
                    "drug": str(drug),
                    "cell_line": str(cell),
                    "base_seed": str(base_seed),
                    "salted_seed": str(salted_seed),
                    **({"model_id": str(model_id)} if model_id is not None else {}),
                    **({"session_id": str(session_id)} if session_id is not None else {}),
                }
            )

        retries = 0
        pred_obj = None

        try:
            while retries < max_retries_per_task:
                try:
                    result = agent(
                        drug=drug,
                        cell_line=cell,
                        experimental_description=experimental_description,
                        output_unit=output_unit,
                        tool_context="None specified.",
                        additional_bio_context="None specified.",
                    )

                    pred_obj = Prediction(
                        drug=str(drug),
                        cell_line=str(cell),
                        target=float(target),
                        predicted=getattr(result, "ic50_pred", None),
                        base_seed=int(base_seed),
                        salted_seed=int(salted_seed),
                        model_id=model_id,
                        session_id=session_id,
                    )
                    results.append(pred_obj)
                    break

                except Exception as e:
                    retries += 1
                    if retries >= max_retries_per_task:
                        pred_obj = Prediction(
                            drug=str(drug),
                            cell_line=str(cell),
                            target=float(target),
                            predicted=None,
                            base_seed=int(base_seed),
                            salted_seed=int(salted_seed),
                            model_id=model_id,
                            session_id=session_id,
                            error=f"prediction_error: {e}",
                        )
                        results.append(pred_obj)
                        break

        finally:
            if run_cm is not None:
                run_cm.__exit__(None, None, None)

    return results


def orchestrate_memless_pred(
    dataset: pd.DataFrame,
    lm_config: dict,
    drug_col: str = "name",
    cell_col: str = "ccle_name",
    target_col: str = "ic50",
    max_retries_per_task: int = 3,
    experimental_description: str = """
    The chemical-perturbation viability screen is conducted in a 8-step,
    4-fold dilution, starting from 10μM.
    """,
    output_unit: str = "uM",
    _n=None,  # subsetting for testing

    # NEW: reproducibility + logging controls
    base_seed: int = 0,
    enable_mlflow: bool = True,
    enable_dspy_autolog: bool = True,
    model_id: Optional[str] = None,
    session_id: Optional[str] = None,
    nested_runs: bool = True,
) -> list[dict]:
    """
    Orchestrate the memless prediction for IC50 values, with per-task salted seeding,
    per-task MLflow tagging, and JSONL/YAML artifact dumps.
    """

    if not all(col in dataset.columns for col in [drug_col, cell_col, target_col]):
        raise ValueError("Dataset must contain specified drug, cell, and target columns.")

    if _n is not None:
        dataset = dataset.head(_n)

    # Always disable caching so each prediction is fresh (matches your current behavior)
    lm_config = dict(lm_config)  # avoid mutating caller dict
    lm_config.update({"cache": False})

    # Build agent ONCE (cheap) – LM will be reconfigured per task to apply per-task seed.
    agent = dspy.ReAct(
        PredictIC50DrugCell,
        tools=[
            search_chembl_id,
            get_compound_properties,
            get_compound_activities,
            get_drug_approval_status,
            get_drug_moa,
            get_drug_indications,
            search_target_id,
            get_target_activities_summary,
            search_pubchem_cid,
            get_properties,
            get_assay_summary,
            get_safety_summary,
            get_drug_summary,
            find_similar_compounds,
            compute_tanimoto,
            search_cellosaurus_ac,
            get_cellosaurus_summary,
        ],
    )

    # Enable DSPy autolog once per parent run context (safe to call multiple times, but avoid spamming)
    if enable_mlflow and enable_dspy_autolog:
        try:
            mlflow.dspy.autolog()
        except Exception:
            # If mlflow.dspy isn't installed/available on some nodes, don't hard-fail orchestrator.
            pass

    pbar = tqdm(
        dataset[[drug_col, cell_col, target_col]].itertuples(index=False),
        total=len(dataset),
        desc="Predicting IC50 values",
    )

    results = []

    for drug, cell, target in pbar:
        # normalize cell naming as in your current code :contentReference[oaicite:2]{index=2}
        cell_parts = str(cell).split("_")
        if len(cell_parts) > 1:
            cell = cell_parts[0]

        salted_seed = _task_seed(base_seed=base_seed, drug=str(drug), cell=str(cell))

        # Configure DSPy LM PER TASK so the salted seed is actually applied.
        # This is the key change vs configuring once for the whole dataset. :contentReference[oaicite:3]{index=3}
        task_lm_config = dict(lm_config)
        task_lm_config["seed"] = salted_seed
        dspy.configure(lm=dspy.LM(**task_lm_config))

        # Context building (same as current, but keep error captured in Prediction)
        try:
            additional_bio_context = (
                build_drug_context(drug_name=drug)
                + "\n"
                + build_pubchem_context(query=drug)
                + "\n"
                + build_cell_context(cell_name=cell)
            )
            ctx_err = None
        except Exception as e:
            additional_bio_context = "None specified."
            ctx_err = f"context_build_error: {e}"

        # Optionally create a nested MLflow run per (drug, cell)
        # NOTE: if you run millions of pairs, nested_runs=True can explode run counts.
        # In that case, set nested_runs=False and rely on JSONL artifacts + tags on parent run.
        run_cm = None
        if enable_mlflow and mlflow.active_run() is not None and nested_runs:
            run_name = f"{drug}__{cell}"
            run_cm = mlflow.start_run(run_name=run_name, nested=True)

        if run_cm is not None:
            run_cm.__enter__()
            mlflow.set_tags(
                {
                    "drug": str(drug),
                    "cell_line": str(cell),
                    "base_seed": str(base_seed),
                    "salted_seed": str(salted_seed),
                    **({"model_id": str(model_id)} if model_id is not None else {}),
                    **({"session_id": str(session_id)} if session_id is not None else {}),
                }
            )

        retries = 0
        pred_obj = None

        try:
            while retries < max_retries_per_task:
                try:
                    result = agent(
                        drug=drug,
                        cell_line=cell,
                        experimental_description=experimental_description,
                        output_unit=output_unit,
                        tool_context="None specified.",
                        additional_bio_context=additional_bio_context,
                    )

                    pred_obj = Prediction(
                        drug=str(drug),
                        cell_line=str(cell),
                        target=float(target),
                        predicted=getattr(result, "ic50_pred", None),
                        base_seed=int(base_seed),
                        salted_seed=int(salted_seed),
                        model_id=model_id,
                        session_id=session_id,
                        error=ctx_err,
                    )
                    results.append(pred_obj)
                    break

                except Exception as e:
                    retries += 1
                    if retries >= max_retries_per_task:
                        pred_obj = Prediction(
                            drug=str(drug),
                            cell_line=str(cell),
                            target=float(target),
                            predicted=None,
                            base_seed=int(base_seed),
                            salted_seed=int(salted_seed),
                            model_id=model_id,
                            session_id=session_id,
                            error=(ctx_err + "\n" if ctx_err else "") + f"prediction_error: {e}",
                        )
                        results.append(pred_obj)
                        break

        finally:
            if run_cm is not None:
                run_cm.__exit__(None, None, None)

    return results
