from dataclasses import dataclass

import pandas as pd
import dspy
from tqdm import tqdm

from dspy_litl_agentic_system.tools.chembl_tools.for_agents import (
    search_chembl_id,
    get_compound_properties,
    get_compound_activities,
    get_drug_approval_status,
    get_drug_moa,
    get_drug_indications,
    search_target_id,
    get_target_activities_summary
)
from dspy_litl_agentic_system.tools.pubchem_tools.for_agents import (
    search_pubchem_cid,
    get_properties,
    get_assay_summary,
    get_safety_summary,
    get_drug_summary,
    find_similar_compounds
)
from dspy_litl_agentic_system.agent.signatures import (
    PredictIC50DrugCell,
)


@dataclass
class Prediction:
    drug: str
    cell_line: str
    target: float
    predicted: float


def orchestrate_memless_pred(
    dataset: pd.DataFrame,
    lm_config: dict,
    drug_col: str = 'name',
    cell_col: str = 'ccle_name',
    target_col: str = 'ic50',
    max_retries_per_task: int = 3,
    experimental_description: str = """
    The chemical-perturbation viability screen is conducted in a 8-step, 
    4-fold dilution, starting from 10μM.
    """,
    output_unit: str = "uM",
    _n=None # subsetting for testing
):
    """
    Orchestrate the memless prediction for IC50 values.
    """
    
    if not all(col in dataset.columns for col in [drug_col, cell_col, target_col]):
        raise ValueError("Dataset must contain specified drug, cell, and target columns.")
    
    if _n is not None:
        dataset = dataset.head(_n)
    
    # disable caching irrespective of user input so that each prediction is fresh
    lm_config.update({"cache": False})
    dspy.configure(
        lm=dspy.LM(**lm_config)
    )

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
            find_similar_compounds
        ]
    )

    results: list[dict] = []
    pbar = tqdm(
        dataset[[drug_col, cell_col, target_col]].itertuples(index=False),
        total=len(dataset), desc="Predicting IC50 values")
    for drug, cell, target in pbar:
        
        retries = 0

        while retries < max_retries_per_task:
            
            try:

                result = agent(
                    drug=drug,
                    cell_line=cell,
                    experimental_description=experimental_description,
                    output_unit=output_unit,
                    tool_context="None specified.",
                    additional_bio_context="None specified."
                )
                results.append(
                    Prediction(
                        drug=drug,
                        cell_line=cell,
                        target=target,
                        predicted=result.ic50_pred
                    ).__dict__
                )
                break  # Exit retry loop on success
            
            except Exception as e:
                
                retries += 1
                print(f"Error during prediction: {e}. Retry {retries}/{max_retries_per_task}")

    return results
