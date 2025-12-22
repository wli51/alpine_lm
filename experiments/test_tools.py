from dspy_litl_agentic_system.tools.chembl_tools.context_inject import (
    build_drug_context,
)
from dspy_litl_agentic_system.tools.pubchem_tools.context_inject import (
    build_pubchem_context
)
from dspy_litl_agentic_system.tools.cellosaurus_tools.context_inject import (
    build_cell_context
)

CELL_QUERY = "U2-OS"
DRUG_QUERY = "barasertib-HQPA"

print(
    build_drug_context(
        drug_name=DRUG_QUERY
    )
)

print(
    build_pubchem_context(
        drug_name=DRUG_QUERY
    )
)

print(
    build_cell_context(
        cell_name=CELL_QUERY
    )
)
