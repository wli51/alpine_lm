from dspy_litl_agentic_system.tools.tool_cache.cache_config import (
    set_default_cache_root,
    set_cache_defaults
)

TOOL_CACHE_PATH = "/scratch/alpine/wli19@xsede.org/.tool_cache"

set_default_cache_root(TOOL_CACHE_PATH)
set_cache_defaults(
    size_limit_bytes=4 * 10**12,    # 4 TB
    expire=None
)
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
        query=DRUG_QUERY
    )
)

print(
    build_cell_context(
        cell_name=CELL_QUERY
    )
)
