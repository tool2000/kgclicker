from kg_gen.models import Graph
from kg_gen.utils.deduplicate import deduplicate_graph
from kg_gen.utils.llm_deduplicate import LLMDeduplicate


def dedup_cluster_graph(graph: Graph) -> Graph:

    # Deduplicate the graph using semantic hashing
    deduplicated_graph = deduplicate_graph(graph)

    # Deduplicate the semantic deduplicated graph using LLM
    llm_deduplicate = LLMDeduplicate(deduplicated_graph)
    llm_deduplicate.cluster()
    llm_deduplicated_graph = llm_deduplicate.deduplicate()
    return llm_deduplicated_graph
