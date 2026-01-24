"""
Graph analysis module for computing various graph metrics and analytics.
Uses NetworkX for graph algorithms.
"""

from typing import Optional
import logging

import networkx as nx
from kg_gen.models import Graph

logger = logging.getLogger(__name__)


def graph_to_networkx(graph: Graph) -> nx.DiGraph:
    """
    Convert a KGGen Graph to a NetworkX DiGraph.

    Args:
        graph: KGGen Graph object

    Returns:
        NetworkX DiGraph
    """
    G = nx.DiGraph()

    # Add nodes
    for entity in graph.entities:
        G.add_node(entity)

    # Add edges with relation labels
    for subject, predicate, obj in graph.relations:
        G.add_edge(subject, obj, relation=predicate, label=predicate)

    return G


def compute_basic_stats(graph: Graph) -> dict:
    """
    Compute basic graph statistics.

    Args:
        graph: KGGen Graph object

    Returns:
        Dictionary of basic statistics
    """
    G = graph_to_networkx(graph)

    num_nodes = G.number_of_nodes()
    num_edges = G.number_of_edges()

    # Compute density
    if num_nodes > 1:
        density = num_edges / (num_nodes * (num_nodes - 1))
    else:
        density = 0

    # Compute average degree
    if num_nodes > 0:
        avg_degree = sum(dict(G.degree()).values()) / num_nodes
    else:
        avg_degree = 0

    # Find isolated nodes (no connections)
    isolated = list(nx.isolates(G))

    # Count unique relation types
    relation_types = set()
    for _, _, data in G.edges(data=True):
        if "relation" in data:
            relation_types.add(data["relation"])

    return {
        "num_entities": num_nodes,
        "num_relations": num_edges,
        "num_relation_types": len(relation_types),
        "density": round(density, 4),
        "average_degree": round(avg_degree, 2),
        "num_isolated_entities": len(isolated),
        "isolated_entities": isolated[:20],  # Limit to first 20
    }


def compute_centrality(graph: Graph, top_k: int = 20) -> dict:
    """
    Compute various centrality metrics for graph nodes.

    Args:
        graph: KGGen Graph object
        top_k: Number of top nodes to return for each metric

    Returns:
        Dictionary containing centrality metrics
    """
    G = graph_to_networkx(graph)

    if G.number_of_nodes() == 0:
        return {
            "degree_centrality": [],
            "pagerank": [],
            "betweenness_centrality": [],
            "closeness_centrality": [],
            "in_degree_centrality": [],
            "out_degree_centrality": [],
        }

    results = {}

    # Degree centrality
    degree_cent = nx.degree_centrality(G)
    results["degree_centrality"] = _top_k_items(degree_cent, top_k)

    # In-degree centrality (importance as target)
    in_degree_cent = nx.in_degree_centrality(G)
    results["in_degree_centrality"] = _top_k_items(in_degree_cent, top_k)

    # Out-degree centrality (importance as source)
    out_degree_cent = nx.out_degree_centrality(G)
    results["out_degree_centrality"] = _top_k_items(out_degree_cent, top_k)

    # PageRank
    try:
        pagerank = nx.pagerank(G, max_iter=100)
        results["pagerank"] = _top_k_items(pagerank, top_k)
    except nx.PowerIterationFailedConvergence:
        logger.warning("PageRank did not converge, using default values")
        results["pagerank"] = []

    # Betweenness centrality (bridge nodes)
    try:
        betweenness = nx.betweenness_centrality(G)
        results["betweenness_centrality"] = _top_k_items(betweenness, top_k)
    except Exception as e:
        logger.warning(f"Betweenness centrality failed: {e}")
        results["betweenness_centrality"] = []

    # Closeness centrality
    try:
        closeness = nx.closeness_centrality(G)
        results["closeness_centrality"] = _top_k_items(closeness, top_k)
    except Exception as e:
        logger.warning(f"Closeness centrality failed: {e}")
        results["closeness_centrality"] = []

    return results


def _top_k_items(scores: dict, k: int) -> list[dict]:
    """Get top k items from a score dictionary."""
    sorted_items = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [
        {"entity": entity, "score": round(score, 4)}
        for entity, score in sorted_items[:k]
    ]


def detect_communities(graph: Graph, method: str = "louvain") -> dict:
    """
    Detect communities/clusters in the graph.

    Args:
        graph: KGGen Graph object
        method: Community detection method ('louvain', 'label_propagation', 'greedy_modularity')

    Returns:
        Dictionary containing community information
    """
    G = graph_to_networkx(graph)

    if G.number_of_nodes() == 0:
        return {
            "num_communities": 0,
            "communities": [],
            "modularity": 0,
        }

    # Convert to undirected for community detection
    G_undirected = G.to_undirected()

    try:
        if method == "louvain":
            # Louvain method (requires python-louvain or networkx >= 3.3)
            try:
                communities = nx.community.louvain_communities(G_undirected)
            except AttributeError:
                # Fallback for older networkx
                communities = list(nx.community.greedy_modularity_communities(G_undirected))

        elif method == "label_propagation":
            communities = list(nx.community.label_propagation_communities(G_undirected))

        elif method == "greedy_modularity":
            communities = list(nx.community.greedy_modularity_communities(G_undirected))

        else:
            logger.warning(f"Unknown method '{method}', using louvain")
            communities = nx.community.louvain_communities(G_undirected)

    except Exception as e:
        logger.warning(f"Community detection failed: {e}")
        return {
            "num_communities": 0,
            "communities": [],
            "modularity": 0,
            "error": str(e),
        }

    # Calculate modularity
    try:
        modularity = nx.community.modularity(G_undirected, communities)
    except Exception:
        modularity = 0

    # Format communities
    community_list = []
    for i, comm in enumerate(communities):
        comm_list = list(comm)
        community_list.append({
            "id": i,
            "size": len(comm_list),
            "members": comm_list[:100],  # Limit members shown
            "sample_members": comm_list[:10],  # Show first 10 as sample
        })

    # Sort by size descending
    community_list.sort(key=lambda x: x["size"], reverse=True)

    return {
        "num_communities": len(communities),
        "communities": community_list,
        "modularity": round(modularity, 4),
        "method": method,
    }


def find_shortest_path(
    graph: Graph,
    source: str,
    target: str,
) -> dict:
    """
    Find the shortest path between two entities.

    Args:
        graph: KGGen Graph object
        source: Source entity name
        target: Target entity name

    Returns:
        Dictionary containing path information
    """
    G = graph_to_networkx(graph)

    if source not in G.nodes():
        return {"error": f"Source entity '{source}' not found in graph"}

    if target not in G.nodes():
        return {"error": f"Target entity '{target}' not found in graph"}

    try:
        # Try directed path first
        path = nx.shortest_path(G, source, target)
        path_length = len(path) - 1

        # Get relations along the path
        path_with_relations = []
        for i in range(len(path) - 1):
            edge_data = G.get_edge_data(path[i], path[i + 1])
            relation = edge_data.get("relation", "related_to") if edge_data else "related_to"
            path_with_relations.append({
                "from": path[i],
                "relation": relation,
                "to": path[i + 1],
            })

        return {
            "source": source,
            "target": target,
            "path": path,
            "path_length": path_length,
            "path_with_relations": path_with_relations,
            "directed": True,
        }

    except nx.NetworkXNoPath:
        # Try undirected
        try:
            G_undirected = G.to_undirected()
            path = nx.shortest_path(G_undirected, source, target)
            path_length = len(path) - 1

            return {
                "source": source,
                "target": target,
                "path": path,
                "path_length": path_length,
                "directed": False,
                "note": "No directed path found; showing undirected path",
            }
        except nx.NetworkXNoPath:
            return {
                "source": source,
                "target": target,
                "error": "No path exists between these entities",
            }


def find_all_paths(
    graph: Graph,
    source: str,
    target: str,
    max_length: int = 5,
    max_paths: int = 10,
) -> dict:
    """
    Find all paths between two entities up to a maximum length.

    Args:
        graph: KGGen Graph object
        source: Source entity name
        target: Target entity name
        max_length: Maximum path length to consider
        max_paths: Maximum number of paths to return

    Returns:
        Dictionary containing all found paths
    """
    G = graph_to_networkx(graph)

    if source not in G.nodes():
        return {"error": f"Source entity '{source}' not found in graph"}

    if target not in G.nodes():
        return {"error": f"Target entity '{target}' not found in graph"}

    try:
        all_paths = list(nx.all_simple_paths(G, source, target, cutoff=max_length))
        all_paths.sort(key=len)
        all_paths = all_paths[:max_paths]

        paths_with_relations = []
        for path in all_paths:
            path_relations = []
            for i in range(len(path) - 1):
                edge_data = G.get_edge_data(path[i], path[i + 1])
                relation = edge_data.get("relation", "related_to") if edge_data else "related_to"
                path_relations.append({
                    "from": path[i],
                    "relation": relation,
                    "to": path[i + 1],
                })
            paths_with_relations.append({
                "path": path,
                "length": len(path) - 1,
                "relations": path_relations,
            })

        return {
            "source": source,
            "target": target,
            "num_paths": len(paths_with_relations),
            "paths": paths_with_relations,
            "max_length_searched": max_length,
        }

    except Exception as e:
        return {"error": str(e)}


def get_entity_neighbors(
    graph: Graph,
    entity: str,
    depth: int = 1,
) -> dict:
    """
    Get neighbors of an entity up to a certain depth.

    Args:
        graph: KGGen Graph object
        entity: Entity name to get neighbors for
        depth: How many hops to explore

    Returns:
        Dictionary containing neighbor information
    """
    G = graph_to_networkx(graph)

    if entity not in G.nodes():
        return {"error": f"Entity '{entity}' not found in graph"}

    # Get outgoing neighbors
    outgoing = []
    for neighbor in G.successors(entity):
        edge_data = G.get_edge_data(entity, neighbor)
        relation = edge_data.get("relation", "related_to") if edge_data else "related_to"
        outgoing.append({
            "entity": neighbor,
            "relation": relation,
        })

    # Get incoming neighbors
    incoming = []
    for neighbor in G.predecessors(entity):
        edge_data = G.get_edge_data(neighbor, entity)
        relation = edge_data.get("relation", "related_to") if edge_data else "related_to"
        incoming.append({
            "entity": neighbor,
            "relation": relation,
        })

    # Get extended neighborhood if depth > 1
    extended = []
    if depth > 1:
        visited = {entity}
        current_level = set(G.successors(entity)) | set(G.predecessors(entity))
        visited.update(current_level)

        for _ in range(depth - 1):
            next_level = set()
            for node in current_level:
                for neighbor in G.successors(node):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
                for neighbor in G.predecessors(node):
                    if neighbor not in visited:
                        next_level.add(neighbor)
                        visited.add(neighbor)
            current_level = next_level

        extended = list(current_level)[:50]  # Limit extended neighbors

    return {
        "entity": entity,
        "outgoing": outgoing,
        "incoming": incoming,
        "total_connections": len(outgoing) + len(incoming),
        "extended_neighbors": extended if depth > 1 else None,
        "depth": depth,
    }


def compute_connected_components(graph: Graph) -> dict:
    """
    Find connected components in the graph.

    Args:
        graph: KGGen Graph object

    Returns:
        Dictionary containing component information
    """
    G = graph_to_networkx(graph)
    G_undirected = G.to_undirected()

    components = list(nx.connected_components(G_undirected))
    components.sort(key=len, reverse=True)

    component_list = []
    for i, comp in enumerate(components):
        comp_list = list(comp)
        component_list.append({
            "id": i,
            "size": len(comp_list),
            "members": comp_list[:100],  # Limit members shown
            "sample_members": comp_list[:10],
        })

    return {
        "num_components": len(components),
        "components": component_list,
        "is_connected": len(components) == 1,
        "largest_component_size": len(components[0]) if components else 0,
    }


def compute_full_analysis(graph: Graph, top_k: int = 20) -> dict:
    """
    Compute a comprehensive analysis of the graph.

    Args:
        graph: KGGen Graph object
        top_k: Number of top nodes to return for rankings

    Returns:
        Dictionary containing all analysis results
    """
    return {
        "basic_stats": compute_basic_stats(graph),
        "centrality": compute_centrality(graph, top_k),
        "communities": detect_communities(graph),
        "components": compute_connected_components(graph),
    }
