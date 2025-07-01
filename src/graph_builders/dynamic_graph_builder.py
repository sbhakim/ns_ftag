# src/graph_builders/dynamic_graph_builder.py


import torch
import networkx as nx
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

class DynamicAttackGraphBuilder:
    def __init__(self, config: Any):
        self.config = config
        self.attack_patterns = {}  # Placeholder for learned patterns (e.g., from Phase 2)

    def build_attack_graph(self, node_entities: List[Any], predictions: Dict[str, Any], attention_map: np.ndarray) -> nx.DiGraph:
        """
        Build an attack graph (NetworkX DiGraph) from neural predictions and attention weights for a single sequence/graph.
        """
        G = nx.DiGraph()

        # Add nodes with their predicted properties
        num_nodes = len(node_entities)  # Number of nodes in the sequence

        for i in range(num_nodes):
            # Extract per-node predictions
            node_attack_prob = predictions['attack_presence'][i, 1]  # Probability of attack (class 1)
            node_severity = predictions['severity'][i, 0]  # Severity score for node i
            node_is_attack_step = node_attack_prob > 0.5

            G.add_node(
                i,
                entity_id=node_entities[i] if i < len(node_entities) else f"node_{i}",
                attack_prob=node_attack_prob,
                severity=node_severity,
                is_attack_step=node_is_attack_step
            )

        # Add edges based on attention weights
        threshold = self.config.attention_threshold
        edges_indices = (torch.tensor(attention_map) > threshold).nonzero(as_tuple=False)

        for edge in edges_indices:
            src, dst = edge[0].item(), edge[1].item()
            weight = attention_map[src, dst].item()
            G.add_edge(src, dst, weight=weight, attention=weight)

        return G

    def extract_attack_paths(self, graph: nx.DiGraph) -> List[List[Any]]:
        """
        Extract potential attack paths from the constructed graph.
        Consider limiting path length or using specific algorithms for large graphs.
        """
        attack_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_attack_step', False)]
        paths = []

        for start_node in attack_nodes:
            for end_node in attack_nodes:
                if start_node != end_node:
                    try:
                        for path in nx.all_simple_paths(graph, source=start_node, target=end_node):
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue

        unique_paths = list(set(tuple(p) for p in paths))
        return [list(p) for p in unique_paths]
