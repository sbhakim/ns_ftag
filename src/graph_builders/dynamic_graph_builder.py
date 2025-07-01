# src/graph_builders/dynamic_graph_builder.py


import torch
import networkx as nx
import numpy as np
import torch.nn.functional as F
from typing import Dict, List, Tuple, Any

# Import the new modular components
from .graph_node_processor import GraphNodeProcessor
from .graph_edge_constructor import GraphEdgeConstructor # Will be created later

class DynamicAttackGraphBuilder:
    def __init__(self, config: Any):
        self.config = config
        self.attack_patterns = {}  # Placeholder for learned patterns (e.g., from Phase 2)

        # Initialize the modular processors/constructors
        self.node_processor = GraphNodeProcessor(config)
        self.edge_constructor = GraphEdgeConstructor(config) # Uncommented as graph_edge_constructor.py is now provided

    def build_attack_graph(self, node_entities: List[Any], predictions: Dict[str, Any], attention_map: np.ndarray, temporal_info: List[Any]) -> nx.DiGraph:
        """
        Build an attack graph (NetworkX DiGraph) from neural predictions and attention weights for a single sequence/graph.
        This method orchestrates node and edge construction using dedicated sub-components.
        """
        G = nx.DiGraph()

        # 1. Process and add nodes with enhanced attributes
        node_attributes_list = self.node_processor.process_nodes(node_entities, predictions, temporal_info)
        for i, attrs in enumerate(node_attributes_list):
            G.add_node(i, **attrs)

        # 2. Construct and validate edges using the dedicated constructor
        # Replace the original basic edge construction logic with calls to the new edge constructor
        edges_to_add = self.edge_constructor.construct_edges(G, attention_map, temporal_info, predictions)
        for src, dst, edge_attrs in edges_to_add:
            G.add_edge(src, dst, **edge_attrs)

        return G

    def extract_attack_paths(self, graph: nx.DiGraph) -> List[List[Any]]:
        """
        Extract potential attack paths from the constructed graph.
        Consider limiting path length or using specific algorithms for large graphs.
        This method will be enhanced later for advanced ranking.
        """
        attack_nodes = [n for n, d in graph.nodes(data=True) if d.get('is_attack_step', False)]
        paths = []

        # Define a default cutoff for simple path extraction to prevent performance issues
        # This can be made configurable via self.config if needed.
        path_cutoff = self.config.path_extraction_cutoff if hasattr(self.config, 'path_extraction_cutoff') else 6


        for start_node in attack_nodes:
            for end_node in attack_nodes:
                if start_node != end_node:
                    try:
                        # Apply cutoff for performance on large graphs
                        for path in nx.all_simple_paths(graph, source=start_node, target=end_node, cutoff=path_cutoff):
                            paths.append(path)
                    except nx.NetworkXNoPath:
                        continue

        unique_paths = list(set(tuple(p) for p in paths))
        return [list(p) for p in unique_paths]

    # Placeholder for advanced path ranking method (from proposed extensions)
    # def extract_ranked_attack_paths(self, graph: nx.DiGraph, max_paths=10) -> List[Dict[str, Any]]:
    #    """Extract and rank attack paths using multiple criteria."""
    #    # Implementation will go here, calling internal helper methods
    #    pass

    # Placeholder for internal helper methods for path scoring
    # def _compute_path_score(self, graph, path): pass
    # def _temporal_coherence_score(self, graph, path): pass
    # def _mitre_coherence_score(self, graph, path): pass

