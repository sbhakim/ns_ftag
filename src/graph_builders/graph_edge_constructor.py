# src/graph_builders/graph_edge_constructor.py


import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple, Any
import networkx as nx
import logging

class GraphEdgeConstructor:
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )

        # Placeholders for domain knowledge (will be loaded/populated in Phase 2)
        # For Phase 1, these can be simple dictionaries or loaded from dummy files.
        self.temporal_constraints = self._load_temporal_constraints()
        self.mitre_attack_flow = self._load_mitre_flows()

    def _load_temporal_constraints(self) -> Dict[str, List[str]]:
        """
        Placeholder to load predefined temporal attack progression constraints.
        Example: {'Reconnaissance': ['Exploitation', 'Credential Access'], ...}
        In Phase 1, this can be a simple hardcoded dict or empty.
        """
        # Mapping numerical attack_type IDs to conceptual names for clarity
        # These names should align with your LabelExtractor's attack_types.
        # Example mapping (adjust to your config.num_attack_types and LabelExtractor):
        attack_type_names = {
            0: 'Benign', 1: 'DDoS', 2: 'PortScan', 3: 'Web Attack', 4: 'Bot', 5: 'Infiltration'
        }
        
        # Define some basic (conceptual) temporal constraints
        # This is highly simplified and needs real domain expertise / MITRE mapping
        constraints = {
            'PortScan': ['Web Attack', 'Infiltration'], # Port scan could precede web attacks or infiltration
            'Web Attack': ['Infiltration'],              # Web attack could lead to infiltration
            'Infiltration': ['Bot', 'DDoS'],             # Infiltration could lead to botnet or DDoS activity
            'DDoS': [],                                  # DDoS is often a terminal attack goal
            'Bot': ['DDoS']                              # Botnet could launch DDoS
        }
        
        # Convert conceptual names to numerical IDs for internal use if needed, or
        # adjust _validate_temporal_constraints to use names.
        # For now, let's assume direct ID usage in validation for simplicity if labels are IDs.
        # This will be properly handled when true MITRE mapping occurs.
        numerical_constraints = {}
        # This is a conceptual mapping. You'll need to align it with LabelExtractor's attack_types.
        # For now, this part is for demonstration.
        # For CICIDS2017 labels, it's safer to use the numerical IDs directly for now if available.
        # Assuming attack_type_dist from predictions already gives you IDs for comparison.

        return constraints # Return conceptual for now, or map to IDs directly if config.attack_types are used.

    def _load_mitre_flows(self) -> Dict[str, List[str]]:
        """
        Placeholder to load MITRE ATT&CK tactical progression flows.
        In Phase 1, this can be a simple hardcoded dict or empty.
        """
        # Example: {'Reconnaissance': ['Resource Development', 'Initial Access'], ...}
        # This will be populated with actual MITRE data in Phase 2
        return {}

    def construct_edges(self, graph: nx.DiGraph, attention_map: np.ndarray, temporal_info: List[Any], predictions: Dict[str, Any]) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Constructs and validates edges between nodes based on attention, temporal constraints,
        and (future) domain knowledge.

        Args:
            graph (nx.DiGraph): The NetworkX graph with nodes already added (from GraphNodeProcessor).
            attention_map (np.ndarray): Attention weights between nodes (num_nodes x num_nodes).
            temporal_info (List[Any]): Temporal information (e.g., timestamps) for each node.
            predictions (Dict[str, Any]): Raw neural predictions for each node.

        Returns:
            List[Tuple[int, int, Dict[str, Any]]]: A list of tuples, each representing an edge
                                                   (source_node_id, target_node_id, edge_attributes).
        """
        candidate_edges_data = self._generate_candidate_edges(attention_map, predictions, graph)
        validated_edges = self._validate_edges_with_domain_knowledge(candidate_edges_data, graph, temporal_info, predictions)
        return validated_edges

    def _generate_candidate_edges(self, attention_map: np.ndarray, predictions: Dict[str, Any], graph: nx.DiGraph) -> List[Dict[str, Any]]:
        """
        Generates candidate edges based on attention weights.
        Adds preliminary edge attributes like weight and confidence.
        """
        candidate_edges = []
        # num_nodes is no longer explicitly needed here, but can be derived from graph.number_of_nodes()
        # if predictions indexing is relative to sequence length.
        # Assuming predictions are for the current sequence, so num_nodes = attention_map.shape[0] is correct for indexing.
        num_nodes = attention_map.shape[0] 
        threshold = self.config.attention_threshold

        # Ensure attention_map is a tensor for thresholding if it's not already
        # In neural_pipeline.py, avg_attention is already numpy.ndarray
        attention_map_processed = torch.tensor(attention_map) if isinstance(attention_map, np.ndarray) else attention_map

        # Get indices where attention is above threshold
        edges_indices = (attention_map_processed > threshold).nonzero(as_tuple=False)

        for edge_idx in edges_indices:
            src_node_id, dst_node_id = edge_idx[0].item(), edge_idx[1].item()
            
            # Skip self-loops, and check if nodes exist (important for robust dummy data)
            if src_node_id == dst_node_id or not graph.has_node(src_node_id) or not graph.has_node(dst_node_id):
                continue

            weight = attention_map[src_node_id, dst_node_id].item()
            
            # --- FIX: Confidence Calculation for numpy arrays ---
            if 'confidence' in predictions:
                confidence_array = predictions['confidence']
                # Check if it's a numpy array with at least 1 dimension (e.g., (N,) or (N,1))
                if isinstance(confidence_array, np.ndarray) and confidence_array.ndim > 0:
                    # Indexing directly into the numpy array confidence_array
                    src_confidence = confidence_array[src_node_id].item() 
                    dst_confidence = confidence_array[dst_node_id].item()
                    confidence = (src_confidence + dst_confidence) / 2.0
                else: # Fallback for scalar or malformed 'confidence'
                    confidence = weight 
            else:
                confidence = weight # Fallback to attention weight if no confidence prediction

            # Ensure node_attack_prob is a scalar, not a 1-element tensor
            src_node_prob = graph.nodes[src_node_id].get('attack_probability', 0.0)
            dst_node_prob = graph.nodes[dst_node_id].get('attack_probability', 0.0)

            # Simple causality based on attention (placeholder)
            causality_score = (src_node_prob * weight) / max(1e-9, (src_node_prob + dst_node_prob)) # Example heuristic

            candidate_edges.append({
                'source': src_node_id,
                'target': dst_node_id,
                'weight': weight,
                'confidence': confidence,
                'causality': causality_score,
                # Add other preliminary attributes that might be useful for validation
            })
        self.logger.info(f"Generated {len(candidate_edges)} raw candidate edges.")
        return candidate_edges

    def _validate_edges_with_domain_knowledge(self, candidate_edges: List[Dict[str, Any]], graph: nx.DiGraph, temporal_info: List[Any], predictions: Dict[str, Any]) -> List[Tuple[int, int, Dict[str, Any]]]:
        """
        Applies temporal and (future) MITRE-based validation to filter and enrich candidate edges.
        """
        validated_edges_data = []
        for edge_data in candidate_edges:
            src, dst = edge_data['source'], edge_data['target']

            # Retrieve predicted attack types/labels for validation
            # Ensure predictions['attack_type'] is processed correctly (e.g., argmax for class ID)
            if 'attack_type' in predictions and isinstance(predictions['attack_type'], np.ndarray) and predictions['attack_type'].ndim > 0:
                src_attack_type_pred = np.argmax(predictions['attack_type'][src]).item() # Use np.argmax for numpy arrays
                dst_attack_type_pred = np.argmax(predictions['attack_type'][dst]).item()
            else:
                # Fallback if attack_type predictions are not available or malformed numpy array
                src_attack_type_pred = 0 # Assume Benign
                dst_attack_type_pred = 0 # Assume Benign


            # 1. Temporal Validation
            # Pass src and dst directly as they are the node indices needed by _validate_temporal_constraints
            temporal_valid = self._validate_temporal_constraints(src, dst, temporal_info, predictions)
            if not temporal_valid:
                continue # Skip if temporal constraint violated

            # 2. MITRE Attack Flow Validation (Placeholder for Phase 2)
            # This would involve looking at predicted MITRE tactics/techniques for src and dst
            # and checking if the transition aligns with known MITRE progressions.
            mitre_relation_valid = True # Default to True for Phase 1
            mitre_relation_type = "generic" # Placeholder

            # If not mitre_relation_valid:
            #    continue

            # 3. Attack Type Compatibility (simple heuristic for Phase 1)
            # Check if attack types conceptually align for a plausible transition
            attack_type_compatibility = 1.0 # Default if no specific rules
            # Example: if self.temporal_constraints.get(src_attack_type_pred) and dst_attack_type_pred in self.temporal_constraints[src_attack_type_pred]:
            #    attack_type_compatibility = 1.0
            # else:
            #    attack_type_compatibility = 0.5 # Lower score for less compatible types

            # Update edge attributes
            edge_data['temporal_validity'] = temporal_valid
            edge_data['mitre_relation'] = mitre_relation_type
            edge_data['attack_type_compatibility'] = attack_type_compatibility

            validated_edges_data.append((src, dst, edge_data))

        self.logger.info(f"Validated {len(validated_edges_data)} edges after temporal and domain checks.")
        return validated_edges_data


    def _validate_temporal_constraints(self, source_idx: int, target_idx: int, temporal_info: List[Any], predictions: Dict[str, Any]) -> bool:
        """
        Validates edges against cybersecurity temporal constraints.
        Returns True if the edge is temporally valid, False otherwise.
        """
        # --- FIX: Added check for temporal_info[source_idx] or temporal_info[target_idx] being None/NaT ---
        # Also, ensure source_idx and target_idx are within bounds of temporal_info
        if (not temporal_info or 
            source_idx >= len(temporal_info) or 
            target_idx >= len(temporal_info) or
            pd.isna(temporal_info[source_idx]) or # Check for NaN/NaT explicitly
            pd.isna(temporal_info[target_idx])):
            self.logger.warning(f"Temporal info missing, NaN, or invalid indices ({source_idx}, {target_idx}) in current sequence. Skipping temporal validation for this edge.")
            return True # Cannot validate, so assume valid for now

        source_time = temporal_info[source_idx]
        target_time = temporal_info[target_idx]

        # Basic temporal ordering: target event must happen after source event
        if target_time <= source_time: # pd.Timestamp handles comparisons with NaT correctly
            # self.logger.debug(f"Temporal order violated: {source_time} -> {target_time}") # Too verbose for full run
            return False

        # Attack progression validation (using conceptual attack type names for now)
        # This part requires the _load_temporal_constraints to provide actual mappings
        # and predictions['attack_type'] to be meaningful.
        # For Phase 1, if self.temporal_constraints is simple, this check can be basic.
        
        # Map predicted numerical attack types to names for lookup in self.temporal_constraints
        # (This is a simplified approach until full MITRE integration in Phase 2)
        # Placeholder for real attack_type_names mapping
        attack_type_id_to_name = {
            0: 'Benign', 1: 'DDoS', 2: 'PortScan', 3: 'Web Attack', 4: 'Bot', 5: 'Infiltration'
        }
        
        # --- FIX: Check for numpy array for predictions['attack_type'] and use src_idx/target_idx ---
        if 'attack_type' in predictions and isinstance(predictions['attack_type'], np.ndarray) and predictions['attack_type'].ndim > 0:
            # Safely get item, assuming predictions['attack_type'] is (num_nodes_in_seq, num_attack_types)
            # The .item() will convert 0-dim array to scalar
            src_attack_type_id = np.argmax(predictions['attack_type'][source_idx]).item() 
            dst_attack_type_id = np.argmax(predictions['attack_type'][target_idx]).item()
        else:
            # Fallback if attack_type predictions are not available or malformed numpy array
            src_attack_type_id = 0 # Assume Benign
            dst_attack_type_id = 0 # Assume Benign

        source_attack_type_name = attack_type_id_to_name.get(src_attack_type_id, 'Unknown')
        target_attack_type_name = attack_type_id_to_name.get(dst_attack_type_id, 'Unknown')

        valid_progressions = self.temporal_constraints.get(source_attack_type_name, [])
        if valid_progressions and target_attack_type_name not in valid_progressions:
            # self.logger.debug(f"Attack progression invalid: {source_attack_type_name} -> {target_attack_type_name}") # Too verbose
            return False # Invalid progression according to defined rules

        # Time window validation (attacks shouldn't be too far apart)
        # Only apply if timestamps are valid datetime objects
        # `pd.isna(source_time)` and `pd.isna(target_time)` already handle NaT
        if pd.api.types.is_datetime64_any_dtype(source_time) and pd.api.types.is_datetime64_any_dtype(target_time):
            time_diff = (target_time - source_time).total_seconds() # Get difference in seconds
            # Use the max_attack_step_gap from config
            max_allowed_gap = self.config.max_attack_step_gap 
            if time_diff > max_allowed_gap:
                # self.logger.debug(f"Time gap too large: {time_diff}s > {max_allowed_gap}s") # Too verbose
                return False

        return True

    # Placeholder for MITRE specific validation (Phase 2)
    # def _is_valid_mitre_progression(self, source_tactics: List[str], target_tactics: List[str]) -> bool:
    #    """
    #    Checks if a tactical progression is valid according to MITRE flows.
    #    Requires self.mitre_attack_flow to be populated.
    #    """
    #    return True # Placeholder for Phase 1