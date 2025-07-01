# src/graph_builders/graph_node_processor.py


import numpy as np
import torch # Import torch to use its methods for tensor checks
import pandas as pd # Import pandas for datetime type checks
from typing import Dict, List, Any
import logging

class GraphNodeProcessor:
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )

        # Placeholder for knowledge bases if needed at node level
        # self.mitre_technique_map = self._load_mitre_technique_map() # Example

    def process_nodes(self, node_entities: List[Any], predictions: Dict[str, Any], temporal_info: List[Any]) -> List[Dict[str, Any]]:
        """
        Processes raw node data and predictions to create a list of enriched node attributes.

        Args:
            node_entities (List[Any]): Original entity IDs/names for each node in the sequence.
            predictions (Dict[str, Any]): Dictionary of raw neural predictions (e.g., logits or probabilities).
            temporal_info (List[Any]): Temporal information (e.g., timestamps) for each node.

        Returns:
            List[Dict[str, Any]]: A list of dictionaries, where each dictionary
                                   contains attributes for a single node, ready to be added to NetworkX.
        """
        num_nodes = len(node_entities)
        enriched_nodes_data = []

        for i in range(num_nodes):
            # Extract basic predictions
            # Ensure predictions are handled correctly whether they are tensors or numpy arrays
            node_attack_prob_val = predictions['attack_presence'][i, 1]
            if isinstance(node_attack_prob_val, torch.Tensor):
                node_attack_prob = node_attack_prob_val.item()
            else: # Assume numpy array
                node_attack_prob = node_attack_prob_val # It's already a scalar or 0-dim array

            node_severity_val = predictions['severity'][i, 0]
            if isinstance(node_severity_val, torch.Tensor):
                node_severity = node_severity_val.item()
            else: # Assume numpy array
                node_severity = node_severity_val # It's already a scalar or 0-dim array

            node_is_attack_step = node_attack_prob > 0.5
            
            # --- FIX: Remove .cpu() if it's already a numpy array ---
            attack_type_dist_val = predictions['attack_type'][i]
            if isinstance(attack_type_dist_val, torch.Tensor):
                attack_type_dist_list = attack_type_dist_val.cpu().tolist()
            else: # Assume numpy array
                attack_type_dist_list = attack_type_dist_val.tolist()


            # Use temporal_info if available, otherwise use index as placeholder
            temporal_position = temporal_info[i] if temporal_info and i < len(temporal_info) else i

            # Node risk score (placeholder, can be enhanced)
            risk_score = self._compute_risk_score(predictions, i, node_severity)

            # Centrality score placeholder (can be calculated based on initial attention or graph structure later)
            # For this node processor, sum of outgoing attention could be a simple proxy
            # Make sure 'attention_weights' is extracted correctly and passed.
            # In neural_pipeline.py, output['attention_weights'][-1] is a torch.Tensor,
            # which is then converted to numpy via .cpu().numpy() before being passed as avg_attention.
            # So, current_attention_map in build_attack_graph is a numpy array.
            if 'attention_weights' in predictions and predictions['attention_weights'] is not None and len(predictions['attention_weights'].shape) == 2: # Check if it's a 2D numpy array for sum
                # 'predictions' dict in process_nodes is actually `current_predictions`, 
                # which contains sliced numpy arrays for pred_probs and scores,
                # BUT 'attention_weights' key in the top-level 'predictions' from pipeline output 
                # is a LIST of original torch TENSORS.
                # The 'attention_map' argument to build_attack_graph is avg_attention (numpy array).
                # This needs careful handling. The centrality score here is being calculated 
                # using the `predictions` dict, not the `attention_map` argument of `build_attack_graph`.
                # If you want it from the actual attention_map, it needs to be passed here from build_attack_graph.
                # For now, I'll use a safer fallback, as predictions['attention_weights'] might not be structured for this.
                centrality_score = 0.0 # Default fallback
                # If predictions['attention_weights'] could somehow carry the processed attention map, use it.
                # Otherwise, this calculation will need to happen after the graph is built, 
                # or a pre-processed attention weight array needs to be passed explicitly.

            else:
                centrality_score = 0.0 # Fallback if attention_weights not directly available or structured as expected here


            node_attrs = {
                'entity_id': node_entities[i] if i < len(node_entities) else f"node_{i}",
                'attack_probability': node_attack_prob, # Already scalar
                'severity': node_severity, # Already scalar
                'is_attack_step': bool(node_is_attack_step), # Ensure boolean
                'attack_type_distribution': attack_type_dist_list, # Already list
                'temporal_position': temporal_position,
                'risk_score': risk_score,
                'centrality_score': centrality_score,
                # 'mitre_techniques': mitre_techniques, # Uncomment and implement later
                # 'mitre_tactics': self._get_tactics_for_techniques(mitre_techniques), # Uncomment and implement later
            }
            enriched_nodes_data.append(node_attrs)
            
        return enriched_nodes_data

    def _compute_risk_score(self, predictions: Dict[str, Any], node_idx: int, default_severity: float) -> float:
        """
        Computes a composite risk score for a node.
        This is a basic placeholder and will be enhanced in later phases.
        """
        # For now, a simple combination of attack probability and severity
        # Ensure conversion to item() if it's a tensor, otherwise assume numpy scalar
        attack_prob_val = predictions['attack_presence'][node_idx, 1]
        if isinstance(attack_prob_val, torch.Tensor):
            attack_prob = attack_prob_val.item()
        else:
            attack_prob = attack_prob_val # Assumes numpy scalar
            
        # You could also integrate specific threat intelligence, CVSS scores, etc. here
        # Example: return (attack_prob * 0.7) + (default_severity * 0.3)
        return default_severity + attack_prob * 0.5 # Simple additive risk
        
    # Placeholder for MITRE techniques extraction (Phase 2)
    # def _get_likely_mitre_techniques(self, predictions: Dict[str, Any], node_idx: int) -> List[str]:
    #     """
    #     Maps neural predictions to likely MITRE ATT&CK techniques.
    #     Requires mapping attack_type/mitre_technique predictions to actual MITRE IDs/names.
    #     """
    #     # Example: return top-K techniques based on predictions['mitre_technique']
    #     return [] # Placeholder

    # Placeholder for getting tactics from techniques (Phase 2)
    # def _get_tactics_for_techniques(self, techniques: List[str]) -> List[str]:
    #     """
    #     Retrieves MITRE ATT&CK tactics corresponding to given techniques.
    #     Requires a loaded MITRE knowledge base.
    #     """
    #     return [] # Placeholder