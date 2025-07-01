# src/neural_components/neural_pipeline.py


import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Any, Dict, List
from graph_builders.dynamic_graph_builder import DynamicAttackGraphBuilder  # Import for graph building

class NeuralAttackGraphPipeline(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config
        from .attack_graph_gnn import AttackGraphGNN  # Local import to avoid circular dependency issues
        from .attack_classifier import AttackStepClassifier
        
        self.gnn_model = AttackGraphGNN(config)
        self.classifier = AttackStepClassifier(
            self.config.tcn_channels[-1],  # Input dim from TCN output
            self.config.num_attack_types,
            self.config.num_mitre_techniques
        )
        self.graph_builder = DynamicAttackGraphBuilder(config)  # Initialize graph builder

    def forward(self, batch: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        # Extract graph embeddings from GNN+TCN
        gnn_output = self.gnn_model(batch)  # Contains 'node_embeddings' and 'attention_weights'

        # Classify attack steps/sequence using the aggregated node embeddings
        predictions = self.classifier(gnn_output['node_embeddings'])

        # Add attention weights for interpretability
        predictions['attention_weights'] = gnn_output['attention_weights']  # List of attention weights from GNN layers

        return predictions

    def get_attack_graph(self, batch: Dict[str, Any]) -> List[Dict[str, Any]]:
        """
        Generate attack graph data from neural predictions and attention for a batch.
        Uses DynamicAttackGraphBuilder to construct graphs and extract edges.
        """
        self.eval()  # Set model to evaluation mode
        with torch.no_grad():
            output = self.forward(batch)

            # Get the attention weights from the last GNN layer
            last_layer_attention = output['attention_weights'][-1]  # (batch_size, num_heads, num_nodes, num_nodes)

            # Average attention across heads for simplicity in graph construction
            avg_attention = torch.mean(last_layer_attention, dim=1).cpu().numpy()  # (batch_size, num_nodes, num_nodes)

            # Convert predictions for graph data (logits to probabilities/labels)
            attack_presence_probs = F.softmax(output['attack_presence'], dim=1).cpu().numpy()  # (total_nodes, 2)
            attack_type_probs = F.softmax(output['attack_type'], dim=1).cpu().numpy()  # (total_nodes, num_attack_types)
            mitre_technique_probs = F.softmax(output['mitre_technique'], dim=1).cpu().numpy()  # (total_nodes, num_mitre_techniques)
            severity_scores = output['severity'].cpu().numpy()  # (total_nodes, 1)
            confidence_scores = output['confidence'].cpu().numpy()  # (total_nodes, 1)

            graphs_data = []
            num_sequences_in_batch = batch['entities'].shape[0]
            nodes_per_sequence = batch['entities'].shape[1]  # Number of nodes in each sequence

            # --- ADDITION: Extract temporal_info from batch ---
            # Assuming 'temporal_info' is a List[List[pd.Timestamp]] or similar
            # that was collated by custom_collate_fn
            temporal_info_batch = batch.get('temporal_info', [[]] * num_sequences_in_batch) 
            # Default to empty lists if not present, to avoid errors


            for i in range(num_sequences_in_batch):
                # Extract attention and predictions for current sequence
                current_attention_map = avg_attention[i]  # (num_nodes, num_nodes)
                current_entities = batch['entities'][i].cpu().tolist()  # Entity IDs as nodes
                current_temporal_info = temporal_info_batch[i] # Get temporal info for current sequence

                # Extract per-node predictions for this sequence
                start_idx = i * nodes_per_sequence
                end_idx = (i + 1) * nodes_per_sequence
                current_predictions = {
                    'attack_presence': attack_presence_probs[start_idx:end_idx],  # (num_nodes, 2)
                    'attack_type': attack_type_probs[start_idx:end_idx],  # (num_nodes, num_attack_types)
                    'mitre_technique': mitre_technique_probs[start_idx:end_idx],  # (num_nodes, num_mitre_techniques)
                    'severity': severity_scores[start_idx:end_idx],  # (num_nodes, 1)
                    'confidence': confidence_scores[start_idx:end_idx]  # (num_nodes, 1)
                }

                # Build graph using DynamicAttackGraphBuilder
                # --- CHANGE: Pass temporal_info to build_attack_graph ---
                graph = self.graph_builder.build_attack_graph(
                    node_entities=current_entities,
                    predictions=current_predictions,
                    attention_map=current_attention_map,
                    temporal_info=current_temporal_info # NEW ARGUMENT
                )

                # Extract edges from the graph
                edges = [[src, dst] for src, dst in graph.edges()]

                graphs_data.append({
                    'node_entities': current_entities,
                    'attention_map': current_attention_map,
                    'predictions': current_predictions,
                    'edges': edges,  # Add edges to the dictionary
                    'temporal_info': current_temporal_info # Also add temporal_info for context in evaluation/logging
                })

            return graphs_data