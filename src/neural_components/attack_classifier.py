# src/neural_components/attack_classifier.py


import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict

class AttackStepClassifier(nn.Module):
    def __init__(self, input_dim: int, num_attack_types: int, num_mitre_techniques: int):
        super().__init__()

        # Multi-task heads
        # Assuming input_dim is the feature dimension from the GNN+TCN output for each node/event
        # These linear layers will operate on each node_embedding independently.
        self.attack_presence = nn.Linear(input_dim, 2)  # Binary classification (e.g., attack/benign)
        self.attack_type = nn.Linear(input_dim, num_attack_types)  # Multi-class classification
        self.mitre_technique = nn.Linear(input_dim, num_mitre_techniques)  # Multi-class classification
        self.severity_score = nn.Linear(input_dim, 1)  # Regression for severity (e.g., 0-1)
        self.confidence_score = nn.Linear(input_dim, 1)  # Regression for confidence (e.g., 0-1)

    def forward(self, node_embeddings: torch.Tensor) -> Dict[str, torch.Tensor]:
        # node_embeddings shape: (batch_size, num_nodes_in_sequence, input_dim)

        # Reshape node_embeddings to (batch_size * num_nodes_in_sequence, input_dim)
        # Ensure tensor is contiguous to avoid RuntimeError in view operation
        flat_node_embeddings = node_embeddings.contiguous().view(-1, node_embeddings.shape[-1])  # (total_nodes_in_batch, input_dim)

        return {
            'attack_presence': self.attack_presence(flat_node_embeddings),  # Logits for binary classification
            'attack_type': self.attack_type(flat_node_embeddings),  # Logits for multi-class classification
            'mitre_technique': self.mitre_technique(flat_node_embeddings),  # Logits for multi-class classification
            'severity': torch.sigmoid(self.severity_score(flat_node_embeddings)),  # Sigmoid for 0-1 range
            'confidence': torch.sigmoid(self.confidence_score(flat_node_embeddings))  # Sigmoid for 0-1 range
        }