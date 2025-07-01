# src/neural_components/gnn_layer.py
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional
from .security_attention import SecurityAwareAttention

class SecurityGNNLayer(nn.Module):
    def __init__(self, input_dim: int, hidden_dim: int, num_heads: int, dropout: float = 0.2, security_feature_dim: int = 10):
        super().__init__()
        self.attention = SecurityAwareAttention(hidden_dim, num_heads, security_feature_dim)
        self.linear1 = nn.Linear(input_dim, hidden_dim)
        self.linear2 = nn.Linear(hidden_dim, hidden_dim) # For feed-forward after attention
        self.dropout = nn.Dropout(dropout)
        self.layer_norm1 = nn.LayerNorm(hidden_dim)
        self.layer_norm2 = nn.LayerNorm(hidden_dim)

    def forward(self, x: torch.Tensor, edge_index: Optional[torch.Tensor], security_features: torch.Tensor):
        # x shape: (batch_size, num_nodes, input_dim)
        # security_features: (batch_size, num_nodes, security_feature_dim)
        # edge_index: (2, num_edges_in_batch) - Used for masking attention if using PyG,
        # but in this simplified version, attention is fully connected across num_nodes within each item in batch.

        # Store residual for first connection
        residual = x

        # Project input features to hidden_dim
        h = F.relu(self.linear1(x))

        # Apply security-aware attention
        # Note: edge_index is currently unused in SecurityAwareAttention for simplicity,
        # as it computes full self-attention within each sequence.
        # For a true GNN layer, attention would typically be masked by edge_index
        # to only consider neighbors.
        attended, attention_weights = self.attention(h, edge_index, security_features)

        # First residual connection and normalization
        output = self.layer_norm1(h + self.dropout(attended))
        
        # Second feed-forward layer for more non-linearity, another residual
        output = self.layer_norm2(output + self.dropout(F.relu(self.linear2(output))))

        return output, attention_weights
