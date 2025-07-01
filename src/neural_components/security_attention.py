import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional

class SecurityAwareAttention(nn.Module):
    def __init__(self, hidden_dim: int, num_heads: int, security_feature_dim: int):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.num_heads = num_heads
        self.head_dim = hidden_dim // num_heads
        assert self.head_dim * num_heads == hidden_dim, "hidden_dim must be divisible by num_heads"

        self.W_q = nn.Linear(hidden_dim, hidden_dim)
        self.W_k = nn.Linear(hidden_dim, hidden_dim)
        self.W_v = nn.Linear(hidden_dim, hidden_dim)

        # Recommendation: Decide early whether attention biases are additive (logits) or multiplicative (scaled scores)
        # and either implement a pairwise feature extractor or reserve it for Phase 2 to avoid re-architecting.
        # Current: Additive bias to logits as a per-query-node influence.
        self.security_bias_proj = nn.Linear(security_feature_dim, num_heads) # Project security features to per-head bias

        self.out_proj = nn.Linear(hidden_dim, hidden_dim)

    def forward(self, node_features: torch.Tensor, edge_index: Optional[torch.Tensor] = None, security_features: Optional[torch.Tensor] = None):
        # node_features: (batch_size, num_nodes, hidden_dim)
        # security_features: (batch_size, num_nodes, security_feature_dim)
        # edge_index: (2, total_num_edges_in_batch) - This tensor contains global indices for the batched graph.

        batch_size, num_nodes, _ = node_features.shape

        Q = self.W_q(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        K = self.W_k(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)
        V = self.W_v(node_features).view(batch_size, num_nodes, self.num_heads, self.head_dim).transpose(1, 2)

        # Q, K, V shape: (batch_size, num_heads, num_nodes, head_dim)

        # Compute attention scores
        attention_scores = torch.matmul(Q, K.transpose(-2, -1))
        # attention_scores shape: (batch_size, num_heads, num_nodes, num_nodes)

        # Add security-aware weighting as an additive bias to attention scores
        if security_features is not None:
            # Project security features to a bias per head per query node
            security_bias = self.security_bias_proj(security_features) # (batch_size, num_nodes, num_heads)
            security_bias = security_bias.permute(0, 2, 1).unsqueeze(-1) # (batch_size, num_heads, num_nodes, 1)
            # This bias is added to each row (query node) of the attention score matrix
            attention_scores = attention_scores + security_bias

        # --- IMPORTANT: Graph Attention Masking ---
        # In a true Graph Attention Network (GAT), attention should only be computed between
        # connected nodes (as defined by edge_index). Currently, this attention is full self-attention
        # within each sequence in the batch.
        #
        # Applying masking based on `edge_index` here is complex with the current batching strategy
        # (where `edge_index` is global for the entire flattened batch, and `node_features` is
        # `(batch_size, num_nodes, dim)`).
        #
        # To correctly mask, one would typically:
        # 1. Obtain the adjacency matrix for each graph in the batch (or a single large sparse adjacency matrix).
        # 2. Convert this adjacency matrix into a mask (e.g., `float('-inf')` for disconnected, `0` for connected).
        # 3. Apply this mask to `attention_scores` *before* the softmax.
        #
        # Example (conceptual, requires `batch_ptr` or PyG):
        # if edge_index is not None:
        #     # This part would require knowing the original graph sizes in the batch
        #     # and mapping global edge_index to per-graph local indices.
        #     # For instance, if using PyG, you'd have a `batch` tensor or `ptr` for this.
        #     # For now, this is a placeholder for future integration of a proper graph library.
        #     mask = build_mask_from_edge_index(edge_index, batch_size, num_nodes, device=node_features.device)
        #     attention_scores = attention_scores.masked_fill(mask == 0, float('-inf')) # Example masking
        #
        # For this incremental step, we proceed with full self-attention within each sequence,
        # acknowledging that true graph-aware message passing will be a key part of future
        # refactoring (e.g., integrating PyTorch-Geometric).
        # -------------------------------------------

        attention_probs = F.softmax(attention_scores / (self.head_dim ** 0.5), dim=-1)
        # attention_probs shape: (batch_size, num_heads, num_nodes, num_nodes)

        # Apply attention to values
        output = torch.matmul(attention_probs, V)
        # output shape: (batch_size, num_heads, num_nodes, head_dim)

        # Concatenate heads and apply final projection
        output = output.transpose(1, 2).contiguous().view(batch_size, num_nodes, self.hidden_dim)
        output = self.out_proj(output)

        return output, attention_probs