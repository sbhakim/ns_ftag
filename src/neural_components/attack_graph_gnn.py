# src/neural_components/attack_graph_gnn.py
import torch
import torch.nn as nn
from typing import Any, Dict, List # ADDED Dict, List
from .gnn_layer import SecurityGNNLayer
from .tcn_layer import TemporalConvolutionalNetwork

class AttackGraphGNN(nn.Module):
    def __init__(self, config: Any):
        super().__init__()
        self.config = config

        # Entity and action embeddings
        # These vocab sizes should be determined after data processing in SecurityEventProcessor
        self.entity_embedding = nn.Embedding(config.entity_vocab_size, config.embedding_dim)
        self.action_embedding = nn.Embedding(config.action_vocab_size, config.embedding_dim)

        # GNN layers
        # The input_dim for the first GNN layer is the sum of embedding_dim for entity and action
        gnn_input_dim = config.embedding_dim * 2 # Assuming concatenation of entity and action embeddings
        self.gnn_layers = nn.ModuleList()
        for i, dim in enumerate(config.gnn_hidden_dims):
            current_input_dim = gnn_input_dim if i == 0 else config.gnn_hidden_dims[i-1]
            self.gnn_layers.append(
                SecurityGNNLayer(current_input_dim, dim, config.gnn_num_heads,
                                 dropout=config.gnn_dropout, security_feature_dim=config.security_feature_dim)
            )

        # TCN for temporal modeling
        # Input channels for TCN will be the output dimension of the last GNN layer
        tcn_input_channels = config.gnn_hidden_dims[-1]
        self.tcn = TemporalConvolutionalNetwork(
            tcn_input_channels,
            config.tcn_channels, # num_channels for TCN
            config.tcn_kernel_size,
            config.gnn_dropout # Using gnn_dropout for TCN as well
        )

    def forward(self, batch: Dict[str, torch.Tensor]):
        # batch['entities']: (batch_size, num_nodes_in_sequence)
        # batch['actions']: (batch_size, num_nodes_in_sequence)
        # batch['edge_index']: (2, total_num_edges_in_batch) - used if GNN layer handles batched graphs (e.g., PyG)
        # batch['security_features']: (batch_size, num_nodes_in_sequence, security_feature_dim)

        # Process entities and actions to get initial node features
        entity_embeds = self.entity_embedding(batch['entities']) # (batch_size, num_nodes, embedding_dim)
        action_embeds = self.action_embedding(batch['actions'])   # (batch_size, num_nodes, embedding_dim)

        # Combine embeddings to form initial node features for each event/node in the sequence
        node_features = torch.cat([entity_embeds, action_embeds], dim=-1) # (batch_size, num_nodes, 2*embedding_dim)

        # Apply GNN layers
        attention_weights = []
        current_node_features = node_features
        for gnn_layer in self.gnn_layers:
            # The SecurityGNNLayer currently processes each item in the batch independently
            # regarding attention calculation (full self-attention per sequence).
            # If using a true batched graph framework (e.g., PyG), `edge_index` would be passed
            # and used for sparse attention/message passing across the whole batch.
            current_node_features, attn = gnn_layer(current_node_features, batch['edge_index'],
                                                    batch['security_features'])
            attention_weights.append(attn)

        # Apply temporal modeling
        # TCN expects (batch, sequence_length, features)
        temporal_features = self.tcn(current_node_features) # Output is (batch, sequence_length, tcn_output_features)

        return {
            'node_embeddings': temporal_features,
            'attention_weights': attention_weights # List of attention weights from each GNN layer
        }