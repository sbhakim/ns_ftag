# src/data_processors/batch_collator.py


import torch
from typing import List, Dict, Any

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching variable-sized graphs."""
    collated_batch = {}
    
    # Collect and concatenate targets (which are tensors)
    targets_batch = [d['targets'] for d in batch]
    collated_batch['targets'] = {}
    for target_key in targets_batch[0].keys():
        # Ensure that target tensors are concatenated even if some are empty for a given key
        # (though typically targets should be consistently present)
        tensors_to_cat = [t[target_key] for t in targets_batch if t[target_key].numel() > 0]
        if tensors_to_cat:
            collated_batch['targets'][target_key] = torch.cat(tensors_to_cat, dim=0)
        else:
            # Handle case where a target type might be completely empty across the batch (e.g., no attacks)
            # This might require checking expected shapes from config if a dummy empty tensor is needed
            # For now, if all are empty, the key might be missing or empty tensor.
            # It's better to return an empty tensor of the expected type if the key must exist.
            pass # Let it be missing if all are empty, or a later stage will handle it.


    node_offset = 0
    all_edge_index = []
    # --- NEW: List to collect temporal_info ---
    all_temporal_info = [] 
    all_true_edges = [] # Collect true_edges as a list of lists

    for i, data_item in enumerate(batch):
        # Unsqueeze(0) is typically used to add a batch dimension of 1.
        # For collate_fn, if items are sequences, we concatenate them along batch dimension.
        # entities, actions, security_features are already 2D (num_nodes, features) for a single item.
        # We want to stack them to (batch_size, num_nodes, features).
        # Assuming data_item['entities'] etc. are already Tensors
        
        # If this is the first item, initialize the collated tensors/lists
        if i == 0:
            collated_batch['entities'] = data_item['entities'].unsqueeze(0) # (1, num_nodes, entity_dim)
            collated_batch['actions'] = data_item['actions'].unsqueeze(0)   # (1, num_nodes, action_dim)
            collated_batch['security_features'] = data_item['security_features'].unsqueeze(0) # (1, num_nodes, feature_dim)
        else:
            # For subsequent items, concatenate along the batch dimension (dim=0)
            collated_batch['entities'] = torch.cat((collated_batch['entities'], data_item['entities'].unsqueeze(0)), dim=0)
            collated_batch['actions'] = torch.cat((collated_batch['actions'], data_item['actions'].unsqueeze(0)), dim=0)
            collated_batch['security_features'] = torch.cat((collated_batch['security_features'], data_item['security_features'].unsqueeze(0)), dim=0)
        
        # Adjust edge_index for global node IDs within the batch
        current_num_nodes = data_item['entities'].shape[0] # Number of nodes in this sequence
        adjusted_edge_index = data_item['edge_index'] + node_offset
        all_edge_index.append(adjusted_edge_index)
        node_offset += current_num_nodes

        # --- NEW: Append temporal_info and true_edges (which are lists) ---
        all_temporal_info.append(data_item['temporal_info'])
        all_true_edges.append(data_item['true_edges'])


    # Final concatenation for edge_index
    collated_batch['edge_index'] = torch.cat(all_edge_index, dim=1) if all_edge_index else torch.tensor([], dtype=torch.long)
    
    # --- NEW: Add collected temporal_info and true_edges to collated_batch ---
    collated_batch['temporal_info'] = all_temporal_info
    collated_batch['true_edges'] = all_true_edges


    return collated_batch