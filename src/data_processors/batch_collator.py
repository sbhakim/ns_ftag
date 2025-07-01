# src/data_processors/batch_collator.py


import torch
from typing import List, Dict, Any

def custom_collate_fn(batch: List[Dict[str, Any]]) -> Dict[str, Any]:
    """Custom collate function for batching variable-sized graphs."""
    collated_batch = {}
    targets_batch = [d['targets'] for d in batch]
    collated_batch['targets'] = {}
    for target_key in targets_batch[0].keys():
        collated_batch['targets'][target_key] = torch.cat([t[target_key] for t in targets_batch], dim=0)

    node_offset = 0
    all_edge_index = []
    for i, data_item in enumerate(batch):
        if i == 0:
            collated_batch['entities'] = data_item['entities'].unsqueeze(0)
            collated_batch['actions'] = data_item['actions'].unsqueeze(0)
            collated_batch['security_features'] = data_item['security_features'].unsqueeze(0)
            collated_batch['true_edges'] = [data_item['true_edges']]
        else:
            collated_batch['entities'] = torch.cat((collated_batch['entities'], data_item['entities'].unsqueeze(0)), dim=0)
            collated_batch['actions'] = torch.cat((collated_batch['actions'], data_item['actions'].unsqueeze(0)), dim=0)
            collated_batch['security_features'] = torch.cat((collated_batch['security_features'], data_item['security_features'].unsqueeze(0)), dim=0)
            collated_batch['true_edges'].append(data_item['true_edges'])
        current_num_nodes = data_item['entities'].shape[0]
        adjusted_edge_index = data_item['edge_index'] + node_offset
        all_edge_index.append(adjusted_edge_index)
        node_offset += current_num_nodes
    collated_batch['edge_index'] = torch.cat(all_edge_index, dim=1) if all_edge_index else torch.tensor([], dtype=torch.long)
    return collated_batch