# src/data_processors/dataset.py

import torch
from torch.utils.data import Dataset
import pandas as pd
from typing import Dict, List, Any

class SecurityEventDataset(Dataset):
    def __init__(self, data_path: str, config: Any, processor: Any = None, entity_manager: Any = None, 
                 relationship_extractor: Any = None, feature_extractor: Any = None, label_extractor: Any = None):
        self.config = config
        
        # === NEW: Auto-select components based on dataset type ===
        if processor is None or entity_manager is None or relationship_extractor is None or feature_extractor is None or label_extractor is None:
            self._auto_select_components()
        else:
            # Use provided components (for backward compatibility)
            self.processor = processor
            self.entity_manager = entity_manager
            self.relationship_extractor = relationship_extractor
            self.feature_extractor = feature_extractor
            self.label_extractor = label_extractor
                
        self.events_df = self.processor.load_data(data_path)
        self.sequences = self._create_sequences()
        self.config.entity_vocab_size = self.entity_manager.get_vocab_sizes()['entity_vocab_size']
        self.config.action_vocab_size = self.entity_manager.get_vocab_sizes()['action_vocab_size']

    def _auto_select_components(self):
        """Auto-select dataset-specific components based on config.dataset_type."""
        if self.config.dataset_type == "optc": 
            from .optc_processor import OpTCProcessor
            from .optc_entity_manager import OpTCEntityManager
            from .optc_relationship_extractor import OpTCRelationshipExtractor
            from .optc_feature_extractor import OpTCFeatureExtractor
            from .optc_label_extractor import OpTCLabelExtractor
                        
            self.processor = OpTCProcessor(self.config)
            self.entity_manager = OpTCEntityManager()
            self.relationship_extractor = OpTCRelationshipExtractor(self.config)
            self.feature_extractor = OpTCFeatureExtractor(self.config)
            self.label_extractor = OpTCLabelExtractor(self.config)
        else:  # Default to CICIDS2017
            from .cicids2017_processor import CICIDS2017Processor
            from .entity_manager import EntityManager
            from .relationship_extractor import RelationshipExtractor
            from .feature_extractor import FeatureExtractor
            from .label_extractor import LabelExtractor
                        
            self.processor = CICIDS2017Processor(self.config)
            self.entity_manager = EntityManager()
            self.relationship_extractor = RelationshipExtractor(self.config)
            self.feature_extractor = FeatureExtractor(self.config)
            self.label_extractor = LabelExtractor(self.config)

    def _create_sequences(self) -> List[Dict[str, Any]]:
        """Create temporal sequences from security events."""
        sequences = []
        window_size = self.config.sequence_window_size
        min_seq_length = self.config.min_sequence_length
        for i in range(0, len(self.events_df), window_size):
            window_events = self.events_df.iloc[i:i + window_size]
            if len(window_events) >= min_seq_length:
                sequence_data = self._process_event_sequence(window_events)
                sequences.append(sequence_data)
        return sequences

    def _process_event_sequence(self, events: pd.DataFrame) -> Dict[str, Any]:
        """Convert event sequence to graph-like representation."""
        entities_actions = self.entity_manager.extract_security_entities(events)
        edge_index = self.relationship_extractor.extract_relationships(events)
        if not edge_index:
            edge_index = [[0, 0]] if len(events) == 1 else [[i, i + 1] for i in range(len(events) - 1)]
        edge_index = torch.tensor(edge_index, dtype=torch.long).t().contiguous()
        security_features = torch.tensor(self.feature_extractor.extract_security_features(events), dtype=torch.float32)
        targets = self.label_extractor.extract_labels(events)
        true_edges = self.relationship_extractor.extract_true_edges(events)
        
        # --- NEW: Extract temporal_info (timestamps) ---
        # Ensure 'timestamp' column exists and is in datetime format from preprocessing
        temporal_info = events['timestamp'].tolist()

        return {
            'entities': torch.tensor(entities_actions['source_entity_ids'], dtype=torch.long),
            'actions': torch.tensor(entities_actions['action_ids'], dtype=torch.long),
            'edge_index': edge_index,
            'security_features': security_features,
            'targets': targets,
            'true_edges': true_edges,
            'temporal_info': temporal_info 
        }

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx]