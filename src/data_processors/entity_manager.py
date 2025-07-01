# src/data_processors/entity_manager.py

from typing import Dict, List, Any
import pandas as pd
import logging

class EntityManager:
    def __init__(self):
        self.entity_vocab = {}
        self.action_vocab = {}
        self._next_entity_id = 0
        self._next_action_id = 0
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )

    def get_entity_id(self, entity_name: str) -> int:
        """Map entity name to numerical ID."""
        if entity_name not in self.entity_vocab:
            self.entity_vocab[entity_name] = self._next_entity_id
            self._next_entity_id += 1
        return self.entity_vocab[entity_name]

    def get_action_id(self, action_name: str) -> int:
        """Map action name to numerical ID."""
        if action_name not in self.action_vocab:
            self.action_vocab[action_name] = self._next_action_id
            self._next_action_id += 1
        return self.action_vocab[action_name]

    def extract_security_entities(self, events: pd.DataFrame) -> Dict[str, List[int]]:
        """
        Extract entity (Source/Destination IP:Port) and action (Protocol) IDs from events.
        Assumes column names are standardized to lowercase with underscores.
        """
        required_columns = ['source_ip', 'source_port', 'destination_ip', 'destination_port', 'protocol']
        missing_columns = [col for col in required_columns if col not in events.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        source_entities = [self.get_entity_id(f"{row['source_ip']}:{row['source_port']}") for _, row in events.iterrows()]
        target_entities = [self.get_entity_id(f"{row['destination_ip']}:{row['destination_port']}") for _, row in events.iterrows()]
        actions = [self.get_action_id(str(row['protocol'])) for _, row in events.iterrows()]
        
        self.logger.info(f"Extracted entities for {len(source_entities)} events")
        return {
            'source_entity_ids': source_entities,
            'target_entity_ids': target_entities,
            'action_ids': actions
        }

    def get_vocab_sizes(self) -> Dict[str, int]:
        """Return entity and action vocabulary sizes."""
        return {
            'entity_vocab_size': len(self.entity_vocab),
            'action_vocab_size': len(self.action_vocab)
        }