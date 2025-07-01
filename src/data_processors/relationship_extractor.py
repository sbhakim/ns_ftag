# src/data_processors/relationship_extractor.py


from typing import List, Any
import pandas as pd
import logging

class RelationshipExtractor:
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )

    def extract_relationships(self, events: pd.DataFrame) -> List[List[int]]:
        """Extract edges based on temporal proximity and shared entities."""
        required_columns = ['source_ip', 'destination_ip']
        missing_columns = [col for col in required_columns if col not in events.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        relationships = []
        window_size = 5  # Temporal window for co-occurrence
        for i in range(len(events)):
            current = events.iloc[i]
            for j in range(i + 1, min(i + window_size, len(events))):
                other = events.iloc[j]
                if current['source_ip'] == other['source_ip'] or current['destination_ip'] == other['destination_ip']:
                    relationships.append([i, j])
        if not relationships and len(events) > 1:
            relationships = [[i, i + 1] for i in range(len(events) - 1)]
        relationships = list(set(tuple(r) for r in relationships))
        self.logger.info(f"Extracted {len(relationships)} relationships")
        return relationships

    def extract_true_edges(self, events: pd.DataFrame) -> List[List[int]]:
        """Extract ground truth edges for attack sequences."""
        required_columns = ['label']
        missing_columns = [col for col in required_columns if col not in events.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        true_edges = []
        attack_flows = events[events['label'] != 'BENIGN']
        for i in range(len(attack_flows) - 1):
            if attack_flows.iloc[i]['label'] == attack_flows.iloc[i + 1]['label']:
                true_edges.append([i, i + 1])  # Sequential edges within same attack
        self.logger.info(f"Extracted {len(true_edges)} true edges")
        return true_edges