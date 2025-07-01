# src/data_processors/feature_extractor.py


import pandas as pd
from typing import List, Any
import logging

class FeatureExtractor:
    def __init__(self, config: Any):
        self.config = config
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )

    def extract_security_features(self, events: pd.DataFrame) -> List[List[float]]:
        """Extract security features for attention mechanism."""
        # Updated required_columns to match the standardized names from CICIDS2017Processor logs
        required_columns = [
            'total_fwd_packets', 'total_backward_packets', 'total_length_of_fwd_packets',
            'total_length_of_bwd_packets', 'flowduration', 'flow_bytes_s', 'flow_packets_s',
            'fwd_packet_length_mean', 'bwd_packet_length_mean', 'label'
        ]
        
        missing_columns = [col for col in required_columns if col not in events.columns]
        if missing_columns:
            self.logger.error(f"Missing required columns: {missing_columns}")
            raise ValueError(f"Missing required columns: {missing_columns}")

        features = []
        for _, row in events.iterrows():
            feature_row = [
                row['total_fwd_packets'],
                row['total_backward_packets'],
                row['total_length_of_fwd_packets'],
                row['total_length_of_bwd_packets'],
                row['flowduration'], # Changed from flow_duration
                row['flow_bytes_s'] if not pd.isna(row['flow_bytes_s']) else 0.0, # Changed from flow_bytess
                row['flow_packets_s'] if not pd.isna(row['flow_packets_s']) else 0.0, # Changed from flow_packetss
                row['fwd_packet_length_mean'],
                row['bwd_packet_length_mean'],
                1.0 if row['label'] != 'BENIGN' else 0.0  # Risk score
            ][:self.config.security_feature_dim]
            features.append(feature_row)
        self.logger.info(f"Extracted features for {len(features)} events")
        return features