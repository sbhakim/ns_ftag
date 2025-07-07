# src/data_processors/optc_feature_extractor.py

import pandas as pd
import logging
import numpy as np
from typing import List, Any, Dict
from .feature_extractor import FeatureExtractor # Inherit from the base FeatureExtractor

class OpTCFeatureExtractor(FeatureExtractor): # Inherit from base for common config/logging
    """
    Specialized feature extractor for OpTC system-level provenance data (eCAR format).
    Converts various event attributes into numerical features for neural networks.
    """
    
    def __init__(self, config: Any):
        super().__init__(config) # Initializes logger and config
        
        # Define the set of numerical features to extract from the preprocessed DataFrame
        # These should align with config.security_feature_dim
        self.feature_columns = [
            # Event-level features
            'event_duration_ms', # Derived: time from previous event in sequence by same process
            'num_properties',    # Derived: count of key-value pairs in 'properties_json'
            
            # Process-related features
            'pid', 'ppid', 'tid',
            
            # File-related features
            'file_depth',        # Derived: depth of file_path
            
            # Network-related features (host-level perspective)
            'network_src_port', 'network_dst_port',
            
            # Action/Syscall features (often one-hot encoded or embedded later)
            # For now, numerical representation if possible (e.g., return_value)
            'return_value_numeric', # Derived: converted from 'return_value'
            
            # Label-related features (used as target in LabelExtractor, but also as input feature here for risk)
            'attack_presence', # Binary: 1 if action/syscall maps to an attack, 0 otherwise
        ]
        
        # Ensure config.security_feature_dim is consistent with the number of features you plan to use
        # If config.security_feature_dim is set, it will slice this list.
        if len(self.feature_columns) < self.config.security_feature_dim:
            self.logger.warning(f"Configured security_feature_dim ({self.config.security_feature_dim}) is larger than defined OpTC features ({len(self.feature_columns)}). Will pad with zeros.")
        elif len(self.feature_columns) > self.config.security_feature_dim:
             self.logger.warning(f"Defined OpTC features ({len(self.feature_columns)}) is larger than configured security_feature_dim ({self.config.security_feature_dim}). Will truncate feature list.")
             self.feature_columns = self.feature_columns[:self.config.security_feature_dim]


    def extract_security_features(self, events_df: pd.DataFrame) -> List[List[float]]:
        """
        Extracts numerical security features from a DataFrame of OpTC events.
        
        Args:
            events_df (pd.DataFrame): A DataFrame of OpTC events for a single sequence,
                                      expected to be sorted by timestamp within subject_id.
        Returns:
            List[List[float]]: A list of feature vectors, one for each event.
        """
        features_list = []
        
        # Ensure required base columns are present after OpTCProcessor
        required_base_cols = [
            'timestamp', 'subject_id', 'pid', 'ppid', 'tid', 'action', 'object_type',
            'file_path', 'network_src_addr', 'network_dst_addr', 'network_src_port', 'network_dst_port',
            'return_value', 'properties_json', 'attack_presence'
        ]
        
        for col in required_base_cols:
            if col not in events_df.columns:
                self.logger.error(f"Missing base column for feature extraction: {col}. Features may be incomplete.")
                # Add dummy column to avoid crash if necessary
                if col in ['pid', 'ppid', 'tid', 'network_src_port', 'network_dst_port', 'attack_presence']:
                    events_df[col] = events_df[col].fillna(-1).astype(int)
                else:
                    events_df[col] = events_df[col].fillna('UNKNOWN').astype(str)

        # --- Derived Features ---
        # Event duration from previous event by same process
        events_df['timestamp_ms'] = events_df['timestamp'].astype(np.int64) // 10**6 # Convert to milliseconds
        events_df['prev_timestamp_ms'] = events_df.groupby('subject_id')['timestamp_ms'].shift(1)
        events_df['event_duration_ms'] = (events_df['timestamp_ms'] - events_df['prev_timestamp_ms']).fillna(0).astype(int)

        # Number of properties
        events_df['num_properties'] = events_df['properties_json'].apply(
            lambda x: len(json.loads(x)) if pd.notna(x) and x != 'UNKNOWN' else 0
        )
        
        # File path depth
        events_df['file_depth'] = events_df['file_path'].apply(
            lambda x: len(os.path.normpath(x).split(os.sep)) if isinstance(x, str) and x != 'UNKNOWN' else 0
        )
        
        # Numeric return value
        events_df['return_value_numeric'] = pd.to_numeric(events_df['return_value'], errors='coerce').fillna(0).astype(float)
        
        # --- Collect features for each event ---
        for index, row in events_df.iterrows():
            feature_vector = []
            for feature_col in self.feature_columns:
                value = row.get(feature_col, 0) # Get value, default to 0 if column is missing (after initial checks)
                
                # Handle specific type conversions if needed for the feature list
                if feature_col in ['pid', 'ppid', 'tid', 'network_src_port', 'network_dst_port', 'event_duration_ms', 'num_properties', 'file_depth']:
                    feature_vector.append(float(value))
                elif feature_col == 'return_value_numeric':
                    feature_vector.append(float(value))
                elif feature_col == 'attack_presence':
                    feature_vector.append(float(value))
                else: # Fallback for any other unexpected features, should be handled by self.feature_columns definition
                    feature_vector.append(0.0) # Default to 0.0 if not explicitly handled
            
            # Pad or truncate to match config.security_feature_dim
            if len(feature_vector) < self.config.security_feature_dim:
                feature_vector.extend([0.0] * (self.config.security_feature_dim - len(feature_vector)))
            elif len(feature_vector) > self.config.security_feature_dim:
                feature_vector = feature_vector[:self.config.security_feature_dim]

            features_list.append(feature_vector)
            
        self.logger.info(f"Extracted {len(features_list)} feature vectors, each of dimension {self.config.security_feature_dim}.")
        return features_list
