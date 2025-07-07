# src/data_processors/optc_relationship_extractor.py

import pandas as pd
import logging
import os
from typing import List, Any, Dict, Tuple, Optional
from .relationship_extractor import RelationshipExtractor # Inherit from the base RelationshipExtractor

class OpTCRelationshipExtractor(RelationshipExtractor): # Inherit from base for common config/logging
    """
    Specialized relationship extractor for OpTC system-level provenance data.
    Focuses on extracting causal relationships between system events based on
    process lineage, file I/O, and host-level network interactions.
    """
    
    def __init__(self, config: Any):
        super().__init__(config) # Initializes logger and config
        
        # Define default time window for temporal causality (e.g., in seconds)
        # This can be configured in NeuralConfig.max_attack_step_gap
        self.causal_time_window_s = getattr(self.config, 'max_attack_step_gap', 3600) # Default to 1 hour
        self.logger = logging.getLogger(__name__)

    def extract_relationships(self, events_df: pd.DataFrame) -> List[List[int]]:
        """
        Extracts causal relationships (edges) from OpTC system events within a sequence.
        Relationships are represented as [source_event_index, target_event_index].
        
        Args:
            events_df (pd.DataFrame): A DataFrame of OpTC events for a single sequence,
                                      expected to be sorted by timestamp within subject_id.
        Returns:
            List[List[int]]: A list of [source_index, target_index] pairs representing causal edges.
        """
        relationships = []
        
        # Ensure essential columns are present
        required_cols = ['event_id', 'timestamp', 'subject_id', 'object_id', 'action', 
                         'object_type', 'pid', 'ppid', 'file_path', 
                         'network_src_addr', 'network_dst_addr']
        
        for col in required_cols:
            if col not in events_df.columns:
                self.logger.warning(f"Missing required column for relationship extraction: {col}. Skipping some causality types.")
                # Fill with dummy/safe values if missing to avoid immediate crash
                if col in ['pid', 'ppid']: events_df[col] = events_df[col].fillna(-1)
                else: events_df[col] = events_df[col].fillna('UNKNOWN')

        # Group by subject_id (process) to find intra-process causal chains
        # Events are expected to be sorted by timestamp within subject_id by OpTCProcessor.
        for subject_id, group in events_df.groupby('subject_id'):
            group_sorted = group.sort_values('timestamp').reset_index(drop=True)
            
            for i in range(len(group_sorted) - 1):
                current_event = group_sorted.iloc[i]
                next_event = group_sorted.iloc[i + 1]
                
                # Convert timestamps to datetime if they aren't already
                ts_curr = pd.to_datetime(current_event['timestamp'])
                ts_next = pd.to_datetime(next_event['timestamp'])

                # Skip if timestamps are invalid or in wrong order (should be handled by processor)
                if pd.isna(ts_curr) or pd.isna(ts_next) or ts_next <= ts_curr:
                    continue

                time_diff_s = (ts_next - ts_curr).total_seconds()
                
                # Temporal proximity (all events by same process within a window)
                if time_diff_s <= self.causal_time_window_s:
                    relationships.append([current_event.name, next_event.name]) # Use original DataFrame index as node ID
                
                # --- Specific Causal Patterns (more explicit provenance) ---
                # 1. Process Spawning (parent->child)
                # Check if next_event is a process creation where current_event is its parent
                if next_event['object_type'] == 'PROCESS' and next_event['action'] == 'CREATE' and \
                   next_event['ppid'] == current_event['pid'] and next_event['pid'] != -1: # Ensures non-unknown PID/PPID
                    relationships.append([current_event.name, next_event.name])
                    self.logger.debug(f"Process spawn: {current_event['process_name']} ({current_event['pid']}) -> {next_event['process_name']} ({next_event['pid']})")

                # 2. File I/O Flow (Write -> Read to same file, potentially different processes)
                # This requires looking up file path and matching timestamps.
                # This needs to be done across different subjects, so it's handled differently than groupby(subject_id).
                # For now, this extractor focuses on within-sequence, general causality.
                # More complex inter-process file I/O causality is often handled at the MultiLevelAttackGraphBuilder.
                # (You implemented a version of this in MultiLevelAttackGraphBuilder._extract_system_causality)

                # 3. Host-level Network Interaction (Process -> Network Flow)
                # If a process event involves network activity, link to a subsequent network object.
                # This depends on your definition of network events in eCAR.
                # If an 'action' is 'CONNECT'/'BIND' for an 'object_type' not 'NETFLOW', it's process-network.
                if current_event['action'] in ['CONNECT', 'BIND', 'ACCEPT', 'SEND', 'RECV'] and \
                   current_event['object_type'] != 'NETFLOW': # Event indicates network interaction by this process
                    # Look for a subsequent NETFLOW event by the same host/IPs if within window
                    # This is better handled by cross-level correlation via ecar-bro or heuristics in SystemEntityManager
                    pass # Keep this for cross-level if not from ecar-bro directly
        
        # Deduplicate relationships (as temporal proximity might create duplicates with explicit rules)
        deduplicated_relationships = list(set(tuple(rel) for rel in relationships))
        self.logger.info(f"Extracted {len(deduplicated_relationships)} relationships for subject {subject_id}.")
        return [list(rel) for rel in deduplicated_relationships]

    def extract_true_edges(self, events: pd.DataFrame) -> List[List[int]]:
        """
        Extracts ground truth edges based on a simplified definition for OpTC.
        This would ideally come from the OpTCRedTeamGroundTruth.pdf for specific attack paths.
        For Phase 1, a simple heuristic (e.g., sequential attack events by same process) is used.
        """
        true_edges = []
        
        # Rely on 'attack_presence' and 'subject_id' created by OpTCProcessor
        if 'attack_presence' not in events.columns or 'subject_id' not in events.columns:
            self.logger.warning("Missing 'attack_presence' or 'subject_id' for true edge extraction. Returning empty list.")
            return []

        # Group by subject_id (process) and look for sequential attack events
        for subject_id, group in events.groupby('subject_id'):
            group_sorted = group.sort_values('timestamp').reset_index(drop=True)
            for i in range(len(group_sorted) - 1):
                current_event = group_sorted.iloc[i]
                next_event = group_sorted.iloc[i + 1]
                
                # If both are classified as attack and are sequential within the same process
                if current_event['attack_presence'] == 1 and next_event['attack_presence'] == 1:
                    # And if they are within a reasonable time window
                    ts_curr = pd.to_datetime(current_event['timestamp'])
                    ts_next = pd.to_datetime(next_event['timestamp'])
                    if pd.isna(ts_curr) or pd.isna(ts_next) or ts_next <= ts_curr:
                        continue
                    time_diff_s = (ts_next - ts_curr).total_seconds()
                    if time_diff_s <= self.causal_time_window_s: # Use the same causal window
                        true_edges.append([current_event.name, next_event.name]) # Use original DataFrame index

        self.logger.info(f"Extracted {len(true_edges)} true edges from {len(events)} events (OpTC heuristic).")
        return true_edges
