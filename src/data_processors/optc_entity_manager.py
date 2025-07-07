# src/data_processors/optc_entity_manager.py

import logging
import pandas as pd
from collections import defaultdict # FIXED: Missing import
from typing import Dict, List, Any, Optional, Tuple
from .system_entity_manager import SystemEntityManager # FIXED: Missing base class
# from .entity_manager import EntityManager # Not needed directly if inheriting from SystemEntityManager

class OpTCEntityManager(SystemEntityManager): # Inherit from SystemEntityManager for multi-level concepts
    """
    Specialized entity manager for OpTC system-level security events (eCAR format).
    Extracts and manages unique IDs for processes, files, syscalls, registry keys,
    and network endpoints from OpTC data.
    """
    
    def __init__(self):
        super().__init__() # Initializes base EntityManager and SystemEntityManager parts
        self.logger = logging.getLogger(__name__)

        # OpTC specific vocabularies (if different from generic SystemEntityManager)
        # For now, base SystemEntityManager's vocabs are directly used and populated.

    def extract_security_entities(self, events: pd.DataFrame) -> Dict[str, List[Any]]:
        """
        Extract primary security entities (process, file, network endpoint, registry) and actions (syscalls/actions)
        from OpTC eCAR events.
        
        This method replaces the generic extract_security_entities from base EntityManager
        with OpTC-specific entity parsing. It will be called by SecurityEventDataset.
        """
        required_cols = [
            'event_id', 'subject_id', 'object_id', 'action', 'object_type', 
            'hostname', 'pid', 'ppid', 'process_name', 'file_path', 
            'network_src_addr', 'network_dst_addr', 'network_src_port', 'network_dst_port', 
            'syscall', 'principal'
        ]
        
        # Ensure required columns exist, fill with UNKNOWN/defaults if missing (should be handled by OpTCProcessor)
        for col in required_cols:
            if col not in events.columns:
                self.logger.warning(f"Missing expected OpTC column for entity extraction: {col}. Filling with defaults.")
                if col in ['pid', 'ppid', 'network_src_port', 'network_dst_port']:
                    events[col] = -1
                else:
                    events[col] = 'UNKNOWN'
        
        # Dictionaries to hold extracted IDs for each entity type for the current batch/sequence
        extracted_ids_detailed = defaultdict(list)
        
        # Prepare lists for the simplified output format expected by SecurityEventDataset._process_event_sequence
        source_entity_ids_for_pipeline = []
        action_ids_for_pipeline = []
        
        for idx, row in events.iterrows():
            # 1. Primary Entity for Pipeline (e.g., the process performing the action)
            process_name_key = row['process_name'] if row['process_name'] != 'UNKNOWN' else f"pid_{row['pid']}"
            process_entity_id = self.get_process_id(process_name_key, str(row['pid']))
            source_entity_ids_for_pipeline.append(process_entity_id)

            # 2. Primary Action for Pipeline (e.g., the eCAR action verb)
            action_id = self.get_action_id(row['action']) # Using base EntityManager's get_action_id
            action_ids_for_pipeline.append(action_id)

            # 3. Detailed OpTC-specific Entity IDs (for MultiLevelAttackGraphBuilder)
            extracted_ids_detailed['event_unique_ids'].append(row['event_id'])
            extracted_ids_detailed['original_subject_ids'].append(row['subject_id']) # Actor UUID
            extracted_ids_detailed['original_object_ids'].append(row['object_id']) # Object UUID
            extracted_ids_detailed['original_hostnames'].append(row['hostname'])

            extracted_ids_detailed['process_entities'].append(process_entity_id) # Store here too for detailed access
            extracted_ids_detailed['syscall_ids'].append(self.get_syscall_id(row['syscall']))

            object_type_val = row['object_type'].upper()
            file_entity_id = 0
            network_entity_id = 0
            registry_entity_id = 0

            if object_type_val == 'FILE':
                file_entity_id = self.get_file_id(row['file_path'])
            elif object_type_val == 'REGISTRYKEY':
                registry_entity_id = self.get_registry_id(row['object_id'])
            elif object_type_val == 'NETFLOW':
                src_ep = f"{row['network_src_addr']}:{row['network_src_port']}"
                dst_ep = f"{row['network_dst_addr']}:{row['network_dst_port']}"
                network_entity_id = self.get_network_endpoint_id(f"{src_ep}-{dst_ep}")

            extracted_ids_detailed['file_entity_ids'].append(file_entity_id)
            extracted_ids_detailed['network_entity_ids'].append(network_entity_id)
            extracted_ids_detailed['registry_entity_ids'].append(registry_entity_id)
            
            # Store original PIDs/PPIDs for graph building
            extracted_ids_detailed['original_pids'].append(row['pid'])
            extracted_ids_detailed['original_ppids'].append(row['ppid'])

        self.logger.info(f"Extracted OpTC-specific entities for {len(events)} events.")
        
        # Combine into the required return format.
        # This dictionary's keys must be matched by SecurityEventDataset._process_event_sequence
        # It's a trade-off: either SecurityEventDataset becomes complex to unwrap,
        # or we flatten/select here. Given the existing pipeline, we adapt.
        
        # The 'source_entity_ids' and 'action_ids' will be directly used for embeddings.
        # The 'optc_metadata' dictionary (NEW) will carry all the rich, granular OpTC info
        # that the MultiLevelAttackGraphBuilder and its components will need.
        return {
            'source_entity_ids': source_entity_ids_for_pipeline,
            'action_ids': action_ids_for_pipeline,
            'target_entity_ids': source_entity_ids_for_pipeline, # Dummy, often same as source for system events
            
            # NEW: Store all granular OpTC data in a dedicated metadata dict
            'optc_metadata': {k: list(v) for k, v in extracted_ids_detailed.items()}
        }

    # get_vocab_sizes is inherited from SystemEntityManager, and it now correctly aggregates
    # all specialized vocabularies. This is called by SecurityEventDataset.
