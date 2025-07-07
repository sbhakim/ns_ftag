# src/data_processors/optc_processor.py

import pandas as pd
import os
import logging
from .base_processor import BaseSecurityEventProcessor
from typing import Any, Dict, List
import json # Explicitly import json module
import sys # For better stream handler setup
import numpy as np # For np.inf, np.nan

class OpTCProcessor(BaseSecurityEventProcessor):
    def __init__(self, config: Any):
        super().__init__(config)
        self.logger = logging.getLogger(__name__)
        # Ensure only one set of handlers to prevent duplicate log messages
        if not self.logger.handlers:
            self.logger.setLevel(logging.INFO)
            file_handler = logging.FileHandler(os.path.join(self.config.log_dir, 'optc_processor.log'))
            stream_handler = logging.StreamHandler(sys.stdout)
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)
        self.logger.info("OpTCProcessor initialized.")

    def load_data(self, file_path: str) -> pd.DataFrame:
        self.logger.info(f"Attempting to load OpTC data from base path: {file_path}")

        # Construct the full path to the ecar JSON files using the config setting
        ecar_full_path = os.path.join(file_path, self.config.optc_ecar_path)
        self.logger.info(f"Searching for OpTC JSON files in: {ecar_full_path}")
        
        all_data_frames = []
        json_files_in_dir = []

        if not os.path.isdir(ecar_full_path):
            self.logger.error(f"OpTC ecar directory not found: {ecar_full_path}")
            raise FileNotFoundError(f"OpTC ecar directory not found: {ecar_full_path}")

        # Collect all JSON files in the target directory
        for filename in os.listdir(ecar_full_path):
            if filename.endswith('.json'):
                json_files_in_dir.append(os.path.join(ecar_full_path, filename))
        
        if not json_files_in_dir:
            self.logger.warning(f"No .json files found directly in the specified OpTC directory: {ecar_full_path}.")
            self.logger.info("Attempting recursive search for JSON files.")
            # If no files found directly, try a recursive walk for nested JSONs
            for root, _, files in os.walk(ecar_full_path):
                for file in files:
                    if file.endswith('.json'):
                        json_files_in_dir.append(os.path.join(root, file))
            
            if not json_files_in_dir:
                self.logger.error(f"No .json files found recursively in {ecar_full_path}. Returning empty DataFrame.")
                return pd.DataFrame() # No JSONs found at all
            else:
                self.logger.info(f"Found {len(json_files_in_dir)} JSON files via recursive search.")

        # Process each found JSON file
        for json_path in json_files_in_dir:
            self.logger.info(f"Attempting to load: {os.path.basename(json_path)}")
            
            df_temp = None
            # Attempt 1: Load as JSON Lines (typical for streaming data like eCAR)
            try:
                df_temp = pd.read_json(json_path, lines=True)
                self.logger.info(f"Successfully loaded {len(df_temp)} records from {os.path.basename(json_path)} as JSON Lines.")
            except ValueError as ve: # pd.read_json raises ValueError if lines=True is wrong
                self.logger.warning(f"Failed to load {os.path.basename(json_path)} as JSON Lines: {ve}. Trying as single JSON object.")
                # Attempt 2: Load as a single JSON object (if lines=True failed)
                try:
                    with open(json_path, 'r') as f:
                        data = json.load(f)
                    # If it's a list of dicts, it's directly DataFrame convertible
                    df_temp = pd.DataFrame(data)
                    self.logger.info(f"Successfully loaded {len(df_temp)} records from {os.path.basename(json_path)} as a single JSON object/array.")
                except Exception as e:
                    self.logger.error(f"Failed to load {os.path.basename(json_path)} as single JSON object: {e}. Skipping this file.")
            except Exception as e:
                self.logger.error(f"Unexpected error loading {os.path.basename(json_path)}: {e}. Skipping this file.")
            
            if df_temp is not None and not df_temp.empty:
                all_data_frames.append(df_temp)

        if not all_data_frames:
            self.logger.error(f"No OpTC JSON data could be successfully loaded from any files in {ecar_full_path}. Check file formats and content.")
            return pd.DataFrame() # Return empty DataFrame if nothing loaded

        combined_df = pd.concat(all_data_frames, ignore_index=True)
        self.logger.info(f"Successfully loaded {len(combined_df)} total OpTC records from {len(json_files_in_dir)} JSON files.")
        return self.preprocess(combined_df) # Always call preprocess

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        initial_rows = len(df)
        self.logger.info(f"Starting preprocessing OpTC DataFrame with {initial_rows} rows.")

        if df.empty:
            self.logger.warning("Empty DataFrame received for preprocessing. Returning empty DataFrame.")
            return pd.DataFrame()

        # --- Step 1: Flattening nested JSON (if 'datum' or other nested fields exist) ---
        # This part is highly dependent on your exact JSON schema.
        unpacked_data = []
        if 'datum' in df.columns:
            self.logger.info("Unpacking 'datum' column...")
            try:
                if not df['datum'].empty and isinstance(df['datum'].iloc[0], dict):
                    first_datum_keys = list(df['datum'].iloc[0].keys())
                    event_key = next((k for k in first_datum_keys if 'Event' in k), None)
                    subject_key = next((k for k in first_datum_keys if 'Subject' in k), None)
                    principle_key = next((k for k in first_datum_keys if 'Principal' in k), None)
                    
                    if event_key:
                        self.logger.info(f"Found event key '{event_key}' in 'datum'. Normalizing events.")
                        unpacked_data = pd.json_normalize(df['datum'].apply(lambda x: x.get(event_key) if isinstance(x, dict) else {}))
                    elif subject_key:
                        self.logger.info(f"Found subject key '{subject_key}' in 'datum'. Normalizing subjects.")
                        unpacked_data = pd.json_normalize(df['datum'].apply(lambda x: x.get(subject_key) if isinstance(x, dict) else {}))
                    elif principle_key:
                        self.logger.info(f"Found principle key '{principle_key}' in 'datum'. Normalizing principles.")
                        unpacked_data = pd.json_normalize(df['datum'].apply(lambda x: x.get(principle_key) if isinstance(x, dict) else {}))
                    else:
                        self.logger.warning("No specific CDM schema key found in 'datum'. Normalizing 'datum' directly (might be flat).")
                        unpacked_data = pd.json_normalize(df['datum'])
                else:
                    self.logger.warning("'datum' column does not contain dictionaries. Skipping unpacking.")
                    unpacked_data = pd.DataFrame()
                
                if not unpacked_data.empty:
                    unpacked_data.columns = [col.replace('.', '_').lower() for col in unpacked_data.columns]
                    df = pd.concat([df.drop(columns=['datum']), unpacked_data], axis=1)
                    self.logger.info(f"Successfully unpacked 'datum' column. New DataFrame shape: {df.shape}")
                else:
                    self.logger.warning("Unpacked 'datum' resulted in empty DataFrame or no data. Original DataFrame used.")

            except Exception as e:
                self.logger.error(f"Error unpacking 'datum' column: {e}. Proceeding with original DataFrame structure.", exc_info=True)

        # --- Step 2: General Column Cleaning and Standardization ---
        df.columns = df.columns.str.lower().str.replace('.', '_', regex=False)
        self.logger.info(f"Columns after initial cleaning: {list(df.columns[:10])}...")

        # --- Step 3: Renaming to standard internal names ---
        # This mapping is CRUCIAL. You must adjust this to match YOUR ACTUAL OpTC JSON fields
        column_rename_map = {
            'uuid': 'event_id',
            'event_timestamp_nanos': 'timestamp',
            'datum_event_type': 'action',
            'datum_subject_uuid': 'subject_id',
            'datum_predicateobject_uuid': 'object_id',
            'datum_predicateobject_path': 'file_path',
            'datum_subject_properties_map_cmdline': 'process_name',
            'datum_subject_pid': 'pid',
            'datum_subject_ppid': 'ppid',
            'datum_netflow_localaddress': 'network_src_addr',
            'datum_netflow_remoteaddress': 'network_dst_addr',
            'datum_netflow_localport': 'network_src_port',
            'datum_netflow_remoteport': 'network_dst_port',
            'datum_syscall': 'syscall',
            'datum_principal_uuid': 'principal_id',
            'datum_host_hostname': 'hostname',
            'datum_thread_tid': 'tid',
            'datum_return_value': 'return_value',
            'datum_properties': 'properties_json'
        }
        df = df.rename(columns=column_rename_map, errors='ignore')
        self.logger.info(f"Columns after renaming to standard names: {list(df.columns[:10])}...")

        # --- Step 4: Apply Dataset Subsetting (New Feature) ---
        if self.config.subset_config and self.config.subset_config.get('enabled', False):
            df = self._apply_subset_filter(df)
            if df.empty:
                self.logger.warning("DataFrame is empty after subset filtering. Returning empty DataFrame.")
                return pd.DataFrame()
        
        # --- Step 5: Ensure essential columns exist and fill with UNKNOWN/defaults ---
        essential_cols_for_pipeline = [
            'event_id', 'subject_id', 'object_id', 'action', 'timestamp',
            'file_path', 'process_name', 'pid', 'ppid', 'hostname',
            'network_src_addr', 'network_dst_addr', 'network_src_port', 'network_dst_port',
            'syscall', 'principal_id', 'object_type', 'tid', 'return_value', 'properties_json'
        ]
        
        for col in essential_cols_for_pipeline:
            if col not in df.columns:
                self.logger.warning(f"Missing essential OpTC column '{col}'. Filling with default/UNKNOWN.")
                if col in ['pid', 'ppid', 'network_src_port', 'network_dst_port', 'tid']:
                    df[col] = -1
                elif col == 'timestamp': # Timestamp handled separately
                    df[col] = pd.NaT 
                else:
                    df[col] = 'UNKNOWN'
            # Ensure proper dtypes for known numeric columns that might be 'UNKNOWN'
            if col in ['pid', 'ppid', 'network_src_port', 'network_dst_port', 'tid', 'return_value']:
                df[col] = pd.to_numeric(df[col], errors='coerce').fillna(-1).astype(int)
            elif col == 'timestamp':
                pass # Handled in step 6
            else:
                df[col] = df[col].astype(str)
        self.logger.info(f"Columns after ensuring essential cols: {list(df.columns[:10])}...")


        # --- Step 6: Timestamp Conversion and Handling ---
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], unit='ns', errors='coerce')

            if df['timestamp'].isna().any():
                self.logger.warning(f"{df['timestamp'].isna().sum()} invalid timestamps after nanosecond conversion. Attempting string parsing.")
                # If your original timestamp string column existed and was renamed, use it here if needed
                # df['timestamp'] = df['timestamp'].fillna(pd.to_datetime(df['original_timestamp_string_col'], errors='coerce'))
                
                common_formats = [
                    '%Y-%m-%dT%H:%M:%S.%fZ',
                    '%Y-%m-%dT%H:%M:%S.%f',
                    '%Y-%m-%dT%H:%M:%SZ',
                    '%Y-%m-%dT%H:%M:%S',
                    '%Y-%m-%d %H:%M:%S.%f',
                    '%Y-%m-%d %H:%M:%S'
                ]
                for fmt in common_formats:
                    if df['timestamp'].isna().any():
                        df.loc[df['timestamp'].isna(), 'timestamp'] = pd.to_datetime(
                            df.loc[df['timestamp'].isna(), 'timestamp'].astype(str), format=fmt, errors='coerce'
                        )
                        self.logger.debug(f"Tried format '{fmt}', NaNs remaining: {df['timestamp'].isna().sum()}")
                
                if df['timestamp'].isna().any():
                    num_invalid_final = df['timestamp'].isna().sum()
                    self.logger.warning(f"Still {num_invalid_final} invalid timestamps after all attempts. Filling with unique sequential dummy values.")
                    nan_indices = df.index[df['timestamp'].isna()]
                    df.loc[nan_indices, 'timestamp'] = pd.to_datetime(pd.Series(range(len(nan_indices))), unit='s', origin='unix') + pd.to_timedelta(pd.Series(nan_indices), unit='ms')
                    self.logger.info("Filled remaining invalid timestamps with unique sequential dummy values.")
        else:
            self.logger.critical("No 'timestamp' column found after preprocessing. This is critical for temporal ordering.")
            raise ValueError("Timestamp column missing. Cannot proceed.")
        
        # --- Step 7: Derive 'object_type' from event structure or heuristics ---
        if 'object_type' not in df.columns:
            self.logger.info("Deriving 'object_type' column based on 'action' or other fields.")
            def derive_object_type(row):
                action = str(row['action']).lower()
                if 'file' in action or 'read' in action or 'write' in action or 'create' in action or 'delete' in action:
                    if 'file_path' in row and str(row['file_path']) != 'UNKNOWN':
                        return 'FILE'
                if 'netflow' in action or 'connect' in action or 'bind' in action or 'accept' in action:
                    if ('network_src_addr' in row and str(row['network_src_addr']) != 'UNKNOWN') or \
                       ('network_dst_addr' in row and str(row['network_dst_addr']) != 'UNKNOWN'):
                        return 'NETFLOW'
                if 'process' in action or 'exec' in action or 'fork' in action:
                    if 'process_name' in row and str(row['process_name']) != 'UNKNOWN':
                        return 'PROCESS'
                if 'registry' in action or 'reg' in action:
                    return 'REGISTRYKEY'
                return 'UNKNOWN'
            df['object_type'] = df.apply(derive_object_type, axis=1)
            self.logger.info(f"Derived 'object_type' distribution: {df['object_type'].value_counts().to_dict()}")


        # --- Step 8: Ground Truth Labeling (CRITICAL for supervised learning) ---
        if 'attack_presence' not in df.columns:
            self.logger.warning("No 'attack_presence' column found. Adding dummy 'attack_presence' (all benign).")
            df['attack_presence'] = 0
        
        if 'label' not in df.columns:
            df['label'] = df['attack_presence'].apply(lambda x: 'BENIGN' if x == 0 else 'ATTACK')
            self.logger.warning("Dummy 'label' column created. Implement real ground truth mapping using OpTCRedTeamGroundTruth.pdf.")
        
        df['label'] = df['label'].astype(str)
        self.logger.info(f"Final label distribution: {df['label'].value_counts().to_dict()}")


        # --- Step 9: Final Cleanup and Sorting ---
        # Drop rows with NaNs in absolutely critical columns after all imputation/derivation.
        final_critical_cols = ['event_id', 'subject_id', 'action', 'timestamp', 'label', 'object_type']
        initial_rows_before_final_dropna = len(df)
        # Ensure all columns in final_critical_cols exist in df before dropping
        existing_critical_cols = [col for col in final_critical_cols if col in df.columns]
        if len(existing_critical_cols) < len(final_critical_cols):
            self.logger.warning(f"Some final critical columns for dropna are missing from DataFrame: {set(final_critical_cols) - set(existing_critical_cols)}. Proceeding with existing.")

        df.dropna(subset=existing_critical_cols, inplace=True)
        dropped_rows_final = initial_rows_before_final_dropna - len(df)
        if dropped_rows_final > 0:
            self.logger.warning(f"Dropped {dropped_rows_final} rows with NaN in final critical columns after all preprocessing steps.")
            
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        self.logger.info("DataFrame sorted by timestamp.")

        self.logger.info(f"Final preprocessed OpTC DataFrame: {len(df)} rows, {len(df.columns)} columns.")
        return df

    def _apply_subset_filter(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies subset selection based on configuration.
        Assumes 'timestamp' column is already in datetime format.
        """
        subset_config = self.config.subset_config
        if not subset_config or not subset_config.get('enabled', False):
            self.logger.info("Subset filtering is disabled. Using full DataFrame.")
            return df

        initial_rows = len(df)
        self.logger.info(f"Applying subset filter to {initial_rows} events.")

        # Temporal filtering
        if subset_config.get('type') == 'temporal_window':
            start_date_str = subset_config.get('start_date')
            end_date_str = subset_config.get('end_date')
            
            if start_date_str and end_date_str and 'timestamp' in df.columns:
                try:
                    start_ts = pd.to_datetime(start_date_str)
                    end_ts = pd.to_datetime(end_date_str)
                    df = df[(df['timestamp'] >= start_ts) & (df['timestamp'] <= end_ts)].copy()
                    self.logger.info(f"Temporal filter applied ({start_ts} to {end_ts}). Events remaining: {len(df)}")
                except Exception as e:
                    self.logger.error(f"Error applying temporal filter: {e}. Skipping temporal filter.", exc_info=True)
        
        # Host sampling
        max_hosts = subset_config.get('max_hosts')
        if max_hosts and 'hostname' in df.columns and df['hostname'].nunique() > max_hosts:
            unique_hosts = df['hostname'].unique()
            # For consistent subsetting, sort hosts before selection
            sorted_hosts = np.sort(unique_hosts)
            selected_hosts = sorted_hosts[:max_hosts]
            df = df[df['hostname'].isin(selected_hosts)].copy()
            self.logger.info(f"Host sampling applied. Selected {len(selected_hosts)} hosts. Events remaining: {len(df)}")

        # Random sampling
        sample_fraction = subset_config.get('sample_fraction')
        if sample_fraction is not None and 0 < sample_fraction < 1:
            df = df.sample(frac=sample_fraction, random_state=42).copy()
            self.logger.info(f"Random sample applied (frac={sample_fraction}). Events remaining: {len(df)}")

        # Max events limit (applied after other filters)
        max_events = subset_config.get('max_events')
        if max_events is not None and len(df) > max_events:
            df = df.head(max_events).copy() # Use .head() or .sample() if random is desired
            self.logger.info(f"Max events limit applied. Retained {len(df)} events.")

        self.logger.info(f"Subset filtering: {len(df)} events retained from {initial_rows} original (post-loading, pre-final-prep).")
        return df