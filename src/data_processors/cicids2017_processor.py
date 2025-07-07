# src/data_processors/cicids2017_processor.py

import pandas as pd
import os
import numpy as np
import glob
import logging
from .base_processor import BaseSecurityEventProcessor

class CICIDS2017Processor(BaseSecurityEventProcessor):
    def __init__(self, config):
        super().__init__(config)
        
        os.makedirs(self.config.log_dir, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            file_handler = logging.FileHandler(os.path.join(self.config.log_dir, 'cicids2017_processor.log'))
            stream_handler = logging.StreamHandler()
            formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
            file_handler.setFormatter(formatter)
            stream_handler.setFormatter(formatter)
            self.logger.addHandler(file_handler)
            self.logger.addHandler(stream_handler)
            self.logger.setLevel(logging.INFO)

        if not hasattr(self.config, 'possible_encodings'):
            self.config.possible_encodings = ['utf-8', 'latin1', 'ISO-8859-1', 'windows-1252']

    def _standardize_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Applies aggressive column cleaning and standardization for CICIDS2017 dataset.
        This includes stripping whitespace, lowercasing, removing non-alphanumeric characters,
        and renaming to a consistent format.
        """
        original_cols = df.columns.tolist()
        
        df.columns = df.columns.str.strip().str.replace(' ', '').str.lower()
        df.columns = df.columns.str.replace(r'[^a-z0-9]', '', regex=True)
        
        column_mapping = {
            'flowid': 'flow_id',
            'sourceip': 'source_ip',
            'sourceport': 'source_port', 
            'destinationip': 'destination_ip',
            'destinationport': 'destination_port',
            'protocol': 'protocol',
            'timestamp': 'timestamp',
            'label': 'label',
            # Feature columns - added for completeness
            'totalfwdpackets': 'total_fwd_packets',
            'totalbackwardpackets': 'total_backward_packets',
            'totallengthoffwdpackets': 'total_length_of_fwd_packets',
            'totallengthofbwdpackets': 'total_length_of_bwd_packets',
            'flowduration': 'flow_duration',
            'flowbytess': 'flow_bytes_s',
            'flowpacketss': 'flow_packets_s',
            'fwdpacketlengthmean': 'fwd_packet_length_mean',
            'bwdpacketlengthmean': 'bwd_packet_length_mean',
            'minpacketlength': 'min_packet_length',
            'maxpacketlength': 'max_packet_length',
            'packetlengthmean': 'packet_length_mean',
            'packetlengthstd': 'packet_length_std',
            'packetlengthvariance': 'packet_length_variance',
            'finflagcount': 'fin_flag_count',
            'synflagcount': 'syn_flag_count',
            'rstflagcount': 'rst_flag_count',
            'pshflagcount': 'psh_flag_count',
            'ackflagcount': 'ack_flag_count',
            'urgflagcount': 'urg_flag_count',
            'cweflagcount': 'cwe_flag_count',
            'eceflagcount': 'ece_flag_count',
            'downupratio': 'down_up_ratio',
            'averagepacketsize': 'average_packet_size',
            'avgfwdsegmentsize': 'avg_fwd_segment_size',
            'avgbwdsegmentsize': 'avg_bwd_segment_size',
            'fwdheaderlength1': 'fwd_header_length_1',
            'fwdavgbytesbulk': 'fwd_avg_bytes_bulk',
            'fwdavgpacketsbulk': 'fwd_avg_packets_bulk',
            'fwdavgbulkrate': 'fwd_avg_bulk_rate',
            'bwdavgbytesbulk': 'bwd_avg_bytes_bulk',
            'bwdavgpacketsbulk': 'bwd_avg_packets_bulk',
            'bwdavgbulkrate': 'bwd_avg_bulk_rate',
            'subflowfwdpackets': 'subflow_fwd_packets',
            'subflowfwdbytes': 'subflow_fwd_bytes',
            'subflowbwdpackets': 'subflow_bwd_packets',
            'subflowbwdbytes': 'subflow_bwd_bytes',
            'init_win_bytes_forward': 'init_win_bytes_forward',
            'init_win_bytes_backward': 'init_win_bytes_backward',
            'act_data_pkt_fwd': 'act_data_pkt_fwd',
            'min_seg_size_forward': 'min_seg_size_forward',
            'activemean': 'active_mean',
            'activestd': 'active_std',
            'activemax': 'active_max',
            'activemin': 'active_min',
            'idlemean': 'idle_mean',
            'idlestd': 'idle_std',
            'idlemax': 'idle_max',
            'idlemin': 'idle_min',
            'fwdheaderlength': 'fwd_header_length',
            'bwdheaderlength': 'bwd_header_length',
            'fwdiattotal': 'fwd_iat_total',
            'fwdiatmean': 'fwd_iat_mean',
            'fwdiatstd': 'fwd_iat_std',
            'fwdiatmax': 'fwd_iat_max',
            'fwdiatmin': 'fwd_iat_min',
            'bwdiattotal': 'bwd_iat_total',
            'bwdiatmean': 'bwd_iat_mean',
            'bwdiatstd': 'bwd_iat_std',
            'bwdiatmax': 'bwd_iat_max',
            'bwdiatmin': 'bwd_iat_min',
            'fwdpshflags': 'fwd_psh_flags',
            'bwdpshflags': 'bwd_psh_flags',
            'fwdurgflags': 'fwd_urg_flags',
            'bwdurgflags': 'bwd_urg_flags',
            'fwdpacketss': 'fwd_packets_s',
            'bwdpacketss': 'bwd_packets_s',
        }
        
        df = df.rename(columns=column_mapping, errors='ignore')
        
        essential_cols = ['flow_id', 'source_ip', 'destination_ip', 'source_port', 'destination_port', 'protocol', 'timestamp', 'label']
        found_cols = [col for col in essential_cols if col in df.columns]
        missing_cols = [col for col in essential_cols if col not in df.columns]
        
        self.logger.info(f"Original columns (first 10): {original_cols[:10]}")
        self.logger.info(f"After cleaning and initial renaming (first 10): {df.columns.tolist()[:10]}")
        self.logger.info(f"Found essential cols: {found_cols}")
        self.logger.info(f"Missing essential cols: {missing_cols}")
        
        # Attempt to fill missing essential columns with defaults rather than failing immediately.
        if len(found_cols) < 4: # Need at least source_ip, dest_ip, protocol, label for basic flow analysis.
            self.logger.error(f"Too few essential columns found for basic flow analysis: {found_cols}. Returning None.")
            return None
            
        for col in missing_cols:
            if col == 'source_port' or col == 'destination_port':
                df[col] = 0 # Default port
                self.logger.warning(f"Missing critical column '{col}' - filling with 0.")
            elif col == 'flow_id':
                df[col] = range(len(df)) # Sequential IDs
                self.logger.warning(f"Missing critical column '{col}' - filling with sequential IDs.")
            elif col == 'source_ip' or col == 'destination_ip':
                self.logger.warning(f"Missing critical column '{col}' - using placeholder 'unknown_{col}'.")
                df[col] = f"unknown_{col}" # Placeholder IP
            elif col == 'protocol': # This is also essential, but might be missing
                df[col] = 'unknown'
                self.logger.warning(f"Missing critical column '{col}' - filling with 'unknown'.")
            elif col == 'timestamp':
                self.logger.warning(f"Timestamp column '{col}' not found after standardizing - will attempt to create in preprocess.")
                df[col] = pd.NaT # Set to NaT, preprocess will handle it
            elif col == 'label':
                df[col] = 'BENIGN' # Default label
                self.logger.warning(f"Label column '{col}' not found - filling with 'BENIGN'.")
                
        return df

    def _load_single_csv(self, file_path: str) -> pd.DataFrame:
        """Load a single CSV file with encoding handling."""
        filename = os.path.basename(file_path)
        for encoding in self.config.possible_encodings:
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                self.logger.info(f"Successfully read {filename} with encoding: {encoding} ({len(df)} rows, {len(df.columns)} columns)")
                return self._standardize_columns(df)
            except UnicodeDecodeError as e:
                self.logger.warning(f"Failed to read {filename} with encoding {encoding}: {e}. Trying next.")
            except pd.errors.EmptyDataError:
                self.logger.warning(f"File {filename} is empty. Skipping.")
                return None
            except Exception as e:
                self.logger.error(f"General error loading {filename} with encoding {encoding}: {e}. Skipping this file.")
                return None
        self.logger.error(f"All attempted encodings failed for {filename}. This file could not be loaded.")
        return None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads CICIDS2017 CSV files dynamically from the specified path and combines them into a single DataFrame.
        """
        self.logger.info(f"Loading data from: {file_path}")
        file_path = os.path.normpath(file_path)

        # Construct the correct subdirectory path for CICIDS2017 CSVs
        # Assuming file_path is the parent directory (e.g., ".../data/datasets")
        # and CICIDS2017 CSVs are in "cic_ids2017/TrafficLabelling" relative to that.
        cicids_specific_path = os.path.join(file_path, 'cic_ids2017', 'TrafficLabelling')
        csv_files = sorted(glob.glob(os.path.join(cicids_specific_path, '*.[cC][sS][vV]')))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {cicids_specific_path}. Check path, permissions, or file extensions.")

        data_frames = []
        for full_path in csv_files:
            if not os.path.exists(full_path):
                self.logger.error(f"File {full_path} does not exist. Skipping.")
                continue
            df = self._load_single_csv(full_path)
            if df is not None and not df.empty:
                self.logger.info(f"Added {os.path.basename(full_path)} with {len(df)} rows to combine list.")
                data_frames.append(df)

        if not data_frames:
            raise ValueError("No valid CSV files were loaded. Check paths, file integrity, column names, and encodings.")

        df_combined = pd.concat(data_frames, ignore_index=True)
        self.logger.info(f"Combined DataFrame: {len(df_combined)} rows, {len(df_combined.columns)} columns")
        return self.preprocess(df_combined)

    def load_data_from_df(self, df: pd.DataFrame) -> pd.DataFrame:
        """Load data from an in-memory DataFrame for testing."""
        self.logger.info("Loading data from in-memory DataFrame")
        # Ensure that an in-memory DataFrame also goes through _standardize_columns
        standardized_df = self._standardize_columns(df.copy())
        if standardized_df is None:
            raise ValueError("In-memory DataFrame could not be standardized.")
        return self.preprocess(standardized_df)

    def _standardize_labels(self, df: pd.DataFrame) -> pd.DataFrame:
        """Standardize label values to handle case/whitespace variations."""
        if 'label' in df.columns:
            # Common CICIDS2017 label variations
            label_mapping = {
                'benign': 'BENIGN',
                'ddos': 'DDoS',  
                'portscan': 'PortScan',
                'port scan': 'PortScan',
                'web attack': 'Web Attack',
                'web attack – brute force': 'Web Attack',
                'web attack – sql injection': 'Web Attack',
                'web attack – xss': 'Web Attack',
                'infilteration': 'Infiltration',  # Common typo
                'infiltration': 'Infiltration',
                'bot': 'Bot',
                'botnet': 'Bot'
            }
            
            # Clean and map labels
            df['label'] = df['label'].astype(str).str.strip().str.lower()
            df['label'] = df['label'].map(label_mapping).fillna(df['label']) # map known, keep original for unknown
            
            # Convert common attack names to your expected format (e.g., "Ddos" to "DDoS")
            df['label'] = df['label'].str.title() # Convert to title case for consistency if not mapped
            # Special case for DDoS to ensure it's uppercase
            df['label'] = df['label'].replace('Ddos', 'DDoS')
            
            self.logger.info(f"Standardized labels: {df['label'].value_counts().to_dict()}")
        
        return df

    def _validate_feature_columns(self, df: pd.DataFrame) -> bool:
        """
        Validate that required feature columns exist and fill with zeros if missing.
        This ensures FeatureExtractor does not fail due to missing columns.
        """
        required_features = [
            'total_fwd_packets', 'total_backward_packets', 
            'total_length_of_fwd_packets', 'total_length_of_bwd_packets',
            'flow_duration', 'flow_bytes_s', 'flow_packets_s',
            'fwd_packet_length_mean', 'bwd_packet_length_mean'
        ]
        
        missing_features = [col for col in required_features if col not in df.columns]
        if missing_features:
            self.logger.warning(f"Missing feature columns for FeatureExtractor: {missing_features}")
            # Add dummy features if needed
            for col in missing_features:
                df[col] = 0.0
                self.logger.info(f"Added dummy column '{col}' with zeros to satisfy FeatureExtractor requirements.")
        
        return True # Return True if processing can continue

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs preprocessing steps: timestamp conversion, NaN/Inf handling, sorting, and diagnostics.
        It ensures essential columns are present and data types are consistent for downstream processing.
        """
        self.logger.info(f"Starting preprocessing for DataFrame with {len(df)} rows.")
        
        essential_cols = ['flow_id', 'source_ip', 'destination_ip', 'source_port', 'destination_port', 'protocol', 'timestamp', 'label']
        
        # Check label distribution BEFORE any processing
        if 'label' in df.columns:
            label_counts = df['label'].astype(str).value_counts()
            self.logger.info(f"Original label distribution: {label_counts.to_dict()}")
            unique_labels = df['label'].astype(str).unique()
            self.logger.info(f"Unique labels found: {list(unique_labels)}")
            if len(unique_labels) == 0 or (len(unique_labels) == 1 and 'nan' in str(unique_labels[0]).lower()):
                self.logger.critical("No valid unique labels found. This will prevent supervised learning.")
        else:
            self.logger.critical("No 'label' column found in raw data after standardization! This is a critical error.")
            df['label'] = 'BENIGN' # Add a dummy label column to prevent immediate crash if missing.
            self.logger.warning("Added dummy 'label' column as it was missing.")


        # Only proceed if essential columns are truly present or filled with defaults
        missing_after_std = [col for col in essential_cols if col not in df.columns]
        if missing_after_std:
            self.logger.critical(f"Still missing essential columns after standardization attempts: {missing_after_std}. Cannot proceed with preprocessing that relies on these.")
            raise ValueError(f"Missing essential columns after standardization: {missing_after_std}")


        # Impute missing values in essential columns (re-run as _standardize_columns might add some defaults)
        initial_rows = len(df)
        for col in ['flow_id', 'source_ip', 'destination_ip', 'protocol', 'label']:
            if df[col].isna().any():
                df[col].fillna('unknown_value', inplace=True)
                self.logger.info(f"Imputed {df[col].isna().sum()} NaNs in {col} with 'unknown_value'")
        for col in ['source_port', 'destination_port']:
            if df[col].isna().any():
                df[col].fillna(0, inplace=True)
                self.logger.info(f"Imputed {df[col].isna().sum()} NaNs in {col} with 0")

        # Log NaN distribution after imputation
        nan_counts = df[essential_cols].isna().sum()
        self.logger.info(f"NaN counts in essential columns after imputation: {nan_counts.to_dict()}")

        # Handle timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = df['timestamp'].astype(str) 
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isna().any():
                num_invalid_timestamps = df['timestamp'].isna().sum()
                self.logger.warning(f"Found {num_invalid_timestamps} rows with invalid timestamps. Attempting format inference.")
                
                invalid_timestamps_sample = df[df['timestamp'].isna()]['timestamp'].head(5).tolist()
                if invalid_timestamps_sample:
                    self.logger.info(f"Sample invalid timestamps: {invalid_timestamps_sample}")
                
                # Try additional CICIDS2017 timestamp formats
                for fmt in ['%d/%m/%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f', '%H:%M:%S']:
                    invalid_rows = df[df['timestamp'].isna()]
                    if not invalid_rows.empty:
                        try:
                            df.loc[invalid_rows.index, 'timestamp'] = pd.to_datetime(invalid_rows['timestamp'], format=fmt, errors='coerce')
                            self.logger.info(f"Parsed {len(invalid_rows) - df['timestamp'].isna().sum()} timestamps with format {fmt}")
                        except ValueError:
                            continue
                
                if df['timestamp'].isna().any():
                    num_invalid = df['timestamp'].isna().sum()
                    self.logger.warning(f"Still {num_invalid} invalid timestamps. Filling with unique sequential dummy values to ensure temporal continuity.")
                    nan_indices = df.index[df['timestamp'].isna()]
                    df.loc[nan_indices, 'timestamp'] = pd.to_datetime(pd.Series(range(len(nan_indices))), unit='s', origin='unix') + pd.to_timedelta(pd.Series(nan_indices), unit='ms')
                    self.logger.info("Filled invalid timestamps with unique sequential dummy values.")
        else:
            self.logger.critical("Timestamp column missing. Creating generic timestamp for all rows.")
            df['timestamp'] = pd.to_datetime(pd.Series(range(len(df))), unit='s', origin='unix')

        # Drop rows with remaining NaNs in essential columns AFTER timestamp handling
        initial_rows_before_final_dropna = len(df)
        df.dropna(subset=essential_cols, inplace=True)
        dropped_rows_final = initial_rows_before_final_dropna - len(df)
        if dropped_rows_final > 0:
            self.logger.info(f"Dropped {dropped_rows_final} rows with NaN in essential columns after final imputation/coercion.")
            
        if 'label' in df.columns:
            df['label'] = df['label'].astype(str)

        df = self._standardize_labels(df)
        self._validate_feature_columns(df)

        df = df.sort_values(by='timestamp').reset_index(drop=True)
        self.logger.info("DataFrame sorted by timestamp.")

        # Handle numeric columns (fill NaNs and Infs)
        numeric_cols_to_process = df.select_dtypes(include=np.number).columns.tolist()
        
        if numeric_cols_to_process:
            df[numeric_cols_to_process] = df[numeric_cols_to_process].fillna(0.0)
            df[numeric_cols_to_process] = df[numeric_cols_to_process].replace([np.inf, -np.inf], 0.0)
            self.logger.info(f"Handled NaNs and infinite values in {len(numeric_cols_to_process)} numeric columns.")
        else:
            self.logger.info("No numeric columns found to handle NaNs/Infs.")

        self.logger.info(f"Final preprocessed DataFrame: {len(df)} rows, {len(df.columns)} columns.")
        return df