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
        """Apply column name standardization and renaming for CICIDS2017 dataset."""
        original_cols = df.columns.tolist()
        df.columns = df.columns.str.strip().str.replace(' ', '').str.lower().str.replace(r'[^a-z0-9]', '', regex=True)
        
        df = df.rename(columns={
            'flowid': 'flow_id',
            'sourceip': 'source_ip',
            'sourceport': 'source_port',
            'destinationip': 'destination_ip',
            'destinationport': 'destination_port',
            'flowbytess': 'flow_bytes_s',
            'flowpacketss': 'flow_packets_s',
            'totalfwdpackets': 'total_fwd_packets',
            'totalbackwardpackets': 'total_backward_packets',
            'totallengthoffwdpackets': 'total_length_of_fwd_packets',
            'totallengthofbwdpackets': 'total_length_of_bwd_packets',
            'fwdpacketlengthmean': 'fwd_packet_length_mean',
            'bwdpacketlengthmean': 'bwd_packet_length_mean'
        }, errors='ignore')
        
        self.logger.info(f"Original columns: {original_cols}")
        self.logger.info(f"Renamed columns: {df.columns.tolist()}")
        
        essential_cols = ['flow_id', 'source_ip', 'destination_ip', 'source_port', 'destination_port', 'protocol', 'timestamp', 'label']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            self.logger.warning(f"Missing essential columns after renaming: {missing_cols}")
            return None
        return df

    def _load_single_csv(self, file_path: str) -> pd.DataFrame:
        """Load a single CSV file with encoding handling."""
        filename = os.path.basename(file_path)
        for encoding in self.config.possible_encodings:
            try:
                df = pd.read_csv(file_path, low_memory=False, encoding=encoding)
                self.logger.info(f"Successfully read {filename} with encoding: {encoding} ({len(df)} rows, {len(df.columns)} columns)")
                return self._standardize_columns(df)
            except UnicodeDecodeError:
                self.logger.warning(f"Failed to read {filename} with encoding {encoding}. Trying next.")
            except pd.errors.EmptyDataError:
                self.logger.warning(f"File {filename} is empty. Skipping.")
                return None
            except Exception as e:
                self.logger.error(f"Error loading {filename} with encoding {encoding}: {e}")
                return None
        return None

    def load_data(self, file_path: str) -> pd.DataFrame:
        """
        Loads CICIDS2017 CSV files dynamically from the specified path and combines them into a single DataFrame.
        """
        self.logger.info(f"Loading data from: {file_path}")
        file_path = os.path.normpath(file_path)
        csv_files = sorted(glob.glob(os.path.join(file_path, '*.[cC][sS][vV]')))
        
        if not csv_files:
            raise ValueError(f"No CSV files found in {file_path}. Check path, permissions, or file extensions.")

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
        return self.preprocess(self._standardize_columns(df.copy()))

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs preprocessing steps: timestamp conversion, NaN/Inf handling, sorting, and diagnostics.
        """
        self.logger.info(f"Starting preprocessing for DataFrame with {len(df)} rows.")
        
        essential_cols = ['flow_id', 'source_ip', 'destination_ip', 'source_port', 'destination_port', 'protocol', 'timestamp', 'label']
        missing_cols = [col for col in essential_cols if col not in df.columns]
        if missing_cols:
            self.logger.critical(f"Missing essential columns after preprocessing: {missing_cols}")
            raise ValueError(f"Missing essential columns: {missing_cols}")

        # Impute missing values in essential columns
        initial_rows = len(df)
        for col in ['flow_id', 'source_ip', 'destination_ip', 'protocol', 'label']:
            if df[col].isna().any():
                df[col].fillna('unknown', inplace=True)
                self.logger.info(f"Imputed {df[col].isna().sum()} NaNs in {col} with 'unknown'")
        for col in ['source_port', 'destination_port']:
            if df[col].isna().any():
                df[col].fillna(0, inplace=True)
                self.logger.info(f"Imputed {df[col].isna().sum()} NaNs in {col} with 0")

        # Log NaN distribution after imputation
        nan_counts = df[essential_cols].isna().sum()
        self.logger.info(f"NaN counts in essential columns after imputation: {nan_counts.to_dict()}")

        # Handle timestamp column
        if 'timestamp' in df.columns:
            df['timestamp'] = pd.to_datetime(df['timestamp'], errors='coerce')
            if df['timestamp'].isna().any():
                num_invalid_timestamps = df['timestamp'].isna().sum()
                self.logger.warning(f"Found {num_invalid_timestamps} rows with invalid timestamps. Attempting format inference.")
                
                # Log sample invalid timestamps
                invalid_timestamps = df[df['timestamp'].isna()]['timestamp'].head(5).tolist()
                if invalid_timestamps:
                    self.logger.info(f"Sample invalid timestamps: {invalid_timestamps}")
                
                # Try additional CICIDS2017 timestamp formats
                for fmt in ['%d/%m/%Y %H:%M:%S', '%Y/%m/%d %H:%M:%S', '%d-%m-%Y %H:%M:%S', '%Y-%m-%d %H:%M:%S.%f']:
                    invalid_rows = df[df['timestamp'].isna()]
                    if not invalid_rows.empty:
                        try:
                            df.loc[invalid_rows.index, 'timestamp'] = pd.to_datetime(invalid_rows['timestamp'], format=fmt, errors='coerce')
                            self.logger.info(f"Parsed {len(invalid_rows) - df['timestamp'].isna().sum()} timestamps with format {fmt}")
                        except ValueError:
                            continue
                
                if df['timestamp'].isna().any():
                    num_invalid = df['timestamp'].isna().sum()
                    self.logger.warning(f"Still {num_invalid} invalid timestamps. Filling with sequential dummy values.")
                    nan_indices = df.index[df['timestamp'].isna()]
                    df.loc[nan_indices, 'timestamp'] = pd.to_datetime(pd.Series(range(num_invalid)), unit='s', origin='unix') + pd.to_timedelta(pd.Series(df.index[nan_indices]), unit='ms')
                    self.logger.info("Filled invalid timestamps with unique sequential dummy values.")
        else:
            self.logger.critical("Timestamp column missing. Creating generic timestamp.")
            df['timestamp'] = pd.to_datetime(pd.Series(range(len(df))), unit='s', origin='unix')

        # Drop rows with remaining NaNs in essential columns
        initial_rows = len(df)
        df.dropna(subset=essential_cols, inplace=True)
        dropped_rows = initial_rows - len(df)
        if dropped_rows > 0:
            self.logger.info(f"Dropped {dropped_rows} rows with NaN in essential columns after imputation.")

        # Sort by timestamp
        df = df.sort_values(by='timestamp').reset_index(drop=True)
        self.logger.info("DataFrame sorted by timestamp.")

        # Handle numeric columns
        numeric_cols = df.select_dtypes(include=np.number).columns.tolist()
        numeric_cols_present = [col for col in numeric_cols if col in df.columns]
        if numeric_cols_present:
            df[numeric_cols_present] = df[numeric_cols_present].fillna(0.0)
            df[numeric_cols_present] = df[numeric_cols_present].replace([np.inf, -np.inf], 0.0)
            self.logger.info(f"Handled NaNs and infinite values in {len(numeric_cols_present)} numeric columns.")
        else:
            self.logger.info("No numeric columns found to handle NaNs/Infs.")

        self.logger.info(f"Final preprocessed DataFrame: {len(df)} rows, {len(df.columns)} columns.")
        return df