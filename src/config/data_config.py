# src/config/data_config.py
from dataclasses import dataclass

@dataclass
class DataConfig:
    # Placeholder for data-specific configurations
    raw_data_path: str = "data/raw/"
    processed_data_path: str = "data/processed/"
    dataset_name: str = "darpa_tc"
    # Add more data specific configs here, e.g., features to extract, normalization params
