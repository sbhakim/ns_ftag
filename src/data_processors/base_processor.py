# src/data_processors/base_processor.py

from abc import ABC, abstractmethod
import pandas as pd
from typing import Dict, List, Any

class BaseSecurityEventProcessor(ABC):
    def __init__(self, config: Any):
        self.config = config

    @abstractmethod
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load and preprocess dataset."""
        pass

    @abstractmethod
    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Apply dataset-specific preprocessing (e.g., timestamp sorting, cleaning)."""
        pass