# src/data_processors/darpa_tc_processor.py

from .base_processor import BaseSecurityEventProcessor
import pandas as pd

class DarpaTCProcessor(BaseSecurityEventProcessor):
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Placeholder for loading DARPA TC system call traces."""
        raise NotImplementedError("DARPA TC data loading not implemented yet.")

    def preprocess(self, df: pd.DataFrame) -> pd.DataFrame:
        """Placeholder for DARPA TC preprocessing."""
        raise NotImplementedError("DARPA TC preprocessing not implemented yet.")