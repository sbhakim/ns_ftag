# src/data_processors/label_extractor.py

import torch
import pandas as pd
from typing import Dict, Any
import logging

class LabelExtractor:
    def __init__(self, config):
        self.config = config
        self.attack_types = {'Benign': 0, 'DDoS': 1, 'PortScan': 2, 'Web Attack': 3, 'Bot': 4, 'Infiltration': 5}
        self.mitre_techniques = {
            'Benign': 0,
            'DDoS': 1,  # T1498: Network Denial of Service
            'PortScan': 2,  # T1595: Active Scanning
            'Web Attack': 3,  # T1190: Exploit Public-Facing Application
            'Bot': 4,  # T1071: Application Layer Protocol
            'Infiltration': 5  # T1078: Valid Accounts
        }
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )

    def extract_labels(self, events: pd.DataFrame) -> Dict[str, torch.Tensor]:
        """Extract labels for attack presence, type, severity, and confidence."""
        if 'label' not in events.columns:
            self.logger.error("Missing required column: 'label'")
            raise ValueError("Missing required column: 'label'")

        labels = [row['label'].strip().capitalize() for _, row in events.iterrows()]
        extracted_labels = {
            'attack_presence': torch.tensor([1 if label != 'Benign' else 0 for label in labels]),
            'attack_type': torch.tensor([self.attack_types.get(label, 0) for label in labels]),
            'mitre_technique': torch.tensor([self.mitre_techniques.get(label, 0) for label in labels]),
            'severity': torch.tensor([0.5 if label != 'Benign' else 0.0 for label in labels]),
            'confidence': torch.tensor([1.0 for _ in labels])
        }
        self.logger.info(f"Extracted labels for {len(labels)} events, MITRE techniques: {set(extracted_labels['mitre_technique'].tolist())}")
        return extracted_labels