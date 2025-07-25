# src/config/neural_config.py

from dataclasses import dataclass, field
from typing import Dict, List, Any # Import Any for subset_config's type
import os
import logging
import stat
import glob
import subprocess

@dataclass
class NeuralConfig:
    # GNN Configuration
    gnn_hidden_dims: List[int] = field(default_factory=lambda: [128, 256, 128])
    gnn_num_heads: int = 8
    gnn_dropout: float = 0.2

    # TCN Configuration
    tcn_channels: List[int] = field(default_factory=lambda: [64, 128, 64])
    tcn_kernel_size: int = 3
    tcn_dilations: List[int] = field(default_factory=lambda: [1, 2, 4, 8])

    # Training hyperparameters
    learning_rate: float = 0.001
    batch_size: int = 32
    max_epochs: int = 100

    # Dynamic Data Path (must be set via environment or CLI override)
    _current_file_dir: str = os.path.dirname(os.path.abspath(__file__))
    _project_root: str = os.path.abspath(os.path.join(_current_file_dir, '..', '..'))
    
    # === NEW: Dataset Selection ===
    dataset_type: str = field(default_factory=lambda: os.environ.get('NS_FTAG_DATASET_TYPE', 'cicids2017'))  # 'cicids2017' or 'optc'
    data_path: str = field(default_factory=lambda: os.environ.get('NS_FTAG_DATA_PATH'))

    # Logging Configuration
    log_dir: str = "logs/"

    # Supported Encodings for CSV Files
    possible_encodings: List[str] = field(default_factory=lambda: ['utf-8', 'latin1', 'ISO-8859-1', 'windows-1252'])

    # Loss Weighting for Multi-Task Heads
    loss_weights: Dict[str, float] = field(default_factory=lambda: {
        'attack_presence': 1.0,
        'attack_type': 1.0,
        'mitre_technique': 1.0,
        'severity': 0.5,
        'confidence': 0.5
    })

    # Security-Specific Settings
    max_sequence_length: int = 1000
    attention_mechanism: str = "security_aware"
    entity_vocab_size: int = 5000       # To be updated dynamically based on dataset type
    action_vocab_size: int = 500        # To be updated dynamically based on dataset type
    embedding_dim: int = 64
    num_attack_types: int = 6           # Updated for CICIDS2017: BENIGN, DDoS, PortScan, Web Attack, Bot, Infiltration
    num_mitre_techniques: int = 20      # Adjust per MITRE ATT&CK subset
    attention_threshold: float = 0.1     # For graph-builder edge pruning
    sequence_window_size: int = 50      # Sliding-window length. Default for CICIDS2017.
    min_sequence_length: int = 10       # Minimum window length
    security_feature_dim: int = 10       # Number of per-node security features

    # --- NEW CONFIGURATIONS FOR ENHANCED GRAPH BUILDER ---
    # Temporal constraint for edge validation (in seconds)
    # E.g., 3600s = 1 hour. Events more than this gap apart won't form an edge (unless specifically allowed)
    max_attack_step_gap: int = 3600 # Default to 1 hour (3600 seconds)

    # Weights for multi-criteria path scoring (for extract_ranked_attack_paths)
    path_scoring_weights: Dict[str, float] = field(default_factory=lambda: {
        'probability': 0.3,          # Weight for average attack probability of nodes in path
        'edge_confidence': 0.3,      # Weight for average confidence of edges in path
        'temporal': 0.2,             # Weight for temporal consistency/validity of path
        'mitre': 0.2                 # Weight for MITRE coherence of path (Phase 2)
    })
    
    # Cutoff for simple path extraction in NetworkX (to prevent computational explosion)
    # e.g., max path length or depth to explore. 0 means no limit.
    path_extraction_cutoff: int = 6 # Default max path length to 6 hops

    # === NEW: OpTC-Specific Parameters ===
    # Updated optc_ecar_path to point to the actual directory containing JSON files
    # This path is relative to the NS_FTAG_DATA_PATH for OpTC, which is /media/safayat/second_ssd/Cyber-Project/ns_ftag_cyber/data/datasets/OpTC-data
    optc_ecar_path: str = os.path.join("ecar", "short", "17-18Sep19", "AIA-176-200")
    optc_evaluation_path: str = "evaluation"
    optc_benign_path: str = "benign"
    optc_ground_truth_path: str = "OpTCRedTeamGroundTruth.pdf"

    # OpTC-Specific Vocabularies (larger than network-based)
    optc_entity_vocab_size: int = 100000  # Much larger for system entities
    optc_action_vocab_size: int = 1000    # More diverse system actions

    # === NEW: Dataset Subsetting Parameters ===
    # Use this to define a subset of data for faster experimentation during development.
    # Set to None to use the full dataset.
    subset_config: Dict[str, Any] = field(default_factory=lambda: {
        'enabled': True, # Set to False to use the full dataset
        'type': 'temporal_window', # Options: 'temporal_window', 'random_sample', 'attack_focus'
        'start_date': '2019-09-17 00:00:00', # For 'temporal_window'
        'end_date': '2019-09-24 23:59:59',   # For 'temporal_window' (e.g., 7 days of data)
        'max_hosts': None, # Optional: Limit number of hosts for 'temporal_window' or 'random_sample'
        'sample_fraction': None, # For 'random_sample': e.g., 0.1 for 10%
        'max_events': 500000 # Optional: Maximum number of events to retain after other filters
    })

    def __post_init__(self):
        """Validate and normalize data path, setup logging, and check file system access."""
        # Setup logging
        os.makedirs(self.log_dir, exist_ok=True)
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s',
            handlers=[
                logging.FileHandler(os.path.join(self.log_dir, 'config.log')),
                logging.StreamHandler()
            ]
        )
        self.logger = logging.getLogger(__name__)

        # Ensure data_path is set
        if not self.data_path:
            raise ValueError(
                "NS_FTAG_DATA_PATH is not set. Please export it or pass --data-path to point to your TrafficLabelling folder."
            )

        # Normalize data_path
        self.data_path = os.path.normpath(self.data_path)
        self.logger.info(f"Configured data_path: {self.data_path}")

        # Resolve symlink if present
        try:
            resolved_path = os.path.realpath(self.data_path)
            self.logger.info(f"Resolved data_path (symlink target): {resolved_path}")
            if os.path.islink(self.data_path):
                self.logger.info(f"data_path {self.data_path} is a symbolic link")
        except Exception as e:
            self.logger.warning(f"Error resolving symlink for {self.data_path}: {e}")

        # Check path existence
        if not os.path.exists(self.data_path):
            self.logger.error(f"data_path {self.data_path!r} does not exist")
            raise ValueError(f"data_path {self.data_path!r} does not exist")

        # Validate directory or file
        if os.path.isdir(self.data_path):
            # List files
            files = os.listdir(self.data_path)
            self.logger.info(f"Files in data_path: {files}")
            # Find CSVs (only relevant for CICIDS2017 detection at this top level)
            csv_files = glob.glob(os.path.join(self.data_path, '*.[cC][sS][vV]'))
            self.logger.info(f"CSV files found: {csv_files}")
            if not csv_files:
                self.logger.warning(f"No CSV files found in {self.data_path}")
            # Permissions
            permissions = os.stat(self.data_path).st_mode
            readable = bool(permissions & stat.S_IRUSR)
            self.logger.info(f"Directory permissions: {oct(permissions)}, Readable: {readable}")
            if not readable:
                self.logger.error(f"No read permission for {self.data_path}")
                raise PermissionError(f"No read permission for {self.data_path}")
        else:
            # Single-file mode
            permissions = os.stat(self.data_path).st_mode
            readable = bool(permissions & stat.S_IRUSR)
            self.logger.info(f"File permissions: {oct(permissions)}, Readable: {readable}")
            if not readable:
                self.logger.error(f"No read permission for {self.data_path}")
                raise PermissionError(f"No read permission for {self.data_path}")

        # Log mount and disk usage
        try:
            mount_output = subprocess.check_output(['mount', '-l'], text=True)
            self.logger.info(f"Mount information: {mount_output}")
            df_output = subprocess.check_output(['df', '-h', self.data_path], text=True)
            self.logger.info(f"Disk usage for {self.data_path}: {df_output}")
        except subprocess.CalledProcessError as e:
            self.logger.warning(f"Error retrieving mount or disk usage info: {e}")

        # === NEW: Dataset-specific vocabulary sizing ===
        if self.dataset_type == "optc":
            self.entity_vocab_size = self.optc_entity_vocab_size
            self.action_vocab_size = self.optc_action_vocab_size
            # Adjust sequence window size if OpTC data typically has longer sequences or needs different chunking
            self.sequence_window_size = 100 # Example adjustment for OpTC
            self.logger.info(f"Configured for OpTC dataset with larger vocabularies and sequence window size: {self.sequence_window_size}")
        else: # Default to CICIDS2017 settings (already defined as default values in dataclass)
            self.logger.info(f"Configured for CICIDS2017 dataset.")