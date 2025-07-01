# src/config/training_config.py
from dataclasses import dataclass

@dataclass
class TrainingConfig:
    # Placeholder for training-specific configurations
    checkpoint_dir: str = "experiments/checkpoints/"
    log_dir: str = "logs/"
    results_dir: str = "results/"
    save_interval: int = 50 # Save model every N epochs
    # Add more training specific configs here, e.g., early stopping, optimizer settings
