# src/train_phase1.py


import torch
from torch.utils.data import DataLoader
import os
import random
import numpy as np
import sys
from typing import Dict, Any, List

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__))))

from neural_components.neural_pipeline import NeuralAttackGraphPipeline
from data_processors.dataset import SecurityEventDataset
# Removed manual imports for processors/managers to rely on auto-selection
# from data_processors.cicids2017_processor import CICIDS2017Processor
# from data_processors.entity_manager import EntityManager
# from data_processors.relationship_extractor import RelationshipExtractor
# from data_processors.feature_extractor import FeatureExtractor
# from data_processors.label_extractor import LabelExtractor
from utils.training_manager import TrainingManager
from utils.device_manager import DeviceManager
from utils.performance_monitor import PerformanceMonitor
from utils.logging_utils import setup_logging
from evaluators.neural_evaluator import NeuralAttackGraphEvaluator
from config.neural_config import NeuralConfig
from config.training_config import TrainingConfig
from data_processors.batch_collator import custom_collate_fn

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def train_phase1():
    set_seed(42)
    neural_config = NeuralConfig()
    training_config = TrainingConfig()
    logger = setup_logging(log_dir=training_config.log_dir, filename="phase1_training.log")
    logger.info("Starting Phase 1 training...")
    logger.info(f"Neural Config: {neural_config}")
    logger.info(f"Training Config: {training_config}")
    logger.info(f"Dataset Type: {neural_config.dataset_type}") # NEW: Log dataset type [cite: 574, 575]

    device_manager = DeviceManager()
    logger.info(f"Using device: {device_manager.get_device()}")

    # === UPDATED: Simplified dataset creation with auto-selection ===
    # Components (processor, entity_manager, etc.) will be auto-selected within SecurityEventDataset
    dataset = SecurityEventDataset(
        data_path=neural_config.data_path,
        config=neural_config
        # No need to pass individual components - they'll be auto-selected
    )
    
    # [cite_start]Vocabulary sizes are now dynamically set in NeuralConfig.__post_init__ based on dataset_type [cite: 574, 575]
    # and then confirmed by the entity_manager during dataset initialization.
    # We re-assign here to ensure the model gets the correct, finalized vocab sizes.
    neural_config.entity_vocab_size = dataset.entity_manager.get_vocab_sizes()['entity_vocab_size']
    neural_config.action_vocab_size = dataset.entity_manager.get_vocab_sizes()['action_vocab_size']
    
    dataloader = DataLoader(
        dataset,
        batch_size=neural_config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    logger.info(f"Dataset loaded with {len(dataset)} sequences.")
    logger.info(f"Final entity_vocab_size: {neural_config.entity_vocab_size}, action_vocab_size: {neural_config.action_vocab_size}")


    model = NeuralAttackGraphPipeline(neural_config).to(device_manager.get_device())
    logger.info(f"Model created with {sum(p.numel() for p in model.parameters())} parameters.")

    trainer = TrainingManager(model, neural_config, device_manager)
    evaluator = NeuralAttackGraphEvaluator(neural_config)
    monitor = PerformanceMonitor()
    
    os.makedirs(training_config.checkpoint_dir, exist_ok=True)

    for epoch in range(neural_config.max_epochs):
        monitor.start(f"Epoch {epoch} Training")
        train_loss = trainer.train_epoch(dataloader)
        monitor.stop(f"Epoch {epoch} Training", {"train_loss": train_loss})
        logger.info(f"Epoch {epoch+1}/{neural_config.max_epochs}, Train Loss: {train_loss:.4f}")

        if (epoch + 1) % 10 == 0 or epoch == neural_config.max_epochs - 1:
            logger.info(f"--- Evaluating after Epoch {epoch+1} ---")
            model.eval()
            all_predictions_from_model: List[Dict[str, torch.Tensor]] = []
            all_targets_from_batch: List[Dict[str, torch.Tensor]] = []
            all_predicted_graphs_data: List[Dict[str, Any]] = []
            all_true_sequences_data: List[Dict[str, Any]] = []

            with torch.no_grad():
                for batch_idx, batch in enumerate(dataloader):
                    batch_on_device = {k: (v.to(device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
                    batch_on_device['targets'] = {k: (v.to(device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch['targets'].items()}
                    predictions = model(batch_on_device)
                    
                    # Detach and move to CPU for appending to lists
                    # Note: attention_weights can be large lists of tensors, might need careful handling if memory becomes an issue
                    all_predictions_from_model.append({k: v.cpu() for k, v in predictions.items() if k != 'attention_weights'})
                    all_targets_from_batch.append({k: v.cpu() for k, v in batch_on_device['targets'].items()})
                    
                    # get_attack_graph handles CPU conversion internally
                    predicted_graphs_batch_data = model.get_attack_graph(batch_on_device)
                    all_predicted_graphs_data.extend(predicted_graphs_batch_data)
                    
                    # Ensure true_sequences_data contains the necessary info for evaluation
                    # For temporal evaluation, 'true_edges' is needed.
                    # For graph construction F1, 'targets' might also be needed for node-level F1, if implemented.
                    for seq_idx in range(batch['entities'].shape[0]):
                        true_seq_item = {'true_edges': batch['true_edges'][seq_idx]}
                        # If node-level F1 on attack steps is desired, you might need to add targets here
                        # true_seq_item['targets'] = {k: v[seq_idx].cpu() for k,v in batch['targets'].items()} 
                        all_true_sequences_data.append(true_seq_item)

            monitor.start(f"Epoch {epoch} Evaluation")
            eval_metrics = evaluator.evaluate_all(all_predictions_from_model, all_targets_from_batch,
                                                  all_predicted_graphs_data, all_true_sequences_data)
            monitor.stop(f"Epoch {epoch} Evaluation", eval_metrics)
            logger.info(f"Evaluation Metrics: {eval_metrics}")
            model.train()

        if (epoch + 1) % training_config.save_interval == 0 or epoch == neural_config.max_epochs - 1:
            checkpoint_path = os.path.join(training_config.checkpoint_dir, f'model_epoch_{epoch+1}.pt')
            torch.save(model.state_dict(), checkpoint_path)
            logger.info(f"Model checkpoint saved to {checkpoint_path}")

    logger.info("Phase 1 training completed!")
    monitor.save_metrics(os.path.join(training_config.results_dir, "phase1_performance_metrics.json"))

if __name__ == "__main__":
    train_phase1()