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
from data_processors.cicids2017_processor import CICIDS2017Processor
from data_processors.entity_manager import EntityManager
from data_processors.relationship_extractor import RelationshipExtractor
from data_processors.feature_extractor import FeatureExtractor
from data_processors.label_extractor import LabelExtractor
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

    device_manager = DeviceManager()
    logger.info(f"Using device: {device_manager.get_device()}")

    processor = CICIDS2017Processor(neural_config)
    entity_manager = EntityManager()
    relationship_extractor = RelationshipExtractor(neural_config)
    feature_extractor = FeatureExtractor(neural_config)
    label_extractor = LabelExtractor(neural_config)

    dataset = SecurityEventDataset(
        data_path=neural_config.data_path,
        config=neural_config,
        processor=processor,
        entity_manager=entity_manager,
        relationship_extractor=relationship_extractor,
        feature_extractor=feature_extractor,
        label_extractor=label_extractor
    )
    neural_config.entity_vocab_size = entity_manager.get_vocab_sizes()['entity_vocab_size']
    neural_config.action_vocab_size = entity_manager.get_vocab_sizes()['action_vocab_size']
    dataloader = DataLoader(
        dataset,
        batch_size=neural_config.batch_size,
        shuffle=True,
        collate_fn=custom_collate_fn
    )
    logger.info(f"Dataset loaded with {len(dataset)} sequences.")

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
                    all_predictions_from_model.append({k: v.cpu() for k, v in predictions.items() if k != 'attention_weights'})
                    all_targets_from_batch.append({k: v.cpu() for k, v in batch_on_device['targets'].items()})
                    predicted_graphs_batch_data = model.get_attack_graph(batch_on_device)
                    all_predicted_graphs_data.extend(predicted_graphs_batch_data)
                    for seq_idx in range(batch['entities'].shape[0]):
                        all_true_sequences_data.append({'true_edges': batch['true_edges'][seq_idx]})

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