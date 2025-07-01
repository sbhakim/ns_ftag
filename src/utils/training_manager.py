# src/utils/training_manager.py
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from typing import Dict, Any

class TrainingManager:
    def __init__(self, model: nn.Module, config: Any, device_manager: Any):
        self.model = model
        self.config = config
        self.device = device_manager.get_device()

        # Multi-task loss
        self.criterion = {
            'attack_presence': nn.CrossEntropyLoss(),
            'attack_type': nn.CrossEntropyLoss(),
            'mitre_technique': nn.CrossEntropyLoss(),
            'severity': nn.MSELoss(),
            'confidence': nn.MSELoss()
        }

        self.optimizer = torch.optim.Adam(model.parameters(), lr=config.learning_rate)

    def compute_loss(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute multi-task loss"""
        total_loss = 0.0
        loss_weights = self.config.loss_weights

        for task, pred in predictions.items():
            if task in targets and task in self.criterion:
                # Ensure target tensor is correctly shaped for the loss function
                target_tensor = targets[task]
                if task in ['attack_presence', 'attack_type', 'mitre_technique']:
                    # For CrossEntropyLoss, targets should be class indices (long type)
                    target_tensor = target_tensor.long()
                elif task in ['severity', 'confidence']:
                    # For MSELoss, targets should match prediction shape
                    target_tensor = target_tensor.float().view_as(pred)

                task_loss = self.criterion[task](pred, target_tensor)
                total_loss += loss_weights.get(task, 1.0) * task_loss
        return total_loss

    def train_epoch(self, dataloader: DataLoader) -> float:
        """Train one epoch"""
        self.model.train()
        total_loss = 0.0

        for batch in dataloader:
            # Move to device
            # Ensure all tensors in batch are moved to device
            batch = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
            batch['targets'] = {k: (v.to(self.device) if isinstance(v, torch.Tensor) else v) for k, v in batch['targets'].items()}

            # Forward pass
            predictions = self.model(batch)
            loss = self.compute_loss(predictions, batch['targets'])

            # Backward pass
            self.optimizer.zero_grad()
            loss.backward()
            self.optimizer.step()

            total_loss += loss.item()
        return total_loss / len(dataloader)
