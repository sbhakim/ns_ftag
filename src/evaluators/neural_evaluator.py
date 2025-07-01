# src/evaluators/neural_evaluator.py


import torch
import numpy as np
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, roc_auc_score
from typing import Dict, List, Any
import networkx as nx
import torch.nn.functional as F
import logging

class NeuralAttackGraphEvaluator:
    def __init__(self, config: Any):
        self.config = config
        self.metrics = {}
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )

    def evaluate_attack_detection(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Evaluate attack step detection performance."""
        metrics = {}
        attack_pred_probs = F.softmax(predictions['attack_presence'], dim=1)
        attack_pred_labels = torch.argmax(attack_pred_probs, dim=1)
        attack_true_labels = targets['attack_presence']

        attack_pred_labels_cpu = attack_pred_labels.cpu().numpy()
        attack_true_labels_cpu = attack_true_labels.cpu().numpy()
        attack_pred_probs_cpu = attack_pred_probs[:, 1].cpu().numpy()

        metrics['attack_accuracy'] = accuracy_score(attack_true_labels_cpu, attack_pred_labels_cpu)

        unique_labels = np.unique(attack_true_labels_cpu)
        if len(unique_labels) < 2:
            self.logger.warning(f"Only one class ({unique_labels[0]}) in true labels for attack presence. Precision/Recall/F1 set to 0.0.")
            prec, rec, f1 = 0.0, 0.0, 0.0
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(
                attack_true_labels_cpu, attack_pred_labels_cpu, average='binary', zero_division=0
            )

        metrics['attack_precision'] = prec
        metrics['attack_recall'] = rec
        metrics['attack_f1'] = f1

        try:
            metrics['attack_roc_auc'] = roc_auc_score(attack_true_labels_cpu, attack_pred_probs_cpu)
        except ValueError:
            metrics['attack_roc_auc'] = np.nan

        if 'attack_type' in predictions and 'attack_type' in targets:
            type_pred_labels = torch.argmax(predictions['attack_type'], dim=1).cpu().numpy()
            type_true_labels = targets['attack_type'].cpu().numpy()
            metrics['attack_type_accuracy'] = accuracy_score(type_true_labels, type_pred_labels)

        if 'mitre_technique' in predictions and 'mitre_technique' in targets:
            mitre_pred_labels = torch.argmax(predictions['mitre_technique'], dim=1).cpu().numpy()
            mitre_true_labels = targets['mitre_technique'].cpu().numpy()
            metrics['mitre_technique_accuracy'] = accuracy_score(mitre_true_labels, mitre_pred_labels)

        self.logger.info(f"Attack detection metrics: {metrics}")
        return metrics

    def evaluate_temporal_sequences(self, predicted_graphs_data: List[Dict[str, Any]], true_sequences_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate temporal sequence accuracy based on predicted vs. true edges."""
        correct_paths_matched = 0
        total_true_paths = 0

        for i, pred_graph_data in enumerate(predicted_graphs_data):
            if i >= len(true_sequences_data):
                continue

            predicted_edges = set(tuple(e) for e in pred_graph_data.get('edges', []))
            true_edges = set(tuple(e) for e in true_sequences_data[i].get('true_edges', [])) or set(
                (k, k + 1) for k in range(len(pred_graph_data['node_entities']) - 1)
            )

            if not true_edges:
                continue

            matched_edges = predicted_edges.intersection(true_edges)
            correct_paths_matched += len(matched_edges)
            total_true_paths += len(true_edges)

        temporal_accuracy = correct_paths_matched / max(total_true_paths, 1)
        metrics = {
            'temporal_accuracy': temporal_accuracy,
            'correct_edges_matched': correct_paths_matched,
            'total_true_edges': total_true_paths
        }
        self.logger.info(f"Temporal sequence metrics: {metrics}")
        return metrics

    def evaluate_graph_construction_f1(self, predicted_graphs_data: List[Dict[str, Any]], true_sequences_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate F1 score for graph construction (nodes and edges)."""
        node_precision, node_recall, node_f1 = 0.0, 0.0, 0.0
        edge_precision, edge_recall, edge_f1 = 0.0, 0.0, 0.0
        total_graphs = min(len(predicted_graphs_data), len(true_sequences_data))

        for i in range(total_graphs):
            pred_data = predicted_graphs_data[i]
            true_data = true_sequences_data[i]

            # Node evaluation (based on attack presence)
            pred_nodes = set(range(len(pred_data['node_entities'])))
            true_nodes = set(range(len(pred_data['node_entities'])))  # Assume same nodes for simplicity
            true_positive_nodes = len(pred_nodes.intersection(true_nodes))
            node_precision += true_positive_nodes / max(len(pred_nodes), 1)
            node_recall += true_positive_nodes / max(len(true_nodes), 1)

            # Edge evaluation
            pred_edges = set(tuple(e) for e in pred_data.get('edges', []))
            true_edges = set(tuple(e) for e in true_data.get('true_edges', [])) or set(
                (k, k + 1) for k in range(len(pred_data['node_entities']) - 1)
            )
            true_positive_edges = len(pred_edges.intersection(true_edges))
            edge_precision += true_positive_edges / max(len(pred_edges), 1)
            edge_recall += true_positive_edges / max(len(true_edges), 1)

        if total_graphs > 0:
            node_precision /= total_graphs
            node_recall /= total_graphs
            node_f1 = 2 * (node_precision * node_recall) / max(node_precision + node_recall, 1e-10)
            edge_precision /= total_graphs
            edge_recall /= total_graphs
            edge_f1 = 2 * (edge_precision * edge_recall) / max(edge_precision + edge_recall, 1e-10)

        metrics = {
            'node_precision': node_precision,
            'node_recall': node_recall,
            'node_f1': node_f1,
            'edge_precision': edge_precision,
            'edge_recall': edge_recall,
            'edge_f1': edge_f1
        }
        self.logger.info(f"Graph construction metrics: {metrics}")
        return metrics

    def evaluate_against_baseline(self, predictions: Dict[str, torch.Tensor], targets: Dict[str, torch.Tensor]) -> Dict[str, float]:
        """Compare model performance against a rule-based baseline."""
        metrics = {}
        # Simple rule-based baseline: Predict attack if flow_bytess > threshold
        baseline_preds = (predictions['attack_presence'][:, 1] > 0.5).long().cpu().numpy()
        true_labels = targets['attack_presence'].cpu().numpy()
        metrics['baseline_accuracy'] = accuracy_score(true_labels, baseline_preds)
        prec, rec, f1, _ = precision_recall_fscore_support(true_labels, baseline_preds, average='binary', zero_division=0)
        metrics['baseline_precision'] = prec
        metrics['baseline_recall'] = rec
        metrics['baseline_f1'] = f1
        self.logger.info(f"Baseline metrics: {metrics}")
        return metrics

    def evaluate_all(self, model_predictions: List[Dict[str, torch.Tensor]], true_targets: List[Dict[str, torch.Tensor]], 
                     predicted_graphs_data: List[Dict[str, Any]], true_sequences_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Comprehensive evaluation across all metrics."""
        all_metrics = {}
        flattened_pred_detection = {k: torch.cat([p[k].view(-1, p[k].shape[-1]) if p[k].dim() > 1 else p[k].view(-1) for p in model_predictions], dim=0)
                                   for k in model_predictions[0] if k != 'attention_weights'}
        flattened_true_detection = {k: torch.cat([t[k].view(-1) for t in true_targets], dim=0)
                                   for k in true_targets[0]}

        detection_metrics = self.evaluate_attack_detection(flattened_pred_detection, flattened_true_detection)
        all_metrics.update(detection_metrics)
        temporal_metrics = self.evaluate_temporal_sequences(predicted_graphs_data, true_sequences_data)
        all_metrics.update(temporal_metrics)
        graph_metrics = self.evaluate_graph_construction_f1(predicted_graphs_data, true_sequences_data)
        all_metrics.update(graph_metrics)
        baseline_metrics = self.evaluate_against_baseline(flattened_pred_detection, flattened_true_detection)
        all_metrics.update(baseline_metrics)

        self.logger.info(f"All evaluation metrics: {all_metrics}")
        return all_metrics
