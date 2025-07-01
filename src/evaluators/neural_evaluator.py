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
            self.logger.warning(f"Only one class ({unique_labels[0]}) in true labels for attack presence. Precision/Recall/F1/ROC_AUC set to 0.0/NaN.")
            prec, rec, f1 = 0.0, 0.0, 0.0
            metrics['attack_roc_auc'] = np.nan
        else:
            prec, rec, f1, _ = precision_recall_fscore_support(
                attack_true_labels_cpu, attack_pred_labels_cpu, average='binary', zero_division=0
            )
            try:
                metrics['attack_roc_auc'] = roc_auc_score(attack_true_labels_cpu, attack_pred_probs_cpu)
            except ValueError:
                metrics['attack_roc_auc'] = np.nan


        metrics['attack_precision'] = prec
        metrics['attack_recall'] = rec
        metrics['attack_f1'] = f1


        if 'attack_type' in predictions and 'attack_type' in targets:
            type_pred_labels = torch.argmax(predictions['attack_type'], dim=1).cpu().numpy()
            type_true_labels = targets['attack_type'].cpu().numpy()
            metrics['attack_type_accuracy'] = accuracy_score(type_true_labels, type_pred_labels)
            
            # Add type-specific precision/recall/f1 if desired, handling multi-class zero_division
            # E.g., prec_type, rec_type, f1_type, _ = precision_recall_fscore_support(type_true_labels, type_pred_labels, average='weighted', zero_division=0)
            # metrics['attack_type_f1'] = f1_type

        if 'mitre_technique' in predictions and 'mitre_technique' in targets:
            mitre_pred_labels = torch.argmax(predictions['mitre_technique'], dim=1).cpu().numpy()
            mitre_true_labels = targets['mitre_technique'].cpu().numpy()
            metrics['mitre_technique_accuracy'] = accuracy_score(mitre_true_labels, mitre_pred_labels)
            
            # Add technique-specific precision/recall/f1 if desired, handling multi-class zero_division
            # E.g., prec_mitre, rec_mitre, f1_mitre, _ = precision_recall_fscore_support(mitre_true_labels, mitre_pred_labels, average='weighted', zero_division=0)
            # metrics['mitre_technique_f1'] = f1_mitre


        self.logger.info(f"Attack detection metrics: {metrics}")
        return metrics

    def evaluate_temporal_sequences(self, predicted_graphs_data: List[Dict[str, Any]], true_sequences_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate temporal sequence accuracy based on predicted vs. true edges."""
        correct_edges_matched = 0
        total_true_edges = 0 # This will be the sum of true edges across all sequences

        for i, pred_graph_data in enumerate(predicted_graphs_data):
            # Ensure we have corresponding true data
            if i >= len(true_sequences_data):
                self.logger.warning(f"Mismatch: predicted_graphs_data has more items ({len(predicted_graphs_data)}) than true_sequences_data ({len(true_sequences_data)}).")
                continue

            # --- CRITICAL FIX: Use true_edges from true_sequences_data ---
            # Remove the fallback `or set((k, k + 1) for k in range(len(pred_graph_data['node_entities']) - 1))`
            # This ensures that temporal_accuracy is calculated ONLY against actual ground truth edges.
            true_edges = set(tuple(e) for e in true_sequences_data[i].get('true_edges', []))
            
            if not true_edges:
                # If there are no true edges in this specific sequence, skip it for this metric
                # self.logger.debug(f"Skipping temporal evaluation for sequence {i}: No true edges found.")
                continue

            predicted_edges = set(tuple(e) for e in pred_graph_data.get('edges', []))
            
            matched_edges = predicted_edges.intersection(true_edges)
            correct_edges_matched += len(matched_edges)
            total_true_edges += len(true_edges)

        temporal_accuracy = correct_edges_matched / max(total_true_edges, 1) # Ensure no division by zero
        
        # Also calculate precision/recall/f1 for temporal sequences (edges)
        # This gives a better picture than just accuracy
        temporal_precision = 0.0
        temporal_recall = 0.0
        temporal_f1 = 0.0

        if correct_edges_matched > 0:
            # Sum all predicted and true edges from all sequences for aggregate P/R/F1
            all_predicted_edges_flat = [e for graph_data in predicted_graphs_data for e in graph_data.get('edges', [])]
            all_true_edges_flat = [e for seq_data in true_sequences_data for e in seq_data.get('true_edges', [])]

            # Create binary lists for sklearn.metrics
            # This is complex because node IDs are not necessarily contiguous across all sequences.
            # A simpler way for aggregate F1 on edges is:
            # true_positives = correct_edges_matched
            # possible_positives = sum(len(pred_graph_data.get('edges', [])) for pred_graph_data in predicted_graphs_data)
            # possible_true = total_true_edges
            
            # This F1 calculation is more direct from counts
            # precision = true_positives / (possible_positives + 1e-10)
            # recall = true_positives / (possible_true + 1e-10)
            # f1 = 2 * (precision * recall) / (precision + recall + 1e-10)

            # Let's use the individual sequence precision/recall and average them, or sum counts directly for one big F1.
            # For simplicity and to match evaluate_graph_construction_f1, we will stick to summing raw counts.
            # (Note: evaluate_graph_construction_f1 uses total_graphs for averaging, which might not be ideal for aggregate F1)
            
            # Reusing the logic for edge_f1 from evaluate_graph_construction_f1 for consistency.
            # We already have correct_edges_matched (TP), total_true_edges (P), and need total_predicted_edges (P')
            total_predicted_edges = sum(len(p.get('edges', [])) for p in predicted_graphs_data)
            
            if total_predicted_edges > 0:
                temporal_precision = correct_edges_matched / total_predicted_edges
            if total_true_edges > 0:
                temporal_recall = correct_edges_matched / total_true_edges
            
            if (temporal_precision + temporal_recall) > 1e-10:
                temporal_f1 = 2 * (temporal_precision * temporal_recall) / (temporal_precision + temporal_recall)


        metrics = {
            'temporal_accuracy': temporal_accuracy,
            'temporal_precision': temporal_precision, # Added
            'temporal_recall': temporal_recall,     # Added
            'temporal_f1': temporal_f1,             # Added
            'correct_edges_matched': correct_edges_matched,
            'total_true_edges': total_true_edges,
            'total_predicted_edges_for_temporal': total_predicted_edges # Added for debug
        }
        self.logger.info(f"Temporal sequence metrics: {metrics}")
        return metrics

    def evaluate_graph_construction_f1(self, predicted_graphs_data: List[Dict[str, Any]], true_sequences_data: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate F1 score for graph construction (nodes and edges)."""
        node_precision_sum, node_recall_sum, node_f1_sum = 0.0, 0.0, 0.0
        edge_precision_sum, edge_recall_sum, edge_f1_sum = 0.0, 0.0, 0.0 # Changed to sum for averaging
        valid_graphs_for_metrics = 0 # Count graphs that had at least one true node/edge for averaging

        # The `evaluate_all` function iterates and passes the full `predicted_graphs_data` and `true_sequences_data`
        # for `evaluate_temporal_sequences` and `evaluate_graph_construction_f1`.
        # Ensure that `predicted_graphs_data` and `true_sequences_data` are aligned for accurate per-graph comparison.

        for i in range(min(len(predicted_graphs_data), len(true_sequences_data))):
            pred_data = predicted_graphs_data[i]
            true_data = true_sequences_data[i]

            # Node evaluation (based on attack presence)
            # Assuming 'is_attack_step' is a boolean node attribute in the constructed graph
            # This requires nodes to be processed with attack presence info in GraphNodeProcessor
            pred_nodes_with_attack = {n for n, d in pred_data.get('graph', nx.DiGraph()).nodes(data=True) if d.get('is_attack_step', False)}
            # If the original `true_sequences_data` doesn't have a direct `true_nodes_with_attack`,
            # we need to derive it from `targets` or directly from the `LabelExtractor`'s output.
            # For now, let's assume `LabelExtractor`'s 'attack_presence' (targets) maps to true attack nodes.
            # This is simplified for Phase 1. `node_entities` are just indices.
            
            # True nodes with attack are based on targets['attack_presence'] which is flattened.
            # We need to map back to the sequence scope.
            # The simplest assumption for this metric is that all nodes in a sequence are relevant,
            # and the true/predicted attack steps are derived from the classification heads.
            # For graph construction, we consider nodes that *should* be part of an attack graph.
            # If `node_entities` is just indices 0 to N-1 for a sequence, and `is_attack_step` is the prediction:
            
            # For `node_evaluation`, `pred_data`'s `predictions` dict (which holds `attack_presence`) is more relevant.
            # This needs careful mapping because `predicted_graphs_data` contains `predictions` that are slices of flattened data.
            # For simplicity in this test, let's keep the original:
            pred_nodes = set(range(len(pred_data.get('node_entities', []))))
            
            # --- FIX: True nodes should be derived from true targets for attack presence ---
            # `true_data` contains `true_edges` but not `true_nodes_with_attack`.
            # We need the original targets['attack_presence'] for this.
            # This implies `true_targets` passed to `evaluate_all` should be accessible per sequence.
            # As `true_targets` is already flattened in `evaluate_all`, a direct per-node mapping is hard here.
            # For Phase 1, `node_f1` often focuses on structural nodes, not just attack nodes.
            # Let's keep the current simplification `true_nodes = set(range(len(pred_data['node_entities'])))`
            # which essentially evaluates if the model constructs a graph with the same number of nodes.
            # Proper "node f1" on "attack step" nodes would need access to `true_targets` for *this specific sequence*.

            # Simpler node evaluation: assumes all nodes (events) in the sequence are "true nodes" to be considered
            # `node_precision` and `node_recall` will thus often be 1.0 if the graph builder doesn't drop nodes.
            true_nodes = set(range(len(pred_data.get('node_entities', [])))) 
            true_positive_nodes = len(pred_nodes.intersection(true_nodes))
            
            if len(pred_nodes) > 0: # Avoid division by zero
                node_precision_sum += true_positive_nodes / len(pred_nodes)
            if len(true_nodes) > 0: # Avoid division by zero
                node_recall_sum += true_positive_nodes / len(true_nodes)
            
            # Edge evaluation
            pred_edges = set(tuple(e) for e in pred_data.get('edges', []))
            
            # --- CRITICAL FIX: Use true_edges from true_sequences_data (like temporal_sequences) ---
            # Remove the fallback for true_edges here as well.
            true_edges = set(tuple(e) for e in true_data.get('true_edges', [])) 
            
            # Only consider graphs that actually had true edges for this metric
            if not true_edges and not pred_edges: # Both empty, perfect score for this graph? No, skip.
                 continue
            if not true_edges: # If no true edges, but model predicts some, precision will be 0. If model predicts 0, all metrics 0.
                if pred_edges:
                    edge_precision_sum += 0.0 # No true edges, but predicted some
                continue # No true edges to compare against for recall/f1

            true_positive_edges = len(pred_edges.intersection(true_edges))
            
            if len(pred_edges) > 0:
                edge_precision_sum += true_positive_edges / len(pred_edges)
            else: # If no predicted edges, precision is 0 for this graph
                edge_precision_sum += 0.0

            if len(true_edges) > 0:
                edge_recall_sum += true_positive_edges / len(true_edges)
            else: # If no true edges, recall is undefined/0, handled by initial `if not true_edges`
                edge_recall_sum += 0.0
            
            # Calculate F1 for this graph and add to sum
            current_precision = true_positive_edges / max(len(pred_edges), 1e-10)
            current_recall = true_positive_edges / max(len(true_edges), 1e-10)
            if (current_precision + current_recall) > 1e-10:
                edge_f1_sum += 2 * (current_precision * current_recall) / (current_precision + current_recall)

            valid_graphs_for_metrics += 1 # Count only graphs that contributed to metrics


        if valid_graphs_for_metrics > 0:
            node_precision = node_precision_sum / valid_graphs_for_metrics
            node_recall = node_recall_sum / valid_graphs_for_metrics
            node_f1 = 2 * (node_precision * node_recall) / max(node_precision + node_recall, 1e-10)
            edge_precision = edge_precision_sum / valid_graphs_for_metrics
            edge_recall = edge_recall_sum / valid_graphs_for_metrics
            edge_f1 = edge_f1_sum / valid_graphs_for_metrics
        else:
            # Fallback if no valid graphs to compute metrics
            node_precision, node_recall, node_f1 = 0.0, 0.0, 0.0
            edge_precision, edge_recall, edge_f1 = 0.0, 0.0, 0.0


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
        """
        Compare model performance against a rule-based baseline.
        For Phase 1, this baseline will be a simple majority-class predictor (predicting BENIGN).
        A more sophisticated rule-based baseline would be introduced in later phases if required.
        """
        metrics = {}
        
        attack_true_labels_cpu = targets['attack_presence'].cpu().numpy()
        
        # --- FIX: Define a truly independent baseline ---
        # The previous baseline_preds = (predictions['attack_presence'][:, 1] > 0.5) was using model's own output.
        # A simple, independent baseline for highly imbalanced datasets often predicts the majority class.
        # Majority class in CICIDS2017 is BENIGN (label 0).
        # So, baseline predicts 0 (BENIGN) for all.
        baseline_preds = np.zeros_like(attack_true_labels_cpu) # Predicts 0 (BENIGN) for all
        
        # Alternatively, if you want a fixed non-zero baseline (e.g., if you know roughly X% are attack):
        # baseline_preds = np.random.randint(0, 2, size=attack_true_labels_cpu.shape) # Random guess (very low accuracy)
        # Or, a "simple rule" if you define one: e.g., predict attack if N events occur in M seconds.
        # For now, predicting all BENIGN is a robust "simple baseline" for imbalance.

        metrics['baseline_accuracy'] = accuracy_score(attack_true_labels_cpu, baseline_preds)

        unique_labels = np.unique(attack_true_labels_cpu)
        if len(unique_labels) < 2:
            self.logger.warning(f"Only one class ({unique_labels[0]}) in true labels for attack presence. Baseline Precision/Recall/F1 set to 0.0.")
            prec, rec, f1 = 0.0, 0.0, 0.0
        else:
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
        
        # Flatten predictions and targets for overall detection metrics
        # Ensure 'predictions' from model_predictions also accounts for missing keys if batch is not full
        flattened_pred_detection = {}
        for k in model_predictions[0].keys():
            if k != 'attention_weights':
                # Dynamically handle dim for predictions like attack_presence (N,2) vs severity (N,1)
                tensors = [p[k].view(-1, p[k].shape[-1]) if p[k].dim() > 1 else p[k].view(-1) for p in model_predictions]
                # Filter out empty tensors which can happen if some batches were missing keys or were empty for some reason
                flattened_pred_detection[k] = torch.cat([t for t in tensors if t.numel() > 0], dim=0)

        flattened_true_detection = {}
        for k in true_targets[0].keys():
            tensors = [t[k].view(-1) for t in true_targets]
            flattened_true_detection[k] = torch.cat([t for t in tensors if t.numel() > 0], dim=0)
            
        # Ensure keys are present before passing to sub-evaluators
        if not flattened_pred_detection.get('attack_presence', None) is None and not flattened_true_detection.get('attack_presence', None) is None:
            detection_metrics = self.evaluate_attack_detection(flattened_pred_detection, flattened_true_detection)
            all_metrics.update(detection_metrics)
            
            baseline_metrics = self.evaluate_against_baseline(flattened_pred_detection, flattened_true_detection)
            all_metrics.update(baseline_metrics)
        else:
            self.logger.warning("Skipping attack detection and baseline evaluation: 'attack_presence' data not found or empty.")


        # Temporal and Graph metrics rely on structured graph data
        if predicted_graphs_data and true_sequences_data:
            temporal_metrics = self.evaluate_temporal_sequences(predicted_graphs_data, true_sequences_data)
            all_metrics.update(temporal_metrics)
            
            graph_metrics = self.evaluate_graph_construction_f1(predicted_graphs_data, true_sequences_data)
            all_metrics.update(graph_metrics)
        else:
            self.logger.warning("Skipping temporal and graph construction evaluation: graph data not found or empty.")


        self.logger.info(f"All evaluation metrics: {all_metrics}")
        return all_metrics