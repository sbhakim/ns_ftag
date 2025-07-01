# src/tests/test_integration.py

import unittest
import torch
import os
import sys
import numpy as np
import pandas as pd
import logging
import time

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from evaluators.neural_evaluator import NeuralAttackGraphEvaluator
from neural_components.neural_pipeline import NeuralAttackGraphPipeline
from data_processors.dataset import SecurityEventDataset
from data_processors.cicids2017_processor import CICIDS2017Processor
from data_processors.entity_manager import EntityManager
from data_processors.relationship_extractor import RelationshipExtractor
from data_processors.feature_extractor import FeatureExtractor
from data_processors.label_extractor import LabelExtractor
from data_processors.batch_collator import custom_collate_fn
from config.neural_config import NeuralConfig
from utils.device_manager import DeviceManager

class TestIntegration(unittest.TestCase):
    def setUp(self):
        self.config = NeuralConfig()
        self.config.entity_vocab_size = 1000
        self.config.action_vocab_size = 100
        self.config.embedding_dim = 64
        self.config.num_attack_types = 6
        self.config.num_mitre_techniques = 6
        self.config.security_feature_dim = 10
        self.device_manager = DeviceManager()
        self.logger = logging.getLogger(__name__)
        if not self.logger.handlers:
            logging.basicConfig(
                level=logging.INFO,
                format='%(asctime)s - %(levelname)s - %(message)s',
                handlers=[logging.StreamHandler()]
            )

    def create_dummy_batch(self, batch_size=2, num_nodes=10):
        """Creates a dummy batch that mimics the output of SecurityEventDataset."""
        entities = torch.randint(0, self.config.entity_vocab_size, (batch_size, num_nodes))
        actions = torch.randint(0, self.config.action_vocab_size, (batch_size, num_nodes))
        total_nodes_in_batch = batch_size * num_nodes
        edge_index = torch.randint(0, total_nodes_in_batch, (2, total_nodes_in_batch * 2))
        security_features = torch.randn(batch_size, num_nodes, self.config.security_feature_dim)
        attack_types = torch.randint(0, self.config.num_attack_types, (batch_size * num_nodes,))
        mitre_techniques = attack_types.clone()
        targets = {
            'attack_presence': torch.tensor([1 if t > 0 else 0 for t in attack_types]),
            'attack_type': attack_types,
            'mitre_technique': mitre_techniques,
            'severity': torch.rand(batch_size * num_nodes, 1),
            'confidence': torch.rand(batch_size * num_nodes, 1)
        }
        return {
            'entities': entities,
            'actions': actions,
            'edge_index': edge_index,
            'security_features': security_features,
            'targets': targets,
            'true_edges': [[[0, 1]] for _ in range(batch_size)]
        }

    def test_end_to_end_pipeline(self):
        """Test complete neural pipeline forward pass."""
        self.logger.info("Starting test_end_to_end_pipeline")
        model = NeuralAttackGraphPipeline(self.config).to(self.device_manager.get_device())
        batch = self.create_dummy_batch(batch_size=self.config.batch_size, num_nodes=self.config.max_sequence_length // 2)
        batch_on_device = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch_on_device['targets'] = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch['targets'].items()}

        predictions = model(batch_on_device)
        self.assertIn('attack_presence', predictions)
        self.assertIn('attack_type', predictions)
        self.assertIn('mitre_technique', predictions)
        self.assertIn('severity', predictions)
        self.assertIn('confidence', predictions)
        self.assertIn('attention_weights', predictions)
        self.assertEqual(predictions['attack_presence'].shape[0], self.config.batch_size * (self.config.max_sequence_length // 2))
        self.assertEqual(predictions['attack_type'].shape[0], self.config.batch_size * (self.config.max_sequence_length // 2))
        self.assertEqual(predictions['mitre_technique'].shape[0], self.config.batch_size * (self.config.max_sequence_length // 2))
        self.assertEqual(predictions['severity'].shape, (self.config.batch_size * (self.config.max_sequence_length // 2), 1))
        self.assertEqual(predictions['confidence'].shape, (self.config.batch_size * (self.config.max_sequence_length // 2), 1))
        self.assertIsInstance(predictions['attention_weights'], list)
        self.assertGreater(len(predictions['attention_weights']), 0)
        self.assertEqual(predictions['attention_weights'][0].shape[0], self.config.batch_size)
        self.assertEqual(predictions['attention_weights'][0].shape[1], self.config.gnn_num_heads)
        self.assertEqual(predictions['attention_weights'][0].shape[2], batch['entities'].shape[1])
        self.logger.info("Completed test_end_to_end_pipeline")

    def test_graph_generation(self):
        """Test attack graph generation method with enhanced validation."""
        self.logger.info("Starting test_graph_generation")
        model = NeuralAttackGraphPipeline(self.config).to(self.device_manager.get_device())
        batch = self.create_dummy_batch(batch_size=1, num_nodes=5)
        batch_on_device = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch_on_device['targets'] = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch['targets'].items()}

        attack_graphs_data = model.get_attack_graph(batch_on_device)
        self.assertIsInstance(attack_graphs_data, list)
        self.assertEqual(len(attack_graphs_data), 1)
        first_graph_data = attack_graphs_data[0]
        self.assertIn('node_entities', first_graph_data)
        self.assertIn('attention_map', first_graph_data)
        self.assertIn('predictions', first_graph_data)
        self.assertIn('edges', first_graph_data)
        self.assertIsInstance(first_graph_data['node_entities'], list)
        self.assertIsInstance(first_graph_data['attention_map'], np.ndarray)
        self.assertIsInstance(first_graph_data['predictions'], dict)
        self.assertIsInstance(first_graph_data['edges'], list)
        self.assertEqual(first_graph_data['attention_map'].shape, (5, 5))
        self.assertEqual(len(first_graph_data['node_entities']), 5, "Node count mismatch")
        self.assertTrue(all(key in first_graph_data['predictions'] for key in ['attack_presence', 'attack_type', 'severity', 'confidence']), "Missing prediction keys")
        true_edges = set(tuple(edge) for edge in batch['true_edges'][0])
        predicted_edges = set(tuple(edge) for edge in first_graph_data['edges'])
        self.assertTrue(len(predicted_edges) > 0, "No edges generated")
        self.assertTrue(true_edges.issubset(predicted_edges), f"True edges {true_edges} not in predicted edges {predicted_edges}")
        self.logger.info("Completed test_graph_generation")

    def test_data_processing_pipeline(self):
        """Test the integration of data processing components with a mock dataset."""
        self.logger.info("Starting test_data_processing_pipeline")
        mock_data = pd.DataFrame({
            'Flow ID': ['1', '2', '3', '4', '5'],
            ' Source IP': ['192.168.1.1', '192.168.1.1', '192.168.1.2', '192.168.1.3', '192.168.1.3'],
            ' Source Port': [12345, 12346, 12347, 12348, 12349],
            ' Destination IP': ['10.0.0.1', '10.0.0.2', '10.0.0.1', '10.0.0.3', '10.0.0.2'],
            ' Destination Port': [80, 80, 443, 80, 443],
            ' Protocol': ['TCP', 'TCP', 'UDP', 'TCP', 'UDP'],
            ' Timestamp': ['7/7/2017 15:00:00', '7/7/2017 15:00:01', '7/7/2017 15:00:02', '7/7/2017 15:00:03', '7/7/2017 15:00:04'],
            ' Total Fwd Packets': [10, 20, 15, 25, 30],
            ' Total Backward Packets': [5, 10, 8, 12, 15],
            ' Total Length of Fwd Packets': [1000, 2000, 1500, 2500, 3000],
            ' Total Length of Bwd Packets': [500, 1000, 800, 1200, 1500],
            ' Flow Duration': [1000000, 2000000, 1500000, 2500000, 3000000],
            ' Flow Bytes/s': [1000.0, 1500.0, 1200.0, 1800.0, 2000.0],
            ' Flow Packets/s': [15.0, 30.0, 23.0, 37.0, 45.0],
            ' Fwd Packet Length Mean': [100.0, 100.0, 100.0, 100.0, 100.0],
            ' Bwd Packet Length Mean': [100.0, 100.0, 100.0, 100.0, 100.0],
            ' Label': ['BENIGN', 'DDoS', 'PortScan', 'Web Attack', 'Bot']
        })

        processor = CICIDS2017Processor(self.config)
        processed_df = processor.preprocess(processor.load_data_from_df(mock_data))
        entity_manager = EntityManager()
        relationship_extractor = RelationshipExtractor(self.config)
        feature_extractor = FeatureExtractor(self.config)
        label_extractor = LabelExtractor(self.config)

        expected_columns = ['flow_id', 'source_ip', 'destination_ip', 'source_port', 'destination_port', 'protocol', 'timestamp', 'label']
        self.assertTrue(all(col in processed_df.columns for col in expected_columns), f"Missing columns: {set(expected_columns) - set(processed_df.columns)}")

        entities = entity_manager.extract_security_entities(processed_df)
        relationships = relationship_extractor.extract_relationships(processed_df)
        features = feature_extractor.extract_security_features(processed_df)
        labels = label_extractor.extract_labels(processed_df)

        self.assertEqual(len(entities['source_entity_ids']), len(processed_df), "Entity extraction length mismatch")
        self.assertGreaterEqual(len(relationships), 0, "No relationships extracted")
        self.assertEqual(len(features), len(processed_df), "Feature extraction length mismatch")
        self.assertEqual(len(labels['attack_presence']), len(processed_df), "Label extraction length mismatch")
        self.assertEqual(len(labels['mitre_technique']), len(processed_df), "MITRE technique extraction length mismatch")
        self.logger.info(f"Completed test_data_processing_pipeline: {len(processed_df)} rows processed")

    def test_data_processing_edge_cases(self):
        """Test data processing with malformed data."""
        self.logger.info("Starting test_data_processing_edge_cases")
        mock_data = pd.DataFrame({
            'Flow ID': ['1', '2', '3'],
            ' Source IP': ['192.168.1.1', None, '192.168.1.2'],
            ' Destination Port': [80, 80, 443],
            ' Protocol': ['TCP', 'TCP', 'UDP'],
            ' Timestamp': ['invalid', '7/7/2017 15:00:01', None],
            ' Label': ['BENIGN', 'DDoS', 'BENIGN']
        })

        processor = CICIDS2017Processor(self.config)
        processed_df = processor.preprocess(processor.load_data_from_df(mock_data))
        self.assertEqual(len(processed_df), 2, "Should drop rows with NaN in essential columns")
        self.assertFalse(processed_df['timestamp'].isna().any(), "Timestamps should be filled")
        self.logger.info("Completed test_data_processing_edge_cases")

    def test_data_processing_missing_columns(self):
        """Test data processing with missing essential columns."""
        self.logger.info("Starting test_data_processing_missing_columns")
        mock_data = pd.DataFrame({
            'Flow ID': ['1', '2', '3'],
            ' Source Port': [12345, 12346, 12347],
            ' Destination Port': [80, 80, 443],
            ' Protocol': ['TCP', 'TCP', 'UDP'],
            ' Timestamp': ['7/7/2017 15:00:00', '7/7/2017 15:00:01', '7/7/2017 15:00:02'],
            ' Label': ['BENIGN', 'DDoS', 'BENIGN']
        })
        processor = CICIDS2017Processor(self.config)
        with self.assertRaises(ValueError, msg="Should raise error for missing essential columns"):
            processor.preprocess(processor.load_data_from_df(mock_data))
        self.logger.info("Completed test_data_processing_missing_columns")

    def test_data_processing_real_subset(self):
        """Test data processing with a subset of real CICIDS2017 data."""
        self.logger.info("Starting test_data_processing_real_subset")
        processor = CICIDS2017Processor(self.config)
        start_time = time.time()
        df = processor.load_data(self.config.data_path).head(1000)
        processed_df = processor.preprocess(df)
        entity_manager = EntityManager()
        relationship_extractor = RelationshipExtractor(self.config)
        entities = entity_manager.extract_security_entities(processed_df)
        relationships = relationship_extractor.extract_relationships(processed_df)
        self.assertEqual(len(entities['source_entity_ids']), len(processed_df), "Entity extraction length mismatch")
        self.assertGreaterEqual(len(relationships), 0, "No relationships extracted")
        elapsed_time = time.time() - start_time
        processing_rate = len(processed_df) / max(elapsed_time, 1e-10)
        self.assertGreater(processing_rate, 1000, f"Processing rate {processing_rate:.2f} events/second below required 1000 events/second")
        self.logger.info(f"Completed test_data_processing_real_subset: {len(processed_df)} rows processed in {elapsed_time:.3f} seconds")

    def test_data_processing_timestamp_formats(self):
        """Test timestamp parsing with CICIDS2017 dataset formats."""
        self.logger.info("Starting test_data_processing_timestamp_formats")
        mock_data = pd.DataFrame({
            'Flow ID': ['1', '2', '3'],
            ' Source IP': ['192.168.1.1', '192.168.1.1', '192.168.1.2'],
            ' Source Port': [12345, 12346, 12347],
            ' Destination IP': ['10.0.0.1', '10.0.0.2', '10.0.0.1'],
            ' Destination Port': [80, 80, 443],
            ' Protocol': ['TCP', 'TCP', 'UDP'],
            ' Timestamp': ['7/7/2017 15:00:00', '07/07/2017 15:00:01', '7-7-2017 15:00:02'],
            ' Label': ['BENIGN', 'DDoS', 'BENIGN']
        })
        processor = CICIDS2017Processor(self.config)
        processed_df = processor.preprocess(processor.load_data_from_df(mock_data))
        self.assertFalse(processed_df['timestamp'].isna().any(), "Timestamps should be parsed without NaNs")
        self.logger.info("Completed test_data_processing_timestamp_formats")

    def test_pipeline_evaluation(self):
        """Test pipeline with evaluator using dummy data."""
        self.logger.info("Starting test_pipeline_evaluation")
        model = NeuralAttackGraphPipeline(self.config).to(self.device_manager.get_device())
        evaluator = NeuralAttackGraphEvaluator(self.config)
        batch = self.create_dummy_batch(batch_size=self.config.batch_size, num_nodes=self.config.max_sequence_length // 2)
        batch_on_device = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch_on_device['targets'] = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch['targets'].items()}

        predictions = model(batch_on_device)
        graphs_data = model.get_attack_graph(batch_on_device)
        true_sequences_data = [{'true_edges': batch['true_edges'][i]} for i in range(len(batch['true_edges']))]

        metrics = evaluator.evaluate_all([predictions], [batch_on_device['targets']], graphs_data, true_sequences_data)
        self.assertIn('attack_accuracy', metrics, "Attack accuracy metric missing")
        self.assertIn('attack_f1', metrics, "Attack F1-score metric missing")
        self.assertIn('temporal_accuracy', metrics, "Temporal accuracy metric missing")
        self.assertIn('node_f1', metrics, "Node F1-score metric missing")
        self.assertIn('edge_f1', metrics, "Edge F1-score metric missing")
        self.assertIn('baseline_accuracy', metrics, "Baseline accuracy metric missing")
        self.assertGreater(metrics['attack_accuracy'], 0.0, "Attack accuracy should be non-zero")
        self.assertGreater(metrics['temporal_accuracy'], 0.0, "Temporal accuracy should be non-zero")
        self.logger.info(f"Completed test_pipeline_evaluation: Metrics {metrics}")

    def test_pipeline_robustness(self):
        """Test pipeline robustness with noisy inputs."""
        self.logger.info("Starting test_pipeline_robustness")
        model = NeuralAttackGraphPipeline(self.config).to(self.device_manager.get_device())
        batch = self.create_dummy_batch(batch_size=1, num_nodes=5)
        batch['security_features'] += torch.randn_like(batch['security_features']) * 0.1
        batch_on_device = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch_on_device['targets'] = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch['targets'].items()}

        predictions = model(batch_on_device)
        self.assertIn('attack_presence', predictions, "Pipeline failed with noisy inputs")
        self.assertEqual(predictions['attack_presence'].shape[0], self.config.batch_size * 5, "Shape mismatch with noisy inputs")
        self.logger.info("Completed test_pipeline_robustness")

    def test_pipeline_temporal_shuffling(self):
        """Test pipeline robustness with shuffled edges."""
        self.logger.info("Starting test_pipeline_temporal_shuffling")
        model = NeuralAttackGraphPipeline(self.config).to(self.device_manager.get_device())
        batch = self.create_dummy_batch(batch_size=1, num_nodes=5)
        batch['edge_index'] = torch.flip(batch['edge_index'], dims=[1])
        batch_on_device = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch_on_device['targets'] = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch['targets'].items()}
        predictions = model(batch_on_device)
        self.assertIn('attack_presence', predictions, "Pipeline failed with shuffled edges")
        self.logger.info("Completed test_pipeline_temporal_shuffling")

    def test_pipeline_adversarial_labels(self):
        """Test pipeline robustness with adversarial label manipulation."""
        self.logger.info("Starting test_pipeline_adversarial_labels")
        model = NeuralAttackGraphPipeline(self.config).to(self.device_manager.get_device())
        batch = self.create_dummy_batch(batch_size=1, num_nodes=5)
        batch['targets']['attack_type'] = torch.roll(batch['targets']['attack_type'], shifts=1, dims=0)
        batch['targets']['mitre_technique'] = torch.roll(batch['targets']['mitre_technique'], shifts=1, dims=0)
        batch_on_device = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch.items()}
        batch_on_device['targets'] = {k: (v.to(self.device_manager.get_device()) if isinstance(v, torch.Tensor) else v) for k, v in batch['targets'].items()}
        predictions = model(batch_on_device)
        self.assertIn('attack_presence', predictions, "Pipeline failed with adversarial labels")
        self.assertEqual(predictions['attack_type'].shape[0], self.config.batch_size * 5, "Shape mismatch with adversarial labels")
        self.logger.info("Completed test_pipeline_adversarial_labels")

if __name__ == '__main__':
    unittest.main()