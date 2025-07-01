# src/tests/test_neural_components.py

import unittest
import torch
import torch.nn.functional as F
import sys
import os
import pandas as pd
import numpy as np

# Add src to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))

from neural_components.security_attention import SecurityAwareAttention
from neural_components.gnn_layer import SecurityGNNLayer
from neural_components.tcn_layer import TemporalConvolutionalNetwork
from neural_components.attack_classifier import AttackStepClassifier
from data_processors.cicids2017_processor import CICIDS2017Processor
from data_processors.entity_manager import EntityManager
from data_processors.relationship_extractor import RelationshipExtractor
from data_processors.feature_extractor import FeatureExtractor
from data_processors.label_extractor import LabelExtractor
from config.neural_config import NeuralConfig

class TestNeuralComponents(unittest.TestCase):
    def setUp(self):
        self.config = NeuralConfig()
        self.config.batch_size = 4
        self.config.max_sequence_length = 20
        self.config.gnn_hidden_dims = [64, 128, 64]
        self.config.gnn_num_heads = 4
        self.config.security_feature_dim = 10
        self.config.embedding_dim = 32
        self.config.tcn_channels = [32, 64]
        self.config.tcn_kernel_size = 3
        self.config.num_attack_types = 6
        self.config.num_mitre_techniques = 6
        self.num_nodes = 10

    def test_security_attention(self):
        """Test security-aware attention mechanism"""
        attention = SecurityAwareAttention(self.config.gnn_hidden_dims[0], self.config.gnn_num_heads, self.config.security_feature_dim)
        node_features = torch.randn(self.config.batch_size, self.num_nodes, self.config.gnn_hidden_dims[0])
        security_features = torch.randn(self.config.batch_size, self.num_nodes, self.config.security_feature_dim)
        output, attention_weights = attention(node_features, None, security_features)
        self.assertEqual(output.shape, node_features.shape)
        self.assertEqual(attention_weights.shape, (self.config.batch_size, self.config.gnn_num_heads, self.num_nodes, self.num_nodes))

    def test_security_gnn_layer(self):
        """Test SecurityGNNLayer"""
        gnn_layer = SecurityGNNLayer(self.config.embedding_dim * 2, self.config.gnn_hidden_dims[0], self.config.gnn_num_heads,
                                     security_feature_dim=self.config.security_feature_dim)
        x = torch.randn(self.config.batch_size, self.num_nodes, self.config.embedding_dim * 2)
        security_features = torch.randn(self.config.batch_size, self.num_nodes, self.config.security_feature_dim)
        edge_index = torch.randint(0, self.num_nodes, (2, self.num_nodes * 2))
        output, attention_weights = gnn_layer(x, edge_index, security_features)
        self.assertEqual(output.shape, (self.config.batch_size, self.num_nodes, self.config.gnn_hidden_dims[0]))
        self.assertEqual(attention_weights.shape, (self.config.batch_size, self.config.gnn_num_heads, self.num_nodes, self.num_nodes))

    def test_tcn_layer(self):
        """Test TemporalConvolutionalNetwork"""
        tcn = TemporalConvolutionalNetwork(self.config.gnn_hidden_dims[-1], self.config.tcn_channels, self.config.tcn_kernel_size, dropout=0.2)
        x = torch.randn(self.config.batch_size, self.num_nodes, self.config.gnn_hidden_dims[-1])
        output = tcn(x)
        self.assertEqual(output.shape, (self.config.batch_size, self.num_nodes, self.config.tcn_channels[-1]))

    def test_attack_classifier(self):
        """Test multi-task attack classifier"""
        classifier = AttackStepClassifier(self.config.gnn_hidden_dims[-1], self.config.num_attack_types,
                                         self.config.num_mitre_techniques)
        embeddings = torch.randn(self.config.batch_size, self.num_nodes, self.config.gnn_hidden_dims[-1])
        predictions = classifier(embeddings)
        self.assertIn('attack_presence', predictions)
        self.assertIn('attack_type', predictions)
        self.assertIn('mitre_technique', predictions)
        self.assertIn('severity', predictions)
        self.assertIn('confidence', predictions)
        self.assertEqual(predictions['attack_presence'].shape, (self.config.batch_size * self.num_nodes, 2))
        self.assertEqual(predictions['attack_type'].shape, (self.config.batch_size * self.num_nodes, self.config.num_attack_types))
        self.assertEqual(predictions['mitre_technique'].shape, (self.config.batch_size * self.num_nodes, self.config.num_mitre_techniques))
        self.assertEqual(predictions['severity'].shape, (self.config.batch_size * self.num_nodes, 1))
        self.assertEqual(predictions['confidence'].shape, (self.config.batch_size * self.num_nodes, 1))

    def test_cicids2017_processor(self):
        """Test CICIDS2017Processor with mock data"""
        mock_data = pd.DataFrame({
            'Flow ID': ['1', '2', '3'],
            ' Source IP': ['192.168.1.1', '192.168.1.1', '192.168.1.2'],
            ' Source Port': [12345, 12346, 12347],
            ' Destination IP': ['10.0.0.1', '10.0.0.2', '10.0.0.1'],
            ' Destination Port': [80, 80, 443],
            ' Protocol': ['TCP', 'TCP', 'UDP'],
            ' Timestamp': ['7/7/2017 15:00:00', '7/7/2017 15:00:01', '7/7/2017 15:00:02'],
            ' Label': ['BENIGN', 'DDoS', 'BENIGN']
        })
        processor = CICIDS2017Processor(self.config)
        df = processor.load_data_from_df(mock_data)
        expected_columns = ['flow_id', 'source_ip', 'destination_ip', 'source_port', 'destination_port', 'protocol', 'timestamp', 'label']
        self.assertTrue(all(col in df.columns for col in expected_columns), f"Missing columns: {set(expected_columns) - set(df.columns)}")
        self.assertEqual(len(df), 3, "Unexpected DataFrame length")

    def test_entity_manager(self):
        """Test EntityManager with mock data"""
        mock_data = pd.DataFrame({
            'Flow ID': ['1', '2'],
            ' Source IP': ['192.168.1.1', '192.168.1.2'],
            ' Source Port': [12345, 12346],
            ' Destination IP': ['10.0.0.1', '10.0.0.2'],
            ' Destination Port': [80, 443],
            ' Protocol': ['TCP', 'UDP'],
            ' Timestamp': ['7/7/2017 15:00:00', '7/7/2017 15:00:01']
        })
        processor = CICIDS2017Processor(self.config)
        df = processor.load_data_from_df(mock_data)
        entity_manager = EntityManager()
        entities = entity_manager.extract_security_entities(df)
        self.assertEqual(len(entities['source_entity_ids']), 2)
        self.assertEqual(len(entities['target_entity_ids']), 2)
        self.assertEqual(len(entities['action_ids']), 2)
        self.assertEqual(entity_manager.get_vocab_sizes()['entity_vocab_size'], 4)
        self.assertEqual(entity_manager.get_vocab_sizes()['action_vocab_size'], 2)

    def test_relationship_extractor(self):
        """Test RelationshipExtractor with mock data"""
        mock_data = pd.DataFrame({
            'Flow ID': ['1', '2', '3'],
            ' Source IP': ['192.168.1.1', '192.168.1.1', '192.168.1.2'],
            ' Destination IP': ['10.0.0.1', '10.0.0.1', '10.0.0.2'],
            ' Timestamp': ['7/7/2017 15:00:00', '7/7/2017 15:00:01', '7/7/2017 15:00:02'],
            ' Label': ['BENIGN', 'DDoS', 'BENIGN']
        })
        processor = CICIDS2017Processor(self.config)
        df = processor.load_data_from_df(mock_data)
        extractor = RelationshipExtractor(self.config)
        relationships = extractor.extract_relationships(df)
        self.assertGreaterEqual(len(relationships), 1, "Should extract at least one relationship")
        true_edges = extractor.extract_true_edges(df)
        self.assertGreaterEqual(len(true_edges), 0, "True edges should be extracted or empty")

    def test_feature_extractor(self):
        """Test FeatureExtractor with mock data"""
        mock_data = pd.DataFrame({
            'Flow ID': ['1', '2'],
            ' Source IP': ['192.168.1.1', '192.168.1.2'],
            ' Total Fwd Packets': [10, 20],
            ' Total Backward Packets': [5, 10],
            ' Total Length of Fwd Packets': [1000, 2000],
            ' Total Length of Bwd Packets': [500, 1000],
            ' Flow Duration': [1000000, 2000000],
            ' Flow Bytes/s': [1000.0, 1500.0],
            ' Flow Packets/s': [15.0, 30.0],
            ' Fwd Packet Length Mean': [100.0, 100.0],
            ' Bwd Packet Length Mean': [100.0, 100.0],
            ' Label': ['BENIGN', 'DDoS']
        })
        processor = CICIDS2017Processor(self.config)
        df = processor.load_data_from_df(mock_data)
        extractor = FeatureExtractor(self.config)
        features = extractor.extract_security_features(df)
        self.assertEqual(len(features), 2)
        self.assertEqual(len(features[0]), self.config.security_feature_dim)
        self.assertEqual(features[0][-1], 0.0)
        self.assertEqual(features[1][-1], 1.0)

    def test_label_extractor(self):
        """Test LabelExtractor with mock data"""
        mock_data = pd.DataFrame({
            'Flow ID': ['1', '2'],
            ' Source IP': ['192.168.1.1', '192.168.1.2'],
            ' Label': ['BENIGN', 'DDoS']
        })
        processor = CICIDS2017Processor(self.config)
        df = processor.load_data_from_df(mock_data)
        extractor = LabelExtractor(self.config)
        labels = extractor.extract_labels(df)
        self.assertEqual(len(labels['attack_presence']), 2)
        self.assertEqual(labels['attack_presence'][0].item(), 0)
        self.assertEqual(labels['attack_presence'][1].item(), 1)
        self.assertEqual(labels['attack_type'][0].item(), 0)
        self.assertEqual(labels['attack_type'][1].item(), 1)

if __name__ == '__main__':
    unittest.main()