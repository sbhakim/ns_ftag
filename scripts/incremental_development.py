# scripts/incremental_development.py


import sys
import os
import torch
import numpy as np
import random
from typing import Dict, Any, List

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config.neural_config import NeuralConfig
from utils.device_manager import DeviceManager
from utils.performance_monitor import PerformanceMonitor
from data_processors.cicids2017_processor import CICIDS2017Processor
from data_processors.entity_manager import EntityManager
from data_processors.relationship_extractor import RelationshipExtractor
from data_processors.feature_extractor import FeatureExtractor
from data_processors.label_extractor import LabelExtractor
from data_processors.dataset import SecurityEventDataset
from data_processors.batch_collator import custom_collate_fn
from neural_components.security_attention import SecurityAwareAttention
from neural_components.gnn_layer import SecurityGNNLayer
from neural_components.tcn_layer import TemporalConvolutionalNetwork
from neural_components.attack_classifier import AttackStepClassifier
from neural_components.neural_pipeline import NeuralAttackGraphPipeline
from graph_builders.dynamic_graph_builder import DynamicAttackGraphBuilder
from evaluators.neural_evaluator import NeuralAttackGraphEvaluator
from utils.training_manager import TrainingManager

def set_seed(seed: int):
    """Set seeds for reproducibility."""
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def create_dummy_batch_for_test(config: Any, batch_size: int = 2, num_nodes: int = 10) -> Dict[str, Any]:
    """Creates a dummy batch that mimics the output of SecurityEventDataset."""
    entities = torch.randint(0, config.entity_vocab_size, (batch_size, num_nodes))
    actions = torch.randint(0, config.action_vocab_size, (batch_size, num_nodes))
    total_nodes_in_batch = batch_size * num_nodes
    edge_index = torch.randint(0, total_nodes_in_batch, (2, total_nodes_in_batch * 2))
    security_features = torch.randn(batch_size, num_nodes, config.security_feature_dim)
    targets = {
        'attack_presence': torch.randint(0, 2, (batch_size * num_nodes,)),
        'attack_type': torch.randint(0, config.num_attack_types, (batch_size * num_nodes,)),
        'mitre_technique': torch.randint(0, config.num_mitre_techniques, (batch_size * num_nodes,)),
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

def test_component(component_name: str, config: NeuralConfig, device: torch.device, monitor: PerformanceMonitor):
    """Test individual components incrementally."""
    set_seed(42)
    print(f"--- Testing {component_name} ---")

    if component_name == "config":
        print(f"Configuration loaded: {config}")

    elif component_name == "device":
        print(f"Device: {device}")
        dummy_tensor = torch.randn(10).to(device)
        print(f"Dummy tensor on device: {dummy_tensor.device}")

    elif component_name == "data_processor":
        processor = CICIDS2017Processor(config)
        entity_manager = EntityManager()
        df = processor.load_data(config.data_path)
        print(f"Data loaded, shape: {df.shape}")
        sample_events = df.head(config.min_sequence_length + 5)
        entities = entity_manager.extract_security_entities(sample_events)
        print(f"Extracted entity/action IDs: {len(entities['source_entity_ids'])} events")
        print(f"Entity vocab size: {entity_manager.get_vocab_sizes()['entity_vocab_size']}")
        relationships = RelationshipExtractor(config).extract_relationships(sample_events)
        print(f"Extracted {len(relationships)} relationships")

    elif component_name == "dataset":
        processor = CICIDS2017Processor(config)
        entity_manager = EntityManager()
        relationship_extractor = RelationshipExtractor(config)
        feature_extractor = FeatureExtractor(config)
        label_extractor = LabelExtractor(config)
        dataset = SecurityEventDataset(
            config.data_path, config, processor, entity_manager,
            relationship_extractor, feature_extractor, label_extractor
        )
        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: entities shape {batch['entities'].shape}, actions shape {batch['actions'].shape}")
            print(f"Batch {i}: security_features shape {batch['security_features'].shape}")
            print(f"Batch {i}: edge_index shape {batch['edge_index'].shape}")
            print(f"Batch {i}: targets keys {batch['targets'].keys()}")
            print(f"Batch {i}: true_edges count {len(batch['true_edges'])}")
            if i == 0: break

    elif component_name == "attention":
        # FIX: Move the attention module to the device
        attention = SecurityAwareAttention(config.gnn_hidden_dims[0], config.gnn_num_heads, config.security_feature_dim).to(device)
        node_features = torch.randn(config.batch_size, config.max_sequence_length // 2, config.gnn_hidden_dims[0]).to(device)
        security_features = torch.randn(config.batch_size, config.max_sequence_length // 2, config.security_feature_dim).to(device)
        output, attn_weights = attention(node_features, None, security_features)
        print(f"Attention output shape: {output.shape}, weights shape: {attn_weights.shape}")

    elif component_name == "gnn_layer":
        gnn_layer = SecurityGNNLayer(config.embedding_dim * 2, config.gnn_hidden_dims[0], config.gnn_num_heads, security_feature_dim=config.security_feature_dim).to(device)
        x = torch.randn(config.batch_size, config.max_sequence_length // 2, config.embedding_dim * 2).to(device)
        security_features = torch.randn(config.batch_size, config.max_sequence_length // 2, config.security_feature_dim).to(device)
        total_nodes = config.batch_size * (config.max_sequence_length // 2)
        edge_index = torch.randint(0, total_nodes, (2, total_nodes * 2)).to(device)
        output, attn_weights = gnn_layer(x, edge_index, security_features)
        print(f"GNN layer output shape: {output.shape}, weights shape: {attn_weights.shape}")

    elif component_name == "tcn":
        tcn = TemporalConvolutionalNetwork(config.gnn_hidden_dims[-1], config.tcn_channels, config.tcn_kernel_size, config.gnn_dropout).to(device)
        x = torch.randn(config.batch_size, config.max_sequence_length // 2, config.gnn_hidden_dims[-1]).to(device)
        output = tcn(x)
        print(f"TCN output shape: {output.shape}")

    elif component_name == "attack_classifier":
        classifier = AttackStepClassifier(config.gnn_hidden_dims[-1], config.num_attack_types, config.num_mitre_techniques).to(device)
        embeddings = torch.randn(config.batch_size, config.max_sequence_length // 2, config.gnn_hidden_dims[-1]).to(device)
        predictions = classifier(embeddings)
        print(f"Classifier predictions keys: {predictions.keys()}")
        print(f"Attack presence shape: {predictions['attack_presence'].shape}")

    elif component_name == "neural_pipeline":
        pipeline = NeuralAttackGraphPipeline(config).to(device)
        dummy_batch = create_dummy_batch_for_test(config, batch_size=config.batch_size, num_nodes=config.max_sequence_length // 2)
        dummy_batch_on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in dummy_batch.items()}
        dummy_batch_on_device['targets'] = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in dummy_batch['targets'].items()}
        predictions = pipeline(dummy_batch_on_device)
        print(f"Pipeline predictions keys: {predictions.keys()}")
        print(f"First attention weights shape: {predictions['attention_weights'][0].shape}")
        graphs_data = pipeline.get_attack_graph(dummy_batch_on_device)
        print(f"Generated {len(graphs_data)} attack graph data items.")
        if graphs_data:
            print(f"First graph node_entities: {graphs_data[0]['node_entities'][:5]}, attention_map shape: {graphs_data[0]['attention_map'].shape}")

    elif component_name == "graph_builder":
        builder = DynamicAttackGraphBuilder(config)
        num_nodes_for_graph = 10
        dummy_nodes = list(range(num_nodes_for_graph))
        dummy_predictions = {
            'attack_presence': np.random.rand(num_nodes_for_graph, 2),
            'attack_type': np.random.rand(num_nodes_for_graph, config.num_attack_types),
            'mitre_technique': np.random.rand(num_nodes_for_graph, config.num_mitre_techniques),
            'severity': np.random.rand(num_nodes_for_graph, 1),
            'confidence': np.random.rand(num_nodes_for_graph, 1)
        }
        dummy_attention_map = np.random.rand(num_nodes_for_graph, num_nodes_for_graph)
        graph = builder.build_attack_graph(dummy_nodes, dummy_predictions, dummy_attention_map)
        print(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        paths = builder.extract_attack_paths(graph)
        print(f"Extracted {len(paths)} dummy paths.")

    elif component_name == "evaluator":
        evaluator = NeuralAttackGraphEvaluator(config)
        total_dummy_samples = config.batch_size * 10
        dummy_preds_flat = {
            'attack_presence': torch.randn(total_dummy_samples, 2),
            'attack_type': torch.randn(total_dummy_samples, config.num_attack_types),
            'mitre_technique': torch.randn(total_dummy_samples, config.num_mitre_techniques),
            'severity': torch.rand(total_dummy_samples, 1),
            'confidence': torch.rand(total_dummy_samples, 1)
        }
        dummy_targets_flat = {
            'attack_presence': torch.randint(0, 2, (total_dummy_samples,)),
            'attack_type': torch.randint(0, config.num_attack_types, (total_dummy_samples,)),
            'mitre_technique': torch.randint(0, config.num_mitre_techniques, (total_dummy_samples,)),
            'severity': torch.rand(total_dummy_samples, 1),
            'confidence': torch.rand(total_dummy_samples, 1)
        }
        dummy_predicted_graphs_data: List[Dict[str, Any]] = [{'nodes': list(range(10)), 'edges': [[0,1], [1,2]], 'predictions': {}, 'attention_map': np.eye(10)} for _ in range(total_dummy_samples // 10)]
        dummy_true_sequences_data: List[Dict[str, Any]] = [{'true_edges': [[0,1]]} for _ in range(total_dummy_samples // 10)]
        detection_metrics = evaluator.evaluate_attack_detection(dummy_preds_flat, dummy_targets_flat)
        print(f"Detection Metrics: {detection_metrics}")
        temporal_metrics = evaluator.evaluate_temporal_sequences(dummy_predicted_graphs_data, dummy_true_sequences_data)
        print(f"Temporal Metrics: {temporal_metrics}")

    else:
        print(f"Unknown component: {component_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Incremental Development Script')
    parser.add_argument('component', type=str,
                        choices=['config', 'device', 'data_processor', 'dataset', 'attention', 'gnn_layer', 'tcn',
                                 'attack_classifier', 'neural_pipeline', 'graph_builder', 'evaluator'],
                        help='Component to test')
    args = parser.parse_args()
    _config = NeuralConfig()
    _device_mgr = DeviceManager()
    _monitor = PerformanceMonitor()
    _monitor.start("direct_test_run")
    test_component(args.component, _config, _device_mgr.get_device(), _monitor)
    _monitor.stop("direct_test_run")
    _monitor.save_metrics(os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')), f"direct_test_{args.component}_performance.json"))