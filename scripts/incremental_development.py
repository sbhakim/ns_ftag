# scripts/incremental_development.py


import sys
import os
import torch
import numpy as np
import random
from typing import Dict, Any, List
import pandas as pd # Import pandas for dummy timestamps

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
    
    # --- ADDITION: Dummy temporal_info ---
    # Generate sequential timestamps for dummy batch
    dummy_timestamps_per_sequence = [
        [pd.Timestamp('2023-01-01 00:00:00') + pd.Timedelta(seconds=j*10) for j in range(num_nodes)]
        for _ in range(batch_size)
    ]

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
        'true_edges': [[[0, 1]] for _ in range(batch_size)],
        'temporal_info': dummy_timestamps_per_sequence # NEW ADDITION
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
            # --- VERIFICATION: Check temporal_info in batch ---
            if 'temporal_info' in batch:
                print(f"Batch {i}: temporal_info count per sequence {len(batch['temporal_info'][0])} (first sequence)")
            else:
                print(f"Batch {i}: temporal_info NOT FOUND in batch!")
            if i == 0: break

    elif component_name == "attention":
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
        # --- FIX: ensure create_dummy_batch_for_test returns temporal_info ---
        dummy_batch = create_dummy_batch_for_test(config, batch_size=config.batch_size, num_nodes=config.max_sequence_length // 2)
        dummy_batch_on_device = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in dummy_batch.items()}
        dummy_batch_on_device['targets'] = {k: (v.to(device) if isinstance(v, torch.Tensor) else v) for k, v in dummy_batch['targets'].items()}
        
        predictions = pipeline(dummy_batch_on_device)
        print(f"Pipeline predictions keys: {predictions.keys()}")
        print(f"First attention weights shape: {predictions['attention_weights'][0].shape}")
        
        # --- FIX: The pipeline's get_attack_graph expects temporal_info in the batch now ---
        graphs_data = pipeline.get_attack_graph(dummy_batch_on_device) 
        
        print(f"Generated {len(graphs_data)} attack graph data items.")
        if graphs_data:
            print(f"First graph node_entities: {graphs_data[0]['node_entities'][:5]}, attention_map shape: {graphs_data[0]['attention_map'].shape}")
            # Optional: Check if temporal_info made it into the graph data as well
            if 'temporal_info' in graphs_data[0]:
                print(f"First graph temporal_info count: {len(graphs_data[0]['temporal_info'])}")

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
        
        # --- FIX: Add dummy temporal_info for graph_builder test ---
        dummy_temporal_info_single_sequence = [pd.Timestamp('2023-01-01 00:00:00') + pd.Timedelta(seconds=j*10) for j in range(num_nodes_for_graph)]
        
        graph = builder.build_attack_graph(
            dummy_nodes, 
            dummy_predictions, 
            dummy_attention_map, 
            temporal_info=dummy_temporal_info_single_sequence # NEW ARGUMENT
        )
        print(f"Built graph with {graph.number_of_nodes()} nodes and {graph.number_of_edges()} edges.")
        paths = builder.extract_attack_paths(graph)
        print(f"Extracted {len(paths)} dummy paths.")
        
        # Optional: Add checks for node/edge attributes to confirm new logic is applied
        # For example, check if 'temporal_position' is set on nodes or 'temporal_validity' on edges
        if graph.number_of_nodes() > 0:
            sample_node = list(graph.nodes(data=True))[0]
            print(f"Sample node attributes: {sample_node[1].keys()}")
        if graph.number_of_edges() > 0:
            sample_edge = list(graph.edges(data=True))[0]
            print(f"Sample edge attributes: {sample_edge[2].keys()}")


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
        
        # --- FIX: Dummy predicted_graphs_data and true_sequences_data now need 'temporal_info' ---
        # to ensure graph_builder received it (via pipeline test) and that evaluator has it.
        # This will be simpler by making create_dummy_batch_for_test pass it.
        dummy_predicted_graphs_data: List[Dict[str, Any]] = []
        dummy_true_sequences_data: List[Dict[str, Any]] = []

        # Re-using create_dummy_batch_for_test to ensure consistency
        # Assuming each dummy batch represents a sequence for evaluation purposes
        for _ in range(total_dummy_samples // (config.max_sequence_length // 2)): # Number of sequences
            dummy_batch_item = create_dummy_batch_for_test(config, batch_size=1, num_nodes=config.max_sequence_length // 2)
            
            # Simulate pipeline's output for predicted_graphs_data
            dummy_graph_nodes = list(range(dummy_batch_item['entities'].shape[1])) # Node indices
            dummy_graph_edges = [[0,1], [1,2]] # Simple dummy edges
            dummy_graph_attention_map = np.eye(dummy_batch_item['entities'].shape[1]) # Identity matrix for simplicity
            
            # Simulate neural predictions for a single sequence
            dummy_preds_single_seq = {
                'attack_presence': F.softmax(torch.randn(dummy_batch_item['entities'].shape[1], 2), dim=1).cpu().numpy(),
                'attack_type': F.softmax(torch.randn(dummy_batch_item['entities'].shape[1], config.num_attack_types), dim=1).cpu().numpy(),
                'mitre_technique': F.softmax(torch.randn(dummy_batch_item['entities'].shape[1], config.num_mitre_techniques), dim=1).cpu().numpy(),
                'severity': torch.rand(dummy_batch_item['entities'].shape[1], 1).cpu().numpy(),
                'confidence': torch.rand(dummy_batch_item['entities'].shape[1], 1).cpu().numpy()
            }
            
            # Simulate true targets for a single sequence
            dummy_targets_single_seq = {
                'attack_presence': torch.randint(0, 2, (dummy_batch_item['entities'].shape[1],)).cpu(),
                'attack_type': torch.randint(0, config.num_attack_types, (dummy_batch_item['entities'].shape[1],)).cpu(),
                'mitre_technique': torch.randint(0, config.num_mitre_techniques, (dummy_batch_item['entities'].shape[1],)).cpu(),
                'severity': torch.rand(dummy_batch_item['entities'].shape[1], 1).cpu(),
                'confidence': torch.rand(dummy_batch_item['entities'].shape[1], 1).cpu()
            }

            # Create a dummy NetworkX graph object as the 'graph' attribute
            temp_graph_obj = nx.DiGraph()
            for k in dummy_graph_nodes:
                temp_graph_obj.add_node(k, is_attack_step=bool(np.random.rand() > 0.5)) # Add some dummy attack steps
            for src, dst in dummy_graph_edges:
                temp_graph_obj.add_edge(src, dst)

            dummy_predicted_graphs_data.append({
                'nodes': dummy_graph_nodes, 
                'edges': dummy_graph_edges, 
                'predictions': dummy_preds_single_seq, 
                'attention_map': dummy_graph_attention_map,
                'temporal_info': dummy_batch_item['temporal_info'][0], # Get temporal info for this sequence
                'graph': temp_graph_obj # Add the NetworkX graph object
            })
            dummy_true_sequences_data.append({
                'true_edges': dummy_batch_item['true_edges'][0], # Get true edges for this sequence
                'targets': dummy_targets_single_seq # Pass targets for node evaluation in graph_construction_f1
            })


        detection_metrics = evaluator.evaluate_attack_detection(dummy_preds_flat, dummy_targets_flat)
        print(f"Detection Metrics: {detection_metrics}")
      
        temporal_metrics = evaluator.evaluate_temporal_sequences(dummy_predicted_graphs_data, dummy_true_sequences_data)
        print(f"Temporal Metrics: {temporal_metrics}")

        graph_metrics = evaluator.evaluate_graph_construction_f1(dummy_predicted_graphs_data, dummy_true_sequences_data)
        print(f"Graph Construction Metrics: {graph_metrics}") # Corrected print

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