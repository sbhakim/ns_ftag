# scripts/incremental_development.py


import sys
import os
import torch
import numpy as np
import random
from typing import Dict, Any, List
import pandas as pd
import logging

# Add src directory to Python path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from config.neural_config import NeuralConfig
from utils.device_manager import DeviceManager
from utils.performance_monitor import PerformanceMonitor
# No longer explicitly import data_processors here, rely on dynamic imports within test_component
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
import networkx as nx
import torch.nn.functional as F

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
    if total_nodes_in_batch > 0:
        edge_index = torch.randint(0, total_nodes_in_batch, (2, total_nodes_in_batch * 2))
    else:
        edge_index = torch.empty((2, 0), dtype=torch.long)

    security_features = torch.randn(batch_size, num_nodes, config.security_feature_dim)
    
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
    true_edges = []
    for _ in range(batch_size):
        if num_nodes > 1:
            true_edges.append([[k, k + 1] for k in range(num_nodes - 1)])
        else:
            true_edges.append([])

    return {
        'entities': entities,
        'actions': actions,
        'edge_index': edge_index,
        'security_features': security_features,
        'targets': targets,
        'true_edges': true_edges,
        'temporal_info': dummy_timestamps_per_sequence
    }

def test_component(component_name: str, config: NeuralConfig, device: torch.device, monitor: PerformanceMonitor):
    """Test individual components incrementally."""
    set_seed(42)
    logger = logging.getLogger(__name__)
    print(f"--- Testing {component_name} (Dataset: {config.dataset_type}) ---")

    if component_name == "config":
        print(f"Configuration loaded: {config}")

    elif component_name == "device":
        print(f"Device: {device}")
        dummy_tensor = torch.randn(10).to(device)
        print(f"Dummy tensor on device: {dummy_tensor.device}")

    elif component_name == "data_processor":
        processor = None
        entity_manager = None
        relationship_extractor = None
        feature_extractor = None
        label_extractor = None

        # Dynamically import and instantiate processor and managers based on dataset type
        if config.dataset_type == "optc":
            from data_processors.optc_processor import OpTCProcessor
            from data_processors.optc_entity_manager import OpTCEntityManager
            from data_processors.optc_relationship_extractor import OpTCRelationshipExtractor
            from data_processors.optc_feature_extractor import OpTCFeatureExtractor
            from data_processors.optc_label_extractor import OpTCLabelExtractor
            
            processor = OpTCProcessor(config)
            entity_manager = OpTCEntityManager()
            relationship_extractor = OpTCRelationshipExtractor(config)
            feature_extractor = OpTCFeatureExtractor(config)
            label_extractor = OpTCLabelExtractor(config)
            logger.info("Testing OpTC processor and managers...")
        else: # Default to CICIDS2017
            from data_processors.cicids2017_processor import CICIDS2017Processor
            from data_processors.entity_manager import EntityManager
            from data_processors.relationship_extractor import RelationshipExtractor
            from data_processors.feature_extractor import FeatureExtractor
            from data_processors.label_extractor import LabelExtractor
            
            processor = CICIDS2017Processor(config)
            entity_manager = EntityManager()
            relationship_extractor = RelationshipExtractor(config)
            feature_extractor = FeatureExtractor(config)
            label_extractor = LabelExtractor(config)
            logger.info("Testing CICIDS2017 processor and managers...")
                    
        df = processor.load_data(config.data_path)
        if df is None or df.empty:
            logger.error("Data loading failed or returned empty DataFrame for data_processor component test. Skipping further processing in this test.")
            return

        print(f"Data loaded, shape: {df.shape}")
        print(f"Columns: {df.columns.tolist()[:10]}...")

        sample_events_len = min(len(df), config.min_sequence_length + 5)
        if sample_events_len == 0:
            logger.warning("No events available in DataFrame for data_processor test after filtering. Skipping entity/relationship/feature/label extraction.")
            return

        sample_events = df.head(sample_events_len)
        
        try:
            entities_actions = entity_manager.extract_security_entities(sample_events)
            print(f"Extracted entity/action IDs: {len(entities_actions['source_entity_ids'])} events")
            print(f"Entity vocab size: {entity_manager.get_vocab_sizes()['entity_vocab_size']}")
            
            relationships = relationship_extractor.extract_relationships(sample_events)
            print(f"Extracted {len(relationships)} relationships")
            
            features = feature_extractor.extract_security_features(sample_events)
            print(f"Extracted {len(features)} security features (dim: {len(features[0]) if features else 'N/A'})")
            
            labels = label_extractor.extract_labels(sample_events)
            print(f"Extracted labels keys: {labels.keys()}")

        except Exception as e:
            logger.error(f"Error during entity/relationship/feature/label extraction: {e}")
            return


    elif component_name == "dataset":
        dataset = SecurityEventDataset(
            config.data_path, config
        )
        config.entity_vocab_size = dataset.entity_manager.get_vocab_sizes()['entity_vocab_size']
        config.action_vocab_size = dataset.entity_manager.get_vocab_sizes()['action_vocab_size']


        dataloader = torch.utils.data.DataLoader(dataset, batch_size=config.batch_size, shuffle=False, collate_fn=custom_collate_fn)
        for i, batch in enumerate(dataloader):
            print(f"Batch {i}: entities shape {batch['entities'].shape}, actions shape {batch['actions'].shape}")
            print(f"Batch {i}: security_features shape {batch['security_features'].shape}")
            print(f"Batch {i}: edge_index shape {batch['edge_index'].shape}")
            print(f"Batch {i}: targets keys {batch['targets'].keys()}")
            print(f"Batch {i}: true_edges count {len(batch['true_edges'])}")
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
        
        if total_nodes > 0:
            edge_index = torch.randint(0, total_nodes, (2, total_nodes * 2)).to(device)
        else:
            edge_index = torch.empty((2,0), dtype=torch.long).to(device)

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
        
        if graph.number_of_nodes() > 0:
            sample_node = list(graph.nodes(data=True))[0]
            print(f"Sample node attributes: {sample_node[1].keys()}")
        if graph.number_of_edges() > 0:
            sample_edge = list(graph.edges(data=True))[0]
            print(f"Sample edge attributes: {sample_edge[2].keys()}")


    elif component_name == "evaluator":
        evaluator = NeuralAttackGraphEvaluator(config)
        total_dummy_samples = config.batch_size * 10
        num_nodes_per_seq = max(1, config.max_sequence_length // 2) 

        dummy_preds_flat = {
            'attack_presence': torch.randn(total_dummy_samples * num_nodes_per_seq, 2),
            'attack_type': torch.randn(total_dummy_samples * num_nodes_per_seq, config.num_attack_types),
            'mitre_technique': torch.randn(total_dummy_samples * num_nodes_per_seq, config.num_mitre_techniques),
            'severity': torch.rand(total_dummy_samples * num_nodes_per_seq, 1),
            'confidence': torch.rand(total_dummy_samples * num_nodes_per_seq, 1)
        }
        dummy_targets_flat = {
            'attack_presence': torch.randint(0, 2, (total_dummy_samples * num_nodes_per_seq,)),
            'attack_type': torch.randint(0, config.num_attack_types, (total_dummy_samples * num_nodes_per_seq,)),
            'mitre_technique': torch.randint(0, config.num_mitre_techniques, (total_dummy_samples * num_nodes_per_seq,)),
            'severity': torch.rand(total_dummy_samples * num_nodes_per_seq, 1),
            'confidence': torch.rand(total_dummy_samples * num_nodes_per_seq, 1)
        }
        
        dummy_predicted_graphs_data: List[Dict[str, Any]] = []
        dummy_true_sequences_data: List[Dict[str, Any]] = []

        for _ in range(total_dummy_samples):
            dummy_batch_item = create_dummy_batch_for_test(config, batch_size=1, num_nodes=num_nodes_per_seq)
            
            dummy_graph_nodes = list(range(num_nodes_per_seq))
            
            dummy_preds_single_seq = {
                'attack_presence': F.softmax(torch.randn(num_nodes_per_seq, 2), dim=1).cpu().numpy(),
                'attack_type': F.softmax(torch.randn(num_nodes_per_seq, config.num_attack_types), dim=1).cpu().numpy(),
                'mitre_technique': F.softmax(torch.randn(num_nodes_per_seq, config.num_mitre_techniques), dim=1).cpu().numpy(),
                'severity': torch.rand(num_nodes_per_seq, 1).cpu().numpy(),
                'confidence': torch.rand(num_nodes_per_seq, 1).cpu().numpy()
            }
            
            dummy_targets_single_seq = {
                'attack_presence': torch.randint(0, 2, (num_nodes_per_seq,)).cpu(),
                'attack_type': torch.randint(0, config.num_attack_types, (num_nodes_per_seq,)).cpu(),
                'mitre_technique': torch.randint(0, config.num_mitre_techniques, (num_nodes_per_seq,)).cpu(),
                'severity': torch.rand(num_nodes_per_seq, 1).cpu(),
                'confidence': torch.rand(num_nodes_per_seq, 1).cpu()
            }

            temp_graph_obj = nx.DiGraph()
            for k in dummy_graph_nodes:
                temp_graph_obj.add_node(k, is_attack_step=bool(np.random.rand() > 0.5))
            
            if num_nodes_per_seq > 1:
                temp_graph_obj.add_edge(0, 1)
            
            item_true_edges = dummy_batch_item['true_edges'][0]

            dummy_predicted_graphs_data.append({
                'nodes': dummy_graph_nodes, 
                'edges': dummy_batch_item['true_edges'][0],
                'predictions': dummy_preds_single_seq, 
                'attention_map': np.eye(num_nodes_per_seq),
                'temporal_info': dummy_batch_item['temporal_info'][0],
                'graph': temp_graph_obj
            })
            dummy_true_sequences_data.append({
                'true_edges': item_true_edges,
                'targets': dummy_targets_single_seq
            })


        detection_metrics = evaluator.evaluate_attack_detection(dummy_preds_flat, dummy_targets_flat)
        print(f"Detection Metrics: {detection_metrics}")
      
        temporal_metrics = evaluator.evaluate_temporal_sequences(dummy_predicted_graphs_data, dummy_true_sequences_data)
        print(f"Temporal Metrics: {temporal_metrics}")

        graph_metrics = evaluator.evaluate_graph_construction_f1(dummy_predicted_graphs_data, dummy_true_sequences_data)
        print(f"Graph Construction Metrics: {graph_metrics}")


    else:
        print(f"Unknown component: {component_name}")

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Incremental Development Script')
    parser.add_argument('component', type=str,
                        choices=['config', 'device', 'data_processor', 'dataset', 'attention', 'gnn_layer', 'tcn',
                                 'attack_classifier', 'neural_pipeline', 'graph_builder', 'evaluator'],
                        help='Component to test')
    parser.add_argument('--dataset-type', choices=['cicids2017', 'optc'], default='cicids2017',
                        help='Select dataset type for testing.')
    parser.add_argument('--data-path', type=str, default=os.environ.get('NS_FTAG_DATA_PATH', './data/CICIDS2017/TrafficLabelling'),
                        help='Path to the dataset root directory (e.g., /path/to/CICIDS2017/TrafficLabelling or /path/to/OpTC/ecar/short)')

    args = parser.parse_args()
    
    _config = NeuralConfig()
    _config.dataset_type = args.dataset_type
    _config.data_path = args.data_path

    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )

    _device_mgr = DeviceManager()
    _monitor = PerformanceMonitor()
    _monitor.start("direct_test_run")
    test_component(args.component, _config, _device_mgr.get_device(), _monitor)
    _monitor.stop("direct_test_run")
    results_dir = os.path.join(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'results')))
    os.makedirs(results_dir, exist_ok=True)
    _monitor.save_metrics(os.path.join(results_dir, f"direct_test_{args.component}_performance.json"))