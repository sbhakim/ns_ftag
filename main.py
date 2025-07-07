# main.py

import argparse
import logging
import sys
import os
import torch
from typing import Any

# Add src and scripts to path
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'scripts')))

from config.neural_config import NeuralConfig
from utils.device_manager import DeviceManager
from utils.performance_monitor import PerformanceMonitor
from train_phase1 import train_phase1 # This function's signature will need to be updated
from incremental_development import test_component # This function's signature will need to be updated

def setup_logging():
    """Sets up basic logging configuration."""
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    return logging.getLogger(__name__)

def run_incremental_test(component: str, config: NeuralConfig, device_mgr: DeviceManager, perf_monitor: PerformanceMonitor):
    """
    Runs an incremental test for a specified component.
    The test_component function will dynamically select data processors based on config.dataset_type.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting incremental test for component: {component} with dataset_type: {config.dataset_type}")
    perf_monitor.start(component)
    # Pass config directly to test_component for dynamic component selection
    test_component(component_name=component, config=config, device=device_mgr.get_device(), monitor=perf_monitor)
    perf_monitor.stop(component)
    logger.info(f"Completed test for {component}")

def run_full_training(config: NeuralConfig, device_mgr: DeviceManager, perf_monitor: PerformanceMonitor):
    """
    Starts the full Phase 1 training process.
    The train_phase1 function will dynamically select data processors based on config.dataset_type.
    """
    logger = logging.getLogger(__name__)
    logger.info(f"Starting full Phase 1 training with dataset_type: {config.dataset_type}")
    perf_monitor.start('full_training')
    # Pass config directly to train_phase1 for dynamic component selection
    train_phase1(config, device_mgr, perf_monitor) # Signature of train_phase1 needs to be updated
    perf_monitor.stop('full_training')
    logger.info("Full training complete")

def main():
    """Main function to parse arguments and run tests or training."""
    logger = setup_logging()
    parser = argparse.ArgumentParser(prog='NS-FTAG_Phase1Runner', description='Incremental testing or full training for NS-FTAG Phase 1')
    
    # --- New Arguments for Dataset Selection ---
    parser.add_argument('--dataset-type', choices=['cicids2017', 'optc'], default='cicids2017',
                        help='Select dataset type: cicids2017 (network flows) or optc (system provenance).')
    parser.add_argument('--data-path', type=str, 
                        help='Path to the dataset root directory (e.g., /path/to/CICIDS2017/TrafficLabelling or /path/to/OpTC/ecar/short)')
    # --- End New Arguments ---

    parser.add_argument('--mode', choices=['test', 'train'], default='test', help='Mode: incremental tests or full training')
    parser.add_argument('--component', default='config',
                        choices=['config', 'device', 'data_processor', 'dataset', 'attention', 'gnn_layer', 'tcn',
                                 'attack_classifier', 'neural_pipeline', 'graph_builder', 'evaluator'],
                        help='Component to test in test mode')
    parser.add_argument('--batch-size', type=int, help='Override config.batch_size')
    parser.add_argument('--lr', type=float, help='Override config.learning_rate')
    parser.add_argument('--epochs', type=int, help='Override config.max_epochs')
    parser.add_argument('--version', action='version', version='%(prog)s 1.0')
    args = parser.parse_args()

    # Initialize NeuralConfig (which has default data_path and dataset_type)
    config = NeuralConfig()
    device_mgr = DeviceManager()
    perf_monitor = PerformanceMonitor()

    # Apply command-line overrides to the config object
    if args.dataset_type is not None:
        config.dataset_type = args.dataset_type
        logger.info(f"Setting dataset_type: {config.dataset_type}")

    if args.data_path is not None:
        config.data_path = args.data_path
        logger.info(f"Overriding data_path: {config.data_path}")
        
    if args.batch_size is not None:
        logger.info(f"Overriding batch_size: {config.batch_size} -> {args.batch_size}")
        config.batch_size = args.batch_size
    if args.lr is not None:
        logger.info(f"Overriding learning_rate: {config.learning_rate} -> {args.lr}")
        config.learning_rate = args.lr
    if args.epochs is not None:
        logger.info(f"Overriding max_epochs: {config.max_epochs} -> {args.epochs}")
        config.max_epochs = args.epochs

    # Run in specified mode
    if args.mode == 'test':
        run_incremental_test(args.component, config, device_mgr, perf_monitor)
    else: # args.mode == 'train'
        run_full_training(config, device_mgr, perf_monitor)

if __name__ == "__main__":
    main()