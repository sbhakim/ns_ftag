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
    
    # --- Dataset Selection Arguments ---
    parser.add_argument('--dataset-type', choices=['cicids2017', 'optc'], default=None, # Set default to None to prioritize env var/config default
                        help='Select dataset type: cicids2017 (network flows) or optc (system provenance).')
    parser.add_argument('--data-path', type=str, default=None, # Set default to None to prioritize env var/config default
                        help='Path to the dataset root directory (e.g., /path/to/CICIDS2017/TrafficLabelling or /path/to/OpTC/ecar/short)')

    # --- Subsetting Arguments ---
    parser.add_argument('--subset-enabled', type=lambda x: (str(x).lower() == 'true'), default=None,
                        help='Enable or disable dataset subsetting (True/False). Overrides config.subset_config.enabled.')
    parser.add_argument('--subset-type', type=str, choices=['temporal_window', 'random_sample', 'attack_focus'], default=None,
                        help='Type of subsetting: temporal_window, random_sample, or attack_focus. Overrides config.subset_config.type.')
    parser.add_argument('--subset-start-date', type=str, default=None,
                        help='Start date for temporal window subsetting (YYYY-MM-DD HH:MM:SS). Overrides config.subset_config.start_date.')
    parser.add_argument('--subset-end-date', type=str, default=None,
                        help='End date for temporal window subsetting (YYYY-MM-DD HH:MM:SS). Overrides config.subset_config.end_date.')
    parser.add_argument('--subset-max-hosts', type=int, default=None,
                        help='Maximum number of hosts to include in subset. Overrides config.subset_config.max_hosts.')
    parser.add_argument('--subset-sample-fraction', type=float, default=None,
                        help='Fraction of data to sample randomly (0.0 to 1.0). Overrides config.subset_config.sample_fraction.')
    parser.add_argument('--subset-max-events', type=int, default=None,
                        help='Maximum number of events to retain in subset. Overrides config.subset_config.max_events.')
    # --- End Subsetting Arguments ---

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
        logger.info(f"Overriding dataset_type: {config.dataset_type}")
    else:
        logger.info(f"Using dataset_type from environment/config default: {config.dataset_type}")


    if args.data_path is not None:
        config.data_path = args.data_path
        logger.info(f"Overriding data_path: {config.data_path}")
    else:
        logger.info(f"Using data_path from environment/config default: {config.data_path}")
        
    if args.batch_size is not None:
        logger.info(f"Overriding batch_size: {config.batch_size} -> {args.batch_size}")
        config.batch_size = args.batch_size
    if args.lr is not None:
        logger.info(f"Overriding learning_rate: {config.learning_rate} -> {args.lr}")
        config.learning_rate = args.lr
    if args.epochs is not None:
        logger.info(f"Overriding max_epochs: {config.max_epochs} -> {args.epochs}")
        config.max_epochs = args.epochs

    # Apply subsetting overrides
    # Only enable subsetting if --subset-enabled is explicitly True or any other subsetting arg is provided
    subset_args_provided = any([
        args.subset_enabled is not None, args.subset_type is not None,
        args.subset_start_date is not None, args.subset_end_date is not None,
        args.subset_max_hosts is not None, args.subset_sample_fraction is not None,
        args.subset_max_events is not None
    ])

    if subset_args_provided:
        # Ensure subset_config is enabled if any subsetting arg is given
        if config.subset_config is None: # Initialize if it was None in config
            config.subset_config = {'enabled': True}
        else:
            config.subset_config['enabled'] = args.subset_enabled if args.subset_enabled is not None else True # Default to True if any subset arg provided

        if args.subset_type is not None:
            config.subset_config['type'] = args.subset_type
        if args.subset_start_date is not None:
            config.subset_config['start_date'] = args.subset_start_date
        if args.subset_end_date is not None:
            config.subset_config['end_date'] = args.subset_end_date
        if args.subset_max_hosts is not None:
            config.subset_config['max_hosts'] = args.subset_max_hosts
        if args.subset_sample_fraction is not None:
            config.subset_config['sample_fraction'] = args.subset_sample_fraction
        if args.subset_max_events is not None:
            config.subset_config['max_events'] = args.subset_max_events
        
        logger.info(f"Subset configuration applied: {config.subset_config}")
    else:
        logger.info(f"Subset configuration (from config default): {config.subset_config}")


    # Run in specified mode
    if args.mode == 'test':
        run_incremental_test(args.component, config, device_mgr, perf_monitor)
    else: # args.mode == 'train'
        run_full_training(config, device_mgr, perf_monitor)

if __name__ == "__main__":
    main()