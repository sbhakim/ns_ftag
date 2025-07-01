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
from train_phase1 import train_phase1
from incremental_development import test_component

def setup_logging():
    if not logging.getLogger().handlers:
        logging.basicConfig(
            level=logging.INFO,
            format='[%(asctime)s] %(levelname)s %(name)s: %(message)s',
            datefmt='%Y-%m-%d %H:%M:%S',
            handlers=[logging.StreamHandler(sys.stdout)]
        )
    return logging.getLogger(__name__)

def run_incremental_test(component: str, config: NeuralConfig, device_mgr: DeviceManager, perf_monitor: PerformanceMonitor):
    logger = logging.getLogger(__name__)
    logger.info(f"Starting incremental test for component: {component}")
    perf_monitor.start(component)
    test_component(component_name=component, config=config, device=device_mgr.get_device(), monitor=perf_monitor)
    perf_monitor.stop(component)
    logger.info(f"Completed test for {component}")

def run_full_training(config: NeuralConfig, device_mgr: DeviceManager, perf_monitor: PerformanceMonitor):
    logger = logging.getLogger(__name__)
    logger.info("Starting full Phase 1 training")
    perf_monitor.start('full_training')
    train_phase1()  # Updated to call train_phase1 without arguments, as it initializes components internally
    perf_monitor.stop('full_training')
    logger.info("Full training complete")

def main():
    logger = setup_logging()
    parser = argparse.ArgumentParser(prog='Phase1Runner', description='Incremental testing or full training for NS-FTAG Phase 1')
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

    config = NeuralConfig()
    device_mgr = DeviceManager()
    perf_monitor = PerformanceMonitor()

    if args.batch_size is not None:
        logger.info(f"Overriding batch_size: {config.batch_size} -> {args.batch_size}")
        config.batch_size = args.batch_size
    if args.lr is not None:
        logger.info(f"Overriding learning_rate: {config.learning_rate} -> {args.lr}")
        config.learning_rate = args.lr
    if args.epochs is not None:
        logger.info(f"Overriding max_epochs: {config.max_epochs} -> {args.epochs}")
        config.max_epochs = args.epochs

    if args.mode == 'test':
        run_incremental_test(args.component, config, device_mgr, perf_monitor)
    else:
        run_full_training(config, device_mgr, perf_monitor)

if __name__ == "__main__":
    main()
