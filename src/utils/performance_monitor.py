# src/utils/performance_monitor.py

import time
import psutil
import torch
import json
import os
from typing import Dict, Any

class PerformanceMonitor:
    def __init__(self):
        self.metrics = {}
        self._timers = {} # Store start times for active operations

    def start(self, operation_name: str): # Renamed from start_monitoring
        """Start monitoring a specific operation."""
        self._timers[operation_name] = time.time()
        # print(f"Monitoring '{operation_name}' started.") # Optional: add detailed logging here

    def stop(self, operation_name: str, extra_info: Dict[str, Any] = None): # Added new stop method
        """Stop monitoring and record metrics for a specific operation."""
        if operation_name not in self._timers:
            print(f"Warning: Attempted to stop '{operation_name}' but it was not started or already stopped.")
            return

        elapsed_time = time.time() - self._timers.pop(operation_name) # Stop timer and remove from active list

        current_metrics = {
            'execution_time_s': elapsed_time,
            'memory_usage_mb': psutil.Process(os.getpid()).memory_info().rss / 1024 / 1024, # Use os.getpid() for robust process check
            'gpu_memory_mb': torch.cuda.memory_allocated() / 1024 / 1024 if torch.cuda.is_available() else 0
        }
        if extra_info:
            current_metrics.update(extra_info)

        # Store metrics for this operation (e.g., append if multiple runs, or overwrite if single)
        # For simplicity, let's just store the last run's metrics for now
        self.metrics[operation_name] = current_metrics
        
        # Print to console immediately
        print(f"[{operation_name}] Time: {current_metrics['execution_time_s']:.3f}s, Mem: {current_metrics['memory_usage_mb']:.1f}MB, GPU Mem: {current_metrics['gpu_memory_mb']:.1f}MB")

    def save_metrics(self, filepath: str):
        """Save all recorded metrics to a file."""
        os.makedirs(os.path.dirname(filepath), exist_ok=True)
        with open(filepath, 'w') as f:
            json.dump(self.metrics, f, indent=2)
        print(f"All performance metrics saved to {filepath}")

    # You can keep the original record_metrics method if you still want to use it
    # for logging specific intermediate metrics that are not part of a timed block.
    # If not, you can remove it. For now, I'll include it.
    def record_metrics(self, operation_name: str, metrics_dict: Dict[str, Any]):
        """Directly record a set of metrics for an operation (e.g., non-timed, intermediate)."""
        if operation_name not in self.metrics:
            self.metrics[operation_name] = {} # Change to dict if storing latest, or list if storing history
        self.metrics[operation_name].update(metrics_dict) # Update existing metrics for the operation
        # print(f"Recorded additional metrics for {operation_name}: {metrics_dict}") # Optional logging