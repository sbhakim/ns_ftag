# src/utils/metrics_collector.py
from typing import Dict, Any

class MetricsCollector:
    def __init__(self):
        self.metrics = {}

    def add_metric(self, name: str, value: Any):
        if name not in self.metrics:
            self.metrics[name] = []
        self.metrics[name].append(value)

    def get_average_metric(self, name: str):
        if name in self.metrics and self.metrics[name]:
            return sum(self.metrics[name]) / len(self.metrics[name])
        return None

    def reset(self):
        self.metrics = {}

    def get_all_metrics(self) -> Dict[str, Any]:
        return {name: self.get_average_metric(name) for name in self.metrics}
