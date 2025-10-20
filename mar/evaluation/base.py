# evaluation/base.py
from abc import ABC, abstractmethod
from typing import Dict

class BaseEvaluator(ABC):
    @abstractmethod
    def evaluate(self, results: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        """Takes results and ground truth, and returns a dictionary of metrics."""
        pass