# evaluation/evaluators.py
from typing import Dict, List
from beir.retrieval.evaluation import EvaluateRetrieval
from mar.evaluation.base import BaseEvaluator

class BeirEvaluator(BaseEvaluator):
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [10, 100]
        self.beir_evaluator = EvaluateRetrieval()
        print(f"📊 BeirEvaluator initialized for k values: {self.k_values}")

    def evaluate(self, results: Dict[str, Dict[str, float]], qrels: Dict[str, Dict[str, int]]) -> Dict[str, float]:
        if not qrels or not results: return {}
        print("--- 📊 Running BEIR Evaluation ---")
        ndcg, _, recall, _ = self.beir_evaluator.evaluate(qrels, results, self.k_values)
        metrics = {}
        metrics.update(ndcg)
        metrics.update(recall)
        print("--- ✅ BEIR Evaluation Complete ---")
        return metrics