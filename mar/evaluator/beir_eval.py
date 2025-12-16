"""
Evaluation utilities for MAR
"""
from typing import Dict, List
from beir.retrieval.evaluation import EvaluateRetrieval
from mar.utils.logging import get_logger

log = get_logger(__name__)

class Evaluator:
    """Wrapper for BEIR evaluation with custom reporting."""
    
    def __init__(self, k_values: List[int] = None):
        self.k_values = k_values or [10, 100]
        self.evaluator = EvaluateRetrieval()
    
    def evaluate(self, qrels: Dict, results: Dict, stage_name: str = "Results") -> Dict:
        if not any(results.values()):
            log.warning(f"No results to evaluate for {stage_name}")
            return {}
        
        log.info(f"\n{'='*54}")
        log.info(f"  {stage_name:^50}")
        log.info(f"{'='*54}")
        
        ndcg, _map, recall, precision = self.evaluator.evaluate(qrels, results, k_values=self.k_values)
        
        # Log results
        log.info(f"  {'Metric':<12} | {'Score':<10}")
        log.info("  " + "-" * 25)
        
        for k in self.k_values:
            ndcg_key = f"NDCG@{k}"
            recall_key = f"Recall@{k}"
            log.info(f"  {ndcg_key:<12} | {ndcg.get(ndcg_key, 0.0):.4f}")
            log.info(f"  {recall_key:<12} | {recall.get(recall_key, 0.0):.4f}")
        
        log.info("="*54)
        
        return {
            "ndcg": ndcg,
            "map": _map,
            "recall": recall,
            "precision": precision
        }
    
    def compare_results(self, qrels: Dict, results_before: Dict, results_after: Dict, before_name: str = "Before", after_name: str = "After") -> Dict:
        before_metrics = self.evaluate(qrels, results_before, before_name)
        after_metrics = self.evaluate(qrels, results_after, after_name)
        
        log.info("\n" + "="*54)
        log.info("  Comparison Summary")
        log.info("="*54)
        
        for k in self.k_values:
            ndcg_key = f"NDCG@{k}"
            ndcg_before = before_metrics.get("ndcg", {}).get(ndcg_key, 0.0)
            ndcg_after = after_metrics.get("ndcg", {}).get(ndcg_key, 0.0)
            diff = ndcg_after - ndcg_before
            log.info(f"  {ndcg_key}: {ndcg_before:.4f} -> {ndcg_after:.4f} ({diff:+.4f})")
        
        log.info("="*54)
        return {before_name: before_metrics, after_name: after_metrics}