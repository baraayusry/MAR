import pickle
import torch
import numpy as np
from sentence_transformers import SentenceTransformer
from scipy.special import softmax
from typing import Dict, Any, List
from mar.utils.logging import get_logger
from mar.config import RETRIEVER_PERFORMANCE_DATA, DATASET_METADATA

log = get_logger()

class QueryAnalysisTool:
    def __init__(self, config: Dict[str, str]):
        log.info("Initializing QueryAnalysisTool...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = SentenceTransformer(config['model_path'], device=self.device)
            with open(config['text_profiles_path'], 'rb') as f:
                self.kmeans_text_profiles = pickle.load(f)
            with open(config['entity_profiles_path'], 'rb') as f:
                self.entity_profiles = pickle.load(f)
            self.dataset_names = list(self.kmeans_text_profiles.keys())
        except Exception as e:
            log.error(f"QueryAnalysisTool FATAL ERROR: {e}")
            raise

    def _calculate_text_scores(self, query_text: str) -> Dict[str, float]:
        query_embedding = self.model.encode(query_text, convert_to_tensor=True).reshape(1, -1)
        scores = {}
        for name in self.dataset_names:
            centroids = self.kmeans_text_profiles.get(name)
            if centroids is None or centroids.shape[0] == 0:
                scores[name] = 0.0
                continue
            centroids_tensor = torch.from_numpy(centroids).to(self.device)
            sims = torch.nn.functional.cosine_similarity(query_embedding, centroids_tensor)
            scores[name] = torch.max(sims).item()
        return scores

    def _calculate_entity_scores(self, query_entities: List[str]) -> Dict[str, float]:
        query_entity_set = set(query_entities)
        scores = {}
        for name in self.dataset_names:
            profile = self.entity_profiles.get(name, {})
            overlapping_entities = query_entity_set.intersection(profile.keys())
            score = sum(profile[entity] for entity in overlapping_entities)
            scores[name] = score
        return scores
        
    def _get_retriever_stats(self, dataset_name: str) -> Dict[str, Any]:
        performance = RETRIEVER_PERFORMANCE_DATA.get(dataset_name)
        if not performance:
            return {"error": f"Performance data not available for dataset: '{dataset_name}'"}
        best_by_ndcg = max(performance.items(), key=lambda item: item[1]['nDCG@10'])
        best_by_recall = max(performance.items(), key=lambda item: item[1]['R@100'])
        return {
            "metadata": DATASET_METADATA.get(dataset_name, {}),
            "best_retrievers": {
                "by_precision_nDCG_at_10": {"name": best_by_ndcg[0], "score": best_by_ndcg[1]['nDCG@10']},
                "by_recall_R_at_100": {"name": best_by_recall[0], "score": best_by_recall[1]['R@100']}
            }, "full_performance_data": performance
        }

    def run(self, query_data: Dict[str, Any], w_text: float = 0.5, w_entity: float = 0.5, power: float = 2.0, temperature: float = 0.05) -> Dict[str, Any]:
        text_scores = self._calculate_text_scores(query_data["query"])
        entity_scores = self._calculate_entity_scores(query_data.get("entities", []))
        raw_final_scores = {}
        min_text, max_text = min(text_scores.values()), max(text_scores.values())
        min_entity, max_entity = min(entity_scores.values()), max(entity_scores.values())
        for name in self.dataset_names:
            norm_text = (text_scores[name] - min_text) / (max_text - min_text + 1e-9) if max_text > min_text else 0
            norm_entity = (entity_scores[name] - min_entity) / (max_entity - min_entity + 1e-9) if max_entity > min_entity else 0
            raw_final_scores[name] = (w_text * (norm_text ** power)) + (w_entity * (norm_entity ** power))
        score_values = np.array(list(raw_final_scores.values()))
        probabilities = softmax(score_values / temperature)
        prob_dist = {name: prob for name, prob in zip(self.dataset_names, probabilities)}
        best_dataset = max(prob_dist, key=prob_dist.get)
        return {
            "predicted_dataset": best_dataset, "confidence": prob_dist[best_dataset],
            "dataset_stats": self._get_retriever_stats(best_dataset), "full_distribution": prob_dist
        }