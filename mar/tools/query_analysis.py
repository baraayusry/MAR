# tools/query_analysis.py
import logging
import pickle
from typing import List, Dict, Any
import numpy as np
import torch
from sentence_transformers import SentenceTransformer
from scipy.special import softmax

log = logging.getLogger(__name__)

RETRIEVER_PERFORMANCE_DATA: Dict[str, Dict[str, Dict[str, float]]] = {
    'trec-covid': {'BM25': {'nDCG@10': 0.595, 'R@100': 0.109}, 'Splade': {'nDCG@10': 0.727, 'R@100': 0.128}, 'Contriever': {'nDCG@10': 0.596, 'R@100': 0.091}, 'BGE': {'nDCG@10': 0.781, 'R@100': 0.141}},
    'nfcorpus': {'BM25': {'nDCG@10': 0.322, 'R@100': 0.246}, 'Splade': {'nDCG@10': 0.347, 'R@100': 0.284}, 'Contriever': {'nDCG@10': 0.328, 'R@100': 0.301}, 'BGE': {'nDCG@10': 0.373, 'R@100': 0.337}},
    'nq': {'BM25': {'nDCG@10': 0.305, 'R@100': 0.751}, 'Splade': {'nDCG@10': 0.538, 'R@100': 0.930}, 'Contriever': {'nDCG@10': 0.498, 'R@100': 0.925}, 'BGE': {'nDCG@10': 0.541, 'R@100': 0.942}},
    'hotpotqa': {'BM25': {'nDCG@10': 0.633, 'R@100': 0.796}, 'Splade': {'nDCG@10': 0.687, 'R@100': 0.818}, 'Contriever': {'nDCG@10': 0.638, 'R@100': 0.777}, 'BGE': {'nDCG@10': 0.726, 'R@100': 0.873}},
    'webis-touche2020': {'BM25': {'nDCG@10': 0.442, 'R@100': 0.582}, 'Splade': {'nDCG@10': 0.247, 'R@100': 0.471}, 'Contriever': {'nDCG@10': 0.204, 'R@100': 0.442}, 'BGE': {'nDCG@10': 0.257, 'R@100': 0.487}},
    'dbpedia-entity': {'BM25': {'nDCG@10': 0.318, 'R@100': 0.468}, 'Splade': {'nDCG@10': 0.437, 'R@100': 0.562}, 'Contriever': {'nDCG@10': 0.413, 'R@100': 0.541}, 'BGE': {'nDCG@10': 0.407, 'R@100': 0.530}},
    'scifact': {'BM25': {'nDCG@10': 0.679, 'R@100': 0.925}, 'Splade': {'nDCG@10': 0.704, 'R@100': 0.935}, 'Contriever': {'nDCG@10': 0.677, 'R@100': 0.947}, 'BGE': {'nDCG@10': 0.741, 'R@100': 0.967}}
}

DATASET_METADATA: Dict[str, Dict[str, str]] = {
    'trec-covid': {'domain': 'Biomedical', 'description': 'Scientific articles about the COVID-19 pandemic.'},
    'nfcorpus': {'domain': 'General / Medical', 'description': 'Nutrition-focused web documents.'},
    'nq': {'domain': 'Open-Domain QA', 'description': 'Natural questions from Google search with answers from Wikipedia.'},
    'hotpotqa': {'domain': 'Complex QA', 'description': 'Multi-hop question answering requiring reasoning over multiple documents.'},
    'webis-touche2020': {'domain': 'Argument Retrieval', 'description': 'Pro/con arguments on controversial topics.'},
    'dbpedia-entity': {'domain': 'Knowledge Base', 'description': 'Entity-centric retrieval against a large knowledge base.'},
    'scifact': {'domain': 'Scientific Fact-Checking', 'description': 'Verifying scientific claims using abstracts from research papers.'}
}


class QueryAnalysisTool:
    def __init__(self, config: Dict[str, str]):
        log.info("Initializing QueryAnalysisTool...")
        self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
        try:
            self.model = SentenceTransformer(config['model_path'], device=self.device)
            with open(config['text_profiles_path'], 'rb') as f:
                self.kmeans_text_profiles: Dict[str, np.ndarray] = pickle.load(f)
            with open(config['entity_profiles_path'], 'rb') as f:
                self.entity_profiles: Dict[str, Dict[str, float]] = pickle.load(f)
            self.dataset_names: List[str] = list(self.kmeans_text_profiles.keys())
        except Exception as e:
            log.warning(f"Could not load QueryAnalysisTool models/profiles: {e}. It will return empty results.")
            self.model = None

    def run(self, query_data: Dict[str, Any], **kwargs) -> Dict[str, Any]:
        if not self.model:
            return {} 
        text_scores = self._calculate_text_scores(query_data["query"])
        entity_scores = self._calculate_entity_scores(query_data.get("entities", []))
        raw_final_scores = {}
        min_text, max_text = min(text_scores.values()), max(text_scores.values())
        min_entity, max_entity = min(entity_scores.values()), max(entity_scores.values())
        for name in self.dataset_names:
            norm_text = (text_scores[name] - min_text) / (max_text - min_text + 1e-9) if max_text > min_text else 0
            norm_entity = (entity_scores[name] - min_entity) / (max_entity - min_entity + 1e-9) if max_entity > min_entity else 0
            raw_final_scores[name] = (0.5 * (norm_text ** 2.0)) + (0.5 * (norm_entity ** 2.0))
        score_values = np.array(list(raw_final_scores.values()))
        probabilities = softmax(score_values / 0.05)
        prob_dist = {name: prob for name, prob in zip(self.dataset_names, probabilities)}
        best_dataset = max(prob_dist, key=prob_dist.get)
        return {
            "predicted_dataset": best_dataset, "confidence": prob_dist[best_dataset],
            "dataset_stats": self._get_retriever_stats(best_dataset),
        }
    
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
            }
        }