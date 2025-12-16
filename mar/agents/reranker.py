import requests
import json
import re
import time
from typing import Dict, List
from mar.utils.logging import get_logger
log = get_logger()

class RerankerAgent:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.session = requests.Session()
        log.info("LLM Reranker initialized.")

    def rerank(self, query_text: str, meta: Dict, corpus: Dict, doc_dict: Dict[str, float]) -> (Dict[str, float], float):
        if not doc_dict: return {}, 0.0
        doc_ids_to_rerank = list(doc_dict.keys())[:self.cfg.get('docs_per_call', 50)]
        original_ids_set = set(doc_ids_to_rerank)
        candidate_docs_str = "\n".join([f"- id: {did}\n  text: {(corpus.get(did, {}).get('title', '') + ' ' + corpus.get(did, {}).get('text', '')).strip()[:1000]}" for did in doc_ids_to_rerank])
        
        prompt = f"""You are an expert relevancy judge. Re-rank the following documents based on relevance to: "{query_text}".\n<CANDIDATE_DOCUMENTS>\n{candidate_docs_str}\n</CANDIDATE_DOCUMENTS>\nReturn a JSON list of IDs in order of relevance."""
        
        payload = {"model": self.cfg['model'], "messages": [{"role": "user", "content": prompt}], "temperature": 0.0, "max_tokens": 4096}
        start = time.monotonic()
        try:
            with self.session.post(f"{self.cfg['base_url']}/chat/completions", headers={"Authorization": f"Bearer {self.cfg['api_key']}"}, json=payload, timeout=1800) as resp:
                resp.raise_for_status()
                text = resp.json()["choices"][0]["message"]["content"]
        except Exception as e:
            log.error(f"Reranker failed: {e}")
            return doc_dict, time.monotonic() - start
            
        duration = time.monotonic() - start
        match = re.search(r"\[[\s\S]*?\]", text)
        reranked_ids = []
        if match:
            try:
                llm_ids = json.loads(match.group(0))
                seen = set()
                for did in llm_ids:
                    did = str(did)
                    if did in original_ids_set and did not in seen:
                        reranked_ids.append(did)
                        seen.add(did)
            except: pass
        
        new_scores = {}
        max_score = float(len(doc_ids_to_rerank))
        for i, did in enumerate(reranked_ids):
            new_scores[did] = max_score - i
        
        missing = [d for d in doc_ids_to_rerank if d not in new_scores]
        for i, did in enumerate(missing):
            new_scores[did] = max_score - len(reranked_ids) - i
            
        return new_scores, duration