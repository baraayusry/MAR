import requests
import re
import json
import time
from typing import Dict, List, Tuple
from mar.utils.logging import get_logger

log = get_logger()

class LLMFilterTool:
    def __init__(self, cfg: Dict):
        self.cfg = cfg
        self.session = requests.Session()

    @staticmethod
    def _extract_json_from_text(text: str) -> Dict[str, str]:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match: return {}
        try:
            obj = json.loads(match.group(0))
            return {str(k): str(v) for k, v in obj.items()}
        except json.JSONDecodeError:
            return {}

    def _call_api_for_batch(self, payload: Dict) -> str:
        url = f"{self.cfg['base_url']}/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg['api_key']}", "Content-Type": "application/json"}
        try:
            with self.session.post(url, headers=headers, json=payload, timeout=1800) as response:
                # Better error handling to see WHY it failed (e.g. context length)
                if response.status_code != 200:
                    log.error(f"LLM Filter Error {response.status_code}: {response.text}")
                    return ""
                
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"] or ""
        except requests.RequestException as e:
            log.error(f"LLM filter API connection failed: {e}")
            return ""

    def build_prompt(self, query_text: str, meta: Dict, candidate_docs: str, retriever_names: List[str]) -> str:
        return f"""
You are a sophisticated relevancy judge for an information retrieval system. Your goal is to filter out documents that are not genuinely relevant to the user's claim.
<QUERY_CONTEXT>
- Claim: "{query_text}"
</QUERY_CONTEXT>
<CANDIDATE_DOCUMENTS>
{candidate_docs}
</CANDIDATE_DOCUMENTS>
Based on all the information above, judge each document.
**CRITICAL RULES:**
1. You MUST return judgments ONLY for the document IDs provided in the <CANDIDATE_DOCUMENTS> section.
2. Do NOT add, invent, or hallucinate any new document IDs.
3. The output MUST be a single JSON object mapping each document's id to either "relevant" or "irrelevant".
Return ONLY the single JSON object.
Example: {{"DOC1":"relevant","DOC2":"irrelevant"}}
""".strip()

    def judge(self, query_text: str, meta: Dict, corpus: Dict, doc_ids: List[str], retriever_names: List[str] = None) -> List[Tuple[Dict, List[str]]]:
        # Suggest lowering this in config if errors persist
        docs_per_call = self.cfg.get('docs_per_call', 5) 
        retriever_names = retriever_names or []
        payloads_and_batches = []
        
        for i in range(0, len(doc_ids), docs_per_call):
            batch_doc_ids = doc_ids[i:i + docs_per_call]
            
            # Reduce text length to avoid context overflow (e.g. 1000 chars instead of 2000)
            candidate_docs_list = []
            for did in batch_doc_ids:
                raw_text = (corpus.get(did, {}).get('title', '') + ' ' + corpus.get(did, {}).get('text', '')).strip()
                # Clean text to prevent JSON breaking
                clean_text = raw_text[:1200].replace('"', "'").replace('\n', ' ')
                candidate_docs_list.append(f"- id: {did}\n  text: {clean_text}")

            candidate_docs_str = "\n".join(candidate_docs_list)
            
            prompt = self.build_prompt(query_text, meta, candidate_docs_str, retriever_names)
            payload = {
                "model": self.cfg['model'],
                "messages": [{"role": "user", "content": prompt}],
                "temperature": self.cfg.get('temperature', 0.0),
                "max_tokens": self.cfg.get('max_tokens', 1024), # Reduced to leave room for input
            }
            payloads_and_batches.append((payload, batch_doc_ids))
            
        return payloads_and_batches