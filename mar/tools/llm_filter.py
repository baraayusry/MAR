# tools/llm_filter_tool.py
import json
import re
import requests
import logging
import concurrent.futures
from typing import Dict, List, Any, Tuple

# Import the HTTPAdapter to manage the connection pool
from requests.adapters import HTTPAdapter

log = logging.getLogger(__name__)

class LLMFilterTool:
    def __init__(self, cfg: Dict, corpus: Dict):
        """
        Initializes the tool.
        A requests.Session is created with a larger connection pool
        to handle many parallel API calls efficiently.
        """
        self.cfg = cfg
        self.corpus = corpus
        
        # Create a session object
        self.session = requests.Session()
        
        # Create an adapter with a larger connection pool (e.g., 50)
        # This prevents the "Connection pool is full" warning
        adapter = HTTPAdapter(pool_connections=50, pool_maxsize=50)
        
        # Mount the adapter to handle all http and https requests
        self.session.mount('http://', adapter)
        self.session.mount('https://', adapter)

    # --- Helper methods ---
    @staticmethod
    def _extract_json_from_text(text: str) -> Dict[str, str]:
        match = re.search(r"\{[\s\S]*\}", text)
        if not match: return {}
        try:
            obj = json.loads(match.group(0))
            return {str(k): str(v) for k, v in obj.items()}
        except json.JSONDecodeError: return {}

    def _call_api_for_batch(self, payload: Dict) -> str:
        url = f"{self.cfg['base_url']}/chat/completions"
        headers = {"Authorization": f"Bearer {self.cfg['api_key']}", "Content-Type": "application/json"}
        try:
            with self.session.post(url, headers=headers, json=payload, timeout=180) as response:
                response.raise_for_status()
                return response.json()["choices"][0]["message"]["content"] or ""
        except requests.RequestException as e:
            log.error(f"LLM filter API call failed: {e}")
            return ""

    def build_prompt(self, query_text: str, meta: Dict, candidate_docs: str) -> str:
        all_entities = meta.get('entities', [])
        all_triples = meta.get('kg', [])
        return f"""
You are a sophisticated relevancy judge. Filter out documents that are not genuinely relevant to the user's claim.

<QUERY_CONTEXT>
- Claim: "{query_text}"
- Key Concepts: {all_entities}
- Structured Knowledge (KG): {all_triples}
</QUERY_CONTEXT>

<CANDIDATE_DOCUMENTS>
{candidate_docs}
</CANDIDATE_DOCUMENTS>

Judge each document. Return ONLY a single JSON object mapping each document's id to "relevant" or "irrelevant".
Example: {{"DOC1":"relevant","DOC2":"irrelevant"}}
""".strip()

    def _prepare_payloads(self, query_text: str, meta: Dict, doc_ids: List[str]) -> List[Dict]:
        docs_per_call = self.cfg.get('docs_per_call', 8)
        payloads = []
        for i in range(0, len(doc_ids), docs_per_call):
            batch_doc_ids = doc_ids[i:i + docs_per_call]
            candidate_docs_str = "\n".join([f"- id: {did}\n  text: {(self.corpus.get(did, {}).get('title', '') + ' ' + self.corpus.get(did, {}).get('text', '')).strip()[:1500]}" for did in batch_doc_ids])
            prompt = self.build_prompt(query_text, meta, candidate_docs_str)
            payloads.append({"model": self.cfg['model'], "messages": [{"role": "user", "content": prompt}], "temperature": self.cfg.get('temperature', 0.0), "max_tokens": self.cfg.get('max_tokens', 2048)})
        return payloads

    # ⚙️ THIS IS THE CORRECTED METHOD ⚙️
    def run(self, query_text: str, meta: Dict, doc_ids: List[str]) -> Tuple[List[str], str]:
        """
        The main entry point. Filters documents and returns a list of relevant IDs AND a log string.
        """
        # --- Log Capturing Setup ---
        log_messages = []
        if not doc_ids:
            return [], ""

        log_messages.append("    L--> 🛠️  **Tool: LLMFilterTool**")
        log_messages.append(f"        |  **Action:** Judging relevancy for {len(doc_ids)} documents.")
        
        payloads = self._prepare_payloads(query_text, meta, doc_ids)
        log_messages.append("        |  **Step 1:** Preparing API payloads in batches...")
        log_messages.append(f"        |  **Step 2:** Executing {len(payloads)} parallel API calls...")
        
        relevant_doc_ids = set()
        with concurrent.futures.ThreadPoolExecutor(max_workers=32) as executor:
            future_to_payload = {executor.submit(self._call_api_for_batch, p): p for p in payloads}
            for future in concurrent.futures.as_completed(future_to_payload):
                try:
                    response_text = future.result()
                    judgments = self._extract_json_from_text(response_text)
                    for doc_id, label in judgments.items():
                        if label.strip().lower() == "relevant":
                            relevant_doc_ids.add(doc_id)
                except Exception as exc:
                    log.error(f"LLM filter batch failed: {exc}")

        final_list = [doc_id for doc_id in doc_ids if doc_id in relevant_doc_ids]
        
        log_messages.append(f"        |  **Outcome:** {len(final_list)} of {len(doc_ids)} documents were judged relevant.")
        log_messages.append("   L--> ✅ **Tool Complete**")

        return final_list, "\n".join(log_messages)