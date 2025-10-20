# agents/retriever_agent.py
import logging
from typing import Dict, Any, Tuple

class RetrieverAgent:
    def __init__(self, name: str, retriever_model: Any, filter_tool: Any = None, filter_policy: dict = None):
        self.name = name
        self.retriever_model = retriever_model
        self.filter_tool = filter_tool
        self.filter_policy = (filter_policy or {"default": "auto", "threshold": 200})
        self.logger = logging.getLogger(f'RetrieverAgent.{self.name}')

    def run(self, query: str, context: Dict[str, Any]) -> Tuple[Dict[str, float], str]:
        log_messages = []
        agent_name_upper = self.name.upper()
        log_messages.append(f"--- 🔎 Agent: {agent_name_upper} ---")
        top_k, query_meta, candidate_docs, stage_id = context.get("top_k", 50), context.get('meta', {}), context.get('candidate_docs'), context.get('stage_id', 1)
        log_messages.append("   |  **State:** Starting its retrieval process.")
        results = self.retriever_model.search_subset(query, candidate_docs, top_k) if candidate_docs else self.retriever_model.search(query, query_meta, top_k)
        log_messages.append(f"   |  **Result:** Retrieved {len(results)} documents.")
        policy, threshold = self.filter_policy.get(f"stage{stage_id}", self.filter_policy["default"]).lower(), self.filter_policy.get("threshold", 200)
        should_filter = bool(self.filter_tool and results and (policy == "always" or (policy == "auto" and len(results) >= threshold)))
        if should_filter:
            log_messages.append(f"   |  **Decision:** Policy is '{policy}', will filter {len(results)} docs.")
            log_messages.append("   |  **Action:** Delegating to LLMFilterTool...")
            relevant_ids, tool_log_string = self.filter_tool.run(query, query_meta, list(results.keys()))
            if tool_log_string: log_messages.append(tool_log_string)
            final_results = {doc_id: score for doc_id, score in results.items() if doc_id in relevant_ids}
        else:
            log_messages.append(f"   |  **Decision:** Policy is '{policy}', skipping filter.")
            final_results = results
        log_messages.append(f"--- ✅ Agent {agent_name_upper} Complete - Final Results: {len(final_results)} docs ---")
        return final_results, "\n".join(log_messages)