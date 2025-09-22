import logging
from typing import Dict, List

class RetrievalAgent:
    """
    AI agent with  tools:
      - RetrieveTool: Tool to retriever relevant docunemtns using a (set) retrieval mthods
      - FilterTool: Tool to binary judge the relevancy of retrieved documents.
    """
    def __init__(self, retrieve_tool, filter_tool=None, filter_policy: dict=None, filter_threshold: int = 200, logger: logging.Logger=None):
        self.retrieve_tool = retrieve_tool
        self.filter_tool = filter_tool
        self.filter_policy = (filter_policy or {"stage1": "auto", "stage2": "never"})
        self.filter_threshold = int(filter_threshold)
        self.logger = logger or logging.getLogger('RetrievalAgent')

    def retrieve(self, query: str, query_meta: Dict, algorithms: List[str], top_k: int, stage_name: str = "stage1") -> Dict[str, float]:
        # 1) retrieve first (no filtering)
        results = self.retrieve_tool.run(query, algorithms, top_k)
        # 2) decide filtering policy
        policy = str(self.filter_policy.get(stage_name.lower(), 'never')).lower()
        if not self.filter_tool or not results:
            return results
        if policy == 'never':
            return results
        if policy == 'always':
            keep = set(self.filter_tool.run(query, query_meta, list(results.keys())))
            return {d: s for d, s in results.items() if d in keep}
        if policy == 'auto':
            if len(results) >= self.filter_threshold:
                keep = set(self.filter_tool.run(query, query_meta, list(results.keys())))
                return {d: s for d, s in results.items() if d in keep}
            return results
        return results