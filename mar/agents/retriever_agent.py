import time
import torch
from typing import Dict, Optional, List, Any
from mar.strategy.base import BaseRetriever
from mar.strategy.bm25 import BM25Retriever
from mar.strategy.splade import SpladeRetriever
from mar.strategy.dense import DenseRetriever
from mar.tools.llm_filter import LLMFilterTool
from mar.utils.logging import get_logger

log = get_logger()

class RetrieverAgent:
    """
    A generic agent that uses a retrieval strategy and a list of tools.
    If an LLMFilterTool is present in the tools list, it filters results automatically.
    """
    def __init__(self, 
                 name: str, 
                 corpus: Dict, 
                 cfg: Dict, 
                 device: torch.device, 
                 tools: List[Any] = None,
                 prompt: str = ""):
        self.name = name
        self.corpus = corpus
        self.cfg = cfg
        self.device = device
        self.tools = tools or []
        self.prompt = prompt  # Can be used as a system prompt or instruction context
        
        self.strategy: Optional[BaseRetriever] = None
        self.init_duration = 0.0

    def initialize(self):
        if self.strategy is not None: return
        log.info(f"Initializing retriever '{self.name}' on demand...")
        start = time.monotonic()
        
        if self.name == "bm25":
            self.strategy = BM25Retriever(self.corpus, self.cfg["retriever_top_k"], self.cfg["bm25_index_path"])
        elif self.name == "contriever":
            self.strategy = DenseRetriever(self.corpus, self.cfg["retriever_top_k"], self.cfg["contriever_model_path"], self.cfg["contriever_embs_path"], self.device, batch_size=128, use_multi_gpu=self.cfg["use_multi_gpu"])
        elif self.name == "splade":
            self.strategy = SpladeRetriever(self.corpus, self.cfg["retriever_top_k"], self.cfg["splade_model_path"], self.cfg["splade_embs_path"], self.device, use_multi_gpu=self.cfg["use_multi_gpu"])
        elif self.name == "bge":
            self.strategy = DenseRetriever(self.corpus, self.cfg["retriever_top_k"], self.cfg["bge_model_path"], self.cfg["bge_embs_path"], self.device, batch_size=128, use_multi_gpu=self.cfg["use_multi_gpu"])
        else:
            raise ValueError(f"Unknown retriever agent name: {self.name}")
            
        self.init_duration = time.monotonic() - start
        log.info(f"Initialized '{self.name}' in {self.init_duration:.2f}s")

    def _apply_tools(self, query_text: str, query_meta: Dict, raw_results: Dict[str, float]) -> Dict[str, float]:
        """
        Iterates through self.tools and applies relevant ones.
        """
        results = raw_results
        
        for tool in self.tools:
            if isinstance(tool, LLMFilterTool):
                results = self._apply_filter_tool(tool, query_text, query_meta, results)
        
        return results

    def _apply_filter_tool(self, filter_tool: LLMFilterTool, query_text: str, query_meta: Dict, results: Dict[str, float]) -> Dict[str, float]:
        if not results: return results
        doc_ids = list(results.keys())
        log.info(f"[{self.name}] Applying LLM Filter Tool to {len(doc_ids)} docs...")

        # 1. Get payloads from tool
        payloads = filter_tool.judge(query_text, query_meta, self.corpus, doc_ids, retriever_names=[self.name])
        relevant_doc_ids = set()

        # 2. Execute payloads (Agent handles execution logic here)
        for payload, _ in payloads:
            response = filter_tool._call_api_for_batch(payload)
            judgments = filter_tool._extract_json_from_text(response)
            for doc_id, label in judgments.items():
                if label.strip().lower() == "relevant":
                    relevant_doc_ids.add(doc_id)

        # 3. Handle results
        if not relevant_doc_ids:
            log.warning(f"[{self.name}] Filter removed all docs. Returning original results (fallback).")
            return results
        
        log.info(f"[{self.name}] Filter kept {len(relevant_doc_ids)} / {len(results)} docs.")
        return {did: score for did, score in results.items() if did in relevant_doc_ids}

    def retrieve(self, query_text: str, query_meta: Dict, top_k: int) -> Dict[str, float]:
        self.initialize()
        results = self.strategy.search(query_text, query_meta, top_k)
        return self._apply_tools(query_text, query_meta, results)

    def retrieve_subset(self, query_text: str, subset_docs: Dict[str, str], top_k: int) -> Dict[str, float]:
        self.initialize()
        results = self.strategy.search_subset(query_text, subset_docs, top_k)
        # We also apply tools on subset retrieval if desired
        query_meta = {} # Subset refinement often doesn't need full meta for search, but tool might.
        return self._apply_tools(query_text, query_meta, results)