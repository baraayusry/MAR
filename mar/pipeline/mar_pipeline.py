import time
import concurrent.futures
import torch
from collections import defaultdict
from tqdm import tqdm

from mar.config.settings import get_config
from mar.utils.logging import get_logger
from mar.utils.helpers import normalize_agents, merge_results
from mar.dataset.loader import BEIRDataset
from mar.tools.query_analysis import QueryAnalysisTool
from mar.tools.llm_filter import LLMFilterTool

from mar.agents.planner import PlannerAgent
from mar.agents.reranker import RerankerAgent
from mar.agents.retriever_agent import RetrieverAgent
from mar.evaluator.beir_eval import Evaluator

log = get_logger()

class MARetrievalPipeline:
    def __init__(self):
        # 1. Load Pydantic Config
        self.cfg = get_config()
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # 2. Load Dataset
        self.dataset = BEIRDataset(self.cfg.dataset_dir, self.cfg.queries_override_jsonl)
        
        # 3. Initialize Tools
        # Pass the specific QueryToolConfig object
        self.query_tool = QueryAnalysisTool(self.cfg.query_tool)
        
        self.filter_tools_list = []
        if self.cfg.use_llm_filter:
            # Pass the specific LLMConfig object for filtering
            self.filter_tools_list.append(LLMFilterTool(self.cfg.filter_llm))

        # 4. Initialize Agents
        # Pass the specific LLMConfig for planning
        self.planner = PlannerAgent(
            cfg=self.cfg.planning_llm, 
            tools=[self.query_tool],
            prompt="" 
        )

        # Pass specific LLMConfig for reranking
        self.reranker = None
        if self.cfg.enable_llm_reranker:
            self.reranker = RerankerAgent(self.cfg.reranker_llm)
        
        self.retriever_pool: dict[str, RetrieverAgent] = {}
        
        # 5. Initialize Evaluator
        self.evaluator = Evaluator(k_values=[10, 100])

        # Stats & Results
        self.query_latencies = defaultdict(lambda: defaultdict(list))
        self.final_results = {}
        self.results_before_rerank = {} 

    def get_agent(self, name: str) -> RetrieverAgent:
        if name not in self.retriever_pool:
           
            self.retriever_pool[name] = RetrieverAgent(
                name=name, 
                corpus=self.dataset.corpus, 
                cfg=self.cfg, # Now passing AppConfig object
                device=self.device,
                tools=self.filter_tools_list,
                prompt="" 
            )
        return self.retriever_pool[name]

    def _execute_stage(self, stage: dict, qtext: str, prev_results: dict, qid: str):
        stage_id = stage["stage_id"]
        agent_names = normalize_agents(stage.get("agents", []))
        is_initial = not ("inputs" in stage or stage_id > 1)
        top_k = stage.get("top_k", self.cfg.retriever_top_k)
        
        retriever_outputs = []
        
        with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent_names)) as executor:
            futures = {}
            for name in agent_names:
                agent = self.get_agent(name)
                start_t = time.monotonic()
                query_meta = self.dataset.queries_meta.get(qid, {})
                
                if is_initial:
                    futures[executor.submit(agent.retrieve, qtext, query_meta, top_k)] = (name, start_t)
                else:
                    if name == 'bm25': continue
                    cand_docs = {d: (self.dataset.corpus[d].get('title','')+ ' ' + self.dataset.corpus[d].get('text','')).strip() for d in prev_results}
                    futures[executor.submit(agent.retrieve_subset, qtext, cand_docs, top_k)] = (name, start_t)
            
            for future in concurrent.futures.as_completed(futures):
                name, start_t = futures[future]
                try:
                    res = future.result()
                    dur = time.monotonic() - start_t
                    self.query_latencies[qid][f"stage_{stage_id}_{name}"].append(dur)
                    if res: retriever_outputs.append(res)
                except Exception as e: log.error(f"Agent {name} failed: {e}")

        return merge_results(retriever_outputs)

    def run(self):
        for qid, qtext in tqdm(self.dataset.queries.items(), desc="Processing"):
            q_start = time.monotonic()
            
            # 1. Planning
            if self.cfg.enable_llm_planning:
                plan = self.planner.generate_plan(qtext, self.dataset.queries_meta.get(qid, {}))
            else:
                plan = {"plan": [{"stage_id": 1, "action": "retrieve", "agents": ["bm25", "splade", "contriever"]}]}
            
            # 2. Execution
            current_results = {}
            sorted_plan = sorted(plan.get("plan", []), key=lambda x: x["stage_id"])
            for stage in sorted_plan:
                current_results = self._execute_stage(stage, qtext, current_results, qid)
            
            # Store results BEFORE reranking
            self.results_before_rerank[qid] = current_results.copy()
            
            # 3. Reranking
            if self.reranker and current_results:
                final, _ = self.reranker.rerank(qtext, self.dataset.queries_meta.get(qid, {}), self.dataset.corpus, current_results)
                self.final_results[qid] = final
            else:
                self.final_results[qid] = current_results
            
            self.query_latencies[qid]["total"].append(time.monotonic() - q_start)
        
        self._evaluate()

    def _evaluate(self):
        log.info("Starting Evaluation...")
        
        if self.cfg.enable_llm_reranker and self.results_before_rerank:
            self.evaluator.compare_results(
                self.dataset.qrels, 
                self.results_before_rerank, 
                self.final_results,
                before_name="Before Reranker",
                after_name="After Reranker (Final)"
            )
        else:
            self.evaluator.evaluate(
                self.dataset.qrels, 
                self.final_results, 
                stage_name="Final Pipeline Results"
            )