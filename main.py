# run_pipeline.py
import os
import json
import logging
import time
import torch
import concurrent.futures
from tqdm import tqdm
from pydantic import BaseModel

# --- Import Schemas ---
from mar.datasets.schemas import DatasetConfig
from mar.retrievers.schemas import RetrieverPaths
from mar.tools.schemas import LLMConfig, FilterLLMConfig, QueryToolConfig

# --- Import Framework Components ---
from mar.datasets.beir_dataloader import BeirDatasetLoader
from mar.agents.planner_agent import PlannerAgent
from mar.agents.retriever_agent import RetrieverAgent
from mar.tools.llm_plan import LLMPlanTool
from mar.tools.llm_filter import LLMFilterTool
from mar.tools.query_analysis import QueryAnalysisTool
from mar.retrievers.lexical.bm25 import BM25Retriever
from mar.retrievers.lexical.splade import SpladeRetriever
from mar.retrievers.dense.bge import BGERetriever
from mar.retrievers.dense.contriever import ContrieverRetriever
from mar.evaluation.beir_evaluator import BeirEvaluator

# --- Define the Master Config Schema ---
class AppConfig(BaseModel):
    datasets: DatasetConfig
    retriever_paths: RetrieverPaths
    planning_llm: LLMConfig
    filter_llm: FilterLLMConfig
    query_tool: QueryToolConfig
    retriever_top_k: int
    use_llm_filter: bool
    cpu_workers: int
    use_multi_gpu: bool

# --- Define the Configuration Data ---
CONFIG_DATA = {
    "datasets": { "loader_type": "beir", "beir": { "dataset_dir": "/leonardo_scratch/fast/L-AUT_024/eelsaada/datasets/webis-touche2020", "queries_override_jsonl": "/leonardo_scratch/fast/L-AUT_024/eelsaada/datasets/webis-touche2020/expanded_queries_llama370.jsonl"}},
    "retriever_paths": {
        "bm25": {"index_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/two_stage_final_results-webis-touche2020/pyserini_index"},
        "splade": {"model_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/splade-v3", "embs_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/two_stage_final_results-webis-touche2020/splade_embs_splade-v3.npz"},
        "bge": {"model_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/bge-large-en-v1.5", "embs_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/two_stage_final_results-webis-touche2020/sbert_embs_bge-large.npy"},
        "contriever": {"model_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/facebook-contriever", "embs_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/two_stage_final_results-webis-touche2020/sbert_embs_facebook-contriever.npy"}
    },
    "query_tool": {"model_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/bge-large-en-v1.5", "text_profiles_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/final_kmeans_text_profiles.pkl", "entity_profiles_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/final_tfidf_entity_profiles.pkl"},
    "planning_llm": {"base_url": "http://lrdn0441:8000/v1", "api_key": "EMPTY", "model": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/llama3", "temperature": 0.0, "max_tokens": 1024},
    "filter_llm": {"base_url": "http://lrdn0441:8000/v1", "api_key": "EMPTY", "model": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/llama3", "temperature": 0.0, "max_tokens": 2048, "docs_per_call": 8},
    "retriever_top_k": 50, "use_llm_filter": True, "cpu_workers": 32, "use_multi_gpu": True
}
AGENT_ALIASES = {"bm25": "bm25", "bge": "bge", "splade": "splade", "contriever": "contriever"}

def normalize_agent_names(agent_names: list) -> list:
    normalized = []
    for name in agent_names:
        for part in name.split(','):
            part = part.strip().lower()
            canonical_name = AGENT_ALIASES.get(part)
            if canonical_name: normalized.append(canonical_name)
            else: logging.warning(f"Unknown agent name '{part}' found in plan. It will be ignored.")
    return list(dict.fromkeys(normalized))

def merge_agent_results(results_list: list) -> dict:
    unique_doc_ids = set()
    for result_dict in results_list:
        if result_dict: unique_doc_ids.update(result_dict.keys())
    return {doc_id: 1.0 for doc_id in unique_doc_ids}

def main():
    try: config = AppConfig(**CONFIG_DATA)
    except Exception as e:
        print(f"❌ Configuration Error: {e}"); return

    os.makedirs(os.path.dirname(config.retriever_paths.bm25.index_path), exist_ok=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    # --- 1. Loading Dataset ---
    loader = BeirDatasetLoader(config=config.datasets.beir)
    dataset = loader.load()

    # --- 2. Initializing Retriever Models ---
    print("\n--- 2. Initializing Retriever Models ---")
    retriever_models = {
        "bm25": BM25Retriever(dataset.corpus, config.retriever_top_k, config.retriever_paths.bm25, config.cpu_workers),
        "splade": SpladeRetriever(dataset.corpus, config.retriever_top_k, config.retriever_paths.splade.model_path, config.retriever_paths.splade.embs_path, device, config.use_multi_gpu),
        "bge": BGERetriever(dataset.corpus, config.retriever_top_k, config.retriever_paths.bge.model_path, config.retriever_paths.bge.embs_path, device, config.use_multi_gpu),
        "contriever": ContrieverRetriever(dataset.corpus, config.retriever_top_k, config.retriever_paths.contriever.model_path, config.retriever_paths.contriever.embs_path, device, config.use_multi_gpu),
    }

    # --- 3. Initializing Tools ---
    print("\n--- 3. Initializing Tools ---")
    query_analyzer = QueryAnalysisTool(config.query_tool.model_dump())
    plan_tool = LLMPlanTool(config.planning_llm.model_dump(), query_analyzer)
    filter_tool = LLMFilterTool(config.filter_llm.model_dump(), dataset.corpus) if config.use_llm_filter else None

    # --- 4. Initializing Agents ---
    print("\n--- 4. Initializing Agents ---")
    planner_agent = PlannerAgent(plan_tool=plan_tool)
    retriever_agents = {name: RetrieverAgent(name, model, filter_tool) for name, model in retriever_models.items()}

    # --- 5. Run Pipeline ---
    print("\n--- 5. 🏃‍♂️ Processing Queries ---")
    all_final_results = {}
    start_time = time.monotonic()
    for qid, qtext in tqdm(dataset.queries.items(), desc="Queries"):
        plan_context = {'meta': dataset.queries_meta.get(qid, {})}
        plan = planner_agent.run(qtext, plan_context)
        print(f"📋 Plan Received: {plan}")
        stage_outputs = {}
        for stage_config in plan.get("plan", []):
            stage_id = stage_config["stage_id"]
            agent_names_for_stage = normalize_agent_names(stage_config.get("agents", []))
            print(f"--- Executing Stage {stage_id} with: {', '.join(agent_names_for_stage)} ---")
            stage_context = {'top_k': config.retriever_top_k,'meta': dataset.queries_meta.get(qid, {}),'stage_id': stage_id}
            if stage_id > 1 and stage_id - 1 in stage_outputs:
                prev_results = stage_outputs[stage_id - 1]
                stage_context['candidate_docs'] = {doc_id: (dataset.corpus[doc_id]['title'] + ' ' + dataset.corpus[doc_id]['text']).strip() for doc_id in prev_results if doc_id in dataset.corpus}
            stage_results_list = []
            with concurrent.futures.ThreadPoolExecutor(max_workers=len(agent_names_for_stage)) as executor:
                future_to_agent = {executor.submit(retriever_agents[name].run, qtext, stage_context): name for name in agent_names_for_stage if name in retriever_agents}
                for future in concurrent.futures.as_completed(future_to_agent):
                    try:
                        results, log_string = future.result(); print(log_string); stage_results_list.append(results)
                    except Exception as exc: print(f"--- ❌ Agent '{future_to_agent[future]}' FAILED: {exc} ---")
            merged_stage_output = merge_agent_results(stage_results_list)
            print(f"--- Stage {stage_id} Complete: Merged into {len(merged_stage_output)} docs ---\n")
            stage_outputs[stage_id] = merged_stage_output
        if stage_outputs: all_final_results[qid] = stage_outputs[max(stage_outputs.keys())]
    end_time = time.monotonic()

    # --- 6. 📊 Evaluating Performance ---
    print("\n\n--- 6. 📊 Evaluating Performance ---")
    total_duration, num_queries = end_time - start_time, len(dataset.queries)
    avg_latency = total_duration / num_queries if num_queries > 0 else 0
    if not all_final_results: print("No results generated. Skipping evaluation."); return
    evaluator = BeirEvaluator(k_values=[10, 100]); metrics = evaluator.evaluate(all_final_results, dataset.qrels)
    print("\n" + "="*50 + "\n      PIPELINE PERFORMANCE REPORT      \n" + "="*50)
    print(f"  {'Total Processing Time':<25}: {total_duration:.2f} seconds\n  {'Total Queries':<25}: {num_queries}\n  {'Average Latency per Query':<25}: {avg_latency:.3f} seconds\n" + "-"*50)
    print(f"  {'Metric':<12} | {'Score':<10}\n  " + "-"*25)
    for name, score in metrics.items(): print(f"  {name:<12} | {score:.4f}")
    print("="*50)

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
    main()