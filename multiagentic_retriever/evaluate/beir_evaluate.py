import logging
from tqdm import tqdm
from beir.retrieval.evaluation import EvaluateRetrieval
from config import CONFIG
from logging_utils import setup_logging
from tools.retrieve_tool import RetrieveTool
from tools.filter_tool import FilterTool
from agents.retrieval_agent import RetrievalAgent
from orchestrator import Planner
import json, os
from beir.datasets.data_loader import GenericDataLoader

class BeirEval():
    def __init__(self):
        self.dummy = 'dummy'
        
    def load_beir(dataset_dir: str):
        return GenericDataLoader(dataset_dir).load(split="test")

    def load_query_meta(path: str):
        meta = {}
        if path and os.path.exists(path):
            with open(path, 'r') as f:
                for line in f:
                    try:
                        d = json.loads(line)
                        meta[d.get('_id')] = d
                    except Exception:
                        pass
        return meta


def build_agent(corpus):
    algo_params = {
        'bm25': {
            'index_path': CONFIG['bm25_index_path'],
            'cpu_workers': CONFIG['cpu_workers'],
        },
        'contriever': {
            'model_path': CONFIG['contriever_model_path'],
            'embs_path': CONFIG['contriever_embs_path'],
        },
        'splade': {
            'model_name': CONFIG['splade_model_path'],
            'embs_path': CONFIG['splade_embs_path'],
            'use_multi_gpu': CONFIG['use_multi_gpu'],
        },
        'bge': {
            'model_path': CONFIG['bge_model_path'],
            'embs_path': None,
        }
    }

    retrieve_tool = RetrieveTool(
        corpus=corpus,
        algo_params=algo_params,
        cpu_workers=CONFIG['cpu_workers']
    )

    filter_tool = FilterTool(
        cfg=CONFIG['filtering_llm'],
        corpus=corpus,
        docs_per_call=CONFIG['filtering_llm'].get('docs_per_call', 10),
        cpu_workers=CONFIG['cpu_workers']
    )

    return RetrievalAgent(
        retrieve_tool=retrieve_tool,
        filter_tool=filter_tool,
        filter_policy=CONFIG.get('filter_policy', {"stage1":"auto","stage2":"never"}),
        filter_threshold=CONFIG.get('filter_threshold', 200),
    )


def run():
    setup_logging()
    log = logging.getLogger('RunDynamic')

    corpus, queries, qrels = load_beir(CONFIG['dataset_dir'])
    queries_meta = load_query_meta(CONFIG['queries_override_jsonl'])

    agent = build_agent(corpus)
    planner = Planner(CONFIG['planning_llm'])

    results = {}
    for qid, qtext in tqdm(queries.items(), desc='Dynamic Plans'):
        plan = planner.plan(qtext)
        log.info(f"Plan for qid={qid}: {plan}")

        stages = sorted(plan.items(), key=lambda kv: kv[0])
        candidate_ids = set()
        final_scores = {}

        for stage_name, algo_str in stages:
            # parse tokens like "bm25 + contriever" or "splade + bge" (no filtered flags)
            parts = [p.strip() for p in algo_str.split('+') if p.strip()]
            algos = [p.lower() for p in parts]

            top_k = CONFIG['retriever_top_k'] if stage_name.lower()=="stage1" else CONFIG.get('stage2_top_k', 20)

            # Agent decides whether to filter
            res = agent.retrieve(
                query=qtext,
                query_meta=queries_meta.get(qid, {}),
                algorithms=algos,
                top_k=top_k,
                stage_name=stage_name,
            )

            if candidate_ids:
                for did, s in res.items():
                    if did in final_scores:
                        final_scores[did] = max(final_scores[did], s)
                    else:
                        final_scores[did] = s
                candidate_ids |= set(res.keys())
            else:
                final_scores = dict(res)
                candidate_ids = set(res.keys())

        if not final_scores and candidate_ids:
            final_scores = {d: 1.0 for d in candidate_ids}

        results[qid] = final_scores

    if any(results.values()):
        evaluator = EvaluateRetrieval()
        ndcg, _, recall, precision = evaluator.evaluate(qrels, results, k_values=[1,5,10,20,50,100])
        print("\n" + "="*60)
        print(" DYNAMIC PIPELINE PERFORMANCE REPORT ")
        print("="*60)
        print(f"nDCG@10   : {ndcg.get('NDCG@10',0.0):.4f}")
        print(f"Recall@100: {recall.get('Recall@100',0.0):.4f}")
        print(f"P@1       : {precision.get('P@1',0.0):.4f}")
        print("="*60)

if __name__ == "__main__":
    run()
