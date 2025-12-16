from mar.config.schemas import AppConfig

_RAW_CONFIG = {
    "dataset_dir": "/leonardo_scratch/fast/L-AUT_024/eelsaada/datasets/trec-dl-2019",
    "output_dir": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/two_stage_final_results-dl19",
    "queries_override_jsonl": "/leonardo_scratch/fast/L-AUT_024/eelsaada/datasets/trec-dl-2019/queries.jsonl",
    "retriever_top_k": 50,

    "enable_llm_planning": True,
    "use_llm_filter": False,
    "enable_llm_reranker": False,
    
    "cpu_workers": 32,
    "use_multi_gpu": True,

    # --- Nested: Model & Embedding Paths ---
    "paths": {
        "contriever_model": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/facebook-contriever",
        "splade_model": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/splade-v3",
        "bge_model": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/bge-large-en-v1.5",
        "bm25_index": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/two_stage_final_results-dl19/pyserini_index_nq",
        
        "contriever_embs": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/two_stage_final_results-dl19/sbert_embs_facebook-contriever.npy",
        "splade_embs": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/two_stage_final_results-dl19/splade_embs_splade-v3.npz",
        "bge_embs": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/two_stage_final_results-dl19/sbert_embs_bge-large.npy",
    },

    # --- Nested: Query Tool ---
    "query_tool": {
        "model_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/bge-large-en-v1.5",
        "text_profiles_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/final_kmeans_text_profiles.pkl",
        "entity_profiles_path": "/leonardo_scratch/fast/L-AUT_024/eelsaada/embs/final_tfidf_entity_profiles.pkl"
    },

    # --- Nested: LLM Configs ---
    "planning_llm": {
        "base_url": "http://lrdn0168:8000/v1",
        "api_key": "EMPTY",
        "model": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/Qwen2.5-32B-Instruct",
        "temperature": 0.0,
        "max_tokens": 1024,
    },

    "filter_llm": {
        "base_url": "http://lrdn0168:8000/v1",
        "api_key": "EMPTY",
        "model": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/Qwen2.5-32B-Instruct",
        "temperature": 0.0,
        "max_tokens": 512,  # Adjusted to prevent 400 Bad Request
        "docs_per_call": 4, # Adjusted to prevent context overflow
    },
    
    "reranker_llm": {
        "base_url": "http://lrdn0168:8000/v1",
        "api_key": "EMPTY",
        "model": "/leonardo_scratch/fast/L-AUT_024/eelsaada/models/Qwen2.5-32B-Instruct",
        "temperature": 0.0,
        "max_tokens": 1024, 
        "docs_per_call": 50, 
    }
}

CONFIG = AppConfig(**_RAW_CONFIG)

def get_config() -> AppConfig:
    return CONFIG