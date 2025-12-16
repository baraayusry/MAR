from mar.config.schemas import AppConfig

_RAW_CONFIG = {
    "dataset_dir": "",
    "output_dir": "",
    "queries_override_jsonl": "",
    "retriever_top_k": 50,

    "enable_llm_planning": True,
    "use_llm_filter": False,
    "enable_llm_reranker": False,
    
    "cpu_workers": 32,
    "use_multi_gpu": True,

    # --- Nested: Model & Embedding Paths ---
    "paths": {
        "contriever_model": "",
        "splade_model": "",
        "bge_model": "",
        "bm25_index": "",
        
        "contriever_embs": "",
        "splade_embs": "",
        "bge_embs": "",
    },

    # --- Nested: Query Tool ---
    "query_tool": {
        "model_path": "",
        "text_profiles_path": "",
        "entity_profiles_path": ""
    },

    # --- Nested: LLM Configs ---
    "planning_llm": {
        "base_url": "",
        "api_key": "EMPTY",
        "model": "",
        "temperature": 0.0,
        "max_tokens": 1024,
    },

    "filter_llm": {
        "base_url": "",
        "api_key": "EMPTY",
        "model": "",
        "temperature": 0.0,
        "max_tokens": 1024,
        "docs_per_call": 4, 
    },
    
    "reranker_llm": {
        "base_url": "",
        "api_key": "EMPTY",
        "model": "",
        "temperature": 0.0,
        "max_tokens": 1024,
        "docs_per_call": 50, 
    }
}

CONFIG = AppConfig(**_RAW_CONFIG)

def get_config() -> AppConfig:
    return CONFIG
