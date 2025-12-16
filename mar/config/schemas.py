from typing import Optional
from pydantic import BaseModel, Field

class LLMConfig(BaseModel):
    base_url: str
    api_key: str = "EMPTY"
    model: str
    temperature: float = 0.0
    max_tokens: int = 512
    docs_per_call: int = 4

class QueryToolConfig(BaseModel):
    model_path: str
    text_profiles_path: str
    entity_profiles_path: str

class RetrieverPaths(BaseModel):
    contriever_model: str
    splade_model: str
    bge_model: str
    bm25_index: str
    contriever_embs: str
    splade_embs: str
    bge_embs: str

class AppConfig(BaseModel):
    dataset_dir: str
    output_dir: str
    queries_override_jsonl: Optional[str] = None
    retriever_top_k: int = 50
    
    # Feature Flags
    enable_llm_planning: bool = True
    use_llm_filter: bool = False
    enable_llm_reranker: bool = False
    
    # Hardware
    cpu_workers: int = 32
    use_multi_gpu: bool = True
    
    # Sub-Configs
    paths: RetrieverPaths
    query_tool: QueryToolConfig
    planning_llm: LLMConfig
    filter_llm: LLMConfig
    reranker_llm: LLMConfig