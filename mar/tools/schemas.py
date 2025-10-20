from pydantic import BaseModel, DirectoryPath, FilePath, Field

class LLMConfig(BaseModel):
    base_url: str
    api_key: str
    model: str
    temperature: float = 0.0
    max_tokens: int

class FilterLLMConfig(LLMConfig):
    docs_per_call: int

class QueryToolConfig(BaseModel):
    model_path: DirectoryPath
    text_profiles_path: FilePath
    entity_profiles_path: FilePath