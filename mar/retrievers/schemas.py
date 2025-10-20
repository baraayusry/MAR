from pydantic import BaseModel, DirectoryPath, FilePath
from typing import Optional

class BM25Paths(BaseModel):
    index_path: str

class DensePaths(BaseModel):
    model_path: DirectoryPath
    embs_path: Optional[FilePath] = None

class RetrieverPaths(BaseModel):
    """Defines all paths for the retriever models and indexes."""
    bm25: BM25Paths
    splade: DensePaths
    bge: DensePaths
    contriever: DensePaths