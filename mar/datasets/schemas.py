# datasets/schemas.py
from pydantic import BaseModel, DirectoryPath, FilePath, Field
from typing import Optional, Literal

class BeirConfig(BaseModel):
    dataset_dir: DirectoryPath
    queries_override_jsonl: Optional[FilePath] = None

class DatasetConfig(BaseModel):
    """Defines the configuration for loading a dataset."""
    loader_type: Literal['beir'] = Field(..., description="The type of loader to use.")
    beir: BeirConfig