

from pydantic import BaseModel
from typing import Dict, Any

class ModelConfig(BaseModel):
    """
    Defines the configuration for creating a LiteLLM-based Model instance.
    This schema matches the desired flexible shape.
    """
    model_id: str
    client_args: Dict[str, Any] = {}
    params: Dict[str, Any] = {}