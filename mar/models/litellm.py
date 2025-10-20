
import litellm
import logging
from typing import List, Dict, Any
from models.base import BaseModelProvider

log = logging.getLogger(__name__)


class LiteLLMProvider(BaseModelProvider):
    """
    A universal model provider that uses LiteLLM to connect to various LLM APIs
    (OpenAI, Azure, Gemini, Ollama, vLLM, etc.).
    """
    def __init__(self, model_id: str, client_args: Dict = None, params: Dict = None):
        """
        Initializes the LiteLLM provider.

        Args:
            model_id (str): The model string LiteLLM uses (e.g., "ollama/llama3", "azure/your-deployment").
            client_args (Dict, optional): Authentication arguments like `api_key`, `api_base`.
            params (Dict, optional): Default generation parameters like `temperature`, `max_tokens`.
        """
        self.model_id = model_id
        self.client_args = client_args or {}
        self.params = params or {}
        print(f"🔧 Initialized LiteLLM provider for model: {self.model_id}")

    def chat(self, messages: List[Dict], **kwargs) -> str:
        """
        Generates a response using litellm.completion.
        """
        # Combine default params with any overrides passed during the call
        request_params = {**self.params, **kwargs}
        
        try:
            # The core LiteLLM call, which handles all provider-specific logic
            response = litellm.completion(
                model=self.model_id,
                messages=messages,
                **self.client_args,  # Unpacks api_key, api_base, etc.
                **request_params     # Unpacks temperature, max_tokens, etc.
            )
            return response.choices[0].message.content or ""
        except Exception as e:
            log.error(f"LiteLLM call for model '{self.model_id}' failed: {e}")
            return f"Error: Could not get a response from the model. Details: {e}"