from abc import ABC, abstractmethod
from typing import List, Dict

class BaseModelProvider(ABC):
    """
    Abstract base class for any LLM provider.
    It defines the standard interface for generating chat completions.
    """
    @abstractmethod
    def chat(self, messages: List[Dict], **kwargs) -> str:
        """Generates a response from the language model."""
        pass