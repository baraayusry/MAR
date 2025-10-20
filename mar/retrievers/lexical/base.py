# retrievers/lexical/base.py
from abc import ABC, abstractmethod
from typing import Dict

class BaseRetriever(ABC):
    def __init__(self, corpus: Dict, top_k: int):
        self.corpus, self.top_k = corpus, top_k
        self._prepare()
    @abstractmethod
    def _prepare(self): ...
    @abstractmethod
    def search(self, query_text: str, query_meta: Dict, top_k: int) -> Dict[str, float]: ...
    @abstractmethod
    def search_subset(self, query_text: str, subset_docs: Dict[str, str], top_k: int) -> Dict[str, float]: ...