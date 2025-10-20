# datasets/base.py
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Dict, Optional

@dataclass
class Dataset:
    """A standard container for holding a dataset's components."""
    corpus: Dict[str, Dict[str, str]]
    queries: Dict[str, str]
    qrels: Dict[str, Dict[str, int]]
    queries_meta: Optional[Dict] = field(default_factory=dict)

class BaseDatasetLoader(ABC):
    @abstractmethod
    def load(self) -> Dataset:
        """Loads the data and returns a standardized Dataset object."""
        pass