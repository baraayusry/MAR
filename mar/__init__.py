"""Top-level package for multiagentic_retriever.

This package provides the public API surface used in examples and docs.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

__all__ = ["Agent", "MultiAgentRetriever", "__version__"]

__version__ = "0.1.0"


@dataclass
class Agent:
    name: str
    retriever: str


class MultiAgentRetriever:
    def __init__(self, agents: Optional[List[Agent]] = None) -> None:
        self.agents: List[Agent] = agents or []

    async def retrieve(self, query: str):
        """Stub async retrieve method.

        Replace with real multi-agent orchestration.
        """
        return {"query": query, "agents": [a.name for a in self.agents], "results": []}

