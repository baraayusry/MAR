# Getting Started

Welcome to Multiagentic Retriever. This page will guide you through installation and a minimal example to get up and running.

```bash
pip install multiagentic-retriever
```

```python
from multiagentic_retriever import MultiAgentRetriever, Agent

retriever = MultiAgentRetriever(agents=[Agent(name="web", retriever="web_search")])
```

