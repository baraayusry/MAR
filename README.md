# Multiagentic Retriever

[![CI](https://github.com/baraayusry/multiagentic-retriever/workflows/CI/badge.svg)](https://github.com/baraayusry/multiagentic-retriever/actions/workflows/ci.yml)
[![Documentation](https://github.com/baraayusry/multiagentic-retriever/workflows/Documentation/badge.svg)](https://baraayusry.github.io/multiagentic-retriever/)
[![PyPI version](https://badge.fury.io/py/multiagentic-retriever.svg)](https://badge.fury.io/py/multiagentic-retriever)
[![Python versions](https://img.shields.io/pypi/pyversions/multiagentic-retriever.svg)](https://pypi.org/project/multiagentic-retriever/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)
[![codecov](https://codecov.io/gh/baraayusry/multiagentic-retriever/branch/main/graph/badge.svg)](https://codecov.io/gh/baraayusry/multiagentic-retriever)

A powerful, extensible framework for building multiagentic retrieval systems that combine multiple AI agents to enhance information retrieval and knowledge discovery.

## 🚀 Features

- **Multi-Agent Architecture**: Coordinate multiple specialized retrieval agents
- **Flexible Retrieval**: Support for various retrieval methods and data sources
- **Async Support**: Built for high-performance concurrent operations
- **Extensible Design**: Easy to add new agents and retrieval strategies
- **Rich Configuration**: Comprehensive configuration options
- **Type Safety**: Full type hints and runtime validation with Pydantic
- **Production Ready**: Comprehensive testing, logging, and monitoring

## 📦 Installation

```bash
pip install multiagentic-retriever
```

### Optional Dependencies

```bash
# For vector search capabilities
pip install multiagentic-retriever[vector]

# For web scraping capabilities  
pip install multiagentic-retriever[web]

# For development
pip install multiagentic-retriever[dev]

# Install all extras
pip install multiagentic-retriever[vector,web,dev]
```

## 🏃 Quick Start

```python
import asyncio
from multiagentic_retriever import MultiAgentRetriever, Agent

async def main():
    # Create specialized agents
    web_agent = Agent(
        name="web_searcher",
        retriever="web_search",
        config={"max_results": 5}
    )
    
    doc_agent = Agent(
        name="document_searcher", 
        retriever="vector_search",
        config={"collection_name": "documents"}
    )
    
    # Initialize the multi-agent retriever
    retriever = MultiAgentRetriever(
        agents=[web_agent, doc_agent],
        aggregation_strategy="ranked_fusion"
    )
    
    # Perform retrieval
    results = await retriever.retrieve(
        query="What are the latest developments in quantum computing?",
        max_results=10
    )
    
    # Process results
    for result in results:
        print(f"Source: {result.source}")
        print(f"Score: {result.score:.3f}")
        print(f"Content: {result.content[:200]}...\n")

if __name__ == "__main__":
    asyncio.run(main())
```

## 🏗️ Architecture

The framework consists of several key components:

- **Agents**: Individual retrieval agents with specialized capabilities
- **Retrievers**: Backend retrieval implementations (vector search, web search, etc.)
- **Orchestrator**: Coordinates agent interactions and result aggregation  
- **Configuration**: Flexible configuration system for different deployment scenarios

## 📚 Documentation

Comprehensive documentation is available at [https://baraayusry.github.io/multiagentic-retriever/](https://baraayusry.github.io/multiagentic-retriever/)

- [Getting Started Guide](https://baraayusry.github.io/multiagentic-retriever/getting-started/)
- [Architecture Overview](https://baraayusry.github.io/multiagentic-retriever/architecture/)
- [API Reference](https://baraayusry.github.io/multiagentic-retriever/api-reference/)
- [Examples](https://baraayusry.github.io/multiagentic-retriever/examples/)

## 🛠️ Development

### Setup Development Environment

```bash
git clone https://github.com/baraayusry/multiagentic-retriever.git
cd multiagentic-retriever

# Create virtual environment
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate

# Install in development mode
pip install -e .[dev]

# Install pre-commit hooks
pre-commit install
```

### Running Tests

```bash
# Run all tests
pytest

# Run with coverage
pytest --cov=src --cov-report=html

# Run specific test file
pytest tests/test_agents.py
```

### Code Quality

```bash
# Format code
black src tests

# Sort imports
isort src tests

# Lint
flake8 src tests

# Type checking
mypy src

# Security check
bandit -r src
```

### Building Documentation

```bash
cd docs
mkdocs serve  # Serve locally
mkdocs build  # Build static site
```

## 🤝 Contributing

We welcome contributions! Please see our [Contributing Guide](CONTRIBUTING.md) for details.

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## 📄 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🎯 Roadmap

- [ ] Support for more retrieval backends (Elasticsearch, Pinecone, etc.)
- [ ] Advanced agent orchestration strategies
- [ ] Built-in caching and rate limiting
- [ ] Monitoring and observability features
- [ ] GraphQL API interface
- [ ] Docker deployment templates

## 📞 Support

- 📖 [Documentation](https://baraayusry.github.io/multiagentic-retriever/)
- 🐛 [Issue Tracker](https://github.com/baraayusry/multiagentic-retriever/issues)
- 💬 [Discussions](https://github.com/baraayusry/multiagentic-retriever/discussions)

## ⭐ Star History

If you find this project useful, please consider giving it a star!

[![Star History Chart](https://api.star-history.com/svg?repos=baraayusry/multiagentic-retriever&type=Date)](https://star-history.com/#baraayusry/multiagentic-retriever&Date)