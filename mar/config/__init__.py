import os



RETRIEVER_PERFORMANCE_DATA = {
    'trec-covid': {'BM25': {'nDCG@10': 0.595, 'R@100': 0.109}, 'Splade': {'nDCG@10': 0.727, 'R@100': 0.128}, 'Contriever': {'nDCG@10': 0.596, 'R@100': 0.091}, 'BGE': {'nDCG@10': 0.781, 'R@100': 0.141}},
    'nfcorpus': {'BM25': {'nDCG@10': 0.322, 'R@100': 0.246}, 'Splade': {'nDCG@10': 0.347, 'R@100': 0.284}, 'Contriever': {'nDCG@10': 0.328, 'R@100': 0.301}, 'BGE': {'nDCG@10': 0.373, 'R@100': 0.337}},
    'nq': {'BM25': {'nDCG@10': 0.305, 'R@100': 0.751}, 'Splade': {'nDCG@10': 0.538, 'R@100': 0.930}, 'Contriever': {'nDCG@10': 0.498, 'R@100': 0.925}, 'BGE': {'nDCG@10': 0.541, 'R@100': 0.942}},
    'hotpotqa': {'BM25': {'nDCG@10': 0.633, 'R@100': 0.796}, 'Splade': {'nDCG@10': 0.687, 'R@100': 0.818}, 'Contriever': {'nDCG@10': 0.638, 'R@100': 0.777}, 'BGE': {'nDCG@10': 0.726, 'R@100': 0.873}},
    'webis-touche2020': {'BM25': {'nDCG@10': 0.442, 'R@100': 0.582}, 'Splade': {'nDCG@10': 0.247, 'R@100': 0.471}, 'Contriever': {'nDCG@10': 0.204, 'R@100': 0.442}, 'BGE': {'nDCG@10': 0.257, 'R@100': 0.487}},
    'dbpedia-entity': {'BM25': {'nDCG@10': 0.318, 'R@100': 0.468}, 'Splade': {'nDCG@10': 0.437, 'R@100': 0.562}, 'Contriever': {'nDCG@10': 0.413, 'R@100': 0.541}, 'BGE': {'nDCG@10': 0.407, 'R@100': 0.530}},
    'scifact': {'BM25': {'nDCG@10': 0.679, 'R@100': 0.925}, 'Splade': {'nDCG@10': 0.704, 'R@100': 0.935}, 'Contriever': {'nDCG@10': 0.677, 'R@100': 0.947}, 'BGE': {'nDCG@10': 0.741, 'R@100': 0.967}}
}

DATASET_METADATA = {
    'trec-covid': {'domain': 'Biomedical', 'description': 'Scientific articles about the COVID-19 pandemic.'},
    'nfcorpus': {'domain': 'General / Medical', 'description': 'Nutrition-focused web documents.'},
    'nq': {'domain': 'Open-Domain QA', 'description': 'Natural questions from Google search with answers from Wikipedia.'},
    'hotpotqa': {'domain': 'Complex QA', 'description': 'Multi-hop question answering requiring reasoning over multiple documents.'},
    'webis-touche2020': {'domain': 'Argument Retrieval', 'description': 'Pro/con arguments on controversial topics.'},
    'dbpedia-entity': {'domain': 'Knowledge Base', 'description': 'Entity-centric retrieval against a large knowledge base.'},
    'scifact': {'domain': 'Scientific Fact-Checking', 'description': 'Verifying scientific claims using abstracts from research papers.'}
}

AGENT_ALIASES = {
    "bm25": "bm25",
    "splade": "splade",
    "contriever": "contriever",
    "bge": "bge",
    "bge-large": "bge",
    "bge_large": "bge",
    "facebook-contriever": "contriever",
    "contriever-msmarco": "contriever",
}

