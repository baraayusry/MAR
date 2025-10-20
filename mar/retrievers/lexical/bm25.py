# retrievers/lexical/bm25.py
import os
import json
import shutil
import logging
from typing import Dict
from pyserini.search.lucene import LuceneSearcher
from mar.retrievers.lexical.base import BaseRetriever
from mar.retrievers.schemas import BM25Paths

log = logging.getLogger(__name__)

class BM25Retriever(BaseRetriever):
    def __init__(self, corpus: Dict, top_k: int, config: BM25Paths, cpu_workers: int):
        self.config = config
        self.cpu_workers = cpu_workers
        super().__init__(corpus, top_k)

    def _prepare(self):
        if not os.path.exists(self.config.index_path):
            log.info(f"BM25 index not found at {self.config.index_path}. Creating...")
            collection_dir = os.path.join(os.path.dirname(self.config.index_path) or ".", "pyserini_temp_collection")
            os.makedirs(collection_dir, exist_ok=True)
            try:
                jsonl_path = os.path.join(collection_dir, "docs.jsonl")
                with open(jsonl_path, 'w', encoding='utf-8') as f:
                    for doc_id, data in self.corpus.items():
                        content = (data.get('title', '') + ' ' + data.get('text', '')).strip()
                        f.write(json.dumps({'id': doc_id, 'contents': content}) + '\n')
                cmd = (f'python -m pyserini.index.lucene --collection JsonCollection --input {collection_dir} --index {self.config.index_path} --generator DefaultLuceneDocumentGenerator --threads {self.cpu_workers}')
                if os.system(cmd) != 0: raise RuntimeError("Pyserini indexing failed.")
            finally: shutil.rmtree(collection_dir)
        else: log.info(f"BM25 index found at {self.config.index_path}.")

    def search(self, query_text: str, query_meta: Dict, top_k: int) -> Dict[str, float]:
        try:
            searcher = LuceneSearcher(self.config.index_path)
            return {hit.docid: float(hit.score) for hit in searcher.search(query_text, k=top_k)}
        except Exception as e:
            log.error(f"BM25 search failed for query '{query_text}': {e}")
            return {}

    def search_subset(self, query_text: str, subset_docs: Dict[str, str], top_k: int) -> Dict[str, float]:
        log.warning("BM25 cannot efficiently search a subset; skipping.")
        return {}