import os
import shutil
import json
from typing import Dict


from mar.strategy.base import BaseRetriever
from mar.utils.logging import get_logger
from mar.config import CONFIG
from pyserini.search.lucene import LuceneSearcher

log = get_logger()

class BM25Retriever(BaseRetriever):
    def __init__(self, corpus: Dict, top_k: int, index_path: str):
        self.index_path = index_path
        super(BM25Retriever, self).__init__(corpus, top_k)
        
    def _prepare(self):
        if not os.path.exists(self.index_path):
            log.info(f"BM25 index not found at {self.index_path}. Creating...")
            collection_dir = os.path.join(os.path.dirname(self.index_path) or ".", "pyserini_temp_collection")
            os.makedirs(collection_dir, exist_ok=True)
            try:
                jsonl_path = os.path.join(collection_dir, "docs.jsonl")
                with open(jsonl_path, 'w', encoding='utf-8') as f:
                    for doc_id, data in self.corpus.items():
                        f.write(json.dumps({'id': doc_id, 'contents': (data.get('title', '') + ' ' + data.get('text', '')).strip()}) + '\n')
                cmd = f'python -m pyserini.index.lucene --collection JsonCollection --input {collection_dir} --index {self.index_path} --generator DefaultLuceneDocumentGenerator --threads {CONFIG["cpu_workers"]}'
                if os.system(cmd) != 0: raise RuntimeError("Pyserini indexing failed.")
            finally:
                shutil.rmtree(collection_dir)
        self.searcher = LuceneSearcher(self.index_path)
        
    def search(self, query_text: str, query_meta: Dict, top_k: int) -> Dict[str, float]:
        return {hit.docid: float(hit.score) for hit in self.searcher.search(query_text, k=top_k)}
        
    def search_subset(self, query_text: str, subset_docs: Dict[str, str], top_k: int) -> Dict[str, float]:
        log.warning("BM25 cannot efficiently search a subset; skipping in refinement stages.")
        return {}