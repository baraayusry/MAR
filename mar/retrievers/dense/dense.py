import logging
import os
from typing import Dict, Optional
import torch
from mar.retrievers.lexical.base import BaseRetriever 
from sentence_transformers import SentenceTransformer, util
import numpy as np

from mar.retrievers.schemas import DensePaths # ⚙️ IMPORT THE SCHEMA

log = logging.getLogger(__name__)

class DenseRetriever(BaseRetriever):
    def __init__(self,
                 corpus: Dict,
                 top_k: int,
                 config: DensePaths, # ⚙️ ACCEPT THE PYDANTIC CONFIG OBJECT
                 device: Optional[torch.device] = None,
                 batch_size: int = 128,
                 use_multi_gpu: bool = True):

        # Set attributes from the config object
        self.config = config
        self.batch_size = batch_size
        self.use_multi_gpu = use_multi_gpu
        self.device = device if device is not None else torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Call the parent constructor which will trigger _prepare()
        super().__init__(corpus, top_k)

    def _prepare(self):
        """
        Loads the model and prepares the document embeddings.
        """
        log.info(f"Initializing DenseRetriever with model: {self.config.model_path}")
        self.model = SentenceTransformer(str(self.config.model_path), device=self.device)
        self.doc_ids = list(self.corpus.keys())

        if self.config.embs_path and self.config.embs_path.exists():
            log.info(f"Loading dense embeddings from {self.config.embs_path}")
            self.doc_embs = np.load(self.config.embs_path)
        else:
            log.info(f"Embeddings not found. Creating new embeddings for {len(self.corpus)} docs...")
            texts = [(d.get("title", "") + " " + d.get("text", "")).strip() for d in self.corpus.values()]

            if self.use_multi_gpu and torch.cuda.device_count() > 1:
                self._encode_corpus_multi_gpu(texts)
            else:
                self._encode_corpus_single_device(texts)
            
            if self.config.embs_path:
                os.makedirs(os.path.dirname(self.config.embs_path), exist_ok=True)
                log.info(f"Saving dense embeddings to {self.config.embs_path}")
                np.save(self.config.embs_path, self.doc_embs)

    def _encode_corpus_multi_gpu(self, texts: list):
        """Encodes the corpus using all available GPUs via multi-processing."""
        num_gpus = torch.cuda.device_count()
        target_devices = [f'cuda:{i}' for i in range(num_gpus)]
        log.info(f"🚀 Using multi-process encoding with {num_gpus} GPUs: {target_devices}")

        # The sentence-transformers library recommends its multi-process pool for this task
        pool = self.model.start_multi_process_pool(target_devices=target_devices)
        self.doc_embs = self.model.encode(texts, pool=pool, batch_size=self.batch_size)
        self.model.stop_multi_process_pool(pool)

    def _encode_corpus_single_device(self, texts: list):
        """Encodes the corpus using a single device."""
        log.info(f"Using single-device encoding on: {self.device}")
        self.doc_embs = self.model.encode(
            texts,
            batch_size=self.batch_size,
            show_progress_bar=True,
            convert_to_numpy=True,
            normalize_embeddings=True,
            device=self.device
        )

    def search(self, query_text: str, query_meta: Dict, top_k: int) -> Dict[str, float]:
        """Performs a search for a single query."""
        query_emb = self.model.encode(query_text, convert_to_numpy=True, normalize_embeddings=True)
        scores = (query_emb @ self.doc_embs.T).flatten()
        k = min(top_k, len(scores))
        if k == 0: return {}
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return {self.doc_ids[i]: float(scores[i]) for i in top_idx}
        
    def search_subset(self, query_text: str, subset_docs: Dict[str, str], top_k: int) -> Dict[str, float]:
        if not subset_docs: return {}
        doc_ids, doc_texts = list(subset_docs.keys()), list(subset_docs.values())
        query_emb = self.model.encode(query_text, convert_to_tensor=True, normalize_embeddings=True)
        doc_embs = self.model.encode(doc_texts, convert_to_tensor=True, batch_size=self.batch_size, normalize_embeddings=True)
        scores = util.cos_sim(query_emb, doc_embs)[0]
        k = min(top_k, len(scores))
        if k == 0: return {}
        top_vals, top_indices = torch.topk(scores, k=k)
        return {doc_ids[i.item()]: float(v.item()) for v, i in zip(top_vals, top_indices)}
    
