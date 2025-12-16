import os
import numpy as np
import torch
from torch.nn.parallel import DataParallel
from transformers import AutoTokenizer, AutoModelForMaskedLM
from scipy.sparse import csr_matrix, save_npz, load_npz, vstack
from typing import Dict, List, Optional
from tqdm import tqdm
from mar.strategy.base import BaseRetriever
from mar.utils.logging import get_logger

log = get_logger()

class SPLADEv3Encoder:
    def __init__(self, model_name: str, device=None, batch_size=32, use_multi_gpu=True):
        self.model_name = model_name
        self.batch_size = batch_size
        self.use_multi_gpu = use_multi_gpu
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu") if device is None else device
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModelForMaskedLM.from_pretrained(model_name)
        if self.use_multi_gpu and torch.cuda.device_count() > 1:
            log.info(f"Using {torch.cuda.device_count()} GPUs for SPLADE encoding")
            self.model = DataParallel(self.model)
        self.model.to(self.device)
        self.model.eval()

    def encode_text(self, texts: List[str], max_length=512, desc="Encoding"):
        all_sparse_matrices = []
        with torch.no_grad():
            for i in tqdm(range(0, len(texts), self.batch_size), desc=desc, leave=False):
                batch_texts = texts[i:i + self.batch_size]
                inputs = self.tokenizer(batch_texts, max_length=max_length, padding=True, truncation=True, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}
                logits = self.model(**inputs).logits
                sparse_vecs = torch.max(torch.log(1 + torch.relu(logits)) * inputs['attention_mask'].unsqueeze(-1), dim=1)[0]
                all_sparse_matrices.append(csr_matrix(sparse_vecs.cpu().numpy()))
        if not all_sparse_matrices:
            vocab_size = self.model.module.config.vocab_size if hasattr(self.model, 'module') else self.model.config.vocab_size
            return csr_matrix((0, vocab_size), dtype=np.float32)
        return vstack(all_sparse_matrices, format='csr')

    def encode_corpus(self, corpus: List[Dict], **kwargs):
        texts = [(doc.get('title', '') + ' ' + doc.get('text', '')).strip() for doc in corpus]
        return self.encode_text(texts, **kwargs)

class SpladeRetriever(BaseRetriever):
    def __init__(self, corpus: Dict, top_k: int, model_name: str, embs_path: Optional[str], device: torch.device, use_multi_gpu: bool = True):
        self.model_name = model_name
        self.embs_path = embs_path
        self.device = device
        self.use_multi_gpu = use_multi_gpu
        super().__init__(corpus, top_k)
    
    def _prepare(self):
        self.encoder = SPLADEv3Encoder(model_name=self.model_name, device=self.device, use_multi_gpu=self.use_multi_gpu)
        self.doc_ids = list(self.corpus.keys())
        self._load_or_compute_embeddings()

    def _load_or_compute_embeddings(self):
        if self.embs_path and os.path.exists(self.embs_path):
            log.info(f"Loading pre-computed sparse SPLADE embeddings from: {self.embs_path}")
            try:
                self.doc_embs = load_npz(self.embs_path)
                return
            except Exception as e:
                log.warning(f"Warning: Could not load embeddings. Error: {e}. Re-encoding corpus.")
        log.info("Encoding corpus with SPLADE...")
        corpus_list = [self.corpus[cid] for cid in self.doc_ids]
        sparse_embs = self.encoder.encode_corpus(corpus_list, desc="Encoding Corpus (SPLADE)")
        self.doc_embs = sparse_embs
        if self.embs_path:
            log.info(f"Saving computed SPLADE embeddings to: {self.embs_path}")
            os.makedirs(os.path.dirname(self.embs_path) or ".", exist_ok=True)
            save_npz(self.embs_path, self.doc_embs)

    def search(self, query_text: str, query_meta: Dict, top_k: int) -> Dict[str, float]:
        query_emb = self.encoder.encode_text([query_text], desc="Encode Query (SPLADE)")
        scores = (query_emb @ self.doc_embs.T).toarray().flatten()
        k = min(top_k, len(scores))
        if k == 0: return {}
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return {self.doc_ids[i]: float(scores[i]) for i in top_idx}

    def search_subset(self, query_text: str, subset_docs: Dict[str, str], top_k: int) -> Dict[str, float]:
        if not subset_docs: return {}
        doc_ids, doc_texts = list(subset_docs.keys()), list(subset_docs.values())
        query_emb = self.encoder.encode_text([query_text], desc="Encode Query (SPLADE)")
        doc_embs = self.encoder.encode_text(doc_texts, desc="Encode Subset (SPLADE)")
        scores = (query_emb @ doc_embs.T).toarray().flatten()
        k = min(top_k, len(scores))
        if k == 0: return {}
        top_idx = np.argpartition(scores, -k)[-k:]
        top_idx = top_idx[np.argsort(scores[top_idx])[::-1]]
        return {doc_ids[i]: float(scores[i]) for i in top_idx}