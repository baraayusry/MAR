import os
import json
from beir.datasets.data_loader import GenericDataLoader
from mar.utils.logging import get_logger

log = get_logger()

class BEIRDataset:
    def __init__(self, dataset_dir: str, queries_override_path: str = None):
        log.info(f"Loading BEIR dataset from: {dataset_dir}")

        self.corpus = {}
        self.queries = {}
        self.qrels = {}

        try:
            self.corpus, self.queries, self.qrels = GenericDataLoader(dataset_dir).load(split="test")
        except Exception as e:
            log.error(f"GenericDataLoader warning (non-fatal): {e}")

        # Safeguard: Ensure variables are dictionaries, not None
        if self.corpus is None: self.corpus = {}
        if self.queries is None: self.queries = {}
        if self.qrels is None: self.qrels = {}

        self.queries_meta = {}
        
        # Load overrides and backfill queries if missing
        if queries_override_path and os.path.exists(queries_override_path):
            log.info(f"Loading query metadata/overrides from: {queries_override_path}")
            with open(queries_override_path, 'r', encoding='utf-8') as f:
                for line in f:
                    if not line.strip(): continue
                    try:
                        data = json.loads(line)
                        qid = data.get('_id')
                        text = data.get('text', '')
                        
                        if qid:
                            # 1. Store metadata
                            self.queries_meta[qid] = data
                            
                            # 2. If the query text was missing from the main loader, add it here
                            if qid not in self.queries and text:
                                self.queries[qid] = text
                    except json.JSONDecodeError:
                        continue

        log.info(f"Dataset Summary: {len(self.corpus)} docs, {len(self.queries)} queries, {len(self.qrels)} qrels.")
        
        if len(self.queries) == 0:
            log.error("CRITICAL: No queries loaded! Check dataset_dir or queries_override_jsonl path.")