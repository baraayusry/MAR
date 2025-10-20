# datasets/loaders.py
import json
from typing import Optional
from beir.datasets.data_loader import GenericDataLoader
from mar.datasets.base import BaseDatasetLoader, Dataset

class BeirDatasetLoader(BaseDatasetLoader):
    def __init__(self, config: BeirConfig):
        self.config = config
        print(f"📦 BeirDatasetLoader initialized for directory: {self.config.dataset_dir}")

    def load(self) -> Dataset:
        print("--- 📦 Loading BEIR dataset ---")
        corpus, queries, qrels = GenericDataLoader(str(self.config.dataset_dir)).load(split="test")
        queries_meta = {}
        if self.config.queries_override_jsonl and self.config.queries_override_jsonl.exists():
            print(f"   |  Loading query metadata from: {self.config.queries_override_jsonl}")
            with open(self.config.queries_override_jsonl, 'r') as f:
                for line in f:
                    data = json.loads(line)
                    queries_meta[data['_id']] = data
        print(f"--- ✅ BEIR data loaded: {len(corpus)} docs, {len(queries)} queries ---")
        return Dataset(corpus=corpus, queries=queries, qrels=qrels, queries_meta=queries_meta)
