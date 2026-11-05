"""Sentence-transformer embeddings and FAISS persistence."""

import json
from pathlib import Path

import faiss
import numpy as np
from sentence_transformers import SentenceTransformer


class VectorStore:
    def __init__(self, model_name: str, index_path: Path, metadata_path: Path):
        self.model = SentenceTransformer(model_name)
        self.index_path = index_path
        self.metadata_path = metadata_path
        self.index = None
        self.records: list[dict] = []
        self.load()

    def load(self) -> None:
        if self.index_path.exists() and self.metadata_path.exists():
            self.index = faiss.read_index(str(self.index_path))
            self.records = json.loads(self.metadata_path.read_text(encoding="utf-8"))

    def add(self, chunks: list[dict]) -> None:
        if not chunks:
            return
        vectors = self.model.encode([chunk["text"] for chunk in chunks], convert_to_numpy=True)
        vectors = np.asarray(vectors, dtype="float32")
        faiss.normalize_L2(vectors)
        if self.index is None:
            self.index = faiss.IndexFlatIP(vectors.shape[1])
        self.index.add(vectors)
        self.records.extend(chunks)
        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(self.index_path))
        self.metadata_path.write_text(json.dumps(self.records, ensure_ascii=False, indent=2), encoding="utf-8")

    def search(self, question: str, top_k: int, threshold: float) -> list[dict]:
        if self.index is None or self.index.ntotal == 0:
            return []
        vector = np.asarray(self.model.encode([question], convert_to_numpy=True), dtype="float32")
        faiss.normalize_L2(vector)
        scores, indices = self.index.search(vector, top_k)
        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx < 0 or float(score) < threshold:
                continue
            results.append({**self.records[int(idx)], "score": float(score)})
        return results
