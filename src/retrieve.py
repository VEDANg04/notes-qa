"""Question retrieval from the persistent vector store."""

from pathlib import Path

from src.embed import VectorStore


def retrieve(question: str, store: VectorStore, top_k: int, threshold: float) -> list[dict]:
    """Return the best matching chunks, or no results below the threshold."""
    return store.search(question, top_k=top_k, threshold=threshold)


def create_store(data_dir: Path, model_name: str) -> VectorStore:
    return VectorStore(model_name, data_dir / "index.faiss", data_dir / "metadata.json")
