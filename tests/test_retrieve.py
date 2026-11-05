from src.embed import VectorStore
from src.ingest import read_pdf
from src.retrieve import retrieve


def test_retrieval_uses_pdf_page_metadata_and_threshold(fixture_pdf, monkeypatch):
    class TinyEmbedder:
        def encode(self, texts, convert_to_numpy=True):
            import numpy as np

            return np.array([[1.0, 0.0] if text.lower() == "meeting" else [1.0, 1.0] for text in texts])

    from src import embed

    monkeypatch.setattr(embed, "SentenceTransformer", lambda _model_name: TinyEmbedder())
    store = VectorStore("test-model", fixture_pdf.parent / "index.faiss", fixture_pdf.parent / "metadata.json")
    store.add(read_pdf(fixture_pdf, 500, 100))

    found = retrieve("meeting", store, top_k=4, threshold=0.5)
    missing = retrieve("meeting", store, top_k=4, threshold=0.95)

    assert found[0]["metadata"] == {"file_name": "notes.pdf", "page_number": 1}
    assert missing == []


def test_store_persists_index_and_metadata(tmp_path, monkeypatch):
    import numpy as np
    import src.embed as embed

    class TinyEmbedder:
        def encode(self, texts, convert_to_numpy=True):
            return np.array([[1.0, 0.0] for _ in texts])

    monkeypatch.setattr(embed, "SentenceTransformer", lambda _model_name: TinyEmbedder())
    index_path = tmp_path / "index.faiss"
    metadata_path = tmp_path / "metadata.json"
    store = VectorStore("test-model", index_path, metadata_path)
    store.add([{"text": "Tuesday meeting", "metadata": {"file_name": "notes.pdf", "page_number": 1}}])

    loaded = VectorStore("test-model", index_path, metadata_path)

    assert loaded.search("meeting", top_k=4, threshold=0.5)[0]["metadata"]["page_number"] == 1
