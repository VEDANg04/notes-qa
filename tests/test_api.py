from dataclasses import replace

from fastapi.testclient import TestClient

from src import api


class FakeStore:
    def __init__(self):
        self.chunks = []

    def add(self, chunks):
        self.chunks.extend(chunks)

    def search(self, question, top_k, threshold):
        return [{"text": "Meeting Tuesday", "metadata": {"file_name": "notes.pdf", "page_number": 1}, "score": 0.8}]


def test_health_endpoint():
    client = TestClient(api.app)
    assert client.get("/health").json() == {"status": "ok"}


def test_ingest_uploads_pdf_and_indexes_chunks(monkeypatch, tmp_path, fixture_pdf):
    store = FakeStore()
    monkeypatch.setattr(api, "get_store", lambda: store)
    monkeypatch.setattr(api, "settings", replace(api.settings, data_dir=tmp_path / "data"))
    response = TestClient(api.app).post(
        "/ingest",
        files={"files": ("notes.pdf", fixture_pdf.read_bytes(), "application/pdf")},
    )

    assert response.status_code == 200
    assert response.json() == {"files": 1, "chunks": 1}
    assert store.chunks[0]["metadata"] == {"file_name": "notes.pdf", "page_number": 1}


def test_ask_endpoint_with_mocked_llm(monkeypatch):
    monkeypatch.setattr(api, "get_store", lambda: FakeStore())
    monkeypatch.setattr(api, "answer_question", lambda question, chunks: "The meeting is Tuesday.")
    response = TestClient(api.app).post("/ask", json={"question": "When is the meeting?"})

    assert response.status_code == 200
    assert response.json() == {
        "answer": "The meeting is Tuesday.",
        "sources": [{"file_name": "notes.pdf", "page_number": 1}],
    }
