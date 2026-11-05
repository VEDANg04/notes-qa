"""FastAPI endpoints for PDF ingestion and document question answering."""

from pathlib import Path

from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel

from src.answer import answer_question
from src.config import settings
from src.ingest import read_pdf
from src.retrieve import create_store, retrieve

app = FastAPI(title="notes-qa")
_store = None


def get_store():
    global _store
    if _store is None:
        settings.data_dir.mkdir(parents=True, exist_ok=True)
        _store = create_store(settings.data_dir, settings.embedding_model)
    return _store


class AskRequest(BaseModel):
    question: str


@app.get("/health")
def health():
    return {"status": "ok"}


@app.post("/ingest")
async def ingest(files: list[UploadFile] = File(...)):
    chunks = []
    upload_dir = settings.data_dir / "uploads"
    upload_dir.mkdir(parents=True, exist_ok=True)
    for upload in files:
        if not upload.filename or Path(upload.filename).suffix.lower() != ".pdf":
            raise HTTPException(status_code=400, detail="Only PDF files are accepted")
        file_name = Path(upload.filename).name
        destination = upload_dir / file_name
        content = await upload.read()
        destination.write_bytes(content)
        try:
            chunks.extend(read_pdf(destination, settings.chunk_size, settings.chunk_overlap))
        except Exception as exc:
            destination.unlink(missing_ok=True)
            raise HTTPException(status_code=400, detail=f"Could not read {file_name} as a PDF") from exc
    if chunks:
        get_store().add(chunks)
    return {"files": len(files), "chunks": len(chunks)}


@app.post("/ask")
def ask(request: AskRequest):
    try:
        chunks = retrieve(request.question, get_store(), settings.top_k, settings.similarity_threshold)
        text = answer_question(request.question, chunks)
    except ValueError as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc
    sources = []
    seen = set()
    for chunk in chunks:
        metadata = chunk["metadata"]
        citation = {"file_name": metadata["file_name"], "page_number": metadata["page_number"]}
        key = (citation["file_name"], citation["page_number"])
        if key not in seen:
            seen.add(key)
            sources.append(citation)
    return {"answer": text, "sources": sources}
