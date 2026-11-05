# notes-qa

A small Retrieval-Augmented Generation service for asking questions about uploaded PDFs. Answers are grounded in retrieved PDF text and returned with source file and page citations.

## Architecture

```text
PDF upload -> pypdf page extraction -> overlapping text chunks
                                      -> Sentence Transformers embeddings
                                      -> FAISS index + metadata.json

Question -> same embedding model -> top 4 FAISS matches -> Ollama or OpenAI
                                                         -> answer + file/page citations
```

## Setup

Requires Python 3.11+ and a running Ollama server by default.

```sh
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
cp .env.example .env
ollama pull llama3.2
uvicorn src.api:app --reload
```

Install and start [Ollama](https://ollama.com/download) before running these commands. Its default model is `llama3.2`; `ollama pull llama3.2` downloads it. Sentence Transformers downloads `all-MiniLM-L6-v2` on first ingestion, so that first run needs network access.

In another terminal, upload the included evaluation PDF:

```sh
curl -F 'files=@eval/notes.pdf' http://localhost:8000/ingest
```

Ask a question:

```sh
curl -X POST http://localhost:8000/ask \
  -H 'Content-Type: application/json' \
  -d '{"question":"When is the project meeting?"}'
```

`GET /health` checks service status. Upload your own PDFs the same way, replacing `eval/notes.pdf`. Keep the API running and run the citation evaluation in a third terminal:

```sh
python eval/run_eval.py
```

The evaluation requires `eval/notes.pdf` to be ingested first. It calls the running API and prints the measured share of expected file/page citations.

To use the OpenAI API, set `LLM_PROVIDER=openai` and `OPENAI_API_KEY` in `.env`. The key is read from the environment and is not stored in the source code. The embedding model is downloaded by Sentence Transformers on first use.

Docker: copy `.env.example` to `.env`, set any needed values, then run `docker compose up --build`. For Ollama running on the host, use `OLLAMA_BASE_URL=http://host.docker.internal:11434` in `.env`.

## Design choices

- **Chunk size: 500 characters, overlap: 100.** Small chunks keep retrieved context focused; overlap preserves sentences near boundaries. Both values are configurable.
- **Top-k: 4.** A short context is easier for a local model to use and explain. It is configurable.
- **Similarity threshold: 0.25.** Similarities use normalized vectors and FAISS inner product (cosine similarity). Matches below the configured threshold are omitted; adjust it against your PDFs. Threshold is configurable.
- **Metadata:** each indexed chunk stores the original upload filename and one-based PDF page number in `metadata.json`.
- **Storage:** FAISS vectors live in `data/index.faiss`; metadata and uploaded PDFs live under `data/`.
- **LLM:** Ollama is the default for local inference. OpenAI is the optional API-key provider.

## Tests and evaluation

Run tests with `pytest`. The API tests mock the LLM; the chunking and retrieval tests use a tiny generated PDF fixture.

Set `NOTES_QA_URL` to evaluate a different local API URL. Edit `eval/qa.json` to match other evaluation PDFs.

## Limitations

- Scanned PDFs need OCR; this project only extracts embedded PDF text.
- Character chunking is intentionally simple and may split sentences.
- Citation correctness depends on retrieval quality and the similarity threshold.
- This is a small local service without authentication, user separation, or upload-size limits.
- Evaluation checks citation file/page only; it does not score answer wording or factual accuracy.
