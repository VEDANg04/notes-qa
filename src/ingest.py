"""PDF extraction and character-based chunking."""

from pathlib import Path

from pypdf import PdfReader


def split_text(text: str, chunk_size: int, overlap: int) -> list[str]:
    """Split text into fixed-size chunks with a fixed overlap."""
    if chunk_size <= 0 or overlap < 0 or overlap >= chunk_size:
        raise ValueError("chunk_size must be positive and overlap smaller than chunk_size")
    text = text.strip()
    if not text:
        return []
    step = chunk_size - overlap
    return [text[start : start + chunk_size] for start in range(0, len(text), step)]


def read_pdf(path: str | Path, chunk_size: int, overlap: int) -> list[dict]:
    """Extract page text and return chunks carrying their source metadata."""
    pdf_path = Path(path)
    reader = PdfReader(str(pdf_path))
    chunks = []
    for page_number, page in enumerate(reader.pages, start=1):
        for chunk in split_text(page.extract_text() or "", chunk_size, overlap):
            chunks.append({
                "text": chunk,
                "metadata": {"file_name": pdf_path.name, "page_number": page_number},
            })
    return chunks
