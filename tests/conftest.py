from pathlib import Path

import pytest
from pypdf import PdfWriter


@pytest.fixture
def fixture_pdf(tmp_path: Path) -> Path:
    """Build a tiny one-page PDF fixture for extraction and retrieval tests."""
    path = tmp_path / "notes.pdf"
    writer = PdfWriter()
    page = writer.add_blank_page(width=300, height=300)
    from pypdf.generic import (
        DecodedStreamObject,
        DictionaryObject,
        NameObject,
    )

    stream = DecodedStreamObject()
    stream.set_data(b"BT /F1 12 Tf 20 200 Td (The project meeting is on Tuesday.) Tj ET")
    page[NameObject("/Contents")] = writer._add_object(stream)
    font = DictionaryObject({
        NameObject("/Type"): NameObject("/Font"),
        NameObject("/Subtype"): NameObject("/Type1"),
        NameObject("/BaseFont"): NameObject("/Helvetica"),
    })
    page[NameObject("/Resources")] = DictionaryObject({
        NameObject("/Font"): DictionaryObject({NameObject("/F1"): writer._add_object(font)})
    })
    with path.open("wb") as output:
        writer.write(output)
    return path
