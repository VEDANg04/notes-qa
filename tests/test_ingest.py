from src.ingest import split_text


def test_split_text_uses_configured_size_and_overlap():
    chunks = split_text("a" * 1200, chunk_size=500, overlap=100)

    assert [len(chunk) for chunk in chunks] == [500, 500, 400]
    assert chunks[0][-100:] == chunks[1][:100]
    assert chunks[1][-100:] == chunks[2][:100]


def test_split_text_handles_short_and_empty_text():
    assert split_text("short", chunk_size=500, overlap=100) == ["short"]
    assert split_text("  ", chunk_size=500, overlap=100) == []
