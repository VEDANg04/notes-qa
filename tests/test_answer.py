from types import SimpleNamespace

from src.answer import UNKNOWN_ANSWER, answer_question


def test_answer_returns_unknown_without_context():
    assert answer_question("What day?", []) == UNKNOWN_ANSWER


def test_answer_prompt_and_unknown_response(monkeypatch):
    captured = {}

    def mock_chat(**kwargs):
        captured.update(kwargs)
        return {"message": {"content": UNKNOWN_ANSWER}}

    monkeypatch.setitem(__import__("sys").modules, "ollama", SimpleNamespace(
        Client=lambda host: SimpleNamespace(chat=mock_chat)
    ))
    chunks = [{"text": "Meeting is Tuesday.", "metadata": {"file_name": "notes.pdf", "page_number": 1}}]

    answer = answer_question("Who is the presenter?", chunks, provider="ollama")

    assert answer == UNKNOWN_ANSWER
    assert UNKNOWN_ANSWER in captured["messages"][0]["content"]
    assert "Answer only from the context" in captured["messages"][0]["content"]
