"""Grounded question answering using the configured LLM provider."""

import os

from src.config import settings

UNKNOWN_ANSWER = "I don't know based on the provided documents."
SYSTEM_PROMPT = (
    "Answer only from the context provided. If the context does not contain the answer, "
    f'reply exactly: "{UNKNOWN_ANSWER}" Do not use outside knowledge.'
)


def answer_question(question: str, chunks: list[dict], provider: str | None = None) -> str:
    if not chunks:
        return UNKNOWN_ANSWER
    context = "\n\n".join(
        f"Source: {chunk['metadata']['file_name']}, page {chunk['metadata']['page_number']}\n{chunk['text']}"
        for chunk in chunks
    )
    selected_provider = (provider or settings.llm_provider).lower()
    if selected_provider == "ollama":
        import ollama

        client = ollama.Client(host=settings.ollama_base_url)
        response = client.chat(
            model=settings.ollama_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return response["message"]["content"].strip()
    if selected_provider == "openai":
        from openai import OpenAI

        key = settings.openai_api_key or os.getenv("OPENAI_API_KEY")
        if not key:
            raise ValueError("OPENAI_API_KEY is required when LLM_PROVIDER=openai")
        response = OpenAI(api_key=key).chat.completions.create(
            model=settings.openai_model,
            messages=[{"role": "system", "content": SYSTEM_PROMPT},
                      {"role": "user", "content": f"Context:\n{context}\n\nQuestion: {question}"}],
        )
        return (response.choices[0].message.content or UNKNOWN_ANSWER).strip()
    raise ValueError("LLM_PROVIDER must be 'ollama' or 'openai'")
