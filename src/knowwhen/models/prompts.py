"""Prompt templates. Fixed here so every run uses identical wording."""
from __future__ import annotations

CLOSED = "Answer the question with a short phrase only. Do not explain.\n\nQuestion: {question}"
RAG = (
    "Use the passages to answer the question with a short phrase only. Do not explain.\n\n"
    "{passages}\n\nQuestion: {question}"
)


def closed_prompt(question: str) -> str:
    return CLOSED.format(question=question)


def rag_prompt(question: str, passages: list[dict]) -> str:
    blocks = [f"Passage {i + 1} ({p['title']}): {p['text']}" for i, p in enumerate(passages)]
    return RAG.format(passages="\n\n".join(blocks), question=question)
