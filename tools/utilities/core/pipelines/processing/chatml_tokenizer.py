"""Tokenization helpers for ChatML records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass
class TokenizedRecord:
    tokens: list[str]
    char_count: int
    record: dict[str, Any]


def tokenize_dataset(records: list[dict[str, Any]], *, tokeniser: Any | None = None) -> list[TokenizedRecord]:
    """Tokenize converted ChatML records.

    This is intentionally lightweight and safe to run without optional tokenizer
    dependencies. If a tokenizer is provided it is used as-is; otherwise a simple
    whitespace tokenizer is applied.
    """

    output: list[TokenizedRecord] = []

    for record in records:
        text_chunks = [
            str(part.get("content", "")) for part in record.get("chatml_messages", []) if isinstance(part, dict)
        ]
        text = "\n".join(text_chunks)

        tokens = list(tokeniser(text)) if tokeniser is not None and callable(tokeniser) else text.split()

        output.append(TokenizedRecord(tokens=tokens, char_count=len(text), record=record))

    return output


__all__ = ["TokenizedRecord", "tokenize_dataset"]
