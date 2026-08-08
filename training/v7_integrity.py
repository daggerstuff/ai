"""V7 MASTER dataset integrity checks.

Validates V7 ChatML records against:
- token limits (per-message and total) via heuristic or tiktoken when available
- role validity (system | user | assistant)
- UTF-8 cleanliness (no replacement chars, no mojibake)
- schema completeness (messages list, role+content strings, non-empty content)

Used by `assemble_v7_master.py` (PIX-4232) to gate the pipeline and by
`tests/test_v7_integrity.py` (PIX-4243) for regression coverage.

CLI: ``python -m training.v7_integrity <file.jsonl>`` exits 0 when clean,
1 when any record fails validation.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Iterable

# Public defaults — callers may override per dataset.
MAX_TOKENS_PER_MESSAGE = 8192
MAX_TOTAL_TOKENS = 32768
VALID_ROLES = frozenset({"system", "user", "assistant"})

# Heuristic: ~4 chars per token. Tuned for English clinical text; conservative
# for code/multilingual content. tiktoken override below when available.
_CHARS_PER_TOKEN = 4

try:  # optional: accurate token counts for OpenAI tokenizers
    import tiktoken  # type: ignore[import-not-found]

    _ENCODER: object | None = tiktoken.get_encoding("cl100k_base")
except Exception:  # pragma: no cover - import-time only, no contract
    _ENCODER = None


def _estimate_tokens(text: str) -> int:
    """Estimate token count for ``text``.

    Uses tiktoken when available; falls back to ``len(text) // 4`` heuristic.
    Empty text returns 0.
    """
    if not text:
        return 0
    if _ENCODER is not None:
        try:
            return len(_ENCODER.encode(text))  # type: ignore[union-attr]
        except Exception:
            # fall through to heuristic on any encoder error
            pass
    return max(1, len(text) // _CHARS_PER_TOKEN)


def _has_replacement_char(text: str) -> bool:
    """True if ``text`` contains U+FFFD (replacement char from bad decoding)."""
    return "\ufffd" in text


def validate_record(
    record: dict,
    *,
    max_tokens_per_message: int = MAX_TOKENS_PER_MESSAGE,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> list[str]:
    """Return list of integrity errors for ``record``. Empty list = valid.

    Checks (in order):
    1. ``messages`` exists and is a non-empty list
    2. Each message is a dict with ``role`` (in VALID_ROLES) and ``content`` (non-empty str)
    3. No message content contains U+FFFD replacement char
    4. Per-message token count <= ``max_tokens_per_message``
    5. Total token count across all messages <= ``max_total_tokens``
    """
    errors: list[str] = []

    messages = record.get("messages")
    if messages is None:
        errors.append("missing 'messages' field")
        return errors
    if not isinstance(messages, list):
        errors.append(f"'messages' must be list, got {type(messages).__name__}")
        return errors
    if not messages:
        errors.append("'messages' is empty list")
        return errors

    total_tokens = 0
    for idx, msg in enumerate(messages):
        if not isinstance(msg, dict):
            errors.append(f"message[{idx}] not dict: {type(msg).__name__}")
            continue

        role = msg.get("role")
        if role is None:
            errors.append(f"message[{idx}] missing 'role'")
        elif not isinstance(role, str):
            errors.append(f"message[{idx}] 'role' not str: {type(role).__name__}")
        elif role not in VALID_ROLES:
            errors.append(f"message[{idx}] invalid role {role!r}; expected one of {sorted(VALID_ROLES)}")

        content = msg.get("content")
        if content is None:
            errors.append(f"message[{idx}] missing 'content'")
        elif not isinstance(content, str):
            errors.append(f"message[{idx}] 'content' not str: {type(content).__name__}")
        else:
            if not content.strip():
                errors.append(f"message[{idx}] 'content' empty/whitespace")
            if _has_replacement_char(content):
                errors.append(f"message[{idx}] 'content' contains U+FFFD replacement char (mojibake)")

            msg_tokens = _estimate_tokens(content)
            total_tokens += msg_tokens
            if msg_tokens > max_tokens_per_message:
                errors.append(f"message[{idx}] {msg_tokens} tokens > limit {max_tokens_per_message}")

    if total_tokens > max_total_tokens:
        errors.append(f"total {total_tokens} tokens > limit {max_total_tokens}")

    return errors


def validate_file(
    path: Path,
    *,
    max_tokens_per_message: int = MAX_TOKENS_PER_MESSAGE,
    max_total_tokens: int = MAX_TOTAL_TOKENS,
) -> tuple[int, list[tuple[int, list[str]]]]:
    """Validate every JSONL line in ``path``.

    Returns ``(total_lines, errors)`` where ``errors`` is a list of
    ``(line_idx, error_list)`` for lines that failed. ``total_lines`` counts
    non-empty lines (blanks skipped).
    """
    if not path.exists():
        raise FileNotFoundError(f"V7 file not found: {path}")

    total = 0
    failures: list[tuple[int, list[str]]] = []
    raw = path.read_bytes()  # read bytes to detect UTF-8 errors before json

    try:
        text = raw.decode("utf-8")
    except UnicodeDecodeError as exc:
        # Whole-file UTF-8 corruption is a single hard failure.
        raise UnicodeDecodeError(
            exc.encoding, exc.object, exc.start, exc.end, f"V7 file {path} not valid UTF-8"
        ) from exc

    for line_idx, line in enumerate(text.splitlines(), start=1):
        stripped = line.strip()
        if not stripped:
            continue
        total += 1
        try:
            record = json.loads(stripped)
        except json.JSONDecodeError as exc:
            failures.append((line_idx, [f"json decode error: {exc}"]))
            continue
        if not isinstance(record, dict):
            failures.append((line_idx, [f"record not dict: {type(record).__name__}"]))
            continue
        errs = validate_record(
            record,
            max_tokens_per_message=max_tokens_per_message,
            max_total_tokens=max_total_tokens,
        )
        if errs:
            failures.append((line_idx, errs))

    return total, failures


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description="Validate V7 MASTER JSONL records for token limits, role validity, UTF-8."
    )
    parser.add_argument("file", type=Path, help="Path to V7_MASTER.jsonl (or shard).")
    parser.add_argument(
        "--max_tokens_per_message",
        type=int,
        default=MAX_TOKENS_PER_MESSAGE,
        help=f"Per-message token limit (default {MAX_TOKENS_PER_MESSAGE}).",
    )
    parser.add_argument(
        "--max_total_tokens",
        type=int,
        default=MAX_TOTAL_TOKENS,
        help=f"Total record token limit (default {MAX_TOTAL_TOKENS}).",
    )
    return parser


def main(argv: Iterable[str] | None = None) -> int:
    args = build_parser().parse_args(list(argv) if argv is not None else None)
    total, failures = validate_file(
        args.file,
        max_tokens_per_message=args.max_tokens_per_message,
        max_total_tokens=args.max_total_tokens,
    )
    if not failures:
        print(f"OK: {total} records clean in {args.file}")
        return 0
    print(f"FAIL: {len(failures)}/{total} records invalid in {args.file}", file=sys.stderr)
    for line_idx, errs in failures:
        for err in errs:
            print(f"  line {line_idx}: {err}", file=sys.stderr)
    return 1


if __name__ == "__main__":
    sys.exit(main())
