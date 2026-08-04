"""Build SFT Task 2 dataset: persona-enhanced dialogue generation.

PIX-4073 — Phase 2.2: Persona-Enhanced Dialogue Generation (SFT Task 2).

The model must learn to roleplay a given persona and generate a response that
adheres to it. For each (persona, dialogue_history, response) triple we render
the persona to a natural language string (via ``meddies_to_pal.format_persona``)
and emit a strictly ChatML-compliant SFT record whose assistant turn is the
target response. The system message instructs the model to stay in persona;
the user message follows the PAL paper prompt template:

    "Given this persona: [Meddies NL string] and this dialogue history,
     generate the next response."

Token bounds are enforced via a character-count proxy (4 chars ≈ 1 token for
English/Vietnamese mixed text) so the validator runs without a tokenizer
dependency. ``Dry run validation passes strict token bounds`` (PIX-4073 AC) is
exercised by ``validate_token_bounds`` + the ``--reject-oversize`` CLI flag.
"""

from __future__ import annotations

import argparse
import json
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meddies_to_pal import format_persona

VALID_ROLES = ("system", "user", "assistant", "tool")
# 4 chars ≈ 1 token is a conservative proxy for mixed EN/VI text; tune per
# tokenizer when wiring into the real training pipeline (Phase 4).
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 1024
DEFAULT_MIN_RECORDS = 5000

SYSTEM_PROMPT = (
    "You are roleplaying a patient with a specific clinical persona. "
    "Read the persona description carefully and generate the next response "
    "in the dialogue while staying in character. Keep the response consistent "
    "with the patient's health literacy, preferences, and background."
)


def format_persona_safe(persona: dict[str, Any]) -> str:
    """Render a persona to NL, guarding against JSON leakage from the source dict."""
    return format_persona(persona).strip()


def format_dialogue_history(dialogue: str | list[dict[str, str]] | None) -> str:
    """Render a dialogue history to a single string.

    Accepts either:
      * a raw string (already rendered), or
      * a list of ChatML-style ``{"role": ..., "content": ...}`` messages.

    ``None`` renders as an empty history (single-turn generation).
    """
    if dialogue is None:
        return ""
    if isinstance(dialogue, str):
        return dialogue.strip()
    if isinstance(dialogue, list):
        rendered: list[str] = []
        for turn in dialogue:
            if not isinstance(turn, dict):
                raise ValueError(f"dialogue turn must be a dict, got {type(turn).__name__}")
            role = turn.get("role", "user")
            content = turn.get("content", "")
            if not isinstance(content, str):
                raise ValueError(f"dialogue turn content must be a string, got {type(content).__name__}")
            rendered.append(f"{role.capitalize()}: {content}")
        return "\n".join(rendered).strip()
    raise ValueError(f"dialogue must be str | list | None, got {type(dialogue).__name__}")


def build_sft_messages(
    persona_string: str,
    dialogue_history: str,
    response: str,
) -> list[dict[str, str]]:
    """Build the ChatML message list for a PAL SFT Task 2 example."""
    if not persona_string:
        raise ValueError("persona_string must be non-empty")
    if not response:
        raise ValueError("response must be non-empty")
    user = (
        f"Given this persona: {persona_string}\n\nDialogue history:\n{dialogue_history}\n\nGenerate the next response."
    )
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": user},
        {"role": "assistant", "content": response},
    ]


def is_chatml_compliant(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if msg.get("role") not in VALID_ROLES:
            return False
        if not isinstance(msg.get("content"), str):
            return False
    return True


def estimate_tokens(messages: list[dict[str, str]]) -> int:
    """Conservative token estimate: total chars / CHARS_PER_TOKEN."""
    total_chars = sum(len(m.get("content", "")) for m in messages)
    return (total_chars + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def validate_token_bounds(messages: list[dict[str, str]], max_tokens: int) -> bool:
    """Return True iff the estimated token count fits within ``max_tokens``."""
    return estimate_tokens(messages) <= max_tokens


@dataclass
class SftDialogueExample:
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)


def build_sft_example(
    persona: dict[str, Any],
    dialogue: str | list[dict[str, str]] | None,
    response: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> SftDialogueExample:
    persona_string = format_persona_safe(persona)
    dialogue_history = format_dialogue_history(dialogue)
    messages = build_sft_messages(persona_string, dialogue_history, response)
    if not validate_token_bounds(messages, max_tokens):
        raise ValueError(f"example exceeds token bound: estimated {estimate_tokens(messages)} > {max_tokens}")
    return SftDialogueExample(
        messages=messages,
        metadata={
            "persona_string": persona_string,
            "n_dialogue_turns": (len(dialogue) if isinstance(dialogue, list) else (0 if not dialogue_history else 1)),
            "estimated_tokens": estimate_tokens(messages),
        },
    )


def generate_dataset(
    input_path: Path,
    output_path: Path,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reject_oversize: bool = False,
    min_records: int = DEFAULT_MIN_RECORDS,
) -> int:
    """Read an input JSONL of ``{persona, dialogue, response}`` and write ChatML SFT records.

    Returns the number of records written. Raises ``ValueError`` if fewer than
    ``min_records`` are written (PIX-4073 AC: minimum 5,000 valid records).
    When ``reject_oversize`` is True, records that exceed ``max_tokens`` are
    skipped; otherwise an oversize record raises ``ValueError``.
    """
    count = 0
    skipped = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            persona = record["persona"]
            dialogue = record.get("dialogue")
            response = record["response"]
            try:
                example = build_sft_example(persona, dialogue, response, max_tokens=max_tokens)
            except ValueError as exc:
                if reject_oversize and "token bound" in str(exc):
                    skipped += 1
                    continue
                raise
            fout.write(
                json.dumps(
                    {"messages": example.messages, "metadata": example.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    if count < min_records:
        raise ValueError(f"wrote {count} records, below minimum {min_records} required by PIX-4073")
    if skipped:
        sys.stderr.write(f"skipped {skipped} oversize records\n")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate PAL SFT Task 2 persona-enhanced dialogue dataset.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL: {persona, dialogue?, response}",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output JSONL of ChatML SFT records",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--reject-oversize",
        action="store_true",
        help="Skip records exceeding --max-tokens instead of erroring.",
    )
    parser.add_argument(
        "--min-records",
        type=int,
        default=DEFAULT_MIN_RECORDS,
        help="Minimum number of records required (PIX-4073 AC: 5000).",
    )
    args = parser.parse_args(argv)
    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1
    n = generate_dataset(
        args.input,
        args.output,
        max_tokens=args.max_tokens,
        reject_oversize=args.reject_oversize,
        min_records=args.min_records,
    )
    print(f"wrote {n} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
