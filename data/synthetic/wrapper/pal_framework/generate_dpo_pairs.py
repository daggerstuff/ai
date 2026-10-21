"""Build DPO preference pairs from PAL persona dialogue responses.

PIX-4074 — Phase 3.1: Preference Pair Construction (DPO).

Constructs preference pairs (chosen vs. rejected) to explicitly penalize
out-of-character or generic generations, per the PAL paper's DPO phase.

Each input record provides:
  - ``persona``: a Meddies patient record (rendered via ``format_persona``)
  - ``dialogue``: dialogue history (string or ChatML message list)
  - ``chosen_response``: a persona-aligned response (low health literacy
    language, cultural biases, in-character)
  - ``rejected_response``: a persona-violating response (high medical jargon,
    overly generic AI tone, out of character)

The contrastive LLM prompting that produces ``chosen_response`` and
``rejected_response`` happens upstream; this module formats, validates, and
enforces the 10,000-pair minimum (PIX-4074 AC).

Output JSONL conforms to the TRL ``DPOTrainer`` schema validated by
``lint_dpo_dataset.py``::

    {"prompt": str, "chosen": list[message], "rejected": list[message],
     "metadata": {...}}

where ``prompt`` is the user query string (system instruction + persona +
dialogue history) and ``chosen`` / ``rejected`` are single-turn assistant
continuations whose content differs.
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
# 4 chars ≈ 1 token — conservative proxy for mixed EN/VI text.
CHARS_PER_TOKEN = 4
DEFAULT_MAX_TOKENS = 1024
# PIX-4074 AC: yields 10,000 DPO pairs.
DEFAULT_MIN_RECORDS = 10_000

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


def build_prompt(persona_string: str, dialogue_history: str) -> str:
    """Build the user query string for the DPO pair.

    Follows the PAL paper prompt template, prefixed by the system instruction
    so ``prompt`` is self-contained for TRL ``DPOTrainer``.
    """
    return (
        f"{SYSTEM_PROMPT}\n\n"
        f"Given this persona: {persona_string}\n\n"
        f"Dialogue history:\n{dialogue_history}\n\n"
        f"Generate the next response."
    )


def _has_json_leakage(text: str) -> bool:
    """Return True if the text contains JSON structural characters.

    Flags curly braces and double quotes but NOT single quotes, which are
    legitimate natural-language punctuation (apostrophes, possessives).
    """
    return any(ch in text for ch in "{}\"")


def _response_has_json_leakage(response: str) -> bool:
    return _has_json_leakage(response)


def is_chatml_compliant(messages: list[dict[str, Any]]) -> bool:
    """Check that ``messages`` is a non-empty list of valid ChatML message dicts."""
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


def estimate_tokens_text(text: str) -> int:
    """Conservative token estimate for a raw string."""
    return (len(text) + CHARS_PER_TOKEN - 1) // CHARS_PER_TOKEN


def validate_token_bounds(
    prompt: str,
    chosen_response: str,
    rejected_response: str,
    max_tokens: int,
) -> bool:
    """Return True iff the full DPO example fits within ``max_tokens``.

    The bound is checked against the larger of chosen/rejected to guarantee
    both sides fit.
    """
    prompt_tokens = estimate_tokens_text(prompt)
    chosen_tokens = estimate_tokens_text(chosen_response)
    rejected_tokens = estimate_tokens_text(rejected_response)
    return prompt_tokens + max(chosen_tokens, rejected_tokens) <= max_tokens


@dataclass
class DpoPairExample:
    prompt: str
    chosen: list[dict[str, str]]
    rejected: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)


def build_dpo_pair(
    persona: dict[str, Any],
    dialogue: str | list[dict[str, str]] | None,
    chosen_response: str,
    rejected_response: str,
    max_tokens: int = DEFAULT_MAX_TOKENS,
) -> DpoPairExample:
    """Build a single DPO preference-pair record.

    Raises ``ValueError`` if:
      - chosen and rejected responses are identical (no preference signal)
      - either response contains JSON formatting leakage
      - the example exceeds ``max_tokens``
    """
    if not chosen_response:
        raise ValueError("chosen_response must be non-empty")
    if not rejected_response:
        raise ValueError("rejected_response must be non-empty")
    if chosen_response == rejected_response:
        raise ValueError("chosen_response and rejected_response must differ")

    if _response_has_json_leakage(chosen_response):
        raise ValueError("chosen_response contains JSON formatting leakage")
    if _response_has_json_leakage(rejected_response):
        raise ValueError("rejected_response contains JSON formatting leakage")

    persona_string = format_persona_safe(persona)
    dialogue_history = format_dialogue_history(dialogue)
    prompt = build_prompt(persona_string, dialogue_history)

    if not validate_token_bounds(prompt, chosen_response, rejected_response, max_tokens):
        prompt_tokens = estimate_tokens_text(prompt)
        resp_tokens = max(
            estimate_tokens_text(chosen_response),
            estimate_tokens_text(rejected_response),
        )
        raise ValueError(f"example exceeds token bound: estimated {prompt_tokens + resp_tokens} > {max_tokens}")

    chosen_messages = [{"role": "assistant", "content": chosen_response}]
    rejected_messages = [{"role": "assistant", "content": rejected_response}]

    n_dialogue_turns: int
    if isinstance(dialogue, list):
        n_dialogue_turns = len(dialogue)
    elif dialogue_history:
        n_dialogue_turns = 1
    else:
        n_dialogue_turns = 0

    return DpoPairExample(
        prompt=prompt,
        chosen=chosen_messages,
        rejected=rejected_messages,
        metadata={
            "persona_string": persona_string,
            "n_dialogue_turns": n_dialogue_turns,
            "chosen_estimated_tokens": estimate_tokens_text(chosen_response),
            "rejected_estimated_tokens": estimate_tokens_text(rejected_response),
            "prompt_estimated_tokens": estimate_tokens_text(prompt),
        },
    )


def generate_dataset(
    input_path: Path,
    output_path: Path,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    reject_oversize: bool = False,
    min_records: int = DEFAULT_MIN_RECORDS,
) -> int:
    """Read input JSONL of DPO pair inputs and write TRL-compatible DPO JSONL.

    Input record schema::

        {"persona": dict, "dialogue": str | list | null,
         "chosen_response": str, "rejected_response": str}

    Returns the number of records written. Raises ``ValueError`` if fewer
    than ``min_records`` are written (PIX-4074 AC: minimum 10,000 pairs).
    When ``reject_oversize`` is True, records exceeding ``max_tokens`` are
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
            chosen_response = record["chosen_response"]
            rejected_response = record["rejected_response"]
            try:
                example = build_dpo_pair(
                    persona,
                    dialogue,
                    chosen_response,
                    rejected_response,
                    max_tokens=max_tokens,
                )
            except ValueError as exc:
                if reject_oversize and "token bound" in str(exc):
                    skipped += 1
                    continue
                raise
            fout.write(
                json.dumps(
                    {
                        "prompt": example.prompt,
                        "chosen": example.chosen,
                        "rejected": example.rejected,
                        "metadata": example.metadata,
                    },
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    if count < min_records:
        raise ValueError(f"wrote {count} records, below minimum {min_records} required by PIX-4074")
    if skipped:
        sys.stderr.write(f"skipped {skipped} oversize records\n")
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        description="Generate PAL Phase 3.1 DPO preference-pair dataset.",
    )
    parser.add_argument(
        "input",
        type=Path,
        help="Input JSONL: {persona, dialogue?, chosen_response, rejected_response}",
    )
    parser.add_argument(
        "output",
        type=Path,
        help="Output JSONL of TRL DPO records",
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
        help="Minimum number of DPO pairs required (PIX-4074 AC: 10000).",
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
    print(f"wrote {n} DPO pairs to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
