"""Build SFT Task 1 dataset: dialogue-informed persona selection.

PIX-4072 — Phase 2.1: Dialogue-Informed Persona Selection (SFT Task 1).

The model must learn to implicitly deduce the patient persona from a dialogue
history. For each synthetic dialogue we render the ground-truth persona plus
3-4 distractor personas to natural language (via
``meddies_to_pal.format_persona``) and emit a strictly ChatML-compliant SFT
record whose target is the correct option index.
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

from meddies_to_pal import format_persona

VALID_ROLES = ("system", "user", "assistant", "tool")
DEFAULT_N_DISTRACTORS = 3


def format_persona_safe(persona: dict[str, Any]) -> str:
    """Render a persona to NL, guarding against JSON leakage from the source dict."""
    return format_persona(persona).strip()


def sample_distractors(
    pool: list[dict[str, Any]],
    correct: dict[str, Any],
    n_distractors: int,
    rng: random.Random,
) -> list[dict[str, Any]]:
    if n_distractors < 1:
        raise ValueError("n_distractors must be >= 1")
    candidates = [p for p in pool if p != correct]
    if len(candidates) < n_distractors:
        raise ValueError(f"persona pool has only {len(candidates)} distractors; need {n_distractors}")
    return rng.sample(candidates, n_distractors)


def build_selection_messages(
    dialogue: str,
    candidates: list[str],
    correct_index: int,
) -> list[dict[str, str]]:
    if not (0 <= correct_index < len(candidates)):
        raise ValueError("correct_index out of range for candidates")
    numbered = "\n".join(f"{i + 1}. {c}" for i, c in enumerate(candidates))
    system = (
        "You are a clinical persona classifier. Given a patient dialogue and a "
        "numbered list of candidate personas, select the single persona that best "
        "matches the dialogue. Respond with only the option number."
    )
    user = (
        f"Dialogue:\n{dialogue}\n\n"
        f"Candidate personas:\n{numbered}\n\n"
        f"Which persona (1-{len(candidates)}) best matches this dialogue? "
        f"Respond with only the number."
    )
    return [
        {"role": "system", "content": system},
        {"role": "user", "content": user},
        {"role": "assistant", "content": str(correct_index + 1)},
    ]


def is_chatml_compliant(messages: list[dict[str, Any]]) -> bool:
    if not messages:
        return False
    for m in messages:
        if not isinstance(m, dict):
            return False
        if m.get("role") not in VALID_ROLES:
            return False
        if not isinstance(m.get("content"), str):
            return False
    return True


@dataclass
class SelectionExample:
    messages: list[dict[str, str]]
    metadata: dict[str, Any] = field(default_factory=dict)


def build_selection_example(
    dialogue: str,
    personas: list[dict[str, Any]],
    correct_index: int,
    n_distractors: int = DEFAULT_N_DISTRACTORS,
    rng: random.Random | None = None,
) -> SelectionExample:
    rng = rng or random.Random()
    distractors = sample_distractors(personas, personas[correct_index], n_distractors, rng)
    options = [personas[correct_index], *distractors]
    rng.shuffle(options)
    chosen_index = options.index(personas[correct_index])
    candidate_strings = [format_persona_safe(p) for p in options]
    messages = build_selection_messages(dialogue, candidate_strings, chosen_index)
    return SelectionExample(
        messages=messages,
        metadata={
            "correct_persona": personas[correct_index],
            "correct_option": chosen_index + 1,
            "n_options": len(options),
        },
    )


def generate_dataset(
    input_path: Path,
    output_path: Path,
    n_distractors: int = DEFAULT_N_DISTRACTORS,
    seed: int | None = None,
) -> int:
    """Read an input JSONL of {dialogue, personas, correct_index} and write ChatML records."""
    rng = random.Random(seed)
    count = 0
    with input_path.open(encoding="utf-8") as fin, output_path.open("w", encoding="utf-8") as fout:
        for raw in fin:
            line = raw.strip()
            if not line:
                continue
            record = json.loads(line)
            dialogue = record["dialogue"]
            personas = record["personas"]
            correct_index = int(record["correct_index"])
            example = build_selection_example(dialogue, personas, correct_index, n_distractors=n_distractors, rng=rng)
            fout.write(
                json.dumps(
                    {"messages": example.messages, "metadata": example.metadata},
                    ensure_ascii=False,
                )
                + "\n"
            )
            count += 1
    return count


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Generate PAL SFT Task 1 persona-selection dataset.")
    parser.add_argument("input", type=Path, help="Input JSONL: {dialogue, personas, correct_index}")
    parser.add_argument("output", type=Path, help="Output JSONL of ChatML SFT records")
    parser.add_argument("--n-distractors", type=int, default=DEFAULT_N_DISTRACTORS)
    parser.add_argument("--seed", type=int, default=None)
    args = parser.parse_args(argv)
    if not args.input.exists():
        print(f"input not found: {args.input}", file=sys.stderr)
        return 1
    n = generate_dataset(args.input, args.output, n_distractors=args.n_distractors, seed=args.seed)
    print(f"wrote {n} records to {args.output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
