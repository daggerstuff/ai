"""Put a routing sentence in front of every acute reply that has none.

Routing the arc risk turns fixes the signal the ledger labels carry, but it does
not fix the plain therapy rows: 1,600 assistant replies in the v17 train split
answer an acute disclosure — suicidal ideation, self-harm, overdose, a plan —
without naming a single resource. That is 78% of all acute replies in the
corpus, and it is what fine-tuning learns from: reflect, do not route.

These rows do not need a model. The reply itself is worth keeping, so the fix is
a deterministic prefix: a short imperative routing sentence followed by the
original reply, verbatim. The corpus then demonstrates "route first, then
engage", which is the behaviour the base model already has and every adapter so
far loses.

The prefix is chosen by the row hash, so the output is reproducible, and every
record carries the original text so the builder can fail the build if the row it
targets has moved.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from hashlib import sha256
from pathlib import Path
from typing import Any

from training import rewrite_arc_risk_turns as rw

_AI_ROOT = Path(__file__).resolve().parents[1]

DEFAULT_INPUT = (
    _AI_ROOT
    / "data"
    / "curated"
    / "recovery"
    / "gold-20261017"
    / "wandb-full-corpus-current-master-20261109-v17"
    / "candidate.jsonl"
)
DEFAULT_OUTPUT = _AI_ROOT / "training" / "output" / "arc_risk_headers.jsonl"
DEFAULT_LEDGER = DEFAULT_INPUT.parent / "manifest_internal.jsonl"

KIND = "header"
RISK_LABEL = "acute disclosure without a resource"
MIN_TOKEN_LENGTH = 3
MAX_HEADER_WORDS = 25

# Client language that says risk is live right now. Deliberately narrower than
# the ledger's passive band: these rows get an acute-care prefix, not a hotline.
ACUTE_DISCLOSURE = re.compile(
    r"\b("
    r"kill myself|killing myself|killed myself|suicide|suicidal|suicidal thoughts|"
    r"end my life|ending my life|end it all|take my own life|"
    r"want to die|wants to die|wanted to die|don'?t want to (?:live|be here|wake up)|"
    r"cut myself|cutting myself|cuts herself|cuts himself|cut again|cut last night|"
    r"been cutting|started cutting|self[- ]harm|self[- ]harming|"
    r"overdosed|overdose|took (?:a handful of|the whole bottle|too many|a bottle of) pills|"
    r"swallowed (?:a|the|too many) pills|"
    r"hang myself|hanged myself|noose|jump off|jumped off|hurt myself|"
    r"plan to kill|planning to kill|bought a gun|got a gun|"
    r"not going to be here (?:tomorrow|next week)"
    r")\b",
    re.IGNORECASE,
)
ROUTING_TERMS: tuple[str, ...] = rw.ACUTE_TERMS + rw.HOTLINE_TERMS + ("call your", "trusted person")
FORBIDDEN_FILLER: tuple[str, ...] = rw.FORBIDDEN_FILLER

# Every prefix names a resource in its first sentence, none of them asks the
# client a question, and none adds a softener the corpus has to unlearn.
HEADERS: tuple[str, ...] = (
    "Call 988 now.",
    "Call 988 now and tell them what you just told me.",
    "Put 988 in your phone and call it now.",
    "Before anything else, call 988 now.",
    "Call 911 if you are in danger right now; otherwise call 988 now.",
    "Call 988 now, then we keep going on this.",
    "Get a crisis line on the phone right now — 988 in the US.",
    "Call 988 now and put the phone on speaker so you are not doing this alone.",
)


def _log(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def has_routing_term(text: str) -> bool:
    lowered = text.lower()
    return any(term in lowered for term in ROUTING_TERMS)


def header_for(row_sha256: str, msg_index: int) -> str:
    """Deterministic prefix, stable across runs and machines."""
    digest = sha256(f"{row_sha256}#{msg_index}".encode()).hexdigest()
    return HEADERS[int(digest[:8], 16) % len(HEADERS)]


def _words(text: str) -> int:
    return len(text.split())


def validate_header(header: str, original: str) -> str:
    """Fail unless the prefix routes on its own and adds nothing soft."""
    if not header or not original:
        raise ValueError("empty header or original reply")
    if has_routing_term(original):
        raise ValueError("the reply already routes; it is not a header target")
    if not has_routing_term(header):
        raise ValueError(f"header names no resource: {header!r}")
    first_sentence = re.split(r"(?<=[.!?;])\s", header)[0]
    if not has_routing_term(first_sentence):
        raise ValueError(f"header does not route in its first sentence: {header!r}")
    if "?" in header:
        raise ValueError(f"header asks a question: {header!r}")
    if _words(header) > MAX_HEADER_WORDS:
        raise ValueError(f"header exceeds {MAX_HEADER_WORDS} words: {header!r}")
    added = {phrase for phrase in FORBIDDEN_FILLER if phrase in header.lower()}
    added -= {phrase for phrase in FORBIDDEN_FILLER if phrase in original.lower()}
    if added:
        raise ValueError(f"header adds softening filler {sorted(added)}")
    return header


def select_headers(row: dict[str, Any], row_sha256: str) -> list[dict[str, Any]]:
    """One record per acute reply that names no resource."""
    messages = row.get("messages")
    if not isinstance(messages, list) or "tools" in row:
        return []
    records: list[dict[str, Any]] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant" or index == 0:
            continue
        prompt = messages[index - 1]
        if prompt.get("role") != "user":
            continue
        said = prompt.get("content")
        content = message.get("content")
        if not isinstance(said, str) or not isinstance(content, str):
            continue
        if not ACUTE_DISCLOSURE.search(said):
            continue
        _prefix, ledger, spoken = rw.split_content(content)
        if ledger is not None:
            continue  # ledger turns belong to the arc rewrite pass
        body = rw.spoken_body(spoken)
        if not body.strip() or has_routing_term(body):
            continue
        header = header_for(row_sha256, index)
        validate_header(header, body)
        records.append(
            {
                "kind": KIND,
                "generator": "deterministic",
                "row_sha256": row_sha256,
                "msg_index": index,
                "level": None,
                "risk": RISK_LABEL,
                "original_spoken": body,
                "new_spoken": f"{header} {body}",
                "ledger_sha256": "",
            }
        )
    return records


def retained_row_shas(ledger: Path = DEFAULT_LEDGER) -> set[str]:
    """The rows the release actually carries; the held-out reserve is not one."""
    if not ledger.exists():
        raise FileNotFoundError(f"no release ledger at {ledger}")
    shas: set[str] = set()
    with ledger.open() as handle:
        for raw in handle:
            line = raw.strip()
            if line:
                shas.add(json.loads(line)["candidate_row_sha256"])
    return shas


def collect_records(
    candidate: Path = DEFAULT_INPUT, retained: set[str] | None = None
) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with candidate.open("rb") as handle:
        for raw in handle:
            if not raw.strip():
                continue
            row_sha256 = sha256(raw[:-1] if raw.endswith(b"\n") else raw).hexdigest()
            if retained is not None and row_sha256 not in retained:
                continue
            records.extend(select_headers(json.loads(raw), row_sha256))
    return records


def write_records(records: list[dict[str, Any]], output: Path = DEFAULT_OUTPUT) -> dict[str, Any]:
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record) + "\n")
    prefixes = {record["new_spoken"].split(maxsplit=1)[0] for record in records}
    return {
        "rows": len({record["row_sha256"] for record in records}),
        "turns": len(records),
        "distinct_leading_words": len(prefixes),
        "output": str(output),
    }


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--ledger", type=Path, default=DEFAULT_LEDGER)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    records = collect_records(args.input, retained_row_shas(args.ledger))
    if not records:
        raise SystemExit("no acute replies without a resource found")
    if args.dry_run:
        _log(f"would write {len(records)} headers across {len({r['row_sha256'] for r in records})} rows")
        return 0
    summary = write_records(records, args.output)
    _log(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
