"""Rewrite the arc corpus therapist turns that teach "risk on the table, reflect".

The arc rows are the strongest risk-labelled signal in the release: 6,518
therapist turns carry an ``active`` or ``passive`` ledger risk, and only 178 of
them name any routing resource. What the corpus therefore teaches is that when
risk is live the therapist keeps assessing and reflecting — which is exactly the
behaviour every SFT adapter shows at a first-exchange crisis disclosure, and
why the base model's routing (11/11) degrades to 8-11/11 after fine-tuning.

This pass does not regenerate sessions. It rewrites the spoken half of the
turns that carry the load-bearing signal:

* the first turn in a session that puts risk on the table,
* any turn where the ledger risk escalates,
* the closing turn when risk is still live.

The ledger think-block is preserved byte for byte; only the spoken text after
it is replaced. Selection is 1,139 turns across 541 of the 1,085 arc sessions.

Output is a resumable JSONL of rewrite records keyed by source label, so a
part-finished run costs only the calls it has not made yet.
"""

from __future__ import annotations

import argparse
import asyncio
import json
import os
import random
import re
import sys
from dataclasses import dataclass
from hashlib import sha256
from pathlib import Path
from typing import Any

import aiohttp
from dotenv import load_dotenv

_AI_ROOT = Path(__file__).resolve().parents[1]
_REPO_ROOT = Path(__file__).resolve().parents[2]
load_dotenv(_REPO_ROOT / ".env", override=True)

DEFAULT_INPUT = (
    _AI_ROOT
    / "data"
    / "curated"
    / "recovery"
    / "gold-20261017"
    / "wandb-full-corpus-current-master-20261109-v17"
    / "candidate.jsonl"
)
DEFAULT_OUTPUT = _AI_ROOT / "training" / "output" / "arc_risk_rewrites.jsonl"

WRITER_URL = os.environ.get(
    "ARC_WRITER_URL",
    "https://api.cloudflare.com/client/v4/accounts/6c9200f8f880a7f846e70cda63baa1be/ai/v1/chat/completions",
)
WRITER_MODEL = os.environ.get("ARC_WRITER_MODEL", "@cf/deepseek-ai/deepseek-v4-flash-0731")
WRITER_KEY_ENV = os.environ.get("ARC_WRITER_KEYS", "CLOUDFLARE_WORKERS_AI_API")
CALL_TIMEOUT = int(os.environ.get("ARC_REWRITE_TIMEOUT", "180"))
CALL_ATTEMPTS = max(1, int(os.environ.get("ARC_REWRITE_ATTEMPTS", "5")))
CONCURRENCY = max(1, int(os.environ.get("ARC_REWRITE_CONCURRENCY", "3")))
BACKOFF_SECONDS = float(os.environ.get("ARC_REWRITE_BACKOFF", "6"))
TEMPERATURE = float(os.environ.get("ARC_REWRITE_TEMPERATURE", "0.3"))
MAX_TOKENS = int(os.environ.get("ARC_REWRITE_MAX_TOKENS", "400"))
ARC_SOURCE_PREFIX = "arc_corpus_"
SPEAKER_PREFIX = ": "
LEVEL_NONE = 1
LEVEL_PASSIVE = 2
LEVEL_ACTIVE = 3
MAX_SPOKEN_WORDS = 75
MIN_ORIGINAL_TOKEN_RECALL = 0.6
MIN_TOKEN_LENGTH = 3
HTTP_OK = 200
HTTP_TOO_MANY_REQUESTS = 429
CONTEXT_TURNS = 6
_THINKING_KWARG = {"thinking": False}

ACUTE_TERMS: tuple[str, ...] = (
    "911",
    "emergency department",
    "emergency room",
    "emergency services",
    "call an ambulance",
    "ambulance",
    "hospital",
    "poison control",
    "urgent care",
)
HOTLINE_TERMS: tuple[str, ...] = (
    "988",
    "crisis line",
    "crisis hotline",
    "hotline",
    "safety plan",
    "are you safe",
    "emergency",
)
FORBIDDEN_FILLER: tuple[str, ...] = (
    "please",
    "i am glad",
    "i'm glad",
    "thank you for",
    "that's a lot",
    "that is a lot",
    "i hear you",
    "i understand",
    "you are not alone",
    "it makes sense",
    "that must be",
    "that sounds",
    "let's",
    "let us",
    "we can",
    "would you",
    "can you tell me",
    "take your time",
    "i am sorry",
    "i'm sorry",
)

SYSTEM_PROMPT = """You rewrite one line of a therapy transcript so it routes risk.

The transcript is a long-form session. Every therapist turn is a ledger JSON
object followed by a newline and the spoken text. You are given the ledger, the
risk it records, the surrounding turns, and the spoken text to rewrite.

Rewrite rules:
- Minimal edit. The client's next turn answers this one, so keep the original
  sentences, questions and assignments verbatim. Do not drop any question.
- One change: put the routing action in the first sentence.
- Phrase the route in this session's own concrete terms. Do not use a stock
  safety-question opener.
- Direct imperative register. No softeners, no validation openers, no
  reflections, no meta-commentary about the therapy.
- Second person, therapist voice, under 70 words.
- Never invent a fact the client has not stated.
- Return only the rewritten spoken text. No ledger, no quotes, no commentary."""

LEVEL_INSTRUCTIONS = {
    LEVEL_ACTIVE: (
        "Risk is active. The first sentence must name an immediate route to care: "
        "911, an emergency department, an ambulance, poison control, or a hospital."
    ),
    LEVEL_PASSIVE: (
        "Risk is passive ideation. The first sentence must name 988, a crisis line, "
        "or a safety plan the client will actually use."
    ),
}


class RewriteRejectedError(ValueError):
    """A generated line failed the register or routing contract."""


class RateLimitedError(RuntimeError):
    """The writer asked us to slow down; the call should be retried later."""


@dataclass(frozen=True)
class Turn:
    index: int
    level: int
    ledger: dict[str, Any]
    spoken: str
    content: str


@dataclass(frozen=True)
class Selection:
    source: str
    row_sha256: str
    msg_index: int
    level: int
    risk: str
    original_spoken: str
    ledger_sha256: str
    context: str


def _log(message: str) -> None:
    sys.stdout.write(message + "\n")
    sys.stdout.flush()


def risk_level(risk: Any) -> int:
    value = str(risk or "").strip().lower()
    if value.startswith("active"):
        return LEVEL_ACTIVE
    if value.startswith("passive"):
        return LEVEL_PASSIVE
    return LEVEL_NONE


def split_content(content: str) -> tuple[str, dict[str, Any] | None, str]:
    """Return (ledger_prefix, ledger, spoken). Non-ledger turns yield ("", None, content)."""
    if not content.startswith("{"):
        return "", None, content
    closing = content.find("}")
    if closing < 0:
        return "", None, content
    try:
        ledger = json.loads(content[: closing + 1])
    except json.JSONDecodeError:
        return "", None, content
    if not isinstance(ledger, dict):
        return "", None, content
    return content[: closing + 1], ledger, content[closing + 1 :].lstrip("\n")


def spoken_body(spoken: str) -> str:
    """Drop the transcript's leading speaker marker; the rebuild restores it."""
    return spoken[len(SPEAKER_PREFIX) :] if spoken.startswith(SPEAKER_PREFIX) else spoken


def therapist_turns(messages: list[dict[str, Any]]) -> list[Turn]:
    turns: list[Turn] = []
    for index, message in enumerate(messages):
        if message.get("role") != "assistant":
            continue
        content = message.get("content") or ""
        _, ledger, spoken = split_content(content)
        spoken = spoken_body(spoken)
        turns.append(
            Turn(
                index=index,
                level=risk_level((ledger or {}).get("risk")),
                ledger=ledger or {},
                spoken=spoken,
                content=content,
            )
        )
    return turns


def _render_context(messages: list[dict[str, Any]], index: int) -> str:
    start = max(0, index - CONTEXT_TURNS)
    lines: list[str] = []
    for position in range(start, index):
        message = messages[position]
        role = "Client" if message.get("role") == "user" else "Therapist"
        _, _, spoken = split_content(message.get("content") or "")
        lines.append(f"{role}: {spoken_body(spoken).strip()}")
    return "\n".join(lines)


def select_risk_turns(messages: list[dict[str, Any]], row_sha256: str = "") -> list[Selection]:
    """First risk engagement, each escalation, and a closing turn with live risk."""
    turns = therapist_turns(messages)
    if not turns:
        return []
    chosen: list[Turn] = []
    previous = LEVEL_NONE
    for turn in turns:
        if turn.level > LEVEL_NONE and turn.level > previous:
            chosen.append(turn)
        previous = turn.level
    if turns[-1].level > LEVEL_NONE and turns[-1] not in chosen:
        chosen.append(turns[-1])
    return [
        Selection(
            source="",
            row_sha256=row_sha256,
            msg_index=turn.index,
            level=turn.level,
            risk=str(turn.ledger.get("risk", "")),
            original_spoken=turn.spoken,
            ledger_sha256=sha256(json.dumps(turn.ledger, sort_keys=True, ensure_ascii=False).encode()).hexdigest(),
            context=_render_context(messages, turn.index),
        )
        for turn in chosen
    ]


def build_prompt(selection: Selection, session_turns: list[Turn]) -> str:
    ledger = next(t.ledger for t in session_turns if t.index == selection.msg_index)
    parts = [
        f"Session context:\n{selection.context or '(this is the opening turn)'}",
        f"Ledger:\n{json.dumps(ledger, ensure_ascii=False)}",
        f"Risk recorded: {selection.risk}",
        LEVEL_INSTRUCTIONS[selection.level],
        f"Spoken text to rewrite:\n{selection.original_spoken.strip()}",
        "Rewrite the spoken text now.",
    ]
    return "\n\n".join(parts)


_STOPWORDS = frozenset(
    (
        "a",
        "an",
        "and",
        "are",
        "as",
        "at",
        "be",
        "been",
        "but",
        "by",
        "can",
        "could",
        "did",
        "do",
        "does",
        "for",
        "from",
        "had",
        "has",
        "have",
        "he",
        "her",
        "him",
        "his",
        "how",
        "i",
        "if",
        "in",
        "into",
        "is",
        "it",
        "its",
        "me",
        "my",
        "no",
        "not",
        "of",
        "on",
        "or",
        "our",
        "she",
        "so",
        "some",
        "than",
        "that",
        "the",
        "their",
        "them",
        "then",
        "there",
        "these",
        "they",
        "this",
        "to",
        "up",
        "was",
        "we",
        "were",
        "what",
        "when",
        "where",
        "which",
        "who",
        "why",
        "will",
        "with",
        "would",
        "you",
        "your",
    )
)


def _tokens(text: str) -> set[str]:
    return {
        word
        for word in re.findall(r"[a-z']+", text.lower())
        if word not in _STOPWORDS and len(word) >= MIN_TOKEN_LENGTH
    }


def _filler_hits(text: str) -> set[str]:
    lowered = text.lower()
    return {phrase for phrase in FORBIDDEN_FILLER if phrase in lowered}


def validate_rewrite(selection: Selection, spoken: str) -> str:
    text = spoken.strip().strip('"').strip()
    if not text:
        raise RewriteRejectedError("empty rewrite")
    lowered = text.lower()

    # Minimal edit: the rewrite may keep the original's own wording, but it may
    # not introduce softening of its own.
    added_filler = _filler_hits(text) - _filler_hits(selection.original_spoken)
    if added_filler:
        raise RewriteRejectedError(f"softening filler {sorted(added_filler)}")

    words = text.split()
    if len(words) > MAX_SPOKEN_WORDS:
        raise RewriteRejectedError(f"{len(words)} words exceeds {MAX_SPOKEN_WORDS}")

    terms = ACUTE_TERMS if selection.level == LEVEL_ACTIVE else HOTLINE_TERMS
    if not any(term in lowered for term in terms):
        raise RewriteRejectedError(f"no routing term from {terms[:3]}...")
    first_sentence = re.split(r"(?<=[.!?])\s", text)[0]
    if not any(term in first_sentence.lower() for term in terms):
        raise RewriteRejectedError("routing term is not in the first sentence")

    # The client's next turn answers this one, so the questions have to survive.
    if text.count("?") < selection.original_spoken.count("?"):
        raise RewriteRejectedError("dropped a question the client's next turn answers")

    original_tokens = _tokens(selection.original_spoken)
    if original_tokens:
        recall = len(original_tokens & _tokens(text)) / len(original_tokens)
        if recall < MIN_ORIGINAL_TOKEN_RECALL:
            raise RewriteRejectedError(f"rewrote the turn instead of routing it ({recall:.0%} kept)")

    if lowered.startswith(("i ", "we ")):
        raise RewriteRejectedError("opens with the therapist rather than the action")
    return text


def rewrite_key(record: dict[str, Any]) -> str:
    return f"{record['row_sha256']}#{record['msg_index']}"


def load_done(path: Path) -> dict[str, dict[str, Any]]:
    done: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return done
    with path.open() as handle:
        for raw in handle:
            stripped = raw.strip()
            if not stripped:
                continue
            record = json.loads(stripped)
            done[rewrite_key(record)] = record
    return done


def collect_selections(candidate: Path) -> list[tuple[Selection, list[Turn]]]:
    out: list[tuple[Selection, list[Turn]]] = []
    with candidate.open("rb") as handle:
        for raw in handle:
            row_sha256 = sha256(raw[:-1] if raw.endswith(b"\n") else raw).hexdigest()
            row = json.loads(raw)
            source = row.get("source") or ""
            if not str(source).startswith(ARC_SOURCE_PREFIX):
                continue
            messages = row.get("messages") or []
            turns = therapist_turns(messages)
            for selection in select_risk_turns(messages, row_sha256):
                out.append(
                    (
                        Selection(
                            source=source,
                            row_sha256=row_sha256,
                            msg_index=selection.msg_index,
                            level=selection.level,
                            risk=selection.risk,
                            original_spoken=selection.original_spoken,
                            ledger_sha256=selection.ledger_sha256,
                            context=selection.context,
                        ),
                        turns,
                    )
                )
    return out


async def call_model(session: aiohttp.ClientSession, api_key: str, prompt: str) -> str:
    payload = {
        "model": WRITER_MODEL,
        "messages": [
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": prompt},
        ],
        "max_tokens": MAX_TOKENS,
        "temperature": TEMPERATURE,
        "chat_template_kwargs": _THINKING_KWARG,
    }
    headers = {"Authorization": f"Bearer {api_key}", "Content-Type": "application/json"}
    timeout = aiohttp.ClientTimeout(total=CALL_TIMEOUT)
    async with session.post(WRITER_URL, json=payload, headers=headers, timeout=timeout) as response:
        if response.status == HTTP_TOO_MANY_REQUESTS:
            raise RateLimitedError("http_429: writer is throttling")
        if response.status != HTTP_OK:
            body = await response.text()
            raise RuntimeError(f"http_{response.status}: {body[:200]}")
        data = json.loads(await response.text())
    choices = data.get("choices") or []
    if not choices:
        raise RuntimeError("no choices in response")
    content = (choices[0].get("message") or {}).get("content") or ""
    if not content.strip():
        raise RuntimeError("empty content")
    return content


async def rewrite_one(
    session: aiohttp.ClientSession,
    api_key: str,
    selection: Selection,
    turns: list[Turn],
) -> tuple[str | None, str | None]:
    prompt = build_prompt(selection, turns)
    last_error: str | None = None
    for attempt in range(1, CALL_ATTEMPTS + 1):
        retry_after = 0.0
        try:
            raw = await call_model(session, api_key, prompt)
            return validate_rewrite(selection, raw), None
        except RewriteRejectedError as error:
            last_error = f"rejected: {error}"
            prompt = build_prompt(selection, turns) + f"\n\nThat attempt was rejected: {error}. Follow every rule."
        except RateLimitedError as error:
            last_error = f"transport: {error}"
            retry_after = BACKOFF_SECONDS * (2 ** (attempt - 1))
        except (TimeoutError, aiohttp.ClientError, RuntimeError) as error:
            last_error = f"transport: {type(error).__name__}: {error}"
            retry_after = BACKOFF_SECONDS
        if retry_after and attempt < CALL_ATTEMPTS:
            await asyncio.sleep(retry_after * (1 + random.random()))
    return None, last_error


async def run(candidate: Path, output: Path, limit: int | None, dry_run: bool) -> int:
    selections = collect_selections(candidate)
    if not selections:
        raise SystemExit("no arc risk turns selected")
    counts: dict[str, int] = {}
    for selection, _ in selections:
        label = "active" if selection.level == LEVEL_ACTIVE else "passive"
        counts[label] = counts.get(label, 0) + 1
    sessions = len({selection.row_sha256 for selection, _ in selections})
    _log(
        f"selected {len(selections)} turns across {sessions} sessions "
        f"(active={counts.get('active', 0)}, passive={counts.get('passive', 0)})"
    )
    if dry_run:
        for selection, _ in selections[:5]:
            _log(f"--- {selection.source} msg {selection.msg_index} | {selection.risk[:80]}")
            _log(f"    original: {selection.original_spoken[:160]}")
        return 0

    done = load_done(output)
    pending = [
        (selection, turns)
        for selection, turns in selections
        if f"{selection.row_sha256}#{selection.msg_index}" not in done
    ]
    if limit is not None:
        pending = pending[:limit]
    _log(f"{len(done)} already rewritten, {len(pending)} to go")
    if not pending:
        return 0

    api_key = os.environ.get(WRITER_KEY_ENV, "").strip()
    if not api_key:
        raise SystemExit(f"missing {WRITER_KEY_ENV}")
    semaphore = asyncio.Semaphore(CONCURRENCY)
    failures = 0
    written = 0

    async def handle(selection: Selection, turns: list[Turn]) -> None:
        nonlocal failures, written
        async with semaphore:
            spoken, error = await rewrite_one(session, api_key, selection, turns)
        record = {
            "source": selection.source,
            "row_sha256": selection.row_sha256,
            "msg_index": selection.msg_index,
            "level": "active" if selection.level == LEVEL_ACTIVE else "passive",
            "risk": selection.risk,
            "ledger_sha256": selection.ledger_sha256,
            "original_spoken": selection.original_spoken,
            "new_spoken": spoken,
            "error": error,
            "model": WRITER_MODEL,
        }
        if spoken is None:
            failures += 1
            _log(f"FAILED {selection.source} msg {selection.msg_index}: {error}")
            return
        with output.open("a") as handle_out:
            handle_out.write(json.dumps(record, ensure_ascii=False) + "\n")
        written += 1
        if written % 50 == 0:
            _log(f"  rewritten {written}/{len(pending)}")

    output.parent.mkdir(parents=True, exist_ok=True)
    async with aiohttp.ClientSession() as session:
        await asyncio.gather(*(handle(selection, turns) for selection, turns in pending))
    _log(f"done: {written} rewritten, {failures} failed")
    return 1 if failures else 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    parser.add_argument("--limit", type=int, default=None)
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    if not args.input.exists():
        parser.error(f"candidate not found: {args.input}")
    return asyncio.run(run(args.input, args.output, args.limit, args.dry_run))


if __name__ == "__main__":
    sys.exit(main())
