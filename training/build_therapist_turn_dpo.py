#!/usr/bin/env python3
"""Build exact-prefix therapist-turn DPO pairs from on-policy sampling.

The arc corpus cannot supply therapist-turn preference pairs directly: every
rejection source on disk (``failed_attempts/*.txt`` and the ``.pre_*`` snapshot
directories) is a *whole-session regeneration*, so the client turns diverge from
the first turn and there is no shared prefix to condition on. Measured on the
2026-11-04 corpus: 0 of 1,189 failed attempts and 0 of 1,085 snapshot revisions
share a client-turn prefix with the accepted session.

This builder creates the pairs instead of mining them. For each accepted client
prefix it samples several therapist replies from a served adapter, scores them
with the deterministic therapist-voice gate in :mod:`ai.training.cliche_gate`
(plus length, echo, and interrogation checks), and emits a pair only when at
least one sample passes and one fails. Both sides are conditioned on the byte
-identical prompt, which is what DPO requires, and the negatives come from the
model's own distribution rather than from a hand-written template.

Output rows are ``{prompt, chosen, rejected, metadata}`` with ``prompt`` being a
list of chat messages, ready for :mod:`ai.training.dpo_trainer`.

Usage::

    PYTHONPATH=/home/vivi/pixelated .venv/bin/python \
        ai/training/build_therapist_turn_dpo.py --dry-run

    PYTHONPATH=/home/vivi/pixelated .venv/bin/python \
        ai/training/build_therapist_turn_dpo.py --max-prefixes 400
"""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import re
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    from ai.training.cliche_gate import is_sycophantic
    from ai.training.wandb_inference import Sampler, api_key
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ai.training.cliche_gate import is_sycophantic
    from ai.training.wandb_inference import Sampler, api_key

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "ai" / "training" / "output" / "arc_corpus"
RECORDS_PATH = CORPUS / "arc_records.jsonl"
RELEASE_DIR = (
    ROOT / "ai" / "data" / "curated" / "recovery" / "gold-20261017" / "wandb-full-corpus-current-master-20261027-v13"
)
OUT_DIR = CORPUS
DEFAULT_MODEL = "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline-ecf35d5e5384:v1"
EXCLUSION_CACHE = CORPUS / "therapist_turn_dpo_excluded_arcs.json"

# Thresholds sit at the corpus's own outlier tail, measured over the 17,853
# accepted therapist replies (sentences: mean 3.26, p90 5, p95 5, max 12;
# chars: mean 161, p90 258, p95 299, max 829). A tighter cap does not detect bad
# therapy, it just labels the corpus's normal register as failure and trains the
# model to be terser than its own gold data.
MAX_SENTENCES = 6
MAX_CHARS = 600
MAX_QUESTIONS = 1
ECHO_NGRAM = 8
EXCLUDED_SPLITS = ("validation", "test", "heldout")

# Position buckets split a session's therapist turns into open / middle / close
# so sampling covers the whole arc instead of clustering at its opening.
BUCKET_OPEN_MAX = 0.34
BUCKET_MIDDLE_MAX = 0.67
BUCKETS = ("open", "middle", "close")

# Deterministic arc-level split: sha256(arc_id) % 100 < 90 -> train.
SPLIT_MODULUS = 100
TRAIN_SPLIT_PERCENT = 90


# --------------------------------------------------------------------------- #
# corpus loading
# --------------------------------------------------------------------------- #
def _norm(text: str) -> str:
    return " ".join(text.split()).strip()


def load_sessions(path: Path) -> dict[tuple[str, int], list[dict[str, Any]]]:
    """Return ``{(arc_id, session_n): turns}`` from an arc-records jsonl."""
    sessions: dict[tuple[str, int], list[dict[str, Any]]] = {}
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            record = json.loads(line)
            for session in record.get("sessions", []):
                sessions[(record["arc_id"], int(session["n"]))] = session["turns"]
    return sessions


def arc_split_membership(release_dir: Path, cache: Path) -> dict[str, set[str]]:
    """Map arc id -> splits it appears in, joined through ``rowrefs.jsonl``.

    ``candidate.jsonl`` is large, so the join result is cached on disk.
    """
    if cache.exists():
        raw = json.loads(cache.read_text(encoding="utf-8"))
        return {arc: set(splits) for arc, splits in raw.items()}

    row_splits: dict[int, str] = {}
    rowrefs = release_dir / "rowrefs.jsonl"
    with rowrefs.open(encoding="utf-8") as handle:
        for line in handle:
            if not line.strip():
                continue
            ref = json.loads(line)
            row_splits[int(ref["candidate_row_number_1based"])] = str(ref.get("provisional_split", ""))

    membership: dict[str, set[str]] = {}
    candidate = release_dir / "candidate.jsonl"
    with candidate.open(encoding="utf-8") as handle:
        for row_number, line in enumerate(handle, start=1):
            if "arc_corpus_" not in line:
                continue
            source = json.loads(line).get("source")
            if not isinstance(source, str) or not source.startswith("arc_corpus_"):
                continue
            arc_id = source[len("arc_corpus_") :]
            split = row_splits.get(row_number)
            if split:
                membership.setdefault(arc_id, set()).add(split)

    cache.write_text(
        json.dumps({arc: sorted(s) for arc, s in membership.items()}, indent=0, sort_keys=True),
        encoding="utf-8",
    )
    return membership


# --------------------------------------------------------------------------- #
# prefix enumeration
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Prefix:
    arc_id: str
    session_n: int
    turn_ordinal: int
    bucket: str
    messages: list[dict[str, str]]
    client_text: str


def therapist_content(turn: dict[str, Any]) -> str | None:
    """Serialise a therapist turn exactly as the training corpus does.

    ``consolidate_arc_corpus.py`` writes the assistant content as
    ``<one-line ledger JSON>\\n<spoken reply>``, so a prompt that carries only
    the spoken reply is off-distribution: the sampler sees assistant turns that
    never look like the turns it was trained to produce. Returns ``None`` when
    the turn has no usable ledger, which marks the whole session as unusable for
    prefix building.
    """
    spoken = turn.get("content")
    ledger = turn.get("ledger")
    if not isinstance(spoken, str) or not spoken.strip():
        return None
    if not isinstance(ledger, dict) or not ledger:
        return None
    return f"{json.dumps(ledger, ensure_ascii=False)}\n{spoken}"


def split_therapist_turn(text: str) -> tuple[dict[str, Any], str] | None:
    """Split a sampled assistant turn into ``(ledger, spoken)``.

    Returns ``None`` when the sample does not carry a parseable leading ledger
    object plus a non-empty spoken reply. Those samples are generation
    breakdowns, not therapist-voice failures, so they must not become negatives.
    """
    stripped = text.lstrip()
    if not stripped.startswith("{"):
        return None
    try:
        ledger, end = json.JSONDecoder().raw_decode(stripped)
    except ValueError:
        return None
    if not isinstance(ledger, dict) or not ledger:
        return None
    spoken = stripped[end:].strip()
    if not spoken:
        return None
    return ledger, spoken


def _alternates(turns: list[dict[str, Any]]) -> bool:
    expected = "client"
    for turn in turns:
        if turn.get("role") != expected:
            return False
        expected = "therapist" if expected == "client" else "client"
    return True


def _bucket_for(share: float) -> str:
    if share < BUCKET_OPEN_MAX:
        return "open"
    return "middle" if share < BUCKET_MIDDLE_MAX else "close"


def _prefix_messages(head: list[dict[str, Any]]) -> list[dict[str, str]] | None:
    """Convert the accepted turns up to the reply into chat messages."""
    if not head or head[-1].get("role") != "client":
        return None
    messages: list[dict[str, str]] = []
    for turn in head:
        if turn["role"] == "client":
            messages.append({"role": "user", "content": turn["content"]})
            continue
        content = therapist_content(turn)
        if content is None:
            return None
        messages.append({"role": "assistant", "content": content})
    if not messages or messages[0]["role"] != "user":
        return None
    return messages


def enumerate_prefixes(
    sessions: dict[tuple[str, int], list[dict[str, Any]]],
    excluded_arcs: set[str],
    rng: random.Random,
) -> list[Prefix]:
    """One prefix per session per position bucket (open / middle / close)."""
    prefixes: list[Prefix] = []
    for (arc_id, session_n), turns in sorted(sessions.items()):
        if arc_id in excluded_arcs:
            continue
        if not turns or not _alternates(turns):
            continue
        positions = [index for index, turn in enumerate(turns) if turn.get("role") == "therapist" and index >= 1]
        if not positions:
            continue
        total = len(positions)
        buckets: dict[str, list[int]] = {bucket: [] for bucket in BUCKETS}
        for ordinal, index in enumerate(positions):
            share = ordinal / max(total - 1, 1)
            buckets[_bucket_for(share)].append(index)

        for bucket, indices in buckets.items():
            if not indices:
                continue
            index = rng.choice(indices)
            messages = _prefix_messages(turns[:index])
            if messages is None:
                continue
            prefixes.append(
                Prefix(
                    arc_id=arc_id,
                    session_n=session_n,
                    turn_ordinal=index,
                    bucket=bucket,
                    messages=messages,
                    client_text=turns[index - 1]["content"],
                )
            )
    return prefixes


# --------------------------------------------------------------------------- #
# deterministic scoring
# --------------------------------------------------------------------------- #
_SENTENCE_SPLIT = re.compile(r"(?<=[.!?])\s+")
_WORD_RE = re.compile(r"[a-z0-9']+")


def _words(text: str) -> list[str]:
    return _WORD_RE.findall(text.lower())


def _echoes_client(reply: str, client_text: str) -> bool:
    """True when the reply reproduces a long verbatim span of the client turn."""
    reply_words = _words(reply)
    client_words = _words(client_text)
    if len(reply_words) < ECHO_NGRAM or len(client_words) < ECHO_NGRAM:
        return False
    client_ngrams = {" ".join(client_words[i : i + ECHO_NGRAM]) for i in range(len(client_words) - ECHO_NGRAM + 1)}
    return any(
        " ".join(reply_words[i : i + ECHO_NGRAM]) in client_ngrams for i in range(len(reply_words) - ECHO_NGRAM + 1)
    )


def score_reply(reply: str | None, client_text: str) -> list[str]:
    """Return failure tags for one sampled therapist reply; empty means pass."""
    if not reply or not reply.strip():
        return ["empty"]

    text = reply.strip()
    failures: list[str] = []

    sycophantic, reason = is_sycophantic(text)
    if sycophantic:
        failures.append(reason or "sycophantic")

    sentences = [part for part in _SENTENCE_SPLIT.split(text) if part.strip()]
    if len(sentences) > MAX_SENTENCES:
        failures.append(f"too_long_sentences:{len(sentences)}")
    if len(text) > MAX_CHARS:
        failures.append(f"too_long_chars:{len(text)}")
    if text.count("?") > MAX_QUESTIONS:
        failures.append(f"interrogation_stack:{text.count('?')}")
    if _echoes_client(text, client_text):
        failures.append("client_echo")

    return failures


# --------------------------------------------------------------------------- #
# sampling
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# pair assembly
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class Stats:
    prefixes: int = 0
    sampled_ok: int = 0
    sampled_failed: int = 0
    replies: int = 0
    passing_replies: int = 0
    pairs: int = 0
    dropped_no_pass: int = 0
    dropped_no_fail: int = 0
    dropped_duplicate: int = 0
    dropped_arc_cap: int = 0
    dropped_unparseable: int = 0
    dropped_no_parse: int = 0
    defect_counts: dict[str, int] = field(default_factory=dict)


@dataclass(slots=True)
class SamplingRun:
    """Provenance recorded on every emitted pair."""

    model: str
    temperature: float
    samples: int
    seed: int
    max_pairs_per_arc: int


def build_pairs(
    prefixes: list[Prefix],
    replies_by_prefix: dict[int, list[str | None]],
    *,
    run: SamplingRun,
    stats: Stats,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    seen: set[str] = set()
    per_arc: dict[str, int] = {}

    for index, prefix in enumerate(prefixes):
        raw_replies = [reply for reply in replies_by_prefix.get(index, []) if reply]
        if not raw_replies:
            continue
        stats.replies += len(raw_replies)

        # Score the spoken reply, not the ledger prefix: the ledger is a fixed
        # nine-field schema the model emits before every reply, so including it
        # would make every sample look long and bury the voice signal.
        parsed: list[tuple[str, str, list[str]]] = []
        for raw in raw_replies:
            split = split_therapist_turn(raw)
            if split is None:
                stats.dropped_unparseable += 1
                continue
            ledger, spoken = split
            content = f"{json.dumps(ledger, ensure_ascii=False)}\n{spoken}"
            parsed.append((content, spoken, score_reply(spoken, prefix.client_text)))
        if not parsed:
            stats.dropped_no_parse += 1
            continue

        passing = [item for item in parsed if not item[2]]
        failing = [item for item in parsed if item[2]]
        stats.passing_replies += len(passing)

        if not passing:
            stats.dropped_no_pass += 1
            continue
        if not failing:
            stats.dropped_no_fail += 1
            continue
        if per_arc.get(prefix.arc_id, 0) >= run.max_pairs_per_arc:
            stats.dropped_arc_cap += 1
            continue

        # Prefer the shortest passing reply (most direct voice) against the
        # reply that fails on the most independent criteria.
        chosen = min(passing, key=lambda item: (len(item[1]), item[0]))[0]
        rejected, _, rejected_tags = max(
            failing,
            key=lambda item: (len(item[2]), len(item[1]), item[0]),
        )

        prompt_json = json.dumps(prefix.messages, ensure_ascii=False, sort_keys=True)
        dedupe_key = hashlib.sha256(f"{prompt_json}\x00{chosen}\x00{rejected}".encode()).hexdigest()
        if dedupe_key in seen:
            stats.dropped_duplicate += 1
            continue
        seen.add(dedupe_key)

        for tag in rejected_tags:
            bucket = tag.split(":", 1)[0]
            stats.defect_counts[bucket] = stats.defect_counts.get(bucket, 0) + 1

        split_bucket = int(hashlib.sha256(prefix.arc_id.encode()).hexdigest(), 16) % SPLIT_MODULUS
        split_name = "train" if split_bucket < TRAIN_SPLIT_PERCENT else "validation"
        rows.append(
            {
                "prompt": prefix.messages,
                "chosen": chosen,
                "rejected": rejected,
                "metadata": {
                    "arc_id": prefix.arc_id,
                    "session_n": prefix.session_n,
                    "turn_ordinal": prefix.turn_ordinal,
                    "position_bucket": prefix.bucket,
                    "prefix_turns": len(prefix.messages),
                    "rejected_defects": rejected_tags,
                    "sampler_model": run.model,
                    "sampler_temperature": run.temperature,
                    "samples_per_prefix": run.samples,
                    "sampler_seed": run.seed,
                    "split": split_name,
                    "pair_hash": dedupe_key,
                },
            }
        )
        per_arc[prefix.arc_id] = per_arc.get(prefix.arc_id, 0) + 1

    stats.pairs = len(rows)
    return rows


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RECORDS_PATH)
    parser.add_argument("--release-dir", type=Path, default=RELEASE_DIR)
    parser.add_argument("--out", type=Path, default=OUT_DIR / "therapist_turn_dpo_pairs.jsonl")
    parser.add_argument("--summary-out", type=Path, default=OUT_DIR / "therapist_turn_dpo_summary.json")
    parser.add_argument("--exclusion-cache", type=Path, default=EXCLUSION_CACHE)
    parser.add_argument("--model", default=DEFAULT_MODEL)
    parser.add_argument("--samples", type=int, default=4)
    parser.add_argument("--temperature", type=float, default=0.9)
    parser.add_argument("--max-prefixes", type=int, default=600)
    parser.add_argument("--max-pairs-per-arc", type=int, default=2)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261104)
    parser.add_argument("--dry-run", action="store_true", help="enumerate prefixes only, no sampling")
    return parser.parse_args(argv)


@dataclass(slots=True)
class Selection:
    """Prefixes chosen for sampling, plus the funnel that produced them."""

    prefixes: list[Prefix]
    bucket_counts: dict[str, int]
    candidates: int
    excluded_arcs: int
    distinct_arcs: int


def _select_prefixes(args: argparse.Namespace, rng: random.Random) -> Selection:
    sessions = load_sessions(args.records)
    membership = arc_split_membership(args.release_dir, args.exclusion_cache)
    excluded = {arc for arc, splits in membership.items() if set(splits) & set(EXCLUDED_SPLITS)}
    print(f"sessions: {len(sessions)} | arcs in release: {len(membership)} | excluded arcs: {len(excluded)}")

    prefixes = enumerate_prefixes(sessions, excluded, rng)
    candidates = len(prefixes)
    print(f"prefix candidates: {candidates}")
    if args.max_prefixes and len(prefixes) > args.max_prefixes:
        prefixes = rng.sample(prefixes, args.max_prefixes)
        prefixes.sort(key=lambda p: (p.arc_id, p.session_n, p.turn_ordinal))
        print(f"sampled prefixes: {len(prefixes)}")

    bucket_counts: dict[str, int] = {}
    for prefix in prefixes:
        bucket_counts[prefix.bucket] = bucket_counts.get(prefix.bucket, 0) + 1
    distinct_arcs = len({prefix.arc_id for prefix in prefixes})
    print(f"bucket mix: {bucket_counts}")
    print(f"distinct arcs: {distinct_arcs}")
    return Selection(prefixes, bucket_counts, candidates, len(excluded), distinct_arcs)


def _sample_all(
    prefixes: list[Prefix],
    sampler: Sampler,
    samples: int,
    concurrency: int,
    stats: Stats,
) -> dict[int, list[str | None]]:
    replies_by_prefix: dict[int, list[str | None]] = {}
    started = time.time()
    with ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = {
            pool.submit(sampler.sample, prefix.messages, samples): index for index, prefix in enumerate(prefixes)
        }
        total = len(futures)
        for done, future in enumerate(as_completed(futures), start=1):
            replies = future.result()
            replies_by_prefix[futures[future]] = replies
            if replies:
                stats.sampled_ok += 1
            else:
                stats.sampled_failed += 1
            if done % 25 == 0 or done == total:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0.0
                print(f"  {done}/{total} prefixes ({rate:.2f}/s)")
    return replies_by_prefix


def _write_outputs(
    rows: list[dict[str, Any]],
    args: argparse.Namespace,
    selection: Selection,
    stats: Stats,
    run: SamplingRun,
) -> dict[str, int]:
    args.out.parent.mkdir(parents=True, exist_ok=True)
    with args.out.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")

    split_counts: dict[str, int] = {}
    for row in rows:
        split = row["metadata"]["split"]
        split_counts[split] = split_counts.get(split, 0) + 1

    summary = {
        "built_at": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "records_path": str(args.records),
        "release_dir": str(args.release_dir),
        "out_path": str(args.out),
        "model": run.model,
        "samples_per_prefix": run.samples,
        "temperature": run.temperature,
        "seed": run.seed,
        "max_prefixes": args.max_prefixes,
        "max_pairs_per_arc": run.max_pairs_per_arc,
        "excluded_arcs": selection.excluded_arcs,
        "prefix_candidates": selection.candidates,
        "bucket_mix": selection.bucket_counts,
        "distinct_arcs": selection.distinct_arcs,
        "stats": {
            "prefixes": stats.prefixes,
            "sampled_ok": stats.sampled_ok,
            "sampled_failed": stats.sampled_failed,
            "replies": stats.replies,
            "passing_replies": stats.passing_replies,
            "pairs": stats.pairs,
            "dropped_no_pass": stats.dropped_no_pass,
            "dropped_no_fail": stats.dropped_no_fail,
            "dropped_duplicate": stats.dropped_duplicate,
            "dropped_arc_cap": stats.dropped_arc_cap,
            "dropped_unparseable": stats.dropped_unparseable,
            "dropped_no_parse": stats.dropped_no_parse,
            "rejected_defect_counts": dict(sorted(stats.defect_counts.items())),
        },
        "split_counts": split_counts,
        "scoring": {
            "max_sentences": MAX_SENTENCES,
            "max_chars": MAX_CHARS,
            "max_questions": MAX_QUESTIONS,
            "echo_ngram": ECHO_NGRAM,
            "gate": "ai.training.cliche_gate.is_sycophantic",
        },
    }
    args.summary_out.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    return split_counts


def main(argv: list[str] | None = None) -> None:
    args = _parse_args(argv)
    rng = random.Random(args.seed)
    selection = _select_prefixes(args, rng)

    if args.dry_run:
        longest = max(len(prefix.messages) for prefix in selection.prefixes)
        print(f"longest prefix: {longest} messages")
        print("dry run: no sampling performed")
        return

    sampler = Sampler(model=args.model, api_key=api_key(), temperature=args.temperature)
    run = SamplingRun(
        model=args.model,
        temperature=args.temperature,
        samples=args.samples,
        seed=args.seed,
        max_pairs_per_arc=args.max_pairs_per_arc,
    )
    stats = Stats(prefixes=len(selection.prefixes))
    replies_by_prefix = _sample_all(selection.prefixes, sampler, args.samples, args.concurrency, stats)
    rows = build_pairs(selection.prefixes, replies_by_prefix, run=run, stats=stats)
    split_counts = _write_outputs(rows, args, selection, stats, run)

    print(f"\npairs: {stats.pairs} -> {args.out}")
    print(f"splits: {split_counts}")
    print(f"defects: {dict(sorted(stats.defect_counts.items()))}")
    print(f"summary -> {args.summary_out}")


if __name__ == "__main__":
    main()
