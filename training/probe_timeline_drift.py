#!/usr/bin/env python3
"""Measure long-context timeline (``tl``) drift on a served adapter.

The arc corpus carries one structural check on long-range temporal coherence:
the ledger's ``tl`` field is append-only, so a therapist turn that stops
carrying an earlier time anchor (``now``, ``-6w``, ``+1y`` ...) fails the
session. That gate is the second most common generation failure on record
(200 of 1,195), and it only fires after a whole session has been written.

This probe measures the same invariant on a single continuation. Each prefix is
an accepted session truncated at a client turn, with the ledgers left exactly
as the training corpus serialises them (``<ledger JSON>\\n<spoken reply>``), so
the model sees the distribution it was trained on. The sampled next turn is
parsed, its ``tl`` anchors compared with the anchors inherited from the prefix,
and any inherited anchor it drops counts as drift.

Prefixes are stratified by context length (short / medium / long / very long)
because the interesting question is not whether drift happens but whether it
grows with the amount of timeline the model has to keep alive. Arcs that appear
in the v13 validation, test, or heldout splits are excluded so the measurement
is on arcs the adapter never trained on.

Usage::

    PYTHONPATH=/home/vivi/pixelated ai/.venv/bin/python \
        ai/training/probe_timeline_drift.py --dry-run

    PYTHONPATH=/home/vivi/pixelated ai/.venv/bin/python \
        ai/training/probe_timeline_drift.py --arm both --per-bucket 12
"""

from __future__ import annotations

import argparse
import json
import random
import sys
import time
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any

try:
    from ai.training.build_therapist_turn_dpo import (
        EXCLUDED_SPLITS,
        arc_split_membership,
        load_sessions,
        therapist_content,
    )
    from ai.training.timeline_gate import accumulate, check_drift, ledger_anchors, parse_ledger
    from ai.training.wandb_inference import BASE_MODEL, SFT_MODEL, Sampler, api_key
except ModuleNotFoundError:  # direct script execution
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ai.training.build_therapist_turn_dpo import (
        EXCLUDED_SPLITS,
        arc_split_membership,
        load_sessions,
        therapist_content,
    )
    from ai.training.timeline_gate import accumulate, check_drift, ledger_anchors, parse_ledger
    from ai.training.wandb_inference import BASE_MODEL, SFT_MODEL, Sampler, api_key

ROOT = Path(__file__).resolve().parents[2]
CORPUS = ROOT / "ai" / "training" / "output" / "arc_corpus"
RECORDS_PATH = CORPUS / "arc_records.jsonl"
RELEASE_DIR = (
    ROOT / "ai" / "data" / "curated" / "recovery" / "gold-20261017" / "wandb-full-corpus-current-master-20261027-v13"
)
EXCLUSION_CACHE = CORPUS / "therapist_turn_dpo_excluded_arcs.json"
OUT_ROOT = ROOT / "ai" / "training" / "output" / "timeline_drift_probe"

# Context-length strata, in prefix messages. The corpus runs 14-74 turns per
# session, so these cut the observed range into four bands instead of inventing
# lengths the writer never produced.
BUCKETS = ("short", "medium", "long", "very_long")
BUCKET_EDGES = ((10, "short"), (20, "medium"), (32, "long"))
ARMS = ("base", "sft")


def bucket_for(message_count: int) -> str:
    """Context-length stratum for a prefix of ``message_count`` messages."""
    for edge, name in BUCKET_EDGES:
        if message_count < edge:
            return name
    return BUCKETS[-1]


# --------------------------------------------------------------------------- #
# prefix construction
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class TimelinePrefix:
    """An accepted session truncated at a client turn."""

    arc_id: str
    session_n: int
    turn_ordinal: int
    bucket: str
    messages: list[dict[str, str]]
    inherited: frozenset[str]
    reference: frozenset[str]

    @property
    def reference_holds(self) -> bool:
        """True when the accepted next turn kept every inherited anchor."""
        # bool(): under the dependency-light ratchet mypy the timeline_gate
        # import root is unresolved (Any); the wrapper keeps the declared
        # return type honest in every environment.
        return bool(check_drift(self.inherited, self.reference).holds)


def build_messages(head: list[dict[str, Any]]) -> list[dict[str, str]] | None:
    """Chat messages for an accepted turn prefix, or ``None`` if unusable."""
    messages: list[dict[str, str]] = []
    for turn in head:
        if turn.get("role") == "client":
            messages.append({"role": "user", "content": str(turn.get("content", ""))})
            continue
        content = therapist_content(turn)
        if content is None:
            return None
        messages.append({"role": "assistant", "content": content})
    if not messages or messages[0]["role"] != "user":
        return None
    return messages


def _session_prefixes(arc_id: str, session_n: int, turns: list[dict[str, Any]]) -> list[TimelinePrefix]:
    """Deepest usable prefix per context bucket, for one session."""
    deepest: dict[str, tuple[int, list[dict[str, str]]]] = {}
    for index in range(2, len(turns), 2):
        if turns[index].get("role") != "client":
            break
        messages = build_messages(turns[:index])
        if messages is None:
            break
        # Keep the deepest prefix in each band: that is where the invariant is
        # under the most pressure for the same bucket label.
        deepest[bucket_for(len(messages))] = (index, messages)

    prefixes: list[TimelinePrefix] = []
    for bucket, (index, messages) in deepest.items():
        if index + 1 >= len(turns) or turns[index + 1].get("role") != "therapist":
            continue
        head = turns[:index]
        inherited = accumulate(t["ledger"] for t in head if t.get("role") == "therapist" and t.get("ledger"))
        reference = ledger_anchors(turns[index + 1].get("ledger") or {})
        prefixes.append(
            TimelinePrefix(
                arc_id=arc_id,
                session_n=session_n,
                turn_ordinal=index,
                bucket=bucket,
                messages=messages,
                inherited=frozenset(inherited),
                reference=frozenset(reference),
            )
        )
    return prefixes


def enumerate_prefixes(
    sessions: dict[tuple[str, int], list[dict[str, Any]]],
    excluded: set[str],
) -> dict[str, list[TimelinePrefix]]:
    """All usable prefixes grouped by context bucket."""
    grouped: dict[str, list[TimelinePrefix]] = {bucket: [] for bucket in BUCKETS}
    for (arc_id, session_n), turns in sorted(sessions.items()):
        if arc_id in excluded or not turns:
            continue
        for prefix in _session_prefixes(arc_id, session_n, turns):
            if not prefix.inherited:
                # Nothing to preserve: the turn cannot drift, so it measures
                # nothing and would dilute the rate.
                continue
            grouped[prefix.bucket].append(prefix)
    return grouped


def _cap(grouped: dict[str, list[TimelinePrefix]], limit: int, rng: random.Random) -> list[TimelinePrefix]:
    """Take up to ``limit`` prefixes per bucket, preferring distinct arcs."""
    chosen: list[TimelinePrefix] = []
    for bucket in BUCKETS:
        candidates = grouped[bucket][:]
        rng.shuffle(candidates)
        used: set[str] = set()
        picked: list[TimelinePrefix] = []
        for prefix in candidates:
            if len(picked) >= limit:
                break
            if prefix.arc_id in used:
                continue
            used.add(prefix.arc_id)
            picked.append(prefix)
        if len(picked) < limit:
            for prefix in candidates:
                if len(picked) >= limit:
                    break
                if prefix in picked:
                    continue
                picked.append(prefix)
        picked.sort(key=lambda p: (p.arc_id, p.session_n, p.turn_ordinal))
        chosen.extend(picked)
    return chosen


# --------------------------------------------------------------------------- #
# sampling and scoring
# --------------------------------------------------------------------------- #
@dataclass(slots=True)
class SampleResult:
    """One sampled continuation, measured against the anchors it inherited."""

    arm: str
    bucket: str
    arc_id: str
    session_n: int
    turn_ordinal: int
    prefix_messages: int
    inherited: list[str]
    emitted: list[str]
    missing: list[str]
    invented: list[str]
    recall: float
    parsed: bool
    collapsed: bool

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "bucket": self.bucket,
            "arc_id": self.arc_id,
            "session_n": self.session_n,
            "turn_ordinal": self.turn_ordinal,
            "prefix_messages": self.prefix_messages,
            "inherited": self.inherited,
            "emitted": self.emitted,
            "missing": self.missing,
            "invented": self.invented,
            "recall": self.recall,
            "parsed": self.parsed,
            "collapsed": self.collapsed,
        }


def score_sample(arm: str, prefix: TimelinePrefix, reply: str | None) -> SampleResult:
    """Measure one sampled continuation against the prefix's inherited anchors."""
    ledger = parse_ledger(reply)
    base: dict[str, Any] = {
        "arm": arm,
        "bucket": prefix.bucket,
        "arc_id": prefix.arc_id,
        "session_n": prefix.session_n,
        "turn_ordinal": prefix.turn_ordinal,
        "prefix_messages": len(prefix.messages),
        "inherited": sorted(prefix.inherited),
    }
    if ledger is None:
        return SampleResult(
            **base,
            emitted=[],
            missing=sorted(prefix.inherited),
            invented=[],
            recall=0.0,
            parsed=False,
            collapsed=False,
        )
    emitted = ledger_anchors(ledger)
    check = check_drift(prefix.inherited, emitted)
    return SampleResult(
        **base,
        emitted=sorted(emitted),
        missing=sorted(check.missing),
        invented=sorted(check.invented),
        recall=check.recall,
        parsed=True,
        collapsed=check.collapsed,
    )


@dataclass(slots=True)
class BucketSummary:
    """Per-arm, per-bucket rollup."""

    arm: str
    bucket: str
    samples: int = 0
    parsed: int = 0
    drift: int = 0
    collapsed: int = 0
    recall_total: float = 0.0
    missing_total: int = 0

    def add(self, result: SampleResult) -> None:
        self.samples += 1
        if not result.parsed:
            return
        self.parsed += 1
        self.recall_total += result.recall
        self.missing_total += len(result.missing)
        if result.missing:
            self.drift += 1
        if result.collapsed:
            self.collapsed += 1

    @property
    def drift_rate(self) -> float:
        return self.drift / self.parsed if self.parsed else 0.0

    @property
    def mean_recall(self) -> float:
        return self.recall_total / self.parsed if self.parsed else 0.0

    def to_dict(self) -> dict[str, Any]:
        return {
            "arm": self.arm,
            "bucket": self.bucket,
            "samples": self.samples,
            "parsed": self.parsed,
            "drift": self.drift,
            "drift_rate": round(self.drift_rate, 4),
            "collapsed": self.collapsed,
            "mean_recall": round(self.mean_recall, 4),
            "missing_anchors": self.missing_total,
        }


@dataclass(slots=True)
class ProbeSettings:
    samples: int
    temperature: float
    concurrency: int
    max_tokens: int


def run_arm(arm: str, model: str, prefixes: list[TimelinePrefix], settings: ProbeSettings) -> list[SampleResult]:
    """Sample and score every prefix for one arm."""
    sampler = Sampler(
        model=model,
        api_key=api_key(),
        temperature=settings.temperature,
        max_tokens=settings.max_tokens,
    )
    results: list[SampleResult] = []
    started = time.time()
    with ThreadPoolExecutor(max_workers=settings.concurrency) as pool:
        futures = {pool.submit(sampler.sample, prefix.messages, settings.samples): prefix for prefix in prefixes}
        total = len(futures)
        for done, future in enumerate(as_completed(futures), start=1):
            prefix = futures[future]
            replies = future.result()
            if not replies:
                print(f"  ! {arm} {prefix.arc_id} s{prefix.session_n}: no replies", file=sys.stderr)
                results.append(score_sample(arm, prefix, None))
                continue
            results.extend(score_sample(arm, prefix, reply) for reply in replies)
            if done % 10 == 0 or done == total:
                elapsed = time.time() - started
                rate = done / elapsed if elapsed else 0.0
                print(f"  {arm}: {done}/{total} prefixes ({rate:.2f}/s)")
    return results


def summarise(results: list[SampleResult]) -> list[BucketSummary]:
    """Roll samples up into arm x bucket rows, ordered deterministically."""
    summaries: dict[tuple[str, str], BucketSummary] = {}
    for result in results:
        key = (result.arm, result.bucket)
        if key not in summaries:
            summaries[key] = BucketSummary(arm=result.arm, bucket=result.bucket)
        summaries[key].add(result)
    return [summaries[key] for key in sorted(summaries)]


def print_report(summaries: list[BucketSummary]) -> None:
    header = f"{'arm':<5} {'bucket':<10} {'n':>4} {'parsed':>6} {'drift':>5} {'rate':>6} {'collapse':>8} {'recall':>7}"
    print(header)
    print("-" * len(header))
    for summary in summaries:
        print(
            f"{summary.arm:<5} {summary.bucket:<10} {summary.samples:>4} {summary.parsed:>6} "
            f"{summary.drift:>5} {summary.drift_rate:>6.1%} {summary.collapsed:>8} {summary.mean_recall:>7.3f}"
        )
    for arm in sorted({summary.arm for summary in summaries}):
        rows = [summary for summary in summaries if summary.arm == arm]
        parsed = sum(row.parsed for row in rows)
        drift = sum(row.drift for row in rows)
        missing = sum(row.missing_total for row in rows)
        rate = drift / parsed if parsed else 0.0
        print(f"{arm} overall: {drift}/{parsed} parsed samples drifted ({rate:.1%}), {missing} anchors lost")


# --------------------------------------------------------------------------- #
# cli
# --------------------------------------------------------------------------- #
def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--records", type=Path, default=RECORDS_PATH)
    parser.add_argument("--release-dir", type=Path, default=RELEASE_DIR)
    parser.add_argument("--exclusion-cache", type=Path, default=EXCLUSION_CACHE)
    parser.add_argument("--out-dir", type=Path, default=OUT_ROOT)
    parser.add_argument("--arm", choices=("both", *ARMS), default="both")
    parser.add_argument("--base-model", default=BASE_MODEL)
    parser.add_argument("--sft-model", default=SFT_MODEL)
    parser.add_argument("--per-bucket", type=int, default=12)
    parser.add_argument("--samples", type=int, default=2)
    parser.add_argument("--temperature", type=float, default=0.7)
    parser.add_argument("--max-tokens", type=int, default=1400)
    parser.add_argument("--concurrency", type=int, default=6)
    parser.add_argument("--seed", type=int, default=20261105)
    parser.add_argument("--dry-run", action="store_true", help="select prefixes and report the strata, no sampling")
    return parser.parse_args(argv)


def _select(args: argparse.Namespace, rng: random.Random) -> list[TimelinePrefix]:
    sessions = load_sessions(args.records)
    membership = arc_split_membership(args.release_dir, args.exclusion_cache)
    excluded = {arc for arc, splits in membership.items() if set(splits) & set(EXCLUDED_SPLITS)}
    print(f"sessions: {len(sessions)} | arcs in release: {len(membership)} | excluded arcs: {len(excluded)}")

    grouped = enumerate_prefixes(sessions, excluded)
    print("available per bucket: " + ", ".join(f"{bucket}={len(grouped[bucket])}" for bucket in BUCKETS))
    prefixes = _cap(grouped, args.per_bucket, rng)
    print(f"selected prefixes: {len(prefixes)} | distinct arcs: {len({p.arc_id for p in prefixes})}")
    for bucket in BUCKETS:
        rows = [p for p in prefixes if p.bucket == bucket]
        if not rows:
            continue
        messages = [len(p.messages) for p in rows]
        anchors = [len(p.inherited) for p in rows]
        print(
            f"  {bucket}: n={len(rows)} messages={min(messages)}-{max(messages)} "
            f"inherited_anchors={min(anchors)}-{max(anchors)}"
        )
    drifted = [p for p in prefixes if not p.reference_holds]
    print(f"reference (accepted next turn) drift: {len(drifted)}/{len(prefixes)}")
    return prefixes


def _write_outputs(
    args: argparse.Namespace,
    prefixes: list[TimelinePrefix],
    results: list[SampleResult],
    summaries: list[BucketSummary],
) -> Path:
    stamp = time.strftime("%Y%m%dT%H%M%SZ", time.gmtime())
    out_dir: Path = args.out_dir / stamp
    out_dir.mkdir(parents=True, exist_ok=True)

    samples_path = out_dir / "samples.jsonl"
    with samples_path.open("w", encoding="utf-8") as handle:
        for result in results:
            handle.write(json.dumps(result.to_dict(), ensure_ascii=False) + "\n")

    payload = {
        "probed_at": stamp,
        "records_path": str(args.records),
        "release_dir": str(args.release_dir),
        "arm": args.arm,
        "base_model": args.base_model,
        "sft_model": args.sft_model,
        "per_bucket": args.per_bucket,
        "samples_per_prefix": args.samples,
        "temperature": args.temperature,
        "seed": args.seed,
        "prefixes": [
            {
                "arc_id": p.arc_id,
                "session_n": p.session_n,
                "turn_ordinal": p.turn_ordinal,
                "bucket": p.bucket,
                "prefix_messages": len(p.messages),
                "inherited": sorted(p.inherited),
                "reference": sorted(p.reference),
                "reference_holds": p.reference_holds,
            }
            for p in prefixes
        ],
        "summaries": [summary.to_dict() for summary in summaries],
    }
    (out_dir / "probe.json").write_text(json.dumps(payload, indent=2, ensure_ascii=False), encoding="utf-8")
    return out_dir


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    rng = random.Random(args.seed)
    prefixes = _select(args, rng)
    if not prefixes:
        print("no prefixes selected", file=sys.stderr)
        return 1
    if args.dry_run:
        return 0

    settings = ProbeSettings(
        samples=args.samples,
        temperature=args.temperature,
        concurrency=args.concurrency,
        max_tokens=args.max_tokens,
    )
    models = {"base": args.base_model, "sft": args.sft_model}
    arms = ARMS if args.arm == "both" else (args.arm,)

    results: list[SampleResult] = []
    for arm in arms:
        print(f"sampling {arm}: {models[arm]}")
        results.extend(run_arm(arm, models[arm], prefixes, settings))

    summaries = summarise(results)
    print_report(summaries)
    out_dir = _write_outputs(args, prefixes, results, summaries)
    print(f"wrote {out_dir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
