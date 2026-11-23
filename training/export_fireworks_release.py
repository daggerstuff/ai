#!/usr/bin/env python3
"""Export a release's train split as a Fireworks supervised fine-tuning dataset.

The W&B path trains on the whole partition. Fireworks charges per training token
and its tunable bases are a different family, so the useful first question there
is narrow: does the same corpus carry the crisis-routing signal onto a non-Qwen
model? That only needs the rows that actually teach the behaviour, not 82k rows
of therapy dialogue.

Selection is therefore explicit and deterministic:

* every chat row whose assistant turns name a routing resource (988, 911, crisis
  line, poison control, emergency department, safety plan) — the v18 signal;
* every chat row that touches acute risk in the client turn, so routing is
  learned against adjacent context rather than as a keyword reflex;
* a deterministic sample of the remaining chat rows, taken by lowest sha256 of
  the emitted payload so the choice does not depend on file order.

Tool rows are excluded by default. Fireworks renders ``tools``/``tool_calls``
from its own template and a plain weighted-messages payload is what its dataset
validator accepts without argument; ``--tool-rows include`` opts in when we have
confirmed the schema against a real job.

Output lines are ``{"messages": [{"role", "content", "weight"}]}`` with user
weight 0 and assistant weight 1, matching the release's own projection so the
model is only ever scored on what it is supposed to say.
"""

from __future__ import annotations

import argparse
import hashlib
import importlib
import json
import os
import sys
from collections import Counter
from collections.abc import Mapping
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any

ROOT = Path(__file__).resolve().parents[2]
DEFAULT_MANIFEST = ROOT / "docs/training/execution-2026-11-09/wandb-full-corpus-current-master-20261109-v18.json"
DEFAULT_OUT_DIR = ROOT / "ai/training/output/fireworks"
DEFAULT_SEED = 20261109
DEFAULT_CHAT_SAMPLE = 4_000
DEFAULT_ROUTING_CAP = 2_000
DEFAULT_MAX_CHARS = 12_000
DEFAULT_SAMPLE_MAX_CHARS = 2_000
CHAR_PER_TOKEN = 4
CRISIS_SOURCE_PREFIX = "crisis_direct_"

ROUTING_TERMS = (
    "988",
    "911",
    "crisis line",
    "crisis hotline",
    "poison control",
    "emergency department",
    "emergency services",
    "ambulance",
    "safety plan",
    "urgent care",
)

ACUTE_TERMS = (
    "kill myself",
    "killing myself",
    "suicide",
    "suicidal",
    "end my life",
    "overdose",
    "took too many",
    "od on",
    "cut myself",
    "cutting",
    "self-harm",
    "self harm",
    "hurt myself",
    "hang myself",
    "pills",
    "no reason to live",
    "better off dead",
    "want to die",
)


@dataclass
class Selection:
    """One release row, the reason it was selected, and its emitted bytes."""

    row_number: int
    reasons: list[str] = field(default_factory=list)
    payload: dict[str, Any] | None = None
    line: str = ""
    chars: int = 0


@dataclass(frozen=True)
class SelectionOptions:
    """How much of the train split to carry over to Fireworks.

    ``max_chars`` drops rows a Fireworks run would only partly see: the routing
    turn sits late in a long arc, so a row longer than the training context is
    paid for in full and teaches nothing. ``sample_max_chars`` keeps the filler
    sample cheap, and ``routing_cap`` bounds the expensive risk rows.

    ``keep_replies`` is a hand-written surface that must reach the dataset whole.
    The crisis family is written as one disclosure and one reply, and its replies
    share the routing vocabulary, so without this set the family is split between
    the capped routing pool and the hash-sampled filler pool and only a fraction
    of it survives. Because the filler sample is keyed on the row's serialized
    bytes, that fraction also moves every time a release is rebuilt.
    """

    seed: int = DEFAULT_SEED
    chat_sample: int = DEFAULT_CHAT_SAMPLE
    routing_cap: int = DEFAULT_ROUTING_CAP
    max_chars: int = DEFAULT_MAX_CHARS
    sample_max_chars: int = DEFAULT_SAMPLE_MAX_CHARS
    include_tool_rows: bool = False
    keep_replies: frozenset[str] | None = None


def _read_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} does not contain a JSON object")
    return payload


def _weighted_messages(row: dict[str, Any]) -> list[dict[str, Any]]:
    """Return the row's messages with the release's user/assistant weighting."""
    messages: list[dict[str, Any]] = []
    for message in row["messages"]:
        role = message.get("role")
        if role not in ("system", "user", "assistant"):
            raise ValueError(f"unexpected role {role!r} in the train payload")
        entry: dict[str, Any] = {"role": role, "content": message.get("content") or ""}
        if role != "system":
            entry["weight"] = 0 if role == "user" else 1
        messages.append(entry)
    return messages


def classify(row: dict[str, Any]) -> list[str]:
    """Return why a row belongs in a Fireworks dataset (empty when it does not)."""
    if "tools" in row:
        return []
    messages = row["messages"]
    assistant = " ".join((m.get("content") or "") for m in messages if m.get("role") == "assistant").lower()
    client = " ".join((m.get("content") or "") for m in messages if m.get("role") == "user").lower()
    reasons: list[str] = []
    if any(term in assistant for term in ROUTING_TERMS):
        reasons.append("routing")
    if any(term in client for term in ACUTE_TERMS):
        reasons.append("acute_client")
    return reasons


def _sample_key(line: str, seed: int) -> str:
    return hashlib.sha256(f"{seed}:{line}".encode()).hexdigest()


def carries_keep_reply(row: dict[str, Any], keep_replies: frozenset[str]) -> bool:
    """Return whether any assistant turn in the row is a keep-set surface."""
    return any(
        message.get("content") in keep_replies for message in row["messages"] if message.get("role") == "assistant"
    )


def crisis_reply_surface(manifest: Mapping[str, Any]) -> frozenset[str]:
    """Return every assistant reply the release appended as a crisis-family row.

    The appended rows carry only ``messages``, so once they are in the train
    split nothing distinguishes them from the general corpus: their replies
    contain the routing vocabulary, which sends them into the capped routing
    pool, and the rest land in the hash-sampled filler pool. That kept about a
    quarter of the family. The manifest still names the builder that produced
    the rows, and that builder's ``collect_rows()`` is the authoritative
    surface, so the keep set is derived rather than restated here.
    """
    sources = (manifest.get("scope") or {}).get("appended_rows_by_source") or {}
    tags = sorted(name[len(CRISIS_SOURCE_PREFIX) :] for name in sources if name.startswith(CRISIS_SOURCE_PREFIX))
    if not tags:
        # A manifest that predates the crisis family has nothing to keep whole.
        return frozenset()
    replies: set[str] = set()
    for tag in tags:
        module_name = f"training.build_agent_tool_sft_{tag}"
        try:
            module = importlib.import_module(module_name)
        except ImportError as error:
            raise ValueError(f"{module_name} is not importable: {error}") from error
        for rows in module.collect_rows().values():
            for row in rows:
                replies.update(
                    message["content"]
                    for message in row["messages"]
                    if message.get("role") == "assistant" and message.get("content")
                )
    return frozenset(replies)


def select_rows(
    train_path: Path,
    options: SelectionOptions,
) -> tuple[list[Selection], Counter[str]]:
    """Select the rows to export and count them by reason."""
    family: list[Selection] = []
    routing: list[Selection] = []
    acute: list[Selection] = []
    filler: list[Selection] = []
    counts: Counter[str] = Counter()

    with train_path.open("r", encoding="utf-8") as handle:
        for row_number, raw in enumerate(handle, start=1):
            row = json.loads(raw)
            if "tools" in row and not options.include_tool_rows:
                counts["tool_rows_excluded"] += 1
                continue
            selection = _write_payload(Selection(row_number=row_number), row)
            if options.keep_replies and carries_keep_reply(row, options.keep_replies):
                # Uncapped and checked first: a keep-set row is the reason the
                # release exists, so neither the routing cap nor the filler
                # sample may decide whether it is trained on.
                if selection.chars > options.max_chars:
                    counts["crisis_too_long_excluded"] += 1
                    continue
                selection.reasons = ["crisis_family"]
                family.append(selection)
                continue
            if selection.chars > options.max_chars:
                counts["too_long_excluded"] += 1
                continue
            reasons = classify(row)
            if "routing" in reasons:
                routing.append(selection)
            elif "acute_client" in reasons:
                acute.append(selection)
            elif selection.chars <= options.sample_max_chars:
                filler.append(selection)

    kept = list(family)
    kept.extend(acute)
    kept.extend(_lowest_by_hash(routing, options.routing_cap, options.seed, "routing"))
    sampled = _lowest_by_hash(filler, options.chat_sample, options.seed, "sample")
    kept.extend(sampled)
    kept.sort(key=lambda selection: selection.row_number)

    counts["crisis_rows"] = len(family)
    counts["routing_rows"] = len(routing)
    counts["routing_kept"] = sum(1 for selection in kept if "routing" in selection.reasons)
    counts["acute_client_rows"] = len(acute)
    counts["sample"] = len(sampled)
    return kept, counts


def _lowest_by_hash(
    pool: list[Selection],
    limit: int,
    seed: int,
    reason: str,
) -> list[Selection]:
    """Take the lowest-hash rows from a pool so the choice ignores file order."""
    if limit <= 0:
        return []
    chosen = sorted(pool, key=lambda selection: _sample_key(selection.line, seed))[:limit]
    for selection in chosen:
        selection.reasons = [reason]
    return chosen


def _write_payload(selection: Selection, row: dict[str, Any]) -> Selection:
    """Attach the emitted payload, bytes, and size to one selection."""
    selection.payload = {"messages": _weighted_messages(row)}
    selection.line = json.dumps(selection.payload, ensure_ascii=False, separators=(",", ":"))
    selection.chars = sum(len(message["content"]) for message in selection.payload["messages"])
    return selection


def _write_jsonl(path: Path, selections: list[Selection]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for selection in selections:
            handle.write(selection.line)
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def export(
    manifest_path: Path,
    out_dir: Path,
    *,
    options: SelectionOptions | None = None,
    dry_run: bool = False,
) -> dict[str, Any]:
    """Write the Fireworks dataset and return its identity record."""
    manifest = _read_json(manifest_path)
    train = manifest["partitions"]["train"]
    train_path = ROOT / train["path"]
    if not train_path.is_file():
        raise FileNotFoundError(f"train partition not found: {train_path}")

    resolved = options or SelectionOptions()
    if resolved.keep_replies is None:
        # Re-derive the keep set from the manifest on every export: a caller that
        # left it unset must still retain the crisis family whole, or the
        # routing cap and filler sample decide which graded rows train.
        resolved = replace(resolved, keep_replies=crisis_reply_surface(manifest))
    selections, counts = select_rows(train_path, resolved)
    if not selections:
        raise ValueError("selection produced no rows")
    chars = sum(selection.chars for selection in selections)
    summary: dict[str, Any] = {
        "release_id": manifest["release_id"],
        "source_manifest": str(manifest_path),
        "source_train": str(train_path),
        "source_train_sha256": train["sha256"],
        "source_train_rows": train["row_count"],
        "selection": {
            "crisis_rows_kept": counts["crisis_rows"],
            "crisis_too_long_excluded": counts["crisis_too_long_excluded"],
            "routing_rows_available": counts["routing_rows"],
            "routing_rows_kept": counts["routing_kept"],
            "acute_client_rows": counts["acute_client_rows"],
            "sampled_rows": counts["sample"],
            "tool_rows_excluded": counts["tool_rows_excluded"],
            "too_long_excluded": counts["too_long_excluded"],
            "seed": resolved.seed,
            "chat_sample": resolved.chat_sample,
            "routing_cap": resolved.routing_cap,
            "max_chars": resolved.max_chars,
            "sample_max_chars": resolved.sample_max_chars,
            "tool_rows_included": resolved.include_tool_rows,
        },
        "rows": len(selections),
        "chars": chars,
        "estimated_tokens": round(chars / CHAR_PER_TOKEN),
        "format": "fireworks_weighted_messages_jsonl_v1",
    }
    if dry_run:
        return summary

    dataset_path = out_dir / "fireworks-sft-train.jsonl"
    _write_jsonl(dataset_path, selections)
    summary["dataset_path"] = str(dataset_path)
    summary["dataset_sha256"] = _sha256(dataset_path)
    summary["dataset_bytes"] = dataset_path.stat().st_size
    identity_path = out_dir / "fireworks-dataset.json"
    identity_path.write_text(json.dumps(summary, indent=1) + "\n", encoding="utf-8")
    return summary


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Export the Fireworks SFT dataset for a release.")
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--seed", type=int, default=DEFAULT_SEED)
    parser.add_argument("--chat-sample", type=int, default=DEFAULT_CHAT_SAMPLE)
    parser.add_argument(
        "--tool-rows",
        choices=("exclude", "include"),
        default="exclude",
        help="whether to carry the agent tool rows (default: exclude)",
    )
    parser.add_argument("--dry-run", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = _parse_args(argv)
    summary = export(
        args.manifest,
        args.out_dir,
        options=SelectionOptions(
            seed=args.seed,
            chat_sample=args.chat_sample,
            include_tool_rows=args.tool_rows == "include",
        ),
        dry_run=args.dry_run,
    )
    sys.stdout.write(json.dumps(summary, indent=1) + "\n")
    return 0


if __name__ == "__main__":
    sys.exit(main())
