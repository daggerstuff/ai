#!/usr/bin/env python3
"""Export a release's train split as tokenized Tinker training datums.

Tinker is not a chat-completions trainer: it takes ``Datum`` records of token
ids with a per-token loss weight and runs ``forward_backward``/``optim_step``
itself. So the release payload has to be converted once, here, rather than
being re-rendered by a provider at training time.

Two things make that conversion more than a ``tokenizer.encode`` call:

* The release's own contract is that the model is scored on assistant turns
  only (``weight`` 1) and never on the user or system turn (``weight`` 0).
  Tinker expresses the same idea as a per-token weight vector, so the mask has
  to be built turn by turn.
* A chat template puts the role header (``<|im_start|>assistant\\n``) *before*
  the content it belongs to. Training on that header would teach the model to
  emit role tags, so the header is masked and only content plus the turn's end
  token are scored.

The mask is built from character offsets rather than from prefix renders: the
conversation is rendered to text once, each message's content is located in that
text in order, and the fast tokenizer's offset map turns those spans into token
positions. Prefix renders cannot do this job — Qwen's template emits an empty
thinking block for every assistant turn *except the last one*, so rendering a
prefix changes tokens that a longer render already produced. The offset build is
checked against the template's own tokenization, so a mismatch fails loudly
instead of silently training on the wrong tokens.

Row selection is ``select_rows``, shared with the Fireworks export so both
providers agree on what a row is, but the Tinker export selects the whole
partition (``SelectionOptions.full_corpus``). The caps in that module are the
Fireworks probe's cost bounds, and inheriting them silently dropped 85% of the
training corpus; ``--selection narrow`` reproduces the historical sample.
Tool rows are excluded by default: Tinker trains on raw tokens, and the tool
schema would have to be rendered into the prompt by us to be meaningful.

Output lines are ``{"input_ids", "target_ids", "weights"}`` where
``target_ids`` is ``input_ids`` shifted by one and ``weights`` is aligned to
``target_ids``, matching Tinker's cross-entropy contract (the target at
position ``t`` is the token the model should have produced after position
``t``).
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

from training.export_fireworks_release import Selection, SelectionOptions, select_rows

ROOT = Path(__file__).resolve().parents[2]
CRISIS_SOURCE_PREFIX = "crisis_direct_"
DEFAULT_MANIFEST = ROOT / "docs/training/execution-2026-11-10/wandb-full-corpus-current-master-20261110-v19.json"
DEFAULT_OUT_DIR = ROOT / "ai/training/output/tinker"

# Must be the release's own base model: the token ids in these datums are only
# meaningful to the tokenizer that produced them, and the v19 release trains
# ``Qwen/Qwen3.6-35B-A3B`` (``Qwen3_5Tokenizer``, 248,077 entries) whose
# vocabulary and chat template both differ from ``Qwen/Qwen3-8B``
# (``Qwen2Tokenizer``, 151,669 entries). Whether Tinker hosts that model is
# confirmed against ``get_server_capabilities`` (see
# ``tinker_sft_release.py --check``).
DEFAULT_BASE_MODEL = "Qwen/Qwen3.6-35B-A3B"

# Tinker bills per training token, and a datum longer than the model's context
# is rejected at forward time rather than at upload time.
DEFAULT_MAX_TOKENS = 8192

# ``max_tokens`` is the history budget: longer arcs are trimmed to their final
# exchange. The exchange itself gets its own allowance, because the collator
# pads each batch to its widest row and a disclosure that does not fit the
# history budget is not a reason to drop the target it was answered with.
DEFAULT_FINAL_EXCHANGE_TOKENS = 1024

# The token that closes a turn. It belongs to the assistant span because it is
# what the model has to learn to emit after its last word. Qwen ends a turn with
# ``<|im_end|>``; Muse-Glimmer ends it with ``<|eot|>``.
DEFAULT_END_TOKEN = "<|im_end|>"

DATASET_SCHEMA = "tinker-tokenized-datums-v1"

# A datum needs at least one input token and one shifted target token.
MIN_DATUM_TOKENS = 2


class TokenizationError(RuntimeError):
    """The chat template is not prefix-consistent, so the mask cannot be built."""


@dataclass
class Datum:
    """One training example in Tinker's shifted-target layout."""

    input_ids: list[int]
    target_ids: list[int]
    weights: list[int]

    @property
    def trained_tokens(self) -> int:
        """Return the number of tokens the loss is computed on."""
        return sum(self.weights)

    def line(self) -> str:
        """Return the JSONL form of this datum."""
        payload = {
            "input_ids": self.input_ids,
            "target_ids": self.target_ids,
            "weights": self.weights,
        }
        return json.dumps(payload, separators=(",", ":"))


@dataclass
class Export:
    """Everything the export produced, for the dataset record and the CLI."""

    rows: list[Datum] = field(default_factory=list)
    counts: Counter[str] = field(default_factory=Counter)
    source_rows: int = 0


def _plain(messages: list[dict[str, Any]]) -> list[dict[str, str]]:
    """Return the messages as the template will see them.

    Drops the release's per-message ``weight`` and reproduces the template's two
    content rewrites, because the mask is measured against the rendered text:

    * the template renders every content with ``|trim``, so leading and trailing
      whitespace is not part of what the model sees;
    * an assistant turn containing ``</think>`` is split into a thinking block
      plus an answer, so the rendered text would stop matching the release text.
      Stripping the reasoning keeps mask and training text identical, and it is
      what the release repair already does for reasoning preambles. The tail is
      trimmed too, because the template trims whatever content it is given.
    """
    plain: list[dict[str, str]] = []
    for message in messages:
        content = message["content"].strip()
        if message["role"] == "assistant" and "</think>" in content:
            content = content.split("</think>")[-1].strip()
        plain.append({"role": message["role"], "content": content})
    return plain


def _render(
    tokenizer: Any,
    messages: list[dict[str, str]],
    *,
    add_generation_prompt: bool,
    template_kwargs: Mapping[str, Any] | None = None,
) -> list[int]:
    """Return the token ids of a rendered conversation."""
    rendered = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=add_generation_prompt,
        **(template_kwargs or {}),
    )
    if isinstance(rendered, Mapping):
        rendered = rendered["input_ids"]
    return [int(token) for token in rendered]


def build_tokens(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    *,
    end_token: str = DEFAULT_END_TOKEN,
    template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[int], list[int]]:
    """Return ``(token_ids, weights)`` for one release row.

    Weights are 1 on assistant content and on the assistant turn's end token,
    and 0 everywhere else (role headers, user turns, system turns).
    ``template_kwargs`` are the template's own inputs (dates, reasoning
    strength); a template that injects the current date needs them pinned or
    the export changes from one day to the next.
    """
    plain = _plain(messages)
    text = tokenizer.apply_chat_template(
        plain,
        tokenize=False,
        add_generation_prompt=False,
        **(template_kwargs or {}),
    )
    encoding = tokenizer(text, add_special_tokens=False, return_offsets_mapping=True)
    ids = [int(token) for token in encoding["input_ids"]]
    offsets = [(int(start), int(end)) for start, end in encoding["offset_mapping"]]

    expected = _render(
        tokenizer,
        plain,
        add_generation_prompt=False,
        template_kwargs=template_kwargs,
    )
    if expected != ids:
        raise TokenizationError("tokenizing the rendered conversation disagrees with the chat template's own render")

    spans: list[tuple[int, int]] = []
    cursor = 0
    for index, message in enumerate(plain):
        content = message["content"]
        start = text.find(content, cursor) if content else cursor
        if start < 0:
            raise TokenizationError(f"message {index} content is not present in the rendered conversation")
        end = start + len(content)
        cursor = end
        if not messages[index].get("weight"):
            continue
        # The end-of-turn token belongs to the assistant turn: it is what the
        # model has to learn to emit after its last word.
        turn_end = text.find(end_token, end)
        spans.append((start, end if turn_end < 0 else turn_end + len(end_token)))

    weights = [
        1 if any(start >= span_start and end <= span_end for span_start, span_end in spans) else 0
        for start, end in offsets
    ]
    if len(weights) != len(ids):
        raise TokenizationError("offset map and token ids disagree in length")
    return ids, weights


MIN_FINAL_EXCHANGE_MESSAGES = 2


@dataclass(frozen=True)
class TokenLimits:
    """The two token budgets a row is checked against."""

    max_tokens: int
    final_exchange_tokens: int

    def budget(self, *, trimmed: bool, message_count: int) -> int:
        """Return the budget for a row after left-trimming.

        A trimmed row is bounded by the final-exchange window it was reduced
        to. A row that arrived as a single exchange keeps the larger of the two
        budgets: it never had history trimmed away, so it must not be dropped
        for exceeding a window it never exceeded, and a final-exchange
        allowance above ``max_tokens`` still applies to it.
        """
        if message_count != MIN_FINAL_EXCHANGE_MESSAGES:
            return self.max_tokens
        if trimmed:
            return self.final_exchange_tokens
        return max(self.max_tokens, self.final_exchange_tokens)


def trim_to_budget(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_tokens: int,
    *,
    end_token: str = DEFAULT_END_TOKEN,
    template_kwargs: Mapping[str, Any] | None = None,
) -> tuple[list[dict[str, Any]], list[int], list[int], bool]:
    ids, weights = build_tokens(
        tokenizer,
        messages,
        end_token=end_token,
        template_kwargs=template_kwargs,
    )
    original_length = len(messages)
    while len(ids) > max_tokens and len(messages) > MIN_FINAL_EXCHANGE_MESSAGES:
        messages = messages[1:]
        while len(messages) > MIN_FINAL_EXCHANGE_MESSAGES and messages[0]["role"] != "user":
            messages = messages[1:]
        ids, weights = build_tokens(
            tokenizer,
            messages,
            end_token=end_token,
            template_kwargs=template_kwargs,
        )
    return messages, ids, weights, len(messages) < original_length


def build_datum(
    tokenizer: Any,
    messages: list[dict[str, Any]],
    max_tokens: int = DEFAULT_MAX_TOKENS,
    *,
    end_token: str = DEFAULT_END_TOKEN,
    template_kwargs: Mapping[str, Any] | None = None,
) -> Datum | None:
    """Return the shifted-target datum for one row, or None if it is unusable."""
    ids, weights = build_tokens(
        tokenizer,
        messages,
        end_token=end_token,
        template_kwargs=template_kwargs,
    )
    if len(ids) < MIN_DATUM_TOKENS:
        return None
    if len(ids) > max_tokens:
        return None
    return Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=weights[1:])


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


def select_for_export(
    manifest: Mapping[str, Any],
    train_path: Path,
    options: SelectionOptions | None,
    limit: int | None,
) -> tuple[list[Selection], Counter[str]]:
    """Select the rows to tokenize, keeping the crisis family whole.

    The crisis family is split between the capped routing pool and the
    hash-sampled filler pool, so which quarter of it survives a rebuild is an
    accident of the row's serialized bytes. Deriving the keep set from the
    release's own appended source and selecting it before either pool means the
    graded rows train every time instead of a different subset each release.
    """
    if options is None:
        options = SelectionOptions.full_corpus()
    if options.keep_replies is None:
        options = replace(options, keep_replies=crisis_reply_surface(manifest))
    selections, counts = select_rows(train_path, options)
    if limit is not None:
        # Smoke exports only: a truncated selection still exercises the real
        # tokenizer and template, but its counts are not a release figure.
        selections = selections[:limit]
    return selections, counts


def _verified_selection(
    manifest: Mapping[str, Any],
    train_path: Path,
    options: SelectionOptions | None,
    limit: int | None,
) -> tuple[list[Selection], Counter[str], str]:
    source_train_sha256 = _sha256(train_path)
    if source_train_sha256 != manifest["partitions"]["train"].get("sha256"):
        raise ValueError("train payload SHA-256 does not match the release manifest")
    selections, counts = select_for_export(manifest, train_path, options, limit)
    if _sha256(train_path) != source_train_sha256:
        raise ValueError("train payload SHA-256 changed during selection")
    return selections, counts, source_train_sha256


def tokenize_selections(
    selections: list[Selection],
    tokenizer: Any,
    *,
    max_tokens: int,
    final_exchange_tokens: int,
    end_token: str,
    template_kwargs: Mapping[str, Any] | None,
) -> tuple[list[Datum], dict[str, int]]:
    trimmed_opening = 0
    trimmed_history = 0
    skipped_too_long = 0
    skipped_empty = 0
    rows: list[Datum] = []
    for selection in selections:
        if selection.payload is None:
            raise ValueError("selected train row has no payload")
        messages = selection.payload["messages"]
        if messages[0]["role"] == "assistant":
            messages = messages[1:]
            trimmed_opening += 1
            if not messages:
                skipped_empty += 1
                continue
        messages, ids, weights, trimmed = trim_to_budget(
            tokenizer,
            messages,
            max_tokens,
            end_token=end_token,
            template_kwargs=template_kwargs,
        )
        if trimmed:
            trimmed_history += 1
        targets = weights[1:]
        if len(ids) < MIN_DATUM_TOKENS or not any(targets):
            skipped_empty += 1
            continue
        limits = TokenLimits(max_tokens=max_tokens, final_exchange_tokens=final_exchange_tokens)
        if len(ids) > limits.budget(trimmed=trimmed, message_count=len(messages)):
            skipped_too_long += 1
            continue
        rows.append(Datum(input_ids=ids[:-1], target_ids=ids[1:], weights=targets))
    return rows, {
        "datums_trimmed_opening": trimmed_opening,
        "datums_left_truncated": trimmed_history,
        "datums_skipped_too_long": skipped_too_long,
        "datums_skipped_no_targets": skipped_empty,
    }


def selection_mode(options: SelectionOptions) -> str:
    """Return the receipt's name for a selection, so the artifact says what it is."""
    caps = (options.chat_sample, options.routing_cap, options.max_chars, options.sample_max_chars)
    return "full_corpus" if all(cap is None for cap in caps) else "narrow"


def write_export(
    out_dir: Path,
    *,
    base_model: str,
    tokenizer: Any,
    end_token: str,
    template_kwargs: Mapping[str, Any] | None,
    manifest_path: Path,
    manifest_bytes: bytes,
    manifest: Mapping[str, Any],
    source_train_sha256: str,
    rows: list[Datum],
    counts: Mapping[str, int],
    source_rows: int,
    max_tokens: int,
    final_exchange_tokens: int,
    selection: str = "narrow",
) -> None:
    out_dir.mkdir(parents=True, exist_ok=True)
    path = out_dir / "tinker-sft-train.jsonl"
    tmp = path.with_name(path.name + ".tmp")
    with tmp.open("w", encoding="utf-8") as handle:
        for datum in rows:
            handle.write(datum.line())
            handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(tmp, path)
    record = {
        "schema": DATASET_SCHEMA,
        "base_model": base_model,
        "tokenizer": getattr(tokenizer, "name_or_path", base_model),
        "end_token": end_token,
        "template_kwargs": dict(template_kwargs or {}),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": hashlib.sha256(manifest_bytes).hexdigest(),
        "source_release_id": manifest.get("release_id"),
        "source_train_sha256": source_train_sha256,
        "path": str(path),
        "sha256": _sha256(path),
        "bytes": path.stat().st_size,
        "datums": len(rows),
        "source_rows": source_rows,
        "counts": dict(counts),
        "max_tokens": max_tokens,
        "max_final_exchange_tokens": final_exchange_tokens,
        "selection_mode": selection,
    }
    (out_dir / "tinker-dataset.json").write_text(json.dumps(record, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def export(
    manifest_path: Path,
    out_dir: Path,
    *,
    base_model: str = DEFAULT_BASE_MODEL,
    tokenizer_model: str | None = None,
    end_token: str = DEFAULT_END_TOKEN,
    template_kwargs: Mapping[str, Any] | None = None,
    options: SelectionOptions | None = None,
    max_tokens: int = DEFAULT_MAX_TOKENS,
    final_exchange_tokens: int = DEFAULT_FINAL_EXCHANGE_TOKENS,
    limit: int | None = None,
    tokenizer: Any | None = None,
    dry_run: bool = False,
) -> Export:
    """Tokenize the selected train rows and write the datum file."""
    manifest_bytes = manifest_path.read_bytes()
    manifest = json.loads(manifest_bytes)
    train_path = Path(manifest["partitions"]["train"]["path"])
    if not train_path.is_absolute():
        train_path = ROOT / train_path

    resolved = options if options is not None else SelectionOptions.full_corpus()
    selections, counts, source_train_sha256 = _verified_selection(manifest, train_path, resolved, limit)
    result = Export(counts=counts, source_rows=len(selections))

    if tokenizer is None:
        from transformers import AutoTokenizer

        # A Fireworks-hosted model is not on the Hub, so its tokenizer is named
        # separately from the model the trainer will load.
        tokenizer = AutoTokenizer.from_pretrained(tokenizer_model or base_model, fast=True)

    rows, token_counts = tokenize_selections(
        selections,
        tokenizer,
        max_tokens=max_tokens,
        final_exchange_tokens=final_exchange_tokens,
        end_token=end_token,
        template_kwargs=template_kwargs,
    )
    result.rows.extend(rows)
    result.counts.update(token_counts)
    result.counts["datums"] = len(result.rows)
    result.counts["trained_tokens"] = sum(datum.trained_tokens for datum in result.rows)
    result.counts["total_tokens"] = sum(len(datum.input_ids) for datum in result.rows)

    if not dry_run:
        write_export(
            out_dir,
            base_model=base_model,
            tokenizer=tokenizer,
            end_token=end_token,
            template_kwargs=template_kwargs,
            manifest_path=manifest_path,
            manifest_bytes=manifest_bytes,
            manifest=manifest,
            source_train_sha256=source_train_sha256,
            rows=result.rows,
            counts=result.counts,
            source_rows=result.source_rows,
            max_tokens=max_tokens,
            final_exchange_tokens=final_exchange_tokens,
            selection=selection_mode(resolved),
        )
    return result


def _sha256(path: Path) -> str:
    """Return the SHA-256 of a file."""
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            digest.update(block)
    return digest.hexdigest()


def _summary(result: Export) -> str:
    """Return the human-readable export summary."""
    counts = result.counts
    tokens = counts["trained_tokens"]
    lines = [
        f"datums: {counts['datums']} from {result.source_rows} selected rows",
        f"tokens: {counts['total_tokens']} total, {tokens} trained",
        f"skipped: too long {counts['datums_skipped_too_long']}, "
        f"no targets {counts['datums_skipped_no_targets']} "
        f"(opening turn trimmed on {counts['datums_trimmed_opening']} rows)",
        f"selection: routing {counts.get('routing_kept', 0)}/"
        f"{counts.get('routing_rows', 0)}, acute {counts.get('acute_client_rows', 0)}, "
        f"sampled {counts.get('sample', 0)}, tool excluded {counts.get('tool_rows_excluded', 0)}, "
        f"label rows excluded {counts.get('label_rows_excluded', 0)}",
    ]
    return "\n".join(lines)


def selection_options(
    selection: str,
    *,
    seed: int,
    chat_sample: int | None,
    routing_cap: int | None,
    include_tool_rows: bool,
) -> SelectionOptions:
    """Return the row selection the export runs with.

    A cap is only meaningful for the narrow sample, so passing one with the
    full partition is an error rather than a silently ignored argument.
    """
    if selection == "full":
        if chat_sample is not None or routing_cap is not None:
            raise ValueError("chat_sample and routing_cap require the narrow selection")
        return SelectionOptions.full_corpus(seed=seed, include_tool_rows=include_tool_rows)
    return SelectionOptions(
        seed=seed,
        chat_sample=SelectionOptions.chat_sample if chat_sample is None else chat_sample,
        routing_cap=SelectionOptions.routing_cap if routing_cap is None else routing_cap,
        include_tool_rows=include_tool_rows,
    )


def _parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    """Parse the export's command line."""
    parser = argparse.ArgumentParser(description=(__doc__ or "Tokenize a research SFT release.").splitlines()[0])
    parser.add_argument("--manifest", type=Path, default=DEFAULT_MANIFEST)
    parser.add_argument("--out-dir", type=Path, default=DEFAULT_OUT_DIR)
    parser.add_argument("--base-model", default=DEFAULT_BASE_MODEL)
    parser.add_argument(
        "--tokenizer-model",
        default=None,
        help="Hub id of the tokenizer (default: --base-model); needed when the base model is hosted",
    )
    parser.add_argument(
        "--end-token",
        default=DEFAULT_END_TOKEN,
        help="the chat template's turn terminator, scored as part of the assistant turn",
    )
    parser.add_argument(
        "--template-kwargs",
        default="{}",
        help="JSON object of pinned chat-template inputs (dates, reasoning strength)",
    )
    parser.add_argument("--max-tokens", type=int, default=DEFAULT_MAX_TOKENS)
    parser.add_argument(
        "--final-exchange-tokens",
        type=int,
        default=DEFAULT_FINAL_EXCHANGE_TOKENS,
        help="budget for the final user+assistant exchange once the history is trimmed",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="tokenize only the first N selected rows (smoke exports)",
    )
    parser.add_argument("--seed", type=int, default=SelectionOptions.seed)
    parser.add_argument(
        "--chat-sample",
        type=int,
        default=None,
        help="cap the filler sample (narrow selection only)",
    )
    parser.add_argument(
        "--routing-cap",
        type=int,
        default=None,
        help="cap the routing pool (narrow selection only)",
    )
    parser.add_argument(
        "--selection",
        choices=("full", "narrow"),
        default="full",
        help="full trains the whole partition; narrow is the Fireworks probe's capped sample",
    )
    parser.add_argument("--include-tool-rows", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args(argv)
    template_kwargs = json.loads(args.template_kwargs)
    if not isinstance(template_kwargs, dict):
        parser.error("--template-kwargs must be a JSON object")
    if args.selection == "full" and (args.chat_sample is not None or args.routing_cap is not None):
        parser.error("--chat-sample and --routing-cap require --selection narrow")
    args.template_kwargs = template_kwargs
    return args


def main(argv: list[str] | None = None) -> int:
    """Run the export from the command line."""
    args = _parse_args(argv)
    template_kwargs = args.template_kwargs
    options = selection_options(
        args.selection,
        seed=args.seed,
        chat_sample=args.chat_sample,
        routing_cap=args.routing_cap,
        include_tool_rows=args.include_tool_rows,
    )
    if args.dry_run:
        from transformers import AutoTokenizer

        tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_model or args.base_model, fast=True)
        result = export(
            args.manifest,
            args.out_dir,
            base_model=args.base_model,
            end_token=args.end_token,
            template_kwargs=template_kwargs,
            options=options,
            max_tokens=args.max_tokens,
            final_exchange_tokens=args.final_exchange_tokens,
            limit=args.limit,
            tokenizer=tokenizer,
            dry_run=True,
        )
        print(_summary(result))
        return 0

    result = export(
        args.manifest,
        args.out_dir,
        base_model=args.base_model,
        tokenizer_model=args.tokenizer_model,
        end_token=args.end_token,
        template_kwargs=template_kwargs,
        options=options,
        max_tokens=args.max_tokens,
        final_exchange_tokens=args.final_exchange_tokens,
        limit=args.limit,
    )
    print(_summary(result))
    print(f"wrote {args.out_dir / 'tinker-sft-train.jsonl'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
