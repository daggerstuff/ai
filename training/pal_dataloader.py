"""PAL-aware dataloaders for HuggingFace `transformers.Trainer`.

PIX-4076 — Phase 4: Torch/Lightning Training Pipeline Integration.

Consumes the JSONL files produced by Phases 1-3 of the PAL framework
(`ai/training_corpus/pal_framework/`):

* SFT JSONL — `{messages: [{role, content}], metadata: {...}}`
  (selection, dialogue, and unified-mix outputs)
* DPO JSONL — `{prompt: str, chosen: [{role, content}], rejected: [{role, content}], metadata: {...}}`
  (TRL DPO conversational format)

The SFT loader renders messages to ChatML text and masks non-assistant turns
in `labels` so the model only learns to produce assistant responses. The DPO
loader preserves the conversational message-list format that TRL `DPOTrainer`
expects when `chat_template` is set on the tokenizer.
"""

from __future__ import annotations

import json
import logging
from collections.abc import Iterable
from pathlib import Path
from typing import TYPE_CHECKING, Any

if TYPE_CHECKING:  # pragma: no cover — typing only
    pass


# --- lazy torch import ---------------------------------------------------
# Tests patch `sys.modules["torch"]` with MagicMock. We import torch lazily
# inside __init__ so the module loads without torch installed, and we expose
# the base class via `torch.utils.data.Dataset` only when torch is real.

_torch = None


def _get_torch() -> Any:
    """Return the real `torch` module. Imported lazily so test mocks win."""
    global _torch
    if _torch is None:
        import torch  # type: ignore[import-not-found]

        _torch = torch
    return _torch


VALID_ROLES: tuple[str, ...] = ("system", "user", "assistant", "tool")
CHATML_HEADER = "<|im_start|>"
CHATML_FOOTER = "<|im_end|>"
CHATML_TEMPLATE = "{header}{role}\n{content}{footer}\n"

logger = logging.getLogger(__name__)


# --- helpers --------------------------------------------------------------


def messages_to_text(messages: list[dict[str, str]]) -> str:
    """Render ChatML messages to a single text block.

    Each turn is emitted as ``<|im_start|>{role}\\n{content}<|im_end|>\\n``.
    Roles outside :data:`VALID_ROLES` raise ``ValueError`` to surface prompt
    injection early.
    """
    if not isinstance(messages, list):
        raise ValueError("messages must be a list of dicts")
    if not messages:
        raise ValueError("messages must not be empty")
    parts: list[str] = []
    for msg in messages:
        if not isinstance(msg, dict):
            raise ValueError("each message must be a dict")
        role = msg.get("role")
        content = msg.get("content")
        if role not in VALID_ROLES:
            raise ValueError(f"invalid role: {role!r}")
        if not isinstance(content, str) or not content:
            raise ValueError("content must be a non-empty string")
        parts.append(CHATML_TEMPLATE.format(header=CHATML_HEADER, role=role, content=content, footer=CHATML_FOOTER))
    return "".join(parts)


def validate_pal_sft_record(record: Any) -> bool:
    """Return ``True`` if *record* matches the PAL SFT schema.

    Required keys: ``messages`` (non-empty list of ChatML dicts).
    Optional: ``metadata`` (dict). Records missing the required shape are
    silently dropped by the loader but logged at WARNING.
    """
    if not isinstance(record, dict):
        return False
    messages = record.get("messages")
    if not isinstance(messages, list) or not messages:
        return False
    for msg in messages:
        if not isinstance(msg, dict):
            return False
        if msg.get("role") not in VALID_ROLES:
            return False
        content = msg.get("content")
        if not isinstance(content, str) or not content:
            return False
    return True


def validate_pal_dpo_record(record: Any) -> bool:
    """Return ``True`` if *record* matches the PAL DPO schema."""
    if not isinstance(record, dict):
        return False
    if not isinstance(record.get("prompt"), str) or not record["prompt"]:
        return False
    for key in ("chosen", "rejected"):
        turns = record.get(key)
        if not isinstance(turns, list) or not turns:
            return False
        for msg in turns:
            if not isinstance(msg, dict):
                return False
            if msg.get("role") not in VALID_ROLES:
                return False
            content = msg.get("content")
            if not isinstance(content, str) or not content:
                return False
    return True


def _iter_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as fh:
        for lineno, raw in enumerate(fh, start=1):
            line = raw.strip()
            if not line:
                continue
            try:
                yield json.loads(line)
            except json.JSONDecodeError as exc:
                logger.warning("skipping malformed JSON at %s:%d: %s", path, lineno, exc)


# --- datasets -------------------------------------------------------------


class PalSftDataset:
    """SFT dataset over PAL ChatML JSONL.

    Each item returns ``{"input_ids", "attention_mask", "labels"}`` where
    non-assistant turns are masked in ``labels`` with ``-100`` so the loss is
    only computed on assistant responses.

    The tokenizer is required at construction time. Tests pass a stub
    tokenizer whose ``__call__`` returns deterministic tensors; in production
    a real HuggingFace tokenizer is used.
    """

    def __init__(self, path: str | Path, tokenizer: Any, max_length: int = 1024) -> None:
        torch = _get_torch()
        # inherit from torch.utils.data.Dataset at runtime
        self._torch = torch
        self.path = Path(path)
        self.tokenizer = tokenizer
        self.max_length = int(max_length)
        self.records: list[dict[str, Any]] = []
        dropped = 0
        for record in _iter_jsonl(self.path):
            if validate_pal_sft_record(record):
                self.records.append(record)
            else:
                dropped += 1
        if dropped:
            logger.warning("dropped %d invalid SFT records from %s", dropped, self.path)
        if not self.records:
            raise ValueError(f"no valid PAL SFT records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        record = self.records[idx]
        messages = record["messages"]
        text = messages_to_text(messages)
        # tokenize the full conversation
        enc = self.tokenizer(
            text,
            truncation=True,
            max_length=self.max_length,
            return_tensors="pt",
        )
        input_ids = enc["input_ids"].squeeze(0)
        attention_mask = enc["attention_mask"].squeeze(0)
        # build labels: mask all turns except assistant
        labels = self._build_labels(messages, input_ids)
        return {
            "input_ids": input_ids,
            "attention_mask": attention_mask,
            "labels": labels,
        }

    def _build_labels(self, messages: list[dict[str, str]], input_ids: Any) -> Any:
        torch = self._torch
        labels = torch.full_like(input_ids, -100)
        # walk through messages, rendering each turn and masking
        cursor = 0
        for msg in messages:
            turn_text = CHATML_TEMPLATE.format(
                header=CHATML_HEADER,
                role=msg["role"],
                content=msg["content"],
                footer=CHATML_FOOTER,
            )
            # tokenize just this turn (no special tokens, no truncation)
            enc = self.tokenizer(turn_text, add_special_tokens=False, return_tensors="pt")
            turn_ids = enc["input_ids"].squeeze(0)
            turn_len = int(turn_ids.shape[0])
            if msg["role"] == "assistant":
                end = min(cursor + turn_len, int(input_ids.shape[0]))
                start = min(cursor, end)
                labels[start:end] = input_ids[start:end]
            cursor += turn_len
        return labels


class PalDpoDataset:
    """DPO dataset over PAL DPO JSONL.

    Returns dicts of ``{"prompt": str, "chosen": list[dict], "rejected": list[dict]}``
    — the TRL conversational format. No tokenization is done here; TRL
    `DPOTrainer` handles chat-template rendering.
    """

    def __init__(self, path: str | Path) -> None:
        torch = _get_torch()
        self._torch = torch
        self.path = Path(path)
        self.records: list[dict[str, Any]] = []
        dropped = 0
        for record in _iter_jsonl(self.path):
            if validate_pal_dpo_record(record):
                clean = {
                    "prompt": record["prompt"],
                    "chosen": record["chosen"],
                    "rejected": record["rejected"],
                }
                if isinstance(record.get("metadata"), dict):
                    clean["metadata"] = record["metadata"]
                self.records.append(clean)
            else:
                dropped += 1
        if dropped:
            logger.warning("dropped %d invalid DPO records from %s", dropped, self.path)
        if not self.records:
            raise ValueError(f"no valid PAL DPO records found in {self.path}")

    def __len__(self) -> int:
        return len(self.records)

    def __getitem__(self, idx: int) -> dict[str, Any]:
        return self.records[idx]

    def to_list(self) -> list[dict[str, Any]]:
        """Return the records as a plain list for `datasets.Dataset.from_list`."""
        return list(self.records)


def load_pal_sft_dataset(path: str | Path, tokenizer: Any, max_length: int = 1024) -> PalSftDataset:
    """Build a :class:`PalSftDataset` (thin factory for test ergonomics)."""
    return PalSftDataset(path, tokenizer, max_length=max_length)


def load_pal_dpo_dataset(path: str | Path) -> PalDpoDataset:
    """Build a :class:`PalDpoDataset` (thin factory for test ergonomics)."""
    return PalDpoDataset(path)
