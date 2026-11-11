"""Tests for the Tinker datum export."""

from __future__ import annotations

import json
import re
from pathlib import Path
from typing import Any

from training import export_tinker_release as tinker

TOKEN_RE = re.compile(r"\S+")
ASSISTANT_REPLY = "Call 988 now."
CLIENT_PROMPT = "I have been thinking about ending it."


class StubTokenizer:
    """A whitespace tokenizer over a minimal ChatML render.

    It exists so the masking tests do not download a real tokenizer. Rendering
    and tokenization are kept separate on purpose: that is the contract
    ``build_tokens`` relies on when it maps character offsets to token weights.
    """

    name_or_path = "stub-chatml"

    def __init__(self) -> None:
        self._ids: dict[str, int] = {}

    def _id(self, token: str) -> int:
        if token not in self._ids:
            self._ids[token] = len(self._ids)
        return self._ids[token]

    def _tokenize(self, text: str) -> tuple[list[int], list[tuple[int, int]]]:
        ids: list[int] = []
        offsets: list[tuple[int, int]] = []
        for match in TOKEN_RE.finditer(text):
            ids.append(self._id(match.group()))
            offsets.append((match.start(), match.end()))
        return ids, offsets

    def render(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = False,
    ) -> str:
        parts = [f"<|{message['role']}|>\n{message['content']}\n<|im_end|>\n" for message in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)

    def apply_chat_template(
        self,
        messages: list[dict[str, Any]],
        tokenize: bool = True,
        add_generation_prompt: bool = False,
        **_: Any,
    ) -> Any:
        text = self.render(messages, add_generation_prompt=add_generation_prompt)
        if not tokenize:
            return text
        return self._tokenize(text)[0]

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        ids, offsets = self._tokenize(text)
        payload: dict[str, Any] = {"input_ids": ids}
        if kwargs.get("return_offsets_mapping"):
            payload["offset_mapping"] = offsets
        return payload

    def decode(self, ids: list[int]) -> str:
        reverse = {value: key for key, value in self._ids.items()}
        return " ".join(reverse.get(token, "?") for token in ids)


class TrimmingTokenizer(StubTokenizer):
    """A ChatML render that trims content, like Qwen3.5's ``|trim``.

    The release tokenizer's template trims every message, so a mask measured
    against untrimmed text would not line up with what the model actually sees.
    """

    name_or_path = "stub-chatml-trim"

    def render(
        self,
        messages: list[dict[str, Any]],
        add_generation_prompt: bool = False,
    ) -> str:
        parts = [f"<|{message['role']}|>\n{message['content'].strip()}\n<|im_end|>\n" for message in messages]
        if add_generation_prompt:
            parts.append("<|assistant|>\n")
        return "".join(parts)


def messages(*turns: tuple[str, str]) -> list[dict[str, Any]]:
    """Build a release payload, weighting assistant turns for training."""
    payload: list[dict[str, Any]] = []
    for role, content in turns:
        entry: dict[str, Any] = {"role": role, "content": content}
        if role != "system":
            entry["weight"] = 0 if role == "user" else 1
        payload.append(entry)
    return payload


def trained_text(tokenizer: StubTokenizer, ids: list[int], weights: list[int]) -> list[str]:
    reverse = {value: key for key, value in tokenizer._ids.items()}
    return [reverse[token] for token, weight in zip(ids, weights, strict=True) if weight]


def test_weights_cover_assistant_content_and_end_token_only() -> None:
    tokenizer = StubTokenizer()
    payload = messages(
        ("system", "You are a therapist."),
        ("user", CLIENT_PROMPT),
        ("assistant", ASSISTANT_REPLY),
    )

    ids, weights = tinker.build_tokens(tokenizer, payload)

    assert trained_text(tokenizer, ids, weights) == ["Call", "988", "now.", "<|im_end|>"]
    assert len(ids) == len(weights)


def test_role_headers_are_never_trained() -> None:
    tokenizer = StubTokenizer()
    ids, weights = tinker.build_tokens(
        tokenizer,
        messages(("user", CLIENT_PROMPT), ("assistant", ASSISTANT_REPLY)),
    )

    reverse = {value: key for key, value in tokenizer._ids.items()}
    for token, weight in zip(ids, weights, strict=True):
        if reverse[token].startswith("<|") and reverse[token] != "<|im_end|>":
            assert weight == 0, f"{reverse[token]} is a role header and must be masked"


def test_user_turn_is_masked() -> None:
    tokenizer = StubTokenizer()
    ids, weights = tinker.build_tokens(
        tokenizer,
        messages(("user", CLIENT_PROMPT), ("assistant", ASSISTANT_REPLY)),
    )

    reverse = {value: key for key, value in tokenizer._ids.items()}
    for token, weight in zip(ids, weights, strict=True):
        if reverse[token] in {"thinking", "about", "ending"}:
            assert weight == 0


def test_leaked_reasoning_is_stripped_from_the_trained_text() -> None:
    tokenizer = StubTokenizer()
    payload = messages(
        ("user", CLIENT_PROMPT),
        ("assistant", "draft thought\n</think>\nCall 988 now."),
    )

    ids, weights = tinker.build_tokens(tokenizer, payload)

    assert trained_text(tokenizer, ids, weights) == ["Call", "988", "now.", "<|im_end|>"]
    assert "draft" not in tokenizer.decode(ids)


def test_whitespace_is_trimmed_the_way_the_template_trims_it() -> None:
    tokenizer = TrimmingTokenizer()
    payload = messages(
        ("user", CLIENT_PROMPT),
        ("assistant", "  Call 988 now.  "),
    )

    ids, weights = tinker.build_tokens(tokenizer, payload)

    assert trained_text(tokenizer, ids, weights) == ["Call", "988", "now.", "<|im_end|>"]


def test_datum_shifts_targets_against_the_mask() -> None:
    tokenizer = StubTokenizer()
    payload = messages(("user", CLIENT_PROMPT), ("assistant", ASSISTANT_REPLY))

    datum = tinker.build_datum(tokenizer, payload)
    assert datum is not None
    ids, weights = tinker.build_tokens(tokenizer, payload)

    assert datum.input_ids == ids[:-1]
    assert datum.target_ids == ids[1:]
    assert datum.weights == weights[1:]
    assert datum.trained_tokens == 3 + 1


def test_datum_line_is_the_tinker_payload_shape() -> None:
    tokenizer = StubTokenizer()
    datum = tinker.build_datum(
        tokenizer,
        messages(("user", CLIENT_PROMPT), ("assistant", ASSISTANT_REPLY)),
    )
    assert datum is not None

    payload = json.loads(datum.line())

    assert set(payload) == {"input_ids", "target_ids", "weights"}
    assert len(payload["input_ids"]) == len(payload["target_ids"]) == len(payload["weights"])


def test_datum_is_dropped_when_it_exceeds_the_token_budget() -> None:
    tokenizer = StubTokenizer()
    payload = messages(("user", CLIENT_PROMPT), ("assistant", ASSISTANT_REPLY))

    assert tinker.build_datum(tokenizer, payload, max_tokens=2) is None


def write_manifest(tmp_path: Path, rows: list[dict[str, Any]]) -> Path:
    train_path = tmp_path / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "release_id": "stub-release",
                "partitions": {"train": {"path": str(train_path), "sha256": "0" * 64}},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_export_trims_a_therapist_opening_turn_instead_of_dropping_the_row(
    tmp_path: Path,
) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [
            {
                "messages": [
                    {"role": "assistant", "content": "What brings you in today?"},
                    {"role": "user", "content": CLIENT_PROMPT},
                    {"role": "assistant", "content": ASSISTANT_REPLY},
                ]
            }
        ],
    )

    result = tinker.export(manifest_path, tmp_path / "out", tokenizer=StubTokenizer())

    assert result.counts["datums"] == 1
    assert result.counts["datums_trimmed_opening"] == 1
    assert (tmp_path / "out" / "tinker-sft-train.jsonl").exists()


def test_a_dry_run_leaves_existing_export_files_alone(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [
            {
                "messages": [
                    {"role": "user", "content": CLIENT_PROMPT},
                    {"role": "assistant", "content": ASSISTANT_REPLY},
                ]
            }
        ],
    )
    out = tmp_path / "out"
    out.mkdir()
    dataset = out / "tinker-sft-train.jsonl"
    record = out / "tinker-dataset.json"
    dataset.write_text("previous dataset\n", encoding="utf-8")
    record.write_text('{"previous": true}\n', encoding="utf-8")

    result = tinker.export(manifest_path, out, tokenizer=StubTokenizer(), dry_run=True)

    assert result.counts["datums"] == 1
    assert dataset.read_text(encoding="utf-8") == "previous dataset\n"
    assert record.read_text(encoding="utf-8") == '{"previous": true}\n'


def test_export_writes_one_jsonl_line_per_datum(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [
            {
                "messages": [
                    {"role": "user", "content": CLIENT_PROMPT},
                    {"role": "assistant", "content": ASSISTANT_REPLY},
                ]
            }
        ],
    )

    result = tinker.export(manifest_path, tmp_path / "out", tokenizer=StubTokenizer())

    path = tmp_path / "out" / "tinker-sft-train.jsonl"
    lines = path.read_text(encoding="utf-8").splitlines()

    assert len(lines) == result.counts["datums"] == 1
    record = json.loads((tmp_path / "out" / "tinker-dataset.json").read_text())
    assert record["datums"] == 1
    assert record["counts"]["trained_tokens"] == result.counts["trained_tokens"]
