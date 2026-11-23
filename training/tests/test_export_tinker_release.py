"""Tests for the Tinker datum export."""

from __future__ import annotations

import hashlib
import json
import re
from pathlib import Path
from typing import Any

import pytest

from training import export_fireworks_release as fireworks, export_tinker_release as tinker
from training.export_fireworks_release import SelectionOptions

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


def test_export_trims_oldest_turns_instead_of_dropping_the_row(tmp_path: Path) -> None:
    tokenizer = StubTokenizer()
    history: list[dict[str, Any]] = []
    for index in range(6):
        history.append({"role": "user", "content": f"older prompt {index} " + "x " * 40})
        history.append({"role": "assistant", "content": f"older reply {index} " + "y " * 40})
    history.append({"role": "user", "content": CLIENT_PROMPT})
    history.append({"role": "assistant", "content": ASSISTANT_REPLY})
    manifest_path = write_manifest(tmp_path, [{"messages": history}])
    full_ids, _ = tinker.build_tokens(tokenizer, history)

    result = tinker.export(
        manifest_path,
        tmp_path / "out",
        tokenizer=tokenizer,
        max_tokens=len(full_ids) - 1,
    )

    assert result.counts["datums"] == 1
    assert result.counts["datums_left_truncated"] == 1
    assert result.counts["datums_skipped_too_long"] == 0
    assert "Call 988 now." in tokenizer.decode(result.rows[0].target_ids)


def test_export_still_drops_a_row_that_cannot_fit_after_truncation(tmp_path: Path) -> None:
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

    result = tinker.export(
        manifest_path,
        tmp_path / "out",
        tokenizer=StubTokenizer(),
        max_tokens=2,
        final_exchange_tokens=2,
    )

    assert result.counts["datums"] == 0
    assert result.counts["datums_skipped_too_long"] == 1


def test_export_keeps_a_final_exchange_that_exceeds_the_history_budget(tmp_path: Path) -> None:
    history = [
        {"role": "user", "content": "older prompt " + "x " * 40},
        {"role": "assistant", "content": "older reply " + "y " * 40},
    ]
    final = messages(("user", " ".join(["detail"] * 40)), ("assistant", ASSISTANT_REPLY))
    manifest_path = write_manifest(tmp_path, [{"messages": history + final}])

    result = tinker.export(manifest_path, tmp_path / "out", tokenizer=StubTokenizer(), max_tokens=10)

    assert result.counts["datums"] == 1
    assert result.counts["datums_left_truncated"] == 1
    assert result.counts["datums_skipped_too_long"] == 0


def test_export_keeps_a_standalone_exchange_under_a_larger_final_exchange_budget(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path, [{"messages": messages(("user", CLIENT_PROMPT), ("assistant", ASSISTANT_REPLY))}]
    )

    result = tinker.export(
        manifest_path,
        tmp_path / "out",
        tokenizer=StubTokenizer(),
        max_tokens=10,
        final_exchange_tokens=1000,
    )

    assert result.counts["datums"] == 1
    assert result.counts["datums_skipped_too_long"] == 0


def test_export_keeps_an_untrimmed_two_message_row_past_the_final_exchange_budget(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path, [{"messages": messages(("user", CLIENT_PROMPT), ("assistant", ASSISTANT_REPLY))}]
    )

    result = tinker.export(
        manifest_path,
        tmp_path / "out",
        tokenizer=StubTokenizer(),
        max_tokens=100,
        final_exchange_tokens=3,
    )

    assert result.counts["datums"] == 1
    assert result.counts["datums_left_truncated"] == 0
    assert result.counts["datums_skipped_too_long"] == 0


def test_export_drops_a_final_exchange_over_its_own_budget(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path,
        [{"messages": messages(("user", " ".join(["detail"] * 40)), ("assistant", ASSISTANT_REPLY))}],
    )

    result = tinker.export(
        manifest_path,
        tmp_path / "out",
        tokenizer=StubTokenizer(),
        max_tokens=10,
        final_exchange_tokens=10,
    )

    assert result.counts["datums"] == 0
    assert result.counts["datums_skipped_too_long"] == 1


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
                "partitions": {
                    "train": {"path": str(train_path), "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}
                },
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


@pytest.mark.parametrize("dry_run", [True, False])
@pytest.mark.parametrize("identity_problem", ["changed_payload", "missing_hash", "invalid_hash"])
def test_export_rejects_unverified_train_inputs(tmp_path: Path, dry_run: bool, identity_problem: str) -> None:
    manifest_path = write_manifest(
        tmp_path, [{"messages": messages(("user", "Fictional input."), ("assistant", "Fictional answer."))}]
    )
    manifest = json.loads(manifest_path.read_text())
    if identity_problem == "changed_payload":
        train_path = Path(manifest["partitions"]["train"]["path"])
        train_path.write_text(train_path.read_text().replace("Fictional input", "Different input"))
    elif identity_problem == "missing_hash":
        del manifest["partitions"]["train"]["sha256"]
    else:
        manifest["partitions"]["train"]["sha256"] = "invalid"
    manifest_path.write_text(json.dumps(manifest))
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=r"train.*SHA-256"):
        tinker.export(manifest_path, out, tokenizer=StubTokenizer(), dry_run=dry_run)

    assert not out.exists()


def test_export_receipt_binds_observed_train_and_manifest(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path, [{"messages": messages(("user", "Fictional input."), ("assistant", "Fictional answer."))}]
    )
    out = tmp_path / "out"

    tinker.export(manifest_path, out, tokenizer=StubTokenizer())

    record = json.loads((out / "tinker-dataset.json").read_bytes())
    assert record["source_train_sha256"] == hashlib.sha256((tmp_path / "train.jsonl").read_bytes()).hexdigest()
    assert record["source_manifest_sha256"] == hashlib.sha256(manifest_path.read_bytes()).hexdigest()
    assert record["selection_mode"] == "full_corpus"


def test_export_receipt_records_the_full_corpus_selection(tmp_path: Path) -> None:
    manifest_path = write_manifest(
        tmp_path, [{"messages": messages(("user", "Fictional input."), ("assistant", "Fictional answer."))}]
    )
    out = tmp_path / "out"

    tinker.export(manifest_path, out, tokenizer=StubTokenizer(), options=SelectionOptions.full_corpus())

    record = json.loads((out / "tinker-dataset.json").read_bytes())
    assert record["selection_mode"] == "full_corpus"


def test_export_rejects_train_changes_during_selection(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    manifest_path = write_manifest(
        tmp_path, [{"messages": messages(("user", "Fictional input."), ("assistant", "Fictional answer."))}]
    )
    original_select = tinker.select_for_export
    train_path = tmp_path / "train.jsonl"

    def mutate_after_select(*args: Any, **kwargs: Any) -> Any:
        result = original_select(*args, **kwargs)
        train_path.write_text(train_path.read_text().replace("Fictional input", "Different input"))
        return result

    monkeypatch.setattr(tinker, "select_for_export", mutate_after_select)
    out = tmp_path / "out"

    with pytest.raises(ValueError, match=r"train.*SHA-256"):
        tinker.export(manifest_path, out, tokenizer=StubTokenizer())

    assert not out.exists()


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


def test_export_limit_keeps_only_the_first_n_selected_rows(tmp_path: Path) -> None:
    """Smoke exports need a handful of real rows, not the whole release."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"{CLIENT_PROMPT} ({index})"},
                {"role": "assistant", "content": ASSISTANT_REPLY},
            ]
        }
        for index in range(3)
    ]
    manifest_path = write_manifest(tmp_path, rows)
    limit = 2

    whole = tinker.export(manifest_path, tmp_path / "all", tokenizer=StubTokenizer())
    limited = tinker.export(manifest_path, tmp_path / "some", tokenizer=StubTokenizer(), limit=limit)

    assert whole.counts["datums"] == len(rows)
    assert limited.counts["datums"] == limited.source_rows == limit


CRISIS_ROWS = 4
CRISIS_REPLY = "Call 911 now and say your head was banged against the wall. That needs a doctor tonight."


class StubCrisisBuilder:
    """A stand-in for ``training.build_agent_tool_sft_vNN``."""

    def __init__(self, replies: list[str]) -> None:
        self._replies = replies

    def collect_rows(self) -> dict[str, list[dict[str, Any]]]:
        return {
            "train": [
                {
                    "messages": [
                        {"role": "user", "content": f"disclosure {index}"},
                        {"role": "assistant", "content": reply},
                    ]
                }
                for index, reply in enumerate(self._replies)
            ]
        }


def write_manifest_with_sources(tmp_path: Path, rows: list[dict[str, Any]], sources: dict[str, int]) -> Path:
    train_path = tmp_path / "train.jsonl"
    with train_path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    manifest_path = tmp_path / "manifest.json"
    manifest_path.write_text(
        json.dumps(
            {
                "release_id": "stub-release",
                "partitions": {
                    "train": {"path": str(train_path), "sha256": hashlib.sha256(train_path.read_bytes()).hexdigest()}
                },
                "scope": {"appended_rows_by_source": sources},
            }
        ),
        encoding="utf-8",
    )
    return manifest_path


def test_crisis_reply_surface_is_empty_before_the_family_exists() -> None:
    """A v19-era manifest appended no crisis rows, so nothing needs keeping."""
    assert (
        tinker.crisis_reply_surface({"scope": {"appended_rows_by_source": {"agent_tool_synthetic_v9": 624}}})
        == frozenset()
    )
    assert tinker.crisis_reply_surface({"scope": {"appended_rows_by_source": {}}}) == frozenset()
    assert tinker.crisis_reply_surface({}) == frozenset()


def test_crisis_reply_surface_reads_the_builder_the_manifest_names(monkeypatch: Any) -> None:
    seen: list[str] = []

    def fake_import(name: str) -> StubCrisisBuilder:
        seen.append(name)
        return StubCrisisBuilder([CRISIS_REPLY, ASSISTANT_REPLY])

    monkeypatch.setattr(fireworks.importlib, "import_module", fake_import)

    surface = tinker.crisis_reply_surface(
        {"scope": {"appended_rows_by_source": {"crisis_direct_v29": 6504, "agent_tool_synthetic_v9": 624}}}
    )

    assert seen == ["training.build_agent_tool_sft_v29"]
    assert surface == frozenset({CRISIS_REPLY, ASSISTANT_REPLY})


def test_crisis_reply_surface_fails_loudly_on_a_missing_builder(monkeypatch: Any) -> None:
    def fake_import(name: str) -> StubCrisisBuilder:
        raise ImportError(f"no module named {name}")

    monkeypatch.setattr(fireworks.importlib, "import_module", fake_import)

    with pytest.raises(ValueError, match=r"training\.build_agent_tool_sft_v99 is not importable"):
        tinker.crisis_reply_surface({"scope": {"appended_rows_by_source": {"crisis_direct_v99": 1}}})


def test_full_selection_keeps_the_whole_partition() -> None:
    options = tinker.selection_options("full", seed=1, chat_sample=None, routing_cap=None, include_tool_rows=False)
    assert options.chat_sample is None
    assert options.routing_cap is None
    assert options.max_chars is None
    assert options.sample_max_chars is None


def test_full_selection_rejects_explicit_caps() -> None:
    with pytest.raises(ValueError, match="narrow"):
        tinker.selection_options("full", seed=1, chat_sample=10, routing_cap=None, include_tool_rows=False)
    with pytest.raises(ValueError, match="narrow"):
        tinker.selection_options("full", seed=1, chat_sample=None, routing_cap=5, include_tool_rows=False)


def test_narrow_selection_keeps_the_probe_caps() -> None:
    options = tinker.selection_options("narrow", seed=1, chat_sample=10, routing_cap=5, include_tool_rows=False)
    assert (options.chat_sample, options.routing_cap) == (10, 5)


def test_narrow_selection_defaults_to_the_probe_caps() -> None:
    options = tinker.selection_options("narrow", seed=1, chat_sample=None, routing_cap=None, include_tool_rows=False)
    assert (options.chat_sample, options.routing_cap) == (
        SelectionOptions.chat_sample,
        SelectionOptions.routing_cap,
    )


def test_export_parser_rejects_caps_with_the_full_selection() -> None:
    with pytest.raises(SystemExit):
        tinker._parse_args(["--chat-sample", "0"])
    with pytest.raises(SystemExit):
        tinker._parse_args(["--routing-cap", "1"])

    args = tinker._parse_args(["--selection", "narrow", "--chat-sample", "0", "--routing-cap", "1"])
    assert (args.chat_sample, args.routing_cap) == (0, 1)


def test_export_parser_defaults_to_the_full_partition() -> None:
    assert tinker._parse_args([]).selection == "full"


def test_export_keeps_the_crisis_family_whole_under_a_routing_cap(tmp_path: Path, monkeypatch: Any) -> None:
    """The regression: the family used to compete for the capped routing pool."""
    rows = [
        {
            "messages": [
                {"role": "user", "content": f"disclosure {index}"},
                {"role": "assistant", "content": CRISIS_REPLY},
            ]
        }
        for index in range(4)
    ]
    rows.append(
        {
            "messages": [
                {"role": "user", "content": "an ordinary routing row"},
                {"role": "assistant", "content": ASSISTANT_REPLY},
            ]
        }
    )
    manifest_path = write_manifest_with_sources(tmp_path, rows, {"crisis_direct_v29": len(rows) - 1})
    monkeypatch.setattr(tinker, "crisis_reply_surface", lambda _manifest: frozenset({CRISIS_REPLY}))

    result = tinker.export(
        manifest_path,
        tmp_path / "out",
        tokenizer=StubTokenizer(),
        options=SelectionOptions(routing_cap=0),
    )

    assert result.counts["crisis_rows"] == CRISIS_ROWS
    assert result.counts["routing_rows"] == 1
    assert result.counts["routing_kept"] == 0
    assert result.counts["datums"] == CRISIS_ROWS
