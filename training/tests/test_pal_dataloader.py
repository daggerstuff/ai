"""Tests for `training.pal_dataloader`.

Uses real torch (installed in `ai/.venv`) but a stub tokenizer so no HuggingFace
download is needed. The stub mimics the `PreTrainedTokenizerBase.__call__`
contract closely enough for the dataset to mask assistant turns correctly.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

import pytest
import torch

from training.pal_dataloader import (
    CHATML_FOOTER,
    CHATML_HEADER,
    CHATML_TEMPLATE,
    PalDpoDataset,
    PalSftDataset,
    load_pal_dpo_dataset,
    load_pal_sft_dataset,
    messages_to_text,
    validate_pal_dpo_record,
    validate_pal_sft_record,
)


class StubTokenizer:
    """Deterministic char-hash tokenizer returning torch tensors."""

    def __init__(self, vocab_size: int = 1000) -> None:
        self.vocab_size = vocab_size

    def __call__(self, text: str, **kwargs: Any) -> dict[str, Any]:
        if not isinstance(text, str):
            raise TypeError("tokenizer expects str input")
        tokens = [ord(c) % self.vocab_size for c in text]
        return_tensors = kwargs.get("return_tensors")
        if return_tensors == "pt":
            ids = torch.tensor([tokens], dtype=torch.long)
            mask = torch.ones_like(ids)
            return {"input_ids": ids, "attention_mask": mask}
        return {"input_ids": list(tokens), "attention_mask": [1] * len(tokens)}


# --- helpers -------------------------------------------------------------


def _make_sft_jsonl(path: Path, n: int = 5) -> Path:
    records = [
        {
            "messages": [
                {"role": "system", "content": "You are a clinical assistant."},
                {"role": "user", "content": f"Hello number {i}"},
                {"role": "assistant", "content": f"Hi patient {i}"},
            ],
            "metadata": {"index": i},
        }
        for i in range(n)
    ]
    return _write_jsonl(path, records)


def _make_dpo_jsonl(path: Path, n: int = 5) -> Path:
    records = [
        {
            "prompt": f"Given this persona: P{i}\n\nDialogue history:\nhi\n\nGenerate the next response.",
            "chosen": [{"role": "assistant", "content": f"chosen response {i}"}],
            "rejected": [{"role": "assistant", "content": f"rejected response {i}"}],
            "metadata": {"persona_string": f"P{i}"},
        }
        for i in range(n)
    ]
    return _write_jsonl(path, records)


def _write_jsonl(path: Path, records: list[dict[str, Any]]) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as fh:
        for rec in records:
            fh.write(json.dumps(rec, ensure_ascii=False) + "\n")
    return path


# --- messages_to_text ----------------------------------------------------


class TestMessagesToText:
    def test_renders_chatml_template(self) -> None:
        out = messages_to_text(
            [
                {"role": "system", "content": "sys"},
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ]
        )
        assert CHATML_HEADER in out
        assert CHATML_FOOTER in out
        assert "system\nsys" in out
        assert "user\nhi" in out
        assert "assistant\nhello" in out
        assert out.endswith("\n")

    def test_rejects_empty(self) -> None:
        with pytest.raises(ValueError):
            messages_to_text([])

    def test_rejects_non_list(self) -> None:
        with pytest.raises(ValueError):
            messages_to_text("not a list")  # type: ignore[arg-type]

    def test_rejects_non_dict_message(self) -> None:
        with pytest.raises(ValueError):
            messages_to_text([{"role": "user", "content": "hi"}, "bad"])  # type: ignore[list-item]

    def test_rejects_invalid_role(self) -> None:
        with pytest.raises(ValueError):
            messages_to_text([{"role": "hacker", "content": "pwned"}])

    def test_rejects_empty_content(self) -> None:
        with pytest.raises(ValueError):
            messages_to_text([{"role": "user", "content": ""}])

    def test_rejects_non_string_content(self) -> None:
        with pytest.raises(ValueError):
            messages_to_text([{"role": "user", "content": 123}])  # type: ignore[dict-item]


# --- validate_pal_sft_record --------------------------------------------


class TestValidatePalSftRecord:
    def test_valid(self) -> None:
        rec = {"messages": [{"role": "system", "content": "sys"}, {"role": "assistant", "content": "hi"}]}
        assert validate_pal_sft_record(rec) is True

    def test_with_metadata(self) -> None:
        rec = {"messages": [{"role": "user", "content": "hi"}], "metadata": {"k": 1}}
        assert validate_pal_sft_record(rec) is True

    def test_non_dict(self) -> None:
        assert validate_pal_sft_record("not dict") is False  # type: ignore[arg-type]

    def test_missing_messages(self) -> None:
        assert validate_pal_sft_record({"metadata": {}}) is False

    def test_empty_messages(self) -> None:
        assert validate_pal_sft_record({"messages": []}) is False

    def test_bad_role(self) -> None:
        assert validate_pal_sft_record({"messages": [{"role": "bot", "content": "x"}]}) is False

    def test_non_string_content(self) -> None:
        assert validate_pal_sft_record({"messages": [{"role": "user", "content": 1}]}) is False  # type: ignore[dict-item]

    def test_non_dict_message(self) -> None:
        assert validate_pal_sft_record({"messages": ["not dict"]}) is False  # type: ignore[list-item]


# --- validate_pal_dpo_record --------------------------------------------


class TestValidatePalDpoRecord:
    def test_valid(self) -> None:
        rec = {
            "prompt": "p",
            "chosen": [{"role": "assistant", "content": "c"}],
            "rejected": [{"role": "assistant", "content": "r"}],
        }
        assert validate_pal_dpo_record(rec) is True

    def test_with_metadata(self) -> None:
        rec = {
            "prompt": "p",
            "chosen": [{"role": "assistant", "content": "c"}],
            "rejected": [{"role": "assistant", "content": "r"}],
            "metadata": {"persona": "P"},
        }
        assert validate_pal_dpo_record(rec) is True

    def test_non_dict(self) -> None:
        assert validate_pal_dpo_record([]) is False  # type: ignore[arg-type]

    def test_missing_prompt(self) -> None:
        assert validate_pal_dpo_record({"chosen": [], "rejected": []}) is False

    def test_empty_prompt(self) -> None:
        assert validate_pal_dpo_record({"prompt": "", "chosen": [], "rejected": []}) is False

    def test_missing_chosen(self) -> None:
        assert validate_pal_dpo_record({"prompt": "p", "rejected": []}) is False

    def test_empty_chosen(self) -> None:
        assert validate_pal_dpo_record({"prompt": "p", "chosen": [], "rejected": []}) is False

    def test_bad_role_in_chosen(self) -> None:
        rec = {
            "prompt": "p",
            "chosen": [{"role": "bot", "content": "x"}],
            "rejected": [{"role": "assistant", "content": "r"}],
        }
        assert validate_pal_dpo_record(rec) is False


# --- PalSftDataset ------------------------------------------------------


class TestPalSftDataset:
    def test_loads_valid_records(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=5)
        ds = PalSftDataset(path, StubTokenizer())
        assert len(ds) == 5

    def test_drops_invalid_records(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = tmp_path / "mixed.jsonl"
        records = [
            {"messages": [{"role": "user", "content": "good"}]},
            {"messages": []},
            {"messages": [{"role": "bot", "content": "bad role"}]},
            {"not_messages": "oops"},
            {"messages": [{"role": "assistant", "content": "ok"}]},
        ]
        _write_jsonl(path, records)
        with caplog.at_level(logging.WARNING):
            ds = PalSftDataset(path, StubTokenizer())
        assert len(ds) == 2
        assert "dropped 3 invalid" in caplog.text

    def test_raises_on_empty_file(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.touch()
        with pytest.raises(ValueError, match="no valid PAL SFT records"):
            PalSftDataset(path, StubTokenizer())

    def test_raises_on_missing_file(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PalSftDataset(tmp_path / "missing.jsonl", StubTokenizer())

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "blanks.jsonl"
        path.write_text(
            json.dumps({"messages": [{"role": "user", "content": "a"}]})
            + "\n\n\n"
            + json.dumps({"messages": [{"role": "user", "content": "b"}]})
            + "\n",
            encoding="utf-8",
        )
        ds = PalSftDataset(path, StubTokenizer())
        assert len(ds) == 2

    def test_getitem_returns_expected_keys(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=3)
        ds = PalSftDataset(path, StubTokenizer())
        item = ds[0]
        assert set(item.keys()) == {"input_ids", "attention_mask", "labels"}
        for key in item:
            assert isinstance(item[key], torch.Tensor)

    def test_getitem_labels_mask_non_assistant(self, tmp_path: Path) -> None:
        # build a record where the assistant turn is the last one
        path = tmp_path / "mask.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "messages": [
                        {"role": "system", "content": "sys"},
                        {"role": "user", "content": "q"},
                        {"role": "assistant", "content": "a"},
                    ]
                }
            ],
        )
        ds = PalSftDataset(path, StubTokenizer())
        item = ds[0]
        labels = item["labels"]
        input_ids = item["input_ids"]
        # assistant turn should have non -100 labels somewhere
        assert (labels != -100).any()
        # the masked positions should equal -100
        assert (labels[labels == -100]).numel() > 0
        # where labels != -100, they should match input_ids
        active = labels != -100
        assert torch.equal(labels[active], input_ids[active])

    def test_hundred_record_subset(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "big.jsonl", n=100)
        ds = PalSftDataset(path, StubTokenizer())
        assert len(ds) == 100
        # all items load without error
        for i in range(100):
            item = ds[i]
            assert item["input_ids"].shape[0] == item["attention_mask"].shape[0]

    def test_factory_load_pal_sft_dataset(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=2)
        ds = load_pal_sft_dataset(path, StubTokenizer(), max_length=512)
        assert isinstance(ds, PalSftDataset)
        assert ds.max_length == 512

    def test_preserves_unicode(self, tmp_path: Path) -> None:
        path = tmp_path / "uni.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "messages": [
                        {"role": "user", "content": "Xin chào 🇻🇳"},
                        {"role": "assistant", "content": "Chào bạn"},
                    ]
                }
            ],
        )
        ds = PalSftDataset(path, StubTokenizer())
        assert len(ds) == 1
        text = messages_to_text(ds.records[0]["messages"])
        assert "🇻🇳" in text
        assert "Chào" in text

    def test_no_json_leakage_in_text(self, tmp_path: Path) -> None:
        path = _make_sft_jsonl(tmp_path / "sft.jsonl", n=3)
        ds = PalSftDataset(path, StubTokenizer())
        for rec in ds.records:
            text = messages_to_text(rec["messages"])
            assert "{" not in text
            assert "}" not in text
            assert '"' not in text


# --- PalDpoDataset ------------------------------------------------------


class TestPalDpoDataset:
    def test_loads_valid_records(self, tmp_path: Path) -> None:
        path = _make_dpo_jsonl(tmp_path / "dpo.jsonl", n=5)
        ds = PalDpoDataset(path)
        assert len(ds) == 5

    def test_drops_invalid(self, tmp_path: Path, caplog: pytest.LogCaptureFixture) -> None:
        path = tmp_path / "mixed.jsonl"
        records = [
            {
                "prompt": "p1",
                "chosen": [{"role": "assistant", "content": "c"}],
                "rejected": [{"role": "assistant", "content": "r"}],
            },
            {"prompt": "", "chosen": [], "rejected": []},
            {"prompt": "p3", "chosen": [{"role": "bot", "content": "x"}], "rejected": []},
            {"not_pal": "oops"},
        ]
        _write_jsonl(path, records)
        with caplog.at_level(logging.WARNING):
            ds = PalDpoDataset(path)
        assert len(ds) == 1
        assert "dropped 3 invalid" in caplog.text

    def test_preserves_metadata(self, tmp_path: Path) -> None:
        path = tmp_path / "meta.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "prompt": "p",
                    "chosen": [{"role": "assistant", "content": "c"}],
                    "rejected": [{"role": "assistant", "content": "r"}],
                    "metadata": {"persona": "P", "tokens": 42},
                }
            ],
        )
        ds = PalDpoDataset(path)
        item = ds[0]
        assert item["metadata"] == {"persona": "P", "tokens": 42}

    def test_raises_on_empty(self, tmp_path: Path) -> None:
        path = tmp_path / "empty.jsonl"
        path.touch()
        with pytest.raises(ValueError, match="no valid PAL DPO records"):
            PalDpoDataset(path)

    def test_raises_on_missing(self, tmp_path: Path) -> None:
        with pytest.raises(FileNotFoundError):
            PalDpoDataset(tmp_path / "missing.jsonl")

    def test_skips_blank_lines(self, tmp_path: Path) -> None:
        path = tmp_path / "blanks.jsonl"
        path.write_text(
            json.dumps(
                {
                    "prompt": "p1",
                    "chosen": [{"role": "assistant", "content": "c"}],
                    "rejected": [{"role": "assistant", "content": "r"}],
                }
            )
            + "\n\n\n"
            + json.dumps(
                {
                    "prompt": "p2",
                    "chosen": [{"role": "assistant", "content": "c2"}],
                    "rejected": [{"role": "assistant", "content": "r2"}],
                }
            )
            + "\n",
            encoding="utf-8",
        )
        ds = PalDpoDataset(path)
        assert len(ds) == 2

    def test_to_list_roundtrip(self, tmp_path: Path) -> None:
        path = _make_dpo_jsonl(tmp_path / "dpo.jsonl", n=3)
        ds = PalDpoDataset(path)
        lst = ds.to_list()
        assert len(lst) == 3
        assert all("prompt" in r and "chosen" in r and "rejected" in r for r in lst)

    def test_factory_load_pal_dpo_dataset(self, tmp_path: Path) -> None:
        path = _make_dpo_jsonl(tmp_path / "dpo.jsonl", n=2)
        ds = load_pal_dpo_dataset(path)
        assert isinstance(ds, PalDpoDataset)

    def test_hundred_record_subset(self, tmp_path: Path) -> None:
        path = _make_dpo_jsonl(tmp_path / "big.jsonl", n=100)
        ds = PalDpoDataset(path)
        assert len(ds) == 100
        for i in range(100):
            item = ds[i]
            assert isinstance(item["prompt"], str)
            assert isinstance(item["chosen"], list)
            assert isinstance(item["rejected"], list)

    def test_preserves_unicode(self, tmp_path: Path) -> None:
        path = tmp_path / "uni.jsonl"
        _write_jsonl(
            path,
            [
                {
                    "prompt": "Xin chào 🇻🇳",
                    "chosen": [{"role": "assistant", "content": "Chào bạn 🇻🇳"}],
                    "rejected": [{"role": "assistant", "content": "Hi there"}],
                }
            ],
        )
        ds = PalDpoDataset(path)
        assert len(ds) == 1
        assert "🇻🇳" in ds[0]["prompt"]
        assert "🇻🇳" in ds[0]["chosen"][0]["content"]
