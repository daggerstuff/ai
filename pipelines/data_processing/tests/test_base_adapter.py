"""Tests for the dataset adapter base infrastructure.

Tests cover:
- BaseDatasetAdapter abstract contract (cannot instantiate directly)
- Concrete test adapter implementing all abstract methods
- validate() filtering logic
- run() orchestration (download -> extract -> convert -> validate -> save)
- _save_as_jsonl output format
- _build_provenance structure
- adapter_factory register_adapter / get_adapter / list_available_adapters
- converters: csv_to_chatml, json_conversation_to_chatml, sharegpt_to_chatml, save_jsonl
- validators: validate_record, filter_valid
"""

from __future__ import annotations

import csv
import json
from typing import Any

import pytest

from ai.pipelines.data_processing.dataset_adapters.adapter_factory import (
    get_adapter,
    list_available_adapters,
    register_adapter,
)
from ai.pipelines.data_processing.dataset_adapters.base_adapter import BaseDatasetAdapter
from ai.pipelines.data_processing.utils.converters import (
    csv_to_chatml,
    json_conversation_to_chatml,
    save_jsonl,
    sharegpt_to_chatml,
)
from ai.pipelines.data_processing.utils.validators import filter_valid, validate_record

# ---------------------------------------------------------------------------
# Concrete test adapter for testing the base class
# ---------------------------------------------------------------------------


@register_adapter("test_dummy")
class _TestDummyAdapter(BaseDatasetAdapter):
    """Minimal concrete adapter for testing base class behavior."""

    def download(self) -> None:
        # Simulate writing a raw file
        raw_file = self._raw_dir / "dummy.json"
        raw_file.write_text(
            json.dumps(
                [
                    {"user": "hello", "assistant": "hi there"},
                    {"user": "how are you", "assistant": "I'm well"},
                ]
            ),
            encoding="utf-8",
        )

    def extract(self) -> list[dict[str, Any]]:
        raw_file = self._raw_dir / "dummy.json"
        data = json.loads(raw_file.read_text(encoding="utf-8"))
        return data

    def convert_to_chatml(self, raw_data: list[dict[str, Any]]) -> list[dict[str, Any]]:
        records: list[dict[str, Any]] = []
        for item in raw_data:
            messages = [
                {"role": "user", "content": item["user"]},
                {"role": "assistant", "content": item["assistant"]},
            ]
            record = {
                "messages": messages,
                "source": "test_dummy",
                "task_type": "therapy_response_generation",
                "diagnostic_tag": None,
                "demographic_tags": [],
                "linguistic_style": "mixed",
                "clinical_reviewed": False,
                "provenance": self._build_provenance(
                    source_url="https://example.com/dummy",
                    access_method="direct_download",
                    original_format="json",
                ),
            }
            records.append(record)
        return records


# ---------------------------------------------------------------------------
# BaseDatasetAdapter tests
# ---------------------------------------------------------------------------


class TestBaseDatasetAdapter:
    """Tests for the abstract base class."""

    def test_cannot_instantiate_abstract_class(self):
        """BaseDatasetAdapter is abstract and cannot be instantiated."""
        with pytest.raises(TypeError):
            BaseDatasetAdapter("foo", "/tmp")  # type: ignore[abstract]

    def test_concrete_adapter_creates_directories(self, tmp_path):
        """Adapter __init__ creates output and raw directories."""
        adapter = _TestDummyAdapter("test_dummy", tmp_path)
        assert (tmp_path / "test_dummy").is_dir()
        assert (tmp_path / "test_dummy" / "raw").is_dir()

    def test_run_orchestration(self, tmp_path):
        """run() executes download -> extract -> convert -> validate -> save."""
        adapter = _TestDummyAdapter("test_dummy", tmp_path)
        output_path = adapter.run()

        assert output_path.exists()
        assert output_path.name == "test_dummy.jsonl"

        # Read the output and verify
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2

        first = json.loads(lines[0])
        assert "messages" in first
        assert first["messages"][0]["role"] == "user"
        assert first["messages"][1]["role"] == "assistant"
        assert first["source"] == "test_dummy"
        assert first["clinical_reviewed"] is False
        assert "provenance" in first
        assert first["provenance"]["source_url"] == "https://example.com/dummy"

    def test_build_provenance_structure(self, tmp_path):
        """_build_provenance returns a well-formed dict."""
        adapter = _TestDummyAdapter("test_dummy", tmp_path)
        prov = adapter._build_provenance(
            source_url="https://example.com",
            access_method="github",
            original_format="csv",
        )
        assert prov["source_url"] == "https://example.com"
        assert prov["access_method"] == "github"
        assert prov["original_format"] == "csv"
        assert "extracted_at" in prov
        assert isinstance(prov["transformations"], list)
        assert len(prov["transformations"]) == 4


# ---------------------------------------------------------------------------
# Adapter factory tests
# ---------------------------------------------------------------------------


class TestAdapterFactory:
    """Tests for the adapter factory."""

    def test_get_adapter_returns_instance(self, tmp_path):
        """get_adapter returns an adapter instance for a registered name."""
        adapter = get_adapter("test_dummy", tmp_path)
        assert isinstance(adapter, _TestDummyAdapter)
        assert adapter.dataset_name == "test_dummy"

    def test_get_adapter_case_insensitive(self, tmp_path):
        """Adapter lookup is case-insensitive."""
        adapter = get_adapter("TEST_DUMMY", tmp_path)
        assert isinstance(adapter, _TestDummyAdapter)

    def test_get_adapter_unknown_raises(self, tmp_path):
        """Unknown adapter name raises ValueError."""
        with pytest.raises(ValueError, match="No adapter for 'nonexistent'"):
            get_adapter("nonexistent", tmp_path)

    def test_list_available_adapters_includes_registered(self):
        """list_available_adapters returns sorted list including test adapter."""
        names = list_available_adapters()
        assert "test_dummy" in names


# ---------------------------------------------------------------------------
# Converter tests
# ---------------------------------------------------------------------------


class TestCsvToChatml:
    """Tests for csv_to_chatml converter."""

    def test_basic_conversion(self, tmp_path):
        """CSV with user/assistant columns converts to ChatML."""
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["user_input", "therapist_response"])
            writer.writeheader()
            writer.writerow({"user_input": "I feel sad", "therapist_response": "Tell me more"})
            writer.writerow({"user_input": "I can't sleep", "therapist_response": "Let's explore that"})

        records = csv_to_chatml(
            csv_path,
            user_column="user_input",
            assistant_column="therapist_response",
            source="test_csv",
        )
        assert len(records) == 2
        assert records[0]["messages"][0]["role"] == "user"
        assert records[0]["messages"][1]["role"] == "assistant"
        assert records[0]["source"] == "test_csv"

    def test_with_system_prompt(self, tmp_path):
        """System prompt is prepended as first message."""
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["u", "a"])
            writer.writeheader()
            writer.writerow({"u": "hi", "a": "hello"})

        records = csv_to_chatml(
            csv_path,
            user_column="u",
            assistant_column="a",
            system_prompt="You are a therapist.",
            source="test",
        )
        assert len(records) == 1
        assert records[0]["messages"][0]["role"] == "system"
        assert records[0]["messages"][0]["content"] == "You are a therapist."
        assert records[0]["messages"][1]["role"] == "user"

    def test_skips_empty_rows(self, tmp_path):
        """Rows with empty user or assistant content are skipped."""
        csv_path = tmp_path / "test.csv"
        with open(csv_path, "w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=["u", "a"])
            writer.writeheader()
            writer.writerow({"u": "valid", "a": "response"})
            writer.writerow({"u": "", "a": "no user"})
            writer.writerow({"u": "no assistant", "a": ""})

        records = csv_to_chatml(csv_path, user_column="u", assistant_column="a", source="test")
        assert len(records) == 1


class TestJsonConversationToChatml:
    """Tests for json_conversation_to_chatml converter."""

    def test_basic_conversion(self):
        """JSON conversations with dialog turns convert correctly."""
        conversations = [
            {
                "dialog": [
                    {"speaker": "seeker", "utterance": "I need help"},
                    {"speaker": "supporter", "utterance": "I'm here for you"},
                ]
            }
        ]
        records = json_conversation_to_chatml(conversations, source="test_json")
        assert len(records) == 1
        assert records[0]["messages"][0]["role"] == "user"
        assert records[0]["messages"][1]["role"] == "assistant"

    def test_skips_empty_conversations(self):
        """Conversations with no turns are skipped."""
        conversations = [{"dialog": []}, {}]
        records = json_conversation_to_chatml(conversations, source="test")
        assert len(records) == 0

    def test_custom_speaker_keys(self):
        """Custom speaker/utterance keys work."""
        conversations = [
            {
                "turns": [
                    {"role": "client", "text": "I'm anxious"},
                    {"role": "counselor", "text": "Let's work through this"},
                ]
            }
        ]
        records = json_conversation_to_chatml(
            conversations,
            speaker_key="role",
            utterance_key="text",
            user_role="client",
            assistant_role="counselor",
            source="test",
        )
        assert len(records) == 1
        assert records[0]["messages"][0]["content"] == "I'm anxious"


class TestSharegptToChatml:
    """Tests for sharegpt_to_chatml converter."""

    def test_basic_conversion(self):
        """ShareGPT format converts to ChatML."""
        conversations = [
            {
                "conversations": [
                    {"from": "human", "value": "What is CBT?"},
                    {"from": "gpt", "value": "Cognitive Behavioral Therapy is..."},
                ]
            }
        ]
        records = sharegpt_to_chatml(conversations, source="test_sgpt")
        assert len(records) == 1
        assert records[0]["messages"][0]["role"] == "user"
        assert records[0]["messages"][1]["role"] == "assistant"

    def test_skips_empty_turns(self):
        """Conversations with no turns are skipped."""
        conversations = [{"conversations": []}]
        records = sharegpt_to_chatml(conversations, source="test")
        assert len(records) == 0


class TestSaveJsonl:
    """Tests for save_jsonl utility."""

    def test_writes_jsonl_file(self, tmp_path):
        """save_jsonl writes records to a .jsonl file."""
        records = [
            {"messages": [{"role": "user", "content": "hi"}]},
            {"messages": [{"role": "user", "content": "bye"}]},
        ]
        output_path = tmp_path / "out" / "test.jsonl"
        count = save_jsonl(records, output_path)
        assert count == 2
        assert output_path.exists()
        lines = output_path.read_text(encoding="utf-8").strip().split("\n")
        assert len(lines) == 2
        assert json.loads(lines[0])["messages"][0]["content"] == "hi"

    def test_creates_parent_dir(self, tmp_path):
        """save_jsonl creates parent directories if needed."""
        records = [{"messages": [{"role": "user", "content": "test"}]}]
        output_path = tmp_path / "deeply" / "nested" / "out.jsonl"
        save_jsonl(records, output_path)
        assert output_path.exists()


# ---------------------------------------------------------------------------
# Validator tests
# ---------------------------------------------------------------------------


class TestValidateRecord:
    """Tests for validate_record."""

    def test_valid_record(self):
        """A well-formed record passes validation."""
        record = {
            "messages": [
                {"role": "user", "content": "I need help"},
                {"role": "assistant", "content": "I'm here"},
            ],
            "source": "test",
            "task_type": "therapy_response_generation",
            "clinical_reviewed": False,
        }
        assert validate_record(record) is True

    def test_missing_messages(self):
        """Record without messages fails."""
        assert validate_record({"source": "test"}) is False

    def test_short_messages(self):
        """Record with fewer than 2 messages fails."""
        assert validate_record({"messages": [{"role": "user", "content": "hi"}]}) is False

    def test_no_user_role(self):
        """Record without user role fails."""
        record = {
            "messages": [
                {"role": "assistant", "content": "hello"},
                {"role": "assistant", "content": "world"},
            ]
        }
        assert validate_record(record) is False

    def test_no_assistant_role(self):
        """Record without assistant role fails."""
        record = {
            "messages": [
                {"role": "user", "content": "hello"},
                {"role": "user", "content": "world"},
            ]
        }
        assert validate_record(record) is False

    def test_empty_content(self):
        """Messages with empty content fail."""
        record = {
            "messages": [
                {"role": "user", "content": ""},
                {"role": "assistant", "content": "valid"},
            ]
        }
        assert validate_record(record) is False

    def test_invalid_task_type(self):
        """Invalid task_type fails."""
        record = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "task_type": "nonexistent_task",
        }
        assert validate_record(record) is False

    def test_invalid_linguistic_style(self):
        """Invalid linguistic_style fails."""
        record = {
            "messages": [
                {"role": "user", "content": "hi"},
                {"role": "assistant", "content": "hello"},
            ],
            "linguistic_style": "purple",
        }
        assert validate_record(record) is False


class TestFilterValid:
    """Tests for filter_valid."""

    def test_filters_mixed_records(self):
        """filter_valid returns only valid records."""
        records = [
            {"messages": [{"role": "user", "content": "hi"}, {"role": "assistant", "content": "hello"}]},
            {"messages": [{"role": "user", "content": "only one"}]},  # invalid: <2 msgs
            {"messages": [{"role": "user", "content": "valid"}, {"role": "assistant", "content": "yes"}]},
        ]
        valid = filter_valid(records)
        assert len(valid) == 2
