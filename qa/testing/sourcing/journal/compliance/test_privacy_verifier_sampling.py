"""Tests for _sample_from_file in PrivacyVerifier."""

import csv
import json
import os
import tempfile
from pathlib import Path

from ai.pipelines.data_processing.journal.compliance.privacy_verifier import PrivacyVerifier


class TestSampleFromFile:
    def test_sample_json(self) -> None:
        data = [{"text": f"item {i}"} for i in range(200)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            json.dump(data, f)
            f.flush()
            result = PrivacyVerifier()._sample_from_file(f.name)
            assert result != ""
            assert "item" in result
            os.unlink(f.name)

    def test_sample_jsonl(self) -> None:
        lines = [json.dumps({"text": f"line {i}"}) for i in range(200)]
        with tempfile.NamedTemporaryFile(mode="w", suffix=".jsonl", delete=False) as f:
            f.write("\n".join(lines))
            f.flush()
            result = PrivacyVerifier()._sample_from_file(f.name)
            assert result != ""
            assert "line" in result
            os.unlink(f.name)

    def test_sample_csv(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".csv", delete=False, newline="") as f:
            writer = csv.DictWriter(f, fieldnames=["name", "value"])
            writer.writeheader()
            for i in range(200):
                writer.writerow({"name": f"name{i}", "value": f"val{i}"})
            f.flush()
            result = PrivacyVerifier()._sample_from_file(f.name)
            assert result != ""
            assert "name" in result
            os.unlink(f.name)

    def test_sample_txt(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False) as f:
            f.write("\n".join(f"line {i}" for i in range(200)))
            f.flush()
            result = PrivacyVerifier()._sample_from_file(f.name)
            assert result != ""
            assert "line" in result
            os.unlink(f.name)

    def test_nonexistent_file(self) -> None:
        result = PrivacyVerifier()._sample_from_file("/nonexistent/file.json")
        assert result == ""

    def test_empty_json(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("[]")
            f.flush()
            result = PrivacyVerifier()._sample_from_file(f.name)
            assert result == ""
            os.unlink(f.name)

    def test_invalid_json(self) -> None:
        with tempfile.NamedTemporaryFile(mode="w", suffix=".json", delete=False) as f:
            f.write("{invalid json")
            f.flush()
            result = PrivacyVerifier()._sample_from_file(f.name)
            assert result == ""
            os.unlink(f.name)
