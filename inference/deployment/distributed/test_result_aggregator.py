#!/usr/bin/env python3
"""
Unit tests for the Result Aggregator CLI report command
"""

import contextlib
import io
import shutil
import tempfile
import unittest
from pathlib import Path
from unittest import mock

from result_aggregator import ResultAggregator, main


def make_result(success=True, quality_score=0.95, worker_id="worker-1"):
    return {
        "task_id": "task-1",
        "file_path": "sample.txt",
        "validation_type": "conversation",
        "success": success,
        "quality_score": quality_score,
        "metrics": {"latency": 0.4},
        "issues": [],
        "processing_time": 1.5,
        "worker_id": worker_id,
        "timestamp": "2026-01-01T00:00:00+00:00",
    }


class TestReportCommand(unittest.TestCase):
    """The report command must surface the generated report"""

    def setUp(self):
        self.temp_dir = Path(tempfile.mkdtemp())
        self.db_path = str(self.temp_dir / "results.db")
        self.batch_id = "batch-1"

        aggregator = ResultAggregator(self.db_path)
        aggregator.add_result(self.batch_id, make_result())
        aggregator.add_result(self.batch_id, make_result(success=False, quality_score=0.4, worker_id="worker-2"))
        assert aggregator.aggregate_results(self.batch_id) is not None

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def _run_cli(self, *argv):
        stdout = io.StringIO()
        with mock.patch("sys.argv", ["result_aggregator", *argv]), contextlib.redirect_stdout(stdout):
            main()
        return stdout.getvalue()

    def test_report_without_output_prints_report_to_stdout(self):
        stdout = self._run_cli("--db-path", self.db_path, "report", self.batch_id)
        assert f"Quality Validation Report - Batch {self.batch_id}" in stdout
        assert "SUMMARY" in stdout
        assert "Total Files: 2" in stdout

    def test_report_with_output_file_keeps_stdout_silent(self):
        output_file = self.temp_dir / "report.txt"
        stdout = self._run_cli("--db-path", self.db_path, "report", self.batch_id, "--output", str(output_file))
        assert stdout == ""
        written = output_file.read_text()
        assert f"Quality Validation Report - Batch {self.batch_id}" in written

    def test_report_for_missing_batch_prints_notice_to_stdout(self):
        stdout = self._run_cli("--db-path", self.db_path, "report", "missing-batch")
        assert "No data found for batch: missing-batch" in stdout

    def test_report_with_output_and_missing_batch_keeps_notice_off_stdout(self):
        output_file = self.temp_dir / "missing-batch.txt"
        stdout, stderr = io.StringIO(), io.StringIO()
        with mock.patch("sys.argv", ["result_aggregator", "--db-path", self.db_path, "report", "missing-batch",
                                     "--output", str(output_file)]), contextlib.redirect_stdout(stdout), \
                contextlib.redirect_stderr(stderr):
            main()
        assert stdout.getvalue() == ""
        assert "No data found for batch: missing-batch" in stderr.getvalue()
        assert not output_file.exists()

