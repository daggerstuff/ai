#!/usr/bin/env python3
"""
Unit tests for the error handling auditor system.
"""

import os
import tempfile

import pytest

from .error_handling_auditor import (
    ErrorHandlingAuditor,
    ErrorHandlingAuditReport,
    ErrorHandlingIssue,
)
from .pathlib import Path


class TestErrorHandlingAuditor:
    """Test cases for the ErrorHandlingAuditor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = ErrorHandlingAuditor()

    def test_auditor_initialization(self):
        """Test auditor initialization."""
        auditor = ErrorHandlingAuditor()
        assert "file_operations" in auditor.risky_operations
        assert "open" in auditor.risky_operations["file_operations"]
        assert "FileNotFoundError" in auditor.common_exceptions
        assert auditor.severity_weights["critical"] == 1.0

    def test_audit_file_with_good_error_handling(self):
        """Test auditing a file with good error handling."""
        good_code = '''
"""Module with good error handling."""
import json

def safe_file_read(filename):
    """Function with proper error handling."""
    try:
        with open(filename, 'r') as f:
            content = f.read()
        return content
    except FileNotFoundError:
        print(f"File not found: {filename}")
        return None
    except IOError as e:
        print(f"I/O error: {e}")
        return None

def safe_json_parse(data):
    """Function with JSON error handling."""
    try:
        return json.loads(data)
    except json.JSONDecodeError as e:
        print(f"JSON decode error: {e}")
        return None
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(good_code)
            temp_file = f.name

        try:
            stats = self.auditor._audit_file(temp_file)

            assert stats["functions"] == 2
            assert stats["functions_with_error_handling"] == 2
            assert stats["risky_operations"] >= 2  # open and json.loads
            assert stats["protected_operations"] >= 1  # At least some operations are protected

        finally:
            os.unlink(temp_file)

    def test_audit_file_with_poor_error_handling(self):
        """Test auditing a file with poor error handling."""
        bad_code = '''
def unsafe_file_read(filename):
    """Function without error handling."""
    with open(filename, 'r') as f:
        content = f.read()
    return content

def function_with_bare_except():
    """Function with bare except."""
    try:
        risky_operation()
    except:
        pass

def function_with_broad_except():
    """Function with broad exception handling."""
    try:
        another_risky_operation()
    except Exception:
        pass
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(bad_code)
            temp_file = f.name

        try:
            # Clear previous issues
            self.auditor.issues = []
            stats = self.auditor._audit_file(temp_file)

            assert stats["functions"] == 3
            assert stats["functions_with_error_handling"] == 2  # Two functions have try-catch

            # Should have issues for poor error handling
            issues = self.auditor.issues
            issue_types = [issue.issue_type for issue in issues]

            assert "unprotected_operation" in issue_types  # unsafe_file_read
            assert "bare_except" in issue_types  # function_with_bare_except
            assert "broad_exception" in issue_types  # function_with_broad_except

        finally:
            os.unlink(temp_file)

    def test_audit_directory(self):
        """Test auditing a directory of Python files."""
        # Create a temporary directory with Python files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with good error handling
            good_file = Path(temp_dir) / "good.py"
            good_file.write_text('''
def safe_function():
    """Function with error handling."""
    try:
        with open("test.txt", "r") as f:
            return f.read()
    except FileNotFoundError:
        return None
''')

            # Create a file with poor error handling
            bad_file = Path(temp_dir) / "bad.py"
            bad_file.write_text('''
def unsafe_function():
    """Function without error handling."""
    with open("test.txt", "r") as f:
        return f.read()
''')

            # Audit the directory
            report = self.auditor.audit_directory(temp_dir)

            assert isinstance(report, ErrorHandlingAuditReport)
            assert report.total_files == 2
            assert report.total_functions >= 2
            assert 0.0 <= report.coverage_score <= 1.0
            assert 0.0 <= report.quality_score <= 1.0
            assert len(report.issues) > 0
            assert len(report.recommendations) > 0

    def test_coverage_score_calculation(self):
        """Test coverage score calculation."""
        # Perfect coverage
        score = self.auditor._calculate_coverage_score(10, 10)
        assert score == 1.0

        # Partial coverage
        score = self.auditor._calculate_coverage_score(5, 10)
        assert score == 0.5

        # No risky operations
        score = self.auditor._calculate_coverage_score(0, 0)
        assert score == 1.0

    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        # No issues
        self.auditor.issues = []
        score = self.auditor._calculate_quality_score()
        assert score == 1.0

        # Some issues
        self.auditor.issues = [
            ErrorHandlingIssue("file1.py", 1, "unprotected_operation", "high", "desc", "sugg", "func1", "code"),
            ErrorHandlingIssue("file2.py", 2, "bare_except", "medium", "desc", "sugg", "func2", "code")
        ]
        score = self.auditor._calculate_quality_score()
        assert 0.0 <= score < 1.0

    def test_generate_report(self):
        """Test report generation."""
        # Create a sample report
        issues = [
            ErrorHandlingIssue("file1.py", 1, "unprotected_operation", "high", "Unprotected file operation", "Add try-catch", "func1", 'open("file")'),
            ErrorHandlingIssue("file2.py", 5, "bare_except", "medium", "Bare except clause", "Specify exception type", "func2", "except:")
        ]

        report = ErrorHandlingAuditReport(
            total_files=2,
            total_functions=5,
            functions_with_error_handling=2,
            risky_operations=10,
            protected_operations=3,
            issues=issues,
            coverage_score=0.3,
            quality_score=0.75,
            recommendations=["Add error handling", "Use specific exceptions"]
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_report = f.name

        try:
            self.auditor.generate_report(report, temp_report)

            # Check that report file was created and has content
            assert os.path.exists(temp_report)
            with open(temp_report) as f:
                content = f.read()
                assert "# Error Handling Audit Report" in content
                assert "Total Files**: 2" in content
                assert "Coverage Score**: 30.00%" in content
                assert "Quality Score**: 75.00%" in content
                assert "Add error handling" in content
                assert "High" in content
                assert "Medium" in content

        finally:
            os.unlink(temp_report)

    def test_error_handling_issue_structure(self):
        """Test ErrorHandlingIssue dataclass structure."""
        issue = ErrorHandlingIssue(
            file_path="test.py",
            line_number=10,
            issue_type="unprotected_operation",
            severity="high",
            description="File operation without error handling",
            suggestion="Add try-catch block",
            function_name="test_function",
            code_snippet='open("file.txt")'
        )

        assert issue.file_path == "test.py"
        assert issue.line_number == 10
        assert issue.issue_type == "unprotected_operation"
        assert issue.severity == "high"
        assert issue.description == "File operation without error handling"
        assert issue.suggestion == "Add try-catch block"
        assert issue.function_name == "test_function"
        assert issue.code_snippet == 'open("file.txt")'

    def test_generate_error_handling_template(self):
        """Test error handling template generation."""
        file_template = self.auditor.generate_error_handling_template("file")
        assert "try:" in file_template
        assert "FileNotFoundError:" in file_template
        assert "except" in file_template

        network_template = self.auditor.generate_error_handling_template("network")
        assert "requests.get" in network_template
        assert "ConnectionError" in network_template

        database_template = self.auditor.generate_error_handling_template("database")
        assert "cursor" in database_template
        assert "rollback" in database_template


def test_main_function():
    """Test the main function runs without error."""
    from .error_handling_auditor import main

    # Should run without raising exceptions
    main()


if __name__ == "__main__":
    pytest.main([__file__])
