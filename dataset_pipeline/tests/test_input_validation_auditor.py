#!/usr/bin/env python3
"""
Unit tests for the input validation auditor system.
"""

import os
import tempfile

import pytest

from .input_validation_auditor import (
    InputValidationAuditor,
    InputValidationAuditReport,
    InputValidationIssue,
)
from .pathlib import Path


class TestInputValidationAuditor:
    """Test cases for the InputValidationAuditor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = InputValidationAuditor()

    def test_auditor_initialization(self):
        """Test auditor initialization."""
        auditor = InputValidationAuditor()
        assert "type_checks" in auditor.validation_keywords
        assert "isinstance" in auditor.validation_keywords["type_checks"]
        assert "file_paths" in auditor.risky_parameter_patterns
        assert auditor.severity_weights["critical"] == 1.0

    def test_audit_file_with_good_validation(self):
        """Test auditing a file with good input validation."""
        good_code = '''
"""Module with good input validation."""

def safe_function(filename: str, count: int):
    """Function with proper input validation."""
    # Type validation
    if not isinstance(filename, str):
        raise TypeError("filename must be a string")

    if not isinstance(count, int):
        raise TypeError("count must be an integer")

    # Value validation
    if not filename.strip():
        raise ValueError("filename cannot be empty")

    if count < 0:
        raise ValueError("count must be non-negative")

    return f"Processing {filename} with count {count}"

def validate_user_input(data: str):
    """Function with user input validation."""
    if not isinstance(data, str):
        raise TypeError("data must be a string")

    if len(data) > 1000:
        raise ValueError("data too long")

    return data.strip()
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(good_code)
            temp_file = f.name

        try:
            stats = self.auditor._audit_file(temp_file)

            assert stats["functions"] == 2
            assert stats["functions_with_validation"] == 2
            assert stats["parameters"] >= 3  # filename, count, data
            assert stats["validated_parameters"] >= 2

        finally:
            os.unlink(temp_file)

    def test_audit_file_with_poor_validation(self):
        """Test auditing a file with poor input validation."""
        bad_code = '''
def unsafe_function(filename, url, user_data):
    """Function without input validation."""
    with open(filename, 'r') as f:
        content = f.read()

    import requests
    response = requests.get(url)

    return content + user_data

def another_unsafe_function(path, count):
    """Another function without validation."""
    return path * count
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(bad_code)
            temp_file = f.name

        try:
            # Clear previous issues
            self.auditor.issues = []
            stats = self.auditor._audit_file(temp_file)

            assert stats["functions"] == 2
            assert stats["parameters"] >= 5  # filename, url, user_data, path, count

            # Should have issues for missing validation
            issues = self.auditor.issues
            assert len(issues) > 0

            # Check for specific risky parameters
            issue_params = [issue.parameter_name for issue in issues]
            assert any("filename" in param or "path" in param for param in issue_params)
            assert any("url" in param for param in issue_params)

        finally:
            os.unlink(temp_file)

    def test_audit_directory(self):
        """Test auditing a directory of Python files."""
        # Create a temporary directory with Python files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a file with good validation
            good_file = Path(temp_dir) / "good.py"
            good_file.write_text('''
def validated_function(filename: str):
    """Function with validation."""
    if not isinstance(filename, str):
        raise TypeError("filename must be string")
    return filename
''')

            # Create a file with poor validation
            bad_file = Path(temp_dir) / "bad.py"
            bad_file.write_text('''
def unvalidated_function(filename, url):
    """Function without validation."""
    return filename + url
''')

            # Audit the directory
            report = self.auditor.audit_directory(temp_dir)

            assert isinstance(report, InputValidationAuditReport)
            assert report.total_files == 2
            assert report.total_functions >= 2
            assert report.total_parameters >= 3
            assert 0.0 <= report.coverage_score <= 1.0
            assert 0.0 <= report.quality_score <= 1.0
            assert len(report.issues) >= 0
            assert len(report.recommendations) > 0
            assert isinstance(report.validation_patterns, dict)

    def test_coverage_score_calculation(self):
        """Test coverage score calculation."""
        # Perfect coverage
        score = self.auditor._calculate_coverage_score(10, 10)
        assert score == 1.0

        # Partial coverage
        score = self.auditor._calculate_coverage_score(5, 10)
        assert score == 0.5

        # No parameters
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
            InputValidationIssue("file1.py", 1, "missing_validation", "high", "desc", "sugg", "func1", "param1", "code"),
            InputValidationIssue("file2.py", 2, "missing_validation", "medium", "desc", "sugg", "func2", "param2", "code")
        ]
        score = self.auditor._calculate_quality_score()
        assert 0.0 <= score < 1.0

    def test_generate_report(self):
        """Test report generation."""
        # Create a sample report
        issues = [
            InputValidationIssue("file1.py", 1, "missing_validation", "high", "Missing validation", "Add validation", "func1", "param1", "code"),
            InputValidationIssue("file2.py", 5, "missing_validation", "medium", "Missing validation", "Add validation", "func2", "param2", "code")
        ]

        validation_patterns = {
            "type_checks": 5,
            "null_checks": 3,
            "length_checks": 2
        }

        report = InputValidationAuditReport(
            total_files=2,
            total_functions=5,
            functions_with_validation=3,
            total_parameters=10,
            validated_parameters=7,
            issues=issues,
            coverage_score=0.7,
            quality_score=0.75,
            recommendations=["Add validation", "Use type hints"],
            validation_patterns=validation_patterns
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_report = f.name

        try:
            self.auditor.generate_report(report, temp_report)

            # Check that report file was created and has content
            assert os.path.exists(temp_report)
            with open(temp_report) as f:
                content = f.read()
                assert "# Input Validation Audit Report" in content
                assert "Total Files**: 2" in content
                assert "Coverage Score**: 70.00%" in content
                assert "Quality Score**: 75.00%" in content
                assert "Type Checks**: 5" in content
                assert "Add validation" in content

        finally:
            os.unlink(temp_report)

    def test_input_validation_issue_structure(self):
        """Test InputValidationIssue dataclass structure."""
        issue = InputValidationIssue(
            file_path="test.py",
            line_number=10,
            issue_type="missing_validation",
            severity="high",
            description="Parameter lacks validation",
            suggestion="Add type and value checks",
            function_name="test_function",
            parameter_name="test_param",
            code_snippet="def test_function(test_param):"
        )

        assert issue.file_path == "test.py"
        assert issue.line_number == 10
        assert issue.issue_type == "missing_validation"
        assert issue.severity == "high"
        assert issue.description == "Parameter lacks validation"
        assert issue.suggestion == "Add type and value checks"
        assert issue.function_name == "test_function"
        assert issue.parameter_name == "test_param"
        assert issue.code_snippet == "def test_function(test_param):"

    def test_generate_validation_template(self):
        """Test validation template generation."""
        file_path_template = self.auditor.generate_validation_template("file_path")
        assert "validate_file_path" in file_path_template
        assert "isinstance" in file_path_template
        assert "ValueError" in file_path_template

        url_template = self.auditor.generate_validation_template("url")
        assert "validate_url" in url_template
        assert "https?" in url_template

        user_input_template = self.auditor.generate_validation_template("user_input")
        assert "validate_user_input" in user_input_template
        assert "max_length" in user_input_template

    def test_parameter_severity_mapping(self):
        """Test parameter severity mapping."""
        assert self.auditor._get_parameter_severity("file_paths") == "high"
        assert self.auditor._get_parameter_severity("urls") == "high"
        assert self.auditor._get_parameter_severity("user_input") == "medium"
        assert self.auditor._get_parameter_severity("numbers") == "low"
        assert self.auditor._get_parameter_severity("unknown") == "medium"


def test_main_function():
    """Test the main function runs without error."""
    from .input_validation_auditor import main

    # Should run without raising exceptions
    main()


if __name__ == "__main__":
    pytest.main([__file__])
