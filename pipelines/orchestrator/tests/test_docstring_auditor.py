#!/usr/bin/env python3
"""
Unit tests for the docstring auditor system.
"""

import os
import tempfile

import pytest

from .docstring_auditor import DocstringAuditor, DocstringAuditReport, DocstringIssue
from .pathlib import Path


class TestDocstringAuditor:
    """Test cases for the DocstringAuditor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = DocstringAuditor()

    def test_auditor_initialization(self):
        """Test auditor initialization with default and custom config."""
        # Default initialization
        auditor = DocstringAuditor()
        assert auditor.min_docstring_length == 20
        assert "Args" in auditor.required_sections
        assert "Returns" in auditor.required_sections

        # Custom configuration
        custom_config = {
            "min_docstring_length": 50,
            "required_sections": ["Parameters", "Returns", "Raises"]
        }
        auditor = DocstringAuditor(custom_config)
        assert auditor.min_docstring_length == 50
        assert auditor.required_sections == ["Parameters", "Returns", "Raises"]

    def test_audit_file_with_good_docstrings(self):
        """Test auditing a file with good docstrings."""
        # Create a temporary file with good docstrings
        good_code = '''
"""Module with good docstrings."""

class GoodClass:
    """
    A well-documented class with comprehensive docstring.

    This class demonstrates proper docstring formatting with
    detailed description and proper structure.
    """

    def good_method(self, param1: str, param2: int) -> str:
        """
        A well-documented method with comprehensive docstring.

        Args:
            param1: Description of the first parameter
            param2: Description of the second parameter

        Returns:
            Description of the return value
        """
        return f"{param1}_{param2}"

def good_function(x: int, y: int) -> int:
    """
    A well-documented function with comprehensive docstring.

    Args:
        x: First integer parameter
        y: Second integer parameter

    Returns:
        Sum of x and y
    """
    return x + y
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(good_code)
            temp_file = f.name

        try:
            stats = self.auditor._audit_file(temp_file)

            assert stats["functions"] == 2  # good_method and good_function
            assert stats["classes"] == 1    # GoodClass
            assert stats["documented_functions"] == 2
            assert stats["documented_classes"] == 1

        finally:
            os.unlink(temp_file)

    def test_audit_file_with_missing_docstrings(self):
        """Test auditing a file with missing docstrings."""
        bad_code = """
class BadClass:
    def bad_method(self, param1, param2):
        return param1 + param2

def bad_function(x, y):
    return x * y
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(bad_code)
            temp_file = f.name

        try:
            # Clear previous issues
            self.auditor.issues = []
            stats = self.auditor._audit_file(temp_file)

            assert stats["functions"] == 2
            assert stats["classes"] == 1
            assert stats["documented_functions"] == 0
            assert stats["documented_classes"] == 0

            # Should have issues for missing docstrings
            missing_docstring_issues = [
                issue for issue in self.auditor.issues
                if issue.issue_type == "missing_docstring"
            ]
            assert len(missing_docstring_issues) == 3  # class + 2 functions

        finally:
            os.unlink(temp_file)

    def test_audit_file_with_quality_issues(self):
        """Test auditing a file with docstring quality issues."""
        quality_issues_code = '''
class ShortDocClass:
    """Short doc."""

    def method_with_args(self, param1, param2):
        """Method without Args section."""
        return param1

    def method_with_return(self):
        """Method without Returns section."""
        return "something"
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(quality_issues_code)
            temp_file = f.name

        try:
            # Clear previous issues
            self.auditor.issues = []
            self.auditor._audit_file(temp_file)

            # Should have quality issues
            short_docstring_issues = [
                issue for issue in self.auditor.issues
                if issue.issue_type == "short_docstring"
            ]
            assert len(short_docstring_issues) >= 1

            missing_args_issues = [
                issue for issue in self.auditor.issues
                if issue.issue_type == "missing_args_section"
            ]
            assert len(missing_args_issues) >= 1

            missing_returns_issues = [
                issue for issue in self.auditor.issues
                if issue.issue_type == "missing_returns_section"
            ]
            assert len(missing_returns_issues) >= 1

        finally:
            os.unlink(temp_file)

    def test_audit_directory(self):
        """Test auditing a directory of Python files."""
        # Create a temporary directory with Python files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a good file
            good_file = Path(temp_dir) / "good.py"
            good_file.write_text('''
"""Good module."""

class GoodClass:
    """Good class with proper docstring."""

    def good_method(self):
        """Good method with proper docstring."""
        pass
''')

            # Create a bad file
            bad_file = Path(temp_dir) / "bad.py"
            bad_file.write_text("""
class BadClass:
    def bad_method(self):
        pass
""")

            # Audit the directory
            report = self.auditor.audit_directory(temp_dir)

            assert isinstance(report, DocstringAuditReport)
            assert report.total_files == 2
            assert report.total_functions >= 2
            assert report.total_classes >= 2
            assert 0.0 <= report.coverage_score <= 1.0
            assert 0.0 <= report.quality_score <= 1.0
            assert len(report.issues) > 0

    def test_coverage_score_calculation(self):
        """Test coverage score calculation."""
        # Perfect coverage
        score = self.auditor._calculate_coverage_score(10, 10, 5, 5)
        assert score == 1.0

        # Partial coverage
        score = self.auditor._calculate_coverage_score(8, 10, 3, 5)
        assert score == 11/15  # (8+3)/(10+5)

        # No items
        score = self.auditor._calculate_coverage_score(0, 0, 0, 0)
        assert score == 1.0

    def test_quality_score_calculation(self):
        """Test quality score calculation."""
        # No issues
        self.auditor.issues = []
        score = self.auditor._calculate_quality_score()
        assert score == 1.0

        # Some issues
        self.auditor.issues = [
            DocstringIssue("file1.py", 1, "missing_docstring", "high", "desc", "sugg"),
            DocstringIssue("file2.py", 2, "short_docstring", "medium", "desc", "sugg")
        ]
        score = self.auditor._calculate_quality_score()
        assert 0.0 <= score < 1.0

    def test_generate_report(self):
        """Test report generation."""
        # Create a sample report
        issues = [
            DocstringIssue("file1.py", 1, "missing_docstring", "critical", "Missing class docstring", "Add docstring"),
            DocstringIssue("file2.py", 5, "short_docstring", "medium", "Short docstring", "Expand docstring")
        ]

        report = DocstringAuditReport(
            total_files=2,
            total_functions=5,
            total_classes=2,
            documented_functions=4,
            documented_classes=1,
            issues=issues,
            coverage_score=0.85,
            quality_score=0.75
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_report = f.name

        try:
            self.auditor.generate_report(report, temp_report)

            # Check that report file was created and has content
            assert os.path.exists(temp_report)
            with open(temp_report) as f:
                content = f.read()
                assert "# Docstring Audit Report" in content
                assert "Total Files**: 2" in content
                assert "Coverage Score**: 85.00%" in content
                assert "Quality Score**: 75.00%" in content
                assert "Critical" in content
                assert "Medium" in content

        finally:
            os.unlink(temp_report)

    def test_docstring_issue_structure(self):
        """Test DocstringIssue dataclass structure."""
        issue = DocstringIssue(
            file_path="test.py",
            line_number=10,
            issue_type="missing_docstring",
            severity="high",
            description="Function missing docstring",
            suggestion="Add comprehensive docstring"
        )

        assert issue.file_path == "test.py"
        assert issue.line_number == 10
        assert issue.issue_type == "missing_docstring"
        assert issue.severity == "high"
        assert issue.description == "Function missing docstring"
        assert issue.suggestion == "Add comprehensive docstring"

    def test_docstring_audit_report_structure(self):
        """Test DocstringAuditReport dataclass structure."""
        issues = [
            DocstringIssue("file1.py", 1, "missing_docstring", "high", "desc", "sugg")
        ]

        report = DocstringAuditReport(
            total_files=5,
            total_functions=20,
            total_classes=8,
            documented_functions=18,
            documented_classes=7,
            issues=issues,
            coverage_score=0.89,
            quality_score=0.92
        )

        assert report.total_files == 5
        assert report.total_functions == 20
        assert report.total_classes == 8
        assert report.documented_functions == 18
        assert report.documented_classes == 7
        assert len(report.issues) == 1
        assert report.coverage_score == 0.89
        assert report.quality_score == 0.92

    def test_fix_generation(self):
        """Test docstring fix generation."""
        issues = [
            DocstringIssue("file1.py", 10, "missing_docstring", "high", "Missing function docstring", "Add docstring")
        ]

        report = DocstringAuditReport(
            total_files=1, total_functions=1, total_classes=0,
            documented_functions=0, documented_classes=0,
            issues=issues, coverage_score=0.0, quality_score=0.5
        )

        fixes = self.auditor.fix_missing_docstrings(report, auto_fix=False)

        assert len(fixes) == 1
        assert '"""' in fixes[0]
        assert "Args:" in fixes[0]
        assert "Returns:" in fixes[0]


def test_main_function():
    """Test the main function runs without error."""
    from .docstring_auditor import main

    # Should run without raising exceptions
    # Note: This will create a report file in the current directory
    main()


if __name__ == "__main__":
    pytest.main([__file__])
