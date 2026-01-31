#!/usr/bin/env python3
"""
Unit tests for the type hint auditor system.
"""

import ast
import os
import tempfile

import pytest

from .pathlib import Path
from .type_hint_auditor import TypeHintAuditor, TypeHintAuditReport, TypeHintIssue


class TestTypeHintAuditor:
    """Test cases for the TypeHintAuditor class."""

    def setup_method(self):
        """Set up test fixtures."""
        self.auditor = TypeHintAuditor()

    def test_auditor_initialization(self):
        """Test auditor initialization with default and custom config."""
        # Default initialization
        auditor = TypeHintAuditor()
        assert "str" in auditor.common_types
        assert "__init__" in auditor.skip_return_hint
        assert auditor.severity_weights["critical"] == 1.0

        # Custom configuration
        custom_config = {
            "min_coverage": 0.8,
            "strict_mode": True
        }
        auditor = TypeHintAuditor(custom_config)
        assert auditor.config["min_coverage"] == 0.8
        assert auditor.config["strict_mode"]

    def test_audit_file_with_good_type_hints(self):
        """Test auditing a file with good type hints."""
        good_code = '''
"""Module with good type hints."""
from .typing import List, Dict, Optional

class GoodClass:
    """A well-typed class."""

    def __init__(self, name: str) -> None:
        self.name = name

    def good_method(self, param1: str, param2: int) -> str:
        """A well-typed method."""
        return f"{param1}_{param2}"

def good_function(x: int, y: int) -> int:
    """A well-typed function."""
    return x + y

def process_data(data: List[Dict[str, str]]) -> Optional[str]:
    """Function with complex type hints."""
    if not data:
        return None
    return data[0].get('key')
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(good_code)
            temp_file = f.name

        try:
            stats = self.auditor._audit_file(temp_file)

            assert stats["functions"] == 2  # good_function and process_data
            assert stats["methods"] == 2    # __init__ and good_method
            assert stats["typed_functions"] == 2
            assert stats["typed_methods"] == 2

        finally:
            os.unlink(temp_file)

    def test_audit_file_with_missing_type_hints(self):
        """Test auditing a file with missing type hints."""
        bad_code = """
class BadClass:
    def __init__(self, name):
        self.name = name

    def bad_method(self, param1, param2):
        return param1 + param2

def bad_function(x, y):
    return x * y

def another_function():
    pass
"""

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(bad_code)
            temp_file = f.name

        try:
            # Clear previous issues
            self.auditor.issues = []
            stats = self.auditor._audit_file(temp_file)

            assert stats["functions"] == 2  # bad_function and another_function
            assert stats["methods"] == 2    # __init__ and bad_method
            assert stats["typed_functions"] == 0
            assert stats["typed_methods"] == 1  # __init__ is considered typed (skip return hint)

            # Should have issues for missing type hints
            missing_hint_issues = [
                issue for issue in self.auditor.issues
                if issue.issue_type == "missing_type_hints"
            ]
            assert len(missing_hint_issues) == 3  # bad_method, bad_function, another_function
            # __init__ is skipped for return type hints but still needs param hints

        finally:
            os.unlink(temp_file)

    def test_audit_file_with_import_issues(self):
        """Test auditing a file with type hint import issues."""
        import_issues_code = '''
def function_with_list(data: List[str]) -> Dict[str, int]:
    """Function using List and Dict without imports."""
    return {item: len(item) for item in data}

def function_with_optional(value: Optional[str]) -> Union[str, int]:
    """Function using Optional and Union without imports."""
    return value or 0
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(import_issues_code)
            temp_file = f.name

        try:
            # Clear previous issues
            self.auditor.issues = []
            self.auditor.import_suggestions = set()
            stats = self.auditor._audit_file(temp_file)

            # Functions should be detected as typed (they have type hints)
            assert stats["functions"] == 2
            assert stats["typed_functions"] == 2

            # The auditor should run without errors
            assert isinstance(self.auditor.issues, list)
            assert isinstance(self.auditor.import_suggestions, set)

        finally:
            os.unlink(temp_file)

    def test_audit_directory(self):
        """Test auditing a directory of Python files."""
        # Create a temporary directory with Python files
        with tempfile.TemporaryDirectory() as temp_dir:
            # Create a well-typed file
            good_file = Path(temp_dir) / "good.py"
            good_file.write_text('''
"""Good module."""
from .typing import List

class GoodClass:
    """Good class."""

    def __init__(self, data: List[str]) -> None:
        self.data = data

    def good_method(self, item: str) -> bool:
        """Good method."""
        return item in self.data

def good_function(x: int) -> int:
    """Good function."""
    return x * 2
''')

            # Create a poorly typed file
            bad_file = Path(temp_dir) / "bad.py"
            bad_file.write_text("""
class BadClass:
    def __init__(self, data):
        self.data = data

    def bad_method(self, item):
        return item in self.data

def bad_function(x):
    return x * 2
""")

            # Audit the directory
            report = self.auditor.audit_directory(temp_dir)

            assert isinstance(report, TypeHintAuditReport)
            assert report.total_files == 2
            assert report.total_functions >= 2
            assert report.total_methods >= 4
            assert 0.0 <= report.coverage_score <= 1.0
            assert 0.0 <= report.quality_score <= 1.0
            assert len(report.issues) > 0

    def test_has_type_hints_detection(self):
        """Test detection of type hints in functions."""
        # Function with parameter hints
        code1 = """
def func_with_param_hints(x: int, y: str):
    return x
"""
        tree1 = ast.parse(code1)
        func1 = tree1.body[0]
        assert self.auditor._has_type_hints(func1)

        # Function with return hint
        code2 = """
def func_with_return_hint(x, y) -> str:
    return str(x)
"""
        tree2 = ast.parse(code2)
        func2 = tree2.body[0]
        assert self.auditor._has_type_hints(func2)

        # Function with no hints
        code3 = """
def func_no_hints(x, y):
    return x + y
"""
        tree3 = ast.parse(code3)
        func3 = tree3.body[0]
        assert not self.auditor._has_type_hints(func3)

        # __init__ method (should be considered typed even without return hint)
        code4 = """
def __init__(self, x: int):
    self.x = x
"""
        tree4 = ast.parse(code4)
        func4 = tree4.body[0]
        assert self.auditor._has_type_hints(func4)

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
            TypeHintIssue("file1.py", 1, "missing_type_hints", "high", "desc", "sugg", "func1"),
            TypeHintIssue("file2.py", 2, "missing_import", "medium", "desc", "sugg", "func2")
        ]
        score = self.auditor._calculate_quality_score()
        assert 0.0 <= score < 1.0

    def test_generate_report(self):
        """Test report generation."""
        # Create a sample report
        issues = [
            TypeHintIssue("file1.py", 1, "missing_type_hints", "high", "Missing type hints", "Add type hints", "func1"),
            TypeHintIssue("file2.py", 5, "missing_import", "medium", "Missing import", "Add import", "func2")
        ]

        report = TypeHintAuditReport(
            total_files=2,
            total_functions=5,
            total_methods=3,
            typed_functions=3,
            typed_methods=2,
            issues=issues,
            coverage_score=0.625,  # (3+2)/(5+3)
            quality_score=0.75,
            import_suggestions={"from .typing import List", "from .typing import Dict"}
        )

        with tempfile.NamedTemporaryFile(mode="w", suffix=".md", delete=False) as f:
            temp_report = f.name

        try:
            self.auditor.generate_report(report, temp_report)

            # Check that report file was created and has content
            assert os.path.exists(temp_report)
            with open(temp_report) as f:
                content = f.read()
                assert "# Type Hint Audit Report" in content
                assert "Total Files**: 2" in content
                assert "Coverage Score**: 62.50%" in content
                assert "Quality Score**: 75.00%" in content
                assert "from .typing import" in content
                assert "High" in content
                assert "Medium" in content

        finally:
            os.unlink(temp_report)

    def test_type_hint_issue_structure(self):
        """Test TypeHintIssue dataclass structure."""
        issue = TypeHintIssue(
            file_path="test.py",
            line_number=10,
            issue_type="missing_type_hints",
            severity="high",
            description="Function missing type hints",
            suggestion="Add type hints",
            function_name="test_function"
        )

        assert issue.file_path == "test.py"
        assert issue.line_number == 10
        assert issue.issue_type == "missing_type_hints"
        assert issue.severity == "high"
        assert issue.description == "Function missing type hints"
        assert issue.suggestion == "Add type hints"
        assert issue.function_name == "test_function"

    def test_type_hint_audit_report_structure(self):
        """Test TypeHintAuditReport dataclass structure."""
        issues = [
            TypeHintIssue("file1.py", 1, "missing_type_hints", "high", "desc", "sugg", "func1")
        ]

        import_suggestions = {"from .typing import List"}

        report = TypeHintAuditReport(
            total_files=5,
            total_functions=20,
            total_methods=15,
            typed_functions=18,
            typed_methods=12,
            issues=issues,
            coverage_score=0.857,  # (18+12)/(20+15)
            quality_score=0.92,
            import_suggestions=import_suggestions
        )

        assert report.total_files == 5
        assert report.total_functions == 20
        assert report.total_methods == 15
        assert report.typed_functions == 18
        assert report.typed_methods == 12
        assert len(report.issues) == 1
        assert report.coverage_score == 0.857
        assert report.quality_score == 0.92
        assert len(report.import_suggestions) == 1

    def test_generate_type_stub(self):
        """Test type stub generation."""
        code = '''
"""Test module."""

class TestClass:
    """Test class."""

    def __init__(self, name):
        self.name = name

    def method(self, value):
        return value * 2

def function(x, y):
    """Test function."""
    return x + y
'''

        with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as f:
            f.write(code)
            temp_file = f.name

        try:
            stub_content = self.auditor.generate_type_stub(temp_file)

            assert "from .typing import Any" in stub_content
            assert "class TestClass:" in stub_content
            assert "def method(self, *args: Any, **kwargs: Any) -> Any: ..." in stub_content
            assert "def function(*args: Any, **kwargs: Any) -> Any: ..." in stub_content

        finally:
            os.unlink(temp_file)

    def test_existing_imports_detection(self):
        """Test detection of existing typing imports."""
        import ast

        code = """
from .typing import List, Dict, Optional
import typing
from .collections import defaultdict
"""

        tree = ast.parse(code)
        imports = self.auditor._get_existing_imports(tree)

        assert "List" in imports
        assert "Dict" in imports
        assert "Optional" in imports
        assert "typing" in imports


def test_main_function():
    """Test the main function runs without error."""
    from .type_hint_auditor import main

    # Should run without raising exceptions
    main()


if __name__ == "__main__":
    pytest.main([__file__])
