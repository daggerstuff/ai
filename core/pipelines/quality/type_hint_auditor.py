#!/usr/bin/env python3
"""
Type Hint Auditor for Task 27
Comprehensive tool to audit, validate, and enhance type hints across the codebase.
"""

import ast
import logging
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class TypeHintIssue:
    """Represents a type hint issue found during audit."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    suggestion: str
    function_name: str

@dataclass
class TypeHintAuditReport:
    """Comprehensive type hint audit report."""
    total_files: int
    total_functions: int
    total_methods: int
    typed_functions: int
    typed_methods: int
    issues: list[TypeHintIssue]
    coverage_score: float
    quality_score: float
    import_suggestions: set[str]

class TypeHintAuditor:
    """
    Comprehensive type hint auditor for Python codebases.

    Analyzes Python files to identify missing, incomplete, or incorrect type hints
    and provides suggestions for improvement.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the type hint auditor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.issues = []
        self.import_suggestions = set()

        # Common type mappings
        self.common_types = {
            "str": "str",
            "int": "int",
            "float": "float",
            "bool": "bool",
            "list": "List",
            "dict": "Dict",
            "tuple": "Tuple",
            "set": "Set",
            "None": "None"
        }

        # Functions that typically don't need return type hints
        self.skip_return_hint = {
            "__init__", "__enter__", "__exit__", "setUp", "tearDown",
            "setup_method", "teardown_method"
        }

        # Severity weights for scoring
        self.severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }

    def audit_directory(self, directory_path: str) -> TypeHintAuditReport:
        """
        Audit all Python files in a directory for type hint quality.

        Args:
            directory_path: Path to directory to audit

        Returns:
            TypeHintAuditReport with comprehensive analysis results
        """
        logger.info(f"Starting type hint audit of directory: {directory_path}")

        self.issues = []
        self.import_suggestions = set()
        total_files = 0
        total_functions = 0
        total_methods = 0
        typed_functions = 0
        typed_methods = 0

        # Find all Python files
        python_files = list(Path(directory_path).glob("*.py"))

        for file_path in python_files:
            if file_path.name.startswith("__"):
                continue  # Skip __init__.py, __pycache__, etc.

            total_files += 1
            file_stats = self._audit_file(str(file_path))

            total_functions += file_stats["functions"]
            total_methods += file_stats["methods"]
            typed_functions += file_stats["typed_functions"]
            typed_methods += file_stats["typed_methods"]

        # Calculate scores
        coverage_score = self._calculate_coverage_score(
            typed_functions, total_functions, typed_methods, total_methods
        )
        quality_score = self._calculate_quality_score()

        report = TypeHintAuditReport(
            total_files=total_files,
            total_functions=total_functions,
            total_methods=total_methods,
            typed_functions=typed_functions,
            typed_methods=typed_methods,
            issues=self.issues.copy(),
            coverage_score=coverage_score,
            quality_score=quality_score,
            import_suggestions=self.import_suggestions.copy()
        )

        logger.info(f"Type hint audit complete: {total_files} files, {len(self.issues)} issues found")
        return report

    def _audit_file(self, file_path: str) -> dict[str, int]:
        """
        Audit a single Python file for type hint issues.

        Args:
            file_path: Path to Python file to audit

        Returns:
            Dictionary with file statistics
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)

            functions = 0
            methods = 0
            typed_functions = 0
            typed_methods = 0

            # Check for existing typing imports
            existing_imports = self._get_existing_imports(tree)

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    is_method = self._is_method(node, tree)

                    if is_method:
                        methods += 1
                        if self._has_type_hints(node):
                            typed_methods += 1
                        else:
                            self._add_type_hint_issue(node, file_path, "method")
                    else:
                        functions += 1
                        if self._has_type_hints(node):
                            typed_functions += 1
                        else:
                            self._add_type_hint_issue(node, file_path, "function")

                    # Check quality of existing type hints
                    self._check_type_hint_quality(node, file_path, existing_imports)

            return {
                "functions": functions,
                "methods": methods,
                "typed_functions": typed_functions,
                "typed_methods": typed_methods
            }

        except Exception as e:
            logger.error(f"Error auditing file {file_path}: {e}")
            return {"functions": 0, "methods": 0, "typed_functions": 0, "typed_methods": 0}

    def _get_existing_imports(self, tree: ast.AST) -> set[str]:
        """Get existing typing imports from the AST."""
        imports = set()

        for node in ast.walk(tree):
            if isinstance(node, ast.ImportFrom) and node.module == "typing":
                for alias in node.names:
                    imports.add(alias.name)
            elif isinstance(node, ast.Import):
                for alias in node.names:
                    if alias.name == "typing":
                        imports.add("typing")

        return imports

    def _is_method(self, node: ast.FunctionDef, tree: ast.AST) -> bool:
        """Check if a function is a method (inside a class)."""
        for parent in ast.walk(tree):
            if isinstance(parent, ast.ClassDef) and node in parent.body:
                return True
        return False

    def _has_type_hints(self, node: ast.FunctionDef) -> bool:
        """Check if a function has type hints."""
        # Check parameters
        has_param_hints = any(arg.annotation is not None for arg in node.args.args)

        # Check return type (skip certain functions)
        has_return_hint = (node.returns is not None or
                          node.name in self.skip_return_hint)

        return has_param_hints or has_return_hint

    def _add_type_hint_issue(self, node: ast.FunctionDef, file_path: str, func_type: str) -> None:
        """Add a type hint issue for a function/method."""
        severity = "high" if func_type == "function" else "medium"

        # Check if it's a test function (lower priority)
        if node.name.startswith("test_") or "test" in file_path.lower():
            severity = "low"

        # Check if it's a private function (lower priority)
        if node.name.startswith("_"):
            severity = "medium" if severity == "high" else "low"

        issue = TypeHintIssue(
            file_path=file_path,
            line_number=node.lineno,
            issue_type="missing_type_hints",
            severity=severity,
            description=f"{func_type.title()} '{node.name}' missing type hints",
            suggestion=self._generate_type_hint_suggestion(node),
            function_name=node.name
        )
        self.issues.append(issue)

    def _check_type_hint_quality(self, node: ast.FunctionDef, file_path: str,
                                existing_imports: set[str]) -> None:
        """Check the quality of existing type hints."""
        # Check for generic types without proper imports
        for arg in node.args.args:
            if arg.annotation:
                self._check_annotation_quality(arg.annotation, file_path, node.lineno,
                                              existing_imports, node.name)

        if node.returns:
            self._check_annotation_quality(node.returns, file_path, node.lineno,
                                          existing_imports, node.name)

    def _check_annotation_quality(self, annotation: ast.AST, file_path: str,
                                 line_number: int, existing_imports: set[str],
                                 function_name: str) -> None:
        """Check the quality of a specific type annotation."""
        if isinstance(annotation, ast.Name):
            type_name = annotation.id

            # Check for generic types that need typing imports
            if type_name in ["List", "Dict", "Tuple", "Set", "Optional", "Union"]:
                if type_name not in existing_imports and "typing" not in existing_imports:
                    self.import_suggestions.add(f"from typing import {type_name}")

                    issue = TypeHintIssue(
                        file_path=file_path,
                        line_number=line_number,
                        issue_type="missing_import",
                        severity="medium",
                        description=f"Type hint '{type_name}' used without import",
                        suggestion=f"Add 'from typing import {type_name}'",
                        function_name=function_name
                    )
                    self.issues.append(issue)

    def _generate_type_hint_suggestion(self, node: ast.FunctionDef) -> str:
        """Generate a type hint suggestion for a function."""
        suggestions = []

        # Parameter suggestions
        if node.args.args:
            param_hints = []
            for arg in node.args.args:
                if arg.arg == "self":
                    continue
                param_hints.append(f"{arg.arg}: Any")

            if param_hints:
                suggestions.append(f"Add parameter type hints: {', '.join(param_hints)}")
                self.import_suggestions.add("from typing import Any")

        # Return type suggestion
        if node.name not in self.skip_return_hint:
            suggestions.append("Add return type hint: -> Any")
            self.import_suggestions.add("from typing import Any")

        return "; ".join(suggestions)

    def _calculate_coverage_score(self, typed_functions: int, total_functions: int,
                                 typed_methods: int, total_methods: int) -> float:
        """Calculate type hint coverage score."""
        if total_functions + total_methods == 0:
            return 1.0

        total_typed = typed_functions + typed_methods
        total_items = total_functions + total_methods

        return total_typed / total_items

    def _calculate_quality_score(self) -> float:
        """Calculate type hint quality score based on issues."""
        if not self.issues:
            return 1.0

        # Weight issues by severity
        total_weight = sum(self.severity_weights.get(issue.severity, 0.5)
                          for issue in self.issues)

        # Normalize by number of issues (lower is better)
        max_possible_weight = len(self.issues) * 1.0  # All critical
        quality_score = 1.0 - (total_weight / max_possible_weight)

        return max(0.0, quality_score)

    def generate_report(self, report: TypeHintAuditReport, output_path: str) -> None:
        """
        Generate a detailed type hint audit report.

        Args:
            report: TypeHintAuditReport to generate report from
            output_path: Path to save the report
        """
        with open(output_path, "w") as f:
            f.write("# Type Hint Audit Report\n\n")
            f.write("## Summary\n")
            f.write(f"- **Total Files**: {report.total_files}\n")
            f.write(f"- **Total Functions**: {report.total_functions}\n")
            f.write(f"- **Total Methods**: {report.total_methods}\n")
            f.write(f"- **Typed Functions**: {report.typed_functions}\n")
            f.write(f"- **Typed Methods**: {report.typed_methods}\n")
            f.write(f"- **Coverage Score**: {report.coverage_score:.2%}\n")
            f.write(f"- **Quality Score**: {report.quality_score:.2%}\n")
            f.write(f"- **Total Issues**: {len(report.issues)}\n\n")

            # Import suggestions
            if report.import_suggestions:
                f.write("## Suggested Imports\n\n")
                for import_stmt in sorted(report.import_suggestions):
                    f.write(f"```python\n{import_stmt}\n```\n\n")

            # Group issues by severity
            issues_by_severity = {}
            for issue in report.issues:
                if issue.severity not in issues_by_severity:
                    issues_by_severity[issue.severity] = []
                issues_by_severity[issue.severity].append(issue)

            f.write("## Issues by Severity\n\n")
            for severity in ["critical", "high", "medium", "low"]:
                if severity in issues_by_severity:
                    f.write(f"### {severity.title()} ({len(issues_by_severity[severity])} issues)\n\n")
                    for issue in issues_by_severity[severity][:20]:  # Limit to first 20
                        f.write(f"- **{issue.file_path}:{issue.line_number}** - {issue.description}\n")
                        f.write(f"  - *Suggestion*: {issue.suggestion}\n\n")

        logger.info(f"Type hint audit report generated: {output_path}")

    def generate_type_stub(self, file_path: str) -> str:
        """
        Generate a type stub (.pyi) file for a Python file.

        Args:
            file_path: Path to Python file

        Returns:
            Type stub content as string
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            stub_lines = []

            # Add imports
            stub_lines.append("from typing import Any, Dict, List, Optional, Tuple, Union")
            stub_lines.append("")

            # Process classes and functions
            for node in ast.walk(tree):
                if isinstance(node, ast.ClassDef):
                    stub_lines.append(f"class {node.name}:")
                    for item in node.body:
                        if isinstance(item, ast.FunctionDef):
                            stub_lines.append(f"    def {item.name}(self, *args: Any, **kwargs: Any) -> Any: ...")
                    stub_lines.append("")

                elif isinstance(node, ast.FunctionDef) and not self._is_method(node, tree):
                    stub_lines.append(f"def {node.name}(*args: Any, **kwargs: Any) -> Any: ...")

            return "\n".join(stub_lines)

        except Exception as e:
            logger.error(f"Error generating type stub for {file_path}: {e}")
            return ""


def main():
    """Main function to run type hint audit."""
    auditor = TypeHintAuditor()

    # Audit the dataset_pipeline directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report = auditor.audit_directory(current_dir)

    # Generate report
    report_path = os.path.join(current_dir, "type_hint_audit_report.md")
    auditor.generate_report(report, report_path)

    # Print summary


if __name__ == "__main__":
    main()
