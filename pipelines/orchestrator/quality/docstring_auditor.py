#!/usr/bin/env python3
"""
Docstring Auditor for Task 26
Comprehensive tool to audit, validate, and enhance docstrings across the codebase.
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
class DocstringIssue:
    """Represents a docstring issue found during audit."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    suggestion: str

@dataclass
class DocstringAuditReport:
    """Comprehensive docstring audit report."""
    total_files: int
    total_functions: int
    total_classes: int
    documented_functions: int
    documented_classes: int
    issues: list[DocstringIssue]
    coverage_score: float
    quality_score: float

class DocstringAuditor:
    """
    Comprehensive docstring auditor for Python codebases.

    Analyzes Python files to identify missing, incomplete, or low-quality docstrings
    and provides suggestions for improvement.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the docstring auditor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.issues = []

        # Quality criteria
        self.min_docstring_length = self.config.get("min_docstring_length", 20)
        self.required_sections = self.config.get("required_sections", ["Args", "Returns"])
        self.severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }

    def audit_directory(self, directory_path: str) -> DocstringAuditReport:
        """
        Audit all Python files in a directory for docstring quality.

        Args:
            directory_path: Path to directory to audit

        Returns:
            DocstringAuditReport with comprehensive analysis results
        """
        logger.info(f"Starting docstring audit of directory: {directory_path}")

        self.issues = []
        total_files = 0
        total_functions = 0
        total_classes = 0
        documented_functions = 0
        documented_classes = 0

        # Find all Python files
        python_files = list(Path(directory_path).glob("*.py"))

        for file_path in python_files:
            if file_path.name.startswith("__"):
                continue  # Skip __init__.py, __pycache__, etc.

            total_files += 1
            file_stats = self._audit_file(str(file_path))

            total_functions += file_stats["functions"]
            total_classes += file_stats["classes"]
            documented_functions += file_stats["documented_functions"]
            documented_classes += file_stats["documented_classes"]

        # Calculate scores
        coverage_score = self._calculate_coverage_score(
            documented_functions, total_functions,
            documented_classes, total_classes
        )
        quality_score = self._calculate_quality_score()

        report = DocstringAuditReport(
            total_files=total_files,
            total_functions=total_functions,
            total_classes=total_classes,
            documented_functions=documented_functions,
            documented_classes=documented_classes,
            issues=self.issues.copy(),
            coverage_score=coverage_score,
            quality_score=quality_score
        )

        logger.info(f"Audit complete: {total_files} files, {len(self.issues)} issues found")
        return report

    def _audit_file(self, file_path: str) -> dict[str, int]:
        """
        Audit a single Python file for docstring issues.

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
            classes = 0
            documented_functions = 0
            documented_classes = 0

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions += 1
                    if self._has_docstring(node):
                        documented_functions += 1
                        self._check_docstring_quality(node, file_path)
                    else:
                        self._add_issue(
                            file_path, node.lineno, "missing_docstring", "high",
                            f"Function '{node.name}' missing docstring",
                            f"Add comprehensive docstring for function '{node.name}'"
                        )

                elif isinstance(node, ast.ClassDef):
                    classes += 1
                    if self._has_docstring(node):
                        documented_classes += 1
                        self._check_docstring_quality(node, file_path)
                    else:
                        self._add_issue(
                            file_path, node.lineno, "missing_docstring", "critical",
                            f"Class '{node.name}' missing docstring",
                            f"Add comprehensive docstring for class '{node.name}'"
                        )

            return {
                "functions": functions,
                "classes": classes,
                "documented_functions": documented_functions,
                "documented_classes": documented_classes
            }

        except Exception as e:
            logger.error(f"Error auditing file {file_path}: {e}")
            return {"functions": 0, "classes": 0, "documented_functions": 0, "documented_classes": 0}

    def _has_docstring(self, node: ast.AST) -> bool:
        """Check if a node has a docstring."""
        return (hasattr(node, "body") and
                len(node.body) > 0 and
                isinstance(node.body[0], ast.Expr) and
                isinstance(node.body[0].value, ast.Constant) and
                isinstance(node.body[0].value.value, str))

    def _get_docstring(self, node: ast.AST) -> str | None:
        """Extract docstring from a node."""
        if self._has_docstring(node):
            return node.body[0].value.value
        return None

    def _check_docstring_quality(self, node: ast.AST, file_path: str) -> None:
        """
        Check the quality of a docstring and add issues if found.

        Args:
            node: AST node with docstring
            file_path: Path to file being audited
        """
        docstring = self._get_docstring(node)
        if not docstring:
            return

        node_name = getattr(node, "name", "unknown")

        # Check length
        if len(docstring.strip()) < self.min_docstring_length:
            self._add_issue(
                file_path, node.lineno, "short_docstring", "medium",
                f"Docstring for '{node_name}' is too short ({len(docstring)} chars)",
                f"Expand docstring for '{node_name}' with more detailed description"
            )

        # Check for required sections (for functions with parameters)
        if isinstance(node, ast.FunctionDef) and node.args.args:
            if "Args:" not in docstring and "Parameters:" not in docstring:
                self._add_issue(
                    file_path, node.lineno, "missing_args_section", "medium",
                    f"Function '{node_name}' missing Args section in docstring",
                    f"Add Args section documenting parameters for '{node_name}'"
                )

        # Check for Returns section (for functions that return values)
        if isinstance(node, ast.FunctionDef):
            has_return = any(isinstance(n, ast.Return) and n.value is not None
                           for n in ast.walk(node))
            if has_return and "Returns:" not in docstring and "Return:" not in docstring:
                self._add_issue(
                    file_path, node.lineno, "missing_returns_section", "medium",
                    f"Function '{node_name}' missing Returns section in docstring",
                    f"Add Returns section documenting return value for '{node_name}'"
                )

        # Check for proper formatting
        if not docstring.strip().startswith('"""') and not docstring.strip().startswith("'''"):
            self._add_issue(
                file_path, node.lineno, "improper_formatting", "low",
                f"Docstring for '{node_name}' not properly formatted",
                f"Use triple quotes for docstring in '{node_name}'"
            )

    def _add_issue(self, file_path: str, line_number: int, issue_type: str,
                   severity: str, description: str, suggestion: str) -> None:
        """Add a docstring issue to the list."""
        issue = DocstringIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            description=description,
            suggestion=suggestion
        )
        self.issues.append(issue)

    def _calculate_coverage_score(self, documented_functions: int, total_functions: int,
                                 documented_classes: int, total_classes: int) -> float:
        """Calculate docstring coverage score."""
        if total_functions + total_classes == 0:
            return 1.0

        total_documented = documented_functions + documented_classes
        total_items = total_functions + total_classes

        return total_documented / total_items

    def _calculate_quality_score(self) -> float:
        """Calculate docstring quality score based on issues."""
        if not self.issues:
            return 1.0

        # Weight issues by severity
        total_weight = sum(self.severity_weights.get(issue.severity, 0.5)
                          for issue in self.issues)

        # Normalize by number of issues (lower is better)
        max_possible_weight = len(self.issues) * 1.0  # All critical
        quality_score = 1.0 - (total_weight / max_possible_weight)

        return max(0.0, quality_score)

    def generate_report(self, report: DocstringAuditReport, output_path: str) -> None:
        """
        Generate a detailed audit report.

        Args:
            report: DocstringAuditReport to generate report from
            output_path: Path to save the report
        """
        with open(output_path, "w") as f:
            f.write("# Docstring Audit Report\n\n")
            f.write("## Summary\n")
            f.write(f"- **Total Files**: {report.total_files}\n")
            f.write(f"- **Total Functions**: {report.total_functions}\n")
            f.write(f"- **Total Classes**: {report.total_classes}\n")
            f.write(f"- **Documented Functions**: {report.documented_functions}\n")
            f.write(f"- **Documented Classes**: {report.documented_classes}\n")
            f.write(f"- **Coverage Score**: {report.coverage_score:.2%}\n")
            f.write(f"- **Quality Score**: {report.quality_score:.2%}\n")
            f.write(f"- **Total Issues**: {len(report.issues)}\n\n")

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
                    for issue in issues_by_severity[severity]:
                        f.write(f"- **{issue.file_path}:{issue.line_number}** - {issue.description}\n")
                        f.write(f"  - *Suggestion*: {issue.suggestion}\n\n")

        logger.info(f"Audit report generated: {output_path}")

    def fix_missing_docstrings(self, report: DocstringAuditReport,
                              auto_fix: bool = False) -> list[str]:
        """
        Generate fixes for missing docstrings.

        Args:
            report: DocstringAuditReport with issues to fix
            auto_fix: Whether to automatically apply fixes

        Returns:
            List of suggested fixes
        """
        fixes = []

        for issue in report.issues:
            if issue.issue_type == "missing_docstring":
                fix = self._generate_docstring_template(issue)
                fixes.append(fix)

                if auto_fix:
                    self._apply_docstring_fix(issue, fix)

        return fixes

    def _generate_docstring_template(self, issue: DocstringIssue) -> str:
        """Generate a docstring template for a missing docstring."""
        # This is a simplified template - in practice, you'd analyze the function signature
        return '''"""
        Brief description of the function/class.

        Args:
            param1: Description of parameter 1
            param2: Description of parameter 2

        Returns:
            Description of return value

        Raises:
            ExceptionType: Description of when this exception is raised
        """'''

    def _apply_docstring_fix(self, issue: DocstringIssue, fix: str) -> None:
        """Apply a docstring fix to a file (placeholder implementation)."""
        # This would require more sophisticated AST manipulation
        logger.info(f"Would apply fix to {issue.file_path}:{issue.line_number}")


def main():
    """Main function to run docstring audit."""
    auditor = DocstringAuditor()

    # Audit the dataset_pipeline directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report = auditor.audit_directory(current_dir)

    # Generate report
    report_path = os.path.join(current_dir, "docstring_audit_report.md")
    auditor.generate_report(report, report_path)

    # Print summary


if __name__ == "__main__":
    main()
