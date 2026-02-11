#!/usr/bin/env python3
"""
Error Handling Auditor for Task 28
Comprehensive tool to audit, validate, and enhance error handling across the codebase.
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
class ErrorHandlingIssue:
    """Represents an error handling issue found during audit."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    suggestion: str
    function_name: str
    code_snippet: str

@dataclass
class ErrorHandlingAuditReport:
    """Comprehensive error handling audit report."""
    total_files: int
    total_functions: int
    functions_with_error_handling: int
    risky_operations: int
    protected_operations: int
    issues: list[ErrorHandlingIssue]
    coverage_score: float
    quality_score: float
    recommendations: list[str]

class ErrorHandlingAuditor:
    """
    Comprehensive error handling auditor for Python codebases.

    Analyzes Python files to identify missing, incomplete, or poor error handling
    and provides suggestions for improvement.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the error handling auditor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.issues = []
        self.recommendations = []

        # Risky operations that should have error handling
        self.risky_operations = {
            "file_operations": ["open", "read", "write", "close", "remove", "rename"],
            "network_operations": ["requests.get", "requests.post", "urllib.request", "socket"],
            "database_operations": ["execute", "commit", "connect", "cursor"],
            "json_operations": ["json.loads", "json.dumps", "json.load", "json.dump"],
            "external_processes": ["subprocess.run", "subprocess.call", "os.system"],
            "type_conversions": ["int(", "float(", "str(", "list(", "dict("],
            "indexing_operations": ["[", "get(", "pop(", "index("],
            "import_operations": ["import", "__import__", "importlib"]
        }

        # Common exception types that should be caught
        self.common_exceptions = {
            "FileNotFoundError", "IOError", "OSError", "PermissionError",
            "ValueError", "TypeError", "KeyError", "IndexError", "AttributeError",
            "ConnectionError", "TimeoutError", "HTTPError",
            "JSONDecodeError", "UnicodeDecodeError", "UnicodeEncodeError",
            "ImportError", "ModuleNotFoundError"
        }

        # Severity weights for scoring
        self.severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }

    def audit_directory(self, directory_path: str) -> ErrorHandlingAuditReport:
        """
        Audit all Python files in a directory for error handling quality.

        Args:
            directory_path: Path to directory to audit

        Returns:
            ErrorHandlingAuditReport with comprehensive analysis results
        """
        logger.info(f"Starting error handling audit of directory: {directory_path}")

        self.issues = []
        self.recommendations = []
        total_files = 0
        total_functions = 0
        functions_with_error_handling = 0
        risky_operations = 0
        protected_operations = 0

        # Find all Python files
        python_files = list(Path(directory_path).glob("*.py"))

        for file_path in python_files:
            if file_path.name.startswith("__"):
                continue  # Skip __init__.py, __pycache__, etc.

            total_files += 1
            file_stats = self._audit_file(str(file_path))

            total_functions += file_stats["functions"]
            functions_with_error_handling += file_stats["functions_with_error_handling"]
            risky_operations += file_stats["risky_operations"]
            protected_operations += file_stats["protected_operations"]

        # Calculate scores
        coverage_score = self._calculate_coverage_score(
            protected_operations, risky_operations
        )
        quality_score = self._calculate_quality_score()

        # Generate recommendations
        self._generate_recommendations()

        report = ErrorHandlingAuditReport(
            total_files=total_files,
            total_functions=total_functions,
            functions_with_error_handling=functions_with_error_handling,
            risky_operations=risky_operations,
            protected_operations=protected_operations,
            issues=self.issues.copy(),
            coverage_score=coverage_score,
            quality_score=quality_score,
            recommendations=self.recommendations.copy()
        )

        logger.info(f"Error handling audit complete: {total_files} files, {len(self.issues)} issues found")
        return report

    def _audit_file(self, file_path: str) -> dict[str, int]:
        """
        Audit a single Python file for error handling issues.

        Args:
            file_path: Path to Python file to audit

        Returns:
            Dictionary with file statistics
        """
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            tree = ast.parse(content)
            lines = content.split("\n")

            functions = 0
            functions_with_error_handling = 0
            risky_operations = 0
            protected_operations = 0

            # Track try-except blocks
            try_blocks = []
            for node in ast.walk(tree):
                if isinstance(node, ast.Try):
                    try_blocks.append(node)

            # Analyze functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions += 1

                    # Check if function has error handling
                    has_error_handling = self._function_has_error_handling(node)
                    if has_error_handling:
                        functions_with_error_handling += 1

                    # Check for risky operations
                    risky_ops = self._find_risky_operations(node, lines)
                    risky_operations += len(risky_ops)

                    # Check which risky operations are protected
                    protected_ops = self._find_protected_operations(node, try_blocks)
                    protected_operations += len(protected_ops)

                    # Generate issues for unprotected risky operations
                    unprotected_ops = []
                    protected_op_lines = {op["line"] for op in protected_ops}

                    for op in risky_ops:
                        if op["line"] not in protected_op_lines:
                            unprotected_ops.append(op)

                    for op in unprotected_ops:
                        self._add_error_handling_issue(
                            file_path, op["line"], "unprotected_operation",
                            "high", f"Risky operation '{op['operation']}' not protected by try-catch",
                            f"Wrap '{op['operation']}' in try-catch block",
                            node.name, op["code"]
                        )

                    # Check for poor error handling practices
                    self._check_error_handling_quality(node, file_path, lines)

            return {
                "functions": functions,
                "functions_with_error_handling": functions_with_error_handling,
                "risky_operations": risky_operations,
                "protected_operations": protected_operations
            }

        except Exception as e:
            logger.error(f"Error auditing file {file_path}: {e}")
            return {
                "functions": 0, "functions_with_error_handling": 0,
                "risky_operations": 0, "protected_operations": 0
            }

    def _function_has_error_handling(self, node: ast.FunctionDef) -> bool:
        """Check if a function has any error handling."""
        return any(isinstance(child, ast.Try) for child in ast.walk(node))

    def _find_risky_operations(self, node: ast.FunctionDef, lines: list[str]) -> list[dict[str, Any]]:
        """Find risky operations in a function."""
        risky_ops = []

        for child in ast.walk(node):
            # File operations
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name):
                    func_name = child.func.id
                    if func_name in self.risky_operations["file_operations"]:
                        risky_ops.append({
                            "operation": func_name,
                            "line": child.lineno,
                            "code": lines[child.lineno - 1].strip() if child.lineno <= len(lines) else ""
                        })

                # Attribute calls (e.g., json.loads, requests.get)
                elif isinstance(child.func, ast.Attribute):
                    attr_name = child.func.attr
                    # Fix: Safely get the base name for the attribute call
                    if isinstance(child.func.value, ast.Name):
                        full_name = f"{child.func.value.id}.{attr_name}"
                    else:
                        # Use ast.unparse if available (Python 3.9+), else fallback to string
                        try:
                            full_name = f"{ast.unparse(child.func.value)}.{attr_name}"
                        except Exception:
                            full_name = f"<expr>.{attr_name}"
                    for _category, ops in self.risky_operations.items():
                        if full_name in ops or attr_name in ops:
                            risky_ops.append({
                                "operation": full_name,
                                "line": child.lineno,
                                "code": lines[child.lineno - 1].strip() if child.lineno <= len(lines) else ""
                            })

            # Subscript operations (indexing)
            elif isinstance(child, ast.Subscript):
                risky_ops.append({
                    "operation": "indexing",
                    "line": child.lineno,
                    "code": lines[child.lineno - 1].strip() if child.lineno <= len(lines) else ""
                })

        return risky_ops

    def _find_protected_operations(self, node: ast.FunctionDef, try_blocks: list[ast.Try]) -> list[dict[str, Any]]:
        """Find operations that are protected by try-catch blocks."""
        protected_ops = []

        for try_block in try_blocks:
            # Check if try block is within this function
            if self._is_node_within_function(try_block, node):
                # Find risky operations within this try block
                for child in ast.walk(try_block):
                    if isinstance(child, ast.Call):
                        if isinstance(child.func, ast.Name):
                            func_name = child.func.id
                            if any(func_name in ops for ops in self.risky_operations.values()):
                                protected_ops.append({
                                    "operation": func_name,
                                    "line": child.lineno
                                })

        return protected_ops

    def _is_node_within_function(self, node: ast.AST, function: ast.FunctionDef) -> bool:
        """Check if a node is within a specific function."""
        # Check if both node and function have lineno attribute
        if not hasattr(node, "lineno") or not hasattr(function, "lineno"):
            return False

        # Get function start line safely
        func_start = getattr(function, "lineno", None)
        if func_start is None:
            return False

        # Get node line safely
        node_line = getattr(node, "lineno", None)
        if node_line is None:
            return False

        # Determine function end line safely
        if hasattr(function, "end_lineno"):
            func_end = function.end_lineno
        elif function.body and hasattr(function.body[-1], "lineno"):
            func_end = function.body[-1].lineno
        else:
            # Fallback: use function start line + 100 as estimate
            func_end = func_start + 100

        return func_start <= node_line <= func_end

    def _check_error_handling_quality(self, node: ast.FunctionDef, file_path: str, lines: list[str]) -> None:
        """Check the quality of existing error handling."""
        for child in ast.walk(node):
            if isinstance(child, ast.Try):
                # Check for bare except clauses
                for handler in child.handlers:
                    if handler.type is None:  # bare except:
                        self._add_error_handling_issue(
                            file_path, handler.lineno, "bare_except",
                            "medium", "Bare 'except:' clause catches all exceptions",
                            "Specify specific exception types to catch",
                            node.name, lines[handler.lineno - 1].strip() if handler.lineno <= len(lines) else ""
                        )

                    # Check for overly broad exception handling
                    elif isinstance(handler.type, ast.Name) and handler.type.id == "Exception":
                        self._add_error_handling_issue(
                            file_path, handler.lineno, "broad_exception",
                            "low", "Catching generic 'Exception' is too broad",
                            "Catch specific exception types instead",
                            node.name, lines[handler.lineno - 1].strip() if handler.lineno <= len(lines) else ""
                        )

                # Check for empty except blocks
                for handler in child.handlers:
                    if len(handler.body) == 1 and isinstance(handler.body[0], ast.Pass):
                        self._add_error_handling_issue(
                            file_path, handler.lineno, "empty_except",
                            "medium", "Empty except block silently ignores errors",
                            "Add proper error handling or logging",
                            node.name, lines[handler.lineno - 1].strip() if handler.lineno <= len(lines) else ""
                        )

    def _add_error_handling_issue(self, file_path: str, line_number: int, issue_type: str,
                                 severity: str, description: str, suggestion: str,
                                 function_name: str, code_snippet: str) -> None:
        """Add an error handling issue to the list."""
        issue = ErrorHandlingIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            description=description,
            suggestion=suggestion,
            function_name=function_name,
            code_snippet=code_snippet
        )
        self.issues.append(issue)

    def _calculate_coverage_score(self, protected_operations: int, risky_operations: int) -> float:
        """Calculate error handling coverage score."""
        if risky_operations == 0:
            return 1.0
        return protected_operations / risky_operations

    def _calculate_quality_score(self) -> float:
        """Calculate error handling quality score based on issues."""
        if not self.issues:
            return 1.0

        # Weight issues by severity
        total_weight = sum(self.severity_weights.get(issue.severity, 0.5)
                          for issue in self.issues)

        # Normalize by number of issues (lower is better)
        max_possible_weight = len(self.issues) * 1.0  # All critical
        quality_score = 1.0 - (total_weight / max_possible_weight)

        return max(0.0, quality_score)

    def _generate_recommendations(self) -> None:
        """Generate recommendations based on audit results."""
        issue_types = {}
        for issue in self.issues:
            issue_types[issue.issue_type] = issue_types.get(issue.issue_type, 0) + 1

        if issue_types.get("unprotected_operation", 0) > 0:
            self.recommendations.append(
                "Implement try-catch blocks around risky operations like file I/O, network calls, and database operations"
            )

        if issue_types.get("bare_except", 0) > 0:
            self.recommendations.append(
                "Replace bare 'except:' clauses with specific exception types"
            )

        if issue_types.get("empty_except", 0) > 0:
            self.recommendations.append(
                "Add proper error handling logic instead of empty except blocks"
            )

        if issue_types.get("broad_exception", 0) > 0:
            self.recommendations.append(
                "Use specific exception types instead of catching generic 'Exception'"
            )

        # General recommendations
        self.recommendations.extend([
            "Add logging to error handlers for better debugging",
            "Consider using context managers (with statements) for resource management",
            "Implement proper cleanup in finally blocks where needed",
            "Add input validation to prevent errors at the source"
        ])

    def generate_report(self, report: ErrorHandlingAuditReport, output_path: str) -> None:
        """
        Generate a detailed error handling audit report.

        Args:
            report: ErrorHandlingAuditReport to generate report from
            output_path: Path to save the report
        """
        with open(output_path, "w") as f:
            f.write("# Error Handling Audit Report\n\n")
            f.write("## Summary\n")
            f.write(f"- **Total Files**: {report.total_files}\n")
            f.write(f"- **Total Functions**: {report.total_functions}\n")
            f.write(f"- **Functions with Error Handling**: {report.functions_with_error_handling}\n")
            f.write(f"- **Risky Operations**: {report.risky_operations}\n")
            f.write(f"- **Protected Operations**: {report.protected_operations}\n")
            f.write(f"- **Coverage Score**: {report.coverage_score:.2%}\n")
            f.write(f"- **Quality Score**: {report.quality_score:.2%}\n")
            f.write(f"- **Total Issues**: {len(report.issues)}\n\n")

            # Recommendations
            if report.recommendations:
                f.write("## Recommendations\n\n")
                for i, rec in enumerate(report.recommendations, 1):
                    f.write(f"{i}. {rec}\n")
                f.write("\n")

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
                        f.write(f"  - *Function*: `{issue.function_name}`\n")
                        f.write(f"  - *Code*: `{issue.code_snippet}`\n")
                        f.write(f"  - *Suggestion*: {issue.suggestion}\n\n")

        logger.info(f"Error handling audit report generated: {output_path}")

    def generate_error_handling_template(self, operation_type: str) -> str:
        """
        Generate error handling template for specific operation types.

        Args:
            operation_type: Type of operation (file, network, database, etc.)

        Returns:
            Template code as string
        """
        templates = {
            "file": """
try:
    with open(filename, 'r') as f:
        content = f.read()
    # Process content
except FileNotFoundError:
    logger.error(f"File not found: {filename}")
    # Handle missing file
except PermissionError:
    logger.error(f"Permission denied: {filename}")
    # Handle permission issues
except IOError as e:
    logger.error(f"I/O error reading {filename}: {e}")
    # Handle other I/O errors
""",
            "network": """
try:
    response = requests.get(url, timeout=30)
    response.raise_for_status()
    data = response.json()
except requests.exceptions.Timeout:
    logger.error(f"Request timeout for {url}")
    # Handle timeout
except requests.exceptions.ConnectionError:
    logger.error(f"Connection error for {url}")
    # Handle connection issues
except requests.exceptions.HTTPError as e:
    logger.error(f"HTTP error for {url}: {e}")
    # Handle HTTP errors
except requests.exceptions.RequestException as e:
    logger.error(f"Request error for {url}: {e}")
    # Handle other request errors
""",
            "database": """
try:
    with connection.cursor() as cursor:
        cursor.execute(query, params)
        result = cursor.fetchall()
    connection.commit()
except DatabaseError as e:
    logger.error(f"Database error: {e}")
    connection.rollback()
    # Handle database errors
except Exception as e:
    logger.error(f"Unexpected error: {e}")
    connection.rollback()
    raise
"""
        }

        return templates.get(operation_type, templates["file"])


def main():
    """Main function to run error handling audit."""
    auditor = ErrorHandlingAuditor()

    # Audit the dataset_pipeline directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report = auditor.audit_directory(current_dir)

    # Generate report
    report_path = os.path.join(current_dir, "error_handling_audit_report.md")
    auditor.generate_report(report, report_path)

    # Print summary


if __name__ == "__main__":
    main()
