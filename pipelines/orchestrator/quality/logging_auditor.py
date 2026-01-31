#!/usr/bin/env python3
"""
Comprehensive Logging Auditor for Pixelated Empathy AI
Analyzes and improves logging throughout the codebase for production monitoring.
"""

import ast
import json
import logging
import re
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class LoggingIssue:
    """Represents a logging issue found in the code."""
    file_path: str
    line_number: int
    function_name: str
    issue_type: str
    severity: str
    description: str
    suggestion: str
    code_snippet: str

@dataclass
class LoggingStats:
    """Statistics about logging coverage."""
    total_functions: int
    functions_with_logging: int
    total_files: int
    files_with_logging: int
    logging_coverage_percent: float
    critical_functions_without_logging: int
    error_handling_with_logging: int
    total_error_handlers: int

class LoggingAuditor:
    """Comprehensive logging auditor for production readiness."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.issues: list[LoggingIssue] = []
        self.stats = LoggingStats(0, 0, 0, 0, 0.0, 0, 0, 0)

        # Critical function patterns that must have logging
        self.critical_patterns = [
            r"def\s+(?:process|validate|generate|create|save|load|export|import)",
            r"def\s+(?:train|predict|infer|analyze|compute|calculate)",
            r"def\s+(?:handle|manage|execute|run|start|stop)",
            r"def\s+(?:connect|disconnect|authenticate|authorize)",
            r"def\s+(?:backup|restore|migrate|sync|update)",
        ]

        # Error handling patterns
        self.error_patterns = [
            r"except\s+\w+",
            r"try:",
            r"raise\s+\w+",
            r"assert\s+",
        ]

        # Logging patterns to detect existing logging (more precise)
        self.logging_patterns = [
            r"logging\.(debug|info|warning|error|critical|exception)",
            r"logger\.(debug|info|warning|error|critical|exception)",
            r"log\.(debug|info|warning|error|critical|exception)",
        ]

        # Setup logging for the auditor itself
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def audit_directory(self, directory: str | None = None) -> dict[str, Any]:
        """Audit logging throughout the specified directory."""
        if directory is None:
            directory = self.root_path

        self.logger.info(f"Starting logging audit of directory: {directory}")

        # Reset stats
        self.issues = []
        self.stats = LoggingStats(0, 0, 0, 0, 0.0, 0, 0, 0)

        # Find all Python files
        python_files = list(Path(directory).rglob("*.py"))
        self.stats.total_files = len(python_files)

        files_with_logging = 0
        total_functions = 0
        functions_with_logging = 0
        critical_functions_without_logging = 0
        error_handlers_with_logging = 0
        total_error_handlers = 0

        for file_path in python_files:
            try:
                file_stats = self._audit_file(file_path)
                total_functions += file_stats["total_functions"]
                functions_with_logging += file_stats["functions_with_logging"]
                critical_functions_without_logging += file_stats["critical_without_logging"]
                error_handlers_with_logging += file_stats["error_handlers_with_logging"]
                total_error_handlers += file_stats["total_error_handlers"]

                if file_stats["has_logging"]:
                    files_with_logging += 1

            except Exception as e:
                self.logger.error(f"Error auditing file {file_path}: {e}")
                self.issues.append(LoggingIssue(
                    file_path=str(file_path),
                    line_number=0,
                    function_name="FILE_ERROR",
                    issue_type="AUDIT_ERROR",
                    severity="HIGH",
                    description=f"Failed to audit file: {e}",
                    suggestion="Fix syntax errors or file encoding issues",
                    code_snippet=""
                ))

        # Update stats
        self.stats.total_functions = total_functions
        self.stats.functions_with_logging = functions_with_logging
        self.stats.files_with_logging = files_with_logging
        self.stats.logging_coverage_percent = (
            (functions_with_logging / total_functions * 100) if total_functions > 0 else 0
        )
        self.stats.critical_functions_without_logging = critical_functions_without_logging
        self.stats.error_handling_with_logging = error_handlers_with_logging
        self.stats.total_error_handlers = total_error_handlers

        self.logger.info(f"Logging audit completed. Found {len(self.issues)} issues.")

        return {
            "stats": asdict(self.stats),
            "issues": [asdict(issue) for issue in self.issues],
            "summary": self._generate_summary()
        }

    def _audit_file(self, file_path: Path) -> dict[str, Any]:
        """Audit logging in a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {
                    "total_functions": 0,
                    "functions_with_logging": 0,
                    "has_logging": False,
                    "critical_without_logging": 0,
                    "error_handlers_with_logging": 0,
                    "total_error_handlers": 0
                }

            lines = content.split("\n")
            has_logging = any(re.search(pattern, content) for pattern in self.logging_patterns)

            # Analyze functions
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            total_functions = len(functions)
            functions_with_logging = 0
            critical_without_logging = 0

            # Analyze error handlers
            error_handlers = self._find_error_handlers(content)
            total_error_handlers = len(error_handlers)
            error_handlers_with_logging = 0

            for func in functions:
                func_content = self._get_function_content(func, lines)
                func_has_logging = any(re.search(pattern, func_content) for pattern in self.logging_patterns)

                if func_has_logging:
                    functions_with_logging += 1
                # Check if it's a critical function
                elif self._is_critical_function(func.name):
                    critical_without_logging += 1
                    self.issues.append(LoggingIssue(
                        file_path=str(file_path),
                        line_number=func.lineno,
                        function_name=func.name,
                        issue_type="MISSING_LOGGING",
                        severity="HIGH",
                        description=f"Critical function '{func.name}' lacks logging",
                        suggestion=self._generate_logging_suggestion(func.name, "critical"),
                        code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content
                    ))
                else:
                    self.issues.append(LoggingIssue(
                        file_path=str(file_path),
                        line_number=func.lineno,
                        function_name=func.name,
                        issue_type="MISSING_LOGGING",
                        severity="MEDIUM",
                        description=f"Function '{func.name}' lacks logging",
                        suggestion=self._generate_logging_suggestion(func.name, "standard"),
                        code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content
                    ))

            # Check error handlers
            for handler_line, handler_content in error_handlers:
                if any(re.search(pattern, handler_content) for pattern in self.logging_patterns):
                    error_handlers_with_logging += 1
                else:
                    self.issues.append(LoggingIssue(
                        file_path=str(file_path),
                        line_number=handler_line,
                        function_name="ERROR_HANDLER",
                        issue_type="MISSING_ERROR_LOGGING",
                        severity="HIGH",
                        description="Error handler without logging",
                        suggestion="Add logging.exception() or logging.error() to capture error details",
                        code_snippet=handler_content
                    ))

            # Check for print statements (should be replaced with proper logging)
            print_statements = re.finditer(r"\bprint\s*\(", content)
            for match in print_statements:
                line_num = content[:match.start()].count("\n") + 1
                self.issues.append(LoggingIssue(
                    file_path=str(file_path),
                    line_number=line_num,
                    function_name="PRINT_STATEMENT",
                    issue_type="IMPROPER_LOGGING",
                    severity="MEDIUM",
                    description="Using print() instead of proper logging",
                    suggestion="Replace print() with appropriate logging level (info, debug, warning, error)",
                    code_snippet=lines[line_num-1] if line_num <= len(lines) else ""
                ))

            return {
                "total_functions": total_functions,
                "functions_with_logging": functions_with_logging,
                "has_logging": has_logging,
                "critical_without_logging": critical_without_logging,
                "error_handlers_with_logging": error_handlers_with_logging,
                "total_error_handlers": total_error_handlers
            }

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return {
                "total_functions": 0,
                "functions_with_logging": 0,
                "has_logging": False,
                "critical_without_logging": 0,
                "error_handlers_with_logging": 0,
                "total_error_handlers": 0
            }

    def _find_error_handlers(self, content: str) -> list[tuple[int, str]]:
        """Find error handling blocks in the code."""
        handlers = []
        lines = content.split("\n")

        for i, line in enumerate(lines):
            stripped_line = line.strip()
            if re.search(r"except\s+\w+", stripped_line):
                # Get the except block content
                block_content = [line]
                j = i + 1
                base_indent = len(line) - len(line.lstrip())

                while j < len(lines):
                    current_line = lines[j]
                    if current_line.strip() == "":
                        j += 1
                        continue

                    current_indent = len(current_line) - len(current_line.lstrip())
                    # If we hit a line at the same or lower indentation level, we're done
                    if current_indent <= base_indent and current_line.strip():
                        break

                    block_content.append(current_line)
                    j += 1

                handlers.append((i + 1, "\n".join(block_content)))

        return handlers

    def _get_function_content(self, func_node: ast.FunctionDef, lines: list[str]) -> str:
        """Extract function content from source lines."""
        start_line = func_node.lineno - 1
        end_line = func_node.end_lineno if hasattr(func_node, "end_lineno") else start_line + 10

        end_line = min(end_line, len(lines))

        return "\n".join(lines[start_line:end_line])

    def _is_critical_function(self, func_name: str) -> bool:
        """Check if a function is critical and must have logging."""
        return any(re.search(pattern, f"def {func_name}") for pattern in self.critical_patterns)

    def _generate_logging_suggestion(self, func_name: str, func_type: str) -> str:
        """Generate logging suggestions for functions."""
        if func_type == "critical":
            return f"""Add comprehensive logging to {func_name}:
logger.info(f"Starting {func_name} with parameters: {{params}}")
try:
    # function logic
    logger.info(f"{func_name} completed successfully")
    return result
except Exception as e:
    logger.error(f"Error in {func_name}: {{e}}", exc_info=True)
    raise"""
        return f"""Add basic logging to {func_name}:
logger.debug(f"Executing {func_name}")
# Add error handling with logging if needed"""

    def _generate_summary(self) -> dict[str, Any]:
        """Generate audit summary."""
        severity_counts = {}
        issue_type_counts = {}

        for issue in self.issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            issue_type_counts[issue.issue_type] = issue_type_counts.get(issue.issue_type, 0) + 1

        return {
            "total_issues": len(self.issues),
            "severity_breakdown": severity_counts,
            "issue_type_breakdown": issue_type_counts,
            "logging_coverage": f"{self.stats.logging_coverage_percent:.1f}%",
            "files_with_logging": f"{self.stats.files_with_logging}/{self.stats.total_files}",
            "critical_functions_missing_logging": self.stats.critical_functions_without_logging,
            "error_handlers_coverage": f"{self.stats.error_handling_with_logging}/{self.stats.total_error_handlers}" if self.stats.total_error_handlers > 0 else "0/0"
        }

    def generate_report(self, output_file: str | None = None) -> str:
        """Generate comprehensive logging audit report."""
        if not output_file:
            output_file = f"logging_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "auditor_version": "1.0.0",
            "stats": asdict(self.stats),
            "summary": self._generate_summary(),
            "issues": [asdict(issue) for issue in self.issues],
            "recommendations": self._generate_recommendations()
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Logging audit report saved to {output_file}")
        return output_file

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations for improving logging."""
        recommendations = []

        if self.stats.logging_coverage_percent < 50:
            recommendations.append("CRITICAL: Logging coverage is below 50%. Implement comprehensive logging strategy.")
        elif self.stats.logging_coverage_percent < 80:
            recommendations.append("WARNING: Logging coverage is below 80%. Consider adding logging to more functions.")

        if self.stats.critical_functions_without_logging > 0:
            recommendations.append(f"HIGH PRIORITY: {self.stats.critical_functions_without_logging} critical functions lack logging.")

        if self.stats.total_error_handlers > 0:
            error_coverage = (self.stats.error_handling_with_logging / self.stats.total_error_handlers) * 100
            if error_coverage < 90:
                recommendations.append(f"Error handler logging coverage is {error_coverage:.1f}%. All error handlers should log exceptions.")

        # Check for print statements
        print_issues = [issue for issue in self.issues if issue.issue_type == "IMPROPER_LOGGING"]
        if print_issues:
            recommendations.append(f"Replace {len(print_issues)} print() statements with proper logging.")

        recommendations.extend([
            "Implement structured logging with consistent format across all modules",
            "Add performance logging for critical operations",
            "Implement log rotation and retention policies for production",
            "Add correlation IDs for request tracing",
            "Configure different log levels for development vs production"
        ])

        return recommendations

    def fix_logging_issues(self, auto_fix: bool = False) -> dict[str, Any]:
        """Generate fixes for logging issues."""
        fixes = []

        for issue in self.issues:
            if issue.issue_type == "MISSING_LOGGING" and issue.severity == "HIGH":
                fix = {
                    "file": issue.file_path,
                    "line": issue.line_number,
                    "type": "ADD_LOGGING",
                    "suggestion": issue.suggestion,
                    "priority": "HIGH"
                }
                fixes.append(fix)

        if auto_fix:
            self.logger.warning("Auto-fix not implemented yet. Manual fixes required.")

        return {
            "total_fixes": len(fixes),
            "fixes": fixes,
            "auto_fix_applied": False
        }

def main():
    """Main function for running the logging auditor."""
    auditor = LoggingAuditor()

    # Audit the current directory
    auditor.audit_directory(".")

    # Generate report
    auditor.generate_report()

    # Print summary

    # Show top issues
    high_severity_issues = [issue for issue in auditor.issues if issue.severity == "HIGH"]
    if high_severity_issues:
        for _issue in high_severity_issues[:5]:  # Show first 5
            pass

if __name__ == "__main__":
    main()
