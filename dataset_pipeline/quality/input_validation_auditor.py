#!/usr/bin/env python3
"""
Input Validation Auditor for Task 29
Comprehensive tool to audit, validate, and enhance input validation across the codebase.
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
class InputValidationIssue:
    """Represents an input validation issue found during audit."""
    file_path: str
    line_number: int
    issue_type: str
    severity: str
    description: str
    suggestion: str
    function_name: str
    parameter_name: str
    code_snippet: str

@dataclass
class InputValidationAuditReport:
    """Comprehensive input validation audit report."""
    total_files: int
    total_functions: int
    functions_with_validation: int
    total_parameters: int
    validated_parameters: int
    issues: list[InputValidationIssue]
    coverage_score: float
    quality_score: float
    recommendations: list[str]
    validation_patterns: dict[str, int]

class InputValidationAuditor:
    """
    Comprehensive input validation auditor for Python codebases.

    Analyzes Python files to identify missing, incomplete, or poor input validation
    and provides suggestions for improvement.
    """

    def __init__(self, config: dict[str, Any] | None = None):
        """
        Initialize the input validation auditor.

        Args:
            config: Optional configuration dictionary
        """
        self.config = config or {}
        self.issues = []
        self.recommendations = []
        self.validation_patterns = {}

        # Common validation patterns to look for
        self.validation_keywords = {
            "type_checks": ["isinstance", "type(", "__class__"],
            "null_checks": ["is None", "== None", "not None", "!= None"],
            "length_checks": ["len(", "__len__"],
            "range_checks": [">", "<", ">=", "<=", "range("],
            "format_checks": ["re.match", "re.search", "match(", "search("],
            "assertion_checks": ["assert", "raise ValueError", "raise TypeError"],
            "library_validation": ["pydantic", "marshmallow", "cerberus", "schema"]
        }

        # Risky parameter types that need validation
        self.risky_parameter_patterns = {
            "file_paths": ["path", "filename", "file", "dir"],
            "urls": ["url", "uri", "link", "endpoint"],
            "user_input": ["input", "data", "content", "text", "message"],
            "ids": ["id", "key", "identifier", "uuid"],
            "numbers": ["count", "size", "length", "amount", "value"],
            "collections": ["list", "dict", "array", "items"]
        }

        # Severity weights for scoring
        self.severity_weights = {
            "critical": 1.0,
            "high": 0.8,
            "medium": 0.5,
            "low": 0.2
        }

    def audit_directory(self, directory_path: str) -> InputValidationAuditReport:
        """
        Audit all Python files in a directory for input validation quality.

        Args:
            directory_path: Path to directory to audit

        Returns:
            InputValidationAuditReport with comprehensive analysis results
        """
        logger.info(f"Starting input validation audit of directory: {directory_path}")

        self.issues = []
        self.recommendations = []
        self.validation_patterns = dict.fromkeys(self.validation_keywords.keys(), 0)

        total_files = 0
        total_functions = 0
        functions_with_validation = 0
        total_parameters = 0
        validated_parameters = 0

        # Find all Python files
        python_files = list(Path(directory_path).glob("*.py"))

        for file_path in python_files:
            if file_path.name.startswith("__"):
                continue  # Skip __init__.py, __pycache__, etc.

            total_files += 1
            file_stats = self._audit_file(str(file_path))

            total_functions += file_stats["functions"]
            functions_with_validation += file_stats["functions_with_validation"]
            total_parameters += file_stats["parameters"]
            validated_parameters += file_stats["validated_parameters"]

        # Calculate scores
        coverage_score = self._calculate_coverage_score(
            validated_parameters, total_parameters
        )
        quality_score = self._calculate_quality_score()

        # Generate recommendations
        self._generate_recommendations()

        report = InputValidationAuditReport(
            total_files=total_files,
            total_functions=total_functions,
            functions_with_validation=functions_with_validation,
            total_parameters=total_parameters,
            validated_parameters=validated_parameters,
            issues=self.issues.copy(),
            coverage_score=coverage_score,
            quality_score=quality_score,
            recommendations=self.recommendations.copy(),
            validation_patterns=self.validation_patterns.copy()
        )

        logger.info(f"Input validation audit complete: {total_files} files, {len(self.issues)} issues found")
        return report

    def _audit_file(self, file_path: str) -> dict[str, int]:
        """
        Audit a single Python file for input validation issues.

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
            functions_with_validation = 0
            parameters = 0
            validated_parameters = 0

            # Analyze functions
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions += 1

                    # Skip special methods and test functions
                    if (node.name.startswith("__") or
                        node.name.startswith("test_") or
                        "test" in file_path.lower()):
                        continue

                    # Count parameters (excluding self)
                    func_params = [arg for arg in node.args.args if arg.arg != "self"]
                    parameters += len(func_params)

                    # Check for validation in function
                    has_validation, validated_param_count = self._analyze_function_validation(
                        node, func_params, file_path, lines
                    )

                    if has_validation:
                        functions_with_validation += 1

                    validated_parameters += validated_param_count

                    # Check for missing validation on risky parameters
                    self._check_risky_parameters(node, func_params, file_path, lines)

            return {
                "functions": functions,
                "functions_with_validation": functions_with_validation,
                "parameters": parameters,
                "validated_parameters": validated_parameters
            }

        except Exception as e:
            logger.error(f"Error auditing file {file_path}: {e}")
            return {
                "functions": 0, "functions_with_validation": 0,
                "parameters": 0, "validated_parameters": 0
            }

    def _analyze_function_validation(self, node: ast.FunctionDef, parameters: list[ast.arg],
                                   file_path: str, lines: list[str]) -> tuple[bool, int]:
        """Analyze validation patterns in a function."""
        function_content = ast.get_source_segment(
            "\n".join(lines), node
        ) if hasattr(ast, "get_source_segment") else ""

        if not function_content:
            # Fallback: get function content by line numbers
            start_line = node.lineno - 1
            end_line = getattr(node, "end_lineno", start_line + 20) or start_line + 20
            function_content = "\n".join(lines[start_line:min(end_line, len(lines))])

        has_validation = False
        validated_param_count = 0

        # Check for validation patterns
        for pattern_type, keywords in self.validation_keywords.items():
            for keyword in keywords:
                if keyword in function_content:
                    has_validation = True
                    if pattern_type in self.validation_patterns:
                        self.validation_patterns[pattern_type] += 1
                    break

        # Count validated parameters (simplified heuristic)
        for param in parameters:
            param_name = param.arg
            if any(param_name in function_content and keyword in function_content
                   for keywords in self.validation_keywords.values()
                   for keyword in keywords):
                validated_param_count += 1

        return has_validation, validated_param_count

    def _check_risky_parameters(self, node: ast.FunctionDef, parameters: list[ast.arg],
                               file_path: str, lines: list[str]) -> None:
        """Check for risky parameters that need validation."""
        for param in parameters:
            param_name = param.arg.lower()

            # Check if parameter name suggests it needs validation
            for risk_type, patterns in self.risky_parameter_patterns.items():
                if any(pattern in param_name for pattern in patterns):
                    # Check if this parameter has validation
                    if not self._parameter_has_validation(node, param.arg, lines):
                        severity = self._get_parameter_severity(risk_type)

                        self._add_validation_issue(
                            file_path, node.lineno, "missing_validation",
                            severity, f"Parameter '{param.arg}' lacks input validation",
                            f"Add validation for {risk_type} parameter '{param.arg}'",
                            node.name, param.arg,
                            lines[node.lineno - 1].strip() if node.lineno <= len(lines) else ""
                        )

    def _parameter_has_validation(self, node: ast.FunctionDef, param_name: str, lines: list[str]) -> bool:
        """Check if a specific parameter has validation."""
        # Simple heuristic: look for parameter name near validation keywords
        start_line = node.lineno - 1
        end_line = getattr(node, "end_lineno", start_line + 20) or start_line + 20
        function_lines = lines[start_line:min(end_line, len(lines))]
        function_content = "\n".join(function_lines)

        # Check if parameter is mentioned with validation keywords
        for keywords in self.validation_keywords.values():
            for keyword in keywords:
                if param_name in function_content and keyword in function_content:
                    # More sophisticated check: parameter and keyword in same line or nearby
                    for line in function_lines:
                        if param_name in line and keyword in line:
                            return True

        return False

    def _get_parameter_severity(self, risk_type: str) -> str:
        """Get severity level for different parameter risk types."""
        severity_map = {
            "file_paths": "high",      # File path injection risks
            "urls": "high",            # URL injection risks
            "user_input": "medium",    # General user input
            "ids": "medium",           # ID validation
            "numbers": "low",          # Numeric validation
            "collections": "medium"    # Collection validation
        }
        return severity_map.get(risk_type, "medium")

    def _add_validation_issue(self, file_path: str, line_number: int, issue_type: str,
                             severity: str, description: str, suggestion: str,
                             function_name: str, parameter_name: str, code_snippet: str) -> None:
        """Add an input validation issue to the list."""
        issue = InputValidationIssue(
            file_path=file_path,
            line_number=line_number,
            issue_type=issue_type,
            severity=severity,
            description=description,
            suggestion=suggestion,
            function_name=function_name,
            parameter_name=parameter_name,
            code_snippet=code_snippet
        )
        self.issues.append(issue)

    def _calculate_coverage_score(self, validated_parameters: int, total_parameters: int) -> float:
        """Calculate input validation coverage score."""
        if total_parameters == 0:
            return 1.0
        return validated_parameters / total_parameters

    def _calculate_quality_score(self) -> float:
        """Calculate input validation quality score based on issues."""
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

        if issue_types.get("missing_validation", 0) > 0:
            self.recommendations.append(
                "Implement input validation for parameters, especially file paths, URLs, and user input"
            )

        # Check which validation patterns are underused
        total_patterns = sum(self.validation_patterns.values())
        if total_patterns > 0:
            for pattern, count in self.validation_patterns.items():
                if count / total_patterns < 0.1:  # Less than 10% usage
                    if pattern == "type_checks":
                        self.recommendations.append("Add more type checking with isinstance()")
                    elif pattern == "null_checks":
                        self.recommendations.append("Add null/None value checks")
                    elif pattern == "length_checks":
                        self.recommendations.append("Add length validation for strings and collections")
                    elif pattern == "range_checks":
                        self.recommendations.append("Add range validation for numeric inputs")
                    elif pattern == "format_checks":
                        self.recommendations.append("Add format validation using regular expressions")

        # General recommendations
        self.recommendations.extend([
            "Consider using validation libraries like Pydantic or Marshmallow",
            "Add comprehensive docstrings documenting expected input formats",
            "Implement consistent error messages for validation failures",
            "Use type hints to document expected parameter types"
        ])

    def generate_report(self, report: InputValidationAuditReport, output_path: str) -> None:
        """
        Generate a detailed input validation audit report.

        Args:
            report: InputValidationAuditReport to generate report from
            output_path: Path to save the report
        """
        with open(output_path, "w") as f:
            f.write("# Input Validation Audit Report\n\n")
            f.write("## Summary\n")
            f.write(f"- **Total Files**: {report.total_files}\n")
            f.write(f"- **Total Functions**: {report.total_functions}\n")
            f.write(f"- **Functions with Validation**: {report.functions_with_validation}\n")
            f.write(f"- **Total Parameters**: {report.total_parameters}\n")
            f.write(f"- **Validated Parameters**: {report.validated_parameters}\n")
            f.write(f"- **Coverage Score**: {report.coverage_score:.2%}\n")
            f.write(f"- **Quality Score**: {report.quality_score:.2%}\n")
            f.write(f"- **Total Issues**: {len(report.issues)}\n\n")

            # Validation patterns usage
            f.write("## Validation Patterns Usage\n\n")
            for pattern, count in report.validation_patterns.items():
                f.write(f"- **{pattern.replace('_', ' ').title()}**: {count} occurrences\n")
            f.write("\n")

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
                        f.write(f"  - *Parameter*: `{issue.parameter_name}`\n")
                        f.write(f"  - *Suggestion*: {issue.suggestion}\n\n")

        logger.info(f"Input validation audit report generated: {output_path}")

    def generate_validation_template(self, parameter_type: str) -> str:
        """
        Generate input validation template for specific parameter types.

        Args:
            parameter_type: Type of parameter (file_path, url, user_input, etc.)

        Returns:
            Template code as string
        """
        templates = {
            "file_path": '''
def validate_file_path(file_path: str) -> str:
    """Validate file path parameter."""
    if not isinstance(file_path, str):
        raise TypeError("File path must be a string")

    if not file_path.strip():
        raise ValueError("File path cannot be empty")

    # Check for path traversal attempts
    if '..' in file_path or file_path.startswith('/'):
        raise ValueError("Invalid file path: potential security risk")

    # Normalize path
    import os
    normalized_path = os.path.normpath(file_path)

    return normalized_path
''',
            "url": r'''
def validate_url(url: str) -> str:
    """Validate URL parameter."""
    if not isinstance(url, str):
        raise TypeError("URL must be a string")

    if not url.strip():
        raise ValueError("URL cannot be empty")

    # Basic URL format validation
    import re
    url_pattern = re.compile(
        r'^https?://'  # http:// or https://
        r'(?:(?:[A-Z0-9](?:[A-Z0-9-]{0,61}[A-Z0-9])?\.)+[A-Z]{2,6}\.?|'  # domain...
        r'localhost|'  # localhost...
        r'\d{1,3}\.\d{1,3}\.\d{1,3}\.\d{1,3})'  # ...or ip
        r'(?::\d+)?'  # optional port
        r'(?:/?|[/?]\S+)$', re.IGNORECASE)

    if not url_pattern.match(url):
        raise ValueError("Invalid URL format")

    return url
''',
            "user_input": '''
def validate_user_input(user_input: str, max_length: int = 1000) -> str:
    """Validate user input parameter."""
    if not isinstance(user_input, str):
        raise TypeError("User input must be a string")

    if len(user_input) > max_length:
        raise ValueError(f"Input too long: {len(user_input)} > {max_length}")

    # Check for potentially malicious content
    dangerous_patterns = ['<script', 'javascript:', 'data:']
    user_input_lower = user_input.lower()

    for pattern in dangerous_patterns:
        if pattern in user_input_lower:
            raise ValueError("Input contains potentially dangerous content")

    return user_input.strip()
'''
        }

        return templates.get(parameter_type, templates["user_input"])


def main():
    """Main function to run input validation audit."""
    auditor = InputValidationAuditor()

    # Audit the dataset_pipeline directory
    current_dir = os.path.dirname(os.path.abspath(__file__))
    report = auditor.audit_directory(current_dir)

    # Generate report
    report_path = os.path.join(current_dir, "input_validation_audit_report.md")
    auditor.generate_report(report, report_path)

    # Print summary


if __name__ == "__main__":
    main()
