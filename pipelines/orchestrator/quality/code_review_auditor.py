#!/usr/bin/env python3
"""
Comprehensive Code Review and Refactoring Auditor for Pixelated Empathy AI
Analyzes code quality, complexity, maintainability, and provides refactoring suggestions.
"""

import ast
import json
import logging
import re
from collections import defaultdict
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class CodeIssue:
    """Represents a code quality issue found during review."""
    file_path: str
    line_number: int
    function_name: str
    class_name: str
    issue_type: str
    severity: str
    description: str
    suggestion: str
    code_snippet: str
    complexity_score: int

@dataclass
class CodeStats:
    """Statistics about code quality and complexity."""
    total_files: int
    total_functions: int
    total_classes: int
    total_lines: int
    average_function_length: float
    average_complexity: float
    high_complexity_functions: int
    duplicate_code_blocks: int
    code_smells: int
    maintainability_score: float

class CodeReviewAuditor:
    """Comprehensive code review and refactoring auditor."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.issues: list[CodeIssue] = []
        self.stats = CodeStats(0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0.0)

        # Code smell patterns
        self.code_smells = {
            "long_function": {"threshold": 50, "severity": "MEDIUM"},
            "too_many_parameters": {"threshold": 5, "severity": "MEDIUM"},
            "deep_nesting": {"threshold": 4, "severity": "HIGH"},
            "long_class": {"threshold": 500, "severity": "MEDIUM"},
            "god_class": {"threshold": 20, "severity": "HIGH"},  # Too many methods
            "duplicate_code": {"threshold": 5, "severity": "HIGH"},  # Similar lines
            "magic_numbers": {"pattern": r"\b\d{2,}\b", "severity": "LOW"},
            "commented_code": {"pattern": r"^\s*#.*def |^\s*#.*class |^\s*#.*import", "severity": "LOW"},
            "todo_fixme": {"pattern": r"#\s*(TODO|FIXME|HACK|XXX)", "severity": "MEDIUM"},
        }

        # Refactoring patterns
        self.refactoring_patterns = {
            "extract_method": "Long functions should be broken into smaller methods",
            "extract_class": "Large classes should be split into focused classes",
            "replace_magic_number": "Magic numbers should be replaced with named constants",
            "simplify_conditional": "Complex conditionals should be simplified",
            "remove_duplicate": "Duplicate code should be extracted to common functions",
            "improve_naming": "Variable and function names should be more descriptive",
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def audit_directory(self, directory: str | None = None) -> dict[str, Any]:
        """Audit code quality throughout the specified directory."""
        if directory is None:
            directory = self.root_path

        self.logger.info(f"Starting code review audit of directory: {directory}")

        # Reset stats
        self.issues = []
        self.stats = CodeStats(0, 0, 0, 0, 0.0, 0.0, 0, 0, 0, 0.0)

        # Find all Python files
        python_files = list(Path(directory).rglob("*.py"))
        self.stats.total_files = len(python_files)

        total_functions = 0
        total_classes = 0
        total_lines = 0
        function_lengths = []
        complexity_scores = []
        high_complexity_count = 0
        code_smells_count = 0

        # Track duplicate code
        code_blocks = defaultdict(list)

        for file_path in python_files:
            try:
                file_stats = self._audit_file(file_path, code_blocks)
                total_functions += file_stats["total_functions"]
                total_classes += file_stats["total_classes"]
                total_lines += file_stats["total_lines"]
                function_lengths.extend(file_stats["function_lengths"])
                complexity_scores.extend(file_stats["complexity_scores"])
                high_complexity_count += file_stats["high_complexity_functions"]
                code_smells_count += file_stats["code_smells"]

            except Exception as e:
                self.logger.error(f"Error auditing file {file_path}: {e}")
                self.issues.append(CodeIssue(
                    file_path=str(file_path),
                    line_number=0,
                    function_name="FILE_ERROR",
                    class_name="",
                    issue_type="AUDIT_ERROR",
                    severity="HIGH",
                    description=f"Failed to audit file: {e}",
                    suggestion="Fix syntax errors or file encoding issues",
                    code_snippet="",
                    complexity_score=0
                ))

        # Detect duplicate code blocks
        duplicate_count = self._detect_duplicate_code(code_blocks)

        # Update stats
        self.stats.total_functions = total_functions
        self.stats.total_classes = total_classes
        self.stats.total_lines = total_lines
        self.stats.average_function_length = sum(function_lengths) / len(function_lengths) if function_lengths else 0
        self.stats.average_complexity = sum(complexity_scores) / len(complexity_scores) if complexity_scores else 0
        self.stats.high_complexity_functions = high_complexity_count
        self.stats.duplicate_code_blocks = duplicate_count
        self.stats.code_smells = code_smells_count
        self.stats.maintainability_score = self._calculate_maintainability_score()

        self.logger.info(f"Code review audit completed. Found {len(self.issues)} issues.")

        return {
            "stats": asdict(self.stats),
            "issues": [asdict(issue) for issue in self.issues],
            "summary": self._generate_summary()
        }

    def _audit_file(self, file_path: Path, code_blocks: dict) -> dict[str, Any]:
        """Audit code quality in a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {
                    "total_functions": 0,
                    "total_classes": 0,
                    "total_lines": 0,
                    "function_lengths": [],
                    "complexity_scores": [],
                    "high_complexity_functions": 0,
                    "code_smells": 0
                }

            lines = content.split("\n")
            total_lines = len(lines)

            # Analyze functions and classes
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            function_lengths = []
            complexity_scores = []
            high_complexity_count = 0
            code_smells_count = 0

            # Analyze functions
            for func in functions:
                func_stats = self._analyze_function(func, lines, str(file_path))
                function_lengths.append(func_stats["length"])
                complexity_scores.append(func_stats["complexity"])

                if func_stats["complexity"] > 10:  # High complexity threshold
                    high_complexity_count += 1

                code_smells_count += func_stats["code_smells"]

                # Store code blocks for duplicate detection
                func_content = self._get_function_content(func, lines)
                code_blocks[self._normalize_code(func_content)].append((str(file_path), func.lineno, func.name))

            # Analyze classes
            for cls in classes:
                class_stats = self._analyze_class(cls, lines, str(file_path))
                code_smells_count += class_stats["code_smells"]

            # Check file-level code smells
            file_smells = self._check_file_level_smells(content, str(file_path))
            code_smells_count += file_smells

            return {
                "total_functions": len(functions),
                "total_classes": len(classes),
                "total_lines": total_lines,
                "function_lengths": function_lengths,
                "complexity_scores": complexity_scores,
                "high_complexity_functions": high_complexity_count,
                "code_smells": code_smells_count
            }

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return {
                "total_functions": 0,
                "total_classes": 0,
                "total_lines": 0,
                "function_lengths": [],
                "complexity_scores": [],
                "high_complexity_functions": 0,
                "code_smells": 0
            }

    def _analyze_function(self, func_node: ast.FunctionDef, lines: list[str], file_path: str) -> dict[str, Any]:
        """Analyze a single function for code quality issues."""
        func_content = self._get_function_content(func_node, lines)
        func_lines = func_content.split("\n")
        func_length = len([line for line in func_lines if line.strip() and not line.strip().startswith("#")])

        # Calculate cyclomatic complexity
        complexity = self._calculate_complexity(func_node)

        # Check for code smells
        code_smells = 0

        # Long function
        if func_length > self.code_smells["long_function"]["threshold"]:
            code_smells += 1
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="LONG_FUNCTION",
                severity=self.code_smells["long_function"]["severity"],
                description=f"Function '{func_node.name}' is too long ({func_length} lines)",
                suggestion="Break this function into smaller, focused functions using Extract Method refactoring",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                complexity_score=complexity
            ))

        # Too many parameters
        param_count = len(func_node.args.args)
        if param_count > self.code_smells["too_many_parameters"]["threshold"]:
            code_smells += 1
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="TOO_MANY_PARAMETERS",
                severity=self.code_smells["too_many_parameters"]["severity"],
                description=f"Function '{func_node.name}' has too many parameters ({param_count})",
                suggestion="Consider using a parameter object or breaking the function into smaller parts",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                complexity_score=complexity
            ))

        # High complexity
        if complexity > 10:
            code_smells += 1
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="HIGH_COMPLEXITY",
                severity="HIGH",
                description=f"Function '{func_node.name}' has high cyclomatic complexity ({complexity})",
                suggestion="Simplify the function by reducing conditional statements and loops",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                complexity_score=complexity
            ))

        # Deep nesting
        max_nesting = self._calculate_max_nesting(func_node)
        if max_nesting > self.code_smells["deep_nesting"]["threshold"]:
            code_smells += 1
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="DEEP_NESTING",
                severity=self.code_smells["deep_nesting"]["severity"],
                description=f"Function '{func_node.name}' has deep nesting ({max_nesting} levels)",
                suggestion="Use early returns and guard clauses to reduce nesting levels",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                complexity_score=complexity
            ))

        return {
            "length": func_length,
            "complexity": complexity,
            "code_smells": code_smells,
            "max_nesting": max_nesting
        }

    def _analyze_class(self, class_node: ast.ClassDef, lines: list[str], file_path: str) -> dict[str, Any]:
        """Analyze a single class for code quality issues."""
        class_content = self._get_class_content(class_node, lines)
        class_lines = len([line for line in class_content.split("\n") if line.strip() and not line.strip().startswith("#")])

        # Count methods
        methods = [node for node in class_node.body if isinstance(node, ast.FunctionDef)]
        method_count = len(methods)

        code_smells = 0

        # Long class
        if class_lines > self.code_smells["long_class"]["threshold"]:
            code_smells += 1
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=class_node.lineno,
                function_name="",
                class_name=class_node.name,
                issue_type="LONG_CLASS",
                severity=self.code_smells["long_class"]["severity"],
                description=f"Class '{class_node.name}' is too long ({class_lines} lines)",
                suggestion="Break this class into smaller, focused classes using Extract Class refactoring",
                code_snippet=class_content[:200] + "..." if len(class_content) > 200 else class_content,
                complexity_score=0
            ))

        # God class (too many methods)
        if method_count > self.code_smells["god_class"]["threshold"]:
            code_smells += 1
            self.issues.append(CodeIssue(
                file_path=file_path,
                line_number=class_node.lineno,
                function_name="",
                class_name=class_node.name,
                issue_type="GOD_CLASS",
                severity=self.code_smells["god_class"]["severity"],
                description=f"Class '{class_node.name}' has too many methods ({method_count})",
                suggestion="Split this class into multiple classes with single responsibilities",
                code_snippet=class_content[:200] + "..." if len(class_content) > 200 else class_content,
                complexity_score=0
            ))

        return {
            "code_smells": code_smells,
            "method_count": method_count,
            "line_count": class_lines
        }

    def _check_file_level_smells(self, content: str, file_path: str) -> int:
        """Check for file-level code smells."""
        code_smells = 0
        lines = content.split("\n")

        # Magic numbers
        for i, line in enumerate(lines):
            if re.search(self.code_smells["magic_numbers"]["pattern"], line):
                # Skip common exceptions (0, 1, 2, etc.)
                numbers = re.findall(r"\b\d{2,}\b", line)
                for num in numbers:
                    if int(num) not in [0, 1, 2, 10, 100, 1000]:  # Common acceptable numbers
                        code_smells += 1
                        self.issues.append(CodeIssue(
                            file_path=file_path,
                            line_number=i + 1,
                            function_name="",
                            class_name="",
                            issue_type="MAGIC_NUMBER",
                            severity=self.code_smells["magic_numbers"]["severity"],
                            description=f"Magic number '{num}' found",
                            suggestion="Replace magic number with a named constant",
                            code_snippet=line.strip()[:200],
                            complexity_score=0
                        ))
                        break  # Only report one per line

        # Commented code
        for i, line in enumerate(lines):
            if re.search(self.code_smells["commented_code"]["pattern"], line):
                code_smells += 1
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i + 1,
                    function_name="",
                    class_name="",
                    issue_type="COMMENTED_CODE",
                    severity=self.code_smells["commented_code"]["severity"],
                    description="Commented out code found",
                    suggestion="Remove commented code or convert to proper documentation",
                    code_snippet=line.strip()[:200],
                    complexity_score=0
                ))

        # TODO/FIXME comments
        for i, line in enumerate(lines):
            if re.search(self.code_smells["todo_fixme"]["pattern"], line):
                code_smells += 1
                self.issues.append(CodeIssue(
                    file_path=file_path,
                    line_number=i + 1,
                    function_name="",
                    class_name="",
                    issue_type="TODO_FIXME",
                    severity=self.code_smells["todo_fixme"]["severity"],
                    description="TODO/FIXME comment found",
                    suggestion="Address the TODO/FIXME or create a proper issue tracker item",
                    code_snippet=line.strip()[:200],
                    complexity_score=0
                ))

        return code_smells

    def _calculate_complexity(self, node: ast.AST) -> int:
        """Calculate cyclomatic complexity of a function."""
        complexity = 1  # Base complexity

        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.ExceptHandler, ast.With, ast.AsyncWith)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1

        return complexity

    def _calculate_max_nesting(self, node: ast.AST) -> int:
        """Calculate maximum nesting depth in a function."""
        def get_nesting_depth(node, current_depth=0):
            max_depth = current_depth

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.If, ast.While, ast.For, ast.AsyncFor, ast.With, ast.AsyncWith, ast.Try)):
                    child_depth = get_nesting_depth(child, current_depth + 1)
                    max_depth = max(max_depth, child_depth)
                else:
                    child_depth = get_nesting_depth(child, current_depth)
                    max_depth = max(max_depth, child_depth)

            return max_depth

        return get_nesting_depth(node)

    def _get_function_content(self, func_node: ast.FunctionDef, lines: list[str]) -> str:
        """Extract function content from source lines."""
        start_line = func_node.lineno - 1
        end_line = func_node.end_lineno if hasattr(func_node, "end_lineno") else start_line + 20

        end_line = min(end_line, len(lines))

        return "\n".join(lines[start_line:end_line])

    def _get_class_content(self, class_node: ast.ClassDef, lines: list[str]) -> str:
        """Extract class content from source lines."""
        start_line = class_node.lineno - 1
        end_line = class_node.end_lineno if hasattr(class_node, "end_lineno") else start_line + 50

        end_line = min(end_line, len(lines))

        return "\n".join(lines[start_line:end_line])

    def _normalize_code(self, code: str) -> str:
        """Normalize code for duplicate detection."""
        # Remove comments and whitespace
        lines = []
        for line in code.split("\n"):
            line = re.sub(r"#.*$", "", line)  # Remove comments
            line = re.sub(r"\s+", " ", line.strip())  # Normalize whitespace
            if line:
                lines.append(line)
        return "\n".join(lines)

    def _detect_duplicate_code(self, code_blocks: dict) -> int:
        """Detect duplicate code blocks."""
        duplicate_count = 0

        for normalized_code, occurrences in code_blocks.items():
            if len(occurrences) > 1 and len(normalized_code.split("\n")) >= self.code_smells["duplicate_code"]["threshold"]:
                duplicate_count += len(occurrences) - 1

                for file_path, line_num, func_name in occurrences:
                    self.issues.append(CodeIssue(
                        file_path=file_path,
                        line_number=line_num,
                        function_name=func_name,
                        class_name="",
                        issue_type="DUPLICATE_CODE",
                        severity=self.code_smells["duplicate_code"]["severity"],
                        description=f"Duplicate code detected in function '{func_name}'",
                        suggestion="Extract common code into a shared function or method",
                        code_snippet=normalized_code[:200] + "..." if len(normalized_code) > 200 else normalized_code,
                        complexity_score=0
                    ))

        return duplicate_count

    def _calculate_maintainability_score(self) -> float:
        """Calculate overall maintainability score (0-100)."""
        if self.stats.total_functions == 0:
            return 100.0

        # Base score
        score = 100.0

        # Penalize high complexity
        if self.stats.average_complexity > 5:
            score -= min(30, (self.stats.average_complexity - 5) * 3)

        # Penalize long functions
        if self.stats.average_function_length > 20:
            score -= min(20, (self.stats.average_function_length - 20) * 0.5)

        # Penalize code smells
        smell_ratio = self.stats.code_smells / self.stats.total_functions
        score -= min(25, smell_ratio * 50)

        # Penalize duplicate code
        if self.stats.duplicate_code_blocks > 0:
            score -= min(15, self.stats.duplicate_code_blocks * 2)

        return max(0.0, score)

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
            "maintainability_score": f"{self.stats.maintainability_score:.1f}/100",
            "average_complexity": f"{self.stats.average_complexity:.1f}",
            "average_function_length": f"{self.stats.average_function_length:.1f} lines",
            "high_complexity_functions": self.stats.high_complexity_functions,
            "duplicate_code_blocks": self.stats.duplicate_code_blocks
        }

    def generate_report(self, output_file: str | None = None) -> str:
        """Generate comprehensive code review report."""
        if not output_file:
            output_file = f"code_review_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "auditor_version": "1.0.0",
            "stats": asdict(self.stats),
            "summary": self._generate_summary(),
            "issues": [asdict(issue) for issue in self.issues],
            "recommendations": self._generate_recommendations(),
            "refactoring_priorities": self._generate_refactoring_priorities()
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Code review report saved to {output_file}")
        return output_file

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations for code improvement."""
        recommendations = []

        if self.stats.maintainability_score < 60:
            recommendations.append("CRITICAL: Maintainability score is below 60. Immediate refactoring required.")
        elif self.stats.maintainability_score < 80:
            recommendations.append("WARNING: Maintainability score is below 80. Consider refactoring.")

        if self.stats.average_complexity > 10:
            recommendations.append(f"High average complexity ({self.stats.average_complexity:.1f}). Simplify complex functions.")

        if self.stats.average_function_length > 30:
            recommendations.append(f"Long average function length ({self.stats.average_function_length:.1f} lines). Break down large functions.")

        if self.stats.duplicate_code_blocks > 0:
            recommendations.append(f"Found {self.stats.duplicate_code_blocks} duplicate code blocks. Extract common functionality.")

        if self.stats.high_complexity_functions > 0:
            recommendations.append(f"{self.stats.high_complexity_functions} functions have high complexity. Prioritize refactoring these.")

        recommendations.extend([
            "Implement consistent naming conventions across the codebase",
            "Add comprehensive unit tests for complex functions",
            "Consider using design patterns to improve code structure",
            "Implement code formatting standards (black, isort)",
            "Set up pre-commit hooks for code quality checks"
        ])

        return recommendations

    def _generate_refactoring_priorities(self) -> list[dict[str, Any]]:
        """Generate prioritized refactoring suggestions."""
        priorities = []

        # Group issues by severity and type
        high_severity_issues = [issue for issue in self.issues if issue.severity == "HIGH"]
        complexity_issues = [issue for issue in self.issues if issue.issue_type == "HIGH_COMPLEXITY"]
        duplicate_issues = [issue for issue in self.issues if issue.issue_type == "DUPLICATE_CODE"]

        if high_severity_issues:
            priorities.append({
                "priority": 1,
                "category": "High Severity Issues",
                "count": len(high_severity_issues),
                "description": "Address critical code quality issues first",
                "estimated_effort": "High"
            })

        if complexity_issues:
            priorities.append({
                "priority": 2,
                "category": "High Complexity Functions",
                "count": len(complexity_issues),
                "description": "Simplify complex functions to improve maintainability",
                "estimated_effort": "Medium"
            })

        if duplicate_issues:
            priorities.append({
                "priority": 3,
                "category": "Duplicate Code",
                "count": len(duplicate_issues),
                "description": "Extract common functionality to reduce duplication",
                "estimated_effort": "Medium"
            })

        return priorities

def main():
    """Main function for running the code review auditor."""
    auditor = CodeReviewAuditor()

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
