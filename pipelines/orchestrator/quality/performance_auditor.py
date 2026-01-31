#!/usr/bin/env python3
"""
Comprehensive Performance Auditor for Pixelated Empathy AI
Analyzes performance bottlenecks, algorithmic complexity, and optimization opportunities.
"""

import ast
import json
import logging
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any


@dataclass
class PerformanceIssue:
    """Represents a performance issue found during analysis."""
    file_path: str
    line_number: int
    function_name: str
    class_name: str
    issue_type: str
    severity: str
    description: str
    suggestion: str
    code_snippet: str
    estimated_impact: str
    optimization_category: str

@dataclass
class PerformanceStats:
    """Statistics about performance characteristics."""
    total_files: int
    total_functions: int
    algorithmic_complexity_issues: int
    io_bottlenecks: int
    memory_inefficiencies: int
    cpu_intensive_operations: int
    database_query_issues: int
    loop_optimization_opportunities: int
    caching_opportunities: int
    overall_performance_score: float

class PerformanceAuditor:
    """Comprehensive performance auditor for production optimization."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.issues: list[PerformanceIssue] = []
        self.stats = PerformanceStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

        # Performance anti-patterns
        self.performance_patterns = {
            "nested_loops": {
                "pattern": r"for\s+\w+.*:\s*\n.*for\s+\w+.*:",
                "severity": "HIGH",
                "category": "ALGORITHMIC_COMPLEXITY"
            },
            "string_concatenation": {
                "pattern": r'\w+\s*\+=\s*["\'].*["\']',
                "severity": "MEDIUM",
                "category": "CPU_INTENSIVE"
            },
            "repeated_db_queries": {
                "pattern": r"for\s+\w+.*:\s*.*\.query\(",
                "severity": "HIGH",
                "category": "DATABASE_BOTTLENECK"
            },
            "file_io_in_loop": {
                "pattern": r"for\s+\w+.*:\s*.*open\(",
                "severity": "HIGH",
                "category": "IO_BOTTLENECK"
            },
            "inefficient_list_operations": {
                "pattern": r"\.append\(.*\)\s*\n.*for\s+",
                "severity": "MEDIUM",
                "category": "MEMORY_INEFFICIENCY"
            }
        }

        # Optimization opportunities
        self.optimization_patterns = {
            "list_comprehension": "Replace loops with list comprehensions for better performance",
            "generator_expression": "Use generator expressions for memory efficiency",
            "caching": "Implement caching for expensive computations",
            "vectorization": "Use numpy/pandas vectorized operations",
            "database_optimization": "Optimize database queries and use bulk operations",
            "async_operations": "Use async/await for I/O bound operations"
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def audit_directory(self, directory: str | None = None) -> dict[str, Any]:
        """Audit performance throughout the specified directory."""
        if directory is None:
            directory = self.root_path

        self.logger.info(f"Starting performance audit of directory: {directory}")

        # Reset stats
        self.issues = []
        self.stats = PerformanceStats(0, 0, 0, 0, 0, 0, 0, 0, 0, 0.0)

        # Find all Python files
        python_files = list(Path(directory).rglob("*.py"))
        self.stats.total_files = len(python_files)

        total_functions = 0
        algorithmic_issues = 0
        io_bottlenecks = 0
        memory_issues = 0
        cpu_issues = 0
        db_issues = 0
        loop_issues = 0
        caching_opportunities = 0

        for file_path in python_files:
            try:
                file_stats = self._audit_file(file_path)
                total_functions += file_stats["total_functions"]
                algorithmic_issues += file_stats["algorithmic_issues"]
                io_bottlenecks += file_stats["io_bottlenecks"]
                memory_issues += file_stats["memory_issues"]
                cpu_issues += file_stats["cpu_issues"]
                db_issues += file_stats["db_issues"]
                loop_issues += file_stats["loop_issues"]
                caching_opportunities += file_stats["caching_opportunities"]

            except Exception as e:
                self.logger.error(f"Error auditing file {file_path}: {e}")
                self.issues.append(PerformanceIssue(
                    file_path=str(file_path),
                    line_number=0,
                    function_name="FILE_ERROR",
                    class_name="",
                    issue_type="AUDIT_ERROR",
                    severity="HIGH",
                    description=f"Failed to audit file: {e}",
                    suggestion="Fix syntax errors or file encoding issues",
                    code_snippet="",
                    estimated_impact="UNKNOWN",
                    optimization_category="ERROR"
                ))

        # Update stats
        self.stats.total_functions = total_functions
        self.stats.algorithmic_complexity_issues = algorithmic_issues
        self.stats.io_bottlenecks = io_bottlenecks
        self.stats.memory_inefficiencies = memory_issues
        self.stats.cpu_intensive_operations = cpu_issues
        self.stats.database_query_issues = db_issues
        self.stats.loop_optimization_opportunities = loop_issues
        self.stats.caching_opportunities = caching_opportunities
        self.stats.overall_performance_score = self._calculate_performance_score()

        self.logger.info(f"Performance audit completed. Found {len(self.issues)} issues.")

        return {
            "stats": asdict(self.stats),
            "issues": [asdict(issue) for issue in self.issues],
            "summary": self._generate_summary()
        }

    def _audit_file(self, file_path: Path) -> dict[str, Any]:
        """Audit performance in a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {
                    "total_functions": 0,
                    "algorithmic_issues": 0,
                    "io_bottlenecks": 0,
                    "memory_issues": 0,
                    "cpu_issues": 0,
                    "db_issues": 0,
                    "loop_issues": 0,
                    "caching_opportunities": 0
                }

            lines = content.split("\n")

            # Analyze functions
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]

            algorithmic_issues = 0
            io_bottlenecks = 0
            memory_issues = 0
            cpu_issues = 0
            db_issues = 0
            loop_issues = 0
            caching_opportunities = 0

            # Analyze each function
            for func in functions:
                func_stats = self._analyze_function_performance(func, lines, str(file_path))
                algorithmic_issues += func_stats["algorithmic_issues"]
                io_bottlenecks += func_stats["io_bottlenecks"]
                memory_issues += func_stats["memory_issues"]
                cpu_issues += func_stats["cpu_issues"]
                db_issues += func_stats["db_issues"]
                loop_issues += func_stats["loop_issues"]
                caching_opportunities += func_stats["caching_opportunities"]

            # Check file-level performance patterns
            file_issues = self._check_file_performance_patterns(content, str(file_path))
            algorithmic_issues += file_issues["algorithmic_issues"]
            io_bottlenecks += file_issues["io_bottlenecks"]
            memory_issues += file_issues["memory_issues"]
            cpu_issues += file_issues["cpu_issues"]
            db_issues += file_issues["db_issues"]

            return {
                "total_functions": len(functions),
                "algorithmic_issues": algorithmic_issues,
                "io_bottlenecks": io_bottlenecks,
                "memory_issues": memory_issues,
                "cpu_issues": cpu_issues,
                "db_issues": db_issues,
                "loop_issues": loop_issues,
                "caching_opportunities": caching_opportunities
            }

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return {
                "total_functions": 0,
                "algorithmic_issues": 0,
                "io_bottlenecks": 0,
                "memory_issues": 0,
                "cpu_issues": 0,
                "db_issues": 0,
                "loop_issues": 0,
                "caching_opportunities": 0
            }

    def _analyze_function_performance(self, func_node: ast.FunctionDef, lines: list[str], file_path: str) -> dict[str, Any]:
        """Analyze a single function for performance issues."""
        func_content = self._get_function_content(func_node, lines)

        algorithmic_issues = 0
        io_bottlenecks = 0
        memory_issues = 0
        cpu_issues = 0
        db_issues = 0
        loop_issues = 0
        caching_opportunities = 0

        # Check for nested loops (O(n²) complexity)
        nested_loops = self._find_nested_loops(func_node)
        if nested_loops > 1:
            algorithmic_issues += 1
            self.issues.append(PerformanceIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="NESTED_LOOPS",
                severity="HIGH",
                description=f"Function '{func_node.name}' has {nested_loops} levels of nested loops",
                suggestion="Consider optimizing algorithm complexity or using vectorized operations",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_impact="HIGH",
                optimization_category="ALGORITHMIC_COMPLEXITY"
            ))

        # Check for inefficient string operations
        string_concat = self._find_string_concatenation(func_node)
        if string_concat > 0:
            cpu_issues += 1
            self.issues.append(PerformanceIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="STRING_CONCATENATION",
                severity="MEDIUM",
                description=f"Function '{func_node.name}' uses inefficient string concatenation",
                suggestion="Use join() method or f-strings for better performance",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_impact="MEDIUM",
                optimization_category="CPU_INTENSIVE"
            ))

        # Check for I/O operations in loops
        io_in_loops = self._find_io_in_loops(func_node)
        if io_in_loops > 0:
            io_bottlenecks += 1
            self.issues.append(PerformanceIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="IO_IN_LOOP",
                severity="HIGH",
                description=f"Function '{func_node.name}' performs I/O operations inside loops",
                suggestion="Move I/O operations outside loops or use batch operations",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_impact="HIGH",
                optimization_category="IO_BOTTLENECK"
            ))

        # Check for database queries in loops
        db_in_loops = self._find_db_queries_in_loops(func_node)
        if db_in_loops > 0:
            db_issues += 1
            self.issues.append(PerformanceIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="DB_QUERY_IN_LOOP",
                severity="HIGH",
                description=f"Function '{func_node.name}' executes database queries inside loops",
                suggestion="Use bulk operations or optimize query structure",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_impact="HIGH",
                optimization_category="DATABASE_BOTTLENECK"
            ))

        # Check for inefficient list operations
        inefficient_lists = self._find_inefficient_list_operations(func_node)
        if inefficient_lists > 0:
            memory_issues += 1
            loop_issues += 1
            self.issues.append(PerformanceIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="INEFFICIENT_LIST_OPS",
                severity="MEDIUM",
                description=f"Function '{func_node.name}' uses inefficient list operations",
                suggestion="Use list comprehensions or generator expressions",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_impact="MEDIUM",
                optimization_category="MEMORY_INEFFICIENCY"
            ))

        # Check for caching opportunities
        expensive_ops = self._find_expensive_operations(func_node)
        if expensive_ops > 0:
            caching_opportunities += 1
            self.issues.append(PerformanceIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="CACHING_OPPORTUNITY",
                severity="MEDIUM",
                description=f"Function '{func_node.name}' has expensive operations that could be cached",
                suggestion="Implement caching using @lru_cache or similar mechanisms",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_impact="MEDIUM",
                optimization_category="CACHING"
            ))

        return {
            "algorithmic_issues": algorithmic_issues,
            "io_bottlenecks": io_bottlenecks,
            "memory_issues": memory_issues,
            "cpu_issues": cpu_issues,
            "db_issues": db_issues,
            "loop_issues": loop_issues,
            "caching_opportunities": caching_opportunities
        }
    def _find_nested_loops(self, node: ast.AST) -> int:
        """Find nested loops in a function."""
        max_depth = 0

        def count_loop_depth(node, current_depth=0):
            nonlocal max_depth
            max_depth = max(max_depth, current_depth)

            for child in ast.iter_child_nodes(node):
                if isinstance(child, (ast.For, ast.While)):
                    count_loop_depth(child, current_depth + 1)
                else:
                    count_loop_depth(child, current_depth)

        count_loop_depth(node)
        return max_depth

    def _find_string_concatenation(self, node: ast.AST) -> int:
        """Find inefficient string concatenation patterns."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.AugAssign) and isinstance(child.op, ast.Add):
                if isinstance(child.target, ast.Name):
                    count += 1
        return count

    def _find_io_in_loops(self, node: ast.AST) -> int:
        """Find I/O operations inside loops."""
        count = 0

        def check_io_in_loop(node, in_loop=False):
            nonlocal count

            if isinstance(node, (ast.For, ast.While)):
                for child in ast.iter_child_nodes(node):
                    check_io_in_loop(child, True)
            elif in_loop and isinstance(node, ast.Call):
                if (isinstance(node.func, ast.Name) and node.func.id in ["open", "read", "write"]) or (isinstance(node.func, ast.Attribute) and node.func.attr in ["read", "write", "open"]):
                    count += 1
            else:
                for child in ast.iter_child_nodes(node):
                    check_io_in_loop(child, in_loop)

        check_io_in_loop(node)
        return count

    def _find_db_queries_in_loops(self, node: ast.AST) -> int:
        """Find database queries inside loops."""
        count = 0

        def check_db_in_loop(node, in_loop=False):
            nonlocal count

            if isinstance(node, (ast.For, ast.While)):
                for child in ast.iter_child_nodes(node):
                    check_db_in_loop(child, True)
            elif in_loop and isinstance(node, ast.Call):
                if isinstance(node.func, ast.Attribute):
                    if node.func.attr in ["query", "execute", "fetch", "fetchall", "fetchone"]:
                        count += 1
            else:
                for child in ast.iter_child_nodes(node):
                    check_db_in_loop(child, in_loop)

        check_db_in_loop(node)
        return count

    def _find_inefficient_list_operations(self, node: ast.AST) -> int:
        """Find inefficient list operations."""
        count = 0

        # Look for list.append() in loops that could be list comprehensions
        for child in ast.walk(node):
            if isinstance(child, ast.For):
                for stmt in child.body:
                    if isinstance(stmt, ast.Expr) and isinstance(stmt.value, ast.Call):
                        if isinstance(stmt.value.func, ast.Attribute) and stmt.value.func.attr == "append":
                            count += 1

        return count

    def _find_expensive_operations(self, node: ast.AST) -> int:
        """Find expensive operations that could benefit from caching."""
        count = 0
        expensive_functions = ["sorted", "max", "min", "sum", "len"]

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                if isinstance(child.func, ast.Name) and child.func.id in expensive_functions:
                    # Check if it's called multiple times (simple heuristic)
                    count += 1
                elif isinstance(child.func, ast.Attribute):
                    if child.func.attr in ["sort", "reverse", "split", "join"]:
                        count += 1

        return count if count > 2 else 0  # Only flag if multiple expensive ops

    def _get_function_content(self, func_node: ast.FunctionDef, lines: list[str]) -> str:
        """Extract function content from source lines."""
        start_line = func_node.lineno - 1
        end_line = func_node.end_lineno if hasattr(func_node, "end_lineno") else start_line + 20

        end_line = min(end_line, len(lines))

        return "\n".join(lines[start_line:end_line])

    def _check_file_performance_patterns(self, content: str, file_path: str) -> dict[str, int]:
        """Check for file-level performance patterns."""
        issues = {
            "algorithmic_issues": 0,
            "io_bottlenecks": 0,
            "memory_issues": 0,
            "cpu_issues": 0,
            "db_issues": 0
        }

        lines = content.split("\n")

        # Check for global performance anti-patterns
        for i, line in enumerate(lines):
            # Check for imports that might indicate performance issues
            if "import pandas" in line and "import numpy" not in content:
                self.issues.append(PerformanceIssue(
                    file_path=file_path,
                    line_number=i + 1,
                    function_name="",
                    class_name="",
                    issue_type="MISSING_NUMPY",
                    severity="MEDIUM",
                    description="Using pandas without numpy may impact performance",
                    suggestion="Consider importing numpy for vectorized operations",
                    code_snippet=line.strip()[:200],
                    estimated_impact="MEDIUM",
                    optimization_category="CPU_INTENSIVE"
                ))
                issues["cpu_issues"] += 1

        return issues

    def _calculate_performance_score(self) -> float:
        """Calculate overall performance score (0-100)."""
        if self.stats.total_functions == 0:
            return 100.0

        # Base score
        score = 100.0

        # Penalize algorithmic complexity issues heavily
        if self.stats.algorithmic_complexity_issues > 0:
            score -= min(40, self.stats.algorithmic_complexity_issues * 10)

        # Penalize I/O bottlenecks
        if self.stats.io_bottlenecks > 0:
            score -= min(30, self.stats.io_bottlenecks * 8)

        # Penalize database issues
        if self.stats.database_query_issues > 0:
            score -= min(25, self.stats.database_query_issues * 7)

        # Penalize memory inefficiencies
        if self.stats.memory_inefficiencies > 0:
            score -= min(20, self.stats.memory_inefficiencies * 5)

        # Penalize CPU intensive operations
        if self.stats.cpu_intensive_operations > 0:
            score -= min(15, self.stats.cpu_intensive_operations * 3)

        return max(0.0, score)

    def _generate_summary(self) -> dict[str, Any]:
        """Generate audit summary."""
        severity_counts = {}
        category_counts = {}

        for issue in self.issues:
            severity_counts[issue.severity] = severity_counts.get(issue.severity, 0) + 1
            category_counts[issue.optimization_category] = category_counts.get(issue.optimization_category, 0) + 1

        return {
            "total_issues": len(self.issues),
            "severity_breakdown": severity_counts,
            "category_breakdown": category_counts,
            "performance_score": f"{self.stats.overall_performance_score:.1f}/100",
            "algorithmic_issues": self.stats.algorithmic_complexity_issues,
            "io_bottlenecks": self.stats.io_bottlenecks,
            "memory_inefficiencies": self.stats.memory_inefficiencies,
            "caching_opportunities": self.stats.caching_opportunities
        }

    def generate_report(self, output_file: str | None = None) -> str:
        """Generate comprehensive performance audit report."""
        if not output_file:
            output_file = f"performance_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

        report = {
            "audit_timestamp": datetime.now().isoformat(),
            "auditor_version": "1.0.0",
            "stats": asdict(self.stats),
            "summary": self._generate_summary(),
            "issues": [asdict(issue) for issue in self.issues],
            "recommendations": self._generate_recommendations(),
            "optimization_priorities": self._generate_optimization_priorities()
        }

        with open(output_file, "w") as f:
            json.dump(report, f, indent=2)

        self.logger.info(f"Performance audit report saved to {output_file}")
        return output_file

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations for performance improvement."""
        recommendations = []

        if self.stats.overall_performance_score < 50:
            recommendations.append("CRITICAL: Performance score is below 50. Immediate optimization required.")
        elif self.stats.overall_performance_score < 70:
            recommendations.append("WARNING: Performance score is below 70. Consider optimization.")

        if self.stats.algorithmic_complexity_issues > 0:
            recommendations.append(f"HIGH PRIORITY: {self.stats.algorithmic_complexity_issues} algorithmic complexity issues found.")

        if self.stats.io_bottlenecks > 0:
            recommendations.append(f"Optimize {self.stats.io_bottlenecks} I/O bottlenecks for better performance.")

        if self.stats.database_query_issues > 0:
            recommendations.append(f"Optimize {self.stats.database_query_issues} database query patterns.")

        if self.stats.caching_opportunities > 0:
            recommendations.append(f"Implement caching for {self.stats.caching_opportunities} expensive operations.")

        recommendations.extend([
            "Profile critical code paths with cProfile for detailed analysis",
            "Consider using async/await for I/O bound operations",
            "Implement connection pooling for database operations",
            "Use vectorized operations with numpy/pandas where applicable",
            "Consider implementing lazy loading for large datasets"
        ])

        return recommendations

    def _generate_optimization_priorities(self) -> list[dict[str, Any]]:
        """Generate prioritized optimization suggestions."""
        priorities = []

        # Group issues by category and severity
        high_severity_issues = [issue for issue in self.issues if issue.severity == "HIGH"]
        algorithmic_issues = [issue for issue in self.issues if issue.optimization_category == "ALGORITHMIC_COMPLEXITY"]
        io_issues = [issue for issue in self.issues if issue.optimization_category == "IO_BOTTLENECK"]

        if high_severity_issues:
            priorities.append({
                "priority": 1,
                "category": "High Severity Performance Issues",
                "count": len(high_severity_issues),
                "description": "Address critical performance bottlenecks first",
                "estimated_effort": "High",
                "expected_impact": "High"
            })

        if algorithmic_issues:
            priorities.append({
                "priority": 2,
                "category": "Algorithmic Complexity",
                "count": len(algorithmic_issues),
                "description": "Optimize algorithm complexity for scalability",
                "estimated_effort": "High",
                "expected_impact": "Very High"
            })

        if io_issues:
            priorities.append({
                "priority": 3,
                "category": "I/O Bottlenecks",
                "count": len(io_issues),
                "description": "Optimize I/O operations and implement async patterns",
                "estimated_effort": "Medium",
                "expected_impact": "High"
            })

        return priorities

def main():
    """Main function for running the performance auditor."""
    auditor = PerformanceAuditor()

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
