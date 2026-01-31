#!/usr/bin/env python3
"""
Comprehensive Memory Usage Auditor for Pixelated Empathy AI
Analyzes memory usage patterns, leaks, and optimization opportunities.
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
class MemoryIssue:
    """Represents a memory usage issue found during analysis."""
    file_path: str
    line_number: int
    function_name: str
    class_name: str
    issue_type: str
    severity: str
    description: str
    suggestion: str
    code_snippet: str
    estimated_memory_impact: str
    optimization_category: str

@dataclass
class MemoryStats:
    """Statistics about memory usage patterns."""
    total_files: int
    total_functions: int
    memory_leak_risks: int
    inefficient_data_structures: int
    large_object_allocations: int
    circular_reference_risks: int
    generator_opportunities: int
    memory_optimization_score: float

class MemoryAuditor:
    """Comprehensive memory usage auditor for optimization."""

    def __init__(self, root_path: str = "."):
        self.root_path = Path(root_path)
        self.issues: list[MemoryIssue] = []
        self.stats = MemoryStats(0, 0, 0, 0, 0, 0, 0, 0.0)

        # Memory anti-patterns
        self.memory_patterns = {
            "global_variables": {
                "pattern": r"^[A-Z_][A-Z0-9_]*\s*=",
                "severity": "MEDIUM",
                "category": "MEMORY_LEAK_RISK"
            },
            "large_list_creation": {
                "pattern": r"\[\s*.*\s*for\s+.*\s+in\s+range\(\d{4,}\)",
                "severity": "HIGH",
                "category": "LARGE_ALLOCATION"
            },
            "string_multiplication": {
                "pattern": r'["\'].*["\']\s*\*\s*\d{3,}',
                "severity": "MEDIUM",
                "category": "LARGE_ALLOCATION"
            },
            "recursive_data_structures": {
                "pattern": r"self\.\w+\s*=.*self",
                "severity": "HIGH",
                "category": "CIRCULAR_REFERENCE"
            }
        }

        # Memory-inefficient patterns
        self.inefficient_patterns = {
            "list_instead_of_generator": "Use generators instead of lists for large datasets",
            "dict_instead_of_slots": "Use __slots__ in classes to reduce memory overhead",
            "string_concatenation": "Use join() instead of += for string concatenation",
            "unnecessary_copies": "Avoid unnecessary data copying",
            "large_constants": "Move large constants to separate modules or files"
        }

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def audit_directory(self, directory: str | None = None) -> dict[str, Any]:
        """Audit memory usage throughout the specified directory."""
        if directory is None:
            directory = self.root_path

        self.logger.info(f"Starting memory audit of directory: {directory}")

        # Reset stats
        self.issues = []
        self.stats = MemoryStats(0, 0, 0, 0, 0, 0, 0, 0.0)

        # Find all Python files
        python_files = list(Path(directory).rglob("*.py"))
        self.stats.total_files = len(python_files)

        total_functions = 0
        memory_leaks = 0
        inefficient_structures = 0
        large_allocations = 0
        circular_refs = 0
        generator_ops = 0

        for file_path in python_files:
            try:
                file_stats = self._audit_file(file_path)
                total_functions += file_stats["total_functions"]
                memory_leaks += file_stats["memory_leaks"]
                inefficient_structures += file_stats["inefficient_structures"]
                large_allocations += file_stats["large_allocations"]
                circular_refs += file_stats["circular_refs"]
                generator_ops += file_stats["generator_opportunities"]

            except Exception as e:
                self.logger.error(f"Error auditing file {file_path}: {e}")
                self.issues.append(MemoryIssue(
                    file_path=str(file_path),
                    line_number=0,
                    function_name="FILE_ERROR",
                    class_name="",
                    issue_type="AUDIT_ERROR",
                    severity="HIGH",
                    description=f"Failed to audit file: {e}",
                    suggestion="Fix syntax errors or file encoding issues",
                    code_snippet="",
                    estimated_memory_impact="UNKNOWN",
                    optimization_category="ERROR"
                ))

        # Update stats
        self.stats.total_functions = total_functions
        self.stats.memory_leak_risks = memory_leaks
        self.stats.inefficient_data_structures = inefficient_structures
        self.stats.large_object_allocations = large_allocations
        self.stats.circular_reference_risks = circular_refs
        self.stats.generator_opportunities = generator_ops
        self.stats.memory_optimization_score = self._calculate_memory_score()

        self.logger.info(f"Memory audit completed. Found {len(self.issues)} issues.")

        return {
            "stats": asdict(self.stats),
            "issues": [asdict(issue) for issue in self.issues],
            "summary": self._generate_summary()
        }

    def _audit_file(self, file_path: Path) -> dict[str, Any]:
        """Audit memory usage in a single file."""
        try:
            with open(file_path, encoding="utf-8") as f:
                content = f.read()

            # Parse AST
            try:
                tree = ast.parse(content)
            except SyntaxError:
                return {
                    "total_functions": 0,
                    "memory_leaks": 0,
                    "inefficient_structures": 0,
                    "large_allocations": 0,
                    "circular_refs": 0,
                    "generator_opportunities": 0
                }

            lines = content.split("\n")

            # Analyze functions and classes
            functions = [node for node in ast.walk(tree) if isinstance(node, ast.FunctionDef)]
            classes = [node for node in ast.walk(tree) if isinstance(node, ast.ClassDef)]

            memory_leaks = 0
            inefficient_structures = 0
            large_allocations = 0
            circular_refs = 0
            generator_ops = 0

            # Analyze each function
            for func in functions:
                func_stats = self._analyze_function_memory(func, lines, str(file_path))
                memory_leaks += func_stats["memory_leaks"]
                inefficient_structures += func_stats["inefficient_structures"]
                large_allocations += func_stats["large_allocations"]
                circular_refs += func_stats["circular_refs"]
                generator_ops += func_stats["generator_opportunities"]

            # Analyze each class
            for cls in classes:
                class_stats = self._analyze_class_memory(cls, lines, str(file_path))
                memory_leaks += class_stats["memory_leaks"]
                inefficient_structures += class_stats["inefficient_structures"]
                circular_refs += class_stats["circular_refs"]

            # Check file-level memory patterns
            file_issues = self._check_file_memory_patterns(content, str(file_path))
            memory_leaks += file_issues["memory_leaks"]
            large_allocations += file_issues["large_allocations"]

            return {
                "total_functions": len(functions),
                "memory_leaks": memory_leaks,
                "inefficient_structures": inefficient_structures,
                "large_allocations": large_allocations,
                "circular_refs": circular_refs,
                "generator_opportunities": generator_ops
            }

        except Exception as e:
            self.logger.error(f"Error processing file {file_path}: {e}")
            return {
                "total_functions": 0,
                "memory_leaks": 0,
                "inefficient_structures": 0,
                "large_allocations": 0,
                "circular_refs": 0,
                "generator_opportunities": 0
            }

    def _analyze_function_memory(self, func_node: ast.FunctionDef, lines: list[str], file_path: str) -> dict[str, Any]:
        """Analyze a single function for memory issues."""
        func_content = self._get_function_content(func_node, lines)

        memory_leaks = 0
        inefficient_structures = 0
        large_allocations = 0
        circular_refs = 0
        generator_ops = 0

        # Check for large list comprehensions that could be generators
        large_comprehensions = self._find_large_comprehensions(func_node)
        if large_comprehensions > 0:
            generator_ops += 1
            self.issues.append(MemoryIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="GENERATOR_OPPORTUNITY",
                severity="MEDIUM",
                description=f"Function '{func_node.name}' has large list comprehensions",
                suggestion="Consider using generator expressions for memory efficiency",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_memory_impact="MEDIUM",
                optimization_category="GENERATOR_OPPORTUNITY"
            ))

        # Check for inefficient data structure usage
        inefficient_ds = self._find_inefficient_data_structures(func_node)
        if inefficient_ds > 0:
            inefficient_structures += 1
            self.issues.append(MemoryIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="INEFFICIENT_DATA_STRUCTURE",
                severity="MEDIUM",
                description=f"Function '{func_node.name}' uses inefficient data structures",
                suggestion="Consider using more memory-efficient data structures",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_memory_impact="MEDIUM",
                optimization_category="DATA_STRUCTURE"
            ))

        # Check for potential memory leaks
        leak_risks = self._find_memory_leak_risks(func_node)
        if leak_risks > 0:
            memory_leaks += 1
            self.issues.append(MemoryIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="MEMORY_LEAK_RISK",
                severity="HIGH",
                description=f"Function '{func_node.name}' has potential memory leak risks",
                suggestion="Ensure proper cleanup of resources and avoid circular references",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_memory_impact="HIGH",
                optimization_category="MEMORY_LEAK"
            ))

        # Check for large object allocations
        large_allocs = self._find_large_allocations(func_node)
        if large_allocs > 0:
            large_allocations += 1
            self.issues.append(MemoryIssue(
                file_path=file_path,
                line_number=func_node.lineno,
                function_name=func_node.name,
                class_name="",
                issue_type="LARGE_ALLOCATION",
                severity="HIGH",
                description=f"Function '{func_node.name}' performs large memory allocations",
                suggestion="Consider streaming or chunking large data operations",
                code_snippet=func_content[:200] + "..." if len(func_content) > 200 else func_content,
                estimated_memory_impact="HIGH",
                optimization_category="LARGE_ALLOCATION"
            ))

        return {
            "memory_leaks": memory_leaks,
            "inefficient_structures": inefficient_structures,
            "large_allocations": large_allocations,
            "circular_refs": circular_refs,
            "generator_opportunities": generator_ops
        }

    def _analyze_class_memory(self, class_node: ast.ClassDef, lines: list[str], file_path: str) -> dict[str, Any]:
        """Analyze a single class for memory issues."""
        class_content = self._get_class_content(class_node, lines)

        memory_leaks = 0
        inefficient_structures = 0
        circular_refs = 0

        # Check if class uses __slots__
        has_slots = any(
            isinstance(node, ast.Assign) and
            any(isinstance(target, ast.Name) and target.id == "__slots__" for target in node.targets)
            for node in class_node.body
        )

        # Count instance variables
        instance_vars = self._count_instance_variables(class_node)

        if not has_slots and instance_vars > 5:
            inefficient_structures += 1
            self.issues.append(MemoryIssue(
                file_path=file_path,
                line_number=class_node.lineno,
                function_name="",
                class_name=class_node.name,
                issue_type="MISSING_SLOTS",
                severity="MEDIUM",
                description=f"Class '{class_node.name}' could benefit from __slots__",
                suggestion="Add __slots__ to reduce memory overhead for instances",
                code_snippet=class_content[:200] + "..." if len(class_content) > 200 else class_content,
                estimated_memory_impact="MEDIUM",
                optimization_category="DATA_STRUCTURE"
            ))

        # Check for circular reference risks
        circular_risk = self._find_circular_reference_risks(class_node)
        if circular_risk > 0:
            circular_refs += 1
            self.issues.append(MemoryIssue(
                file_path=file_path,
                line_number=class_node.lineno,
                function_name="",
                class_name=class_node.name,
                issue_type="CIRCULAR_REFERENCE_RISK",
                severity="HIGH",
                description=f"Class '{class_node.name}' has circular reference risks",
                suggestion="Use weak references or careful cleanup to avoid memory leaks",
                code_snippet=class_content[:200] + "..." if len(class_content) > 200 else class_content,
                estimated_memory_impact="HIGH",
                optimization_category="CIRCULAR_REFERENCE"
            ))

        return {
            "memory_leaks": memory_leaks,
            "inefficient_structures": inefficient_structures,
            "circular_refs": circular_refs
        }

    def _find_large_comprehensions(self, node: ast.AST) -> int:
        """Find large list comprehensions that could be generators."""
        count = 0
        for child in ast.walk(node):
            if isinstance(child, ast.ListComp):
                # Simple heuristic: nested comprehensions or range with large numbers
                if any(isinstance(n, ast.ListComp) for n in ast.walk(child)) or any(isinstance(n, ast.Call) and
                        isinstance(n.func, ast.Name) and n.func.id == "range" and
                        len(n.args) > 0 and isinstance(n.args[0], ast.Constant) and
                        isinstance(n.args[0].value, int) and n.args[0].value > 1000
                        for n in ast.walk(child)):
                    count += 1
        return count

    def _find_inefficient_data_structures(self, node: ast.AST) -> int:
        """Find inefficient data structure usage."""
        count = 0

        # Look for repeated list.append() that could be list comprehension
        for child in ast.walk(node):
            if isinstance(child, ast.For):
                append_calls = 0
                for stmt in ast.walk(child):
                    if isinstance(stmt, ast.Call) and isinstance(stmt.func, ast.Attribute):
                        if stmt.func.attr == "append":
                            append_calls += 1
                if append_calls > 0:
                    count += 1

        return count

    def _find_memory_leak_risks(self, node: ast.AST) -> int:
        """Find potential memory leak risks."""
        count = 0

        # Look for global variable assignments
        for child in ast.walk(node):
            if isinstance(child, ast.Global):
                count += 1
            elif isinstance(child, ast.Assign):
                # Check for assignments to module-level variables
                for target in child.targets:
                    if isinstance(target, ast.Name) and target.id.isupper():
                        count += 1

        return count

    def _find_large_allocations(self, node: ast.AST) -> int:
        """Find large memory allocations."""
        count = 0

        for child in ast.walk(node):
            # Large list creation
            if isinstance(child, ast.List) and len(child.elts) > 1000:
                count += 1
            # Large string multiplication
            elif isinstance(child, ast.BinOp) and isinstance(child.op, ast.Mult):
                if isinstance(child.right, ast.Constant) and isinstance(child.right.value, int):
                    if child.right.value > 1000:
                        count += 1

        return count

    def _count_instance_variables(self, class_node: ast.ClassDef) -> int:
        """Count instance variables in a class."""
        instance_vars = set()

        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self":
                            instance_vars.add(target.attr)

        return len(instance_vars)

    def _find_circular_reference_risks(self, class_node: ast.ClassDef) -> int:
        """Find circular reference risks in a class."""
        count = 0

        for node in ast.walk(class_node):
            if isinstance(node, ast.Assign):
                for target in node.targets:
                    if isinstance(target, ast.Attribute) and isinstance(target.value, ast.Name):
                        if target.value.id == "self":
                            # Check if assigned value references self
                            if any(isinstance(n, ast.Name) and n.id == "self" for n in ast.walk(node.value)):
                                count += 1

        return count

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

    def _check_file_memory_patterns(self, content: str, file_path: str) -> dict[str, int]:
        """Check for file-level memory patterns."""
        issues = {
            "memory_leaks": 0,
            "large_allocations": 0
        }

        lines = content.split("\n")

        # Check for global variables
        for i, line in enumerate(lines):
            if re.match(r"^[A-Z_][A-Z0-9_]*\s*=", line.strip()):
                self.issues.append(MemoryIssue(
                    file_path=file_path,
                    line_number=i + 1,
                    function_name="",
                    class_name="",
                    issue_type="GLOBAL_VARIABLE",
                    severity="MEDIUM",
                    description="Global variable found",
                    suggestion="Consider using class attributes or function parameters instead",
                    code_snippet=line.strip()[:200],
                    estimated_memory_impact="MEDIUM",
                    optimization_category="MEMORY_LEAK"
                ))
                issues["memory_leaks"] += 1

        return issues

    def _calculate_memory_score(self) -> float:
        """Calculate overall memory optimization score (0-100)."""
        if self.stats.total_functions == 0:
            return 100.0

        # Base score
        score = 100.0

        # Penalize memory leak risks heavily
        if self.stats.memory_leak_risks > 0:
            score -= min(40, self.stats.memory_leak_risks * 8)

        # Penalize large allocations
        if self.stats.large_object_allocations > 0:
            score -= min(30, self.stats.large_object_allocations * 6)

        # Penalize circular reference risks
        if self.stats.circular_reference_risks > 0:
            score -= min(25, self.stats.circular_reference_risks * 5)

        # Penalize inefficient data structures
        if self.stats.inefficient_data_structures > 0:
            score -= min(20, self.stats.inefficient_data_structures * 3)

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
            "memory_score": f"{self.stats.memory_optimization_score:.1f}/100",
            "memory_leak_risks": self.stats.memory_leak_risks,
            "large_allocations": self.stats.large_object_allocations,
            "generator_opportunities": self.stats.generator_opportunities
        }

    def generate_report(self, output_file: str | None = None) -> str:
        """Generate comprehensive memory audit report."""
        if not output_file:
            output_file = f"memory_audit_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"

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

        self.logger.info(f"Memory audit report saved to {output_file}")
        return output_file

    def _generate_recommendations(self) -> list[str]:
        """Generate recommendations for memory optimization."""
        recommendations = []

        if self.stats.memory_optimization_score < 50:
            recommendations.append("CRITICAL: Memory optimization score is below 50. Immediate action required.")
        elif self.stats.memory_optimization_score < 70:
            recommendations.append("WARNING: Memory optimization score is below 70. Consider optimization.")

        if self.stats.memory_leak_risks > 0:
            recommendations.append(f"HIGH PRIORITY: Address {self.stats.memory_leak_risks} memory leak risks.")

        if self.stats.large_object_allocations > 0:
            recommendations.append(f"Optimize {self.stats.large_object_allocations} large memory allocations.")

        if self.stats.generator_opportunities > 0:
            recommendations.append(f"Implement generators for {self.stats.generator_opportunities} opportunities.")

        recommendations.extend([
            "Use __slots__ in classes with many instances",
            "Implement object pooling for frequently created objects",
            "Use weak references to break circular dependencies",
            "Profile memory usage with tracemalloc in production",
            "Consider using memory-mapped files for large datasets"
        ])

        return recommendations

def main():
    """Main function for running the memory auditor."""
    auditor = MemoryAuditor()

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
