#!/usr/bin/env python3
"""
Test Infrastructure Fix Script
Systematically fixes test collection errors and import issues identified in Task 82.

This script addresses the critical test infrastructure problems preventing proper
test coverage measurement, as identified in the test coverage validation report.
"""

import re
import subprocess
import sys
from pathlib import Path

MAX_COLLECTION_ERRORS = 10


class TestInfrastructureFixer:
    """Fixes test infrastructure issues to enable proper coverage measurement."""

    def __init__(self, project_root: str = "/home/vivi/pixelated/ai"):
        self.project_root = Path(project_root)
        self.test_files = []
        self.errors_fixed = 0
        self.import_fixes = 0

    def scan_test_files(self) -> list[Path]:
        """Scan for all test files in the project."""
        test_files = []
        for pattern in ["test_*.py", "*_test.py"]:
            test_files.extend(self.project_root.rglob(pattern))

        self.test_files = test_files
        return test_files

    def analyze_import_errors(self) -> dict[str, list[str]]:
        """Analyze import errors in test files."""

        # Run pytest collect-only to capture errors
        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            shell=False,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        errors = {}
        current_file = None

        for line in result.stderr.split("\n"):
            if "ERROR collecting" in line:
                # Extract file path
                match = re.search(r"ERROR collecting (.+\.py)", line)
                if match:
                    current_file = match.group(1)
                    errors[current_file] = []
            elif current_file and ("ModuleNotFoundError" in line or "ImportError" in line):
                errors[current_file].append(line.strip())

        return errors

    def fix_import_paths(self, test_file: Path) -> bool:
        """Fix import paths in a test file."""
        try:
            with open(test_file) as f:
                content = f.read()

            original_content = content

            # Common import fixes
            fixes = [
                # Fix relative imports for modules in same directory
                (r"from (\w+) import", r"from .\1 import"),
                # Fix missing type imports
                (
                    r"from typing import",
                    r"from typing import Tuple, Callable, Dict, List, Optional, Union, Any",
                ),
                # Fix dataset_pipeline imports
                (r"from dataset_pipeline\.", r"from ai.pipelines.orchestrator."),
                # Fix pixel imports
                (r"from pixel\.", r"from ai.models.pixel_core."),
                # Fix inference imports
                (r"from inference\.", r"from ai.inference."),
            ]

            for pattern, replacement in fixes:
                content = re.sub(pattern, replacement, content)

            # Add missing imports at the top
            if "import pytest" not in content:
                content = "import pytest\n" + content

            if "from unittest.mock import" not in content and "mock" in content.lower():
                content = "from unittest.mock import Mock, patch, MagicMock\n" + content

            # Write back if changed
            if content != original_content:
                with open(test_file, "w") as f:
                    f.write(content)
                self.import_fixes += 1
                return True

        except Exception:
            return False

        return False

    def create_missing_modules(self) -> None:
        """Create missing __init__.py files and stub modules."""

        # Ensure __init__.py files exist
        directories = [
            self.project_root / "dataset_pipeline",
            self.project_root / "pixel",
            self.project_root / "inference",
            self.project_root / "tests",
            self.project_root / "tests" / "dataset_pipeline",
            self.project_root / "tests" / "pixel",
            self.project_root / "tests" / "inference",
        ]

        for directory in directories:
            if directory.exists():
                init_file = directory / "__init__.py"
                if not init_file.exists():
                    init_file.write_text("# Auto-generated __init__.py\n")

    def organize_test_files(self) -> None:
        """Organize test files into proper test directory structure."""

        tests_dir = self.project_root / "tests"
        tests_dir.mkdir(exist_ok=True)

        # Create test subdirectories
        for subdir in ["dataset_pipeline", "pixel", "inference", "tools", "qa"]:
            (tests_dir / subdir).mkdir(exist_ok=True)
            (tests_dir / subdir / "__init__.py").touch()

    def fix_common_test_issues(self, test_file: Path) -> bool:
        """Fix common issues in test files."""
        try:
            with open(test_file) as f:
                content = f.read()

            original_content = content

            # Fix common test issues

            # Add proper test structure if missing
            if "class Test" not in content and "def test_" in content:
                # Wrap standalone test functions in a test class
                lines = content.split("\n")
                new_lines = []
                in_test_function = False

                for line in lines:
                    if line.startswith("def test_"):
                        if not in_test_function:
                            new_lines.append("\nclass TestModule(unittest.TestCase):")
                            in_test_function = True
                        new_lines.append("    " + line)
                    elif in_test_function and line.strip() and not line.startswith(" "):
                        in_test_function = False
                        new_lines.append(line)
                    elif in_test_function:
                        new_lines.append("    " + line)
                    else:
                        new_lines.append(line)

                content = "\n".join(new_lines)

            # Write back if changed
            if content != original_content:
                with open(test_file, "w") as f:
                    f.write(content)
                return True

        except Exception:
            return False

        return False

    def validate_fixes(self) -> dict[str, int]:
        """Validate that fixes worked by running pytest collect."""

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--collect-only", "-q"],
            shell=False,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        # Count collected tests and errors
        collected = len([line for line in result.stdout.split("\n") if "::test_" in line])
        errors = len([line for line in result.stderr.split("\n") if "ERROR collecting" in line])

        return {
            "collected_tests": collected,
            "collection_errors": errors,
            "exit_code": result.returncode,
        }

    def run_coverage_test(self) -> dict[str, float]:
        """Run a quick coverage test to see improvement."""

        result = subprocess.run(
            [sys.executable, "-m", "pytest", "--cov=.", "--cov-report=term", "-x"],
            shell=False,
            cwd=self.project_root,
            capture_output=True,
            text=True,
            check=False,
        )

        # Extract coverage percentage
        coverage_match = re.search(r"TOTAL\s+\d+\s+\d+\s+(\d+)%", result.stdout)
        coverage_percent = float(coverage_match.group(1)) if coverage_match else 0.0

        return {
            "coverage_percent": coverage_percent,
            "tests_run": len([line for line in result.stdout.split("\n") if "PASSED" in line]),
            "tests_failed": len([line for line in result.stdout.split("\n") if "FAILED" in line]),
        }

    def generate_report(self, validation_results: dict, coverage_results: dict) -> str:
        """Generate a fix report."""
        status = (
            "✅ Infrastructure fixes successful"
            if validation_results["collection_errors"] < MAX_COLLECTION_ERRORS
            else "❌ More fixes needed"
        )
        return f"""
# Test Infrastructure Fix Report

## Summary
- Test files scanned: {len(self.test_files)}
- Import fixes applied: {self.import_fixes}
- Errors fixed: {self.errors_fixed}

## Validation Results
- Tests collected: {validation_results["collected_tests"]}
- Collection errors: {validation_results["collection_errors"]}
- Exit code: {validation_results["exit_code"]}

## Coverage Results
- Coverage percentage: {coverage_results["coverage_percent"]}%
- Tests run: {coverage_results["tests_run"]}
- Tests failed: {coverage_results["tests_failed"]}

## Status
{status}
"""

    def run_full_fix(self) -> None:
        """Run the complete fix process."""

        # Step 1: Scan test files
        self.scan_test_files()

        # Step 2: Analyze current errors
        self.analyze_import_errors()

        # Step 3: Create missing infrastructure
        self.create_missing_modules()
        self.organize_test_files()

        # Step 4: Fix import issues in test files
        for test_file in self.test_files:
            if self.fix_import_paths(test_file):
                pass

            if self.fix_common_test_issues(test_file):
                pass

        # Step 5: Validate fixes
        validation_results = self.validate_fixes()
        coverage_results = self.run_coverage_test()

        # Step 6: Generate report
        report = self.generate_report(validation_results, coverage_results)

        # Save report
        report_file = self.project_root / "qa" / "reports" / "test_infrastructure_fix_report.md"
        report_file.write_text(report)


def main():
    """Main entry point."""
    fixer = TestInfrastructureFixer()
    fixer.run_full_fix()


if __name__ == "__main__":
    main()
