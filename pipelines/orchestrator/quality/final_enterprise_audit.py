#!/usr/bin/env python3
"""
FINAL ENTERPRISE AUDIT SCRIPT
Comprehensive verification of all 36 tasks claimed in phase6.md
Tests for enterprise-grade quality and production readiness
"""

import importlib.util
import json
import subprocess
import sys
from datetime import datetime
from pathlib import Path
from typing import Any


class FinalEnterpriseAuditor:
    """Final enterprise audit for Task 6.0"""

    def __init__(self):
        self.ai_root = Path("/home/vivi/pixelated/ai")
        self.results = {}
        self.total_tasks = 36
        self.enterprise_criteria = {
            "min_file_size": 5000,  # 5KB minimum for enterprise code
            "min_functions": 5,     # Minimum functions for meaningful implementation
            "min_classes": 1,       # At least one main class
            "requires_docstring": True,
            "requires_error_handling": True,
            "requires_logging": True
        }

    def find_file_in_repo(self, filename: str) -> list[str]:
        """Find file anywhere in the AI repository"""
        try:
            result = subprocess.run(
                ["find", str(self.ai_root), "-name", filename, "-type", "f"],
                check=False, capture_output=True, text=True
            )
            if result.returncode == 0:
                return [path.strip() for path in result.stdout.split("\n") if path.strip()]
            return []
        except Exception:
            return []

    def test_file_functionality(self, file_path: str) -> dict[str, Any]:
        """Test file functionality and enterprise quality"""
        test_result = {
            "syntax_valid": False,
            "imports_successfully": False,
            "has_main_function": False,
            "runs_without_error": False,
            "enterprise_quality": False,
            "quality_issues": []
        }

        try:
            # Read file content
            with open(file_path) as f:
                content = f.read()

            # Basic syntax check
            try:
                compile(content, file_path, "exec")
                test_result["syntax_valid"] = True
            except SyntaxError as e:
                test_result["quality_issues"].append(f"Syntax error: {e}")
                return test_result

            # Check for main function
            if "def main(" in content or 'if __name__ == "__main__"' in content:
                test_result["has_main_function"] = True

            # Try to import the module
            try:
                spec = importlib.util.spec_from_file_location("test_module", file_path)
                if spec and spec.loader:
                    module = importlib.util.module_from_spec(spec)
                    spec.loader.exec_module(module)
                    test_result["imports_successfully"] = True

                    # Try to run main function if it exists
                    if hasattr(module, "main") and callable(module.main):
                        try:
                            # Capture output to avoid cluttering
                            import contextlib
                            import io

                            f = io.StringIO()
                            with contextlib.redirect_stdout(f), contextlib.redirect_stderr(f):
                                module.main()
                            test_result["runs_without_error"] = True
                        except Exception as e:
                            test_result["quality_issues"].append(f"Runtime error in main(): {e}")
                    else:
                        test_result["runs_without_error"] = True  # No main to test

            except Exception as e:
                test_result["quality_issues"].append(f"Import error: {e}")

            # Enterprise quality checks
            test_result["enterprise_quality"] = self._assess_enterprise_quality(content, file_path)

        except Exception as e:
            test_result["quality_issues"].append(f"File access error: {e}")

        return test_result

    def _assess_enterprise_quality(self, content: str, file_path: str) -> bool:
        """Assess if file meets enterprise quality standards"""
        issues = []

        # Size check
        if len(content) < self.enterprise_criteria["min_file_size"]:
            issues.append(f"File too small: {len(content)} bytes < {self.enterprise_criteria['min_file_size']}")

        # Function count
        function_count = content.count("def ")
        if function_count < self.enterprise_criteria["min_functions"]:
            issues.append(f"Too few functions: {function_count} < {self.enterprise_criteria['min_functions']}")

        # Class count
        class_count = content.count("class ")
        if class_count < self.enterprise_criteria["min_classes"]:
            issues.append(f"Too few classes: {class_count} < {self.enterprise_criteria['min_classes']}")

        # Docstring check
        if self.enterprise_criteria["requires_docstring"]:
            if '"""' not in content and "'''" not in content:
                issues.append("No docstrings found")

        # Error handling check
        if self.enterprise_criteria["requires_error_handling"]:
            if "try:" not in content and "except" not in content:
                issues.append("No error handling found")

        # Logging check
        if self.enterprise_criteria["requires_logging"]:
            if "logging" not in content and "logger" not in content:
                issues.append("No logging implementation found")

        return len(issues) == 0

    def audit_task(self, task_id: str, expected_filename: str, description: str) -> dict[str, Any]:
        """Comprehensive audit of a single task"""

        result = {
            "task_id": task_id,
            "expected_filename": expected_filename,
            "description": description,
            "files_found": [],
            "primary_file": None,
            "file_stats": {},
            "functionality_test": {},
            "enterprise_ready": False,
            "status": "MISSING",
            "issues": [],
            "recommendations": []
        }

        # Find files
        found_files = self.find_file_in_repo(expected_filename)

        if not found_files:
            result["status"] = "MISSING"
            result["issues"].append("File not found in repository")
            return result

        result["files_found"] = found_files

        # Choose primary file (prefer dataset_pipeline)
        primary_file = None
        for file_path in found_files:
            if "dataset_pipeline/" in file_path and "ecosystem" not in file_path:
                primary_file = file_path
                break

        if not primary_file:
            for file_path in found_files:
                if "ecosystem" in file_path:
                    primary_file = file_path
                    break

        if not primary_file:
            primary_file = found_files[0]

        result["primary_file"] = primary_file

        # Get file statistics
        try:
            with open(primary_file) as f:
                content = f.read()

            result["file_stats"] = {
                "size_bytes": len(content),
                "size_kb": len(content) / 1024,
                "lines": len(content.split("\n")),
                "classes": content.count("class "),
                "functions": content.count("def "),
                "has_docstring": '"""' in content or "'''" in content,
                "has_logging": "logging" in content or "logger" in content,
                "has_error_handling": "try:" in content and "except" in content
            }


        except Exception as e:
            result["issues"].append(f"Could not read file: {e}")
            return result

        # Test functionality
        result["functionality_test"] = self.test_file_functionality(primary_file)

        # Determine status and enterprise readiness
        func_test = result["functionality_test"]
        file_stats = result["file_stats"]

        if func_test["syntax_valid"] and func_test["imports_successfully"]:
            if func_test["runs_without_error"] and func_test["enterprise_quality"]:
                result["status"] = "ENTERPRISE_READY"
                result["enterprise_ready"] = True
            elif func_test["runs_without_error"]:
                result["status"] = "FUNCTIONAL"
            else:
                result["status"] = "PARTIAL"
        else:
            result["status"] = "BROKEN"

        # Add issues and recommendations
        if func_test["quality_issues"]:
            result["issues"].extend(func_test["quality_issues"])

        if not func_test["enterprise_quality"]:
            if file_stats["size_kb"] < 5:
                result["recommendations"].append("Increase implementation depth")
            if file_stats["functions"] < 5:
                result["recommendations"].append("Add more comprehensive functionality")
            if not file_stats["has_docstring"]:
                result["recommendations"].append("Add comprehensive documentation")
            if not file_stats["has_error_handling"]:
                result["recommendations"].append("Add robust error handling")
            if not file_stats["has_logging"]:
                result["recommendations"].append("Add enterprise logging")

        return result

    def run_final_audit(self) -> dict[str, Any]:
        """Run final comprehensive audit"""

        # Define all 36 tasks as claimed in phase6.md
        tasks = [
            # Phase 1: Ecosystem-Scale Data Processing Pipeline
            ("6.1", "distributed_architecture.py", "Distributed processing architecture"),
            ("6.2", "data_fusion_engine.py", "Intelligent data fusion algorithms"),
            ("6.3", "quality_assessment_framework.py", "Hierarchical quality assessment framework"),
            ("6.4", "deduplication.py", "Automated conversation deduplication"),
            ("6.5", "cross_dataset_linker.py", "Cross-dataset conversation linking"),
            ("6.6", "metadata_schema.py", "Unified metadata schema"),

            # Phase 2: Advanced Therapeutic Intelligence
            ("6.7", "therapeutic_intelligence.py", "Comprehensive therapeutic approach classification"),
            ("6.8", "condition_pattern_recognition.py", "Mental health condition pattern recognition"),
            ("6.9", "outcome_prediction.py", "Therapeutic outcome prediction models"),
            ("6.10", "crisis_intervention_detector.py", "Crisis intervention detection and escalation"),
            ("6.11", "personality_adapter.py", "Personality-aware conversation adaptation"),
            ("6.12", "cultural_competency_generator.py", "Cultural competency and diversity-aware response generation"),

            # Phase 3: Multi-Modal Integration
            ("6.13", "audio_emotion_integration.py", "Audio emotion recognition integration"),
            ("6.14", "multimodal_disorder_analysis.py", "Multi-modal mental disorder analysis pipeline"),
            ("6.15", "emotion_cause_extraction.py", "Emotion cause extraction and intervention mapping"),
            ("6.16", "tfidf_clusterer.py", "TF-IDF feature-based conversation clustering"),
            ("6.17", "temporal_reasoner.py", "Temporal reasoning integration"),
            ("6.18", "evidence_validator.py", "Scientific evidence-based practice validation"),

            # Phase 4: Intelligent Dataset Balancing & Optimization
            ("6.19", "priority_weighted_sampler.py", "Priority-weighted sampling algorithms"),
            ("6.20", "condition_balancer.py", "Condition-specific balancing system"),
            ("6.21", "approach_diversity_optimizer.py", "Therapeutic approach diversity optimization"),
            ("6.22", "demographic_balancer.py", "Demographic and cultural diversity balancing"),
            ("6.23", "complexity_stratifier.py", "Conversation complexity stratification"),
            ("6.24", "crisis_routine_balancer.py", "Crisis-to-routine conversation ratio optimization"),

            # Phase 5: Advanced Quality Validation & Safety Systems
            ("6.25", "multi_tier_validator.py", "Multi-tier quality validation system"),
            ("6.26", "dsm5_accuracy_validator.py", "DSM-5 therapeutic accuracy validation"),
            ("6.27", "safety_ethics_validator.py", "Conversation safety and ethics validation"),
            ("6.28", "effectiveness_predictor.py", "Therapeutic effectiveness prediction"),
            ("6.29", "coherence_validator.py", "Conversation coherence validation using CoT reasoning"),
            ("6.30", "realtime_quality_monitor.py", "Real-time conversation quality monitoring"),

            # Phase 6: Production Deployment & Adaptive Learning
            ("6.31", "production_exporter.py", "Production-ready dataset export with tiered access"),
            ("6.32", "adaptive_learner.py", "Adaptive learning pipeline"),
            ("6.33", "analytics_dashboard.py", "Comprehensive analytics dashboard"),
            ("6.34", "automated_maintenance.py", "Automated dataset update and maintenance procedures"),
            ("6.35", "feedback_loops.py", "Conversation effectiveness feedback loops"),
            ("6.36", "comprehensive_api.py", "Comprehensive documentation and API"),
        ]

        # Audit each task
        for task_id, filename, description in tasks:
            result = self.audit_task(task_id, filename, description)
            self.results[task_id] = result

        return self.generate_final_report()

    def generate_final_report(self) -> dict[str, Any]:
        """Generate final audit report"""

        # Count statuses
        enterprise_ready = sum(1 for r in self.results.values() if r["status"] == "ENTERPRISE_READY")
        functional = sum(1 for r in self.results.values() if r["status"] == "FUNCTIONAL")
        partial = sum(1 for r in self.results.values() if r["status"] == "PARTIAL")
        broken = sum(1 for r in self.results.values() if r["status"] == "BROKEN")
        missing = sum(1 for r in self.results.values() if r["status"] == "MISSING")


        working_total = enterprise_ready + functional

        # Phase breakdown
        phases = {
            "Phase 1": [f"6.{i}" for i in range(1, 7)],
            "Phase 2": [f"6.{i}" for i in range(7, 13)],
            "Phase 3": [f"6.{i}" for i in range(13, 19)],
            "Phase 4": [f"6.{i}" for i in range(19, 25)],
            "Phase 5": [f"6.{i}" for i in range(25, 31)],
            "Phase 6": [f"6.{i}" for i in range(31, 37)],
        }

        for _phase_name, task_ids in phases.items():
            phase_enterprise = sum(1 for tid in task_ids if self.results.get(tid, {}).get("status") == "ENTERPRISE_READY")
            phase_working = sum(1 for tid in task_ids if self.results.get(tid, {}).get("status") in ["ENTERPRISE_READY", "FUNCTIONAL"])
            total_phase = len(task_ids)

            if total_phase in (phase_enterprise, phase_working) or phase_working > 0:
                pass
            else:
                pass


        # Enterprise ready files
        if enterprise_ready > 0:
            for result in self.results.values():
                if result["status"] == "ENTERPRISE_READY":
                    result["file_stats"]

        # Functional but not enterprise ready
        if functional > 0:
            for result in self.results.values():
                if result["status"] == "FUNCTIONAL":
                    result["file_stats"]

        # Issues found
        issues_found = [r for r in self.results.values() if r["issues"]]
        if issues_found:
            for result in issues_found:
                if result["issues"]:
                    pass

        # Missing files
        missing_files = [r for r in self.results.values() if r["status"] == "MISSING"]
        if missing_files:
            for result in missing_files:
                pass

        # Final assessment
        if enterprise_ready == self.total_tasks:
            overall_status = "ENTERPRISE_READY"
        elif working_total == self.total_tasks:
            overall_status = "FUNCTIONALLY_COMPLETE"
        elif working_total >= self.total_tasks * 0.9:
            overall_status = "NEARLY_COMPLETE"
        else:
            overall_status = "INCOMPLETE"

        # Business impact assessment
        if overall_status in {"ENTERPRISE_READY", "FUNCTIONALLY_COMPLETE"}:
            pass
        else:
            pass

        return {
            "audit_date": datetime.now().isoformat(),
            "total_tasks": self.total_tasks,
            "enterprise_ready": enterprise_ready,
            "functional": functional,
            "partial": partial,
            "broken": broken,
            "missing": missing,
            "working_total": working_total,
            "enterprise_readiness_rate": enterprise_ready / self.total_tasks,
            "functional_completion_rate": working_total / self.total_tasks,
            "overall_status": overall_status,
            "detailed_results": self.results
        }

def main():
    """Run final enterprise audit"""
    auditor = FinalEnterpriseAuditor()
    report = auditor.run_final_audit()

    # Save detailed report
    with open("/home/vivi/pixelated/ai/pipelines/orchestrator/final_enterprise_audit_report.json", "w") as f:
        json.dump(report, f, indent=2, default=str)


    # Return appropriate exit code
    if report["overall_status"] == "ENTERPRISE_READY":
        return 0
    if report["overall_status"] == "FUNCTIONALLY_COMPLETE":
        return 1
    return 2

if __name__ == "__main__":
    sys.exit(main())
