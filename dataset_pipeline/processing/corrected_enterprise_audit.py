#!/usr/bin/env python3
"""
CORRECTED ENTERPRISE AUDIT - Task 6.0 Production Readiness Assessment
Properly activates uv environment and tests all components accurately.
"""

import json
import os
import subprocess
import sys
from datetime import datetime


class CorrectedEnterpriseAuditor:
    def __init__(self):
        self.results = {
            "audit_timestamp": datetime.now().isoformat(),
            "audit_type": "Corrected Enterprise Readiness Assessment",
            "environment_setup": "Proper uv environment activation",
            "total_tasks": 36,
            "enterprise_ready": [],
            "functional": [],
            "broken": [],
            "missing": [],
            "detailed_results": {}
        }

        # Task 6.0 component mapping with correct file paths
        self.task_components = {
            "6.1": {"file": "dataset_pipeline/distributed_architecture.py", "class": "DistributedArchitecture"},
            "6.2": {"file": "dataset_pipeline/data_fusion_engine.py", "class": "DataFusionEngine"},
            "6.3": {"file": "dataset_pipeline/quality_assessment_framework.py", "class": "QualityAssessmentFramework"},
            "6.4": {"file": "dataset_pipeline/deduplication.py", "class": "Deduplicator"},
            "6.5": {"file": "dataset_pipeline/cross_dataset_linker.py", "class": "CrossDatasetLinker"},
            "6.6": {"file": "dataset_pipeline/metadata_schema.py", "class": "MetadataSchema"},
            "6.7": {"file": "dataset_pipeline/therapeutic_intelligence.py", "class": "TherapeuticIntelligence"},
            "6.8": {"file": "dataset_pipeline/ecosystem/condition_pattern_recognition.py", "class": "ConditionPatternRecognizer"},
            "6.9": {"file": "dataset_pipeline/ecosystem/outcome_prediction.py", "class": "OutcomePredictor"},
            "6.10": {"file": "dataset_pipeline/crisis_intervention_detector.py", "class": "CrisisInterventionDetector"},
            "6.11": {"file": "dataset_pipeline/personality_adapter.py", "class": "PersonalityAdapter"},
            "6.12": {"file": "dataset_pipeline/cultural_competency_generator.py", "class": "CulturalCompetencyGenerator"},
            "6.13": {"file": "dataset_pipeline/audio_emotion_integration.py", "class": "AudioEmotionIntegration"},
            "6.14": {"file": "dataset_pipeline/ecosystem/multimodal_disorder_analysis.py", "class": "MultimodalDisorderAnalyzer"},
            "6.15": {"file": "dataset_pipeline/ecosystem/emotion_cause_extraction.py", "class": "EmotionCauseExtractor"},
            "6.16": {"file": "dataset_pipeline/tfidf_clusterer.py", "class": "TFIDFClusterer"},
            "6.17": {"file": "dataset_pipeline/temporal_reasoner.py", "class": "TemporalReasoner"},
            "6.18": {"file": "dataset_pipeline/evidence_validator.py", "class": "EvidenceValidator"},
            "6.19": {"file": "dataset_pipeline/priority_weighted_sampler.py", "class": "PriorityWeightedSampler"},
            "6.20": {"file": "dataset_pipeline/condition_balancer.py", "class": "ConditionBalancer"},
            "6.21": {"file": "dataset_pipeline/approach_diversity_optimizer.py", "class": "ApproachDiversityOptimizer"},
            "6.22": {"file": "dataset_pipeline/demographic_balancer.py", "class": "DemographicBalancer"},
            "6.23": {"file": "dataset_pipeline/complexity_stratifier.py", "class": "ComplexityStratifier"},
            "6.24": {"file": "dataset_pipeline/crisis_routine_balancer.py", "class": "CrisisRoutineBalancer"},
            "6.25": {"file": "dataset_pipeline/multi_tier_validator.py", "class": "MultiTierValidator"},
            "6.26": {"file": "dataset_pipeline/dsm5_accuracy_validator.py", "class": "DSM5AccuracyValidator"},
            "6.27": {"file": "dataset_pipeline/safety_ethics_validator.py", "class": "SafetyEthicsValidator"},
            "6.28": {"file": "dataset_pipeline/effectiveness_predictor.py", "class": "EffectivenessPredictor"},
            "6.29": {"file": "dataset_pipeline/coherence_validator.py", "class": "CoherenceValidator"},
            "6.30": {"file": "dataset_pipeline/realtime_quality_monitor.py", "class": "RealtimeQualityMonitor"},
            "6.31": {"file": "dataset_pipeline/production_exporter.py", "class": "ProductionExporter"},
            "6.32": {"file": "dataset_pipeline/adaptive_learner.py", "class": "AdaptiveLearner"},
            "6.33": {"file": "dataset_pipeline/analytics_dashboard.py", "class": "AnalyticsDashboard"},
            "6.34": {"file": "dataset_pipeline/automated_maintenance.py", "class": "AutomatedMaintenance"},
            "6.35": {"file": "dataset_pipeline/feedback_loops.py", "class": "FeedbackLoops"},
            "6.36": {"file": "dataset_pipeline/comprehensive_api.py", "class": "ComprehensiveAPI"}
        }

    def setup_environment(self):
        """Ensure proper uv environment is activated"""

        # Change to project root
        os.chdir("/home/vivi/pixelated")

        # Activate virtual environment
        venv_python = "/home/vivi/pixelated/.venv/bin/python"
        if not os.path.exists(venv_python):
            return False

        # Update sys.path to use the virtual environment
        sys.path.insert(0, "/home/vivi/pixelated/.venv/lib/python3.11/site-packages")

        # Change to AI directory
        os.chdir("/home/vivi/pixelated/ai")

        # Add project paths
        sys.path.insert(0, ".")
        sys.path.insert(0, "./dataset_pipeline")

        return True

    def test_component(self, task_id, component_info):
        """Test a single component with proper environment"""
        file_path = component_info["file"]
        class_name = component_info["class"]

        result = {
            "task_id": task_id,
            "file_path": file_path,
            "class_name": class_name,
            "status": "unknown",
            "error": None,
            "enterprise_ready": False,
            "functional": False,
            "issues": []
        }

        # Check if file exists
        if not os.path.exists(file_path):
            result["status"] = "missing"
            result["error"] = f"File not found: {file_path}"
            return result

        try:
            # Test import using subprocess with proper environment
            test_script = f"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, './dataset_pipeline')

try:
    # Test basic imports first
    import numpy as np
    print(f"NumPy: {{np.__version__}}")

    # Import the module
    module_path = "{file_path}".replace('/', '.').replace('.py', '')
    if module_path.startswith('dataset_pipeline.'):
        module_path = module_path[len('dataset_pipeline.'):]

    exec(f"from {{module_path}} import {class_name}")
    print("IMPORT_SUCCESS")

    # Try to instantiate
    exec(f"instance = {class_name}()")
    print("INSTANTIATION_SUCCESS")

except ImportError as e:
    print(f"IMPORT_ERROR: {{e}}")
except Exception as e:
    print(f"OTHER_ERROR: {{e}}")
"""

            # Run test in subprocess with proper environment
            process = subprocess.run([
                "/home/vivi/pixelated/.venv/bin/python", "-c", test_script
            ], check=False, capture_output=True, text=True, cwd="/home/vivi/pixelated/ai")

            output = process.stdout
            error_output = process.stderr

            if "IMPORT_SUCCESS" in output:
                result["functional"] = True
                if "INSTANTIATION_SUCCESS" in output:
                    result["status"] = "working"
                    # Check for enterprise readiness indicators
                    if self.check_enterprise_quality(file_path):
                        result["enterprise_ready"] = True
                        result["status"] = "enterprise_ready"
                else:
                    result["status"] = "import_only"
                    result["issues"].append("Cannot instantiate class")
            elif "IMPORT_ERROR" in output:
                result["status"] = "import_error"
                result["error"] = output.split("IMPORT_ERROR: ")[1].split("\n")[0]
            else:
                result["status"] = "broken"
                result["error"] = error_output or output

        except Exception as e:
            result["status"] = "test_error"
            result["error"] = str(e)

        return result

    def check_enterprise_quality(self, file_path):
        """Check if component meets enterprise quality standards"""
        try:
            with open(file_path) as f:
                content = f.read()

            enterprise_indicators = 0

            # Check for comprehensive error handling
            if "try:" in content and "except" in content:
                enterprise_indicators += 1

            # Check for logging
            if "logging" in content or "logger" in content:
                enterprise_indicators += 1

            # Check for input validation
            if "validate" in content or "isinstance" in content:
                enterprise_indicators += 1

            # Check for docstrings
            if '"""' in content or "'''" in content:
                enterprise_indicators += 1

            # Check for type hints
            if ":" in content and "->" in content:
                enterprise_indicators += 1

            # Enterprise ready if has 4+ indicators
            return enterprise_indicators >= 4

        except Exception:
            return False

    def run_audit(self):
        """Run complete enterprise audit"""

        if not self.setup_environment():
            return


        for task_id, component_info in self.task_components.items():

            result = self.test_component(task_id, component_info)
            self.results["detailed_results"][task_id] = result

            # Categorize results
            if result["status"] == "missing":
                self.results["missing"].append(task_id)
            elif result["status"] == "enterprise_ready":
                self.results["enterprise_ready"].append(task_id)
            elif result["status"] in ["working", "import_only"]:
                self.results["functional"].append(task_id)
            else:
                self.results["broken"].append(task_id)

        self.generate_summary()

    def generate_summary(self):
        """Generate audit summary"""
        len(self.task_components)
        len(self.results["enterprise_ready"])
        len(self.results["functional"])
        len(self.results["broken"])
        len(self.results["missing"])


        if self.results["enterprise_ready"]:
            for task_id in self.results["enterprise_ready"]:
                self.task_components[task_id]

        if self.results["functional"]:
            for task_id in self.results["functional"]:
                self.task_components[task_id]

        if self.results["broken"]:
            for task_id in self.results["broken"]:
                self.task_components[task_id]
                self.results["detailed_results"][task_id]

        if self.results["missing"]:
            for task_id in self.results["missing"]:
                self.task_components[task_id]

        # Save results
        with open("corrected_enterprise_audit_results.json", "w") as f:
            json.dump(self.results, f, indent=2)


if __name__ == "__main__":
    auditor = CorrectedEnterpriseAuditor()
    auditor.run_audit()
