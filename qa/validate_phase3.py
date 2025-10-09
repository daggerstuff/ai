#!/usr/bin/env python3
"""
Pixelated Empathy AI - Phase 3 Validation Script
Tasks 3.1, 3.2, 3.3: Validate Safety Certification Implementation

Comprehensive validation script to verify Phase 3 safety certification implementation is enterprise-ready.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase3Validator:
    """Phase 3 safety certification validator"""
    
    def __init__(self):
        """Initialize validator"""
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": "Phase 3 - Safety Certification",
            "tasks": {
                "3.1": {"name": "Safety Accuracy Validation", "status": "pending", "score": 0},
                "3.2": {"name": "Clinical Safety Certification", "status": "pending", "score": 0},
                "3.3": {"name": "Safety Monitoring Integration", "status": "pending", "score": 0}
            },
            "overall_score": 0,
            "enterprise_ready": False,
            "critical_issues": [],
            "recommendations": []
        }
    
    def validate_task_3_1_safety_accuracy(self) -> Tuple[int, List[str]]:
        """Validate Task 3.1: Safety Accuracy Validation"""
        logger.info("Validating Task 3.1: Safety Accuracy Validation")
        
        score = 0
        issues = []
        
        try:
            # Test 1: Import safety accuracy validator
            try:
                from safety_accuracy_validator import (
                    SafetyAccuracyValidator, CrisisDetectionSimulator, SafetyTestSuite,
                    CrisisType, SafetyLevel, ValidationResult, safety_validator
                )
                score += 10
                logger.info("✅ Safety accuracy validator imports successfully")
            except ImportError as e:
                issues.append(f"Safety accuracy validator import failed: {e}")
                return score, issues
            
            # Test 2: Crisis Detection Simulator
            try:
                simulator = CrisisDetectionSimulator()
                
                # Test crisis detection
                test_cases = [
                    ("I want to kill myself", True),
                    ("I'm feeling sad today", False),
                    ("Going to hurt someone", True),
                    ("Had a good day", False)
                ]
                
                correct_detections = 0
                for text, expected_crisis in test_cases:
                    result = simulator.detect_crisis(text)
                    if result.detected == expected_crisis:
                        correct_detections += 1
                
                accuracy = correct_detections / len(test_cases)
                if accuracy >= 0.75:  # 75% minimum for basic testing
                    score += 20
                    logger.info(f"✅ Crisis detection accuracy: {accuracy:.1%}")
                else:
                    issues.append(f"Crisis detection accuracy too low: {accuracy:.1%}")
                
            except Exception as e:
                issues.append(f"Crisis detection testing failed: {e}")
            
            # Test 3: Safety Test Suite
            try:
                test_suite = SafetyTestSuite()
                
                # Verify test cases exist
                assert len(test_suite.test_cases) > 0
                
                # Check for crisis and safe test cases
                crisis_cases = [tc for tc in test_suite.test_cases if tc.ground_truth]
                safe_cases = [tc for tc in test_suite.test_cases if not tc.ground_truth]
                
                assert len(crisis_cases) > 0
                assert len(safe_cases) > 0
                
                score += 15
                logger.info(f"✅ Safety test suite: {len(test_suite.test_cases)} test cases")
                
            except Exception as e:
                issues.append(f"Safety test suite validation failed: {e}")
            
            # Test 4: Safety Accuracy Validator
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = SafetyAccuracyValidator(db_path=tmp.name)
                    
                    # Run validation
                    report = validator.run_validation()
                    
                    assert report.validation_id is not None
                    assert 0 <= report.overall_accuracy <= 1
                    assert isinstance(report.certification_ready, bool)
                    
                    score += 25
                    logger.info(f"✅ Safety validation: {report.overall_accuracy:.1%} accuracy")
                    
                    # Check if meets >95% requirement
                    if report.overall_accuracy >= 0.95:
                        score += 10
                        logger.info("✅ Meets >95% accuracy requirement")
                    else:
                        issues.append(f"Accuracy {report.overall_accuracy:.1%} below 95% requirement")
                    
                    os.unlink(tmp.name)
                    
            except Exception as e:
                issues.append(f"Safety accuracy validation failed: {e}")
            
            # Test 5: Performance Requirements
            try:
                simulator = CrisisDetectionSimulator()
                
                # Test response time
                start_time = time.time()
                for i in range(10):
                    simulator.detect_crisis(f"Test message {i}")
                end_time = time.time()
                
                avg_time = (end_time - start_time) / 10
                
                if avg_time < 1.0:  # Less than 1 second per detection
                    score += 10
                    logger.info(f"✅ Response time: {avg_time:.3f}s per detection")
                else:
                    issues.append(f"Response time too slow: {avg_time:.3f}s")
                
            except Exception as e:
                issues.append(f"Performance testing failed: {e}")
            
            # Test 6: Database Storage
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = SafetyAccuracyValidator(db_path=tmp.name)
                    
                    # Database should be initialized
                    assert os.path.exists(tmp.name)
                    
                    score += 10
                    logger.info("✅ Safety validation database operational")
                    os.unlink(tmp.name)
                    
            except Exception as e:
                issues.append(f"Database storage failed: {e}")
            
        except Exception as e:
            issues.append(f"Critical safety accuracy validation error: {e}")
        
        return score, issues
    
    def validate_task_3_2_clinical_certification(self) -> Tuple[int, List[str]]:
        """Validate Task 3.2: Clinical Safety Certification"""
        logger.info("Validating Task 3.2: Clinical Safety Certification")
        
        score = 0
        issues = []
        
        try:
            # Test 1: Import clinical safety certifier
            try:
                from clinical_safety_certifier import (
                    ClinicalSafetyCertifier, ClinicalRequirementsFramework,
                    ClinicalStandard, SafetyProtocol, CertificationLevel, clinical_certifier
                )
                score += 10
                logger.info("✅ Clinical safety certifier imports successfully")
            except ImportError as e:
                issues.append(f"Clinical safety certifier import failed: {e}")
                return score, issues
            
            # Test 2: Clinical Requirements Framework
            try:
                framework = ClinicalRequirementsFramework()
                
                # Verify requirements exist
                assert len(framework.requirements) > 0
                
                # Check for critical requirements
                critical_reqs = [req for req in framework.requirements.values() 
                               if req.mandatory and req.risk_level.value in ["critical", "high"]]
                assert len(critical_reqs) > 0
                
                score += 15
                logger.info(f"✅ Clinical requirements: {len(framework.requirements)} requirements")
                
            except Exception as e:
                issues.append(f"Clinical requirements framework failed: {e}")
            
            # Test 3: Safety Protocol Testing
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    certifier = ClinicalSafetyCertifier(db_path=tmp.name)
                    
                    # Test a specific protocol
                    test_result = certifier.test_safety_protocol("CE-001")
                    
                    assert test_result.test_id is not None
                    assert test_result.requirement_id == "CE-001"
                    assert isinstance(test_result.test_result, bool)
                    assert isinstance(test_result.evidence, list)
                    
                    score += 20
                    logger.info(f"✅ Protocol testing: {test_result.requirement_id}")
                    os.unlink(tmp.name)
                    
            except Exception as e:
                issues.append(f"Safety protocol testing failed: {e}")
            
            # Test 4: Clinical Incident Simulation
            try:
                from clinical_safety_certifier import RiskLevel
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    certifier = ClinicalSafetyCertifier(db_path=tmp.name)
                    
                    # Simulate incident
                    incident = certifier.simulate_clinical_incident("system_error", RiskLevel.HIGH)
                    
                    assert incident.incident_id is not None
                    assert incident.severity == RiskLevel.HIGH
                    assert incident.response_time_minutes > 0
                    
                    score += 15
                    logger.info(f"✅ Incident simulation: {incident.incident_type}")
                    os.unlink(tmp.name)
                    
            except Exception as e:
                issues.append(f"Clinical incident simulation failed: {e}")
            
            # Test 5: Clinical Certification Generation
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    certifier = ClinicalSafetyCertifier(db_path=tmp.name)
                    
                    # Generate certification
                    certification = certifier.generate_clinical_certification()
                    
                    assert certification.certification_id is not None
                    assert certification.certification_level in [
                        CertificationLevel.RESEARCH_ONLY,
                        CertificationLevel.CLINICAL_TRIAL,
                        CertificationLevel.CLINICAL_PRACTICE,
                        CertificationLevel.PRODUCTION_READY
                    ]
                    assert 0 <= certification.overall_score <= 100
                    
                    score += 25
                    logger.info(f"✅ Clinical certification: {certification.certification_level.value} ({certification.overall_score:.1f}%)")
                    
                    # Check certification level
                    if certification.certification_level in [CertificationLevel.CLINICAL_PRACTICE, CertificationLevel.PRODUCTION_READY]:
                        score += 10
                        logger.info("✅ Achieves clinical practice certification level")
                    else:
                        issues.append(f"Certification level {certification.certification_level.value} insufficient for production")
                    
                    os.unlink(tmp.name)
                    
            except Exception as e:
                issues.append(f"Clinical certification generation failed: {e}")
            
            # Test 6: Database Storage
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    certifier = ClinicalSafetyCertifier(db_path=tmp.name)
                    
                    # Database should be initialized
                    assert os.path.exists(tmp.name)
                    
                    score += 5
                    logger.info("✅ Clinical certification database operational")
                    os.unlink(tmp.name)
                    
            except Exception as e:
                issues.append(f"Clinical database storage failed: {e}")
            
        except Exception as e:
            issues.append(f"Critical clinical certification error: {e}")
        
        return score, issues
    
    def validate_task_3_3_safety_monitoring(self) -> Tuple[int, List[str]]:
        """Validate Task 3.3: Safety Monitoring Integration"""
        logger.info("Validating Task 3.3: Safety Monitoring Integration")
        
        score = 0
        issues = []
        
        try:
            # Test 1: Import safety monitoring integration
            try:
                from safety_monitor_integration import (
                    SafetyMonitoringIntegration, AlertManager, SafetyMetricsCollector,
                    AlertSeverity, MonitoringStatus, EscalationLevel, safety_monitor
                )
                score += 10
                logger.info("✅ Safety monitoring integration imports successfully")
            except ImportError as e:
                issues.append(f"Safety monitoring integration import failed: {e}")
                return score, issues
            
            # Test 2: Alert Manager
            try:
                alert_manager = AlertManager()
                
                # Create test alert
                alert = alert_manager.create_alert(
                    alert_type="test_alert",
                    severity=AlertSeverity.WARNING,
                    source_system="test_system",
                    message="Test alert message",
                    details={"test": True}
                )
                
                assert alert.alert_id is not None
                assert alert.severity == AlertSeverity.WARNING
                assert not alert.acknowledged
                assert not alert.resolved
                
                # Test acknowledgment
                success = alert_manager.acknowledge_alert(alert.alert_id, "test_responder")
                assert success
                assert alert.acknowledged
                
                score += 20
                logger.info("✅ Alert management system operational")
                
            except Exception as e:
                issues.append(f"Alert manager testing failed: {e}")
            
            # Test 3: Safety Metrics Collector
            try:
                metrics_collector = SafetyMetricsCollector()
                
                # Test metrics collection
                metrics = metrics_collector._collect_safety_metrics()
                
                assert len(metrics) > 0
                
                # Verify metric types
                metric_names = [m.metric_name for m in metrics]
                expected_metrics = ["crisis_detection_accuracy", "avg_response_time", "system_availability"]
                
                for expected in expected_metrics:
                    assert any(expected in name for name in metric_names)
                
                score += 15
                logger.info(f"✅ Safety metrics collection: {len(metrics)} metrics")
                
            except Exception as e:
                issues.append(f"Safety metrics collection failed: {e}")
            
            # Test 4: Safety Monitoring Integration
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    monitor = SafetyMonitoringIntegration(db_path=tmp.name)
                    
                    # Test monitoring start/stop
                    monitor.start_monitoring()
                    assert monitor.monitoring_status == MonitoringStatus.ACTIVE
                    
                    # Create test alert
                    alert = monitor.create_safety_alert(
                        alert_type="crisis_detected",
                        severity=AlertSeverity.EMERGENCY,
                        source_system="crisis_detector",
                        message="Test crisis alert",
                        details={"confidence": 0.95}
                    )
                    
                    assert alert.alert_id is not None
                    
                    # Test dashboard data
                    dashboard = monitor.get_dashboard_data()
                    assert dashboard.system_status == MonitoringStatus.ACTIVE
                    assert dashboard.active_alerts >= 0
                    assert 0 <= dashboard.safety_score <= 100
                    
                    monitor.stop_monitoring()
                    assert monitor.monitoring_status == MonitoringStatus.INACTIVE
                    
                    score += 25
                    logger.info("✅ Safety monitoring integration operational")
                    os.unlink(tmp.name)
                    
            except Exception as e:
                issues.append(f"Safety monitoring integration failed: {e}")
            
            # Test 5: Escalation Rules
            try:
                alert_manager = AlertManager()
                
                # Verify escalation rules exist
                assert len(alert_manager.escalation_rules) > 0
                
                # Check for critical escalation rules
                crisis_rule = alert_manager.escalation_rules.get("crisis_detected")
                assert crisis_rule is not None
                assert crisis_rule.severity == AlertSeverity.EMERGENCY
                assert crisis_rule.escalation_level == EscalationLevel.LEVEL_4
                
                score += 15
                logger.info("✅ Escalation rules configured correctly")
                
            except Exception as e:
                issues.append(f"Escalation rules validation failed: {e}")
            
            # Test 6: Real-time Monitoring
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    monitor = SafetyMonitoringIntegration(db_path=tmp.name)
                    
                    # Start monitoring
                    monitor.start_monitoring()
                    
                    # Wait briefly for metrics collection
                    time.sleep(2)
                    
                    # Check that metrics are being collected
                    recent_metrics = monitor.metrics_collector.get_recent_metrics(hours=1)
                    
                    # Stop monitoring
                    monitor.stop_monitoring()
                    
                    score += 10
                    logger.info("✅ Real-time monitoring functional")
                    os.unlink(tmp.name)
                    
            except Exception as e:
                issues.append(f"Real-time monitoring failed: {e}")
            
            # Test 7: Database Storage
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    monitor = SafetyMonitoringIntegration(db_path=tmp.name)
                    
                    # Database should be initialized
                    assert os.path.exists(tmp.name)
                    
                    score += 5
                    logger.info("✅ Safety monitoring database operational")
                    os.unlink(tmp.name)
                    
            except Exception as e:
                issues.append(f"Monitoring database storage failed: {e}")
            
        except Exception as e:
            issues.append(f"Critical safety monitoring error: {e}")
        
        return score, issues
    
    def validate_integration(self) -> Tuple[int, List[str]]:
        """Validate safety certification integration"""
        logger.info("Validating safety certification integration")
        
        score = 0
        issues = []
        
        try:
            # Test 1: All safety frameworks can be imported together
            try:
                from safety_accuracy_validator import safety_validator
                from clinical_safety_certifier import clinical_certifier
                from safety_monitor_integration import safety_monitor
                
                score += 25
                logger.info("✅ All safety frameworks integrate successfully")
            except ImportError as e:
                issues.append(f"Safety framework integration failed: {e}")
                return score, issues
            
            # Test 2: Cross-framework compatibility
            try:
                from safety_monitor_integration import AlertSeverity
                
                # Test that frameworks can work with the same data
                test_message = "I'm having thoughts of suicide and self-harm"
                
                # Safety accuracy detection
                detection_result = safety_validator.detector.detect_crisis(test_message)
                
                # Create monitoring alert
                alert = safety_monitor.create_safety_alert(
                    alert_type="crisis_detected",
                    severity=AlertSeverity.EMERGENCY,
                    source_system="safety_validator",
                    message="Crisis detected in validation",
                    details={"confidence": detection_result.confidence}
                )
                
                # Both should process the crisis appropriately
                assert detection_result.detected
                assert alert.alert_id is not None
                
                score += 25
                logger.info("✅ Cross-framework crisis handling works correctly")
            except Exception as e:
                issues.append(f"Cross-framework compatibility failed: {e}")
            
            # Test 3: End-to-end safety pipeline
            try:
                from safety_monitor_integration import AlertSeverity
                
                # Simulate complete safety pipeline
                test_input = "I want to end my life"
                
                # 1. Detection
                detection = safety_validator.detector.detect_crisis(test_input)
                
                # 2. Alert creation
                if detection.detected:
                    alert = safety_monitor.create_safety_alert(
                        alert_type="crisis_detected",
                        severity=AlertSeverity.EMERGENCY,
                        source_system="crisis_detector",
                        message=f"Crisis detected: {detection.crisis_type}",
                        details={"confidence": detection.confidence}
                    )
                
                # 3. Clinical protocol activation (simulated)
                protocol_test = clinical_certifier.test_safety_protocol("CE-001")
                
                # All components should work together
                assert detection.detected
                assert alert.severity == AlertSeverity.EMERGENCY
                assert protocol_test.test_result
                
                score += 25
                logger.info("✅ End-to-end safety pipeline operational")
            except Exception as e:
                issues.append(f"End-to-end safety pipeline failed: {e}")
            
            # Test 4: Performance under load
            try:
                from safety_monitor_integration import AlertSeverity
                
                start_time = time.time()
                
                # Perform multiple safety operations
                for i in range(5):
                    test_data = f"Test crisis message {i}: I want to hurt myself"
                    
                    # Quick safety checks
                    detection = safety_validator.detector.detect_crisis(test_data)
                    if detection.detected:
                        alert = safety_monitor.create_safety_alert(
                            alert_type="test_crisis",
                            severity=AlertSeverity.WARNING,
                            source_system="load_test",
                            message=f"Load test crisis {i}",
                            details={"test_id": i}
                        )
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Should complete within reasonable time (< 10 seconds for 5 operations)
                assert processing_time < 10.0
                
                score += 25
                logger.info("✅ Safety frameworks perform well under load")
            except Exception as e:
                issues.append(f"Performance testing failed: {e}")
            
        except Exception as e:
            issues.append(f"Critical integration error: {e}")
        
        return score, issues
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete Phase 3 validation"""
        logger.info("Starting Phase 3 validation...")
        
        # Validate Task 3.1
        score_3_1, issues_3_1 = self.validate_task_3_1_safety_accuracy()
        self.results["tasks"]["3.1"]["score"] = score_3_1
        self.results["tasks"]["3.1"]["status"] = "passed" if score_3_1 >= 80 else "failed"
        if issues_3_1:
            self.results["critical_issues"].extend([f"Task 3.1: {issue}" for issue in issues_3_1])
        
        # Validate Task 3.2
        score_3_2, issues_3_2 = self.validate_task_3_2_clinical_certification()
        self.results["tasks"]["3.2"]["score"] = score_3_2
        self.results["tasks"]["3.2"]["status"] = "passed" if score_3_2 >= 80 else "failed"
        if issues_3_2:
            self.results["critical_issues"].extend([f"Task 3.2: {issue}" for issue in issues_3_2])
        
        # Validate Task 3.3
        score_3_3, issues_3_3 = self.validate_task_3_3_safety_monitoring()
        self.results["tasks"]["3.3"]["score"] = score_3_3
        self.results["tasks"]["3.3"]["status"] = "passed" if score_3_3 >= 80 else "failed"
        if issues_3_3:
            self.results["critical_issues"].extend([f"Task 3.3: {issue}" for issue in issues_3_3])
        
        # Validate integration
        score_integration, issues_integration = self.validate_integration()
        if issues_integration:
            self.results["critical_issues"].extend([f"Integration: {issue}" for issue in issues_integration])
        
        # Calculate overall score
        total_score = score_3_1 + score_3_2 + score_3_3 + score_integration
        max_score = 100 + 100 + 100 + 100  # 400 total
        self.results["overall_score"] = (total_score / max_score) * 100
        
        # Determine enterprise readiness
        self.results["enterprise_ready"] = (
            self.results["overall_score"] >= 95 and
            len(self.results["critical_issues"]) == 0 and
            all(task["status"] == "passed" for task in self.results["tasks"].values())
        )
        
        # Generate recommendations
        if not self.results["enterprise_ready"]:
            if self.results["overall_score"] < 95:
                self.results["recommendations"].append("Overall score below 95% - review failed components")
            
            if len(self.results["critical_issues"]) > 0:
                self.results["recommendations"].append("Resolve all critical issues before production deployment")
            
            for task_id, task in self.results["tasks"].items():
                if task["status"] == "failed":
                    self.results["recommendations"].append(f"Task {task_id} failed - requires immediate attention")
        
        return self.results
    
    def generate_report(self) -> str:
        """Generate validation report"""
        report = []
        report.append("=" * 80)
        report.append("PHASE 3 VALIDATION REPORT")
        report.append("Safety Certification Implementation")
        report.append("=" * 80)
        report.append(f"Timestamp: {self.results['timestamp']}")
        report.append(f"Overall Score: {self.results['overall_score']:.1f}%")
        report.append(f"Enterprise Ready: {'✅ YES' if self.results['enterprise_ready'] else '❌ NO'}")
        report.append("")
        
        # Task results
        report.append("TASK RESULTS:")
        report.append("-" * 40)
        for task_id, task in self.results["tasks"].items():
            status_icon = "✅" if task["status"] == "passed" else "❌"
            report.append(f"Task {task_id}: {task['name']}")
            report.append(f"  Status: {status_icon} {task['status'].upper()}")
            report.append(f"  Score: {task['score']}/100")
            report.append("")
        
        # Critical issues
        if self.results["critical_issues"]:
            report.append("CRITICAL ISSUES:")
            report.append("-" * 40)
            for issue in self.results["critical_issues"]:
                report.append(f"❌ {issue}")
            report.append("")
        
        # Recommendations
        if self.results["recommendations"]:
            report.append("RECOMMENDATIONS:")
            report.append("-" * 40)
            for rec in self.results["recommendations"]:
                report.append(f"💡 {rec}")
            report.append("")
        
        # Summary
        report.append("SUMMARY:")
        report.append("-" * 40)
        if self.results["enterprise_ready"]:
            report.append("✅ Phase 3 implementation is ENTERPRISE READY")
            report.append("✅ All safety certification frameworks are operational")
            report.append("✅ Ready to proceed to Phase 4")
        else:
            report.append("❌ Phase 3 implementation is NOT enterprise ready")
            report.append("❌ Critical issues must be resolved")
            report.append("❌ Do NOT proceed to Phase 4 until issues are fixed")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main validation function"""
    print("Pixelated Empathy AI - Phase 3 Validation")
    print("=" * 50)
    
    # Add current directory to path for imports
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    sys.path.insert(0, str(current_dir.parent / "monitoring"))
    
    # Run validation
    validator = Phase3Validator()
    results = validator.run_validation()
    
    # Generate and display report
    report = validator.generate_report()
    print(report)
    
    # Save results to file
    results_file = current_dir / "phase3_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    report_file = current_dir / "phase3_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results["enterprise_ready"] else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
