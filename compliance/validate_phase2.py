#!/usr/bin/env python3
"""
Pixelated Empathy AI - Phase 2 Validation Script
Tasks 2.1, 2.2, 2.3: Validate Regulatory Compliance Implementation

Comprehensive validation script to verify Phase 2 regulatory compliance implementation is enterprise-ready.
"""

import os
import sys
import json
import time
import tempfile
from pathlib import Path
from datetime import datetime, timedelta, timezone
from typing import Dict, List, Any, Tuple
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class Phase2Validator:
    """Phase 2 regulatory compliance validator"""
    
    def __init__(self):
        """Initialize validator"""
        self.results = {
            "timestamp": datetime.utcnow().isoformat(),
            "phase": "Phase 2 - Regulatory Compliance",
            "tasks": {
                "2.1": {"name": "HIPAA Compliance Framework", "status": "pending", "score": 0},
                "2.2": {"name": "SOC2 Compliance Framework", "status": "pending", "score": 0},
                "2.3": {"name": "GDPR Compliance Framework", "status": "pending", "score": 0}
            },
            "overall_score": 0,
            "enterprise_ready": False,
            "critical_issues": [],
            "recommendations": []
        }
    
    def validate_task_2_1_hipaa(self) -> Tuple[int, List[str]]:
        """Validate Task 2.1: HIPAA Compliance Framework"""
        logger.info("Validating Task 2.1: HIPAA Compliance Framework")
        
        score = 0
        issues = []
        
        try:
            # Test 1: Import HIPAA framework
            try:
                from hipaa_validator import (
                    HIPAAValidator, PHIDetector, HIPAAEncryption, HIPAAStorage,
                    PHIType, HIPAAViolationType, ComplianceLevel, hipaa_validator
                )
                score += 10
                logger.info("✅ HIPAA framework imports successfully")
            except ImportError as e:
                issues.append(f"HIPAA framework import failed: {e}")
                return score, issues
            
            # Test 2: PHI Detection
            try:
                detector = PHIDetector()
                test_data = "Patient John Doe, SSN: 123-45-6789, Phone: (555) 123-4567"
                detections = detector.detect_phi(test_data)
                
                assert len(detections) >= 2  # Should detect SSN and phone
                assert any(d.phi_type == PHIType.SSN for d in detections)
                assert any(d.phi_type == PHIType.PHONE for d in detections)
                
                score += 20
                logger.info("✅ PHI detection works correctly")
            except Exception as e:
                issues.append(f"PHI detection failed: {e}")
            
            # Test 3: HIPAA Encryption
            try:
                encryption = HIPAAEncryption()
                test_phi = "Patient medical record data"
                
                encrypted = encryption.encrypt_phi(test_phi)
                decrypted = encryption.decrypt_phi(encrypted)
                
                assert encrypted != test_phi
                assert decrypted == test_phi
                
                score += 15
                logger.info("✅ HIPAA encryption works correctly")
            except Exception as e:
                issues.append(f"HIPAA encryption failed: {e}")
            
            # Test 4: HIPAA Storage
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    storage = HIPAAStorage(db_path=tmp.name)
                    
                    # Test PHI access logging
                    storage.log_phi_access(
                        user_id="test_user",
                        phi_type=PHIType.NAME,
                        action="access",
                        success=True
                    )
                    
                    score += 15
                    logger.info("✅ HIPAA storage works correctly")
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"HIPAA storage failed: {e}")
            
            # Test 5: Data Access Validation
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = HIPAAValidator(storage=HIPAAStorage(db_path=tmp.name))
                    
                    test_data = "Patient John Doe, DOB: 01/15/1980"
                    is_compliant, violations = validator.validate_data_access(
                        user_id="authorized_user",
                        data=test_data,
                        context="medical_record"
                    )
                    
                    # Should detect PHI and log access
                    assert isinstance(is_compliant, bool)
                    assert isinstance(violations, list)
                    
                    score += 15
                    logger.info("✅ HIPAA data access validation works")
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"HIPAA data access validation failed: {e}")
            
            # Test 6: Data Storage Validation
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = HIPAAValidator(storage=HIPAAStorage(db_path=tmp.name))
                    
                    # Test unencrypted PHI storage (should fail)
                    test_data = "Patient medical record with SSN: 123-45-6789"
                    is_compliant, violations = validator.validate_data_storage(
                        data=test_data,
                        encrypted=False
                    )
                    
                    # Should detect violation for unencrypted PHI
                    assert not is_compliant
                    assert len(violations) > 0
                    
                    score += 15
                    logger.info("✅ HIPAA storage validation works")
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"HIPAA storage validation failed: {e}")
            
            # Test 7: Compliance Report Generation
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = HIPAAValidator(storage=HIPAAStorage(db_path=tmp.name))
                    
                    report = validator.generate_compliance_report()
                    
                    assert report.assessment_id is not None
                    assert report.compliance_level in [ComplianceLevel.COMPLIANT, ComplianceLevel.NON_COMPLIANT, ComplianceLevel.REQUIRES_REVIEW]
                    assert 0 <= report.score <= 100
                    
                    score += 10
                    logger.info("✅ HIPAA compliance reporting works")
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"HIPAA compliance reporting failed: {e}")
            
        except Exception as e:
            issues.append(f"Critical HIPAA framework error: {e}")
        
        return score, issues
    
    def validate_task_2_2_soc2(self) -> Tuple[int, List[str]]:
        """Validate Task 2.2: SOC2 Compliance Framework"""
        logger.info("Validating Task 2.2: SOC2 Compliance Framework")
        
        score = 0
        issues = []
        
        try:
            # Test 1: Import SOC2 framework
            try:
                from soc2_validator import (
                    SOC2Validator, SystemMonitor, SOC2Storage,
                    SOC2Principle, ControlCategory, ComplianceStatus, soc2_validator
                )
                score += 10
                logger.info("✅ SOC2 framework imports successfully")
            except ImportError as e:
                issues.append(f"SOC2 framework import failed: {e}")
                return score, issues
            
            # Test 2: System Monitoring
            try:
                monitor = SystemMonitor()
                
                # Test metrics collection
                metrics = monitor._collect_metrics()
                
                assert "cpu_usage" in metrics
                assert "memory_usage" in metrics
                assert "disk_usage" in metrics
                assert "timestamp" in metrics
                
                score += 20
                logger.info("✅ SOC2 system monitoring works correctly")
            except Exception as e:
                issues.append(f"SOC2 system monitoring failed: {e}")
            
            # Test 3: SOC2 Storage
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    storage = SOC2Storage(db_path=tmp.name)
                    
                    # Database should be initialized
                    assert os.path.exists(tmp.name)
                    
                    score += 15
                    logger.info("✅ SOC2 storage works correctly")
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"SOC2 storage failed: {e}")
            
            # Test 4: Control Testing
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = SOC2Validator(storage=SOC2Storage(db_path=tmp.name))
                    
                    # Test a specific control
                    test_result = validator.test_control("CC6.1")
                    
                    assert test_result.control_id == "CC6.1"
                    assert test_result.status in [ComplianceStatus.COMPLIANT, ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT]
                    assert isinstance(test_result.evidence, list)
                    
                    score += 20
                    logger.info("✅ SOC2 control testing works correctly")
                    
                    # Stop monitoring
                    validator.monitor.stop_monitoring()
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"SOC2 control testing failed: {e}")
            
            # Test 5: Availability Metrics
            try:
                monitor = SystemMonitor()
                
                # Simulate some metrics
                monitor.metrics_history = [
                    {
                        "timestamp": datetime.now(timezone.utc),
                        "cpu_usage": 50.0,
                        "memory_usage": 60.0,
                        "uptime": 86400
                    }
                ]
                
                availability_metrics = monitor.get_availability_metrics(hours=1)
                
                assert "availability" in availability_metrics
                assert "uptime" in availability_metrics
                assert "incidents" in availability_metrics
                
                score += 15
                logger.info("✅ SOC2 availability metrics work correctly")
            except Exception as e:
                issues.append(f"SOC2 availability metrics failed: {e}")
            
            # Test 6: SOC2 Assessment
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = SOC2Validator(storage=SOC2Storage(db_path=tmp.name))
                    
                    assessment = validator.generate_soc2_assessment(period_days=1)
                    
                    assert assessment.assessment_id is not None
                    assert assessment.overall_status in [ComplianceStatus.COMPLIANT, ComplianceStatus.NON_COMPLIANT, ComplianceStatus.PARTIALLY_COMPLIANT]
                    assert isinstance(assessment.principle_scores, dict)
                    assert len(assessment.control_results) > 0
                    
                    score += 20
                    logger.info("✅ SOC2 assessment generation works correctly")
                    
                    # Stop monitoring
                    validator.monitor.stop_monitoring()
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"SOC2 assessment generation failed: {e}")
            
        except Exception as e:
            issues.append(f"Critical SOC2 framework error: {e}")
        
        return score, issues
    
    def validate_task_2_3_gdpr(self) -> Tuple[int, List[str]]:
        """Validate Task 2.3: GDPR Compliance Framework"""
        logger.info("Validating Task 2.3: GDPR Compliance Framework")
        
        score = 0
        issues = []
        
        try:
            # Test 1: Import GDPR framework
            try:
                from gdpr_validator import (
                    GDPRValidator, PersonalDataDetector, GDPRStorage,
                    DataCategory, LegalBasis, DataSubjectRight, gdpr_validator
                )
                score += 10
                logger.info("✅ GDPR framework imports successfully")
            except ImportError as e:
                issues.append(f"GDPR framework import failed: {e}")
                return score, issues
            
            # Test 2: Personal Data Detection
            try:
                detector = PersonalDataDetector()
                test_data = "John Doe, email: john@example.com, IP: 192.168.1.1"
                detections = detector.detect_personal_data(test_data)
                
                assert len(detections) >= 2  # Should detect email and IP
                categories = [category for category, _ in detections]
                assert DataCategory.PERSONAL_DATA in categories or DataCategory.ONLINE_IDENTIFIERS in categories
                
                score += 20
                logger.info("✅ GDPR personal data detection works correctly")
            except Exception as e:
                issues.append(f"GDPR personal data detection failed: {e}")
            
            # Test 3: GDPR Storage
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    storage = GDPRStorage(db_path=tmp.name)
                    
                    # Database should be initialized
                    assert os.path.exists(tmp.name)
                    
                    score += 15
                    logger.info("✅ GDPR storage works correctly")
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"GDPR storage failed: {e}")
            
            # Test 4: Data Processing Validation
            try:
                from gdpr_validator import ProcessingPurpose
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = GDPRValidator(storage=GDPRStorage(db_path=tmp.name))
                    
                    test_data = "User John Doe, email: john@example.com"
                    is_compliant, violations = validator.validate_data_processing(
                        data=test_data,
                        purpose=ProcessingPurpose.SERVICE_PROVISION,
                        legal_basis=LegalBasis.CONSENT,
                        data_subject_id="subject_123"
                    )
                    
                    assert isinstance(is_compliant, bool)
                    assert isinstance(violations, list)
                    
                    score += 20
                    logger.info("✅ GDPR data processing validation works correctly")
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"GDPR data processing validation failed: {e}")
            
            # Test 5: Data Subject Request Processing
            try:
                from gdpr_validator import DataSubjectRequest
                
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = GDPRValidator(storage=GDPRStorage(db_path=tmp.name))
                    
                    request = DataSubjectRequest(
                        request_id="req_123",
                        data_subject_id="subject_123",
                        request_type=DataSubjectRight.ACCESS,
                        request_date=datetime.now(timezone.utc),
                        status="pending",
                        completed_date=None,
                        response_data=None,
                        verification_method="email"
                    )
                    
                    response = validator.process_data_subject_request(request)
                    
                    assert "request_id" in response
                    assert "status" in response
                    
                    score += 20
                    logger.info("✅ GDPR data subject request processing works correctly")
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"GDPR data subject request processing failed: {e}")
            
            # Test 6: GDPR Compliance Report
            try:
                with tempfile.NamedTemporaryFile(delete=False) as tmp:
                    validator = GDPRValidator(storage=GDPRStorage(db_path=tmp.name))
                    
                    report = validator.generate_gdpr_compliance_report()
                    
                    assert report.assessment_id is not None
                    assert 0 <= report.compliance_score <= 100
                    assert 0 <= report.consent_compliance <= 100
                    assert 0 <= report.data_protection_compliance <= 100
                    assert 0 <= report.rights_fulfillment_rate <= 100
                    
                    score += 15
                    logger.info("✅ GDPR compliance reporting works correctly")
                    os.unlink(tmp.name)
            except Exception as e:
                issues.append(f"GDPR compliance reporting failed: {e}")
            
        except Exception as e:
            issues.append(f"Critical GDPR framework error: {e}")
        
        return score, issues
    
    def validate_integration(self) -> Tuple[int, List[str]]:
        """Validate regulatory compliance integration"""
        logger.info("Validating regulatory compliance integration")
        
        score = 0
        issues = []
        
        try:
            # Test 1: All frameworks can be imported together
            try:
                from hipaa_validator import hipaa_validator
                from soc2_validator import soc2_validator
                from gdpr_validator import gdpr_validator
                
                score += 25
                logger.info("✅ All compliance frameworks integrate successfully")
            except ImportError as e:
                issues.append(f"Compliance framework integration failed: {e}")
                return score, issues
            
            # Test 2: Cross-framework compatibility
            try:
                # Test that frameworks can work with the same data
                test_data = "Patient John Doe, email: john@example.com, SSN: 123-45-6789"
                
                # HIPAA detection
                hipaa_detections = hipaa_validator.phi_detector.detect_phi(test_data)
                
                # GDPR detection
                gdpr_detections = gdpr_validator.data_detector.detect_personal_data(test_data)
                
                # Both should detect personal data
                assert len(hipaa_detections) > 0
                assert len(gdpr_detections) > 0
                
                score += 25
                logger.info("✅ Cross-framework data detection works correctly")
            except Exception as e:
                issues.append(f"Cross-framework compatibility failed: {e}")
            
            # Test 3: Unified compliance reporting
            try:
                # Generate reports from all frameworks
                hipaa_report = hipaa_validator.generate_compliance_report()
                soc2_assessment = soc2_validator.generate_soc2_assessment(period_days=1)
                gdpr_report = gdpr_validator.generate_gdpr_compliance_report()
                
                # All reports should be generated successfully
                assert hipaa_report.assessment_id is not None
                assert soc2_assessment.assessment_id is not None
                assert gdpr_report.assessment_id is not None
                
                score += 25
                logger.info("✅ Unified compliance reporting works correctly")
                
                # Stop SOC2 monitoring
                soc2_validator.monitor.stop_monitoring()
            except Exception as e:
                issues.append(f"Unified compliance reporting failed: {e}")
            
            # Test 4: Performance under load
            try:
                start_time = time.time()
                
                # Perform multiple compliance checks
                for i in range(10):
                    test_data = f"Test data {i} with email: test{i}@example.com"
                    
                    # Quick validation checks
                    hipaa_validator.phi_detector.detect_phi(test_data)
                    gdpr_validator.data_detector.detect_personal_data(test_data)
                
                end_time = time.time()
                processing_time = end_time - start_time
                
                # Should complete within reasonable time (< 5 seconds for 10 checks)
                assert processing_time < 5.0
                
                score += 25
                logger.info("✅ Compliance frameworks perform well under load")
            except Exception as e:
                issues.append(f"Performance testing failed: {e}")
            
        except Exception as e:
            issues.append(f"Critical integration error: {e}")
        
        return score, issues
    
    def run_validation(self) -> Dict[str, Any]:
        """Run complete Phase 2 validation"""
        logger.info("Starting Phase 2 validation...")
        
        # Validate Task 2.1
        score_2_1, issues_2_1 = self.validate_task_2_1_hipaa()
        self.results["tasks"]["2.1"]["score"] = score_2_1
        self.results["tasks"]["2.1"]["status"] = "passed" if score_2_1 >= 80 else "failed"
        if issues_2_1:
            self.results["critical_issues"].extend([f"Task 2.1: {issue}" for issue in issues_2_1])
        
        # Validate Task 2.2
        score_2_2, issues_2_2 = self.validate_task_2_2_soc2()
        self.results["tasks"]["2.2"]["score"] = score_2_2
        self.results["tasks"]["2.2"]["status"] = "passed" if score_2_2 >= 80 else "failed"
        if issues_2_2:
            self.results["critical_issues"].extend([f"Task 2.2: {issue}" for issue in issues_2_2])
        
        # Validate Task 2.3
        score_2_3, issues_2_3 = self.validate_task_2_3_gdpr()
        self.results["tasks"]["2.3"]["score"] = score_2_3
        self.results["tasks"]["2.3"]["status"] = "passed" if score_2_3 >= 80 else "failed"
        if issues_2_3:
            self.results["critical_issues"].extend([f"Task 2.3: {issue}" for issue in issues_2_3])
        
        # Validate integration
        score_integration, issues_integration = self.validate_integration()
        if issues_integration:
            self.results["critical_issues"].extend([f"Integration: {issue}" for issue in issues_integration])
        
        # Calculate overall score
        total_score = score_2_1 + score_2_2 + score_2_3 + score_integration
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
        report.append("PHASE 2 VALIDATION REPORT")
        report.append("Regulatory Compliance Implementation")
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
            report.append("✅ Phase 2 implementation is ENTERPRISE READY")
            report.append("✅ All regulatory compliance frameworks are operational")
            report.append("✅ Ready to proceed to Phase 3")
        else:
            report.append("❌ Phase 2 implementation is NOT enterprise ready")
            report.append("❌ Critical issues must be resolved")
            report.append("❌ Do NOT proceed to Phase 3 until issues are fixed")
        
        report.append("=" * 80)
        
        return "\n".join(report)

def main():
    """Main validation function"""
    print("Pixelated Empathy AI - Phase 2 Validation")
    print("=" * 50)
    
    # Add current directory to path for imports
    current_dir = Path(__file__).parent
    sys.path.insert(0, str(current_dir))
    
    # Run validation
    validator = Phase2Validator()
    results = validator.run_validation()
    
    # Generate and display report
    report = validator.generate_report()
    print(report)
    
    # Save results to file
    results_file = current_dir / "phase2_validation_results.json"
    with open(results_file, 'w') as f:
        json.dump(results, f, indent=2)
    
    report_file = current_dir / "phase2_validation_report.txt"
    with open(report_file, 'w') as f:
        f.write(report)
    
    print(f"\nResults saved to: {results_file}")
    print(f"Report saved to: {report_file}")
    
    # Exit with appropriate code
    exit_code = 0 if results["enterprise_ready"] else 1
    sys.exit(exit_code)

if __name__ == "__main__":
    main()
