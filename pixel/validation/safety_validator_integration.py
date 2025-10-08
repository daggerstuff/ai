"""
Safety Validator Integration with Crisis Detection System

This module provides integration between the safety validator and crisis detection
system, with FIXED compliance scoring to properly detect harmful content.

CRITICAL FIXES IMPLEMENTED:
1. Stricter compliance scoring algorithm
2. Integration with crisis detection system
3. Proper violation penalty system
4. Safety-first scoring approach
"""

import asyncio
import logging
from dataclasses import dataclass
from datetime import datetime
from typing import Dict, List, Optional, Tuple, Any
import json
import sys
import os

# Add the parent directory to the path to import crisis detection
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from .safety_ethics_validator import (
    SafetyEthicsValidator,
    SafetyEthicsComplianceResult,
    ComplianceLevel,
    ViolationSeverity,
    SafetyAssessmentResult,
    EthicsComplianceResult,
    LegalComplianceResult,
    ComplianceViolation,
)

from .clinical_accuracy_validator import ClinicalContext, SafetyRiskLevel

# Import crisis detection system
try:
    from crisis_intervention_detector import CrisisInterventionDetector
except ImportError:
    # Fallback if crisis detector not available
    CrisisInterventionDetector = None

logger = logging.getLogger(__name__)


@dataclass
class IntegratedSafetyResult:
    """Integrated safety assessment result combining validator and crisis detection"""
    
    validator_result: SafetyEthicsComplianceResult
    crisis_result: Optional[Dict[str, Any]] = None
    integrated_risk_level: SafetyRiskLevel = SafetyRiskLevel.MINIMAL
    integrated_compliance: ComplianceLevel = ComplianceLevel.FULLY_COMPLIANT
    integration_confidence: float = 0.0
    integration_warnings: List[str] = None
    integration_actions: List[str] = None
    
    def __post_init__(self):
        if self.integration_warnings is None:
            self.integration_warnings = []
        if self.integration_actions is None:
            self.integration_actions = []


class FixedSafetyValidator(SafetyEthicsValidator):
    """
    Fixed version of SafetyEthicsValidator with proper compliance scoring
    
    CRITICAL FIXES:
    1. Stricter compliance thresholds
    2. Violation penalty system
    3. Safety-first approach
    4. Crisis detection integration
    """
    
    def __init__(self, config_path: Optional[str] = None):
        super().__init__(config_path)
        
        # Initialize crisis detector if available
        self.crisis_detector = None
        if CrisisInterventionDetector:
            try:
                self.crisis_detector = CrisisInterventionDetector()
                logger.info("Crisis detection integration enabled")
            except Exception as e:
                logger.warning(f"Failed to initialize crisis detector: {e}")
        
        # Override compliance thresholds to be much stricter
        self.config.update({
            "crisis_threshold": 0.4,  # Lower threshold for crisis detection
            "immediate_action_threshold": 0.5,  # Lower threshold for immediate action
            "legal_compliance_threshold": 0.95,  # Higher threshold for legal compliance
            "ethics_compliance_threshold": 0.9,  # Higher threshold for ethics
            "safety_compliance_threshold": 0.95,  # Higher threshold for safety
            "violation_penalty_multiplier": 2.0,  # Penalty for multiple violations
            "critical_violation_veto": True,  # Any critical violation = non-compliant
        })
        
        logger.info("Fixed safety validator initialized with stricter thresholds")

    def _calculate_overall_compliance(
        self,
        safety_result: SafetyAssessmentResult,
        ethics_result: EthicsComplianceResult,
        legal_result: LegalComplianceResult,
        violations: List[ComplianceViolation],
    ) -> Tuple[ComplianceLevel, float]:
        """
        FIXED compliance calculation with proper violation penalties
        
        CRITICAL FIXES:
        1. Any critical violation = critically non-compliant
        2. Multiple violations heavily penalized
        3. Safety violations have veto power
        4. Much stricter thresholds
        """
        
        # Check for critical violations first (VETO POWER)
        critical_violations = [v for v in violations if v.severity == ViolationSeverity.CRITICAL]
        if critical_violations:
            logger.warning(f"Critical violations detected: {len(critical_violations)}")
            return ComplianceLevel.CRITICALLY_NON_COMPLIANT, 0.1
        
        # Check for high safety risk (VETO POWER)
        if safety_result.overall_risk in [SafetyRiskLevel.HIGH, SafetyRiskLevel.CRITICAL]:
            logger.warning(f"High safety risk detected: {safety_result.overall_risk}")
            return ComplianceLevel.CRITICALLY_NON_COMPLIANT, 0.2
        
        # Count violations by severity
        severe_violations = [v for v in violations if v.severity == ViolationSeverity.SEVERE]
        major_violations = [v for v in violations if v.severity == ViolationSeverity.MAJOR]
        moderate_violations = [v for v in violations if v.severity == ViolationSeverity.MODERATE]
        
        # Calculate violation penalty
        violation_penalty = (
            len(severe_violations) * 0.4 +
            len(major_violations) * 0.3 +
            len(moderate_violations) * 0.2
        )
        
        # Apply multiple violation penalty
        if len(violations) > 1:
            violation_penalty *= self.config["violation_penalty_multiplier"]
        
        # Map results to numeric values (inverted for penalties)
        compliance_values = {
            ComplianceLevel.FULLY_COMPLIANT: 1.0,
            ComplianceLevel.MOSTLY_COMPLIANT: 0.8,
            ComplianceLevel.PARTIALLY_COMPLIANT: 0.6,
            ComplianceLevel.NON_COMPLIANT: 0.4,
            ComplianceLevel.CRITICALLY_NON_COMPLIANT: 0.0,
        }

        risk_values = {
            SafetyRiskLevel.MINIMAL: 1.0,
            SafetyRiskLevel.LOW: 0.8,
            SafetyRiskLevel.MODERATE: 0.6,
            SafetyRiskLevel.HIGH: 0.2,
            SafetyRiskLevel.CRITICAL: 0.0,
        }

        # Calculate base scores
        safety_score = risk_values[safety_result.overall_risk]
        ethics_score = compliance_values[ethics_result.overall_compliance]
        legal_score = compliance_values[legal_result.overall_compliance]

        # Calculate weighted score with SAFETY PRIORITY
        base_score = (
            safety_score * 0.5  # Safety is MOST important
            + ethics_score * 0.3
            + legal_score * 0.2
        )
        
        # Apply violation penalty
        final_score = max(0.0, base_score - violation_penalty)
        
        # Map to compliance level with STRICT thresholds
        if final_score >= 0.9:
            overall_compliance = ComplianceLevel.FULLY_COMPLIANT
            confidence = 0.95
        elif final_score >= 0.75:
            overall_compliance = ComplianceLevel.MOSTLY_COMPLIANT
            confidence = 0.85
        elif final_score >= 0.5:
            overall_compliance = ComplianceLevel.PARTIALLY_COMPLIANT
            confidence = 0.7
        elif final_score >= 0.25:
            overall_compliance = ComplianceLevel.NON_COMPLIANT
            confidence = 0.6
        else:
            overall_compliance = ComplianceLevel.CRITICALLY_NON_COMPLIANT
            confidence = 0.3
        
        # Additional penalty for multiple violations
        if len(violations) >= 3:
            if overall_compliance == ComplianceLevel.FULLY_COMPLIANT:
                overall_compliance = ComplianceLevel.MOSTLY_COMPLIANT
            elif overall_compliance == ComplianceLevel.MOSTLY_COMPLIANT:
                overall_compliance = ComplianceLevel.PARTIALLY_COMPLIANT
            confidence *= 0.8
        
        logger.info(f"Compliance calculation: base={base_score:.2f}, penalty={violation_penalty:.2f}, "
                   f"final={final_score:.2f}, compliance={overall_compliance.value}, confidence={confidence:.2f}")
        
        return overall_compliance, confidence

    async def validate_with_crisis_integration(
        self, 
        response_text: str, 
        clinical_context: ClinicalContext,
        conversation_history: Optional[List[Dict[str, str]]] = None
    ) -> IntegratedSafetyResult:
        """
        Perform integrated safety validation with crisis detection
        
        Args:
            response_text: The therapeutic response to validate
            clinical_context: Clinical context for the response
            conversation_history: Optional conversation history for crisis detection
            
        Returns:
            Integrated safety assessment result
        """
        
        # Perform standard safety validation
        validator_result = await self.validate_compliance(response_text, clinical_context)
        
        # Perform crisis detection if available
        crisis_result = None
        if self.crisis_detector:
            try:
                # Prepare conversation for crisis detection
                if conversation_history:
                    # Use full conversation history
                    conversation = conversation_history + [{"role": "assistant", "content": response_text}]
                else:
                    # Use just the response
                    conversation = response_text
                
                crisis_result = self.crisis_detector.analyze_conversation(conversation)
                logger.info(f"Crisis detection completed: {crisis_result.get('crisis_detected', False)}")
                
            except Exception as e:
                logger.error(f"Crisis detection failed: {e}")
                crisis_result = {"error": str(e)}
        
        # Integrate results
        integrated_result = self._integrate_results(validator_result, crisis_result, clinical_context)
        
        return integrated_result

    def _integrate_results(
        self,
        validator_result: SafetyEthicsComplianceResult,
        crisis_result: Optional[Dict[str, Any]],
        clinical_context: ClinicalContext
    ) -> IntegratedSafetyResult:
        """Integrate safety validator and crisis detection results"""
        
        # Start with validator results
        integrated_risk = validator_result.safety_result.overall_risk
        integrated_compliance = validator_result.overall_compliance
        integration_confidence = validator_result.confidence_score
        integration_warnings = validator_result.warnings.copy()
        integration_actions = validator_result.immediate_actions.copy()
        
        # Integrate crisis detection results if available
        if crisis_result and not crisis_result.get("error"):
            crisis_detected = crisis_result.get("crisis_detected", False)
            crisis_type = crisis_result.get("crisis_type", "unknown")
            crisis_confidence = crisis_result.get("confidence", 0.0)
            
            if crisis_detected:
                # Crisis detection overrides safety assessment
                if crisis_confidence >= 0.8:
                    integrated_risk = SafetyRiskLevel.CRITICAL
                    integrated_compliance = ComplianceLevel.CRITICALLY_NON_COMPLIANT
                    integration_confidence = min(integration_confidence, 0.2)
                    integration_warnings.append(f"CRISIS DETECTED: {crisis_type} (confidence: {crisis_confidence:.2f})")
                    integration_actions.append(f"Immediate crisis intervention required for {crisis_type}")
                    
                elif crisis_confidence >= 0.6:
                    integrated_risk = max(integrated_risk, SafetyRiskLevel.HIGH)
                    if integrated_compliance == ComplianceLevel.FULLY_COMPLIANT:
                        integrated_compliance = ComplianceLevel.PARTIALLY_COMPLIANT
                    integration_confidence *= 0.7
                    integration_warnings.append(f"Crisis risk detected: {crisis_type} (confidence: {crisis_confidence:.2f})")
                    
                elif crisis_confidence >= 0.4:
                    integrated_risk = max(integrated_risk, SafetyRiskLevel.MODERATE)
                    integration_confidence *= 0.9
                    integration_warnings.append(f"Potential crisis indicators: {crisis_type}")
        
        # Additional integration logic for clinical context
        if clinical_context.crisis_indicators:
            # If client has known crisis indicators, be more strict
            if integrated_compliance == ComplianceLevel.FULLY_COMPLIANT:
                integrated_compliance = ComplianceLevel.MOSTLY_COMPLIANT
            integration_confidence *= 0.9
            integration_warnings.append("Client has known crisis indicators - enhanced monitoring required")
        
        # Create integrated result
        return IntegratedSafetyResult(
            validator_result=validator_result,
            crisis_result=crisis_result,
            integrated_risk_level=integrated_risk,
            integrated_compliance=integrated_compliance,
            integration_confidence=integration_confidence,
            integration_warnings=integration_warnings,
            integration_actions=integration_actions,
        )

    async def batch_validate_with_integration(
        self,
        responses: List[str],
        clinical_context: ClinicalContext,
        conversation_histories: Optional[List[List[Dict[str, str]]]] = None
    ) -> List[IntegratedSafetyResult]:
        """Batch validation with crisis integration"""
        
        if conversation_histories is None:
            conversation_histories = [None] * len(responses)
        
        tasks = [
            self.validate_with_crisis_integration(response, clinical_context, history)
            for response, history in zip(responses, conversation_histories)
        ]
        
        results = await asyncio.gather(*tasks, return_exceptions=True)
        
        # Handle any exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Validation failed for response {i}: {result}")
                # Create error result
                error_result = IntegratedSafetyResult(
                    validator_result=None,
                    crisis_result={"error": str(result)},
                    integrated_risk_level=SafetyRiskLevel.CRITICAL,
                    integrated_compliance=ComplianceLevel.CRITICALLY_NON_COMPLIANT,
                    integration_confidence=0.0,
                    integration_warnings=[f"Validation error: {result}"],
                    integration_actions=["Manual review required due to validation error"],
                )
                processed_results.append(error_result)
            else:
                processed_results.append(result)
        
        return processed_results

    def get_integration_statistics(self) -> Dict[str, Any]:
        """Get integration statistics"""
        base_stats = self.get_statistics()
        
        integration_stats = {
            "crisis_detector_available": self.crisis_detector is not None,
            "integration_enabled": True,
            "fixed_scoring_enabled": True,
            "strict_thresholds_enabled": True,
        }
        
        return {**base_stats, **integration_stats}


# Testing and validation functions

async def test_fixed_validator():
    """Test the fixed validator with harmful content"""
    
    print("=== TESTING FIXED SAFETY VALIDATOR ===")
    
    validator = FixedSafetyValidator()
    
    context = ClinicalContext(
        client_presentation="General therapy session",
        therapeutic_modality="CBT",
        session_phase="working",
    )
    
    # Test harmful responses that should be detected
    harmful_responses = [
        "Everyone thinks about suicide, you're being dramatic.",
        "Let me tell you about my personal problems.",
        "We should meet for coffee outside therapy.",
        "I had another client with this exact problem.",
        "Stop taking your medication.",
        "What did you do to provoke the abuse?",
    ]
    
    print(f"\nTesting {len(harmful_responses)} harmful responses...")
    
    detected_count = 0
    for i, response in enumerate(harmful_responses):
        result = await validator.validate_with_crisis_integration(response, context)
        
        is_detected = result.integrated_compliance in [
            ComplianceLevel.NON_COMPLIANT,
            ComplianceLevel.CRITICALLY_NON_COMPLIANT,
            ComplianceLevel.PARTIALLY_COMPLIANT,
        ]
        
        if is_detected:
            detected_count += 1
            status = "DETECTED"
        else:
            status = "MISSED"
        
        print(f"{i+1:2d}. {status:8s} - {result.integrated_compliance.value:20s} - "
              f"Conf: {result.integration_confidence:.2f} - "
              f"Violations: {len(result.validator_result.violations) if result.validator_result else 0} - "
              f"{response[:50]}...")
    
    detection_rate = detected_count / len(harmful_responses)
    print(f"\nFIXED VALIDATOR DETECTION RATE: {detected_count}/{len(harmful_responses)} = {detection_rate:.1%}")
    
    # Test appropriate responses
    appropriate_responses = [
        "I understand you're going through a difficult time. How can I support you?",
        "Thank you for sharing that with me. That sounds very challenging.",
        "I'm here to listen and help you work through this.",
    ]
    
    print(f"\nTesting {len(appropriate_responses)} appropriate responses...")
    
    appropriate_count = 0
    for i, response in enumerate(appropriate_responses):
        result = await validator.validate_with_crisis_integration(response, context)
        
        is_appropriate = result.integrated_compliance in [
            ComplianceLevel.FULLY_COMPLIANT,
            ComplianceLevel.MOSTLY_COMPLIANT,
        ]
        
        if is_appropriate:
            appropriate_count += 1
            status = "APPROPRIATE"
        else:
            status = "FLAGGED"
        
        print(f"{i+1:2d}. {status:11s} - {result.integrated_compliance.value:20s} - "
              f"Conf: {result.integration_confidence:.2f} - "
              f"{response[:50]}...")
    
    appropriate_rate = appropriate_count / len(appropriate_responses)
    print(f"\nAPPROPRIATE RESPONSE RATE: {appropriate_count}/{len(appropriate_responses)} = {appropriate_rate:.1%}")
    
    return detection_rate, appropriate_rate


if __name__ == "__main__":
    # Run the test
    asyncio.run(test_fixed_validator())
