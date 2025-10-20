
import unittest
from .content_filtering import (
    ContentFilter, PiiDetectionResult, ContentValidationResult, SafetyGateResult,
    PiiType, ValidationSeverity, SafetyGateType
)

class TestContentFilter(unittest.TestCase):

    def setUp(self):
        self.content_filter = ContentFilter()

    # =============================================================================
    # PII Detection Tests
    # =============================================================================

    def test_detect_pii_email(self):
        """Test email PII detection."""
        text = "My email is test@example.com."
        results = self.content_filter.detect_pii(text)
        
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.pii_type, PiiType.EMAIL)
        self.assertEqual(result.value, "test@example.com")
        self.assertEqual(result.start, 12)
        self.assertEqual(result.end, 28)
        self.assertGreater(result.confidence, 0.9)

    def test_detect_pii_phone(self):
        """Test phone number PII detection."""
        text = "My phone number is 123-456-7890."
        results = self.content_filter.detect_pii(text)
        
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.pii_type, PiiType.PHONE)
        self.assertIn("123", result.value)
        self.assertGreater(result.confidence, 0.8)

    def test_detect_pii_ssn(self):
        """Test SSN PII detection."""
        text = "My SSN is 123-45-6789."
        results = self.content_filter.detect_pii(text)
        
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.pii_type, PiiType.SSN)
        self.assertEqual(result.value, "123-45-6789")

    def test_detect_pii_credit_card(self):
        """Test credit card PII detection."""
        text = "My card number is 1234-5678-9012-3456."
        results = self.content_filter.detect_pii(text)
        
        self.assertEqual(len(results), 1)
        result = results[0]
        self.assertEqual(result.pii_type, PiiType.CREDIT_CARD)
        self.assertEqual(result.value, "1234-5678-9012-3456")

    def test_detect_pii_multiple_types(self):
        """Test detection of multiple PII types in one text."""
        text = "Contact John at john@example.com or 555-123-4567. His SSN is 123-45-6789."
        results = self.content_filter.detect_pii(text)
        
        self.assertGreaterEqual(len(results), 3)
        pii_types = [r.pii_type for r in results]
        self.assertIn(PiiType.EMAIL, pii_types)
        self.assertIn(PiiType.PHONE, pii_types)
        self.assertIn(PiiType.SSN, pii_types)

    def test_detect_pii_name_context(self):
        """Test name detection in context."""
        text = "My name is John Smith and I need help."
        results = self.content_filter.detect_pii(text)
        
        name_results = [r for r in results if r.pii_type == PiiType.NAME]
        self.assertGreaterEqual(len(name_results), 1)

    # =============================================================================
    # PII Removal Tests
    # =============================================================================

    def test_remove_pii_basic(self):
        """Test basic PII removal."""
        text = "Contact me at test@example.com or 123-456-7890."
        result = self.content_filter.remove_pii(text)
        
        self.assertNotIn("test@example.com", result)
        self.assertNotIn("123-456-7890", result)
        self.assertIn("[EMAIL_REDACTED]", result)
        self.assertIn("[PHONE_REDACTED]", result)

    def test_remove_pii_preserve_crisis_keywords(self):
        """Test PII removal while preserving crisis keywords."""
        text = "I want to kill myself. My email is help@crisis.com."
        result = self.content_filter.remove_pii(text, preserve_crisis_keywords=True)
        
        # Crisis content should be preserved
        self.assertIn("kill myself", result)
        # Email should still be redacted
        self.assertIn("[EMAIL_REDACTED]", result)

    def test_remove_pii_crisis_overlap_protection(self):
        """Test protection against removing crisis-related content."""
        text = "I feel suicidal and my number is 988 (crisis line)."
        result = self.content_filter.remove_pii(text, preserve_crisis_keywords=True)
        
        # Should preserve crisis context even if it looks like PII
        self.assertIn("suicidal", result)

    # =============================================================================
    # Content Validation Tests
    # =============================================================================

    def test_validate_content_too_short(self):
        """Test validation of too-short content."""
        text = "Hi"
        results = self.content_filter.validate_content(text)
        
        min_length_errors = [r for r in results if r.validation_rule == "MIN_LENGTH"]
        self.assertEqual(len(min_length_errors), 1)
        self.assertFalse(min_length_errors[0].is_valid)
        self.assertEqual(min_length_errors[0].severity, ValidationSeverity.ERROR)

    def test_validate_content_too_long(self):
        """Test validation of too-long content."""
        text = "a" * 15000  # Exceeds max length
        results = self.content_filter.validate_content(text)
        
        max_length_errors = [r for r in results if r.validation_rule == "MAX_LENGTH"]
        self.assertEqual(len(max_length_errors), 1)
        self.assertFalse(max_length_errors[0].is_valid)

    def test_validate_content_valid_length(self):
        """Test validation of appropriately-sized content."""
        text = "This is a valid therapeutic message about feelings and support."
        results = self.content_filter.validate_content(text)
        
        length_errors = [r for r in results if r.validation_rule in ["MIN_LENGTH", "MAX_LENGTH"]]
        self.assertEqual(len(length_errors), 0)

    def test_validate_content_therapeutic_context(self):
        """Test therapeutic context validation."""
        # Text without therapeutic context
        text = "The weather is nice today and I like pizza."
        results = self.content_filter.validate_content(text)
        
        context_issues = [r for r in results if r.validation_rule == "THERAPEUTIC_CONTEXT"]
        self.assertEqual(len(context_issues), 1)
        self.assertFalse(context_issues[0].is_valid)

    def test_validate_content_with_therapeutic_context(self):
        """Test content with good therapeutic context."""
        text = "I'm feeling anxious about my therapy session and need support coping with stress."
        results = self.content_filter.validate_content(text)
        
        context_issues = [r for r in results if r.validation_rule == "THERAPEUTIC_CONTEXT"]
        self.assertEqual(len(context_issues), 0)

    def test_validate_content_coherence(self):
        """Test coherence validation."""
        # Incoherent text
        incoherent_text = "word random another sentence fragments incomplete"
        results = self.content_filter.validate_content(incoherent_text)
        
        coherence_issues = [r for r in results if r.validation_rule == "COHERENCE"]
        self.assertGreaterEqual(len(coherence_issues), 1)

    # =============================================================================
    # Safety Gate Tests
    # =============================================================================

    def test_safety_gates_profanity(self):
        """Test profanity safety gate."""
        text = "This is fucking ridiculous and makes me angry."
        results = self.content_filter.enforce_safety_gates(text)
        
        profanity_violations = [r for r in results if r.gate_type == SafetyGateType.PROFANITY]
        self.assertEqual(len(profanity_violations), 1)
        self.assertFalse(profanity_violations[0].passed)
        self.assertEqual(profanity_violations[0].severity, ValidationSeverity.WARNING)

    def test_safety_gates_hate_speech(self):
        """Test hate speech safety gate."""
        text = "I hate myself and want to kill myself."
        results = self.content_filter.enforce_safety_gates(text)
        
        hate_violations = [r for r in results if r.gate_type == SafetyGateType.HATE_SPEECH]
        self.assertEqual(len(hate_violations), 1)
        self.assertFalse(hate_violations[0].passed)
        self.assertEqual(hate_violations[0].severity, ValidationSeverity.CRITICAL)

    def test_safety_gates_medical_advice(self):
        """Test medical advice safety gate."""
        text = "You should stop taking your antidepressants and increase the dosage of your medication."
        results = self.content_filter.enforce_safety_gates(text)
        
        medical_violations = [r for r in results if r.gate_type == SafetyGateType.MEDICAL_ADVICE]
        self.assertEqual(len(medical_violations), 1)
        self.assertFalse(medical_violations[0].passed)
        self.assertEqual(medical_violations[0].severity, ValidationSeverity.ERROR)

    def test_safety_gates_personal_disclosure(self):
        """Test personal disclosure safety gate."""
        text = "I am a licensed therapist and work as a psychiatrist at the hospital."
        results = self.content_filter.enforce_safety_gates(text)
        
        disclosure_violations = [r for r in results if r.gate_type == SafetyGateType.PERSONAL_DISCLOSURE]
        self.assertEqual(len(disclosure_violations), 1)
        self.assertFalse(disclosure_violations[0].passed)

    def test_safety_gates_empty_content(self):
        """Test empty content safety gate."""
        text = ""
        results = self.content_filter.enforce_safety_gates(text)
        
        empty_violations = [r for r in results if r.gate_name == "EMPTY_CONTENT"]
        self.assertEqual(len(empty_violations), 1)
        self.assertFalse(empty_violations[0].passed)

    def test_safety_gates_clean_content(self):
        """Test safety gates with clean, appropriate content."""
        text = "I'm feeling better today after our therapy session. Thank you for your support."
        results = self.content_filter.enforce_safety_gates(text)
        
        violations = [r for r in results if not r.passed]
        self.assertEqual(len(violations), 0)

    # =============================================================================
    # Integration Tests
    # =============================================================================

    def test_comprehensive_filtering_pipeline(self):
        """Test the complete filtering pipeline."""
        text = """
        Hi, I'm John Smith (john@email.com, 555-123-4567). 
        I'm feeling really fucking depressed and want to kill myself. 
        Should I stop taking my medication? My SSN is 123-45-6789.
        """
        
        # Test PII detection
        pii_results = self.content_filter.detect_pii(text)
        self.assertGreater(len(pii_results), 3)  # Should detect multiple PII types
        
        # Test PII removal with crisis preservation
        filtered_text = self.content_filter.remove_pii(text, preserve_crisis_keywords=True)
        self.assertNotIn("john@email.com", filtered_text)
        self.assertNotIn("555-123-4567", filtered_text)
        self.assertNotIn("123-45-6789", filtered_text)
        self.assertIn("kill myself", filtered_text)  # Crisis content preserved
        
        # Test content validation
        validation_results = self.content_filter.validate_content(text)
        # Note: This text actually passes validation (has therapeutic context, good length, coherent)
        # So we don't expect validation errors for this specific text
        
        # Test safety gates
        safety_results = self.content_filter.enforce_safety_gates(text)
        safety_violations = [r for r in safety_results if not r.passed]
        self.assertGreaterEqual(len(safety_violations), 2)  # Should catch profanity, hate speech

    def test_crisis_integration_disabled(self):
        """Test behavior when crisis integration is disabled."""
        filter_no_crisis = ContentFilter(enable_crisis_integration=False)
        text = "I want to kill myself. Call me at 555-123-4567."
        
        filtered_text = filter_no_crisis.remove_pii(text, preserve_crisis_keywords=False)
        # Should remove phone number regardless of crisis context
        self.assertNotIn("555-123-4567", filtered_text)

    # =============================================================================
    # Helper Method Tests
    # =============================================================================

    def test_calculate_pii_confidence(self):
        """Test PII confidence calculation."""
        # High confidence email
        high_conf = self.content_filter._calculate_pii_confidence(PiiType.EMAIL, "test@example.com")
        self.assertGreater(high_conf, 0.9)
        
        # Lower confidence incomplete phone
        low_conf = self.content_filter._calculate_pii_confidence(PiiType.PHONE, "123-45")
        self.assertLess(low_conf, 0.8)

    def test_therapeutic_context_detection(self):
        """Test therapeutic context detection."""
        # Text with therapeutic context
        therapeutic_text = "I'm struggling with anxiety and depression, need therapy support."
        result = self.content_filter._check_therapeutic_context(therapeutic_text)
        self.assertTrue(result['has_context'])
        
        # Text without therapeutic context
        non_therapeutic_text = "The weather is nice and I like pizza."
        result = self.content_filter._check_therapeutic_context(non_therapeutic_text)
        self.assertFalse(result['has_context'])

    def test_coherence_assessment(self):
        """Test text coherence assessment."""
        # Coherent text
        coherent_text = "I am feeling anxious about my upcoming job interview. It's causing me stress."
        coherence_score = self.content_filter._assess_coherence(coherent_text)
        self.assertGreater(coherence_score, 0.6)
        
        # Incoherent text
        incoherent_text = "word random fragments incomplete sentence structure bad"
        coherence_score = self.content_filter._assess_coherence(incoherent_text)
        self.assertLess(coherence_score, 0.6)

if __name__ == '__main__':
    unittest.main()

