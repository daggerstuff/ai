"""
Inference safety filtering integration system.
Ensures all inference endpoints pass safety and content filters before returning content.
"""

import hashlib
import json
import logging
import time
from dataclasses import dataclass
from datetime import UTC, datetime
from enum import Enum
from typing import Any

# from ..dataset_pipeline.traceability_system import TraceabilityManager  # TODO: fix import
from ..monitoring.observability import observability

# Import our existing systems
from .enhanced_safety_filter import (
    EnhancedSafetyFilter,
    SafetyCategory,
    SafetyCheckResult,
    SafetyLevel,
)

# R1: Receipt emission
try:
    from ai.receipts.receipt import ReceiptEnvelope, Ledger
except ImportError:
    ReceiptEnvelope = None  # type: ignore
    Ledger = None  # type: ignore

logger = logging.getLogger(__name__)

# Module-level ledger for receipt chaining
_receipt_ledger = None
if Ledger is not None:
    _receipt_ledger = Ledger()


class SafetyFilterMode(Enum):
    """Modes for safety filtering"""

    BLOCK_ALL = "block_all"  # Block all unsafe content
    FILTER_AND_WARN = "filter_and_warn"  # Filter unsafe content and warn
    LOG_ONLY = "log_only"  # Only log unsafe content, don't block
    DISABLED = "disabled"  # No safety filtering


@dataclass
class InferenceSafetyResult:
    """Result of safety filtering for inference"""

    is_safe: bool
    content_filtered: bool
    original_content: str
    filtered_content: str
    safety_score: float
    flagged_categories: list[str]
    crisis_detected: bool
    crisis_info: dict[str, Any] | None = None
    filter_reason: str | None = None
    processing_time_ms: float = 0.0
    traceability_info: dict[str, Any] | None = None
    metadata: dict[str, Any] | None = None
    receipt_root_hash: str | None = None


class InferenceSafetyFilter:
    """Main safety filter integration for inference endpoints"""

    def __init__(
        self,
        safety_level: SafetyLevel = SafetyLevel.MODERATE,
        filter_mode: SafetyFilterMode = SafetyFilterMode.FILTER_AND_WARN,
    ):
        self.safety_filter = EnhancedSafetyFilter(safety_level)
        self.filter_mode = filter_mode
        self.logger = logging.getLogger(__name__)
        self.traceability_manager = None
        self.blocked_content_count = 0
        self.filtered_content_count = 0
        self.total_requests = 0
        self.last_filtered_content = None

    def set_traceability_manager(self, traceability_manager: "TraceabilityManager"):
        """Set the traceability manager for tracking filtered content"""
        self.traceability_manager = traceability_manager
        self.logger.info("Traceability manager set for inference safety filter")

    def set_safety_level(self, safety_level: SafetyLevel):
        """Update the safety level"""
        self.safety_filter.safety_level = safety_level
        self.logger.info(f"Safety level updated to: {safety_level.value}")

    def set_filter_mode(self, filter_mode: SafetyFilterMode):
        """Update the filter mode"""
        self.filter_mode = filter_mode
        self.logger.info(f"Filter mode updated to: {filter_mode.value}")

    def filter_inference_output(
        self,
        content: str,
        user_context: dict[str, Any] | None = None,
        request_metadata: dict[str, Any] | None = None,
        model_info: dict[str, Any] | None = None,
    ) -> InferenceSafetyResult:
        """Filter inference output through safety checks"""
        start_time = time.time()
        self.total_requests += 1

        # Validate input
        if not content or not isinstance(content, str):
            return InferenceSafetyResult(
                is_safe=True,
                content_filtered=False,
                original_content=content or "",
                filtered_content=content or "",
                safety_score=1.0,
                flagged_categories=[],
                crisis_detected=False,
                processing_time_ms=(time.time() - start_time) * 1000,
                metadata={"reason": "empty_or_invalid_content"},
            )

        # Perform safety check
        safety_result = self.safety_filter.check_output_safety(content, user_context, request_metadata)

        processing_time_ms = (time.time() - start_time) * 1000

        # Handle crisis detection specially
        crisis_info = None
        crisis_detected = False
        if SafetyCategory.CRISIS in safety_result.flagged_categories:
            crisis_detected = True
            # Extract crisis information
            crisis_detection_result = self.safety_filter._check_crisis_content(content)
            if crisis_detection_result.is_crisis:
                crisis_info = {
                    "type": crisis_detection_result.crisis_type,
                    "confidence": crisis_detection_result.confidence,
                    "urgency_level": crisis_detection_result.urgency_level,
                    "recommended_action": crisis_detection_result.recommended_action,
                    "resources": crisis_detection_result.resources or [],
                    "timestamp": crisis_detection_result.timestamp,
                }

        # Determine filtering action based on mode and safety result
        content_filtered = False
        filtered_content = content
        filter_reason = None

        if not safety_result.is_safe:
            if self.filter_mode == SafetyFilterMode.BLOCK_ALL:
                # Completely block unsafe content
                filtered_content = "[Response blocked for safety reasons - content requires human review]"
                content_filtered = True
                self.blocked_content_count += 1
                filter_reason = "blocked_unsafe_content"

            elif self.filter_mode == SafetyFilterMode.FILTER_AND_WARN:
                # Filter unsafe content and provide warning
                if safety_result.filtered_content:
                    filtered_content = safety_result.filtered_content
                    content_filtered = True
                    self.filtered_content_count += 1
                    filter_reason = "filtered_unsafe_content"
                else:
                    # Fallback to generic filtering
                    filtered_content = "[Response filtered for safety - content requires review]"
                    content_filtered = True
                    self.filtered_content_count += 1
                    filter_reason = "fallback_filtered_unsafe_content"

            elif self.filter_mode == SafetyFilterMode.LOG_ONLY:
                # Only log unsafe content, don't block
                filter_reason = "logged_unsafe_content"
                # Content remains unchanged

            elif self.filter_mode == SafetyFilterMode.DISABLED:
                # No filtering
                filter_reason = "safety_filtering_disabled"
                # Content remains unchanged

        # Record the filtering action
        if content_filtered or not safety_result.is_safe:
            self.last_filtered_content = {
                "original": content,
                "filtered": filtered_content,
                "safety_result": safety_result,
                "filter_reason": filter_reason,
                "timestamp": datetime.now(UTC).isoformat(),
            }

        # Create traceability record if traceability manager is available
        traceability_info = None
        if self.traceability_manager and model_info:
            try:
                traceability_info = self._create_traceability_record(
                    content, filtered_content, safety_result, model_info
                )
            except Exception as e:
                self.logger.warning(f"Failed to create traceability record: {e}")

        # Log the safety check result
        self.logger.info(
            f"Safety check completed - Safe: {safety_result.is_safe}, "
            f"Filtered: {content_filtered}, Time: {processing_time_ms:.2f}ms"
        )

        # Record metrics
        observability.metrics_collector.record_histogram("inference_safety_check_time_ms", processing_time_ms)
        observability.metrics_collector.record_metric(
            "inference_safety_score",
            safety_result.overall_score,
            "gauge",
        )
        if content_filtered:
            observability.metrics_collector.increment_counter("inference_content_filtered")
        if crisis_detected:
            observability.metrics_collector.increment_counter("inference_crisis_detected")

        # R1: Emit cryptographic receipt for this inference turn
        receipt_root_hash = None
        if ReceiptEnvelope is not None and _receipt_ledger is not None:
            try:
                model_fingerprint = model_info.get("model_fingerprint") if model_info else None
                if not model_fingerprint:
                    model_fingerprint = model_info.get("name", "unknown") if model_info else "unknown"

                prompt_hash = None
                if user_context and "prompt" in user_context:
                    prompt_hash = hashlib.sha256(user_context["prompt"].encode()).hexdigest()
                elif request_metadata and "prompt_hash" in request_metadata:
                    prompt_hash = request_metadata["prompt_hash"]

                output_hash = hashlib.sha256(content.encode()).hexdigest()

                fhe_ciphertext_hash = None
                if request_metadata and "fhe_ciphertext_hash" in request_metadata:
                    fhe_ciphertext_hash = request_metadata["fhe_ciphertext_hash"]

                bias_score = safety_result.overall_score

                parent_receipt_hash = None
                if request_metadata and "parent_receipt_hash" in request_metadata:
                    parent_receipt_hash = request_metadata["parent_receipt_hash"]
                elif _receipt_ledger._receipts:
                    parent_receipt_hash = _receipt_ledger._receipts[-1].receipt_hash

                receipt = ReceiptEnvelope.compute(
                    prev_hash=parent_receipt_hash or "0" * 64,
                    model_fingerprint=model_fingerprint,
                    prompt_hash=prompt_hash or "0" * 64,
                    output_hash=output_hash,
                    fhe_ciphertext_hash=fhe_ciphertext_hash or "0" * 64,
                )
                _receipt_ledger.append(receipt)
                receipt_root_hash = _receipt_ledger.root_hash()
                logger.debug(f"Receipt emitted: root_hash={receipt_root_hash[:16]}...")
            except Exception as e:
                logger.warning(f"Failed to emit receipt: {e}")

        return InferenceSafetyResult(
            is_safe=safety_result.is_safe,
            content_filtered=content_filtered,
            original_content=content,
            filtered_content=filtered_content,
            safety_score=safety_result.overall_score,
            flagged_categories=[cat.value for cat in safety_result.flagged_categories],
            crisis_detected=crisis_detected,
            crisis_info=crisis_info,
            filter_reason=filter_reason,
            processing_time_ms=processing_time_ms,
            traceability_info=traceability_info,
            metadata={
                "safety_check_timestamp": safety_result.timestamp,
                "safety_confidence": safety_result.confidence,
                "category_scores": {cat.value: score for cat, score in safety_result.category_scores.items()},
                "filter_mode": self.filter_mode.value,
                "safety_level": self.safety_filter.safety_level.value,
            },
            receipt_root_hash=receipt_root_hash,
        )

    def _create_traceability_record(
        self,
        original_content: str,
        filtered_content: str,
        safety_result: SafetyCheckResult,
        model_info: dict[str, Any],
    ) -> dict[str, Any]:
        """Create traceability record for filtered content"""
        if not self.traceability_manager:
            return None

        # Create traceability record for the safety filtering event
        record_id = f"safety_filter_{datetime.now(UTC).strftime('%Y%m%d_%H%M%S')}_{hashlib.md5(original_content.encode()).hexdigest()[:8]}"

        return {
            "record_id": record_id,
            "type": "safety_filtering",
            "original_content_hash": hashlib.sha256(original_content.encode()).hexdigest(),
            "filtered_content_hash": hashlib.sha256(filtered_content.encode()).hexdigest(),
            "is_safe": safety_result.is_safe,
            "content_filtered": original_content != filtered_content,
            "safety_score": safety_result.overall_score,
            "flagged_categories": [cat.value for cat in safety_result.flagged_categories],
            "confidence": safety_result.confidence,
            "explanation": safety_result.explanation,
            "model_info": model_info,
            "timestamp": datetime.now(UTC).isoformat(),
            "metadata": {
                "filter_mode": self.filter_mode.value,
                "safety_level": self.safety_filter.safety_level.value,
                "processing_time_ms": (time.time() - time.time()) * 1000,  # This would be filled in properly
            },
        }

        # In a real implementation, you'd store this in the traceability system
        # For now, we'll just return the data structure

    def get_filtering_statistics(self) -> dict[str, Any]:
        """Get statistics about content filtering"""
        return {
            "total_requests": self.total_requests,
            "blocked_content_count": self.blocked_content_count,
            "filtered_content_count": self.filtered_content_count,
            "safe_content_count": self.total_requests - self.blocked_content_count - self.filtered_content_count,
            "block_rate": self.blocked_content_count / self.total_requests if self.total_requests > 0 else 0,
            "filter_rate": self.filtered_content_count / self.total_requests if self.total_requests > 0 else 0,
            "safe_rate": (self.total_requests - self.blocked_content_count - self.filtered_content_count)
            / self.total_requests
            if self.total_requests > 0
            else 0,
            "last_filtered_content": self.last_filtered_content,
            "current_safety_level": self.safety_filter.safety_level.value,
            "current_filter_mode": self.filter_mode.value,
        }

    def reset_statistics(self):
        """Reset filtering statistics"""
        self.blocked_content_count = 0
        self.filtered_content_count = 0
        self.total_requests = 0
        self.last_filtered_content = None
        self.logger.info("Safety filtering statistics reset")

    def get_recent_filtered_content(self, _limit: int = 10) -> list[dict[str, Any]]:
        """Get recent filtered content for review"""
        # In a real implementation, this would query a database or log system
        # For now, we'll just return the last filtered content if available
        if self.last_filtered_content:
            return [self.last_filtered_content]
        return []


class SafetyAwareInferenceAPI:
    """Wrapper for inference APIs that ensures safety filtering"""

    def __init__(self, safety_filter: InferenceSafetyFilter):
        self.safety_filter = safety_filter
        self.logger = logging.getLogger(__name__)

    def safe_inference_call(
        self,
        inference_function: callable,
        *args,
        user_context: dict[str, Any] | None = None,
        request_metadata: dict[str, Any] | None = None,
        model_info: dict[str, Any] | None = None,
        **kwargs,
    ) -> tuple[bool, Any, InferenceSafetyResult]:
        """Call an inference function with automatic safety filtering"""
        start_time = time.time()

        try:
            # Call the inference function
            inference_result = inference_function(*args, **kwargs)

            # Extract content to filter
            content_to_filter = self._extract_content_from_result(inference_result)

            # Apply safety filtering
            safety_result = self.safety_filter.filter_inference_output(
                content_to_filter,
                user_context=user_context,
                request_metadata=request_metadata,
                model_info=model_info,
            )

            # Update the inference result with filtered content if needed
            updated_result = self._update_inference_result(
                inference_result,
                safety_result.filtered_content if safety_result.content_filtered else content_to_filter,
            )

            # Log the complete inference with safety check
            processing_time_ms = (time.time() - start_time) * 1000
            self.logger.info(
                f"Safe inference completed - Safe: {safety_result.is_safe}, "
                f"Filtered: {safety_result.content_filtered}, Time: {processing_time_ms:.2f}ms"
            )

            # Record metrics
            observability.metrics_collector.record_histogram("safe_inference_time_ms", processing_time_ms)
            if not safety_result.is_safe:
                observability.metrics_collector.increment_counter("unsafe_inference_responses")

            return safety_result.is_safe, updated_result, safety_result

        except Exception as e:
            error_time_ms = (time.time() - start_time) * 1000
            self.logger.error(f"Safe inference failed after {error_time_ms:.2f}ms: {e}")

            # Record error metrics
            observability.metrics_collector.increment_counter("inference_errors")
            observability.metrics_collector.record_histogram("inference_error_time_ms", error_time_ms)

            # Return error result
            safety_result = InferenceSafetyResult(
                is_safe=False,
                content_filtered=True,
                original_content=str(e),
                filtered_content="[Inference failed - error occurred]",
                safety_score=0.0,
                flagged_categories=["system_error"],
                crisis_detected=False,
                filter_reason="inference_error",
                processing_time_ms=error_time_ms,
                metadata={"error": str(e)},
            )

            return False, {"error": str(e)}, safety_result

    def _extract_content_from_result(self, result: Any) -> str:
        """Extract content from inference result"""
        if isinstance(result, str):
            return result
        if isinstance(result, dict):
            # Look for common content fields
            content_fields = ["content", "response", "text", "output", "generated_text"]
            for field in content_fields:
                if field in result and isinstance(result[field], str):
                    return result[field]
            # If no content field found, convert dict to string
            return json.dumps(result, default=str)
        if isinstance(result, list):
            # Join list elements
            return " ".join(str(item) for item in result)
        return str(result)

    def _update_inference_result(self, result: Any, filtered_content: str) -> Any:
        """Update inference result with filtered content"""
        if isinstance(result, str):
            return filtered_content
        if isinstance(result, dict):
            # Update content fields with filtered content
            updated_result = result.copy()
            content_fields = ["content", "response", "text", "output", "generated_text"]
            for field in content_fields:
                if field in updated_result:
                    updated_result[field] = filtered_content
                    break
            else:
                # If no content field found, add a content field
                updated_result["content"] = filtered_content
            return updated_result
        if isinstance(result, list):
            # Replace first element or append filtered content
            if result:
                updated_result = result.copy()
                updated_result[0] = filtered_content
                return updated_result
            return [filtered_content]
        return filtered_content


# Decorator for automatic safety filtering on inference endpoints
def safety_filtered_inference(
    filter_mode: SafetyFilterMode = SafetyFilterMode.FILTER_AND_WARN,
):
    """Decorator to automatically apply safety filtering to inference endpoints"""

    def decorator(func):
        # Create safety filter for this endpoint
        safety_filter = InferenceSafetyFilter(filter_mode=filter_mode)
        safety_aware_api = SafetyAwareInferenceAPI(safety_filter)

        def wrapper(*args, **kwargs):
            # Extract user context and request metadata from function arguments
            user_context = kwargs.get("user_context") or getattr(args[0] if args else None, "user_context", None)
            request_metadata = kwargs.get("request_metadata") or getattr(
                args[0] if args else None, "request_metadata", None
            )
            model_info = kwargs.get("model_info") or getattr(args[0] if args else None, "model_info", None)

            # Call the function with safety filtering
            is_safe, result, safety_result = safety_aware_api.safe_inference_call(
                func,
                *args,
                user_context=user_context,
                request_metadata=request_metadata,
                model_info=model_info,
                **kwargs,
            )

            # Add safety information to the result
            if isinstance(result, dict):
                result["safety_filtered"] = not is_safe
                result["safety_score"] = safety_result.safety_score
                result["flagged_categories"] = safety_result.flagged_categories
                result["crisis_detected"] = safety_result.crisis_detected
                if safety_result.crisis_info:
                    result["crisis_info"] = safety_result.crisis_info
                result["filter_reason"] = safety_result.filter_reason
                result["processing_time_ms"] = safety_result.processing_time_ms

            # Handle crisis situations
            if safety_result.crisis_detected and safety_result.crisis_info:
                # Log crisis detection
                logger.critical(f"CRISIS DETECTED: {safety_result.crisis_info}")

                # In a real system, you would:
                # 1. Notify crisis intervention team
                # 2. Provide crisis resources to user
                # 3. Escalate to appropriate authorities if needed
                # 4. Log the incident for review

                # For now, we'll just add crisis information to the response
                if isinstance(result, dict):
                    result["crisis_response"] = {
                        "detected": True,
                        "type": safety_result.crisis_info.get("type", "unknown"),
                        "urgency": safety_result.crisis_info.get("urgency_level", "unknown"),
                        "resources": safety_result.crisis_info.get("resources", []),
                        "recommended_action": safety_result.crisis_info.get(
                            "recommended_action", "seek professional help"
                        ),
                    }

            return result

        return wrapper

    return decorator


# Global safety filter instances
default_safety_filter = InferenceSafetyFilter(SafetyLevel.MODERATE, SafetyFilterMode.FILTER_AND_WARN)
strict_safety_filter = InferenceSafetyFilter(SafetyLevel.STRICT, SafetyFilterMode.BLOCK_ALL)
paranoid_safety_filter = InferenceSafetyFilter(SafetyLevel.PARANOID, SafetyFilterMode.BLOCK_ALL)

# Safety-aware API wrappers
default_safety_api = SafetyAwareInferenceAPI(default_safety_filter)
strict_safety_api = SafetyAwareInferenceAPI(strict_safety_filter)
paranoid_safety_api = SafetyAwareInferenceAPI(paranoid_safety_filter)


# Integration functions for the inference API
def integrate_safety_filtering_with_inference_api(_api_app):
    """Integrate safety filtering with the inference API"""
    # This function would be called during API initialization
    # to set up safety filtering for all endpoints

    logger.info("Safety filtering integrated with inference API")

    # Example of how to wrap API endpoints with safety filtering
    def create_safety_wrapper(safety_filter: InferenceSafetyFilter):
        safety_aware_api = SafetyAwareInferenceAPI(safety_filter)

        def safety_wrapper(endpoint_func):
            def wrapped(*args, **kwargs):
                # Apply safety filtering to the endpoint
                is_safe, result, safety_result = safety_aware_api.safe_inference_call(endpoint_func, *args, **kwargs)

                # Add safety metadata to response
                if hasattr(result, "__dict__"):
                    result.safety_filtered = not is_safe
                    result.safety_score = safety_result.safety_score
                    result.flagged_categories = safety_result.flagged_categories
                elif isinstance(result, dict):
                    result["safety_filtered"] = not is_safe
                    result["safety_score"] = safety_result.safety_score
                    result["flagged_categories"] = safety_result.flagged_categories
                    result["crisis_detected"] = safety_result.crisis_detected
                    if safety_result.crisis_info:
                        result["crisis_info"] = safety_result.crisis_info
                    result["filter_reason"] = safety_result.filter_reason
                    result["processing_time_ms"] = safety_result.processing_time_ms

                return result

            return wrapped

        return safety_wrapper

    return create_safety_wrapper(default_safety_filter)


# Example usage and testing
def test_inference_safety_filtering():
    """Test the inference safety filtering system"""
    logger.info("Testing Inference Safety Filtering System...")

    # Test basic safety filtering

    # Create safety filter
    safety_filter = InferenceSafetyFilter(SafetyLevel.MODERATE, SafetyFilterMode.FILTER_AND_WARN)

    # Test safe content
    safe_content = "Hello, how can I help you today? I'm here to listen and support you."
    result = safety_filter.filter_inference_output(safe_content)

    # Test unsafe content
    unsafe_content = "I'm thinking about killing myself. Life isn't worth living anymore."
    result = safety_filter.filter_inference_output(unsafe_content)

    if result.crisis_info:
        pass

    # Test filtering modes

    # Block all mode
    block_filter = InferenceSafetyFilter(SafetyLevel.MODERATE, SafetyFilterMode.BLOCK_ALL)
    result = block_filter.filter_inference_output(unsafe_content)

    # Log only mode
    log_filter = InferenceSafetyFilter(SafetyLevel.MODERATE, SafetyFilterMode.LOG_ONLY)
    result = log_filter.filter_inference_output(unsafe_content)

    # Disabled mode
    disabled_filter = InferenceSafetyFilter(SafetyLevel.MODERATE, SafetyFilterMode.DISABLED)
    result = disabled_filter.filter_inference_output(unsafe_content)

    # Test with different safety levels

    # Strict level
    strict_filter = InferenceSafetyFilter(SafetyLevel.STRICT, SafetyFilterMode.FILTER_AND_WARN)
    result = strict_filter.filter_inference_output(safe_content)

    # Paranoid level
    paranoid_filter = InferenceSafetyFilter(SafetyLevel.PARANOID, SafetyFilterMode.FILTER_AND_WARN)
    result = paranoid_filter.filter_inference_output(safe_content)

    # Test statistics
    safety_filter.get_filtering_statistics()

    # Test safety-aware API wrapper

    def mock_inference_function(prompt: str) -> str:
        """Mock inference function"""
        if "suicide" in prompt.lower():
            return "That sounds very serious. Have you been thinking about hurting yourself?"
        return "I understand how you're feeling. Can you tell me more about what's been going on?"

    safety_aware_api = SafetyAwareInferenceAPI(safety_filter)

    # Test safe inference
    is_safe, result, safety_result = safety_aware_api.safe_inference_call(
        mock_inference_function, "I'm feeling anxious about my therapy session."
    )

    # Test unsafe inference
    _is_safe, result, _safety_result = safety_aware_api.safe_inference_call(
        mock_inference_function, "I've been thinking about suicide lately."
    )

    # Test decorated function

    @safety_filtered_inference(SafetyFilterMode.FILTER_AND_WARN)
    def decorated_inference_function(prompt: str) -> str:
        """Decorated inference function"""
        if "hurt myself" in prompt.lower():
            return "I'm really concerned about what you're going through. Those thoughts can be dangerous."
        return "Thank you for sharing that with me. How can I support you today?"

    result = decorated_inference_function("I've been having thoughts about hurting myself.")

    if isinstance(result, dict):
        pass


if __name__ == "__main__":
    test_inference_safety_filtering()
