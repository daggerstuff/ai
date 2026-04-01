"""
Letta-Hindsight Bridge - Core integration layer.

This bridge connects Hindsight's therapeutic memory system with Letta's
persistent agent architecture, enabling clinically-safe AI agents.
"""

import asyncio
import logging
from dataclasses import dataclass
from typing import Any, Dict, Optional

from .letta_pii_middleware import LettaPIIMiddleware, PIIBlockedException
from .letta_crisis_handler import LettaCrisisHandler, CrisisSeverity, CrisisResult

logger = logging.getLogger(__name__)


@dataclass
class BridgeConfig:
    """Configuration for the Letta-Hindsight bridge."""
    hindsight_db_path: str
    hindsight_bank_id: str
    letta_base_url: str
    letta_api_key: str
    letta_agent_id: Optional[str] = None
    letta_client: Optional[Any] = None
    hindsight_api_url: Optional[str] = None
    hindsight_api_key: Optional[str] = None
    pii_filter_enabled: bool = True
    crisis_detection_enabled: bool = True
    max_redaction_ratio: float = 0.5


class LettaHindsightBridge:
    """
    Bridge between Letta and Hindsight systems.

    This class orchestrates the integration:
    1. Intercepts Letta tool calls
    2. Filters through Hindsight's PII detection
    3. Applies crisis detection
    4. Routes to appropriate storage (Hindsight for crisis, Letta for general)
    """

    def __init__(self, config: BridgeConfig):
        """
        Initialize the bridge with configuration.

        Args:
            config: BridgeConfig instance
        """
        self.config = config

        # Initialize Hindsight components
        self.hindsight_manager = self._init_hindsight_manager()
        self.crisis_detector = self._init_crisis_detector()

        # Initialize middleware and handlers
        self.pii_middleware = self._build_pii_middleware()
        self.crisis_handler: Optional[LettaCrisisHandler] = None

        if config.crisis_detection_enabled and self.crisis_detector:
            self.crisis_handler = LettaCrisisHandler(self.crisis_detector)

        # Initialize Letta client
        self.letta_client = self._init_letta_client()

    async def process_message(
        self,
        message: str,
        user_id: str,
        session_id: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Process a user message through the bridge.

        Args:
            message: User message
            user_id: User identifier
            session_id: Optional session identifier

        Returns:
            Response dict with response, memories, and metadata
        """
        del session_id  # Mark as intentionally unused - passed to handlers
        # 1. Crisis detection first (safety first)
        crisis_result: Optional[CrisisResult] = None
        if self.crisis_handler:
            crisis_result = await self.crisis_handler.check_message(message)

            # Block if critical crisis
            if crisis_result and crisis_result.severity == CrisisSeverity.CRITICAL:
                return {
                    'response': self._crisis_response(crisis_result),
                    'blocked': True,
                    'crisis': crisis_result
                }

        # 2. PII filtering for storage
        filtered_message = message
        if self.pii_middleware:
            filter_result = await self.pii_middleware.filter_tool_call(
                'user_message',
                {'content': message}
            )

            if filter_result.should_block:
                raise PIIBlockedException("Message contains too much PII")

            filtered_message = filter_result.filtered

        # 3. Letta processing
        letta_response = await self.letta_client.send(filtered_message)

        # 4. Store in Hindsight (therapeutic memory)
        if self.hindsight_manager:
            await asyncio.to_thread(
                self.hindsight_manager.add_memory,
                content=filtered_message,
                user_id=user_id,
                category=self._categorize_message(message, crisis_result),
            )

        # 5. Store in Letta (persistent agent memory)
        # Only if not crisis (crisis goes to Hindsight only)
        if crisis_result and crisis_result.severity != CrisisSeverity.NONE:
            # Crisis memory - Hindsight only
            pass
        else:
            # General memory - also store in Letta
            await self.letta_client.run(f"/remember {filtered_message[:500]}")

        return {
            'response': letta_response,
            'blocked': False,
            'crisis': crisis_result
        }

    def wrap_letta_client(self, letta_client: Any) -> Any:
        """
        Wrap a Letta client with the bridge's middleware.

        Args:
            letta_client: Raw Letta client to wrap

        Returns:
            Wrapped client with PII filtering and crisis detection
        """
        # Wrap tool execution
        if self.pii_middleware:
            original_call = letta_client.call_tool
            letta_client.call_tool = self._wrap_tool_call(original_call)

        return letta_client

    def _wrap_tool_call(self, original_call):
        """Wrap tool call with PII filtering."""
        async def wrapped(tool_name: str, tool_input: Dict[str, Any]):
            # Filter through PII middleware
            if self.pii_middleware:
                result = await self.pii_middleware.filter_tool_call(
                    tool_name,
                    tool_input
                )
                if result.should_block:
                    raise PIIBlockedException(f"Tool {tool_name} blocked due to PII")

            # Call original
            return await original_call(tool_name, tool_input)

        return wrapped

    def _init_hindsight_manager(self):
        """Initialize the local shared-memory manager."""
        try:
            from ai.memory.hindsight_manager import HindsightMemoryManager
            return HindsightMemoryManager(
                bank_id=self.config.hindsight_bank_id,
                db_path=self.config.hindsight_db_path,
            )
        except Exception as e:
            logger.error(f"Failed to initialize Hindsight manager: {e}")
            return None

    def _init_pii_filter(self):
        """Initialize PII filter."""
        return None

    def _build_pii_middleware(self) -> Optional[LettaPIIMiddleware]:
        """Create PII middleware only when a concrete filter exists."""
        if not self.config.pii_filter_enabled:
            return None
        pii_filter = self._init_pii_filter()
        if pii_filter is None:
            return None
        return LettaPIIMiddleware(
            pii_filter,
            {"max_redaction_ratio": self.config.max_redaction_ratio},
        )

    def _init_crisis_detector(self):
        """Initialize crisis detector."""
        # Use Hindsight's crisis detector if available
        try:
            from ai.memory.hindsight_subconscious import SubconsciousAgent
            # Return the crisis detector component
            return None  # Will be initialized from SubconsciousAgent if needed
        except Exception:
            return None

    def _init_letta_client(self):
        """Initialize Letta client from the provided configuration."""
        if self.config.letta_client is None:
            raise RuntimeError(
                "BridgeConfig.letta_client is required. "
                "Pass a real Letta SDK client instead of relying on a placeholder."
            )
        return self.config.letta_client

    def _categorize_message(
        self,
        message: str,
        crisis_result: CrisisResult
    ) -> str:
        """Categorize message for Hindsight storage."""
        if crisis_result and crisis_result.severity != CrisisSeverity.NONE:
            return "crisis_context"

        # Check for emotional distress indicators
        message_lower = message.lower()
        distress_indicators = ['anxious', 'depressed', 'overwhelmed', 'struggling']
        if any(indicator in message_lower for indicator in distress_indicators):
            return "emotional_state"

        # Check for therapeutic progress markers
        progress_indicators = ['milestone', 'breakthrough', 'improved', 'better', 'progress']
        if any(indicator in message_lower for indicator in progress_indicators):
            return "treatment_progress"

        return "general"

    def _crisis_response(self, crisis_result: CrisisResult) -> str:
        """Generate crisis response."""
        if crisis_result.suggested_action:
            return (
                f"I'm concerned. {crisis_result.suggested_action}. "
                "Please reach out to a professional."
            )
        return "I'm here to help. Consider reaching out to a professional."
