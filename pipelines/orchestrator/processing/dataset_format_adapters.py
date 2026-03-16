"""
Dataset Format Adapters for Mental Health Integration

Provides adapter pattern for converting various dataset formats
into standardized Conversation schema.
"""

import logging
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

from conversation_schema import Conversation, Message

logger = logging.getLogger(__name__)


class DatasetFormatAdapter(ABC):
    """
    Abstract base class for dataset format adapters.
    
    Each adapter knows how to convert a specific dataset format
    into the standardized Conversation schema.
    """

    @abstractmethod
    def adapt(self, raw_data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Convert raw dataset entry to Conversation.
        
        Args:
            raw_data: Raw data dict from dataset
            
        Returns:
            Conversation object or None if conversion fails
        """
        pass

    @abstractmethod
    def get_format_name(self) -> str:
        """Get the name of this format."""
        pass

    def _create_message(
        self,
        role: str,
        content: str,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> Message:
        """Helper to create a Message object."""
        return Message(
            role=role,
            content=content.strip() if content else "",
            metadata=metadata or {},
        )


class Psych8kAdapter(DatasetFormatAdapter):
    """
    Adapter for Psych8k dataset format.
    
    Psych8k contains professional therapy conversations with
    'Context' and 'Response' fields.
    """

    def get_format_name(self) -> str:
        return "psych8k"

    def adapt(self, raw_data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Convert Psych8k format to Conversation.
        
        Expected format:
        {
            "Context": "patient message",
            "Response": "therapist response",
            "metadata": {...}
        }
        """
        try:
            context = raw_data.get("Context", "")
            response = raw_data.get("Response", "")
            
            if not context or not response:
                return None
            
            messages = [
                self._create_message("user", context),
                self._create_message("assistant", response),
            ]
            
            conversation = Conversation(
                messages=messages,
                metadata={
                    "source": "psych8k",
                    "format": "professional_therapy",
                    **raw_data.get("metadata", {}),
                },
            )
            
            return conversation
            
        except Exception as e:
            logger.warning(f"Failed to adapt Psych8k entry: {e}")
            return None


class MentalHealthCounselingAdapter(DatasetFormatAdapter):
    """
    Adapter for mental_health_counseling_conversations format.
    
    Contains 'questionTitle', 'questionText', and 'answerText'.
    """

    def get_format_name(self) -> str:
        return "mental_health_counseling"

    def adapt(self, raw_data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Convert mental health counseling format to Conversation.
        
        Expected format:
        {
            "questionTitle": "title",
            "questionText": "question",
            "answerText": "answer"
        }
        """
        try:
            question_title = raw_data.get("questionTitle", "")
            question_text = raw_data.get("questionText", "")
            answer_text = raw_data.get("answerText", "")
            
            # Combine title and question if both present
            question = f"{question_title}\n\n{question_text}".strip() if question_title else question_text
            
            if not question or not answer_text:
                return None
            
            messages = [
                self._create_message("user", question),
                self._create_message("assistant", answer_text),
            ]
            
            conversation = Conversation(
                messages=messages,
                metadata={
                    "source": "mental_health_counseling",
                    "format": "qa_counseling",
                    "question_title": question_title,
                },
            )
            
            return conversation
            
        except Exception as e:
            logger.warning(f"Failed to adapt mental health counseling entry: {e}")
            return None


class SoulChatAdapter(DatasetFormatAdapter):
    """
    Adapter for SoulChat2.0 format.
    
    Advanced psychological counselor conversations.
    """

    def get_format_name(self) -> str:
        return "soulchat"

    def adapt(self, raw_data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Convert SoulChat format to Conversation.
        
        Expected format:
        {
            "instruction": "user message",
            "output": "counselor response",
            "input": optional context
        }
        """
        try:
            instruction = raw_data.get("instruction", "")
            output = raw_data.get("output", "")
            input_context = raw_data.get("input", "")
            
            # Combine instruction with input if present
            user_message = f"{instruction}\n{input_context}".strip() if input_context else instruction
            
            if not user_message or not output:
                return None
            
            messages = [
                self._create_message("user", user_message),
                self._create_message("assistant", output),
            ]
            
            conversation = Conversation(
                messages=messages,
                metadata={
                    "source": "soulchat",
                    "format": "psychological_counselor",
                },
            )
            
            return conversation
            
        except Exception as e:
            logger.warning(f"Failed to adapt SoulChat entry: {e}")
            return None


class CounselChatAdapter(DatasetFormatAdapter):
    """
    Adapter for counsel-chat format.
    
    Professional counseling Q&A archive.
    """

    def get_format_name(self) -> str:
        return "counsel_chat"

    def adapt(self, raw_data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Convert counsel-chat format to Conversation.
        
        Expected format:
        {
            "questionText": "client question",
            "answerText": "therapist answer",
            "topic": "category"
        }
        """
        try:
            question = raw_data.get("questionText", "")
            answer = raw_data.get("answerText", "")
            topic = raw_data.get("topic", "")
            
            if not question or not answer:
                return None
            
            messages = [
                self._create_message("user", question),
                self._create_message("assistant", answer),
            ]
            
            conversation = Conversation(
                messages=messages,
                metadata={
                    "source": "counsel_chat",
                    "format": "professional_counseling",
                    "topic": topic,
                },
            )
            
            return conversation
            
        except Exception as e:
            logger.warning(f"Failed to adapt counsel-chat entry: {e}")
            return None


class LLAMA3MentalAdapter(DatasetFormatAdapter):
    """
    Adapter for LLAMA3_Mental_Counseling_Data format.
    """

    def get_format_name(self) -> str:
        return "llama3_mental"

    def adapt(self, raw_data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Convert LLAMA3 mental counseling format to Conversation.
        
        Expected format:
        {
            "conversations": [
                {"from": "human", "value": "..."},
                {"from": "gpt", "value": "..."}
            ]
        }
        """
        try:
            conversations = raw_data.get("conversations", [])
            
            if not conversations:
                return None
            
            messages = []
            for conv in conversations:
                from_role = conv.get("from", "")
                value = conv.get("value", "")
                
                if not value:
                    continue
                
                # Map roles
                role = "user" if from_role == "human" else "assistant"
                messages.append(self._create_message(role, value))
            
            if len(messages) < 2:
                return None
            
            conversation = Conversation(
                messages=messages,
                metadata={
                    "source": "llama3_mental",
                    "format": "ai_counseling",
                },
            )
            
            return conversation
            
        except Exception as e:
            logger.warning(f"Failed to adapt LLAMA3 mental entry: {e}")
            return None


class TherapistSFTAdapter(DatasetFormatAdapter):
    """
    Adapter for therapist-sft-format.
    """

    def get_format_name(self) -> str:
        return "therapist_sft"

    def adapt(self, raw_data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Convert therapist SFT format to Conversation.
        
        Expected format:
        {
            "input": "client input",
            "output": "therapist output"
        }
        """
        try:
            input_text = raw_data.get("input", "")
            output_text = raw_data.get("output", "")
            
            if not input_text or not output_text:
                return None
            
            messages = [
                self._create_message("user", input_text),
                self._create_message("assistant", output_text),
            ]
            
            conversation = Conversation(
                messages=messages,
                metadata={
                    "source": "therapist_sft",
                    "format": "supervised_fine_tuning",
                },
            )
            
            return conversation
            
        except Exception as e:
            logger.warning(f"Failed to adapt therapist SFT entry: {e}")
            return None


class NeuroQAAdapter(DatasetFormatAdapter):
    """
    Adapter for neuro_qa_SFT_Trainer format.
    
    Neurology/psychology Q&A with 35K+ entries.
    """

    def get_format_name(self) -> str:
        return "neuro_qa"

    def adapt(self, raw_data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Convert neuro QA format to Conversation.
        
        Expected format:
        {
            "question": "...",
            "answer": "..."
        }
        """
        try:
            question = raw_data.get("question", "")
            answer = raw_data.get("answer", "")
            
            if not question or not answer:
                return None
            
            messages = [
                self._create_message("user", question),
                self._create_message("assistant", answer),
            ]
            
            conversation = Conversation(
                messages=messages,
                metadata={
                    "source": "neuro_qa",
                    "format": "neurology_psychology_qa",
                },
            )
            
            return conversation
            
        except Exception as e:
            logger.warning(f"Failed to adapt neuro QA entry: {e}")
            return None


class GenericChatMLAdapter(DatasetFormatAdapter):
    """
    Generic adapter for ChatML-style formats.
    
    Handles standard conversation formats with role/content structure.
    """

    def get_format_name(self) -> str:
        return "generic_chatml"

    def adapt(self, raw_data: Dict[str, Any]) -> Optional[Conversation]:
        """
        Convert generic ChatML format to Conversation.
        
        Expected format:
        {
            "messages": [
                {"role": "user", "content": "..."},
                {"role": "assistant", "content": "..."}
            ]
        }
        """
        try:
            messages_data = raw_data.get("messages", [])
            
            if not messages_data:
                return None
            
            messages = []
            for msg in messages_data:
                role = msg.get("role", "")
                content = msg.get("content", "")
                
                if not content or role not in ["user", "assistant", "system"]:
                    continue
                
                messages.append(self._create_message(role, content))
            
            if len(messages) < 2:
                return None
            
            conversation = Conversation(
                messages=messages,
                metadata={
                    "source": raw_data.get("source", "unknown"),
                    "format": "generic_chatml",
                },
            )
            
            return conversation
            
        except Exception as e:
            logger.warning(f"Failed to adapt generic ChatML entry: {e}")
            return None


class AdapterRegistry:
    """
    Registry for dataset format adapters.
    
    Manages adapter selection and fallback strategies.
    """

    def __init__(self):
        self.adapters: Dict[str, DatasetFormatAdapter] = {}
        self._register_default_adapters()

    def _register_default_adapters(self):
        """Register all default adapters."""
        adapters = [
            Psych8kAdapter(),
            MentalHealthCounselingAdapter(),
            SoulChatAdapter(),
            CounselChatAdapter(),
            LLAMA3MentalAdapter(),
            TherapistSFTAdapter(),
            NeuroQAAdapter(),
            GenericChatMLAdapter(),
        ]
        
        for adapter in adapters:
            self.register_adapter(adapter)

    def register_adapter(self, adapter: DatasetFormatAdapter):
        """Register a new adapter."""
        name = adapter.get_format_name()
        self.adapters[name] = adapter
        logger.info(f"Registered adapter: {name}")

    def get_adapter(self, format_name: str) -> Optional[DatasetFormatAdapter]:
        """Get adapter by format name."""
        return self.adapters.get(format_name)

    def adapt_with_fallback(
        self,
        raw_data: Dict[str, Any],
        primary_format: str,
        source_name: str = "unknown",
    ) -> Optional[Conversation]:
        """
        Adapt data using primary format, with generic fallback.
        
        Args:
            raw_data: Raw data to adapt
            primary_format: Primary format name to try
            source_name: Source dataset name for metadata
            
        Returns:
            Conversation object or None
        """
        # Try primary adapter
        adapter = self.get_adapter(primary_format)
        if adapter:
            result = adapter.adapt(raw_data)
            if result:
                return result
        
        # Try generic ChatML adapter as fallback
        generic_adapter = self.get_adapter("generic_chatml")
        if generic_adapter:
            # Add source to raw data for metadata
            raw_data["source"] = source_name
            result = generic_adapter.adapt(raw_data)
            if result:
                logger.debug(
                    f"Used generic fallback for {source_name} (primary: {primary_format})"
                )
                return result
        
        logger.warning(
            f"Failed to adapt data from {source_name} with format {primary_format}"
        )
        return None

    def list_adapters(self) -> List[str]:
        """Get list of registered adapter names."""
        return list(self.adapters.keys())
