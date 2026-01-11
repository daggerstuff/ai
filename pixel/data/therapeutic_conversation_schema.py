"""
Therapeutic Conversation Schema

Defines the data structures and validation for therapeutic conversations,
supporting Tiers 1.1, 1.2, and 1.3 functionality.
"""
from __future__ import annotations

import uuid
from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Any, Dict, List, Optional


class Role(Enum):
    USER = "user"
    ASSISTANT = "assistant" 
    SYSTEM = "system"


# Alias for backwards compatibility
ConversationRole = Role


class TherapeuticModality(Enum):
    """Therapeutic modalities/approaches."""
    CBT = "cbt"
    DBT = "dbt"
    PSYCHODYNAMIC = "psychodynamic"
    HUMANISTIC = "humanistic"
    EXISTENTIAL = "existential"
    SOLUTION_FOCUSED = "solution_focused"
    NARRATIVE = "narrative"
    GESTALT = "gestalt"
    EMDR = "emdr"
    SOMATIC = "somatic"


@dataclass
class ClinicalContext:
    """Clinical context for sessions."""
    session_type: str = "ongoing" 
    dsm5_categories: List[str] = field(default_factory=list)
    severity_level: Optional[str] = None
    risk_factors: List[str] = field(default_factory=list)
    
    # Enum-like constants for backwards compatibility
    INTAKE = "intake"
    ONGOING = "ongoing" 
    CRISIS = "crisis"
    TERMINATION = "termination"
    ASSESSMENT = "assessment"
    TREATMENT_PLANNING = "treatment_planning"


class ClinicalSeverity(Enum):
    """Clinical severity levels."""
    MINIMAL = "minimal"
    MILD = "mild"
    MODERATE = "moderate"
    SEVERE = "severe"
    CRITICAL = "critical"
    CRISIS = "crisis"  # Add CRISIS for backwards compatibility


class ComplexityLevel(Enum):
    """Conversation complexity levels."""
    BASIC = "basic"
    INTERMEDIATE = "intermediate"
    ADVANCED = "advanced"
    EXPERT = "expert"


class TherapeuticContext(Enum):
    GENERAL = "general"
    CBT = "cbt"
    DBT = "dbt"
    TRAUMA = "trauma"
    GRIEF = "grief"
    ANXIETY = "anxiety"
    DEPRESSION = "depression"
    CRISIS = "crisis"


@dataclass
class TherapeuticMessage:
    role: Role
    content: str
    timestamp: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    therapeutic_context: Optional[TherapeuticContext] = None
    emotional_tone: Optional[str] = None
    therapeutic_techniques: List[str] = field(default_factory=list)
    crisis_indicators: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        return {
            "role": self.role.value,
            "content": self.content,
            "timestamp": self.timestamp,
            "therapeutic_context": self.therapeutic_context.value if self.therapeutic_context else None,
            "emotional_tone": self.emotional_tone,
            "therapeutic_techniques": self.therapeutic_techniques,
            "crisis_indicators": self.crisis_indicators,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TherapeuticMessage:
        return cls(
            role=Role(data["role"]),
            content=data["content"],
            timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
            therapeutic_context=TherapeuticContext(data["therapeutic_context"]) if data.get("therapeutic_context") else None,
            emotional_tone=data.get("emotional_tone"),
            therapeutic_techniques=data.get("therapeutic_techniques", []),
            crisis_indicators=data.get("crisis_indicators", []),
            metadata=data.get("metadata", {}),
        )


@dataclass
class TherapeuticConversation:
    conversation_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    messages: List[TherapeuticMessage] = field(default_factory=list)
    session_context: TherapeuticContext = TherapeuticContext.GENERAL
    created_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    updated_at: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())
    client_id: Optional[str] = None
    session_goals: List[str] = field(default_factory=list)
    therapeutic_alliance_score: Optional[float] = None
    risk_level: str = "low"  # low, medium, high, crisis
    metadata: Dict[str, Any] = field(default_factory=dict)

    def add_message(self, role: Role, content: str, **kwargs) -> None:
        """Add a message to the conversation."""
        message = TherapeuticMessage(role=role, content=content, **kwargs)
        self.messages.append(message)
        self.updated_at = datetime.now(timezone.utc).isoformat()

    def get_latest_exchange(self) -> tuple[Optional[TherapeuticMessage], Optional[TherapeuticMessage]]:
        """Get the latest user message and assistant response pair."""
        user_msg = None
        assistant_msg = None
        
        # Look backwards for the latest pair
        for msg in reversed(self.messages):
            if msg.role == Role.ASSISTANT and assistant_msg is None:
                assistant_msg = msg
            elif msg.role == Role.USER and user_msg is None:
                user_msg = msg
                break
                
        return user_msg, assistant_msg

    def get_conversation_length(self) -> int:
        """Get number of messages in conversation."""
        return len(self.messages)

    def has_crisis_indicators(self) -> bool:
        """Check if any message contains crisis indicators."""
        for msg in self.messages:
            if msg.crisis_indicators:
                return True
        return False

    def to_dict(self) -> Dict[str, Any]:
        return {
            "conversation_id": self.conversation_id,
            "messages": [msg.to_dict() for msg in self.messages],
            "session_context": self.session_context.value,
            "created_at": self.created_at,
            "updated_at": self.updated_at,
            "client_id": self.client_id,
            "session_goals": self.session_goals,
            "therapeutic_alliance_score": self.therapeutic_alliance_score,
            "risk_level": self.risk_level,
            "metadata": self.metadata,
        }

    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> TherapeuticConversation:
        messages = [TherapeuticMessage.from_dict(msg_data) for msg_data in data.get("messages", [])]
        return cls(
            conversation_id=data.get("conversation_id", str(uuid.uuid4())),
            messages=messages,
            session_context=TherapeuticContext(data.get("session_context", "general")),
            created_at=data.get("created_at", datetime.now(timezone.utc).isoformat()),
            updated_at=data.get("updated_at", datetime.now(timezone.utc).isoformat()),
            client_id=data.get("client_id"),
            session_goals=data.get("session_goals", []),
            therapeutic_alliance_score=data.get("therapeutic_alliance_score"),
            risk_level=data.get("risk_level", "low"),
            metadata=data.get("metadata", {}),
        )


@dataclass
class ConversationMemory:
    """Memory structure for conversation context and history."""
    conversation_id: str
    short_term_memory: List[str] = field(default_factory=list)  # Recent key points
    long_term_memory: Dict[str, Any] = field(default_factory=dict)  # Persistent themes/patterns
    emotional_state_history: List[Dict[str, Any]] = field(default_factory=list)
    therapeutic_progress: Dict[str, Any] = field(default_factory=dict)
    last_updated: str = field(default_factory=lambda: datetime.now(timezone.utc).isoformat())

    def add_to_short_term(self, memory_item: str, max_items: int = 10) -> None:
        """Add item to short-term memory with size limit."""
        self.short_term_memory.append(memory_item)
        if len(self.short_term_memory) > max_items:
            self.short_term_memory.pop(0)
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def update_long_term(self, key: str, value: Any) -> None:
        """Update long-term memory with persistent information."""
        self.long_term_memory[key] = value
        self.last_updated = datetime.now(timezone.utc).isoformat()

    def get_context_summary(self) -> str:
        """Generate a context summary for therapeutic responses."""
        recent_items = self.short_term_memory[-3:] if len(self.short_term_memory) >= 3 else self.short_term_memory
        summary = "Recent context: " + "; ".join(recent_items) if recent_items else "No recent context"
        
        if self.long_term_memory:
            persistent_themes = [f"{k}: {v}" for k, v in self.long_term_memory.items()]
            summary += f"\nPersistent themes: {'; '.join(persistent_themes)}"
            
        return summary


@dataclass
class ConversationTurn:
    """A single turn in a therapeutic conversation (user + assistant pair)."""
    user_message: TherapeuticMessage
    assistant_message: Optional[TherapeuticMessage] = None
    turn_number: int = 0
    therapeutic_goal: Optional[str] = None
    intervention_used: Optional[str] = None
    effectiveness_score: Optional[float] = None

    def is_complete(self) -> bool:
        """Check if turn has both user and assistant messages."""
        return self.assistant_message is not None


@dataclass
class ConversationTemplate:
    """Template for generating structured therapeutic conversations."""
    name: str
    context: TherapeuticContext
    modality: TherapeuticModality
    session_goals: List[str] = field(default_factory=list)
    typical_flow: List[str] = field(default_factory=list)  # List of intervention types
    contraindications: List[str] = field(default_factory=list)


class ConversationQualityValidator:
    """Validates the quality and appropriateness of therapeutic conversations."""
    
    def __init__(self):
        self.min_response_length = 10
        self.max_response_length = 500
    
    def validate_message(self, message: TherapeuticMessage) -> Dict[str, Any]:
        """Validate a single therapeutic message."""
        issues = []
        
        if len(message.content) < self.min_response_length:
            issues.append("Response too short")
        if len(message.content) > self.max_response_length:
            issues.append("Response too long")
        
        # Check for therapeutic appropriateness
        if message.role == Role.ASSISTANT:
            if any(phrase in message.content.lower() for phrase in ["i don't know", "just get over it", "that's not my problem"]):
                issues.append("Non-therapeutic language detected")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "message_id": message.timestamp
        }
    
    def validate_conversation(self, conversation: TherapeuticConversation) -> Dict[str, Any]:
        """Validate an entire therapeutic conversation."""
        issues = []
        message_validations = []
        
        if len(conversation.messages) == 0:
            issues.append("Empty conversation")
        
        # Validate each message
        for msg in conversation.messages:
            validation = self.validate_message(msg)
            message_validations.append(validation)
            if not validation["valid"]:
                issues.extend(validation["issues"])
        
        # Check conversation flow
        user_count = sum(1 for msg in conversation.messages if msg.role == Role.USER)
        assistant_count = sum(1 for msg in conversation.messages if msg.role == Role.ASSISTANT)
        
        if abs(user_count - assistant_count) > 1:
            issues.append("Unbalanced conversation flow")
        
        return {
            "valid": len(issues) == 0,
            "issues": issues,
            "message_validations": message_validations,
            "conversation_id": conversation.conversation_id
        }