"""
Human-in-the-loop labeling interface for edge cases and low-confidence items.
Provides tools for human annotators to review, validate, and correct automated labels.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Callable
import uuid
import json
from enum import Enum
import logging
from .label_taxonomy import (
    LabelBundle, TherapeuticResponseLabel, CrisisLabel, TherapyModalityLabel,
    MentalHealthConditionLabel, DemographicLabel, LabelProvenanceType,
    TherapeuticResponseType, CrisisLevelType, TherapyModalityType, 
    MentalHealthConditionType, DemographicType
)
from .conversation_schema import Conversation
from .automated_labeler import AutomatedLabeler, ConfidenceScore

logger = logging.getLogger(__name__)


class AnnotationAction(Enum):
    """Types of annotation actions"""
    CONFIRM = "confirm"
    CORRECT = "correct"
    REJECT = "reject"
    ADD = "add"
    SKIP = "skip"


class ReviewPriority(Enum):
    """Priority levels for human review"""
    HIGH = "high"  # Low confidence automated labels
    MEDIUM = "medium"  # Medium confidence or critical labels
    LOW = "low"  # High confidence but important for quality assurance


@dataclass
class AnnotationTask:
    """A single annotation task for a human annotator"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    conversation_id: str
    conversation: Conversation
    original_labels: LabelBundle
    suggested_labels: LabelBundle
    review_priority: ReviewPriority
    assigned_annotator: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    due_date: Optional[str] = None
    status: str = "pending"  # pending, in_progress, completed, rejected
    notes: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AnnotationResult:
    """Result from a human annotator's work on a task"""
    task_id: str
    annotator_id: str
    action: AnnotationAction
    applied_labels: LabelBundle
    confidence_adjustment: Optional[float] = None  # Adjustment to confidence after human review
    feedback: Optional[str] = None
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    metadata: Dict[str, Any] = field(default_factory=dict)


class AnnotationQueue:
    """Queue for managing human annotation tasks"""
    
    def __init__(self):
        self.tasks: Dict[str, AnnotationTask] = {}
        self.results: Dict[str, AnnotationResult] = {}
        self.completed_tasks: List[str] = []
        self.active_annotators: Dict[str, int] = {}  # annotator_id -> task_count

    def add_task(self, task: AnnotationTask) -> str:
        """Add a task to the queue"""
        self.tasks[task.task_id] = task
        logger.info(f"Added annotation task {task.task_id} for conversation {task.conversation_id}")
        return task.task_id

    def get_next_task(self, annotator_id: str, min_priority: ReviewPriority = ReviewPriority.HIGH) -> Optional[AnnotationTask]:
        """Get the next task for an annotator based on priority and availability"""
        # Filter tasks by status and priority
        available_tasks = [
            task for task in self.tasks.values() 
            if task.status == "pending" and 
            ReviewPriority(task.review_priority) >= min_priority
        ]
        
        # Sort by priority (high first) and creation time
        available_tasks.sort(
            key=lambda t: (ReviewPriority(t.review_priority).value, t.created_at),
            reverse=True
        )
        
        if available_tasks:
            task = available_tasks[0]
            task.status = "in_progress"
            task.assigned_annotator = annotator_id
            self.active_annotators[annotator_id] = self.active_annotators.get(annotator_id, 0) + 1
            logger.info(f"Assigned task {task.task_id} to annotator {annotator_id}")
            return task
        
        return None

    def submit_result(self, result: AnnotationResult):
        """Submit annotation result and update the task"""
        if result.task_id in self.tasks:
            task = self.tasks[result.task_id]
            task.status = "completed"
            self.completed_tasks.append(result.task_id)
            
            # Decrement active task count for annotator
            if task.assigned_annotator:
                self.active_annotators[task.assigned_annotator] = max(0, 
                    self.active_annotators.get(task.assigned_annotator, 0) - 1)
            
            self.results[result.task_id] = result
            logger.info(f"Annotator {result.annotator_id} completed task {result.task_id}")
        else:
            logger.warning(f"Attempted to submit result for non-existent task {result.task_id}")

    def get_task_status(self, task_id: str) -> str:
        """Get the status of a specific task"""
        if task_id in self.tasks:
            return self.tasks[task_id].status
        return "not_found"

    def get_annotator_stats(self, annotator_id: str) -> Dict[str, int]:
        """Get statistics for a specific annotator"""
        completed_count = len([r for r in self.results.values() if r.annotator_id == annotator_id])
        active_count = self.active_annotators.get(annotator_id, 0)
        
        return {
            "completed_tasks": completed_count,
            "active_tasks": active_count,
            "total_assigned": completed_count + active_count
        }


class HumanInLoopLabeler:
    """
    System to manage human-in-the-loop labeling of edge cases and low-confidence items.
    Integrates with the automated labeler to identify items needing human review.
    """
    
    def __init__(self, automated_labeler: AutomatedLabeler, confidence_threshold: float = 0.7):
        self.automated_labeler = automated_labeler
        self.confidence_threshold = confidence_threshold
        self.annotation_queue = AnnotationQueue()
        self.logger = logging.getLogger(__name__)

    def identify_low_confidence_items(self, label_bundle: LabelBundle) -> List[str]:
        """Identify labels in a bundle that have low confidence and need human review"""
        low_confidence_items = []
        
        # Check therapeutic response labels
        for i, label in enumerate(label_bundle.therapeutic_response_labels):
            if label.metadata.confidence < self.confidence_threshold:
                low_confidence_items.append(f"therapeutic_response_{i}")
        
        # Check crisis label
        if label_bundle.crisis_label and label_bundle.crisis_label.metadata.confidence < self.confidence_threshold:
            low_confidence_items.append("crisis_label")
        
        # Check therapy modality label
        if label_bundle.therapy_modality_label and label_bundle.therapy_modality_label.metadata.confidence < self.confidence_threshold:
            low_confidence_items.append("therapy_modality")
        
        # Check mental health condition label
        if label_bundle.mental_health_condition_label and label_bundle.mental_health_condition_label.metadata.confidence < self.confidence_threshold:
            low_confidence_items.append("mental_health_condition")
        
        # Check demographic label
        if label_bundle.demographic_label and label_bundle.demographic_label.metadata.confidence < self.confidence_threshold:
            low_confidence_items.append("demographic")
        
        return low_confidence_items

    def should_escalate_to_human(self, label_bundle: LabelBundle, conversation: Conversation) -> bool:
        """Determine if a label bundle should be escalated to human review"""
        low_confidence_items = self.identify_low_confidence_items(label_bundle)
        
        # Also escalate if crisis level is medium or higher (safety first)
        if (label_bundle.crisis_label and 
            label_bundle.crisis_label.crisis_level in [CrisisLevelType.MEDIUM_RISK, 
                                                     CrisisLevelType.HIGH_RISK, 
                                                     CrisisLevelType.IMMEDIATE_RISK]):
            return True
        
        # Escalate if many items need review
        if len(low_confidence_items) > 2:
            return True
            
        # Escalate if any critical labels are low confidence
        for item in low_confidence_items:
            if any(crit in item for crit in ['crisis', 'condition']):
                return True
        
        return len(low_confidence_items) > 0

    def create_annotation_task(self, conversation: Conversation, 
                             suggested_labels: LabelBundle,
                             priority: ReviewPriority = ReviewPriority.HIGH,
                             notes: Optional[List[str]] = None) -> AnnotationTask:
        """Create an annotation task for human review"""
        # Get original automated labels (if existing)
        original_labels = suggested_labels  # In this case, suggested_labels are original automated ones
        
        task = AnnotationTask(
            conversation_id=conversation.conversation_id,
            conversation=conversation,
            original_labels=original_labels,
            suggested_labels=original_labels,
            review_priority=priority,
            notes=notes or []
        )
        
        return task

    def process_conversation_for_human_review(self, conversation: Conversation) -> Optional[str]:
        """Process a conversation and create annotation task if needed"""
        # Generate automated labels
        automated_labels = self.automated_labeler.label_conversation(conversation)
        
        # Check if human review is needed
        if self.should_escalate_to_human(automated_labels, conversation):
            # Determine priority based on confidence and content
            priority = ReviewPriority.LOW
            if any(label.metadata.confidence < 0.5 for label in automated_labels.therapeutic_response_labels):
                priority = ReviewPriority.HIGH
            elif automated_labels.crisis_label and automated_labels.crisis_label.metadata.confidence < 0.8:
                priority = ReviewPriority.HIGH
            
            # Create annotation task
            notes = []
            low_confidence_items = self.identify_low_confidence_items(automated_labels)
            if low_confidence_items:
                notes.append(f"Low confidence on: {', '.join(low_confidence_items)}")
            
            if automated_labels.crisis_label:
                notes.append(f"Crisis level: {automated_labels.crisis_label.crisis_level.value}")
            
            task = self.create_annotation_task(
                conversation=conversation,
                suggested_labels=automated_labels,
                priority=priority,
                notes=notes
            )
            
            task_id = self.annotation_queue.add_task(task)
            self.logger.info(f"Created annotation task {task_id} for conversation {conversation.conversation_id}")
            return task_id
        
        return None  # No human review needed

    def integrate_human_labels(self, result: AnnotationResult, original_bundle: LabelBundle) -> LabelBundle:
        """Integrate human-generated labels with original bundle"""
        # For now, we'll replace the original bundle with human labels
        # In a real system, you might want to merge them intelligently
        updated_bundle = result.applied_labels
        
        # Update provenance to reflect human involvement
        for label in updated_bundle.therapeutic_response_labels:
            if label.metadata.provenance == LabelProvenanceType.AUTOMATED_MODEL:
                label.metadata.provenance = LabelProvenanceType.COMBINED_MODEL_HUMAN
        
        if updated_bundle.crisis_label and updated_bundle.crisis_label.metadata.provenance == LabelProvenanceType.AUTOMATED_MODEL:
            updated_bundle.crisis_label.metadata.provenance = LabelProvenanceType.COMBINED_MODEL_HUMAN
            
        if updated_bundle.therapy_modality_label and updated_bundle.therapy_modality_label.metadata.provenance == LabelProvenanceType.AUTOMATED_MODEL:
            updated_bundle.therapy_modality_label.metadata.provenance = LabelProvenanceType.COMBINED_MODEL_HUMAN
            
        if updated_bundle.mental_health_condition_label and updated_bundle.mental_health_condition_label.metadata.provenance == LabelProvenanceType.AUTOMATED_MODEL:
            updated_bundle.mental_health_condition_label.metadata.provenance = LabelProvenanceType.COMBINED_MODEL_HUMAN
            
        if updated_bundle.demographic_label and updated_bundle.demographic_label.metadata.provenance == LabelProvenanceType.AUTOMATED_MODEL:
            updated_bundle.demographic_label.metadata.provenance = LabelProvenanceType.COMBINED_MODEL_HUMAN
        
        return updated_bundle

    def get_review_statistics(self) -> Dict[str, Any]:
        """Get statistics about the human review process"""
        total_tasks = len(self.annotation_queue.tasks)
        completed_tasks = len(self.annotation_queue.completed_tasks)
        pending_tasks = len([t for t in self.annotation_queue.tasks.values() if t.status == "pending"])
        in_progress_tasks = len([t for t in self.annotation_queue.tasks.values() if t.status == "in_progress"])
        
        return {
            "total_tasks": total_tasks,
            "completed_tasks": completed_tasks,
            "pending_tasks": pending_tasks,
            "in_progress_tasks": in_progress_tasks,
            "completion_rate": completed_tasks / total_tasks if total_tasks > 0 else 0
        }


# Web API interface for the human-in-the-loop system
class HumanInLoopAPI:
    """API endpoints for the human-in-the-loop interface"""
    
    def __init__(self, human_in_loop: HumanInLoopLabeler):
        self.human_in_loop = human_in_loop

    def get_next_annotation_task(self, annotator_id: str) -> Optional[Dict[str, Any]]:
        """Get the next annotation task for an annotator"""
        task = self.human_in_loop.annotation_queue.get_next_task(annotator_id)
        if not task:
            return None
        
        # Convert task to JSON-serializable format
        return {
            "task_id": task.task_id,
            "conversation_id": task.conversation_id,
            "conversation": task.conversation.to_dict(),
            "original_labels": self._bundle_to_dict(task.original_labels),
            "suggested_labels": self._bundle_to_dict(task.suggested_labels),
            "review_priority": task.review_priority.value,
            "created_at": task.created_at,
            "notes": task.notes,
            "metadata": task.metadata
        }

    def submit_annotation_result(self, result_data: Dict[str, Any]) -> bool:
        """Submit annotation result from human annotator"""
        try:
            # Convert data back to objects
            task_id = result_data["task_id"]
            annotator_id = result_data["annotator_id"]
            action = AnnotationAction(result_data["action"])
            
            applied_labels = self._dict_to_bundle(result_data["applied_labels"])
            
            result = AnnotationResult(
                task_id=task_id,
                annotator_id=annotator_id,
                action=action,
                applied_labels=applied_labels,
                confidence_adjustment=result_data.get("confidence_adjustment"),
                feedback=result_data.get("feedback")
            )
            
            self.human_in_loop.annotation_queue.submit_result(result)
            
            # Optionally integrate the human labels back into the system
            if task_id in self.human_in_loop.annotation_queue.tasks:
                original_task = self.human_in_loop.annotation_queue.tasks[task_id]
                integrated_labels = self.human_in_loop.integrate_human_labels(result, original_task.original_labels)
                
            return True
        except Exception as e:
            logger.error(f"Error submitting annotation result: {e}")
            return False

    def _bundle_to_dict(self, bundle: LabelBundle) -> Dict[str, Any]:
        """Convert LabelBundle to dictionary for JSON serialization"""
        return {
            "conversation_id": bundle.conversation_id,
            "label_id": bundle.label_id,
            "therapeutic_response_labels": [
                {
                    "response_type": label.response_type.value,
                    "effectiveness_score": label.effectiveness_score,
                    "technique_usage_accuracy": label.technique_usage_accuracy,
                    "skill_level": label.skill_level,
                    "metadata": {
                        "confidence": label.metadata.confidence,
                        "confidence_explanation": label.metadata.confidence_explanation,
                        "provenance": label.metadata.provenance.value,
                        "annotator_id": label.metadata.annotator_id,
                        "model_name": label.metadata.model_name,
                        "model_version": label.metadata.model_version
                    }
                }
                for label in bundle.therapeutic_response_labels
            ],
            "crisis_label": {
                "crisis_level": bundle.crisis_label.crisis_level.value,
                "crisis_types": bundle.crisis_label.crisis_types,
                "risk_factors": bundle.crisis_label.risk_factors,
                "protection_factors": bundle.crisis_label.protection_factors,
                "estimated_risk_probability": bundle.crisis_label.estimated_risk_probability,
                "intervention_needed": bundle.crisis_label.intervention_needed,
                "metadata": {
                    "confidence": bundle.crisis_label.metadata.confidence,
                    "confidence_explanation": bundle.crisis_label.metadata.confidence_explanation,
                    "provenance": bundle.crisis_label.metadata.provenance.value,
                    "annotator_id": bundle.crisis_label.metadata.annotator_id,
                    "model_name": bundle.crisis_label.metadata.model_name,
                    "model_version": bundle.crisis_label.metadata.model_version
                }
            } if bundle.crisis_label else None,
            "therapy_modality_label": {
                "modality": bundle.therapy_modality_label.modality.value,
                "modality_specific_techniques": bundle.therapy_modality_label.modality_specific_techniques,
                "modality_adherence_score": bundle.therapy_modality_label.modality_adherence_score,
                "metadata": {
                    "confidence": bundle.therapy_modality_label.metadata.confidence,
                    "confidence_explanation": bundle.therapy_modality_label.metadata.confidence_explanation,
                    "provenance": bundle.therapy_modality_label.metadata.provenance.value,
                    "annotator_id": bundle.therapy_modality_label.metadata.annotator_id,
                    "model_name": bundle.therapy_modality_label.metadata.model_name,
                    "model_version": bundle.therapy_modality_label.metadata.model_version
                }
            } if bundle.therapy_modality_label else None,
            "mental_health_condition_label": {
                "conditions": [c.value for c in bundle.mental_health_condition_label.conditions] if bundle.mental_health_condition_label else [],
                "severity": bundle.mental_health_condition_label.severity if bundle.mental_health_condition_label else None,
                "primary_condition": bundle.mental_health_condition_label.primary_condition.value if bundle.mental_health_condition_label and bundle.mental_health_condition_label.primary_condition else None,
                "co_morbidities": [c.value for c in bundle.mental_health_condition_label.co_morbidities] if bundle.mental_health_condition_label else [],
                "metadata": {
                    "confidence": bundle.mental_health_condition_label.metadata.confidence if bundle.mental_health_condition_label else None,
                    "confidence_explanation": bundle.mental_health_condition_label.metadata.confidence_explanation if bundle.mental_health_condition_label else None,
                    "provenance": bundle.mental_health_condition_label.metadata.provenance.value if bundle.mental_health_condition_label else None,
                    "annotator_id": bundle.mental_health_condition_label.metadata.annotator_id if bundle.mental_health_condition_label else None,
                    "model_name": bundle.mental_health_condition_label.metadata.model_name if bundle.mental_health_condition_label else None,
                    "model_version": bundle.mental_health_condition_label.metadata.model_version if bundle.mental_health_condition_label else None
                }
            } if bundle.mental_health_condition_label else None,
            "demographic_label": {
                "demographics": [d.value for d in bundle.demographic_label.demographics] if bundle.demographic_label else [],
                "estimated_accuracy": bundle.demographic_label.estimated_accuracy if bundle.demographic_label else None,
                "metadata": {
                    "confidence": bundle.demographic_label.metadata.confidence if bundle.demographic_label else None,
                    "confidence_explanation": bundle.demographic_label.metadata.confidence_explanation if bundle.demographic_label else None,
                    "provenance": bundle.demographic_label.metadata.provenance.value if bundle.demographic_label else None,
                    "annotator_id": bundle.demographic_label.metadata.annotator_id if bundle.demographic_label else None,
                    "model_name": bundle.demographic_label.metadata.model_name if bundle.demographic_label else None,
                    "model_version": bundle.demographic_label.metadata.model_version if bundle.demographic_label else None
                }
            } if bundle.demographic_label else None,
            "created_at": bundle.created_at,
            "version": bundle.version,
            "additional_labels": bundle.additional_labels
        }

    def _dict_to_bundle(self, data: Dict[str, Any]) -> LabelBundle:
        """Convert dictionary back to LabelBundle"""
        # This is a simplified version - in practice you'd need more validation
        therapeutic_labels = []
        for label_data in data.get("therapeutic_response_labels", []):
            label = TherapeuticResponseLabel(
                response_type=TherapeuticResponseType(label_data["response_type"]),
                effectiveness_score=label_data.get("effectiveness_score"),
                technique_usage_accuracy=label_data.get("technique_usage_accuracy"),
                skill_level=label_data.get("skill_level"),
                metadata=LabelMetadata(
                    confidence=label_data["metadata"]["confidence"],
                    confidence_explanation=label_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(label_data["metadata"]["provenance"]),
                    annotator_id=label_data["metadata"]["annotator_id"],
                    model_name=label_data["metadata"]["model_name"],
                    model_version=label_data["metadata"]["model_version"]
                )
            )
            therapeutic_labels.append(label)

        crisis_label = None
        if data.get("crisis_label"):
            crisis_data = data["crisis_label"]
            crisis_label = CrisisLabel(
                crisis_level=CrisisLevelType(crisis_data["crisis_level"]),
                crisis_types=crisis_data.get("crisis_types", []),
                risk_factors=crisis_data.get("risk_factors", []),
                protection_factors=crisis_data.get("protection_factors", []),
                estimated_risk_probability=crisis_data.get("estimated_risk_probability"),
                intervention_needed=crisis_data.get("intervention_needed", False),
                metadata=LabelMetadata(
                    confidence=crisis_data["metadata"]["confidence"],
                    confidence_explanation=crisis_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(crisis_data["metadata"]["provenance"]),
                    annotator_id=crisis_data["metadata"]["annotator_id"],
                    model_name=crisis_data["metadata"]["model_name"],
                    model_version=crisis_data["metadata"]["model_version"]
                )
            )

        modality_label = None
        if data.get("therapy_modality_label"):
            modality_data = data["therapy_modality_label"]
            modality_label = TherapyModalityLabel(
                modality=TherapyModalityType(modality_data["modality"]),
                modality_specific_techniques=modality_data.get("modality_specific_techniques", []),
                modality_adherence_score=modality_data.get("modality_adherence_score"),
                metadata=LabelMetadata(
                    confidence=modality_data["metadata"]["confidence"],
                    confidence_explanation=modality_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(modality_data["metadata"]["provenance"]),
                    annotator_id=modality_data["metadata"]["annotator_id"],
                    model_name=modality_data["metadata"]["model_name"],
                    model_version=modality_data["metadata"]["model_version"]
                )
            )

        condition_label = None
        if data.get("mental_health_condition_label"):
            condition_data = data["mental_health_condition_label"]
            condition_label = MentalHealthConditionLabel(
                conditions=[MentalHealthConditionType(c) for c in condition_data.get("conditions", [])],
                severity=condition_data.get("severity"),
                primary_condition=MentalHealthConditionType(condition_data["primary_condition"]) if condition_data.get("primary_condition") else None,
                co_morbidities=[MentalHealthConditionType(c) for c in condition_data.get("co_morbidities", [])],
                metadata=LabelMetadata(
                    confidence=condition_data["metadata"]["confidence"],
                    confidence_explanation=condition_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(condition_data["metadata"]["provenance"]),
                    annotator_id=condition_data["metadata"]["annotator_id"],
                    model_name=condition_data["metadata"]["model_name"],
                    model_version=condition_data["metadata"]["model_version"]
                )
            )

        demographic_label = None
        if data.get("demographic_label"):
            demographic_data = data["demographic_label"]
            demographic_label = DemographicLabel(
                demographics=[DemographicType(d) for d in demographic_data.get("demographics", [])],
                estimated_accuracy=demographic_data.get("estimated_accuracy"),
                metadata=LabelMetadata(
                    confidence=demographic_data["metadata"]["confidence"],
                    confidence_explanation=demographic_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(demographic_data["metadata"]["provenance"]),
                    annotator_id=demographic_data["metadata"]["annotator_id"],
                    model_name=demographic_data["metadata"]["model_name"],
                    model_version=demographic_data["metadata"]["model_version"]
                )
            )

        return LabelBundle(
            conversation_id=data["conversation_id"],
            label_id=data.get("label_id", str(uuid.uuid4())),
            therapeutic_response_labels=therapeutic_labels,
            crisis_label=crisis_label,
            therapy_modality_label=modality_label,
            mental_health_condition_label=condition_label,
            demographic_label=demographic_label,
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            version=data.get("version", "1.0"),
            additional_labels=data.get("additional_labels", {})
        )


def create_human_in_loop_system() -> HumanInLoopLabeler:
    """Create a default human-in-the-loop system"""
    automated_labeler = AutomatedLabeler()
    return HumanInLoopLabeler(automated_labeler, confidence_threshold=0.7)


# Example usage
def test_human_in_loop():
    """Test the human-in-the-loop system"""
    from .conversation_schema import Conversation, Message
    
    # Create a test conversation
    conversation = Conversation()
    conversation.add_message("therapist", "How are you feeling today?")
    conversation.add_message("client", "I've been really down lately and having thoughts about not wanting to be here anymore.")
    conversation.add_message("therapist", "I'm sorry to hear you're feeling this way. Can you tell me more about these thoughts?")
    
    # Create the human-in-the-loop system
    human_loop = create_human_in_loop_system()
    
    # Process the conversation - should identify it needs human review
    task_id = human_loop.process_conversation_for_human_review(conversation)
    
    if task_id:
        print(f"Created annotation task {task_id} for human review")
        
        # Get task statistics
        stats = human_loop.get_review_statistics()
        print(f"Review statistics: {stats}")
        
        # Get the next task for an annotator
        task = human_loop.annotation_queue.get_next_task("test_annotator")
        if task:
            print(f"Annotator received task for conversation {task.conversation_id}")
            print(f"Review priority: {task.review_priority.value}")
            print(f"Notes: {task.notes}")
            
            # Create a mock annotation result
            from .label_taxonomy import LabelMetadata
            updated_bundle = human_loop.automated_labeler.label_conversation(conversation)
            
            result = AnnotationResult(
                task_id=task_id,
                annotator_id="test_annotator",
                action=AnnotationAction.CONFIRM,
                applied_labels=updated_bundle
            )
            
            human_loop.annotation_queue.submit_result(result)
            print("Submitted annotation result")
            
            # Check updated statistics
            stats = human_loop.get_review_statistics()
            print(f"Updated statistics: {stats}")
    else:
        print("No human review needed for this conversation")


if __name__ == "__main__":
    test_human_in_loop()