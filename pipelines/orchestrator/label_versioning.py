"""
Label versioning and provenance tracking system.
Implements comprehensive tracking of label changes, history, and sources.
"""

from dataclasses import dataclass, field
from datetime import datetime
from typing import List, Optional, Dict, Any, Union
import uuid
import json
import hashlib
from enum import Enum
import logging
from .label_taxonomy import (
    LabelBundle, LabelMetadata, LabelProvenanceType,
    TherapeuticResponseLabel, CrisisLabel, TherapyModalityLabel,
    MentalHealthConditionLabel, DemographicLabel
)
from .conversation_schema import Conversation

logger = logging.getLogger(__name__)


class VersionAction(Enum):
    """Actions that can be performed on labels"""
    CREATED = "created"
    UPDATED = "updated"
    CONFIRMED = "confirmed"
    CORRECTED = "corrected"
    REJECTED = "rejected"
    MERGED = "merged"
    SPLIT = "split"


@dataclass
class LabelVersion:
    """Represents a single version of a label with metadata"""
    version_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    parent_version_id: Optional[str] = None  # Previous version ID
    label_bundle_id: str  # ID of the label bundle this version belongs to
    conversation_id: str
    label_data: Dict[str, Any]  # Serialized label data
    version_number: int = 1
    action: VersionAction = VersionAction.CREATED
    actor: str = "system"  # Who performed the action (user ID, system, etc.)
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    description: Optional[str] = None
    confidence_change: Optional[float] = None  # Change in confidence from previous version
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ProvenanceRecord:
    """Record of how a label was created/modified"""
    record_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    label_bundle_id: str
    source: LabelProvenanceType
    timestamp: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    version_id: Optional[str] = None  # Associated version ID
    model_name: Optional[str] = None
    model_version: Optional[str] = None
    human_annotator: Optional[str] = None
    human_annotation_task_id: Optional[str] = None
    algorithm: Optional[str] = None  # Specific algorithm used
    parameters: Dict[str, Any] = field(default_factory=dict)  # Algorithm parameters
    input_features: List[str] = field(default_factory=list)  # Features used for labeling
    validation_status: str = "pending"  # pending, validated, invalidated
    validation_timestamp: Optional[str] = None
    validation_annotator: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class LabelHistory:
    """Complete history of all versions and provenance for a label bundle"""
    label_bundle_id: str
    conversation_id: str
    versions: List[LabelVersion] = field(default_factory=list)
    provenance_records: List[ProvenanceRecord] = field(default_factory=list)
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    last_modified: str = field(default_factory=lambda: datetime.utcnow().isoformat())

    def add_version(self, version: LabelVersion):
        """Add a new version to the history"""
        self.versions.append(version)
        self.last_modified = datetime.utcnow().isoformat()

    def add_provenance_record(self, record: ProvenanceRecord):
        """Add a provenance record to the history"""
        self.provenance_records.append(record)
        self.last_modified = datetime.utcnow().isoformat()

    def get_current_version(self) -> Optional[LabelVersion]:
        """Get the most recent version"""
        if self.versions:
            return max(self.versions, key=lambda v: v.version_number)
        return None

    def get_version(self, version_number: int) -> Optional[LabelVersion]:
        """Get a specific version by number"""
        for version in self.versions:
            if version.version_number == version_number:
                return version
        return None

    def get_versions_by_actor(self, actor: str) -> List[LabelVersion]:
        """Get all versions created by a specific actor"""
        return [v for v in self.versions if v.actor == actor]

    def get_versions_by_action(self, action: VersionAction) -> List[LabelVersion]:
        """Get all versions with a specific action type"""
        return [v for v in self.versions if v.action == action]


class LabelVersionManager:
    """Manages label versioning and provenance tracking"""

    def __init__(self):
        self.label_histories: Dict[str, LabelHistory] = {}
        self.logger = logging.getLogger(__name__)

    def create_initial_version(self, label_bundle: LabelBundle, actor: str = "system", 
                              description: Optional[str] = None) -> LabelVersion:
        """Create the initial version for a label bundle"""
        version = LabelVersion(
            label_bundle_id=label_bundle.label_id,
            conversation_id=label_bundle.conversation_id,
            label_data=self._serialize_label_bundle(label_bundle),
            version_number=1,
            action=VersionAction.CREATED,
            actor=actor,
            description=description or "Initial label bundle creation"
        )
        
        # Create provenance record
        provenance = self._create_provenance_record(label_bundle, version.version_id)
        
        # Store in history
        history = self._get_or_create_history(label_bundle.label_id, label_bundle.conversation_id)
        history.add_version(version)
        history.add_provenance_record(provenance)
        
        self.logger.info(f"Created initial version {version.version_number} for label bundle {label_bundle.label_id}")
        return version

    def update_label_bundle(self, label_bundle: LabelBundle, 
                           previous_version: LabelVersion,
                           actor: str,
                           action: VersionAction = VersionAction.UPDATED,
                           description: Optional[str] = None,
                           confidence_change: Optional[float] = None) -> LabelVersion:
        """Update a label bundle and create a new version"""
        new_version = LabelVersion(
            parent_version_id=previous_version.version_id,
            label_bundle_id=label_bundle.label_id,
            conversation_id=label_bundle.conversation_id,
            label_data=self._serialize_label_bundle(label_bundle),
            version_number=previous_version.version_number + 1,
            action=action,
            actor=actor,
            description=description or f"Update to label bundle version {previous_version.version_number + 1}",
            confidence_change=confidence_change
        )
        
        # Create provenance record
        provenance = self._create_provenance_record(label_bundle, new_version.version_id, actor=actor)
        
        # Store in history
        history = self._get_or_create_history(label_bundle.label_id, label_bundle.conversation_id)
        history.add_version(new_version)
        history.add_provenance_record(provenance)
        
        self.logger.info(f"Updated label bundle {label_bundle.label_id} to version {new_version.version_number}")
        return new_version

    def get_history(self, label_bundle_id: str) -> Optional[LabelHistory]:
        """Get the complete history for a label bundle"""
        return self.label_histories.get(label_bundle_id)

    def get_current_bundle(self, label_bundle_id: str) -> Optional[LabelBundle]:
        """Get the current (most recent) version of a label bundle"""
        history = self.get_history(label_bundle_id)
        if not history:
            return None
        
        current_version = history.get_current_version()
        if not current_version:
            return None
        
        return self._deserialize_label_bundle(current_version.label_data)

    def _get_or_create_history(self, label_bundle_id: str, conversation_id: str) -> LabelHistory:
        """Get existing history or create new one"""
        if label_bundle_id not in self.label_histories:
            self.label_histories[label_bundle_id] = LabelHistory(
                label_bundle_id=label_bundle_id,
                conversation_id=conversation_id
            )
        return self.label_histories[label_bundle_id]

    def _serialize_label_bundle(self, bundle: LabelBundle) -> Dict[str, Any]:
        """Serialize a label bundle to dictionary format"""
        return {
            "label_id": bundle.label_id,
            "conversation_id": bundle.conversation_id,
            "created_at": bundle.created_at,
            "version": bundle.version,
            "therapeutic_response_labels": [
                {
                    "response_type": label.response_type.value,
                    "effectiveness_score": label.effectiveness_score,
                    "technique_usage_accuracy": label.technique_usage_accuracy,
                    "skill_level": label.skill_level,
                    "metadata": {
                        "created_at": label.metadata.created_at,
                        "version": label.metadata.version,
                        "confidence": label.metadata.confidence,
                        "confidence_explanation": label.metadata.confidence_explanation,
                        "provenance": label.metadata.provenance.value,
                        "annotator_id": label.metadata.annotator_id,
                        "model_name": label.metadata.model_name,
                        "model_version": label.metadata.model_version,
                        "additional_context": label.metadata.additional_context
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
                    "created_at": bundle.crisis_label.metadata.created_at,
                    "version": bundle.crisis_label.metadata.version,
                    "confidence": bundle.crisis_label.metadata.confidence,
                    "confidence_explanation": bundle.crisis_label.metadata.confidence_explanation,
                    "provenance": bundle.crisis_label.metadata.provenance.value,
                    "annotator_id": bundle.crisis_label.metadata.annotator_id,
                    "model_name": bundle.crisis_label.metadata.model_name,
                    "model_version": bundle.crisis_label.metadata.model_version,
                    "additional_context": bundle.crisis_label.metadata.additional_context
                }
            } if bundle.crisis_label else None,
            "therapy_modality_label": {
                "modality": bundle.therapy_modality_label.modality.value,
                "modality_specific_techniques": bundle.therapy_modality_label.modality_specific_techniques,
                "modality_adherence_score": bundle.therapy_modality_label.modality_adherence_score,
                "metadata": {
                    "created_at": bundle.therapy_modality_label.metadata.created_at,
                    "version": bundle.therapy_modality_label.metadata.version,
                    "confidence": bundle.therapy_modality_label.metadata.confidence,
                    "confidence_explanation": bundle.therapy_modality_label.metadata.confidence_explanation,
                    "provenance": bundle.therapy_modality_label.metadata.provenance.value,
                    "annotator_id": bundle.therapy_modality_label.metadata.annotator_id,
                    "model_name": bundle.therapy_modality_label.metadata.model_name,
                    "model_version": bundle.therapy_modality_label.metadata.model_version,
                    "additional_context": bundle.therapy_modality_label.metadata.additional_context
                }
            } if bundle.therapy_modality_label else None,
            "mental_health_condition_label": {
                "conditions": [c.value for c in bundle.mental_health_condition_label.conditions] if bundle.mental_health_condition_label else [],
                "severity": bundle.mental_health_condition_label.severity,
                "primary_condition": bundle.mental_health_condition_label.primary_condition.value if bundle.mental_health_condition_label and bundle.mental_health_condition_label.primary_condition else None,
                "co_morbidities": [c.value for c in bundle.mental_health_condition_label.co_morbidities] if bundle.mental_health_condition_label else [],
                "metadata": {
                    "created_at": bundle.mental_health_condition_label.metadata.created_at,
                    "version": bundle.mental_health_condition_label.metadata.version,
                    "confidence": bundle.mental_health_condition_label.metadata.confidence,
                    "confidence_explanation": bundle.mental_health_condition_label.metadata.confidence_explanation,
                    "provenance": bundle.mental_health_condition_label.metadata.provenance.value,
                    "annotator_id": bundle.mental_health_condition_label.metadata.annotator_id,
                    "model_name": bundle.mental_health_condition_label.metadata.model_name,
                    "model_version": bundle.mental_health_condition_label.metadata.model_version,
                    "additional_context": bundle.mental_health_condition_label.metadata.additional_context
                }
            } if bundle.mental_health_condition_label else None,
            "demographic_label": {
                "demographics": [d.value for d in bundle.demographic_label.demographics] if bundle.demographic_label else [],
                "estimated_accuracy": bundle.demographic_label.estimated_accuracy,
                "metadata": {
                    "created_at": bundle.demographic_label.metadata.created_at,
                    "version": bundle.demographic_label.metadata.version,
                    "confidence": bundle.demographic_label.metadata.confidence,
                    "confidence_explanation": bundle.demographic_label.metadata.confidence_explanation,
                    "provenance": bundle.demographic_label.metadata.provenance.value,
                    "annotator_id": bundle.demographic_label.metadata.annotator_id,
                    "model_name": bundle.demographic_label.metadata.model_name,
                    "model_version": bundle.demographic_label.metadata.model_version,
                    "additional_context": bundle.demographic_label.metadata.additional_context
                }
            } if bundle.demographic_label else None,
            "additional_labels": bundle.additional_labels
        }

    def _deserialize_label_bundle(self, data: Dict[str, Any]) -> LabelBundle:
        """Deserialize dictionary data back to a label bundle"""
        from .label_taxonomy import (
            TherapeuticResponseType, CrisisLevelType, TherapyModalityType,
            MentalHealthConditionType, DemographicType
        )
        
        # Deserialize therapeutic response labels
        therapeutic_labels = []
        for label_data in data.get("therapeutic_response_labels", []):
            label = TherapeuticResponseLabel(
                response_type=TherapeuticResponseType(label_data["response_type"]),
                effectiveness_score=label_data.get("effectiveness_score"),
                technique_usage_accuracy=label_data.get("technique_usage_accuracy"),
                skill_level=label_data.get("skill_level"),
                metadata=LabelMetadata(
                    created_at=label_data["metadata"]["created_at"],
                    version=label_data["metadata"]["version"],
                    confidence=label_data["metadata"]["confidence"],
                    confidence_explanation=label_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(label_data["metadata"]["provenance"]),
                    annotator_id=label_data["metadata"].get("annotator_id"),
                    model_name=label_data["metadata"].get("model_name"),
                    model_version=label_data["metadata"].get("model_version"),
                    additional_context=label_data["metadata"].get("additional_context", {})
                )
            )
            therapeutic_labels.append(label)

        # Deserialize crisis label
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
                    created_at=crisis_data["metadata"]["created_at"],
                    version=crisis_data["metadata"]["version"],
                    confidence=crisis_data["metadata"]["confidence"],
                    confidence_explanation=crisis_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(crisis_data["metadata"]["provenance"]),
                    annotator_id=crisis_data["metadata"].get("annotator_id"),
                    model_name=crisis_data["metadata"].get("model_name"),
                    model_version=crisis_data["metadata"].get("model_version"),
                    additional_context=crisis_data["metadata"].get("additional_context", {})
                )
            )

        # Deserialize therapy modality label
        modality_label = None
        if data.get("therapy_modality_label"):
            modality_data = data["therapy_modality_label"]
            modality_label = TherapyModalityLabel(
                modality=TherapyModalityType(modality_data["modality"]),
                modality_specific_techniques=modality_data.get("modality_specific_techniques", []),
                modality_adherence_score=modality_data.get("modality_adherence_score"),
                metadata=LabelMetadata(
                    created_at=modality_data["metadata"]["created_at"],
                    version=modality_data["metadata"]["version"],
                    confidence=modality_data["metadata"]["confidence"],
                    confidence_explanation=modality_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(modality_data["metadata"]["provenance"]),
                    annotator_id=modality_data["metadata"].get("annotator_id"),
                    model_name=modality_data["metadata"].get("model_name"),
                    model_version=modality_data["metadata"].get("model_version"),
                    additional_context=modality_data["metadata"].get("additional_context", {})
                )
            )

        # Deserialize mental health condition label
        condition_label = None
        if data.get("mental_health_condition_label"):
            condition_data = data["mental_health_condition_label"]
            condition_label = MentalHealthConditionLabel(
                conditions=[MentalHealthConditionType(c) for c in condition_data.get("conditions", [])],
                severity=condition_data.get("severity"),
                primary_condition=MentalHealthConditionType(condition_data["primary_condition"]) if condition_data.get("primary_condition") else None,
                co_morbidities=[MentalHealthConditionType(c) for c in condition_data.get("co_morbidities", [])] if condition_data.get("co_morbidities") else [],
                metadata=LabelMetadata(
                    created_at=condition_data["metadata"]["created_at"],
                    version=condition_data["metadata"]["version"],
                    confidence=condition_data["metadata"]["confidence"],
                    confidence_explanation=condition_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(condition_data["metadata"]["provenance"]),
                    annotator_id=condition_data["metadata"].get("annotator_id"),
                    model_name=condition_data["metadata"].get("model_name"),
                    model_version=condition_data["metadata"].get("model_version"),
                    additional_context=condition_data["metadata"].get("additional_context", {})
                )
            )

        # Deserialize demographic label
        demographic_label = None
        if data.get("demographic_label"):
            demo_data = data["demographic_label"]
            demographic_label = DemographicLabel(
                demographics=[DemographicType(d) for d in demo_data.get("demographics", [])],
                estimated_accuracy=demo_data.get("estimated_accuracy"),
                metadata=LabelMetadata(
                    created_at=demo_data["metadata"]["created_at"],
                    version=demo_data["metadata"]["version"],
                    confidence=demo_data["metadata"]["confidence"],
                    confidence_explanation=demo_data["metadata"]["confidence_explanation"],
                    provenance=LabelProvenanceType(demo_data["metadata"]["provenance"]),
                    annotator_id=demo_data["metadata"].get("annotator_id"),
                    model_name=demo_data["metadata"].get("model_name"),
                    model_version=demo_data["metadata"].get("model_version"),
                    additional_context=demo_data["metadata"].get("additional_context", {})
                )
            )

        return LabelBundle(
            conversation_id=data["conversation_id"],
            label_id=data["label_id"],
            therapeutic_response_labels=therapeutic_labels,
            crisis_label=crisis_label,
            therapy_modality_label=modality_label,
            mental_health_condition_label=condition_label,
            demographic_label=demographic_label,
            created_at=data.get("created_at", datetime.utcnow().isoformat()),
            version=data.get("version", "1.0"),
            additional_labels=data.get("additional_labels", {})
        )

    def _create_provenance_record(self, label_bundle: LabelBundle, 
                                 version_id: Optional[str] = None,
                                 actor: Optional[str] = None) -> ProvenanceRecord:
        """Create a provenance record for a label bundle"""
        # Determine the primary source based on the metadata in the labels
        primary_source = LabelProvenanceType.AUTOMATED_MODEL  # Default
        
        # Check the types of provenance in the bundle to determine the primary type
        all_provenances = []
        for label in label_bundle.therapeutic_response_labels:
            all_provenances.append(label.metadata.provenance)
        if label_bundle.crisis_label:
            all_provenances.append(label_bundle.crisis_label.metadata.provenance)
        if label_bundle.therapy_modality_label:
            all_provenances.append(label_bundle.therapy_modality_label.metadata.provenance)
        if label_bundle.mental_health_condition_label:
            all_provenances.append(label_bundle.mental_health_condition_label.metadata.provenance)
        if label_bundle.demographic_label:
            all_provenances.append(label_bundle.demographic_label.metadata.provenance)
        
        # If any are human-related, mark as human
        if any(p in [LabelProvenanceType.HUMAN_EXPERT, LabelProvenanceType.HUMAN_IN_THE_LOOP] for p in all_provenances):
            primary_source = LabelProvenanceType.HUMAN_IN_THE_LOOP
        elif any(p == LabelProvenanceType.COMBINED_MODEL_HUMAN for p in all_provenances):
            primary_source = LabelProvenanceType.COMBINED_MODEL_HUMAN
        elif any(p == LabelProvenanceType.SYNTHETIC for p in all_provenances):
            primary_source = LabelProvenanceType.SYNTHETIC

        # Extract model info if available
        model_name = None
        model_version = None
        human_annotator = None
        
        # Find first model name/version or human annotator in the bundle
        for label in label_bundle.therapeutic_response_labels:
            if label.metadata.model_name:
                model_name = label.metadata.model_name
                model_version = label.metadata.model_version
            if label.metadata.annotator_id:
                human_annotator = label.metadata.annotator_id
                primary_source = LabelProvenanceType.HUMAN_EXPERT
        if not model_name and label_bundle.crisis_label:
            model_name = label_bundle.crisis_label.metadata.model_name
            model_version = label_bundle.crisis_label.metadata.model_version
            if label_bundle.crisis_label.metadata.annotator_id:
                human_annotator = label_bundle.crisis_label.metadata.annotator_id
                primary_source = LabelProvenanceType.HUMAN_EXPERT

        record = ProvenanceRecord(
            label_bundle_id=label_bundle.label_id,
            source=primary_source,
            version_id=version_id,
            model_name=model_name,
            model_version=model_version,
            human_annotator=human_annotator,
            timestamp=datetime.utcnow().isoformat()
        )
        
        return record

    def compute_bundle_hash(self, label_bundle: LabelBundle) -> str:
        """Compute a hash for a label bundle to detect changes"""
        # Serialize the bundle to a canonical form
        serialized = json.dumps(self._serialize_label_bundle(label_bundle), sort_keys=True)
        # Create hash
        return hashlib.sha256(serialized.encode()).hexdigest()

    def get_label_change_summary(self, label_bundle_id: str) -> Dict[str, Any]:
        """Get a summary of changes across all versions of a label bundle"""
        history = self.get_history(label_bundle_id)
        if not history or len(history.versions) < 2:
            return {"message": "No changes to report - only one version exists"}

        changes = {
            "total_versions": len(history.versions),
            "version_range": f"v1 to v{max(v.version_number for v in history.versions)}",
            "actions": {},
            "actors": {},
            "time_span": None
        }

        # Count actions and actors
        for version in history.versions:
            action_key = version.action.value
            changes["actions"][action_key] = changes["actions"].get(action_key, 0) + 1

            actor_key = version.actor
            changes["actors"][actor_key] = changes["actors"].get(actor_key, 0) + 1

        # Calculate time span
        if history.versions:
            timestamps = [datetime.fromisoformat(v.timestamp.replace('Z', '+00:00')) for v in history.versions]
            start_time = min(timestamps)
            end_time = max(timestamps)
            changes["time_span"] = {
                "start": start_time.isoformat(),
                "end": end_time.isoformat(),
                "duration_days": (end_time - start_time).days
            }

        return changes


class ProvenanceAnalyzer:
    """Analyzer for provenance data across multiple label bundles"""

    def __init__(self, version_manager: LabelVersionManager):
        self.version_manager = version_manager

    def analyze_provenance_distribution(self) -> Dict[str, int]:
        """Analyze the distribution of provenance types across all label bundles"""
        provenance_counts = {}
        
        for history in self.version_manager.label_histories.values():
            if history.provenance_records:
                # Use the most recent provenance record for the bundle
                latest_record = max(history.provenance_records, key=lambda r: r.timestamp)
                provenance_type = latest_record.source.value
                provenance_counts[provenance_type] = provenance_counts.get(provenance_type, 0) + 1
        
        return provenance_counts

    def find_low_confidence_provenance_chains(self) -> List[Dict[str, Any]]:
        """Find label bundles with chains of low-confidence labels"""
        low_confidence_chains = []
        
        for history in self.version_manager.label_histories.values():
            low_confidence_versions = []
            for version in history.versions:
                # Deserialize the label bundle to check confidence levels
                bundle = self.version_manager._deserialize_label_bundle(version.label_data)
                if self._has_low_confidence_labels(bundle):
                    low_confidence_versions.append({
                        "version_id": version.version_id,
                        "version_number": version.version_number,
                        "timestamp": version.timestamp,
                        "confidence_issue": self._get_confidence_issues(bundle)
                    })
            
            if len(low_confidence_versions) > 1:
                low_confidence_chains.append({
                    "label_bundle_id": history.label_bundle_id,
                    "conversation_id": history.conversation_id,
                    "low_confidence_versions": low_confidence_versions
                })
        
        return low_confidence_chains

    def _has_low_confidence_labels(self, bundle: LabelBundle) -> bool:
        """Check if any labels in the bundle have low confidence"""
        for label in bundle.therapeutic_response_labels:
            if label.metadata.confidence < 0.7:
                return True
        if bundle.crisis_label and bundle.crisis_label.metadata.confidence < 0.8:
            return True
        if bundle.therapy_modality_label and bundle.therapy_modality_label.metadata.confidence < 0.7:
            return True
        if bundle.mental_health_condition_label and bundle.mental_health_condition_label.metadata.confidence < 0.7:
            return True
        if bundle.demographic_label and bundle.demographic_label.metadata.confidence < 0.6:
            return True
        return False

    def _get_confidence_issues(self, bundle: LabelBundle) -> List[str]:
        """Get details about low confidence issues in a bundle"""
        issues = []
        for i, label in enumerate(bundle.therapeutic_response_labels):
            if label.metadata.confidence < 0.7:
                issues.append(f"Therapeutic response {i}: {label.response_type.value} confidence {label.metadata.confidence}")
        if bundle.crisis_label and bundle.crisis_label.metadata.confidence < 0.8:
            issues.append(f"Crisis label: {bundle.crisis_label.crisis_level.value} confidence {bundle.crisis_label.metadata.confidence}")
        if bundle.therapy_modality_label and bundle.therapy_modality_label.metadata.confidence < 0.7:
            issues.append(f"Modality label: {bundle.therapy_modality_label.modality.value} confidence {bundle.therapy_modality_label.metadata.confidence}")
        if bundle.mental_health_condition_label and bundle.mental_health_condition_label.metadata.confidence < 0.7:
            issues.append(f"Condition label confidence {bundle.mental_health_condition_label.metadata.confidence}")
        if bundle.demographic_label and bundle.demographic_label.metadata.confidence < 0.6:
            issues.append(f"Demographic label confidence {bundle.demographic_label.metadata.confidence}")
        return issues

    def get_provenance_timeline(self, label_bundle_id: str) -> List[Dict[str, Any]]:
        """Get a timeline of provenance changes for a specific label bundle"""
        history = self.version_manager.get_history(label_bundle_id)
        if not history:
            return []

        timeline = []
        for record in sorted(history.provenance_records, key=lambda r: r.timestamp):
            timeline.append({
                "timestamp": record.timestamp,
                "source": record.source.value,
                "model_name": record.model_name,
                "model_version": record.model_version,
                "human_annotator": record.human_annotator,
                "version_id": record.version_id
            })

        return timeline


def create_version_manager() -> LabelVersionManager:
    """Create a default label version manager"""
    return LabelVersionManager()


# Example usage
def test_label_versioning():
    """Test the label versioning system"""
    from .label_taxonomy import (
        TherapeuticResponseLabel, CrisisLabel, LabelMetadata, LabelProvenanceType,
        TherapeuticResponseType, CrisisLevelType
    )
    from .conversation_schema import Conversation
    
    # Create a test label bundle
    conversation = Conversation()
    
    # Create initial bundle
    initial_bundle = LabelBundle(conversation_id=conversation.conversation_id)
    
    # Add some labels
    initial_bundle.therapeutic_response_labels.append(
        TherapeuticResponseLabel(
            response_type=TherapeuticResponseType.EMPATHY,
            metadata=LabelMetadata(
                confidence=0.85,
                provenance=LabelProvenanceType.AUTOMATED_MODEL
            )
        )
    )
    
    initial_bundle.crisis_label = CrisisLabel(
        crisis_level=CrisisLevelType.LOW_RISK,
        metadata=LabelMetadata(
            confidence=0.9,
            provenance=LabelProvenanceType.AUTOMATED_MODEL
        )
    )

    # Create version manager
    vm = create_version_manager()
    
    # Create initial version
    initial_version = vm.create_initial_version(initial_bundle, "system", "Initial automated labels")
    print(f"Created initial version: {initial_version.version_number}")
    
    # Update the bundle (simulate human correction)
    updated_bundle = initial_bundle
    updated_bundle.therapeutic_response_labels[0].metadata.confidence = 0.95
    updated_bundle.therapeutic_response_labels[0].metadata.provenance = LabelProvenanceType.HUMAN_EXPERT
    updated_bundle.therapeutic_response_labels[0].metadata.annotator_id = "human_expert_123"
    
    # Create updated version
    updated_version = vm.update_label_bundle(
        updated_bundle, 
        initial_version, 
        "human_expert_123", 
        VersionAction.CORRECTED, 
        "Human expert confirmed and improved confidence"
    )
    print(f"Created updated version: {updated_version.version_number}")
    
    # Get the current bundle
    current_bundle = vm.get_current_bundle(initial_bundle.label_id)
    if current_bundle:
        print(f"Current bundle has {len(current_bundle.therapeutic_response_labels)} therapeutic labels")
        print(f"First label confidence: {current_bundle.therapeutic_response_labels[0].metadata.confidence}")
        print(f"First label provenance: {current_bundle.therapeutic_response_labels[0].metadata.provenance.value}")
    
    # Get history
    history = vm.get_history(initial_bundle.label_id)
    if history:
        print(f"Total versions: {len(history.versions)}")
        print(f"Total provenance records: {len(history.provenance_records)}")
        
        # Get change summary
        summary = vm.get_label_change_summary(initial_bundle.label_id)
        print(f"Change summary: {summary}")
        
        # Use provenance analyzer
        analyzer = ProvenanceAnalyzer(vm)
        provenance_dist = analyzer.analyze_provenance_distribution()
        print(f"Provenance distribution: {provenance_dist}")


if __name__ == "__main__":
    test_label_versioning()