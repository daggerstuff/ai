"""Controller Orchestrator for the DeepRare multi-agent rare disease diagnosis system.

Enterprise-grade central controller agent that receives a patient case, decomposes
it into sub-tasks for specialized sub-agents, maintains a differential diagnosis
list with probability weights, and determines when to request additional tests
or specialist consultation.

Enterprise enhancements:
- Structured logging via Python ``logging`` with case_id correlation
- Clinical safety gates (red-flag detection, life-threatening condition protection)
- Per-iteration audit trail recording
- Configurable timeout per diagnosis
- Confidence interval propagation on hypotheses
- Convergence criteria: top-5 stable across ``convergence_window`` iterations OR
  all sub-agents agree on the top diagnosis

Based on the DeepRare architecture (arXiv 2506.20430).
"""

from __future__ import annotations

import logging
import time
from typing import TYPE_CHECKING, Any

from .agents.literature_matcher import LiteratureMatcher
from .agents.symptom_analyzer import SymptomAnalyzer
from .agents.test_interpreter import TestInterpreter
from .clinical_safety import (
    AuditAction,
    AuditTrail,
    ClinicalSafetyContext,
    ClinicalSafetyGate,
    RedFlagDetector,
)
from .differential import DifferentialDiagnosisManager
from .schema import (
    DiagnosisResult,
    DifferentialDiagnosis,
    Evidence,
    Hypothesis,
    PatientCase,
    RareDiseaseState,
)

if TYPE_CHECKING:
    from .knowledge_base import RareDiseaseKnowledgeBase

logger = logging.getLogger("deep_rare.orchestrator")


class ControllerOrchestrator:
    """Central controller agent orchestrating the multi-agent diagnosis pipeline.

    Workflow per iteration:
    1. Symptom Analyzer analyzes patient symptoms → generates hypotheses
    2. Test Interpreter interprets available tests → Bayesian updates
    3. Literature Matcher searches for matching case reports
    4. Differential Diagnosis Manager ranks and prunes hypotheses
    5. Check convergence → stop or request more information

    Enterprise features:
    - Clinical safety gates prevent elimination of life-threatening conditions
    - Audit trail records every agent action for regulatory compliance
    - Structured logging with case_id correlation
    - Configurable timeout per diagnosis
    - Red-flag detection for immediate referral scenarios
    """

    def __init__(
        self,
        kb: RareDiseaseKnowledgeBase,
        max_iterations: int = 10,
        convergence_window: int = 3,
        pruning_threshold: float = 0.01,
        timeout_seconds: float = 60.0,
        enable_safety_gates: bool = True,
        enable_audit_trail: bool = True,
        enable_red_flag_detection: bool = True,
    ) -> None:
        self._kb = kb
        self._symptom_analyzer = SymptomAnalyzer(kb)
        self._test_interpreter = TestInterpreter(kb)
        self._literature_matcher = LiteratureMatcher(kb)
        self._differential_manager = DifferentialDiagnosisManager(
            kb=kb,
            pruning_threshold=pruning_threshold,
        )
        self._max_iterations = max_iterations
        self._convergence_window = convergence_window
        self._timeout_seconds = timeout_seconds

        # Clinical safety components
        self._safety_gate: ClinicalSafetyGate | None = ClinicalSafetyGate() if enable_safety_gates else None
        self._red_flag_detector: RedFlagDetector | None = RedFlagDetector() if enable_red_flag_detection else None
        self._audit_trail: AuditTrail | None = AuditTrail() if enable_audit_trail else None
        self._safety_context: ClinicalSafetyContext | None = None
        if enable_safety_gates and enable_red_flag_detection and enable_audit_trail:
            self._safety_context = ClinicalSafetyContext(
                safety_gate=self._safety_gate,
                red_flag_detector=self._red_flag_detector,
                audit_trail=self._audit_trail,
            )

    def diagnose(self, case: PatientCase) -> DiagnosisResult:
        """Run the full diagnostic pipeline on a patient case.

        Orchestrates sub-agents across multiple iterations until convergence
        or max iterations reached. Enforces clinical safety gates, records
        audit trail entries, and respects configurable timeout.

        Args:
            case: Patient case with symptoms, history, and available tests.

        Returns:
            DiagnosisResult with differential, state, safety flags, and audit trail.
        """
        start_time = time.time()
        logger.info(
            "diagnosis_started",
            extra={"case_id": case.case_id, "urgency": case.clinical_urgency},
        )

        # Pre-diagnosis safety check — red flag detection
        safety_flags: list[dict[str, Any]] = []
        red_flags: list[Any] = []
        if self._red_flag_detector:
            red_flags = self._red_flag_detector.detect(case)
            if red_flags:
                for flag in red_flags:
                    safety_flags.append(flag.model_dump())
                    logger.warning(
                        "red_flag_detected",
                        extra={
                            "case_id": case.case_id,
                            "flag_type": flag.flag_type,
                            "urgency": flag.urgency,
                        },
                    )
                if self._red_flag_detector.should_block(red_flags):
                    logger.critical(
                        "diagnosis_blocked_by_red_flags",
                        extra={"case_id": case.case_id},
                    )
                    return self._blocked_result(case, red_flags, start_time, safety_flags)

        # Initialize state
        state = RareDiseaseState(
            max_iterations=self._max_iterations,
            convergence_window=self._convergence_window,
        )
        agent_traces: dict[str, str] = {}
        safety_violations: list[dict[str, Any]] = []

        # Reset safety gate for this diagnosis
        if self._safety_gate:
            self._safety_gate.reset_iteration()

        # Register life-threatening diseases as protected
        if self._safety_gate:
            for hyp_candidate in self._kb.search_by_symptoms([s.name for s in case.presenting_symptoms]):
                profile = self._kb.get_disease(hyp_candidate)
                if profile and self._is_life_threatening(profile):
                    self._safety_gate.register_protected_disease(hyp_candidate)

        # Record audit: diagnosis started
        self._record_audit(
            AuditAction.HYPOTHESIS_CREATED,
            "orchestrator",
            case.case_id,
            {"urgency": case.clinical_urgency, "symptom_count": len(case.presenting_symptoms)},
        )

        for iteration in range(self._max_iterations):
            state.iteration = iteration

            # Check timeout
            elapsed = time.time() - start_time
            if elapsed > self._timeout_seconds:
                logger.warning(
                    "diagnosis_timeout",
                    extra={"case_id": case.case_id, "elapsed": elapsed, "iteration": iteration},
                )
                self._record_audit(
                    AuditAction.ITERATION_COMPLETED,
                    "orchestrator",
                    case.case_id,
                    {"reason": "timeout", "iteration": iteration, "elapsed": elapsed},
                )
                break

            # Reset safety gate per-iteration elimination counter
            if self._safety_gate:
                self._safety_gate.reset_iteration()

            logger.debug(
                "iteration_started",
                extra={"case_id": case.case_id, "iteration": iteration},
            )

            # === Phase 1: Symptom Analysis ===
            symptom_result = self._symptom_analyzer.analyze(case)
            agent_traces["symptom_analyzer"] = symptom_result.reasoning

            self._record_audit(
                AuditAction.HYPOTHESIS_CREATED,
                "symptom_analyzer",
                case.case_id,
                {
                    "new_hypotheses": len(symptom_result.new_hypotheses),
                    "pathognomonic": len(symptom_result.identified_pathognomonic),
                },
            )

            if iteration == 0:
                for hyp in symptom_result.new_hypotheses:
                    state.add_hypothesis(hyp)
                    self._mark_life_threatening(hyp)
            else:
                for new_hyp in symptom_result.new_hypotheses:
                    existing = next(
                        (h for h in state.active_hypotheses if h.disease_name == new_hyp.disease_name),
                        None,
                    )
                    if existing:
                        existing.posterior_probability = max(
                            existing.posterior_probability,
                            new_hyp.posterior_probability,
                        )
                        for s in new_hyp.matching_symptoms:
                            if s not in existing.matching_symptoms:
                                existing.matching_symptoms.append(s)
                        # Propagate confidence intervals
                        if new_hyp.confidence_interval_lower > 0:
                            existing.confidence_interval_lower = max(
                                existing.confidence_interval_lower,
                                new_hyp.confidence_interval_lower,
                            )
                        if new_hyp.confidence_interval_upper > 0:
                            existing.confidence_interval_upper = max(
                                existing.confidence_interval_upper,
                                new_hyp.confidence_interval_upper,
                            )
                    elif new_hyp.posterior_probability > 0.05:
                        state.add_hypothesis(new_hyp)
                        self._mark_life_threatening(new_hyp)

            # Add evidence from pathognomonic symptoms
            for patho_symptom in symptom_result.identified_pathognomonic:
                for hyp in state.active_hypotheses:
                    profile = self._kb.get_disease(hyp.disease_name)
                    if profile and patho_symptom in profile.pathognomonic_symptoms:
                        hyp.add_evidence(
                            Evidence(
                                source="symptom_analyzer",
                                description=f"Pathognomonic symptom: {patho_symptom}",
                                supports=True,
                                weight=0.9,
                                confidence_level="very_high",
                                provenance="symptom_analysis",
                                evidence_grade="A",
                            )
                        )

            # === Phase 2: Test Interpretation ===
            test_result = self._test_interpreter.interpret(case, state.active_hypotheses)
            agent_traces["test_interpreter"] = test_result.reasoning

            self._record_audit(
                AuditAction.TEST_INTERPRETED,
                "test_interpreter",
                case.case_id,
                {
                    "updated": len(test_result.updated_probabilities),
                    "eliminated": len(test_result.eliminated_hypotheses),
                    "criteria_met": len(test_result.diagnostic_criteria_met),
                },
            )

            for disease_name, updated_prob in test_result.updated_probabilities.items():
                hyp = next(
                    (h for h in state.active_hypotheses if h.disease_name == disease_name),
                    None,
                )
                if hyp:
                    hyp.posterior_probability = updated_prob
                    if disease_name in test_result.eliminated_hypotheses:
                        # Safety gate check before elimination
                        can_eliminate, reason = (True, None)
                        if self._safety_gate:
                            can_eliminate, reason = self._safety_gate.can_eliminate(hyp, hyp.evidence_list)
                        if can_eliminate:
                            hyp.status = "eliminated"
                            state.eliminate(disease_name)
                        else:
                            safety_violations.append(
                                {
                                    "type": "elimination_blocked",
                                    "disease": disease_name,
                                    "reason": reason,
                                    "iteration": iteration,
                                }
                            )
                            logger.warning(
                                "elimination_blocked",
                                extra={"case_id": case.case_id, "disease": disease_name, "reason": reason},
                            )
                    for disease, criteria in test_result.diagnostic_criteria_met.items():
                        if disease == disease_name:
                            for crit in criteria:
                                hyp.add_evidence(
                                    Evidence(
                                        source="test_interpreter",
                                        description=f"Diagnostic criterion met: {crit}",
                                        supports=True,
                                        weight=0.8,
                                        confidence_level="high",
                                        provenance="test_interpretation",
                                        evidence_grade="B",
                                    )
                                )
                    # Safety gate check for confirmation
                    if self._safety_gate and hyp.posterior_probability > 0.85:
                        can_confirm, reason = self._safety_gate.can_confirm(hyp)
                        if can_confirm:
                            hyp.status = "confirmed"
                            self._record_audit(
                                AuditAction.HYPOTHESIS_CONFIRMED,
                                "test_interpreter",
                                case.case_id,
                                {"disease": disease_name, "probability": hyp.posterior_probability},
                            )

            # === Phase 3: Literature Matching ===
            lit_result = self._literature_matcher.search(case, state.active_hypotheses)
            agent_traces["literature_matcher"] = lit_result.reasoning

            self._record_audit(
                AuditAction.LITERATURE_RETRIEVED,
                "literature_matcher",
                case.case_id,
                {
                    "matches": len(lit_result.matches),
                    "best_score": max((m.similarity_score for m in lit_result.matches), default=0.0),
                },
            )

            for match in lit_result.matches[:5]:
                hyp = next(
                    (h for h in state.active_hypotheses if h.disease_name == match.matched_disease),
                    None,
                )
                if hyp:
                    hyp.add_evidence(
                        Evidence(
                            source="literature_matcher",
                            description=f"Literature match: {match.title} (sim={match.similarity_score:.2f})",
                            supports=True,
                            weight=match.similarity_score,
                            confidence_level="moderate" if match.similarity_score > 0.5 else "low",
                            provenance=match.title,
                            evidence_grade="C",
                        )
                    )
                    hyp.confidence_score = max(
                        hyp.confidence_score,
                        match.similarity_score,
                    )

            # === Phase 4: Differential Diagnosis Management ===
            self._differential_manager.update(state)

            self._record_audit(
                AuditAction.DIFFERENTIAL_UPDATED,
                "orchestrator",
                case.case_id,
                {
                    "active": len(state.active_hypotheses),
                    "eliminated": len(state.eliminated_conditions),
                    "iteration": iteration,
                },
            )

            # Record top hypotheses for convergence check
            state.record_top_hypotheses()

            # Check convergence
            if state.check_convergence():
                state.is_converged = True
                self._record_audit(
                    AuditAction.CONVERGENCE_REACHED,
                    "orchestrator",
                    case.case_id,
                    {"iteration": iteration, "window": self._convergence_window},
                )
                logger.info(
                    "convergence_reached",
                    extra={"case_id": case.case_id, "iteration": iteration},
                )
                break

            # Check if top hypothesis has high confidence
            top_hyp = (
                max(state.active_hypotheses, key=lambda h: h.posterior_probability) if state.active_hypotheses else None
            )
            if top_hyp and top_hyp.confidence_score > 0.85 and iteration >= 1:
                state.is_converged = True
                self._record_audit(
                    AuditAction.CONVERGENCE_REACHED,
                    "orchestrator",
                    case.case_id,
                    {"iteration": iteration, "reason": "high_confidence", "confidence": top_hyp.confidence_score},
                )
                logger.info(
                    "convergence_high_confidence",
                    extra={"case_id": case.case_id, "iteration": iteration, "confidence": top_hyp.confidence_score},
                )
                break

            self._record_audit(
                AuditAction.ITERATION_COMPLETED,
                "orchestrator",
                case.case_id,
                {"iteration": iteration, "active_hypotheses": len(state.active_hypotheses)},
            )

        # Build final differential diagnosis
        differential = self._differential_manager.build_differential(state)

        elapsed = time.time() - start_time
        next_steps = self._recommend_next_steps(state, differential)

        # Calculate clinical confidence
        clinical_confidence = self._calculate_clinical_confidence(differential, state, red_flags)

        # Determine if human review required
        requires_review = (
            not state.is_converged
            or clinical_confidence < 0.7
            or bool(safety_flags)
            or bool(safety_violations)
            or case.clinical_urgency in ("emergent", "life_threatening")
        )

        # Collect audit trail for this case
        audit_trail_data: list[dict[str, Any]] = []
        if self._audit_trail:
            audit_trail_data = [e.model_dump() for e in self._audit_trail.get_case_trail(case.case_id)]

        self._record_audit(
            AuditAction.DIAGNOSIS_FINALIZED,
            "orchestrator",
            case.case_id,
            {
                "iterations": state.iteration + 1,
                "converged": state.is_converged,
                "elapsed": elapsed,
                "clinical_confidence": clinical_confidence,
            },
        )

        logger.info(
            "diagnosis_completed",
            extra={
                "case_id": case.case_id,
                "iterations": state.iteration + 1,
                "converged": state.is_converged,
                "elapsed": elapsed,
                "confidence": clinical_confidence,
            },
        )

        return DiagnosisResult(
            case_id=case.case_id,
            differential=differential,
            state=state,
            iterations=state.iteration + 1,
            time_seconds=elapsed,
            converged=state.is_converged,
            agent_outputs=agent_traces,
            recommended_next_steps=next_steps,
            evaluation=None,
            safety_flags=safety_flags,
            audit_trail=audit_trail_data,
            safety_violations=safety_violations,
            clinical_confidence=clinical_confidence,
            requires_human_review=requires_review,
            phi_deidentified=case.phi_protected,
        )

    def _recommend_next_steps(
        self,
        state: RareDiseaseState,
        differential: DifferentialDiagnosis,
    ) -> list[str]:
        """Generate clinical next-step recommendations."""
        steps: list[str] = []
        top = differential.top_disease()
        if top:
            steps.append(f"Primary diagnostic consideration: {top.disease_name} (P={top.probability:.2f})")
        if state.pending_inquiries:
            steps.append(f"Pending inquiries: {', '.join(state.pending_inquiries[:3])}")
        if len(differential.ranked_list) > 1 and differential.ranked_list[0].probability < 0.5:
            steps.append("Consider additional genetic testing to narrow differential.")
        if top and top.is_pathognomonic_match:
            steps.append(f"Pathognomonic symptom matched for {top.disease_name} — high-confidence diagnostic lead.")
        if not state.is_converged:
            steps.append("Convergence not achieved — recommend specialist consultation.")
        return steps

    def _single_agent_baseline(self, case: PatientCase) -> DifferentialDiagnosis:
        """Run a single-agent baseline (symptom analysis only, no multi-agent)."""
        symptom_result = self._symptom_analyzer.analyze(case)
        state = RareDiseaseState(max_iterations=1, convergence_window=1)
        for hyp in symptom_result.new_hypotheses:
            state.add_hypothesis(hyp)
            self._mark_life_threatening(hyp)
        state.iteration = 0
        state.record_top_hypotheses()
        state.is_converged = True
        return self._differential_manager.build_differential(state)

    def _record_audit(
        self,
        action: AuditAction,
        agent_name: str,
        case_id: str,
        details: dict[str, Any],
        evidence_refs: list[str] | None = None,
    ) -> None:
        """Record an audit trail entry if audit is enabled."""
        if self._audit_trail:
            self._audit_trail.record(
                action=action,
                agent_name=agent_name,
                case_id=case_id,
                details=details,
                evidence_refs=evidence_refs or [],
            )

    def _mark_life_threatening(self, hyp: Hypothesis) -> None:
        """Mark a hypothesis as life-threatening based on KB profile."""
        profile = self._kb.get_disease(hyp.disease_name)
        if profile and self._is_life_threatening(profile):
            hyp.is_life_threatening = True

    @staticmethod
    def _is_life_threatening(profile: Any) -> bool:
        """Check if a disease profile indicates a life-threatening condition."""
        life_threatening_organs = {"cardiovascular", "neurological", "metabolic"}
        if hasattr(profile, "organ_system") and profile.organ_system in life_threatening_organs:
            # Check for specific life-threatening symptoms
            lt_symptoms = {"cardiomyopathy", "heart failure", "respiratory failure", "seizure", "sudden cardiac death"}
            all_symptoms = set(profile.pathognomonic_symptoms + profile.common_symptoms)
            if all_symptoms & lt_symptoms:
                return True
        return False

    @staticmethod
    def _calculate_clinical_confidence(
        differential: DifferentialDiagnosis,
        state: RareDiseaseState,
        red_flags: list[Any],
    ) -> float:
        """Calculate overall clinical confidence score for the diagnosis.

        Factors:
        - Top diagnosis probability
        - Convergence achieved
        - Evidence count
        - Red flag presence (reduces confidence)
        """
        if not differential.ranked_list:
            return 0.0
        top = differential.ranked_list[0]
        base = top.probability * 0.5
        convergence_bonus = 0.2 if state.is_converged else 0.0
        evidence_bonus = min(0.2, top.evidence_count * 0.04)
        pathognomonic_bonus = 0.1 if top.is_pathognomonic_match else 0.0
        red_flag_penalty = min(0.3, len(red_flags) * 0.1)
        return max(0.0, min(1.0, base + convergence_bonus + evidence_bonus + pathognomonic_bonus - red_flag_penalty))

    @staticmethod
    def _blocked_result(
        case: PatientCase,
        red_flags: list[Any],
        start_time: float,
        safety_flags: list[dict[str, Any]],
    ) -> DiagnosisResult:
        """Create a DiagnosisResult when diagnosis is blocked by red flags."""
        elapsed = time.time() - start_time
        state = RareDiseaseState(max_iterations=0, convergence_window=0)
        steps = [
            f"IMMEDIATE REFERRAL REQUIRED: {len(red_flags)} red flag(s) detected.",
            "Do not proceed with routine differential diagnosis.",
        ]
        for flag in red_flags:
            steps.append(f"  - {flag.flag_type}: {flag.description}")

        return DiagnosisResult(
            case_id=case.case_id,
            differential=DifferentialDiagnosis(
                ranked_list=[],
                eliminated=[],
                total_hypotheses_considered=0,
                iterations_used=0,
                convergence_achieved=False,
                reasoning_trace="Diagnosis blocked by red flag detection",
            ),
            state=state,
            iterations=0,
            time_seconds=elapsed,
            converged=False,
            agent_outputs={"orchestrator": "Blocked by red flag detection"},
            recommended_next_steps=steps,
            evaluation=None,
            safety_flags=safety_flags,
            audit_trail=[],
            safety_violations=[{"type": "red_flag_block", "count": len(red_flags)}],
            clinical_confidence=0.0,
            requires_human_review=True,
            phi_deidentified=case.phi_protected,
        )
