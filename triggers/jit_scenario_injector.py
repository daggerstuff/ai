"""JIT scenario top-up injection for flagged clinicians.

When the foresight JIT trigger engine decides a clinician should be offered
additional training (``TriggerDecision.should_trigger is True``), this module
calls the Nightmare Fuel scenario generator with a targeted spec (domain gap +
difficulty) and injects the resulting scenario into the clinician's next
session via the foresight block system (block label ``jit_scenario_topup``,
injected pre-prompt).

``foresight`` is an optional integration: if it is unavailable at runtime this
module degrades to a no-op and never breaks the calling flow.
"""

import importlib
import logging
from dataclasses import dataclass

logger = logging.getLogger(__name__)

JIT_SCENARIO_BLOCK_LABEL = "jit_scenario_topup"
JIT_SCENARIO_INJECTION_POINT = "pre_prompt"
DEFAULT_DIFFICULTY = "high"
DEFAULT_DOMAIN_GAP = "resistant patient with complex comorbidities"


@dataclass(frozen=True, slots=True)
class TriggerDecision:
    """Decoupled mirror of ``foresight.foresight.triggers.TriggerDecision``."""

    should_trigger: bool
    matching_flags: int = 0
    clinician_id: str | None = None


def _load_foresight_block_system():
    """Lazily import the foresight block registry (None when unavailable)."""
    try:
        return importlib.import_module("foresight.block_registry")
    except ImportError:
        logger.debug("foresight.block_registry unavailable; JIT scenario injection disabled")
        return None


def _load_foresight_event_bus():
    """Lazily import the foresight EventBus (None when unavailable)."""
    try:
        return importlib.import_module("foresight.event_bus")
    except ImportError:
        logger.debug("foresight.event_bus unavailable; EventBus subscription disabled")
        return None


def _default_generator(domain_gap=None, difficulty=None):
    """Default scenario generator, resolved lazily to keep heavy imports out."""
    from training.nightmare_fuel_generator import generate_nightmare_scenario

    return generate_nightmare_scenario(domain_gap=domain_gap, difficulty=difficulty)


class JITScenarioInjector:
    """Generates and injects a JIT nightmare scenario for a clinician.

    Supports two usage modes:
    1. Direct: call ``inject_for_decision()`` with a TriggerDecision
    2. EventBus: call ``subscribe_to_event_bus()`` to auto-inject on trigger events
    """

    def __init__(self, generator=None):
        self._generator = generator if generator is not None else _default_generator
        self._event_bus_subscribed = False

    def subscribe_to_event_bus(self, event_bus=None):
        """Subscribe to EventBus trigger events for automatic injection.

        When a BIAS_THRESHOLD_EXCEEDED or CRISIS_THRESHOLD_EXCEEDED event arrives,
        automatically evaluate and inject a scenario if the threshold is met.

        Args:
            event_bus: Optional EventBus instance. If None, attempts to load from foresight.

        Returns:
            True if subscription succeeded, False if EventBus unavailable.
        """
        if event_bus is None:
            event_bus_mod = _load_foresight_event_bus()
            if event_bus_mod is None:
                return False
            event_bus = event_bus_mod.get_event_bus()

        if event_bus is None:
            return False

        try:
            event_bus_mod = _load_foresight_event_bus()
            if event_bus_mod:
                event_bus.subscribe(event_bus_mod.EventType.BIAS_THRESHOLD_EXCEEDED, self._on_trigger_event)
                event_bus.subscribe(event_bus_mod.EventType.CRISIS_THRESHOLD_EXCEEDED, self._on_trigger_event)
                self._event_bus_subscribed = True
                logger.info("JITScenarioInjector subscribed to EventBus trigger events")
                return True
        except Exception:
            logger.warning("Failed to subscribe to EventBus", exc_info=True)
        return False

    def _on_trigger_event(self, event):
        """EventBus handler: convert event to TriggerDecision and inject."""
        payload = event.payload
        clinician_id = payload.get("clinician_id") or payload.get("user_id") or "unknown"
        bias_score = payload.get("overall_bias_score", 0.0)

        decision = TriggerDecision(
            should_trigger=True,
            matching_flags=1,
            clinician_id=clinician_id,
        )

        domain_gap = payload.get("detected_biases", [None])[0] if payload.get("detected_biases") else None
        difficulty = "critical" if bias_score >= 0.7 else "high"

        self.inject_for_decision(decision, domain_gap=domain_gap, difficulty=difficulty)

    def inject_for_decision(self, decision, *, domain_gap=None, difficulty=None, session_context=None):
        """Generate + inject a scenario block when the decision triggers.

        Args:
            decision: TriggerDecision with should_trigger=True to proceed
            domain_gap: Optional domain gap for targeted scenario generation
            difficulty: Optional difficulty level ("high", "critical")
            session_context: Optional dict with session metadata for HIPAA gating.
                             Must contain ``is_training_session=True`` to allow injection.

        Returns:
            The stored block dict on success, ``None`` on no-trigger or
            any failure (foresight missing, generator error, block error, HIPAA gate).
        """
        if not decision.should_trigger:
            return None

        if not self._check_hipaa_gate(session_context):
            return None

        try:
            scenario = self._generator(
                domain_gap=domain_gap or DEFAULT_DOMAIN_GAP,
                difficulty=difficulty or DEFAULT_DIFFICULTY,
            )
        except Exception:
            logger.warning("Failed to generate JIT nightmare scenario", exc_info=True)
            return None

        block_system = _load_foresight_block_system()
        if block_system is None:
            logger.warning("foresight block system unavailable; JIT scenario not injected")
            return None

        try:
            registry = block_system.get_registry()
            if registry.get_schema(JIT_SCENARIO_BLOCK_LABEL) is None:
                registry.register(
                    block_system.MemoryBlockSchema(
                        label=JIT_SCENARIO_BLOCK_LABEL,
                        description="JIT nightmare-fuel scenario top-up for next session",
                        retention_policy=block_system.RetentionPolicy.EPHEMERAL,
                        merge_strategy=block_system.MergeStrategy.REPLACE,
                        injection_point=block_system.InjectionPoint.PRE_PROMPT,
                        scope=block_system.BlockScope.SESSION,
                    )
                )
            clinician = decision.clinician_id or "unknown"
            block_content = (
                f"[JIT scenario top-up for clinician {clinician}]\n[is_training_edge_case: true]\n\n{scenario}"
            )
            block = registry.create_block(
                JIT_SCENARIO_BLOCK_LABEL,
                content=block_content,
            )
            registry.set_block(JIT_SCENARIO_BLOCK_LABEL, block)
            logger.info(
                "JIT scenario injected",
                clinician=clinician,
                domain_gap=domain_gap,
                difficulty=difficulty,
            )
            return block.to_dict()
        except Exception:
            logger.warning("Failed to inject JIT scenario block", exc_info=True)
            return None

    def _check_hipaa_gate(self, session_context):
        """HIPAA gating: never inject in non-training (real patient) sessions.

        Args:
            session_context: Dict with session metadata. Must contain
                             ``is_training_session=True`` to allow injection.

        Returns:
            True if injection is allowed, False if blocked by HIPAA gate.
        """
        if session_context is None:
            logger.debug("No session context provided; allowing injection (assume training)")
            return True

        is_training = session_context.get("is_training_session", False)
        if not is_training:
            logger.info(
                "HIPAA gate: JIT scenario injection blocked for non-training session",
                session_id=session_context.get("session_id"),
            )
            return False
        return True


_injector = JITScenarioInjector()


def inject_for_decision(decision, *, domain_gap=None, difficulty=None, session_context=None):
    """Module-level convenience wrapper around a shared ``JITScenarioInjector``."""
    return _injector.inject_for_decision(
        decision, domain_gap=domain_gap, difficulty=difficulty, session_context=session_context
    )


def subscribe_to_event_bus(event_bus=None):
    """Module-level convenience wrapper for EventBus subscription."""
    return _injector.subscribe_to_event_bus(event_bus)
