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


def _default_generator(domain_gap=None, difficulty=None):
    """Default scenario generator, resolved lazily to keep heavy imports out."""
    from training.nightmare_fuel_generator import generate_nightmare_scenario

    return generate_nightmare_scenario(domain_gap=domain_gap, difficulty=difficulty)


class JITScenarioInjector:
    """Generates and injects a JIT nightmare scenario for a clinician."""

    def __init__(self, generator=None):
        self._generator = generator if generator is not None else _default_generator

    def inject_for_decision(self, decision, *, domain_gap=None, difficulty=None):
        """Generate + inject a scenario block when the decision triggers.

        Returns the stored block dict on success, ``None`` on no-trigger or
        any failure (foresight missing, generator error, block error).
        """
        if not decision.should_trigger:
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
            block = registry.create_block(
                JIT_SCENARIO_BLOCK_LABEL,
                content=f"[JIT scenario top-up for clinician {clinician}]\n\n{scenario}",
            )
            registry.set_block(JIT_SCENARIO_BLOCK_LABEL, block)
            return block.to_dict()
        except Exception:
            logger.warning("Failed to inject JIT scenario block", exc_info=True)
            return None


_injector = JITScenarioInjector()


def inject_for_decision(decision, *, domain_gap=None, difficulty=None):
    """Module-level convenience wrapper around a shared ``JITScenarioInjector``."""
    return _injector.inject_for_decision(decision, domain_gap=domain_gap, difficulty=difficulty)
