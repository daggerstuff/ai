"""Quadit — four-persona adversarial audit for clinical AI content.

A generic audit core: three clinical judge personas plus one or more
adversarial auditor personas (loaded from TOML descriptors such as
``brene_brown.toml``). The core is content-source-agnostic — adapters
feed it :class:`~ai.research.quadit.models.AuditItem` sequences, so the
same judges can audit pe AI responses, training-dataset records, or
synthetic-corpus months.

History: ported from the hackathon corpus workspace
(``~/hackathon/corpus/pixelated_empathy``) where the 3-persona monthly
adversarial review (voice fidelity / clinical accuracy / training signal)
and the Brené Brown ``quadit`` auditor were first built.
"""

from ai.research.quadit.dataset_gate import (
    DatasetGateError,
    audit_dataset_records,
    chatml_to_audit_item,
    gate_should_block,
    record_to_audit_item,
)
from ai.research.quadit.models import (
    AuditItem,
    Finding,
    PersonaVerdict,
    QuadAuditReport,
)
from ai.research.quadit.personas import (
    CLINICAL_ACCURACY_JUDGE,
    TRAINING_SIGNAL_JUDGE,
    VOICE_FIDELITY_JUDGE,
    load_auditor_descriptor,
)
from ai.research.quadit.review import QuaditLLMClient, run_quadit_audit
from ai.research.quadit.rubric import PASS_THRESHOLD, SEVERITY_ORDER, severity_weight

__all__ = [
    "CLINICAL_ACCURACY_JUDGE",
    "PASS_THRESHOLD",
    "SEVERITY_ORDER",
    "TRAINING_SIGNAL_JUDGE",
    "VOICE_FIDELITY_JUDGE",
    "AuditItem",
    "DatasetGateError",
    "Finding",
    "PersonaVerdict",
    "QuadAuditReport",
    "QuaditLLMClient",
    "audit_dataset_records",
    "chatml_to_audit_item",
    "gate_should_block",
    "load_auditor_descriptor",
    "record_to_audit_item",
    "run_quadit_audit",
    "severity_weight",
]
