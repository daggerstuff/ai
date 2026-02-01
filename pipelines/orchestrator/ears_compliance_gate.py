"""Minimal EARS compliance gate.

This module implements a stub compliance gate that initially rejects
all inputs to represent a failing state. It will be hardened later
to enforce the >95% sensitivity requirement.
"""

class EarsComplianceGate:
    """
    Minimal implementation of an EARS compliance validation gate.
    """

    def validate_compliance(self, data: dict) -> bool:
        """
        Stub validate_compliance method.
        Currently always returns False to represent a failing gate.
        """
        return False