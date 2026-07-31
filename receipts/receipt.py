"""R1: Cryptographic receipts for audit trail.

Deterministic ReceiptEnvelope + Merkle-root Ledger.
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field
from typing import Self


@dataclass(frozen=True, slots=True)
class ReceiptEnvelope:
    """Immutable receipt for a single inference turn."""

    prev_hash: str
    model_fingerprint: str
    prompt_hash: str
    output_hash: str
    fhe_ciphertext_hash: str
    receipt_hash: str = field(init=False)

    def __post_init__(self) -> None:
        # Deterministic hash over all fields
        content = (
            self.prev_hash + self.model_fingerprint + self.prompt_hash + self.output_hash + self.fhe_ciphertext_hash
        )
        object.__setattr__(self, "receipt_hash", hashlib.sha256(content.encode()).hexdigest())

    @classmethod
    def compute(
        cls,
        prev_hash: str,
        model_fingerprint: str,
        prompt_hash: str,
        output_hash: str,
        fhe_ciphertext_hash: str,
    ) -> Self:
        return cls(
            prev_hash=prev_hash,
            model_fingerprint=model_fingerprint,
            prompt_hash=prompt_hash,
            output_hash=output_hash,
            fhe_ciphertext_hash=fhe_ciphertext_hash,
        )

    def to_json(self) -> str:
        return json.dumps(self.to_dict(), sort_keys=True)

    def to_dict(self) -> dict[str, str]:
        return {
            "prev_hash": self.prev_hash,
            "model_fingerprint": self.model_fingerprint,
            "prompt_hash": self.prompt_hash,
            "output_hash": self.output_hash,
            "fhe_ciphertext_hash": self.fhe_ciphertext_hash,
            "receipt_hash": self.receipt_hash,
        }

    @classmethod
    def from_dict(cls, data: dict[str, str]) -> Self:
        """Rebuild a receipt from its serialized fields.

        The stored ``receipt_hash`` is cross-checked against a recomputation;
        a mismatch raises ``ValueError`` (tamper detection on load).
        """
        receipt = cls(
            prev_hash=data["prev_hash"],
            model_fingerprint=data["model_fingerprint"],
            prompt_hash=data["prompt_hash"],
            output_hash=data["output_hash"],
            fhe_ciphertext_hash=data["fhe_ciphertext_hash"],
        )
        stored_hash = data.get("receipt_hash")
        if stored_hash is not None and stored_hash != receipt.receipt_hash:
            raise ValueError(f"receipt_hash mismatch for {receipt.receipt_hash!r}")
        return receipt

    @classmethod
    def from_json(cls, raw: str) -> Self:
        return cls.from_dict(json.loads(raw))


class Ledger:
    """Append-only Merkle-root ledger of receipts."""

    def __init__(self) -> None:
        self._receipts: list[ReceiptEnvelope] = []
        self._leaves: list[str] = []

    def append(self, receipt: ReceiptEnvelope) -> None:
        self._receipts.append(receipt)
        self._leaves.append(receipt.receipt_hash)

    def root_hash(self) -> str:
        if not self._leaves:
            return hashlib.sha256(b"empty-ledger").hexdigest()

        leaves = self._leaves[:]
        while len(leaves) > 1:
            if len(leaves) % 2 == 1:
                leaves.append(leaves[-1])
            next_level = []
            for i in range(0, len(leaves), 2):
                combined = leaves[i] + leaves[i + 1]
                next_level.append(hashlib.sha256(combined.encode()).hexdigest())
            leaves = next_level
        return leaves[0]

    def __len__(self) -> int:
        return len(self._receipts)

    def __iter__(self):
        return iter(self._receipts)

    def verify_chain(self) -> bool:
        """Verify hash integrity and prev-hash linkage of every receipt.

        Genesis receipt may carry an arbitrary ``prev_hash`` (callers use
        ``"0"*64``); every later receipt must chain to its predecessor.
        An empty ledger is vacuously valid.
        """
        previous_hash: str | None = None
        for receipt in self._receipts:
            recomputed = ReceiptEnvelope.compute(
                prev_hash=receipt.prev_hash,
                model_fingerprint=receipt.model_fingerprint,
                prompt_hash=receipt.prompt_hash,
                output_hash=receipt.output_hash,
                fhe_ciphertext_hash=receipt.fhe_ciphertext_hash,
            )
            if recomputed.receipt_hash != receipt.receipt_hash:
                return False
            if previous_hash is not None and receipt.prev_hash != previous_hash:
                return False
            previous_hash = receipt.receipt_hash
        return True
