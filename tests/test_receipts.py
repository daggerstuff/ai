"""R1: Cryptographic receipts for audit trail.

Test asserts deterministic ReceiptEnvelope + Merkle-root Ledger.
"""

from __future__ import annotations

import hashlib

from ai.receipts.receipt import Ledger, ReceiptEnvelope


def _sha256(data: str) -> str:
    return hashlib.sha256(data.encode()).hexdigest()


def test_receipt_envelope_deterministic() -> None:
    """Identical inputs produce identical receipts."""
    prev_hash = _sha256("genesis")
    model_fp = _sha256("model-v1-weights")
    prompt_hash = _sha256("prompt content")
    output_hash = _sha256("output content")
    fhe_hash = _sha256("fhe-ciphertext")

    r1 = ReceiptEnvelope.compute(
        prev_hash=prev_hash,
        model_fingerprint=model_fp,
        prompt_hash=prompt_hash,
        output_hash=output_hash,
        fhe_ciphertext_hash=fhe_hash,
    )
    r2 = ReceiptEnvelope.compute(
        prev_hash=prev_hash,
        model_fingerprint=model_fp,
        prompt_hash=prompt_hash,
        output_hash=output_hash,
        fhe_ciphertext_hash=fhe_hash,
    )

    assert r1.receipt_hash == r2.receipt_hash


def test_receipt_chain_breaks_on_bitflip() -> None:
    """Single-bit change in any input breaks receipt chain."""
    prev_hash = _sha256("genesis")
    model_fp = _sha256("model-v1-weights")
    prompt_hash = _sha256("prompt content")
    output_hash = _sha256("output content")
    fhe_hash = _sha256("fhe-ciphertext")

    r1 = ReceiptEnvelope.compute(
        prev_hash=prev_hash,
        model_fingerprint=model_fp,
        prompt_hash=prompt_hash,
        output_hash=output_hash,
        fhe_ciphertext_hash=fhe_hash,
    )

    # Flip one bit in output_hash
    flipped = _sha256("output content\x00")

    r2 = ReceiptEnvelope.compute(
        prev_hash=prev_hash,
        model_fingerprint=model_fp,
        prompt_hash=prompt_hash,
        output_hash=flipped,
        fhe_ciphertext_hash=fhe_hash,
    )

    assert r1.receipt_hash != r2.receipt_hash


def test_ledger_merkle_root_reproducible() -> None:
    """Ledger.append returns same root_hash for identical receipt sequence."""
    ledger1 = Ledger()
    ledger2 = Ledger()

    prev = "0" * 64
    for i in range(5):
        r = ReceiptEnvelope.compute(
            prev_hash=prev,
            model_fingerprint=_sha256(f"model-{i}"),
            prompt_hash=_sha256(f"prompt-{i}"),
            output_hash=_sha256(f"output-{i}"),
            fhe_ciphertext_hash=_sha256(f"fhe-{i}"),
        )
        prev = r.receipt_hash
        ledger1.append(r)
        ledger2.append(r)

    assert ledger1.root_hash() == ledger2.root_hash()


def test_ledger_root_changes_on_new_receipt() -> None:
    """Appending a new receipt changes root_hash."""
    ledger = Ledger()
    r1 = ReceiptEnvelope.compute(
        prev_hash="0" * 64,
        model_fingerprint=_sha256("model"),
        prompt_hash=_sha256("prompt"),
        output_hash=_sha256("output"),
        fhe_ciphertext_hash=_sha256("fhe"),
    )
    ledger.append(r1)
    root1 = ledger.root_hash()

    r2 = ReceiptEnvelope.compute(
        prev_hash=r1.receipt_hash,
        model_fingerprint=_sha256("model"),
        prompt_hash=_sha256("prompt"),
        output_hash=_sha256("output"),
        fhe_ciphertext_hash=_sha256("fhe"),
    )
    ledger.append(r2)
    root2 = ledger.root_hash()

    assert root1 != root2
