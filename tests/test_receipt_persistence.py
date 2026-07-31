"""R5/INT-5: Receipt-ledger persistence + audit export tests."""

import hashlib

import pytest

from ai.receipts.persistence import (
    GENESIS_PREV_HASH,
    PersistentLedger,
    PostgresReceiptStore,
    SQLiteReceiptStore,
    get_persistent_ledger,
)
from ai.receipts.receipt import Ledger, ReceiptEnvelope


def _sha256(value: str) -> str:
    return hashlib.sha256(value.encode()).hexdigest()


def _receipt(prev_hash: str = GENESIS_PREV_HASH, payload: str = "turn") -> ReceiptEnvelope:
    return ReceiptEnvelope.compute(
        prev_hash=prev_hash,
        model_fingerprint="model-v1",
        prompt_hash=_sha256(f"prompt-{payload}"),
        output_hash=_sha256(f"output-{payload}"),
        fhe_ciphertext_hash=_sha256(f"cipher-{payload}"),
    )


def _chain(length: int = 5) -> list[ReceiptEnvelope]:
    receipts = []
    prev = GENESIS_PREV_HASH
    for i in range(length):
        receipt = _receipt(prev_hash=prev, payload=f"turn-{i}")
        receipts.append(receipt)
        prev = receipt.receipt_hash
    return receipts


# --- ReceiptEnvelope round-trip / tamper detection --------------------------


def test_envelope_to_dict_from_dict_roundtrip():
    receipt = _receipt()
    rebuilt = ReceiptEnvelope.from_dict(receipt.to_dict())
    assert rebuilt == receipt
    assert rebuilt.receipt_hash == receipt.receipt_hash


def test_envelope_from_dict_rejects_tampered_hash():
    data = _receipt().to_dict()
    data["receipt_hash"] = "0" * 64
    with pytest.raises(ValueError, match="receipt_hash mismatch"):
        ReceiptEnvelope.from_dict(data)


def test_envelope_from_json_roundtrip():
    receipt = _receipt(payload="json")
    assert ReceiptEnvelope.from_json(receipt.to_json()) == receipt


# --- Ledger.verify_chain ----------------------------------------------------


def test_verify_chain_valid():
    ledger = Ledger()
    for receipt in _chain(5):
        ledger.append(receipt)
    assert ledger.verify_chain() is True


def test_verify_chain_empty_is_valid():
    assert Ledger().verify_chain() is True


def test_verify_chain_detects_broken_prev_hash_link():
    receipts = _chain(3)
    # Re-link receipt 2 to a stale previous hash (chain discontinuity).
    receipts[2] = _receipt(prev_hash=GENESIS_PREV_HASH, payload="turn-2")
    ledger = Ledger()
    for receipt in receipts:
        ledger.append(receipt)
    assert ledger.verify_chain() is False


def test_verify_chain_detects_tampered_field():
    receipts = _chain(3)
    ledger = Ledger()
    for receipt in receipts:
        ledger.append(receipt)
    # Mutate the stored envelope's field hash (frozen dataclass → rebuild).
    tampered = ReceiptEnvelope.compute(
        prev_hash=receipts[1].prev_hash,
        model_fingerprint=receipts[1].model_fingerprint,
        prompt_hash=_sha256("tampered-prompt"),
        output_hash=receipts[1].output_hash,
        fhe_ciphertext_hash=receipts[1].fhe_ciphertext_hash,
    )
    ledger._receipts[1] = tampered
    ledger._leaves[1] = tampered.receipt_hash
    assert ledger.verify_chain() is False


# --- SQLiteReceiptStore -----------------------------------------------------


def test_sqlite_append_load_roundtrip(tmp_path):
    store = SQLiteReceiptStore(tmp_path / "ledger.db")
    receipts = _chain(3)
    for receipt in receipts:
        store.append(receipt)
    assert store.count() == 3
    assert store.load_all() == receipts


def test_sqlite_persists_across_reopen(tmp_path):
    db_path = tmp_path / "ledger.db"
    store = SQLiteReceiptStore(db_path)
    store.append(_receipt(payload="persist"))
    assert SQLiteReceiptStore(db_path).count() == 1


def test_sqlite_insert_or_ignore_duplicate(tmp_path):
    store = SQLiteReceiptStore(tmp_path / "ledger.db")
    receipt = _receipt()
    store.append(receipt)
    store.append(receipt)
    assert store.count() == 1


def test_sqlite_load_order_preserved(tmp_path):
    store = SQLiteReceiptStore(tmp_path / "ledger.db")
    for receipt in _chain(4):
        store.append(receipt)
    loaded = store.load_all()
    assert [r.receipt_hash for r in loaded] == [r.receipt_hash for r in _chain(4)]


# --- PersistentLedger -------------------------------------------------------


def test_persistent_ledger_writes_through_and_restores(tmp_path):
    store = SQLiteReceiptStore(tmp_path / "ledger.db")
    ledger = PersistentLedger(store)
    for receipt in _chain(5):
        ledger.append(receipt)

    restored = PersistentLedger(SQLiteReceiptStore(tmp_path / "ledger.db"))
    assert restored.restore() == 5
    assert restored.root_hash() == ledger.root_hash()
    assert restored.verify_chain() is True
    assert len(restored) == 5


def test_persistent_ledger_restore_empty(tmp_path):
    ledger = PersistentLedger(SQLiteReceiptStore(tmp_path / "ledger.db"))
    assert ledger.restore() == 0
    assert ledger.root_hash() == Ledger().root_hash()
    assert ledger.verify_chain() is True


# --- Audit export -----------------------------------------------------------


def test_export_audit_jsonl_roundtrip(tmp_path):
    store = SQLiteReceiptStore(tmp_path / "ledger.db")
    ledger = PersistentLedger(store)
    for receipt in _chain(3):
        ledger.append(receipt)

    audit_path = tmp_path / "audit.jsonl"
    payload = ledger.export_audit(audit_path)
    assert payload == audit_path.read_text(encoding="utf-8")
    lines = [line for line in payload.splitlines() if line]
    assert len(lines) == 3
    rebuilt = [ReceiptEnvelope.from_json(line) for line in lines]
    assert [r.receipt_hash for r in rebuilt] == [r.receipt_hash for r in _chain(3)]


def test_export_audit_empty(tmp_path):
    ledger = PersistentLedger(SQLiteReceiptStore(tmp_path / "ledger.db"))
    assert ledger.export_audit() == ""


# --- PostgresReceiptStore (driver-gated) ------------------------------------


def test_postgres_store_requires_dsn(monkeypatch):
    monkeypatch.delenv("RECEIPT_LEDGER_POSTGRES_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    with pytest.raises(ValueError, match="RECEIPT_LEDGER_POSTGRES_URL or DATABASE_URL"):
        PostgresReceiptStore()


def test_postgres_store_requires_driver(monkeypatch):
    import sys

    # sys.modules None-entry makes `import psycopg[2]` raise ImportError.
    monkeypatch.setitem(sys.modules, "psycopg", None)
    monkeypatch.setitem(sys.modules, "psycopg2", None)
    monkeypatch.setenv("RECEIPT_LEDGER_POSTGRES_URL", "postgres://localhost:5432/ledger")
    with pytest.raises(ImportError, match="requires 'psycopg'"):
        PostgresReceiptStore()


# --- get_persistent_ledger --------------------------------------------------


def test_get_persistent_ledger_defaults_to_sqlite(monkeypatch, tmp_path):
    import ai.receipts.persistence as persistence

    monkeypatch.setenv("RECEIPT_LEDGER_DB_PATH", str(tmp_path / "ledger.db"))
    monkeypatch.delenv("RECEIPT_LEDGER_POSTGRES_URL", raising=False)
    monkeypatch.delenv("DATABASE_URL", raising=False)
    ledger = get_persistent_ledger()
    assert isinstance(ledger, PersistentLedger)
    assert isinstance(ledger.store, SQLiteReceiptStore)
