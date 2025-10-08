"""Small smoke runner for LocalFileConnector that avoids pytest/conftest.

Run with: uv run python ai/dataset_pipeline/tests/smoke_local_connector.py
"""
import sys
import tempfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))

from ai.dataset_pipeline.ingestion_interface import IngestRecord, LocalFileConnector


class DummyQuarantine:
    def quarantine_record(self, rec, errors):
        pass


def dummy_validate(record: IngestRecord):
    # Accept all records for this smoke run; simulate schema normalization by returning the same object
    return record


def main():
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "a.bin"
        p.write_bytes(b"hello world")

        # Monkeypatch minimal functions used by connector
        from ai.dataset_pipeline import quarantine, validation

        validation.validate_record = lambda rec: rec
        quarantine.get_quarantine_store = lambda: DummyQuarantine()

        conn = LocalFileConnector(directory=td, rate_limit={"capacity": 1, "refill_rate": 0.5})
        conn.connect()
        for _rec in conn.fetch():
            pass
        conn.close()


if __name__ == "__main__":
    main()
