"""Tests for build_arc_dpo.py — DPO pair builder (Phase B pairing)."""

import json
import sys
from pathlib import Path
from typing import Any

import pytest

_HERE = Path(__file__).resolve()
for _p in (_HERE.parents[1], _HERE.parents[2]):
    if str(_p) not in sys.path:
        sys.path.insert(0, str(_p))

import training.build_arc_dpo as bd  # noqa: E402
from training.generate_arc_corpus import SYSTEM_PROMPT, build_session_prompt  # noqa: E402

LEDGER = {
    "dx": "none",
    "def": "none",
    "soma": "none",
    "risk": "none",
    "hx": "none",
    "onset": "not stated",
    "track": "none",
    "tx": "none",
    "tl": "now: opening (told)",
}

PLAN = {
    "arc_id": "arc_9001",
    "title": "Test Arc",
    "client": {"name": "J", "age": 34, "occupation": "welder", "speech_style": "terse", "notes": "some notes"},
    "sessions": [
        {"n": 1, "turns": 6, "gap_before": None, "focus": "opening"},
        {"n": 2, "turns": 6, "gap_before": "one week", "focus": "work"},
    ],
    "timeline": [{"anchor": "now", "event": "stuck", "provenance": "told"}],
    "surface_subject": "schedule trouble",
    "real_subject": {"session": 2, "surfaces_around_turn": 4, "content": "grief"},
    "beats": [],
    "ending": {"session": 2, "turn": 10, "requirement": "concrete plan"},
}


SHA256_HEX_LEN = 64


def _turns(tag: str, n_pairs: int = 3, with_ledger: bool = True) -> list[dict[str, Any]]:
    turns: list[dict[str, Any]] = []
    for i in range(n_pairs):
        turns.append({"role": "client", "content": f"{tag} client {i}"})
        t: dict[str, Any] = {"role": "therapist", "content": f"{tag} therapist {i}"}
        if with_ledger:
            t["ledger"] = dict(LEDGER, tl=f"now: {tag} {i} (told)")
        turns.append(t)
    return turns


def _accept_record(arc: str, flags: list[dict[str, Any]] | None = None) -> dict[str, Any]:
    return {"arc_id": arc, "audit": {"verdict": "accept", "flags_final": flags or []}}


def _hr_record(arc: str) -> dict[str, Any]:
    return {"arc_id": arc, "audit": {"verdict": "hr", "flags_final": []}}


class TestGateHeads:
    def test_single_gate_head(self) -> None:
        assert bd._gate_heads("tl_drift: 2 ledgers dropped") == ["tl_drift"]

    def test_multi_hit_garbage_fragments_skipped(self) -> None:
        # arc_0039_s1_a1 shape: second fragment is a bare anchor list.
        header = "tl_drift: 2 ledgers dropped earlier anchors: ['now']; ['-4y']"
        assert bd._gate_heads(header) == ["tl_drift"]

    def test_gate_detail_semicolon_does_not_create_fake_head(self) -> None:
        header = (
            "cliche_gate: 2 banned-phrase hits: "
            "[banned_parroting_opener: 'you said'] line one; "
            "[banned_parroting_opener: 'you said'] line two"
        )
        assert bd._gate_heads(header) == ["cliche_gate"]

    def test_multiple_real_heads_deduped_in_order(self) -> None:
        header = "tl_drift: 1 x; spoken_specificity: 1 y; tl_drift: 1 z"
        assert bd._gate_heads(header) == ["tl_drift", "spoken_specificity"]

    def test_unknown_header_falls_back_to_unknown(self) -> None:
        assert bd._gate_heads("mystery: stuff") == ["unknown"]


class TestLengthOk:
    def test_within_bounds(self) -> None:
        assert bd._length_ok("x" * 100, "y" * 80)

    def test_too_short_rejected(self) -> None:
        assert not bd._length_ok("x" * 100, "y" * 10)

    def test_too_long_rejected(self) -> None:
        assert not bd._length_ok("x" * 100, "y" * 250)

    def test_empty_rejected_rejected(self) -> None:
        assert not bd._length_ok("x" * 100, "")


class TestRenderSession:
    def test_client_and_therapist_with_ledger(self) -> None:
        out = bd._render_session(_turns("t", 1))
        assert out is not None
        assert out.startswith("[C] t client 0")
        assert "[T|THINK] " in out
        assert '"tl"' in out
        assert "[T] t therapist 0" in out

    def test_therapist_without_ledger_is_allowed(self) -> None:
        out = bd._render_session(_turns("t", 1, with_ledger=False))
        assert out is not None
        assert "[T|THINK]" not in out
        assert "[T] t therapist 0" in out

    def test_unknown_role_returns_none(self) -> None:
        assert bd._render_session([{"role": "nurse", "content": "hi"}]) is None

    def test_empty_content_returns_none(self) -> None:
        assert bd._render_session([{"role": "client", "content": "  "}]) is None

    def test_empty_turn_list_returns_none(self) -> None:
        assert bd._render_session([]) is None


class TestFlagSessions:
    def test_groups_by_session(self) -> None:
        record = _accept_record(
            "arc_1",
            flags=[
                {"session": 1, "category": "fabrication"},
                {"session": 1, "category": "fabrication"},
                {"session": 2, "category": "tl_drift"},
                {"session": None, "category": "junk"},
            ],
        )
        assert bd._flag_sessions(record) == {1: ["fabrication", "fabrication"], 2: ["tl_drift"]}

    def test_empty_flags(self) -> None:
        assert bd._flag_sessions(_accept_record("arc_1")) == {}


def _write_checkpoint(path: Path, rows: list[dict[str, Any]]) -> None:
    path.write_text("\n".join(json.dumps(r) for r in rows) + "\n", encoding="utf-8")


class TestSessionPairs:
    def _current(self) -> dict[tuple[str, int], dict[str, Any]]:
        return {
            ("arc_9001", 1): {"turns": _turns("good", 3), "writer_model": "w1"},
            ("arc_9001", 2): {"turns": _turns("g2", 3), "writer_model": "w1"},
        }

    def test_accept_only_and_defect_metadata(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr(bd, "_load_plan", lambda _arc_id: dict(PLAN))
        snap = tmp_path / ".pre_test"
        snap.mkdir()
        _write_checkpoint(
            snap / "sessions_checkpoint.jsonl",
            [
                {"arc_id": "arc_9001", "session_n": 1, "turns": _turns("bad", 3)},
                {"arc_id": "arc_9002", "session_n": 1, "turns": _turns("hr", 3)},
            ],
        )
        records = {
            "arc_9001": _accept_record("arc_9001", flags=[{"session": 1, "category": "fabrication"}]),
            "arc_9002": _hr_record("arc_9002"),
        }
        pairs = bd._session_pairs(records, self._current(), snap / "sessions_checkpoint.jsonl", True)
        assert len(pairs) == 1
        p = pairs[0]
        assert p["chosen"].startswith("[C] good")
        assert p["rejected"].startswith("[C] bad")
        assert p["metadata"]["mode"] == "session"
        assert p["metadata"]["arc_id"] == "arc_9001"
        assert p["metadata"]["defect"] == ["fabrication"]
        assert p["metadata"]["rejected_source"] == ".pre_test"

    def test_identical_transcripts_skipped(self, tmp_path: Path) -> None:
        snap = tmp_path / ".pre_same"
        snap.mkdir()
        turns = _turns("same", 3)
        _write_checkpoint(snap / "sessions_checkpoint.jsonl", [{"arc_id": "arc_9001", "session_n": 1, "turns": turns}])
        records = {"arc_9001": _accept_record("arc_9001")}
        current = {("arc_9001", 1): {"turns": turns, "writer_model": "w"}}
        assert bd._session_pairs(records, current, snap / "sessions_checkpoint.jsonl", True) == []

    def test_length_ratio_out_of_bounds_skipped(self, tmp_path: Path) -> None:
        snap = tmp_path / ".pre_short"
        snap.mkdir()
        _write_checkpoint(
            snap / "sessions_checkpoint.jsonl",
            [
                {"arc_id": "arc_9001", "session_n": 1, "turns": [{"role": "client", "content": "tiny"}]},
            ],
        )
        records = {"arc_9001": _accept_record("arc_9001")}
        pairs = bd._session_pairs(records, self._current(), snap / "sessions_checkpoint.jsonl", True)
        assert pairs == []


class TestAttemptPairs:
    def _run(
        self,
        tmp_path: Path,
        monkeypatch: pytest.MonkeyPatch,
        files: dict[str, str],
        current: dict[tuple[str, int], dict[str, Any]],
    ) -> list[dict[str, Any]]:
        fdir = tmp_path / "failed_attempts"
        fdir.mkdir()
        for name, body in files.items():
            (fdir / name).write_text(body, encoding="utf-8")
        monkeypatch.setattr(bd, "FAILED_ATTEMPTS_DIR", fdir)
        monkeypatch.setattr(bd, "_load_plan", lambda _arc_id: dict(PLAN))
        return bd._attempt_pairs({"arc_9001": _accept_record("arc_9001")}, current, True)

    def test_defect_heads_from_multihit_header(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        current = {("arc_9001", 2): {"turns": _turns("good", 3), "writer_model": "w"}}
        rejected_body = bd._render_session(_turns("bad", 3))
        assert rejected_body is not None
        body = (
            "# gates: tl_drift: 2 ledgers dropped earlier anchors: "
            "['now']; ['-4y']; spoken_specificity: 1 untraceable "
            "speech items: t8 spoken 'five mornings a week'\n" + rejected_body + "\n"
        )
        pairs = self._run(tmp_path, monkeypatch, {"arc_9001_s2_a1.txt": body}, current)
        assert len(pairs) == 1
        assert pairs[0]["metadata"]["defect"] == ["tl_drift", "spoken_specificity"]
        assert pairs[0]["metadata"]["attempt"] == 1
        assert pairs[0]["metadata"]["rejected_source"] == ("failed_attempts/arc_9001_s2_a1.txt")
        assert pairs[0]["prompt"].startswith("You author")
        assert "[C] bad client 0" in pairs[0]["rejected"]

    def test_non_accept_or_missing_session_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        current = {("arc_9001", 2): {"turns": _turns("good", 3), "writer_model": "w"}}
        rejected_body = bd._render_session(_turns("bad", 3))
        assert rejected_body is not None
        body = f"# gates: cliche_gate: 1 hit: x\n{rejected_body}\n"
        monkeypatch.setattr(bd, "_load_plan", lambda _arc_id: dict(PLAN))
        fdir = tmp_path / "fa"
        fdir.mkdir()
        monkeypatch.setattr(bd, "FAILED_ATTEMPTS_DIR", fdir)
        (fdir / "arc_9002_s1_a1.txt").write_text(body, encoding="utf-8")
        (fdir / "arc_9001_s3_a1.txt").write_text(body, encoding="utf-8")
        pairs = bd._attempt_pairs(
            {"arc_9002": _hr_record("arc_9002"), "arc_9001": _accept_record("arc_9001")}, current, True
        )
        assert pairs == []

    def test_rejected_equal_to_chosen_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        chosen = bd._render_session(_turns("good", 3))
        current = {("arc_9001", 1): {"turns": _turns("good", 3), "writer_model": "w"}}
        pairs = self._run(
            tmp_path, monkeypatch, {"arc_9001_s1_a1.txt": f"# gates: cliche_gate: 1 hit\n{chosen}\n"}, current
        )
        assert pairs == []

    def test_malformed_filename_skipped(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        current = {("arc_9001", 1): {"turns": _turns("good", 3), "writer_model": "w"}}
        pairs = self._run(tmp_path, monkeypatch, {"nonsense.txt": "# gates: x\n[C] bad\n"}, current)
        assert pairs == []


class TestDpoHash:
    def test_case_insensitive_and_stable(self) -> None:
        a = bd._dpo_hash("Prompt", "A", "B")
        b = bd._dpo_hash("prompt", "a", "b")
        assert a == b
        assert len(a) == SHA256_HEX_LEN


class TestWritePairs:
    def test_append_and_idempotency(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        out = tmp_path / "pairs.jsonl"
        monkeypatch.setattr(bd, "PAIRS_PATH", out)
        pair = {"prompt": "p", "chosen": "c", "rejected": "r", "metadata": {"mode": "session"}}
        assert bd._write_pairs([pair, dict(pair)], dry_run=False)["emitted"] == 1
        assert out.read_text().count("\n") == 1
        assert bd._write_pairs([dict(pair)], dry_run=False)["emitted"] == 0
        assert bd._write_pairs([dict(pair)], dry_run=True)["emitted"] == 0


def test_module_imports_offline(monkeypatch: pytest.MonkeyPatch) -> None:
    """Import must not touch network or require fastembed."""
    monkeypatch.setenv("PYTHONDONTWRITEBYTECODE", "1")
    assert SYSTEM_PROMPT
    assert build_session_prompt is not None
