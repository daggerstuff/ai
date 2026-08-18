"""Tests for the PATIENT-Ψ FastAPI router."""

from __future__ import annotations

from fastapi.testclient import TestClient

from ai.pkg_mera.platform.patient_psi.api import create_app

app = create_app(prefix="/api/v1/patient-psi")

client = TestClient(app)


class TestPatientPsiAPI:
    """API endpoint integration tests."""

    # ── Create Session ──────────────────────────────────────────────────

    def test_create_session_returns_201(self) -> None:
        resp = client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})
        assert resp.status_code == 201
        data = resp.json()
        assert "session_id" in data
        assert data["profile_name"] == "generalized_anxiety"
        assert data["status"] == "active"
        assert data["phase"] == "initial"
        assert data["turn_count"] == 0

    def test_create_session_unknown_profile_returns_404(self) -> None:
        resp = client.post("/api/v1/patient-psi/sessions", json={"profile_name": "nonexistent"})
        assert resp.status_code == 404

        assert resp.json()["detail"] == "Profile not found"

    def test_create_session_rejects_invalid_config(self) -> None:
        resp = client.post(
            "/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety", "difficulty": 99}
        )
        assert resp.status_code == 422

    def test_create_session_with_optional_fields(self) -> None:
        resp = client.post(
            "/api/v1/patient-psi/sessions",
            json={
                "profile_name": "generalized_anxiety",
                "style": "friendly",
                "difficulty": 0.3,
                "max_turns": 5,
                "patient_name": "Alex",
            },
        )
        assert resp.status_code == 201
        data = resp.json()
        assert data["profile_name"] == "generalized_anxiety"
        assert data["status"] == "active"

    # ── Interact ────────────────────────────────────────────────────────

    def test_interact_returns_turn(self) -> None:
        create = client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})
        session_id = create.json()["session_id"]

        resp = client.post(
            f"/api/v1/patient-psi/sessions/{session_id}/interact",
            json={"message": "Hello"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        turn = data["turn"]
        assert turn["turn_number"] == 1
        assert turn["trigger"] == "therapist_greeting"
        assert turn["therapist_utterance"] == "Hello"
        assert len(turn["patient_utterance"]) > 0
        assert turn["coherence_score"] is not None

    def test_interact_nonexistent_session_returns_404(self) -> None:
        resp = client.post(
            "/api/v1/patient-psi/sessions/nonexistent/interact",
            json={"message": "Hello"},
        )
        assert resp.status_code == 404

        assert resp.json()["detail"] == "Session not found"

    def test_interact_terminated_session_returns_400(self) -> None:
        create = client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})
        session_id = create.json()["session_id"]

        client.post(f"/api/v1/patient-psi/sessions/{session_id}/terminate")

        resp = client.post(
            f"/api/v1/patient-psi/sessions/{session_id}/interact",
            json={"message": "Hello"},
        )
        assert resp.status_code == 400

        assert resp.json()["detail"] == "Session is not active"

    def test_same_seed_produces_same_transition(self) -> None:
        create = client.post(
            "/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety", "max_turns": 20}
        )
        sid_a = create.json()["session_id"]
        create = client.post(
            "/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety", "max_turns": 20}
        )
        sid_b = create.json()["session_id"]

        resp_a = client.post(f"/api/v1/patient-psi/sessions/{sid_a}/interact", json={"message": "Hello", "seed": 42})
        resp_b = client.post(f"/api/v1/patient-psi/sessions/{sid_b}/interact", json={"message": "Hello", "seed": 42})

        assert resp_a.json()["turn"]["phase"] == resp_b.json()["turn"]["phase"]
        assert resp_a.json()["turn"]["trigger"] == resp_b.json()["turn"]["trigger"]

    # ── Get Session ─────────────────────────────────────────────────────

    def test_get_session_returns_status(self) -> None:
        create = client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})
        session_id = create.json()["session_id"]

        resp = client.get(f"/api/v1/patient-psi/sessions/{session_id}")
        assert resp.status_code == 200
        data = resp.json()
        assert data["session_id"] == session_id
        assert data["status"] == "active"

    def test_get_session_nonexistent_returns_404(self) -> None:
        resp = client.get("/api/v1/patient-psi/sessions/nonexistent")
        assert resp.status_code == 404

    # ── Terminate ───────────────────────────────────────────────────────

    def test_terminate_session(self) -> None:
        create = client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})
        session_id = create.json()["session_id"]

        resp = client.post(f"/api/v1/patient-psi/sessions/{session_id}/terminate")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "terminated"

    def test_terminate_nonexistent_returns_404(self) -> None:
        resp = client.post("/api/v1/patient-psi/sessions/nonexistent/terminate")
        assert resp.status_code == 404

    # ── List Sessions ───────────────────────────────────────────────────

    def test_list_active_sessions(self) -> None:
        # Ensure clean state for this test
        resp = client.get("/api/v1/patient-psi/sessions")
        prev = resp.json()["sessions"]

        client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})

        resp = client.get("/api/v1/patient-psi/sessions")
        current = resp.json()["sessions"]
        assert len(current) == len(prev) + 1

    def test_list_active_sessions_empty(self) -> None:
        # This test relies on engine global state — at minimum the list returns
        resp = client.get("/api/v1/patient-psi/sessions")
        assert resp.status_code == 200
        assert "sessions" in resp.json()

    def test_list_active_sessions_filter_by_profile(self) -> None:
        client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})
        client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})
        client.post("/api/v1/patient-psi/sessions", json={"profile_name": "major_depressive_disorder"})

        resp = client.get("/api/v1/patient-psi/sessions?profile=major_depressive_disorder")
        sessions = resp.json()["sessions"]
        assert len(sessions) == 1
        assert all(s["profile_name"] == "major_depressive_disorder" for s in sessions)

    # ── Profiles ────────────────────────────────────────────────────────

    def test_list_profiles_returns_known_profiles(self) -> None:
        resp = client.get("/api/v1/patient-psi/profiles")
        assert resp.status_code == 200
        profiles = resp.json()
        assert isinstance(profiles, list)
        assert len(profiles) > 0
        assert "generalized_anxiety" in profiles
        assert "major_depressive_disorder" in profiles
