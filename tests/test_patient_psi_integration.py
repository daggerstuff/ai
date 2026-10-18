"""Integration tests for PATIENT-Ψ — engine + API + Starlette mount.

Tests cover:
- Patient-psi API via ``create_app()`` (isolated sub-app)
- Multi-turn session lifecycle through the full engine → API pipeline
- Seeded deterministic flows across multiple endpoints
- Safe error handling (404, 400, 422)
- No route conflicts with the main Starlette app
"""

from __future__ import annotations

from fastapi.testclient import TestClient

from ai.pkg_mera.platform.patient_psi.api import create_app

# ── Helpers ───────────────────────────────────────────────────────────


def _make_client() -> TestClient:
    """Return a TestClient for the standalone patient-psi FastAPI app."""
    app = create_app(prefix="/api/v1/patient-psi")
    return TestClient(app)


def _draw(seed: int) -> tuple[str, str, str]:
    """Deterministically create a session, interact, return (session_id, phase, trigger)."""
    client = _make_client()
    create = client.post(
        "/api/v1/patient-psi/sessions",
        json={"profile_name": "generalized_anxiety", "max_turns": 20},
    )
    assert create.status_code == 201
    session_id = create.json()["session_id"]

    interact = client.post(
        f"/api/v1/patient-psi/sessions/{session_id}/interact",
        json={"message": "Hello", "seed": seed},
    )
    turn = interact.json()["turn"]
    return session_id, turn["phase"], turn["trigger"]


# ── Integration tests ─────────────────────────────────────────────────


class TestPatientPsiIntegration:
    """End-to-end: create → interact → list → terminate → verify."""

    def test_full_session_lifecycle(self) -> None:
        """Create a session, interact multiple times, list, terminate."""
        client = _make_client()

        # Create
        create = client.post(
            "/api/v1/patient-psi/sessions",
            json={"profile_name": "major_depressive_disorder", "style": "melancholic", "difficulty": 0.5},
        )
        assert create.status_code == 201
        session_id = create.json()["session_id"]

        # Interact × 3
        for i, msg in enumerate(["Hello", "Tell me more", "How does that make you feel?"], 1):
            resp = client.post(
                f"/api/v1/patient-psi/sessions/{session_id}/interact",
                json={"message": msg},
            )
            assert resp.status_code == 200, f"interact {i} failed: {resp.text}"
            turn = resp.json()["turn"]
            assert turn["turn_number"] == i
            assert len(turn["patient_utterance"]) > 0

        # List — the created session is active
        listing = client.get("/api/v1/patient-psi/sessions")
        assert listing.status_code == 200
        ids = [s["session_id"] for s in listing.json()["sessions"]]
        assert session_id in ids

        # Terminate
        term = client.post(f"/api/v1/patient-psi/sessions/{session_id}/terminate")
        assert term.status_code == 200
        assert term.json()["status"] == "terminated"

        # Interact after termination → 400
        after = client.post(
            f"/api/v1/patient-psi/sessions/{session_id}/interact",
            json={"message": "Hello"},
        )
        assert after.status_code == 400

    def test_seeded_determinism(self) -> None:
        """Same seed → same phase + trigger across independent sessions."""
        _, phase_a, trigger_a = _draw(42)
        _, phase_b, trigger_b = _draw(42)
        assert phase_a == phase_b
        assert trigger_a == trigger_b

    def test_different_seed_different_outcome(self) -> None:
        """Different seeds → can produce different phase/trigger."""
        outcomes = {_draw(s)[1:] for s in [1, 2, 3, 4, 5]}  # (phase, trigger)
        # At least 2 distinct outcomes across 5 seeds
        assert len(outcomes) >= 2

    def test_list_active_sessions_filter_by_profile(self) -> None:
        """Filtering by profile returns only matching sessions."""
        client = _make_client()

        client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})
        client.post("/api/v1/patient-psi/sessions", json={"profile_name": "generalized_anxiety"})
        client.post("/api/v1/patient-psi/sessions", json={"profile_name": "major_depressive_disorder"})

        resp = client.get("/api/v1/patient-psi/sessions?profile=generalized_anxiety")
        assert resp.status_code == 200
        sessions = resp.json()["sessions"]
        assert len(sessions) >= 2
        assert all(s["profile_name"] == "generalized_anxiety" for s in sessions)

    def test_list_profiles_returns_all(self) -> None:
        """All 20+ profiles are listed with no duplicates."""
        client = _make_client()
        resp = client.get("/api/v1/patient-psi/profiles")
        assert resp.status_code == 200
        profiles = resp.json()
        assert len(profiles) >= 20
        assert len(profiles) == len(set(profiles))  # no dupes
        assert "generalized_anxiety" in profiles
        assert "major_depressive_disorder" in profiles

    def test_error_unknown_profile_404(self) -> None:
        """Creating a session with a nonexistent profile yields 404."""
        client = _make_client()
        resp = client.post("/api/v1/patient-psi/sessions", json={"profile_name": "made_up_123"})
        assert resp.status_code == 404

    def test_error_invalid_config_422(self) -> None:
        """Invalid difficulty value yields 422."""
        client = _make_client()
        resp = client.post(
            "/api/v1/patient-psi/sessions",
            json={"profile_name": "generalized_anxiety", "difficulty": 5},
        )
        assert resp.status_code == 422

    def test_error_nonexistent_session_404(self) -> None:
        """GET/POST on a nonexistent session yields 404."""
        client = _make_client()

        resp = client.get("/api/v1/patient-psi/sessions/does-not-exist")
        assert resp.status_code == 404

        resp = client.post(
            "/api/v1/patient-psi/sessions/does-not-exist/interact",
            json={"message": "Hi"},
        )
        assert resp.status_code == 404


# ── Starlette mount tests ─────────────────────────────────────────────


class TestStarletteMount:
    """Patient-psi sub-app mounted under Starlette — route isolation."""

    def test_patient_psi_routes_accessible(self) -> None:
        """Endpoints respond through the mounted sub-app."""
        from ai.api.index import app as starlette_app

        client = TestClient(starlette_app)

        # Profiles (fast, no session needed)
        resp = client.get("/api/v1/patient-psi/profiles")
        assert resp.status_code == 200
        assert "generalized_anxiety" in resp.json()

    def test_health_still_works(self) -> None:
        """Existing Starlette routes are unaffected by the mount."""
        from ai.api.index import app as starlette_app

        client = TestClient(starlette_app)
        resp = client.get("/health")
        assert resp.status_code == 200

    def test_root_still_works(self) -> None:
        """Root route is unaffected."""
        from ai.api.index import app as starlette_app

        client = TestClient(starlette_app)
        resp = client.get("/")
        assert resp.status_code == 200

    def test_no_prefix_conflict(self) -> None:
        """Mounted routes don't shadow main app routes."""
        from ai.api.index import app as starlette_app

        client = TestClient(starlette_app)
        # The health endpoint is at /health — not a sub-path of /api/v1/patient-psi
        resp = client.get("/nonexistent")
        assert resp.status_code in (404, 405)
