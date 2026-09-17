"""Tests for PIX-4382 placeholder completions.

Covers:
- ai.research.foresight_local_retention.scope_metadata (RetainScope kwargs
  and object form, used by the MCP foresight retain route)
- ai.inference.services.dream.consolidation.list_active_users
  (whitelist / Redis / in-memory sources)
- ai.models.artifacts.multimodal_fusion.TextToSpeechGenerator
  (real HF text-to-audio synthesis, env overrides, graceful degradation)
"""

from __future__ import annotations

import sys
import types
import wave
from unittest.mock import MagicMock

import anyio
import numpy as np
import pytest
from flask import Flask

import ai.inference.services.dream.consolidation as dream_consolidation
from ai.inference.services.dream.consolidation import _dream_user_whitelist
from ai.models.artifacts.multimodal_fusion import TextToSpeechGenerator
from ai.research.foresight_local_retention import (
    RetainScope,
    scope_metadata,
)

# Standard output of the default MMS TTS model
MMS_SAMPLING_RATE = 16000

# Minimum sample count the default model emits for a short sentence
MIN_TTS_SAMPLES = 1000

# Fixture WAV length
TTS_TEST_SAMPLES = 3200

# Minimum live-TTS duration (seconds) for the short sample sentence
MIN_TTS_DURATION_S = 0.1


# ---------------------------------------------------------------------------
# foresight_local_retention.scope_metadata
# ---------------------------------------------------------------------------


class TestScopeMetadata:
    def test_unscoped_returns_empty(self) -> None:
        assert scope_metadata() == {}

    def test_kwargs_form_produces_scope_keys(self) -> None:
        metadata = scope_metadata(
            org_id="org-1",
            project_id="proj-1",
            session_id="sess-1",
            agent_id="agent-1",
            run_id="run-1",
        )
        assert metadata == {
            "org_id": "org-1",
            "project_id": "proj-1",
            "session_id": "sess-1",
            "agent_id": "agent-1",
            "run_id": "run-1",
            "visibility": "private",
        }

    def test_visibility_explicit_shared(self) -> None:
        metadata = scope_metadata(org_id="org-1", visibility="shared")
        assert metadata == {"org_id": "org-1", "visibility": "shared"}

    def test_visibility_alone_is_recorded(self) -> None:
        assert scope_metadata(visibility="org") == {"visibility": "org"}

    def test_object_form_matches_kwargs_form(self) -> None:
        via_object = scope_metadata(
            RetainScope(org_id="org-1", session_id="sess-1", visibility="private")
        )
        via_kwargs = scope_metadata(
            org_id="org-1", session_id="sess-1", visibility="private"
        )
        assert via_object == via_kwargs

    def test_route_shape_accepts_header_kwargs(self) -> None:
        """The MCP retain route calls scope_metadata(org_id=..., ...) —
        previously a TypeError against the empty stub."""
        metadata = scope_metadata(
            org_id=None,
            project_id=None,
            session_id="sess-1",
            agent_id=None,
            run_id=None,
            visibility=None,
        )
        assert metadata == {"session_id": "sess-1", "visibility": "private"}


# ---------------------------------------------------------------------------
# dream consolidation — list_active_users
# ---------------------------------------------------------------------------


@pytest.fixture
def memory_store(monkeypatch: pytest.MonkeyPatch) -> dict:
    """Force the in-memory store and expose it for assertions."""
    store: dict = {}
    monkeypatch.setattr(dream_consolidation, "_redis_client", None)
    if not hasattr(dream_consolidation, "_dream_cycles"):
        # The module defines _dream_cycles only when REDIS_URL is unset at
        # import time; create it for the process-local path under test.
        monkeypatch.setattr(
            dream_consolidation, "_dream_cycles", store, raising=False
        )
    else:
        monkeypatch.setattr(dream_consolidation, "_dream_cycles", store)
    return store


@pytest.fixture
def flask_app():
    app = Flask(__name__)
    app.register_blueprint(dream_consolidation.dream_bp)
    return app


class TestListActiveUsers:
    def test_whitelist_wins_over_redis(
        self, monkeypatch: pytest.MonkeyPatch, flask_app
    ) -> None:
        monkeypatch.setenv("DREAM_USER_WHITELIST", "user-a, user-b ,,user-c")
        fake_redis = MagicMock()
        fake_redis.smembers.return_value = {"redis-user"}
        monkeypatch.setattr(dream_consolidation, "_redis_client", fake_redis)

        body = flask_app.test_client().get("/api/dream/users").get_json()
        assert body["success"] is True
        assert body["users"] == ["user-a", "user-b", "user-c"]
        assert body["source"] == "whitelist"
        fake_redis.smembers.assert_not_called()

    def test_redis_source_lists_dream_users(
        self, monkeypatch: pytest.MonkeyPatch, flask_app
    ) -> None:
        monkeypatch.delenv("DREAM_USER_WHITELIST", raising=False)
        fake_redis = MagicMock()
        fake_redis.smembers.return_value = {"user-1", "user-2"}
        monkeypatch.setattr(dream_consolidation, "_redis_client", fake_redis)

        body = flask_app.test_client().get("/api/dream/users").get_json()
        assert body["success"] is True
        assert body["source"] == "redis"
        assert body["users"] == ["user-1", "user-2"]

    def test_memory_source_from_user_keys(
        self, memory_store: dict, flask_app
    ) -> None:
        memory_store["dream:user:u-1"] = []
        memory_store["dream:user:u-2"] = ["dream-1"]

        body = flask_app.test_client().get("/api/dream/users").get_json()
        assert body["success"] is True
        assert body["source"] == "memory"
        assert body["users"] == ["u-1", "u-2"]

    def test_memory_source_from_dream_records(
        self, memory_store: dict, flask_app
    ) -> None:
        memory_store["dream-1"] = {"user_id": "u-9", "status": "completed"}

        body = flask_app.test_client().get("/api/dream/users").get_json()
        assert body["users"] == ["u-9"]

    def test_memory_source_deduplicates(
        self, memory_store: dict, flask_app
    ) -> None:
        memory_store["dream:user:u-1"] = ["dream-1"]
        memory_store["dream-2"] = {"user_id": "u-1"}

        body = flask_app.test_client().get("/api/dream/users").get_json()
        assert body["users"] == ["u-1"]

    def test_no_longer_reports_not_implemented(
        self, flask_app
    ) -> None:
        body = flask_app.test_client().get("/api/dream/users").get_json()
        assert "not yet implemented" not in body["message"]


class TestDreamUserWhitelist:
    def test_unset_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.delenv("DREAM_USER_WHITELIST", raising=False)
        assert _dream_user_whitelist() is None

    def test_empty_string_returns_none(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DREAM_USER_WHITELIST", "")
        assert _dream_user_whitelist() is None

    def test_parses_and_trims(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("DREAM_USER_WHITELIST", " a , b,,c ")
        assert _dream_user_whitelist() == ["a", "b", "c"]


# ---------------------------------------------------------------------------
# multimodal_fusion — TextToSpeechGenerator
# ---------------------------------------------------------------------------


def _stub_transformers(
    monkeypatch: pytest.MonkeyPatch, stub_pipeline: MagicMock
) -> None:
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = MagicMock(return_value=stub_pipeline)
    fake_transformers.AutoFeatureExtractor = MagicMock()
    fake_transformers.AutoModelForSequenceClassification = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


def _stub_transformers_factory(
    monkeypatch: pytest.MonkeyPatch, pipeline_factory: MagicMock
) -> None:
    """Stub where the pipeline factory itself raises (load-time failure)."""
    fake_transformers = types.ModuleType("transformers")
    fake_transformers.pipeline = pipeline_factory
    fake_transformers.AutoFeatureExtractor = MagicMock()
    fake_transformers.AutoModelForSequenceClassification = MagicMock()
    monkeypatch.setitem(sys.modules, "transformers", fake_transformers)


class TestTextToSpeechGenerator:
    def test_real_model_synthesizes_audio(self) -> None:
        """Live integration: the default MMS TTS model produces real PCM."""
        generator = TextToSpeechGenerator()

        result = anyio.run(lambda: generator.synthesize("Hello there.", "session-tts-1"))

        assert result["status"] == "success"
        assert result["sampling_rate"] == MMS_SAMPLING_RATE
        audio = result["audio"]
        # MMS produces a non-trivial amount of samples for a short sentence
        assert len(audio) > MIN_TTS_SAMPLES
        assert float(result["audio_duration_s"]) > MIN_TTS_DURATION_S

    def test_synthesize_returns_audio_and_metadata(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stub = MagicMock(
            return_value={
                "audio": np.array([0.1, -0.2, 0.3], dtype=np.float32),
                "sampling_rate": MMS_SAMPLING_RATE,
            }
        )
        _stub_transformers(monkeypatch, stub)
        monkeypatch.setenv("TTS_MODEL", "stub/tts-model")
        generator = TextToSpeechGenerator(device="cpu")

        result = anyio.run(
            lambda: generator.synthesize(
                "All right.", "session-1", emotional_state={"valence": 0.6}
            )
        )

        assert result["status"] == "success"
        assert result["audio"].shape == (3,)
        assert result["sampling_rate"] == MMS_SAMPLING_RATE
        assert result["audio_duration_s"] == pytest.approx(3 / MMS_SAMPLING_RATE)
        assert result["model"] == "stub/tts-model"
        assert result["emotional_prosody"] == {"valence": 0.6}
        assert "audio_path" not in result

    def test_empty_text_returns_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        _stub_transformers(monkeypatch, MagicMock())
        generator = TextToSpeechGenerator(device="cpu")

        result = anyio.run(lambda: generator.synthesize("", "session-1"))
        assert result == {"error": "Empty text"}

    def test_model_load_failure_degrades_gracefully(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The factory itself raises — surfacing the failure at load time
        _stub_transformers_factory(
            monkeypatch, MagicMock(side_effect=RuntimeError("weights missing"))
        )
        generator = TextToSpeechGenerator(device="cpu")

        result = anyio.run(lambda: generator.synthesize("Text.", "session-1"))
        assert result["status"] == "unavailable"
        assert "failed to load" in result["reason"]

    def test_synthesis_failure_returns_error(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The pipeline constructs but raises at synthesis (call) time
        failing_synth = MagicMock(side_effect=RuntimeError("weights missing"))
        _stub_transformers(monkeypatch, failing_synth)
        generator = TextToSpeechGenerator(device="cpu")

        result = anyio.run(lambda: generator.synthesize("Text.", "session-1"))
        assert result == {"error": "weights missing"}

    def test_output_dir_writes_wav(
        self, monkeypatch: pytest.MonkeyPatch, tmp_path
    ) -> None:
        stub = MagicMock(
            return_value={
                "audio": np.zeros(TTS_TEST_SAMPLES, dtype=np.float32),
                "sampling_rate": MMS_SAMPLING_RATE,
            }
        )
        _stub_transformers(monkeypatch, stub)
        monkeypatch.setenv("TTS_OUTPUT_DIR", str(tmp_path))
        generator = TextToSpeechGenerator(device="cpu")

        result = anyio.run(lambda: generator.synthesize("Write me.", "session-9"))

        audio_path = result.get("audio_path")
        assert audio_path is not None
        with wave.open(str(audio_path), "rb") as wav_file:
            assert wav_file.getframerate() == MMS_SAMPLING_RATE
            assert wav_file.getnframes() == TTS_TEST_SAMPLES

    def test_env_model_override(self, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setenv("TTS_MODEL", "custom/tts")
        stub_pipeline = MagicMock(
            return_value={"audio": [0.0], "sampling_rate": MMS_SAMPLING_RATE}
        )
        _stub_transformers(monkeypatch, stub_pipeline)

        generator = TextToSpeechGenerator()
        assert generator.model_name == "custom/tts"
