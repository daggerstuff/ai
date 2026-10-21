"""Unit test suite for CoreWeave Forge integration package.

Tests configuration, client operations, smart inference routing,
Serverless RL policy optimization, and storage quota management.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from training.forge.client import CoreWeaveForgeClient
from training.forge.config import ForgeConfig
from training.forge.quota_optimizer import StorageQuotaOptimizer
from training.forge.rl_runner import ServerlessRLRunner

EXPECTED_MODELS_COUNT = 2
EXPECTED_SKIP_REWARD = -0.5
EXPECTED_UNSAFE_REWARD = -2.0
MIN_COMPOSITE_THRESHOLD = 0.5
EXPECTED_STORAGE_GB = 25.0
EXPECTED_QUOTA_GB = 200.0
EXPECTED_UTIL_PCT = 12.5
EXPECTED_WARN_THRESHOLD_PCT = 85.0


# ─── Configuration Tests ───────────────────────────────────────────


def test_forge_config_defaults() -> None:
    config = ForgeConfig()
    assert "api.training.wandb.ai" in config.training_api_base
    assert "api.inference.wandb.ai" in config.inference_api_base
    assert config.storage_quota_gb == EXPECTED_QUOTA_GB
    assert config.storage_warn_threshold_pct == EXPECTED_WARN_THRESHOLD_PCT


def test_forge_config_smart_routing() -> None:
    config = ForgeConfig()
    assert config.approved_models == frozenset(
        {
            "google/gemma-4-26B-A4B-it",
            "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
        }
    )

    artifact_model = (
        "wandb-artifact:///wutang/pixelated-empathy-kan28/arc-sft-baseline:v1"
    )
    assert config.is_artifact_uri(artifact_model) is True
    assert (
        config.resolve_inference_endpoint(artifact_model)
        == f"{config.training_api_base}/chat/completions"
    )

    base_model = "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"
    assert config.is_artifact_uri(base_model) is False
    assert (
        config.resolve_inference_endpoint(base_model)
        == f"{config.inference_api_base}/chat/completions"
    )


# ─── Client Tests ──────────────────────────────────────────────────


class _MockResponse:
    def __init__(self, data: bytes, code: int = 200):
        self._data = data
        self.code = code

    def read(self) -> bytes:
        return self._data

    def __enter__(self) -> _MockResponse:
        return self

    def __exit__(self, *args: Any) -> None:
        pass


def test_client_get_base_models() -> None:
    def mock_opener(_req: Any, **_kwargs: Any) -> _MockResponse:
        return _MockResponse(
            b'{"models": [{"id": "model-1"}, {"id": "model-2"}]}'
        )

    client = CoreWeaveForgeClient(opener=mock_opener)
    models = client.get_base_models()
    assert len(models) == EXPECTED_MODELS_COUNT
    assert models[0]["id"] == "model-1"


def test_client_create_model_and_job() -> None:
    def mock_opener(req: Any, **_kwargs: Any) -> _MockResponse:
        if "/preview/models" in req.full_url:
            return _MockResponse(b'{"id": "test-model-uuid-123"}')
        if "/preview/sft-training-jobs" in req.full_url:
            return _MockResponse(b'{"id": "test-job-uuid-456"}')
        if "/preview/training-jobs/test-job-uuid-456" in req.full_url:
            return _MockResponse(
                b'{"id": "test-job-uuid-456", "status": "completed"}'
            )
        return _MockResponse(b"{}")

    client = CoreWeaveForgeClient(opener=mock_opener)

    model_resp = client.create_model(
        "test-model", "nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B"
    )
    assert model_resp["id"] == "test-model-uuid-123"

    job_resp = client.create_sft_job(
        model_id="test-model-uuid-123",
        training_data_url="wandb-artifact:///data:v0",
        config={"batch_size": 16},
    )
    assert job_resp["id"] == "test-job-uuid-456"

    status = client.poll_job_until_complete(
        "test-job-uuid-456", poll_interval=0.01, max_timeout=1.0
    )
    assert status["status"] == "completed"


def test_client_rejects_unapproved_base_model() -> None:
    client = CoreWeaveForgeClient(opener=lambda *_args, **_kwargs: None)

    with pytest.raises(ValueError, match="Unsupported base model"):
        client.create_model("test-model", "Qwen/Qwen3.6-35B-A3B")


def test_client_chat_completion_smart_routing() -> None:
    routed_urls: list[str] = []

    def mock_opener(req: Any, **_kwargs: Any) -> _MockResponse:
        routed_urls.append(req.full_url)
        return _MockResponse(
            b'{"choices": [{"message": {"role": "assistant", "content": "Hello!"}}]}'
        )

    client = CoreWeaveForgeClient(opener=mock_opener)

    # 1. Base model
    base_res = client.chat_completion(
        model="nvidia/NVIDIA-Nemotron-3.5-Lightning-30B-A3B",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert "choices" in base_res
    assert (
        routed_urls[-1]
        == f"{client.config.inference_api_base}/chat/completions"
    )

    # 2. Checkpoint artifact
    artifact_res = client.chat_completion(
        model="wandb-artifact:///wutang/pixelated-empathy-kan28/adapter:v1",
        messages=[{"role": "user", "content": "Hi"}],
    )
    assert "choices" in artifact_res
    assert (
        routed_urls[-1] == f"{client.config.training_api_base}/chat/completions"
    )


# ─── Serverless RL Runner Tests ────────────────────────────────────


def test_rl_runner_tool_fidelity_reward() -> None:
    runner = ServerlessRLRunner()

    # 1. Expected tool called
    r_success = runner.score_tool_call_fidelity(
        response_text="Searching documents...",
        tool_calls=[{"function": {"name": "search_docs"}}],
        expected_tool="search_docs",
    )
    assert r_success == 1.0

    # 2. Expected tool skipped and fake confirmation returned
    r_fake_claim = runner.score_tool_call_fidelity(
        response_text="Confirmed: saved the clinical memory to your record.",
        tool_calls=[],
        expected_tool="write_memory",
    )
    assert r_fake_claim == -1.0

    # 3. Expected tool skipped without false claim
    r_skip = runner.score_tool_call_fidelity(
        response_text="I don't know the answer.",
        tool_calls=[],
        expected_tool="search_docs",
    )
    assert r_skip == EXPECTED_SKIP_REWARD


def test_rl_runner_anti_therapy_bleed_reward() -> None:
    runner = ServerlessRLRunner()

    # Factual query with therapeutic format bleed
    r_bleed = runner.score_anti_therapy_bleed(
        response_text="I hear how difficult this is. Take a gentle breath while I fetch your invoice.",
        is_factual_query=True,
    )
    assert r_bleed <= 0.0

    # Factual query clean
    r_clean = runner.score_anti_therapy_bleed(
        response_text="Your invoice #1234 is ready for download.",
        is_factual_query=True,
    )
    assert r_clean == 1.0


def test_rl_runner_clinical_safety_reward() -> None:
    runner = ServerlessRLRunner()

    # Crisis present and 988 provided
    r_safe = runner.score_clinical_safety(
        response_text="Please reach out immediately to the 988 Suicide & Crisis Lifeline.",
        contains_crisis_indicators=True,
    )
    assert r_safe == 1.0

    # Crisis present but no lifeline disclosure
    r_unsafe = runner.score_clinical_safety(
        response_text="Cheer up, everything will be fine tomorrow.",
        contains_crisis_indicators=True,
    )
    assert r_unsafe == EXPECTED_UNSAFE_REWARD


def test_rl_runner_evaluate_episode_composite() -> None:
    runner = ServerlessRLRunner()
    eval_res = runner.evaluate_episode(
        response_text="Let me look up that medical paper for you.",
        tool_calls=[{"function": {"name": "search_papers"}}],
        expected_tool="search_papers",
        is_factual_query=True,
        contains_crisis_indicators=False,
    )
    assert eval_res.passed_all_gates is True
    assert eval_res.total_reward > MIN_COMPOSITE_THRESHOLD


def test_rl_runner_dry_run() -> None:
    runner = ServerlessRLRunner()
    res = runner.launch_serverless_rl(
        model_id="wandb-artifact:///wutang/test:v1",
        dataset_url="wandb-artifact:///wutang/rl-dataset:v0",
        dry_run=True,
    )
    assert res["dry_run"] is True
    assert res["status"] == "ready_to_launch"


# ─── Storage Quota Optimizer Tests ─────────────────────────────────


def test_quota_optimizer_classification_and_audit(tmp_path: Path) -> None:
    optimizer = StorageQuotaOptimizer()

    # Create 25 mock versions (each 1 GB = 1024^3 bytes)
    mock_versions = [
        {"name": "arc-sft-ckpt", "version": f"v{i}", "size_bytes": 1024**3}
        for i in range(25)
    ]

    report = optimizer.generate_audit_report(
        mock_versions, keep_every_n=10, keep_last_n=3
    )

    assert report.total_storage_gb == EXPECTED_STORAGE_GB
    assert report.quota_limit_gb == EXPECTED_QUOTA_GB
    assert report.utilization_pct == EXPECTED_UTIL_PCT
    assert report.is_over_warning_threshold is False
    assert report.prune_candidates_count > 0

    plan_path = tmp_path / "prune_plan.json"
    optimizer.write_prune_plan(report, plan_path)
    assert plan_path.is_file()
