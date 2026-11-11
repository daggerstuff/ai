import hashlib
import importlib.util
import json
import sys
from pathlib import Path
from urllib.error import HTTPError

import pytest

EXPECTED_TIMEOUT_SECONDS = 60

MODULE_NAME = "wandb_sft_g04"
MODULE_PATH = Path(__file__).parents[1] / "launch" / f"{MODULE_NAME}.py"
SPEC = importlib.util.spec_from_file_location(MODULE_NAME, MODULE_PATH)
if SPEC is None or SPEC.loader is None:
    raise ImportError(f"cannot load test module at {MODULE_PATH}")
launcher = importlib.util.module_from_spec(SPEC)
sys.modules[MODULE_NAME] = launcher
SPEC.loader.exec_module(launcher)


def make_release(tmp_path: Path, status: str = "approved", group_key: str = "seed_group_sha256") -> tuple[Path, bytes]:
    payload_bytes = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Synthetic prompt", "weight": 0},
                    {"role": "assistant", "content": "Synthetic response", "weight": 1},
                ]
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    (tmp_path / "wandb-train.jsonl").write_bytes(payload_bytes)
    val_bytes = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Synthetic validation prompt", "weight": 0},
                    {"role": "assistant", "content": "Synthetic validation response", "weight": 1},
                ]
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    test_bytes = (
        json.dumps(
            {
                "messages": [
                    {"role": "user", "content": "Synthetic test prompt", "weight": 0},
                    {"role": "assistant", "content": "Synthetic test response", "weight": 1},
                ]
            },
            separators=(",", ":"),
        )
        + "\n"
    ).encode()
    evidence_bytes = b"synthetic group-disjointness evidence\n"
    source_rows = b'{"source":"synthetic"}\n'
    (tmp_path / "source.jsonl").write_bytes(source_rows)
    split_identity = []
    rowrefs = []
    for split in ("train", "validation", "test"):
        candidate_hash = hashlib.sha256(f"{split}-candidate".encode()).hexdigest()
        group_hash = hashlib.sha256(f"{split}-group".encode()).hexdigest()
        split_identity.append({"candidate_row_sha256": candidate_hash, group_key: group_hash, "split": split})
        rowrefs.append({"candidate_row_sha256": candidate_hash, group_key: group_hash, "provisional_split": split})
    internal_bytes = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in split_identity)
    rowref_bytes = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rowrefs)
    (tmp_path / "internal.jsonl").write_bytes(internal_bytes)
    (tmp_path / "rowrefs.jsonl").write_bytes(rowref_bytes)
    (tmp_path / "heldout.jsonl").write_bytes(b"")
    (tmp_path / "wandb-val.jsonl").write_bytes(val_bytes)
    (tmp_path / "wandb-test.jsonl").write_bytes(test_bytes)
    (tmp_path / "group-evidence.json").write_bytes(evidence_bytes)
    manifest = {
        "status": "approved_for_research_training" if status == "approved" else status,
        "upload_gate": True,
        "provider": "wandb",
        "scope": {"use": "research_fine_tuning_candidate_only", "clinical_or_production_readiness": False},
        "limitations": ["synthetic fixture only"],
        "release_id": "g04-test-release",
        "source_candidate_sha256": "a" * 64,
        "source_candidate": {"path": "source.jsonl", "sha256": hashlib.sha256(source_rows).hexdigest(), "row_count": 1},
        "internal_manifest": {
            "path": "internal.jsonl",
            "sha256": hashlib.sha256(internal_bytes).hexdigest(),
            "row_count": 3,
        },
        "partitions": {
            "train": {
                "split": "train",
                "path": "wandb-train.jsonl",
                "sha256": hashlib.sha256(payload_bytes).hexdigest(),
                "row_count": 1,
                "source_group_count": 1,
            },
            "validation": {
                "split": "validation",
                "path": "wandb-val.jsonl",
                "sha256": hashlib.sha256(val_bytes).hexdigest(),
                "row_count": 1,
                "source_group_count": 1,
            },
            "test": {
                "split": "test",
                "path": "wandb-test.jsonl",
                "sha256": hashlib.sha256(test_bytes).hexdigest(),
                "row_count": 1,
                "source_group_count": 1,
            },
        },
        "evidence": {
            "group_disjointness": {
                "status": "verified",
                "group_key": group_key,
                "train_group_count": 1,
                "validation_group_count": 1,
                "test_group_count": 1,
                "pairwise_intersections": {"train_validation": 0, "train_test": 0, "validation_test": 0},
                "reserved_eval_group_count": 0,
                "evidence_path": "rowrefs.jsonl",
                "evidence_sha256": hashlib.sha256(rowref_bytes).hexdigest(),
                "heldout_exclusions_path": "heldout.jsonl",
                "heldout_exclusions_sha256": hashlib.sha256(b"").hexdigest(),
            }
        },
    }
    manifest["source_candidate_sha256"] = manifest["source_candidate"]["sha256"]
    path = tmp_path / "release.json"
    path.write_text(json.dumps(manifest), encoding="utf-8")
    return path, payload_bytes


def test_rejects_uncleared_g04_release_without_preparing_artifact(tmp_path: Path) -> None:
    manifest, _ = make_release(tmp_path, status="diagnostic")
    with pytest.raises(launcher.QualificationError, match="not approved"):
        launcher.resolve_train_payload(manifest)
    assert not (tmp_path / "artifacts").exists()


def test_rejects_approved_status_when_upload_gate_is_false(tmp_path: Path) -> None:
    manifest, _ = make_release(tmp_path)
    value = json.loads(manifest.read_text())
    value["upload_gate"] = False
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(launcher.QualificationError, match="upload_gate"):
        launcher.resolve_train_payload(manifest)


def test_accepts_verified_prompt_disjoint_group_evidence(tmp_path: Path) -> None:
    manifest, _ = make_release(tmp_path, group_key="prompt_group_sha256")

    _, payload = launcher.resolve_train_payload(manifest)

    assert payload.rows == 1


def test_distinguishes_repeated_row_hashes_by_candidate_row_number() -> None:
    candidate_hash = hashlib.sha256(b"identical candidate row").hexdigest()
    group_hash = hashlib.sha256(b"same content group").hexdigest()
    rows = [
        {
            "candidate_row_number_1based": row_number,
            "candidate_row_sha256": candidate_hash,
            "content_group_sha256": group_hash,
            "split": "train",
        }
        for row_number in (17, 18)
    ]

    membership, row_ids = launcher._membership(rows, "split", "internal manifest", "content_group_sha256")

    assert membership["train"] == {group_hash}
    assert row_ids == {"row:17": candidate_hash, "row:18": candidate_hash}


def test_rejects_payload_hash_mismatch(tmp_path: Path) -> None:
    manifest, _ = make_release(tmp_path)
    (tmp_path / "wandb-train.jsonl").write_text("{}\n", encoding="utf-8")
    with pytest.raises(launcher.QualificationError, match="file/hash"):
        launcher.resolve_train_payload(manifest)


def test_rejects_group_membership_evidence_that_disagrees(tmp_path: Path) -> None:
    manifest, _ = make_release(tmp_path)
    value = json.loads(manifest.read_text())
    rowrefs = (
        (tmp_path / "rowrefs.jsonl")
        .read_text()
        .replace('"provisional_split":"train"', '"provisional_split":"validation"', 1)
    )
    (tmp_path / "rowrefs.jsonl").write_text(rowrefs)
    value["evidence"]["group_disjointness"]["evidence_sha256"] = hashlib.sha256(rowrefs.encode()).hexdigest()
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(launcher.QualificationError, match="rowref evidence"):
        launcher.resolve_train_payload(manifest)


def test_recomputes_group_intersections_from_internal_rows(tmp_path: Path) -> None:
    manifest, _ = make_release(tmp_path)
    value = json.loads(manifest.read_text())
    internal_path = tmp_path / "internal.jsonl"
    rows = [json.loads(line) for line in internal_path.read_text().splitlines()]
    rows[1]["seed_group_sha256"] = rows[0]["seed_group_sha256"]
    internal_bytes = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in rows)
    internal_path.write_bytes(internal_bytes)
    value["internal_manifest"]["sha256"] = hashlib.sha256(internal_bytes).hexdigest()
    rowrefs_path = tmp_path / "rowrefs.jsonl"
    ref_rows = [json.loads(line) for line in rowrefs_path.read_text().splitlines()]
    ref_rows[1]["seed_group_sha256"] = ref_rows[0]["seed_group_sha256"]
    ref_bytes = b"".join((json.dumps(row, separators=(",", ":")) + "\n").encode() for row in ref_rows)
    rowrefs_path.write_bytes(ref_bytes)
    value["evidence"]["group_disjointness"]["evidence_sha256"] = hashlib.sha256(ref_bytes).hexdigest()
    manifest.write_text(json.dumps(value), encoding="utf-8")
    with pytest.raises(launcher.QualificationError, match="not group-disjoint"):
        launcher.resolve_train_payload(manifest)


def test_accepts_zero_weight_system_context_and_preserves_assistant_loss_contract(tmp_path: Path) -> None:
    manifest, payload = make_release(tmp_path)
    row = {
        "messages": [
            {"role": "user", "content": "Synthetic prompt", "weight": 0},
            {"role": "system", "content": "Unexpected system message", "weight": 0},
            {"role": "assistant", "content": "Synthetic response", "weight": 1},
        ]
    }
    bad = (json.dumps(row, separators=(",", ":")) + "\n").encode()
    (tmp_path / "wandb-train.jsonl").write_bytes(bad)
    updated = json.loads(manifest.read_text())
    updated["partitions"]["train"]["sha256"] = hashlib.sha256(bad).hexdigest()
    manifest.write_text(json.dumps(updated), encoding="utf-8")
    _, system_payload = launcher.resolve_train_payload(manifest)
    assert system_payload.rows == 1

    (tmp_path / "wandb-train.jsonl").write_bytes(payload)
    updated["partitions"]["train"]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest.write_text(json.dumps(updated), encoding="utf-8")
    wrong_mask = payload.replace(b'"weight":1', b'"weight":0')
    (tmp_path / "wandb-train.jsonl").write_bytes(wrong_mask)
    updated["partitions"]["train"]["sha256"] = hashlib.sha256(wrong_mask).hexdigest()
    manifest.write_text(json.dumps(updated), encoding="utf-8")
    with pytest.raises(launcher.QualificationError, match="loss mask"):
        launcher.resolve_train_payload(manifest)

    (tmp_path / "wandb-train.jsonl").write_bytes(payload)
    updated["partitions"]["train"]["sha256"] = hashlib.sha256(payload).hexdigest()
    manifest.write_text(json.dumps(updated), encoding="utf-8")
    parsed_manifest, parsed_payload = launcher.resolve_train_payload(manifest)
    packaged = launcher.make_training_artifact(
        parsed_payload, parsed_manifest, launcher.BASELINE_CONFIG, tmp_path / "artifacts"
    )
    assert (tmp_path / "artifacts" / "trajectories.jsonl").read_bytes() == payload
    assert packaged["metadata"]["loss_mask"] == "assistant turns only"
    assert packaged["metadata"]["assistant_turns_config"] == "all"
    gemma = launcher.make_training_artifact(
        parsed_payload,
        parsed_manifest,
        launcher.BASELINE_CONFIG,
        tmp_path / "gemma-artifacts",
        model_id="google/gemma-4-26B-A4B-it",
    )
    assert gemma["metadata"]["base_model"] == "google/gemma-4-26B-A4B-it"


class FakeAPI:
    def __init__(self) -> None:
        self.calls: list[tuple[str, str]] = []
        self.request_bodies: list[dict | None] = []

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        self.calls.append((method, path))
        self.request_bodies.append(body)
        if method == "GET" and path == "/preview/base-models":
            return {"models": [{"id": model_id} for model_id in launcher.ALLOWED_MODEL_IDS]}
        if method == "POST" and path == "/preview/models":
            return {"id": "00000000-0000-0000-0000-000000000001"}
        if method == "POST" and path == "/preview/sft-training-jobs":
            return {
                "id": "00000000-0000-0000-0000-000000000002",
                "config": {"batch_size": 8, "learning_rate": 0.0001, "api_key": "must-not-persist"},
            }
        if method == "GET" and path.endswith("00000000-0000-0000-0000-000000000002"):
            return {
                "id": "00000000-0000-0000-0000-000000000002",
                "status": "running",
                "resolved_config": {"batch_size": 8, "learning_rate": 0.0001, "api_key": "must-not-persist"},
                "api_key": "must-not-persist",
            }
        raise AssertionError(f"unexpected fake API request: {method} {path}")


def test_resume_does_not_duplicate_model_or_job_creation(tmp_path: Path) -> None:
    manifest, _ = make_release(tmp_path)
    api = FakeAPI()
    uploads: list[str] = []

    def upload(directory: Path, name: str) -> str:
        assert directory.is_dir()
        uploads.append(name)
        return f"wandb-artifact:///{launcher.ENTITY}/{launcher.PROJECT}/{name}:v0"

    def private_check(_entity: str, _project: str, _key: str) -> None:
        return None

    request = launcher.ExecutionRequest(
        manifest_path=manifest,
        arm="baseline",
        state_dir=tmp_path / "state",
        work_dir=tmp_path / "artifact-work",
        api=api,
        api_key="fake-key-for-test",
        private_check=private_check,
        artifact_upload=upload,
        max_polls=0,
    )
    first = launcher.execute(request)
    second = launcher.execute(request)
    assert first["training_job_id"] == second["training_job_id"]
    assert api.calls.count(("POST", "/preview/models")) == 1
    assert api.calls.count(("POST", "/preview/sft-training-jobs")) == 1
    assert len(uploads) == 1
    state = json.loads(Path(first["state_path"]).read_text())
    assert state["provider_reported_config"] == {"batch_size": 8, "learning_rate": 0.0001}
    assert "must-not-persist" not in json.dumps(state)


def test_unknown_job_create_outcome_blocks_duplicate_post(tmp_path: Path) -> None:
    manifest, _ = make_release(tmp_path)
    api = FakeAPI()

    def uploads(directory: Path, name: str) -> str:
        assert directory.is_dir()
        return f"wandb-artifact:///{launcher.ENTITY}/{launcher.PROJECT}/{name}:v0"

    def private_check(_entity: str, _project: str, _key: str) -> None:
        return None

    request = launcher.ExecutionRequest(
        manifest_path=manifest,
        arm="baseline",
        state_dir=tmp_path / "state",
        work_dir=tmp_path / "artifact-work",
        api=api,
        api_key="fake-key-for-test",
        private_check=private_check,
        artifact_upload=uploads,
        max_polls=0,
    )
    result = launcher.execute(request)
    state_path = Path(result["state_path"])
    state = json.loads(state_path.read_text())
    state["phase"] = "job_creation_submitted_unknown"
    state.pop("training_job_id", None)
    state_path.write_text(json.dumps(state))
    before = api.calls.count(("POST", "/preview/sft-training-jobs"))
    with pytest.raises(launcher.QualificationError, match="outcome is unknown"):
        launcher.execute(request)
    assert api.calls.count(("POST", "/preview/sft-training-jobs")) == before


class JobListAPI(FakeAPI):
    """FakeAPI plus the job-enumeration route reconcile depends on."""

    def __init__(self, jobs: list[dict] | Exception) -> None:
        super().__init__()
        self.jobs = jobs

    def request(self, method: str, path: str, body: dict | None = None) -> dict:
        if method == "GET" and path == "/preview/sft-training-jobs":
            self.calls.append((method, path))
            self.request_bodies.append(body)
            if isinstance(self.jobs, Exception):
                raise self.jobs
            return {"jobs": list(self.jobs)}
        return super().request(method, path, body)


MODEL_UUID = "00000000-0000-0000-0000-000000000001"
JOB_UUID = "00000000-0000-0000-0000-000000000002"


def prepare_run(
    tmp_path: Path, api: FakeAPI
) -> tuple[launcher.ExecutionRequest, launcher.TrainingPlan, Path, list[str]]:
    """Run a full launch to get a real state file, then return the pieces reconcile needs."""
    manifest, _ = make_release(tmp_path)
    uploads: list[str] = []

    def upload(directory: Path, name: str) -> str:
        assert directory.is_dir()
        uploads.append(name)
        return f"wandb-artifact:///{launcher.ENTITY}/{launcher.PROJECT}/{name}:v0"

    request = launcher.ExecutionRequest(
        manifest_path=manifest,
        arm="baseline",
        state_dir=tmp_path / "state",
        work_dir=tmp_path / "artifact-work",
        api=api,
        api_key="fake-key-for-test",
        private_check=lambda *_args: None,
        artifact_upload=upload,
        max_polls=0,
    )
    result = launcher.execute(request)
    parsed_manifest, payload = launcher.resolve_train_payload(manifest, "baseline")
    work_path = request.work_dir / parsed_manifest["release_id"] / request.arm
    artifact = launcher.make_training_artifact(payload, parsed_manifest, launcher.BASELINE_CONFIG, work_path)
    plan = launcher.TrainingPlan(parsed_manifest, payload, launcher.BASELINE_CONFIG, artifact, result["task_id"])
    return request, plan, Path(result["state_path"]), uploads


def read_state(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def write_state(path: Path, state: dict) -> None:
    path.write_text(json.dumps(state), encoding="utf-8")


def test_reconcile_adopts_a_model_created_before_the_outage(tmp_path: Path) -> None:
    api = FakeAPI()
    request, plan, state_path, _ = prepare_run(tmp_path, api)
    state = read_state(state_path)
    state["phase"] = "model_creation_submitted_unknown"
    state.pop("model_id", None)
    write_state(state_path, state)
    before = api.calls.count(("POST", "/preview/models"))

    outcome = launcher.reconcile(request, plan, state, state_path)

    assert outcome["resolved"] == ["model_adopted"]
    assert outcome["phase"] == "model_ready"
    assert api.calls.count(("POST", "/preview/models")) == before + 1
    saved = read_state(state_path)
    assert saved["model_id"] == MODEL_UUID
    assert saved["reconciled_phases"] == ["model_creation_submitted_unknown"]


def test_reconcile_confirms_a_recorded_artifact_instead_of_re_uploading(tmp_path: Path) -> None:
    api = FakeAPI()
    request, plan, state_path, uploads = prepare_run(tmp_path, api)
    state = read_state(state_path)
    state["phase"] = "artifact_upload_submitted_unknown"
    write_state(state_path, state)
    uploaded = len(uploads)

    outcome = launcher.reconcile(request, plan, state, state_path)

    assert outcome["resolved"] == ["artifact_upload_confirmed"]
    assert outcome["phase"] == "artifact_uploaded"
    assert len(uploads) == uploaded


def test_reconcile_repeats_an_artifact_upload_with_no_recorded_uri(tmp_path: Path) -> None:
    api = FakeAPI()
    request, plan, state_path, uploads = prepare_run(tmp_path, api)
    state = read_state(state_path)
    state["phase"] = "artifact_upload_submitted_unknown"
    state.pop("training_data_url", None)
    write_state(state_path, state)
    uploaded = len(uploads)

    outcome = launcher.reconcile(request, plan, state, state_path)

    assert outcome["resolved"] == ["artifact_upload_repeated"]
    assert outcome["phase"] == "artifact_uploaded"
    assert len(uploads) == uploaded + 1
    assert read_state(state_path)["training_data_url"].startswith("wandb-artifact:///")


def test_reconcile_adopts_the_single_job_the_provider_created(tmp_path: Path) -> None:
    api = JobListAPI([])
    request, plan, state_path, _ = prepare_run(tmp_path, api)
    state = read_state(state_path)
    state["phase"] = "job_creation_submitted_unknown"
    state.pop("training_job_id", None)
    write_state(state_path, state)
    api.jobs = [{"id": JOB_UUID, "model_id": MODEL_UUID, "training_data_url": state["training_data_url"]}]

    outcome = launcher.reconcile(request, plan, state, state_path)

    assert outcome["resolved"] == ["job_adopted"]
    assert outcome["phase"] == "job_created"
    assert read_state(state_path)["training_job_id"] == JOB_UUID
    assert api.calls.count(("POST", "/preview/sft-training-jobs")) == 1


def test_reconcile_refuses_to_guess_between_two_matching_jobs(tmp_path: Path) -> None:
    api = JobListAPI([])
    request, plan, state_path, _ = prepare_run(tmp_path, api)
    state = read_state(state_path)
    state["phase"] = "job_creation_submitted_unknown"
    state.pop("training_job_id", None)
    write_state(state_path, state)
    api.jobs = [
        {"id": JOB_UUID, "model_id": MODEL_UUID, "training_data_url": state["training_data_url"]},
        {"id": "00000000-0000-0000-0000-000000000003", "model_id": MODEL_UUID},
    ]

    with pytest.raises(launcher.QualificationError, match="refusing to guess"):
        launcher.reconcile(request, plan, state, state_path)
    assert api.calls.count(("POST", "/preview/sft-training-jobs")) == 1


def test_reconcile_fails_closed_when_the_provider_cannot_list_jobs(tmp_path: Path) -> None:
    api = JobListAPI([])
    request, plan, state_path, _ = prepare_run(tmp_path, api)
    state = read_state(state_path)
    state["phase"] = "job_creation_submitted_unknown"
    state.pop("training_job_id", None)
    write_state(state_path, state)
    api.jobs = launcher.QualificationError("HTTP 404 for GET /preview/sft-training-jobs")

    with pytest.raises(launcher.QualificationError, match="no usable job list"):
        launcher.reconcile(request, plan, state, state_path)
    assert api.calls.count(("POST", "/preview/sft-training-jobs")) == 1


def test_reconcile_retries_job_creation_only_when_nothing_exists(tmp_path: Path) -> None:
    api = JobListAPI([])
    request, plan, state_path, _ = prepare_run(tmp_path, api)
    state = read_state(state_path)
    state["phase"] = "job_creation_submitted_unknown"
    state.pop("training_job_id", None)
    write_state(state_path, state)

    outcome = launcher.reconcile(request, plan, state, state_path)

    assert outcome["resolved"] == ["job_creation_retried"]
    assert outcome["phase"] == "job_creation_retried"
    assert "training_job_id" not in read_state(state_path)


def test_reconcile_leaves_a_known_phase_alone(tmp_path: Path) -> None:
    api = FakeAPI()
    request, plan, state_path, _ = prepare_run(tmp_path, api)
    state = read_state(state_path)
    before = list(api.calls)

    outcome = launcher.reconcile(request, plan, state, state_path)

    assert outcome == {"resolved": [], "phase": state["phase"]}
    assert api.calls == before


def test_load_state_only_accepts_an_uncertain_phase_when_reconciling(tmp_path: Path) -> None:
    api = FakeAPI()
    _, _, state_path, _ = prepare_run(tmp_path, api)
    state = read_state(state_path)
    state["phase"] = "artifact_upload_submitted_unknown"
    write_state(state_path, state)

    with pytest.raises(launcher.QualificationError, match="outcome is unknown"):
        launcher._load_state(state_path, state["task_id"], state["payload_sha256"])
    loaded = launcher._load_state(state_path, state["task_id"], state["payload_sha256"], allow_uncertain=True)
    assert loaded["phase"] == "artifact_upload_submitted_unknown"


def test_http_error_does_not_echo_body_or_api_key() -> None:
    def denied(request, timeout):
        assert timeout == EXPECTED_TIMEOUT_SECONDS
        assert request.get_header("User-agent") == launcher.TRAINING_API_USER_AGENT
        raise HTTPError("https://api.training.wandb.ai/v1/x", 400, "bad", {}, None)

    client = launcher.TrainingAPI("never-log-this-token", opener=denied)
    with pytest.raises(launcher.QualificationError) as error:
        client.request("GET", "/preview/base-models")
    assert "never-log-this-token" not in str(error.value)
