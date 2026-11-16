"""Tests for the R2 storage layer that replaced the OVH volume for eval results.

The point of the module is that a GPU job can save its results without holding a
credential, so the interesting behaviour is the presigned-URL path: the map that
carries the URLs into the job, and the plain-HTTP PUT that uses them.
"""

from __future__ import annotations

import io
import urllib.error
from pathlib import Path
from typing import Any

import pytest

from training.launch import r2_sync


class FakeS3:
    """Records every call and serves a paged listing like the real client."""

    def __init__(self, objects: dict[str, bytes] | None = None) -> None:
        self.objects = objects or {}
        self.puts: list[dict[str, Any]] = []
        self.list_calls: list[dict[str, Any]] = []
        self.presigned: list[tuple[str, dict[str, Any], int]] = []

    def put_object(self, **kwargs: Any) -> dict[str, Any]:
        self.puts.append(kwargs)
        self.objects[kwargs["Key"]] = kwargs["Body"]
        return {}

    def get_object(self, *, Bucket: str, Key: str) -> dict[str, Any]:
        return {"Body": io.BytesIO(self.objects[Key])}

    def list_objects_v2(self, **kwargs: Any) -> dict[str, Any]:
        self.list_calls.append(kwargs)
        prefix = kwargs.get("Prefix", "")
        keys = sorted(key for key in self.objects if key.startswith(prefix))
        # One object per page, so the pagination branch is exercised.
        start = int(kwargs.get("ContinuationToken", "0"))
        page = keys[start : start + 1]
        response: dict[str, Any] = {
            "Contents": [{"Key": key, "Size": len(self.objects[key])} for key in page],
            "IsTruncated": start + 1 < len(keys),
        }
        if response["IsTruncated"]:
            response["NextContinuationToken"] = str(start + 1)
        return response

    def generate_presigned_url(self, operation: str, *, Params: dict[str, str], ExpiresIn: int) -> str:
        self.presigned.append((operation, Params, ExpiresIn))
        return f"https://r2.example/{Params['Key']}?sig={operation}"


def test_upload_dir_mirrors_the_tree_and_skips_bytecode(tmp_path: Path) -> None:
    run = tmp_path / "run"
    (run / "__pycache__").mkdir(parents=True)
    (run / "results.jsonl").write_text('{"a": 1}\n')
    (run / "manifest.json").write_text("{}")
    (run / "__pycache__" / "results.cpython-314.pyc").write_bytes(b"\x00")
    s3 = FakeS3()

    keys = r2_sync.upload_dir(run, bucket="training", prefix="evals/v22/tag/run", s3_client=s3)

    assert keys == ["evals/v22/tag/run/manifest.json", "evals/v22/tag/run/results.jsonl"]
    types = {put["Key"]: put["ContentType"] for put in s3.puts}
    assert types["evals/v22/tag/run/results.jsonl"] == "application/x-ndjson"
    assert types["evals/v22/tag/run/manifest.json"] == "application/json"


def test_download_prefix_follows_pagination_and_mirrors_keys(tmp_path: Path) -> None:
    s3 = FakeS3(
        {
            "evals/v22/tag/run/results.jsonl": b"one\n",
            "evals/v22/tag/run/nested/manifest.json": b"{}",
            "evals/v22/other/run/results.jsonl": b"two\n",
        }
    )

    written = r2_sync.download_prefix(
        bucket="training", prefix="evals/v22/tag", destination=tmp_path / "out", s3_client=s3
    )

    assert sorted(path.relative_to(tmp_path / "out").as_posix() for path in written) == [
        "run/nested/manifest.json",
        "run/results.jsonl",
    ]
    assert (tmp_path / "out" / "run" / "results.jsonl").read_bytes() == b"one\n"
    # Two matching objects, one per page: the second call must carry a token.
    assert len(s3.list_calls) == 2
    assert "ContinuationToken" not in s3.list_calls[0]
    assert s3.list_calls[1]["ContinuationToken"] == "1"


def test_presign_uses_the_operation_that_matches_the_method() -> None:
    s3 = FakeS3()

    puts = r2_sync.presign(["run/results.jsonl"], bucket="training", method="put", expires=600, s3_client=s3)
    gets = r2_sync.presign(["run/results.jsonl"], bucket="training", method="get", expires=600, s3_client=s3)

    assert puts["run/results.jsonl"].endswith("sig=put_object")
    assert gets["run/results.jsonl"].endswith("sig=get_object")
    assert [call[2] for call in s3.presigned] == [600, 600]


def test_put_file_sends_a_content_length_and_retries_a_failed_attempt(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = tmp_path / "results.jsonl"
    payload.write_text('{"a": 1}\n')
    attempts: list[Any] = []

    class Handle:
        status = 200

        def __enter__(self) -> Handle:
            return self

        def __exit__(self, *_: Any) -> None:
            return None

    def fake_urlopen(request: Any, timeout: float) -> Handle:
        attempts.append(request)
        if len(attempts) == 1:
            raise urllib.error.URLError("connection reset")
        return Handle()

    monkeypatch.setattr(r2_sync.urllib.request, "urlopen", fake_urlopen)
    monkeypatch.setattr(r2_sync.time, "sleep", lambda _seconds: None)

    assert r2_sync.put_file("https://r2.example/key?sig=put_object", payload) == 200
    assert len(attempts) == 2
    # A presigned PUT without Content-Length is sent chunked, which S3 rejects.
    assert attempts[1].get_header("Content-length") == str(len(payload.read_bytes()))


def test_put_file_gives_up_after_the_attempt_budget(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    payload = tmp_path / "results.jsonl"
    payload.write_text("x\n")
    monkeypatch.setattr(r2_sync.time, "sleep", lambda _seconds: None)

    def always_fail(request: Any, timeout: float) -> Any:
        raise urllib.error.HTTPError(request.full_url, 403, "SignatureDoesNotMatch", {}, None)

    monkeypatch.setattr(r2_sync.urllib.request, "urlopen", always_fail)

    with pytest.raises(RuntimeError, match="upload failed after 3 attempts"):
        r2_sync.put_file("https://r2.example/key", payload, attempts=3)


def test_the_job_map_survives_a_round_trip_through_an_environment_variable() -> None:
    urls = {
        f"eval-bf16-dev/{name}": f"https://r2.example/{name}?sig=put_object"
        for name in ("results.jsonl", "manifest.json")
    }

    blob = r2_sync.encode_map({"upload": urls, "tag": "l4d"})

    # It has to fit in one env var, so the compression is part of the contract.
    assert len(blob) < 400
    assert r2_sync.decode_map(blob) == {"upload": urls, "tag": "l4d"}


def test_upload_run_sends_only_the_files_that_were_signed(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    run = tmp_path / "eval-nf4-sealed"
    run.mkdir()
    (run / "results.jsonl").write_text("{}\n")
    (run / "manifest.json").write_text("{}")
    sent: list[tuple[str, Path]] = []
    monkeypatch.setattr(r2_sync, "put_file", lambda url, path: sent.append((url, path)) or 200)

    keys = r2_sync.upload_run(run, {"eval-nf4-sealed/results.jsonl": "https://r2.example/one"})

    assert keys == ["eval-nf4-sealed/results.jsonl"]
    assert sent[0][1].name == "results.jsonl"


def test_credentials_explain_what_is_missing(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in r2_sync.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)

    with pytest.raises(RuntimeError, match="R2_ACCESS_KEY_ID, R2_SECRET_ACCESS_KEY, R2_ENDPOINT is unset"):
        r2_sync.credentials(env_file=tmp_path / "absent.env")


def test_credentials_strip_the_quotes_a_dotenv_file_keeps(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    for name in r2_sync.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    dotenv = tmp_path / ".env"
    dotenv.write_text(
        "# comment\nR2_ACCESS_KEY_ID='be4c9f'\nR2_SECRET_ACCESS_KEY='b4289e'\n"
        "R2_ENDPOINT='https://account.r2.cloudflarestorage.com'\nR2_BUCKET='training'\n"
    )

    settings = r2_sync.credentials(env_file=dotenv)

    # A single-quoted value is what made the old naive regex sign with quotes.
    assert settings["R2_ACCESS_KEY_ID"] == "be4c9f"
    assert settings["R2_ENDPOINT"] == "https://account.r2.cloudflarestorage.com"
    assert settings["R2_BUCKET"] == "training"


def test_the_upload_run_command_needs_no_credential(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """The job side of the contract: a map in the environment is all it takes."""
    run = tmp_path / "eval-bf16-dev"
    run.mkdir()
    (run / "results.jsonl").write_text('{"case_id": "dev-crisis-suicide-plan"}\n')
    (run / "manifest.json").write_text("{}")
    for name in r2_sync.REQUIRED_ENV:
        monkeypatch.delenv(name, raising=False)
    monkeypatch.setenv(
        "R2_MAP_B64",
        r2_sync.encode_map({"upload": {"eval-bf16-dev/results.jsonl": "https://r2.example/one"}}),
    )
    sent: list[str] = []
    monkeypatch.setattr(r2_sync, "put_file", lambda _url, _path: sent.append(_url) or 200)

    assert r2_sync.main(["upload-run", "--run-dir", str(run)]) == 0
    assert sent == ["https://r2.example/one"]


def test_the_upload_run_command_degrades_when_the_map_is_absent(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    """Losing the map must not fail a job that already produced the results."""
    run = tmp_path / "eval-bf16-dev"
    run.mkdir()
    (run / "results.jsonl").write_text("{}\n")
    sent: list[str] = []
    monkeypatch.delenv("R2_MAP_B64", raising=False)
    monkeypatch.setattr(r2_sync, "put_file", lambda _url, _path: sent.append(_url) or 200)

    assert r2_sync.main(["upload-run", "--run-dir", str(run)]) == 2
    assert sent == []
    assert "R2_MAP_B64 is unset" in capsys.readouterr().err
