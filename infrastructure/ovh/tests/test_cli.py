"""Tests for the OVHcloud CLI wiring: every handler gets the knobs it reads."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import pytest

from ai.infrastructure.ovh import cli
from ai.infrastructure.ovh.catalog import API_VERSIONS, Catalog, Operation
from ai.infrastructure.ovh.client import DEFAULT_API_VERSION


def _fake_download(seen: dict[str, Any]) -> Any:
    def download(_cache_dir: Path, **kwargs: Any) -> dict[str, Any]:
        seen["versions"] = list(kwargs["versions"])
        return {"failures": []}

    return download


def test_me_receives_an_api_version() -> None:
    args = cli.build_parser().parse_args(["me"])

    assert args.api_version == DEFAULT_API_VERSION


def test_me_accepts_an_explicit_api_version() -> None:
    args = cli.build_parser().parse_args(["me", "--api-version", "v2"])

    assert args.api_version == "v2"


def test_refresh_sends_one_version_not_its_characters(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "download_catalog", _fake_download(seen))
    args = cli.build_parser().parse_args(["--version", "v2", "refresh"])

    assert cli.cmd_refresh(args) == 0
    assert seen["versions"] == ["v2"]


def test_refresh_defaults_to_every_version(monkeypatch: pytest.MonkeyPatch) -> None:
    seen: dict[str, Any] = {}
    monkeypatch.setattr(cli, "download_catalog", _fake_download(seen))
    args = cli.build_parser().parse_args(["refresh"])

    assert cli.cmd_refresh(args) == 0
    assert seen["versions"] == list(API_VERSIONS)


def test_ops_filters_by_version_before_the_limit(
    monkeypatch: pytest.MonkeyPatch, capsys: pytest.CaptureFixture[str]
) -> None:
    legacy_operations = 5
    operations = tuple(
        Operation(
            version="1.0" if index < legacy_operations else "v2",
            section="public-cloud",
            method="GET",
            path=f"/resource/{index}",
            summary="",
        )
        for index in range(10)
    )
    catalog = Catalog(cache_dir=Path("/tmp/specs"), operations=operations)
    monkeypatch.setattr(cli, "_catalog", lambda _args: catalog)
    args = cli.build_parser().parse_args(["--version", "v2", "--limit", "3", "ops"])

    assert cli.cmd_ops(args) == 0

    payload = json.loads(capsys.readouterr().out)
    assert [row["path"] for row in payload] == ["/resource/5", "/resource/6", "/resource/7"]


def test_call_substitutes_path_params_and_keeps_the_rest_as_query() -> None:
    args = cli.build_parser().parse_args(
        [
            "call",
            "GET",
            "/cloud/project/{serviceName}/quota",
            "--param",
            "serviceName=6d7e4e17",
            "--param",
            "region=GRA",
        ]
    )

    path, query = cli.substitute_path_params(args.path, cli.parse_params(args.param))

    assert path == "/cloud/project/6d7e4e17/quota"
    assert query == {"region": "GRA"}
