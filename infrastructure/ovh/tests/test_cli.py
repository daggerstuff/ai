"""Tests for the OVHcloud CLI wiring: every handler gets the knobs it reads."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pytest

from ai.infrastructure.ovh import cli
from ai.infrastructure.ovh.catalog import API_VERSIONS
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
