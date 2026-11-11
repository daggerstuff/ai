"""Command line entry point for the OVHcloud API client.

Examples::

    python -m ai.infrastructure.ovh.cli token
    python -m ai.infrastructure.ovh.cli me
    python -m ai.infrastructure.ovh.cli refresh
    python -m ai.infrastructure.ovh.cli sections --version v2
    python -m ai.infrastructure.ovh.cli ops "public-ip"
    python -m ai.infrastructure.ovh.cli describe "GET /me"
    python -m ai.infrastructure.ovh.cli call GET /cloud/project/{serviceName}/quota \\
        --param serviceName=6d7e4e17cbf14daabf7dee0210ab9cfc

The AI Training API is a separate host; use the ``ovhai`` CLI for it.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
from typing import Any

from ai.infrastructure.ovh.catalog import (
    API_VERSIONS,
    DEFAULT_BASE_URL,
    Catalog,
    download_catalog,
    load_catalog,
)
from ai.infrastructure.ovh.client import (
    DEFAULT_API_VERSION,
    MissingTokenError,
    OvhApiError,
    OvhClient,
    resolve_token_with_source,
)

CACHE_ENV = "OVH_SPEC_CACHE"
DEFAULT_CACHE = Path.home() / ".cache" / "pixelated" / "ovh-specs"
EXIT_API_ERROR = 1
EXIT_NO_TOKEN = 2


def cache_dir_from(args: argparse.Namespace) -> Path:
    if args.cache_dir:
        return Path(args.cache_dir).expanduser()
    from_env = os.environ.get(CACHE_ENV)
    return Path(from_env).expanduser() if from_env else DEFAULT_CACHE


def emit(payload: Any) -> None:
    print(json.dumps(payload, indent=2, sort_keys=True))


def cmd_token(_args: argparse.Namespace) -> int:
    try:
        _token, source = resolve_token_with_source()
    except MissingTokenError as exc:
        print(f"missing token: {exc}", file=sys.stderr)
        return EXIT_NO_TOKEN
    emit({"resolved": True, "source": source})
    return 0


def cmd_me(args: argparse.Namespace) -> int:
    with OvhClient(api_version=args.api_version) as client:
        emit(client.get("/me"))
    return 0


def cmd_refresh(args: argparse.Namespace) -> int:
    versions = args.version or list(API_VERSIONS)
    summary = download_catalog(
        cache_dir_from(args),
        base_url=args.base_url,
        versions=versions,
        workers=args.workers,
    )
    emit(summary)
    return 0 if not summary["failures"] else EXIT_API_ERROR


def _catalog(args: argparse.Namespace) -> Catalog:
    return load_catalog(cache_dir_from(args))


def cmd_sections(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    rows = [
        {
            "version": section.version,
            "section": section.path,
            "name": section.name,
            "operations": section.operation_count,
        }
        for section in catalog.sections
        if args.version in (None, section.version)
    ]
    emit(rows)
    return 0


def cmd_ops(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    hits = catalog.find(args.pattern or "", limit=args.limit)
    if args.version:
        hits = [op for op in hits if op.version == args.version]
    emit([op.as_dict() for op in hits])
    return 0


def cmd_describe(args: argparse.Namespace) -> int:
    catalog = _catalog(args)
    target = args.target.strip()
    method, _, path = target.partition(" ")
    if path:
        matches = [op for op in catalog.for_path(path) if op.method.upper() == method.upper()]
    else:
        matches = catalog.find(target, limit=args.limit)
    if not matches:
        print(f"no operation matched {target!r} in {catalog.cache_dir}", file=sys.stderr)
        return EXIT_API_ERROR
    emit([op.as_dict() for op in matches])
    return 0


def parse_params(raw: list[str]) -> dict[str, str]:
    params: dict[str, str] = {}
    for item in raw:
        key, sep, value = item.partition("=")
        if not sep or not key:
            raise SystemExit(f"--param expects key=value, got {item!r}")
        params[key] = value
    return params


def cmd_call(args: argparse.Namespace) -> int:
    body: Any = None
    if args.json is not None:
        body = json.loads(args.json)
    elif args.json_file:
        body = json.loads(Path(args.json_file).read_text(encoding="utf-8"))
    with OvhClient(api_version=args.api_version, base_url=args.base_url) as client:
        emit(client.request(args.method, args.path, params=parse_params(args.param), body=body))
    return 0


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="ovh-api",
        description="OVHcloud public API client and offline schema catalog.",
    )
    parser.add_argument("--cache-dir", help=f"schema cache directory (default: ${CACHE_ENV} or {DEFAULT_CACHE})")
    parser.add_argument("--base-url", default=DEFAULT_BASE_URL, help="API base URL")
    parser.add_argument("--version", help="API version filter for schema commands (1.0 or v2)")
    parser.add_argument("--limit", type=int, default=40, help="maximum rows to print")
    sub = parser.add_subparsers(dest="command", required=True)

    sub.add_parser("token", help="report where the bearer token comes from").set_defaults(func=cmd_token)
    sub.add_parser("me", help="GET /me to verify the token").set_defaults(func=cmd_me)

    refresh = sub.add_parser("refresh", help="download navigation trees and section specs")
    refresh.add_argument("--workers", type=int, default=8, help="parallel downloads")
    refresh.set_defaults(func=cmd_refresh)

    sub.add_parser("sections", help="list documented sections from the cache").set_defaults(func=cmd_sections)

    ops = sub.add_parser("ops", help="search cached operations")
    ops.add_argument("pattern", nargs="?", default="", help="case-insensitive substring")
    ops.set_defaults(func=cmd_ops)

    describe = sub.add_parser("describe", help="show one operation (e.g. 'GET /me')")
    describe.add_argument("target")
    describe.set_defaults(func=cmd_describe)

    call = sub.add_parser("call", help="issue an authenticated API call")
    call.add_argument("method", help="HTTP method")
    call.add_argument("path", help="API path, e.g. /me")
    call.add_argument("--param", action="append", default=[], metavar="KEY=VALUE", help="query parameter (repeatable)")
    call.add_argument("--json", help="request body as a JSON string")
    call.add_argument("--json-file", help="request body read from a file")
    call.add_argument("--api-version", default=DEFAULT_API_VERSION, help="1.0 (default) or v2")
    call.set_defaults(func=cmd_call)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    try:
        return int(args.func(args))
    except MissingTokenError as exc:
        print(f"missing token: {exc}", file=sys.stderr)
        return EXIT_NO_TOKEN
    except OvhApiError as exc:
        print(str(exc), file=sys.stderr)
        return EXIT_API_ERROR


if __name__ == "__main__":
    raise SystemExit(main())
