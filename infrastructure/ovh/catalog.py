"""Harvest and index the OVHcloud developer-hub OpenAPI schemas.

The public API publishes its documentation as JSON, so the whole surface can be
downloaded and searched offline:

* navigation tree — ``{base}/{version}/__schemas/developer-hub-navigation.json``
* section spec    — ``{base}/{version}/__schemas/developer-hub-navigation/{section}.json?format=openapi3``

``section`` is the slug chain of a *leaf* node joined with ``/`` (branch nodes
404). The navigation document is public, but every real API call needs a bearer
token — see :mod:`ai.infrastructure.ovh.client`.
"""

from __future__ import annotations

import json
from collections.abc import Iterator, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import httpx

DEFAULT_BASE_URL = "https://api.us.ovhcloud.com"
API_VERSIONS = ("1.0", "v2")
SCHEMA_ROOT = "__schemas/developer-hub-navigation"
NAVIGATION_FILE = "navigation-{version}.json"
HTTP_METHODS = ("get", "post", "put", "patch", "delete")


def navigation_url(base_url: str = DEFAULT_BASE_URL, version: str = "1.0") -> str:
    return f"{base_url.rstrip('/')}/{version.strip('/')}/{SCHEMA_ROOT}.json"


def section_url(section: str, base_url: str = DEFAULT_BASE_URL, version: str = "1.0") -> str:
    return f"{base_url.rstrip('/')}/{version.strip('/')}/{SCHEMA_ROOT}/{section.strip('/')}.json?format=openapi3"


def spec_filename(version: str, section: str) -> str:
    return f"{version}__{section.strip('/').replace('/', '__')}.json"


def fetch_json(url: str, *, transport: httpx.BaseTransport | None = None, timeout: float = 60.0) -> Any:
    with httpx.Client(timeout=timeout, transport=transport) as client:
        response = client.get(url, headers={"accept": "application/json"})
        response.raise_for_status()
        return response.json()


@dataclass(frozen=True)
class Section:
    path: str
    name: str
    operation_count: int
    version: str


def iter_sections(navigation: dict[str, Any], version: str) -> Iterator[Section]:
    """Yield every leaf section in a navigation document, depth-first."""

    def walk(nodes: Sequence[dict[str, Any]], trail: tuple[str, ...]) -> Iterator[Section]:
        for node in nodes:
            slugs = (*trail, str(node.get("slug", "")))
            children = node.get("children") or []
            if children:
                yield from walk(children, slugs)
            else:
                yield Section(
                    path="/".join(slugs),
                    name=str(node.get("name", slugs[-1])),
                    operation_count=int(node.get("operationCount") or 0),
                    version=version,
                )

    yield from walk(navigation.get("navigation") or [], ())


@dataclass(frozen=True)
class Parameter:
    name: str
    location: str
    required: bool = False


@dataclass(frozen=True)
class Operation:
    version: str
    section: str
    method: str
    path: str
    summary: str
    operation_id: str | None = None
    parameters: tuple[Parameter, ...] = ()
    request_schema: str | None = None

    @property
    def key(self) -> str:
        return f"{self.method} {self.path}"

    @property
    def qualified_key(self) -> str:
        return f"{self.version} {self.method} {self.path}"

    def matches(self, pattern: str) -> bool:
        needle = pattern.casefold()
        haystack = f"{self.version} {self.method} {self.path} {self.summary} {self.section}".casefold()
        return needle in haystack

    def as_dict(self) -> dict[str, Any]:
        return {
            "version": self.version,
            "section": self.section,
            "method": self.method,
            "path": self.path,
            "summary": self.summary,
            "operation_id": self.operation_id,
            "parameters": [{"name": p.name, "in": p.location, "required": p.required} for p in self.parameters],
            "request_schema": self.request_schema,
        }


def _request_schema(operation: dict[str, Any]) -> str | None:
    body = operation.get("requestBody")
    if not isinstance(body, dict):
        return None
    content = body.get("content") or {}
    schema = (content.get("application/json") or {}).get("schema") or {}
    ref = schema.get("$ref")
    if isinstance(ref, str):
        return ref.rsplit("/", 1)[-1]
    return str(schema.get("type") or "object")


def _parameters(operation: dict[str, Any]) -> tuple[Parameter, ...]:
    collected: list[Parameter] = []
    for raw in operation.get("parameters") or []:
        if not isinstance(raw, dict) or not raw.get("name"):
            continue
        collected.append(
            Parameter(
                name=str(raw["name"]),
                location=str(raw.get("in") or "query"),
                required=bool(raw.get("required")),
            )
        )
    return tuple(collected)


def operations_from_spec(spec: dict[str, Any], *, version: str, section: str) -> list[Operation]:
    operations: list[Operation] = []
    for path, node in (spec.get("paths") or {}).items():
        if not isinstance(node, dict):
            continue
        for method, operation in node.items():
            if method.lower() not in HTTP_METHODS or not isinstance(operation, dict):
                continue
            operations.append(
                Operation(
                    version=version,
                    section=section,
                    method=method.upper(),
                    path=path,
                    summary=str(operation.get("summary") or "").strip(),
                    operation_id=operation.get("operationId"),
                    parameters=_parameters(operation),
                    request_schema=_request_schema(operation),
                )
            )
    return operations


@dataclass
class Catalog:
    """The downloaded schema set for one or more API versions."""

    cache_dir: Path
    sections: tuple[Section, ...] = ()
    operations: tuple[Operation, ...] = ()
    failures: tuple[str, ...] = field(default_factory=tuple)

    def find(self, pattern: str, *, limit: int = 40) -> list[Operation]:
        if not pattern:
            return list(self.operations[:limit])
        hits = [op for op in self.operations if op.matches(pattern)]
        return hits[:limit]

    def for_path(self, path: str) -> list[Operation]:
        wanted = path.rstrip("/") or "/"
        return [op for op in self.operations if op.path.rstrip("/") == wanted]

    def sections_for(self, version: str) -> list[Section]:
        return [s for s in self.sections if s.version == version]


def _write_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=1, sort_keys=True), encoding="utf-8")


def download_catalog(
    cache_dir: Path,
    *,
    base_url: str = DEFAULT_BASE_URL,
    versions: Sequence[str] = API_VERSIONS,
    workers: int = 8,
    transport: httpx.BaseTransport | None = None,
) -> dict[str, Any]:
    """Download every navigation tree and section spec into ``cache_dir``."""
    cache_dir = Path(cache_dir)
    cache_dir.mkdir(parents=True, exist_ok=True)
    summary: dict[str, Any] = {"cache_dir": str(cache_dir), "versions": {}, "failures": []}

    for version in versions:
        navigation = fetch_json(navigation_url(base_url, version), transport=transport)
        _write_json(cache_dir / NAVIGATION_FILE.format(version=version), navigation)
        sections = list(iter_sections(navigation, version))

        def _fetch(section: Section, version: str = version) -> tuple[Section, dict[str, Any]]:
            spec = fetch_json(section_url(section.path, base_url, version), transport=transport)
            return section, spec

        written = 0
        with ThreadPoolExecutor(max_workers=max(1, workers)) as pool:
            futures = {pool.submit(_fetch, section): section for section in sections}
            for future in as_completed(futures):
                section = futures[future]
                try:
                    _, spec = future.result()
                except (httpx.HTTPError, ValueError, OSError) as exc:
                    summary["failures"].append(f"{version} {section.path}: {exc}")
                    continue
                _write_json(cache_dir / spec_filename(version, section.path), spec)
                written += 1

        summary["versions"][version] = {
            "sections": len(sections),
            "operations": sum(s.operation_count for s in sections),
            "specs_written": written,
        }

    return summary


def load_catalog(cache_dir: Path) -> Catalog:
    """Build a searchable :class:`Catalog` from a previously downloaded cache."""
    cache_dir = Path(cache_dir)
    sections: list[Section] = []
    operations: list[Operation] = []
    failures: list[str] = []

    for nav_path in sorted(cache_dir.glob("navigation-*.json")):
        version = nav_path.stem.split("-", 1)[1]
        try:
            navigation = json.loads(nav_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            failures.append(f"{nav_path.name}: {exc}")
            continue
        for section in iter_sections(navigation, version):
            sections.append(section)
            spec_path = cache_dir / spec_filename(version, section.path)
            if not spec_path.is_file():
                failures.append(f"missing spec: {version} {section.path}")
                continue
            try:
                spec = json.loads(spec_path.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                failures.append(f"{spec_path.name}: {exc}")
                continue
            operations.extend(operations_from_spec(spec, version=version, section=section.path))

    return Catalog(
        cache_dir=cache_dir,
        sections=tuple(sections),
        operations=tuple(operations),
        failures=tuple(failures),
    )
