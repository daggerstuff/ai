"""Stage 0 ingest router — normalizes multi-source raw data into JSONL shards.

Routes by ``source_type`` to an extractor, attaches provenance via
``provenance.build_provenance``, enforces the SPDX license gate, and emits
raw shards to ``ai/data/raw/<source_type>/`` at 50K records per shard.

Heavy extractors (trafilatura, python-docx) are imported lazily at call
sites so this module imports cleanly even when those deps are absent.

See ``docs/training-pipeline-blueprint-2026-08-10.md`` Appendix B.1.
"""

from __future__ import annotations

import hashlib
import json
import logging
import re
from collections.abc import Callable, Iterable, Mapping
from dataclasses import dataclass, field
from datetime import UTC, datetime
from enum import Enum
from pathlib import Path
from typing import Any
from urllib.parse import urlparse
from urllib.robotparser import RobotFileParser

import httpx

from training.book_pdf_converter import _chunk_text, _extract_epub, _extract_pdf
from training.provenance import ProvenanceOptions, build_provenance, validate_license

logger = logging.getLogger("ingest_router")

RETRYABLE_HTTP_STATUS_CODES: frozenset[int] = frozenset({429, 500, 502, 503, 504})
NEMO_RETRY_DELAYS: tuple[int, ...] = (1, 2, 4)
RECORDS_PER_SHARD = 50_000
DEFAULT_USER_AGENT = "pixelated-ingest/1.0 (+https://pixelated-empathy.org/bot)"
WEB_REQUEST_INTERVAL_SECONDS = 2.0
WEB_TIMEOUT_SECONDS = 30.0
API_TIMEOUT_SECONDS = 60.0

THERAPIST_TURN = "Therapist"
PATIENT_TURN = "Patient"
SPEAKER_TURN_RE = re.compile(
    r"^\s*(Therapist|Patient|Counselor|Client|Doctor|Dr\.?)\s*[:\-]\s*(.*)$",
    re.IGNORECASE | re.MULTILINE,
)


class SourceType(str, Enum):
    WEB = "web"
    DOCX = "docx"
    PDF = "pdf"
    EPUB = "epub"
    API = "api"

    @classmethod
    def from_value(cls, value: str) -> SourceType:
        normalized = value.strip().lower()
        try:
            return cls(normalized)
        except ValueError as exc:
            allowed = ", ".join(s.value for s in cls)
            raise ValueError(f"Unsupported source_type '{value}'. Expected one of: {allowed}") from exc


@dataclass
class IngestRecord:
    """One normalized raw record emitted by an extractor.

    ``payload`` is the raw content + any source-specific fields.  Provenance
    is attached at emit time so callers can override license/metadata before
    the shard writer stamps the record.
    """

    source_url: str
    source_type: SourceType
    text: str
    payload: Mapping[str, Any] = field(default_factory=dict)
    license_id: str = "NOASSERTION"
    metadata: Mapping[str, Any] | None = None
    transformations: tuple[str, ...] = ()

    def with_provenance(self) -> dict[str, Any]:
        provenance = build_provenance(
            self.source_url,
            self.source_type.value,
            options=ProvenanceOptions(
                license_id=self.license_id,
                transformations=self.transformations,
            ),
            metadata=dict(self.metadata) if self.metadata else None,
        )
        return {
            "text": self.text,
            "payload": dict(self.payload),
            "source_url": self.source_url,
            "source_type": self.source_type.value,
            "provenance": provenance,
        }


# ---------------------------------------------------------------------------
# Speaker-turn chunking for therapy transcripts in DOCX/structured docs.
# ---------------------------------------------------------------------------


def chunk_by_speaker_turn(text: str, *, min_chunk_chars: int = 120) -> list[str]:
    """Chunk text on Therapist/Patient turn boundaries.

    Groups consecutive turns under the same speaker (normalizing counselor/
    doctor/dr aliases to Therapist, everything else to Patient) into one
    chunk.  Falls back to a single chunk when no turn markers are present
    (e.g. a narrative DOCX with no speaker prefixes).
    """

    if not text or not text.strip():
        return []

    def _normalize(label: str) -> str:
        if label in {THERAPIST_TURN, "Counselor", "Doctor", "Dr"}:
            return THERAPIST_TURN
        return PATIENT_TURN

    matches = list(SPEAKER_TURN_RE.finditer(text))
    if not matches:
        return [text.strip()]

    chunks: list[str] = []
    current_label: str | None = None
    current_parts: list[str] = []

    def flush() -> None:
        if current_label is None or not current_parts:
            return
        joined = "\n".join(p for p in current_parts if p).strip()
        if joined:
            chunks.append(f"{current_label}: {joined}")

    for match in matches:
        label = _normalize(match.group(1))
        content = match.group(2).strip()
        if label != current_label:
            flush()
            current_parts = []
            current_label = label
        if content:
            current_parts.append(content)
    flush()
    return [c for c in chunks if len(c) >= min_chunk_chars] or chunks


# ---------------------------------------------------------------------------
# Domain rate-limiting for web fetches (1 req per 2s default, obey Crawl-Delay).
# ---------------------------------------------------------------------------


class _DomainGate:
    """Per-domain last-request timestamp + interval, honoring robots.txt Crawl-Delay."""

    def __init__(self) -> None:
        self._last_seen: dict[str, datetime] = {}
        self._interval: dict[str, float] = {}
        self._robots_cache: dict[str, RobotFileParser] = {}

    def _robots(self, url: str, client: httpx.AsyncClient) -> RobotFileParser:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        cached = self._robots_cache.get(host)
        if cached is not None:
            return cached
        parser = RobotFileParser()
        parser.set_url(f"{host}/robots.txt")
        # Fetch live so async context can populate rules.  Best-effort: a
        # failure leaves the parser with no rules, which our caller treats
        # as disallow-by-default for safety.
        try:
            resp = httpx.get(f"{host}/robots.txt", timeout=WEB_TIMEOUT_SECONDS, headers={"User-Agent": DEFAULT_USER_AGENT})
            if resp.status_code == 200:
                parser.parse(resp.text.splitlines())
            crawl_delay = parser.crawl_delay(DEFAULT_USER_AGENT)
            if crawl_delay:
                self._interval[host] = max(float(crawl_delay), WEB_REQUEST_INTERVAL_SECONDS)
            else:
                self._interval[host] = WEB_REQUEST_INTERVAL_SECONDS
        except httpx.HTTPError:
            self._interval[host] = WEB_REQUEST_INTERVAL_SECONDS
        self._robots_cache[host] = parser
        return parser

    def can_fetch(self, url: str, client: httpx.AsyncClient) -> bool:
        parser = self._robots(url, client)
        return parser.can_fetch(DEFAULT_USER_AGENT, url)

    def interval(self, url: str) -> float:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        return self._interval.get(host, WEB_REQUEST_INTERVAL_SECONDS)

    def mark(self, url: str) -> None:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        self._last_seen[host] = datetime.now(UTC)

    def wait_seconds(self, url: str, now: datetime) -> float:
        parsed = urlparse(url)
        host = f"{parsed.scheme}://{parsed.netloc}"
        last = self._last_seen.get(host)
        if last is None:
            return 0.0
        elapsed = (now - last).total_seconds()
        interval = self.interval(url)
        return max(0.0, interval - elapsed)


# ---------------------------------------------------------------------------
# Extractors — each returns IngestRecord(s) for a single source.
# ---------------------------------------------------------------------------


_extractor_registry: dict[SourceType, Any] = {}


def register(source_type: SourceType) -> Any:
    def decorator(fn: Any) -> Any:
        _extractor_registry[source_type] = fn
        return fn

    return decorator


async def _web_fetch(url: str, gate: _DomainGate, client: httpx.AsyncClient) -> str:
    wait = gate.wait_seconds(url, datetime.now(UTC))
    if wait > 0:
        await _async_sleep(wait)
    gate.mark(url)
    headers = {"User-Agent": DEFAULT_USER_AGENT}
    resp = await client.get(url, headers=headers, follow_redirects=True)
    if resp.status_code in RETRYABLE_HTTP_STATUS_CODES:
        raise RetryableHTTPError(url, resp.status_code)
    resp.raise_for_status()
    return resp.text


async def _async_sleep(seconds: float) -> None:
    import anyio  # lazy — httpx pulls anyio transitively

    await anyio.sleep(seconds)


class RetryableHTTPError(Exception):
    def __init__(self, url: str, status_code: int) -> None:
        super().__init__(f"Retryable HTTP {status_code} for {url}")
        self.url = url
        self.status_code = status_code


@register(SourceType.WEB)
async def _extract_web(
    *,
    source_url: str,
    gate: _DomainGate,
    client: httpx.AsyncClient,
    license_id: str = "NOASSERTION",
    metadata: Mapping[str, Any] | None = None,
    **_opts: Any,
) -> list[IngestRecord]:
    if not gate.can_fetch(source_url, client):
        logger.warning("robots.txt disallows fetch: %s", source_url)
        return []
    try:
        from trafilatura import extract as trafilatura_extract
    except ImportError as exc:  # pragma: no cover — dep must be installed for web ingest
        raise ImportError(
            "trafilatura is required for web ingestion. Install with: uv sync --extra ingest-web"
        ) from exc
    html = await _web_fetch(source_url, gate, client)
    text = trafilatura_extract(html, include_comments=False, include_tables=True) or ""
    if not text.strip():
        logger.info("No extractable text from %s", source_url)
        return []
    fetch_meta = {
        "fetch_user_agent": DEFAULT_USER_AGENT,
        "crawl_delay_seconds": gate.interval(source_url),
        **(dict(metadata) if metadata else {}),
    }
    return [
        IngestRecord(
            source_url=source_url,
            source_type=SourceType.WEB,
            text=text,
            payload={"html_sha256": hashlib.sha256(html.encode("utf-8")).hexdigest()},
            license_id=license_id,
            metadata=fetch_meta,
            transformations=("trafilatura_extract",),
        )
    ]


@register(SourceType.DOCX)
async def _extract_docx(
    *,
    source_url: str,
    path: Path | None = None,
    license_id: str = "NOASSERTION",
    metadata: Mapping[str, Any] | None = None,
    chunk_size: int = 4000,
    chunk_by_turns: bool = True,
    **_opts: Any,
) -> list[IngestRecord]:
    if path is None:
        raise ValueError("DOCX ingestion requires a local `path` (download first)")
    try:
        from docx import Document
    except ImportError as exc:  # pragma: no cover — dep optional
        raise ImportError(
            "python-docx is required for DOCX ingestion. Install with: uv sync --extra ingest-docx"
        ) from exc
    document = Document(str(path))
    paragraphs = [p.text for p in document.paragraphs if p.text and p.text.strip()]
    full_text = "\n".join(paragraphs)
    if chunk_by_turns:
        chunks = chunk_by_speaker_turn(full_text)
        if not chunks:
            chunks = _chunk_text(full_text, chunk_size=chunk_size)
    else:
        chunks = _chunk_text(full_text, chunk_size=chunk_size)
    records: list[IngestRecord] = []
    for index, chunk in enumerate(chunks):
        records.append(
            IngestRecord(
                source_url=source_url,
                source_type=SourceType.DOCX,
                text=chunk,
                payload={"part_index": index, "part_count": len(chunks), "docx_path": str(path)},
                license_id=license_id,
                metadata=dict(metadata) if metadata else None,
                transformations=("python-docx", "speaker_turn_chunk" if chunk_by_turns else "fixed_chunk"),
            )
        )
    return records


@register(SourceType.PDF)
async def _extract_pdf(
    *,
    source_url: str,
    path: Path | None = None,
    license_id: str = "NOASSERTION",
    metadata: Mapping[str, Any] | None = None,
    chunk_size: int = 4000,
    **_opts: Any,
) -> list[IngestRecord]:
    if path is None:
        raise ValueError("PDF ingestion requires a local `path`")
    full_text = _extract_pdf(path)
    chunks = _chunk_text(full_text, chunk_size=chunk_size)
    records: list[IngestRecord] = []
    for index, chunk in enumerate(chunks):
        records.append(
            IngestRecord(
                source_url=source_url,
                source_type=SourceType.PDF,
                text=chunk,
                payload={"part_index": index, "part_count": len(chunks), "pdf_path": str(path)},
                license_id=license_id,
                metadata=dict(metadata) if metadata else None,
                transformations=("book_pdf_converter._extract_pdf", "fixed_chunk"),
            )
        )
    return records


@register(SourceType.EPUB)
async def _extract_epub_(
    *,
    source_url: str,
    path: Path | None = None,
    license_id: str = "NOASSERTION",
    metadata: Mapping[str, Any] | None = None,
    chunk_size: int = 4000,
    **_opts: Any,
) -> list[IngestRecord]:
    if path is None:
        raise ValueError("EPUB ingestion requires a local `path`")
    full_text = _extract_epub(path)
    chunks = _chunk_text(full_text, chunk_size=chunk_size)
    records: list[IngestRecord] = []
    for index, chunk in enumerate(chunks):
        records.append(
            IngestRecord(
                source_url=source_url,
                source_type=SourceType.EPUB,
                text=chunk,
                payload={"part_index": index, "part_count": len(chunks), "epub_path": str(path)},
                license_id=license_id,
                metadata=dict(metadata) if metadata else None,
                transformations=("book_pdf_converter._extract_epub", "fixed_chunk"),
            )
        )
    return records


@register(SourceType.API)
async def _extract_api(
    *,
    source_url: str,
    gate: _DomainGate,
    client: httpx.AsyncClient,
    headers: Mapping[str, str] | None = None,
    params: Mapping[str, str] | None = None,
    json_body: Mapping[str, Any] | None = None,
    method: str = "GET",
    license_id: str = "NOASSERTION",
    metadata: Mapping[str, Any] | None = None,
    **_opts: Any,
) -> list[IngestRecord]:
    wait = gate.wait_seconds(source_url, datetime.now(UTC))
    if wait > 0:
        await _async_sleep(wait)
    gate.mark(source_url)
    request_headers = {"User-Agent": DEFAULT_USER_AGENT}
    if headers:
        request_headers.update(headers)
    last_exc: Exception | None = None
    for attempt, delay in enumerate(NEMO_RETRY_DELAYS):
        try:
            resp = await client.request(
                method,
                source_url,
                headers=request_headers,
                params=dict(params) if params else None,
                json=dict(json_body) if json_body else None,
                timeout=API_TIMEOUT_SECONDS,
                follow_redirects=True,
            )
            if resp.status_code in RETRYABLE_HTTP_STATUS_CODES:
                last_exc = RetryableHTTPError(source_url, resp.status_code)
                logger.warning("API %s retryable %d (attempt %d)", source_url, resp.status_code, attempt + 1)
                await _async_sleep(float(delay))
                continue
            resp.raise_for_status()
            api_meta = {
                "request_method": method,
                "request_params": dict(params) if params else {},
                **(dict(metadata) if metadata else {}),
            }
            payload: dict[str, Any] = {"status_code": resp.status_code}
            try:
                payload["json"] = resp.json()
                text = json.dumps(payload["json"], ensure_ascii=False, sort_keys=True)
            except json.JSONDecodeError:
                payload["text"] = resp.text
                text = resp.text
            return [
                IngestRecord(
                    source_url=source_url,
                    source_type=SourceType.API,
                    text=text,
                    payload=payload,
                    license_id=license_id,
                    metadata=api_meta,
                    transformations=("httpx_api", "json_or_text"),
                )
            ]
        except httpx.HTTPError as exc:
            last_exc = exc
            logger.warning("API %s HTTP error (attempt %d): %s", source_url, attempt + 1, exc)
            await _async_sleep(float(delay))
    raise last_exc if last_exc else RuntimeError(f"API ingestion failed for {source_url}")


# ---------------------------------------------------------------------------
# Public entry: route + emit shards.
# ---------------------------------------------------------------------------


async def route_ingest(
    source_type: str,
    source_url: str,
    *,
    raw_dir: Path | str = "ai/data/raw",
    license_id: str = "NOASSERTION",
    metadata: Mapping[str, Any] | None = None,
    client: httpx.AsyncClient | None = None,
    **opts: Any,
) -> list[Path]:
    """Route one source to its extractor, emit raw JSONL shards.

    Raises ValueError on unsupported source_type, unlicensed content, or
    missing required path for document sources.
    """

    validate_license(license_id)
    stype = SourceType.from_value(source_type)
    extractor = _extractor_registry.get(stype)
    if extractor is None:
        raise ValueError(f"No extractor registered for {stype.value}")
    owns_client = client is None
    client = client or httpx.AsyncClient(timeout=httpx.Timeout(WEB_TIMEOUT_SECONDS))
    gate = _DomainGate()
    try:
        records = await extractor(
            source_url=source_url,
            gate=gate,
            client=client,
            license_id=license_id,
            metadata=metadata,
            **opts,
        )
    finally:
        if owns_client:
            await client.aclose()
    if not records:
        logger.info("No records extracted for %s %s", stype.value, source_url)
        return []
    return _emit_shards(records, stype, Path(raw_dir))


def _emit_shards(records: Iterable[IngestRecord], stype: SourceType, raw_dir: Path) -> list[Path]:
    shard_dir = raw_dir / stype.value
    shard_dir.mkdir(parents=True, exist_ok=True)
    shard_paths: list[Path] = []
    current: list[dict[str, Any]] = []
    shard_index = 0

    def flush_current() -> None:
        nonlocal shard_index
        if not current:
            return
        shard_path = shard_dir / f"shard-{shard_index:05d}.jsonl"
        with open(shard_path, "w", encoding="utf-8") as f:
            for record in current:
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        shard_paths.append(shard_path)
        logger.info("Wrote %d records to %s", len(current), shard_path)
        current.clear()
        shard_index += 1

    for record in records:
        current.append(record.with_provenance())
        if len(current) >= RECORDS_PER_SHARD:
            flush_current()
    flush_current()
    return shard_paths


def iter_shard_records(shard_path: Path | str) -> Iterable[Mapping[str, Any]]:
    """Read a shard back as a stream of provenance-bearing records."""

    with open(shard_path, encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if line:
                yield json.loads(line)
