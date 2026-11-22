"""Store eval artifacts in Cloudflare R2 instead of trusting the OVH volume.

The OVH object-storage bucket is mounted at ``/workspace`` and synced when a job
finalises. That push failed on all three eval jobs (``SYNC_FAILED :
DATASYNC_RETRY_ERROR``), so the only surviving copy of each run was the job log.
R2 is S3-compatible, so the runner can write its results straight there and the
volume stops mattering.

Credentials never go into a job spec. The submitter signs a presigned URL per
object and hands the job a compact map of ``key -> url``; the job then does a
plain HTTP ``PUT`` with :func:`put_file`, which needs no credentials.
Presigned URLs are scoped to one object and expire, so a leaked job spec leaks
nothing but the ability to overwrite one result file before it expires.

Usage::

    python -m training.launch.r2_sync list --prefix evals/v22
    python -m training.launch.r2_sync upload --source out/run --prefix evals/v22/tag/run
    python -m training.launch.r2_sync download --prefix evals/v22/tag --destination /tmp/r2
    python -m training.launch.r2_sync presign --keys run/results.jsonl --method put
    python -m training.launch.r2_sync put-file --url "$URL" --path run/results.jsonl
"""

from __future__ import annotations

import argparse
import base64
import gzip
import json
import mimetypes
import os
import sys
import time
import urllib.error
import urllib.request
from collections.abc import Callable
from pathlib import Path
from typing import Any

REQUIRED_ENV = ("R2_ACCESS_KEY_ID", "R2_SECRET_ACCESS_KEY", "R2_ENDPOINT")
DEFAULT_BUCKET_ENV = "R2_BUCKET"
DEFAULT_BUCKET = "training"
# Longer than any eval job, short enough that a leaked URL stops working.
PRESIGN_EXPIRES_SECONDS = 12 * 3600
UPLOAD_ATTEMPTS = 3
# Bytecode and editor droppings must never be mirrored into the bucket.
SKIP_PARTS = frozenset({"__pycache__", ".git", ".pytest_cache", ".ruff_cache"})
SKIP_SUFFIXES = frozenset({".pyc", ".pyo"})

# R2 ignores most of these, but a browser or a `curl -O` should still get the
# right file, and JSON-lines has no mimetypes entry on a bare container image.
CONTENT_TYPES = {
    ".jsonl": "application/x-ndjson",
    ".json": "application/json",
    ".md": "text/markdown",
    ".txt": "text/plain",
    ".sh": "text/x-shellscript",
}


def _read_env_file(path: Path) -> dict[str, str]:
    """Parse the ``KEY='value'`` subset of a dotenv file.

    The repo stores every value single-quoted, which is why a naive regex over
    ``.env`` hands back the quotes and the API rejects the signature.
    """
    values: dict[str, str] = {}
    for line in path.read_text().splitlines():
        stripped = line.strip()
        if not stripped or stripped.startswith("#") or "=" not in stripped:
            continue
        key, _, raw = stripped.partition("=")
        values[key.strip()] = raw.strip().strip("'\"")
    return values


def credentials(*, env_file: Path | None = None) -> dict[str, str]:
    """The three R2 settings, from the environment or a dotenv file."""
    settings = {name: os.environ[name] for name in REQUIRED_ENV if name in os.environ}
    if env_file is not None and env_file.exists():
        stored = _read_env_file(env_file)
        for name in REQUIRED_ENV:
            settings.setdefault(name, stored.get(name, ""))
    missing = [name for name in REQUIRED_ENV if not settings.get(name)]
    if missing:
        raise RuntimeError(f"R2 is not configured: {', '.join(missing)} is unset")
    settings.setdefault(DEFAULT_BUCKET_ENV, os.environ.get(DEFAULT_BUCKET_ENV, DEFAULT_BUCKET))
    return settings


def client(*, env_file: Path | None = None) -> Any:
    """A boto3 S3 client pointed at R2.

    boto3 is imported lazily: the GPU image runs this module for the
    stdlib-only put-file/fetch-file path and has no SDK installed, so an
    eager import would break result upload from the job even before
    credentials are ever needed.
    """
    import boto3
    from botocore.config import Config

    settings = credentials(env_file=env_file)
    return boto3.client(
        "s3",
        endpoint_url=settings["R2_ENDPOINT"],
        aws_access_key_id=settings["R2_ACCESS_KEY_ID"],
        aws_secret_access_key=settings["R2_SECRET_ACCESS_KEY"],
        # R2 has no regions; "auto" is the documented placeholder.
        region_name="auto",
        config=Config(
            signature_version="s3v4",
            retries={"max_attempts": 5, "mode": "standard"},
            connect_timeout=15,
            read_timeout=120,
        ),
    )


def content_type(path: Path) -> str:
    return CONTENT_TYPES.get(path.suffix) or mimetypes.guess_type(path.name)[0] or "application/octet-stream"


def _skip(path: Path) -> bool:
    return any(part in SKIP_PARTS for part in path.parts) or path.suffix in SKIP_SUFFIXES


def upload_dir(
    source: Path,
    *,
    bucket: str,
    prefix: str,
    s3_client: Any | None = None,
    skip_unchanged: bool = False,
) -> list[str]:
    """Mirror a local directory under ``prefix``. Returns the uploaded keys.

    ``skip_unchanged`` compares against the objects already under ``prefix`` and
    uploads only what is missing or has a different size. The input bundle holds
    two 344 MB adapters, so re-sending them on every submit is not free.

    Callers that need a non-default ``env_file`` build the client with
    :func:`client` and pass it in: credentials are resolved in exactly one
    place, and nothing here reads the environment behind the caller's back.
    """
    s3 = s3_client if s3_client is not None else client()
    base = prefix.strip("/")
    existing: dict[str, int] = {}
    if skip_unchanged:
        existing = {entry["key"]: entry["size"] for entry in list_prefix(bucket=bucket, prefix=base, s3_client=s3)}
    uploaded: list[str] = []
    for path in sorted(source.rglob("*")):
        if not path.is_file() or _skip(path.relative_to(source)):
            continue
        key = f"{base}/{path.relative_to(source).as_posix()}" if base else path.relative_to(source).as_posix()
        if skip_unchanged and existing.get(key) == path.stat().st_size:
            continue
        s3.put_object(Bucket=bucket, Key=key, Body=path.read_bytes(), ContentType=content_type(path))
        uploaded.append(key)
    return uploaded


def download_prefix(
    *,
    bucket: str,
    prefix: str,
    destination: Path,
    s3_client: Any | None = None,
) -> list[Path]:
    """Pull every object under ``prefix`` into ``destination``, mirroring keys."""
    s3 = s3_client if s3_client is not None else client()
    base = prefix.strip("/")
    written: list[Path] = []
    token: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": base + "/" if base else ""}
        if token:
            request["ContinuationToken"] = token
        page = s3.list_objects_v2(**request)
        for entry in page.get("Contents", []):
            key = entry["Key"]
            relative = key[len(base) + 1 :] if base and key.startswith(base + "/") else key
            target = destination / relative
            target.parent.mkdir(parents=True, exist_ok=True)
            body = s3.get_object(Bucket=bucket, Key=key)["Body"].read()
            target.write_bytes(body)
            written.append(target)
        if not page.get("IsTruncated"):
            return written
        token = page.get("NextContinuationToken")


def list_prefix(
    *,
    bucket: str,
    prefix: str,
    s3_client: Any | None = None,
) -> list[dict[str, Any]]:
    """Every object under ``prefix`` as ``{key, size}``."""
    s3 = s3_client if s3_client is not None else client()
    base = prefix.strip("/")
    found: list[dict[str, Any]] = []
    token: str | None = None
    while True:
        request: dict[str, Any] = {"Bucket": bucket, "Prefix": base + "/" if base else ""}
        if token:
            request["ContinuationToken"] = token
        page = s3.list_objects_v2(**request)
        found += [{"key": entry["Key"], "size": entry["Size"]} for entry in page.get("Contents", [])]
        if not page.get("IsTruncated"):
            return found
        token = page.get("NextContinuationToken")


def presign(
    keys: list[str],
    *,
    bucket: str,
    method: str = "put",
    expires: int = PRESIGN_EXPIRES_SECONDS,
    s3_client: Any | None = None,
) -> dict[str, str]:
    """``key -> presigned url`` for a known set of objects."""
    operation = "put_object" if method == "put" else "get_object"
    s3 = s3_client if s3_client is not None else client()
    return {
        key: s3.generate_presigned_url(operation, Params={"Bucket": bucket, "Key": key}, ExpiresIn=expires)
        for key in keys
    }


def put_file(url: str, path: Path, *, attempts: int = UPLOAD_ATTEMPTS, timeout: float = 300.0) -> int:
    """PUT a file to a presigned URL with the standard library only.

    ``Content-Length`` is set explicitly: without it urllib switches to chunked
    transfer encoding, which S3 rejects for a presigned PUT.
    """
    payload = path.read_bytes()
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            data=payload,
            method="PUT",
            headers={
                "Content-Type": content_type(path),
                "Content-Length": str(len(payload)),
                "User-Agent": "pixelated-empathy-r2-sync/1.0",
            },
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                return int(handle.status)
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
            last = error
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"upload failed after {attempts} attempts: {last}")


def fetch_file(url: str, destination: Path, *, attempts: int = UPLOAD_ATTEMPTS, timeout: float = 600.0) -> Path:
    """GET a presigned URL to a local path with the standard library only."""
    destination.parent.mkdir(parents=True, exist_ok=True)
    last: Exception | None = None
    for attempt in range(attempts):
        request = urllib.request.Request(
            url,
            headers={"User-Agent": "pixelated-empathy-r2-sync/1.0"},
        )
        try:
            with urllib.request.urlopen(request, timeout=timeout) as handle:
                destination.write_bytes(handle.read())
                return destination
        except (urllib.error.URLError, urllib.error.HTTPError, TimeoutError, OSError) as error:
            last = error
            if attempt + 1 < attempts:
                time.sleep(2.0 * (attempt + 1))
    raise RuntimeError(f"download failed after {attempts} attempts: {last}")


def encode_map(payload: dict[str, Any]) -> str:
    """Gzip and base64 a map so it fits in one job environment variable."""
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return base64.urlsafe_b64encode(gzip.compress(raw, 9)).decode()


def decode_map(text: str) -> dict[str, Any]:
    decoded: dict[str, Any] = json.loads(gzip.decompress(base64.urlsafe_b64decode(text.encode())))
    return decoded


def run_dir_uploads(run_dir: Path, urls: dict[str, str]) -> list[tuple[Path, str]]:
    """Pair each file in ``run_dir`` with the URL signed for it, if any.

    The map is keyed ``<run-dir-name>/<file-name>``, i.e. relative to the run
    root, so the same map works wherever the job wrote the directory.
    """
    pairs: list[tuple[Path, str]] = []
    for path in sorted(run_dir.iterdir()):
        if not path.is_file():
            continue
        url = urls.get(f"{run_dir.name}/{path.name}")
        if url:
            pairs.append((path, url))
    return pairs


def upload_run(run_dir: Path, urls: dict[str, str]) -> list[str]:
    """PUT every signed file in a finished run directory. Returns the keys sent."""
    sent: list[str] = []
    for path, url in run_dir_uploads(run_dir, urls):
        put_file(url, path)
        sent.append(f"{run_dir.name}/{path.name}")
    return sent


def upload_map_from_env() -> dict[str, str]:
    """The ``upload`` half of ``R2_MAP_B64``: run-relative key -> presigned URL."""
    text = os.environ.get("R2_MAP_B64", "")
    if not text:
        raise RuntimeError("R2_MAP_B64 is unset")
    upload = decode_map(text).get("upload", {})
    if not isinstance(upload, dict):
        raise RuntimeError("R2_MAP_B64 has no upload map")
    return {str(key): str(url) for key, url in upload.items()}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--env-file", type=Path, default=Path(".env") if Path(".env").exists() else None)
    parser.add_argument("--bucket", default=None)
    sub = parser.add_subparsers(dest="command", required=True)

    listing = sub.add_parser("list", help="list objects under a prefix")
    listing.add_argument("--prefix", default="")

    upload = sub.add_parser("upload", help="mirror a local directory into the bucket")
    upload.add_argument("--source", type=Path, required=True)
    upload.add_argument("--prefix", required=True)

    download = sub.add_parser("download", help="pull a prefix into a local directory")
    download.add_argument("--prefix", required=True)
    download.add_argument("--destination", type=Path, required=True)

    sign = sub.add_parser("presign", help="sign one URL per key")
    sign.add_argument("--keys", nargs="+", required=True)
    sign.add_argument("--method", choices=("put", "get"), default="put")
    sign.add_argument("--expires", type=int, default=PRESIGN_EXPIRES_SECONDS)

    put = sub.add_parser("put-file", help="PUT one file to a presigned URL (no SDK needed)")
    put.add_argument("--url", required=True)
    put.add_argument("--path", type=Path, required=True)

    fetch = sub.add_parser("fetch-file", help="GET one presigned URL to a local path")
    fetch.add_argument("--url", required=True)
    fetch.add_argument("--path", type=Path, required=True)

    decode = sub.add_parser("decode-map", help="write R2_MAP_B64 to a JSON file")
    decode.add_argument("--out", type=Path, required=True)

    upload_run_parser = sub.add_parser("upload-run", help="PUT a finished run dir using R2_MAP_B64")
    upload_run_parser.add_argument("--run-dir", type=Path, required=True)
    return parser


def _signed_target(args: argparse.Namespace) -> tuple[Any, str]:
    """The client and bucket the credential-bearing subcommands need."""
    settings = credentials(env_file=args.env_file)
    return client(env_file=args.env_file), args.bucket or settings[DEFAULT_BUCKET_ENV]


def _run_decode_map(args: argparse.Namespace) -> int:
    text = os.environ.get("R2_MAP_B64", "")
    if not text:
        sys.stderr.write("R2_MAP_B64 is unset\n")
        return 2
    args.out.write_text(json.dumps(decode_map(text), indent=2, sort_keys=True) + "\n")
    return 0


def _run_upload_run(args: argparse.Namespace) -> int:
    try:
        urls = upload_map_from_env()
    except RuntimeError as error:
        # The stdout RECORD lines are the fallback copy, so a missing map
        # must not fail the job that produced the results.
        sys.stderr.write(f"{error}; results stay in the log only\n")
        return 2
    sent = upload_run(args.run_dir, urls)
    sys.stdout.write(json.dumps({"run": args.run_dir.name, "uploaded": sent}, sort_keys=True) + "\n")
    return 0


def _run_put_file(args: argparse.Namespace) -> int:
    put_file(args.url, args.path)
    sys.stdout.write(f"uploaded {args.path.name}\n")
    return 0


def _run_fetch_file(args: argparse.Namespace) -> int:
    fetch_file(args.url, args.path)
    sys.stdout.write(f"fetched {args.path}\n")
    return 0


def _run_list(args: argparse.Namespace) -> int:
    s3, bucket = _signed_target(args)
    for entry in list_prefix(bucket=bucket, prefix=args.prefix, s3_client=s3):
        sys.stdout.write(f"{entry['size']:>12}  {entry['key']}\n")
    return 0


def _run_upload(args: argparse.Namespace) -> int:
    s3, bucket = _signed_target(args)
    keys = upload_dir(args.source, bucket=bucket, prefix=args.prefix, s3_client=s3)
    sys.stdout.write(f"uploaded {len(keys)} objects to {bucket}/{args.prefix}\n")
    return 0


def _run_download(args: argparse.Namespace) -> int:
    s3, bucket = _signed_target(args)
    paths = download_prefix(bucket=bucket, prefix=args.prefix, destination=args.destination, s3_client=s3)
    sys.stdout.write(f"downloaded {len(paths)} objects into {args.destination}\n")
    return 0


def _run_presign(args: argparse.Namespace) -> int:
    s3, bucket = _signed_target(args)
    urls = presign(args.keys, bucket=bucket, method=args.method, expires=args.expires, s3_client=s3)
    sys.stdout.write(json.dumps(urls, indent=2, sort_keys=True) + "\n")
    return 0


# One handler per subcommand, so main() never grows a branch per command and a
# new subcommand cannot be added without wiring it here.
COMMANDS: dict[str, Callable[[argparse.Namespace], int]] = {
    "decode-map": _run_decode_map,
    "upload-run": _run_upload_run,
    "put-file": _run_put_file,
    "fetch-file": _run_fetch_file,
    "list": _run_list,
    "upload": _run_upload,
    "download": _run_download,
    "presign": _run_presign,
}


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    handler = COMMANDS.get(args.command)
    if handler is None:
        sys.stderr.write(f"unknown command: {args.command}\n")
        return 2
    return handler(args)


if __name__ == "__main__":
    raise SystemExit(main())
