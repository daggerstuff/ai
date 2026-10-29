#!/usr/bin/env python3
"""Sync Fireworks fine-tuned adapters to local disk and track export access.

Two independent gates block `firectl model download`:

  1. Per-model allowlist. Fireworks gates weight export per model ID. A blocked
     model returns ``FailedPrecondition: model downloading is restricted``.
     There is no self-serve switch and the restriction is invisible in the
     model record -- the REST and gRPC metadata for a blocked model are
     identical to an allowed one -- so the only fix is a support request
     naming the model IDs. ``request`` emits that email from the ledger.
  2. Session-only auth. API keys cannot download (``PermissionDenied: API keys
     cannot download models``); only the Cognito session written by
     ``firectl signin`` works. This tool therefore never passes ``--api-key``
     and reports session expiry up front via ``check-auth``.

An API key is still used, but only to list models over REST -- never to export.

Usage:
  python ai/training/fireworks_model_sync.py check-auth
  python ai/training/fireworks_model_sync.py status
  python ai/training/fireworks_model_sync.py sync [--only NAME ...]
  python ai/training/fireworks_model_sync.py normalize
  python ai/training/fireworks_model_sync.py verify
  python ai/training/fireworks_model_sync.py request
"""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import os
import shutil
import struct
import subprocess
import sys
import time
import urllib.request
from pathlib import Path
from typing import Any

ROOT = Path("/home/vivi/pixelated")
MODELS_DIR = ROOT / "ai/training/models/fireworks"
LEDGER = MODELS_DIR / ".sync-ledger.json"
AUTH_INI = Path.home() / ".fireworks/auth.ini"
API_BASE = "https://api.fireworks.ai/v1"
FIRECTL = shutil.which("firectl") or "firectl"

DOWNLOAD_TIMEOUT = 3600

# Ledger states that mean "not usable locally, and retrying will not fix it".
BLOCKING = ("restricted", "auth", "not_found", "other", "broken")


def now() -> str:
    return time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())


# --------------------------------------------------------------------------
# auth / api
# --------------------------------------------------------------------------


def _env_value(key: str) -> str | None:
    if os.environ.get(key):
        return os.environ[key]
    env_file = ROOT / ".env"
    if not env_file.exists():
        return None
    for line in env_file.read_text().splitlines():
        s = line.strip()
        if s and not s.startswith("#") and "=" in s:
            k, _, value = s.partition("=")
            if k.strip() == key:
                return value.strip().strip('"').strip("'")
    return None


def api_key() -> str:
    key = _env_value("FIREWORKS_API_KEY")
    if not key:
        raise SystemExit("FIREWORKS_API_KEY not set (env or .env); needed to list models")
    return key


def api_get(path: str) -> dict[str, Any]:
    req = urllib.request.Request(API_BASE + path)
    req.add_header("Authorization", "Bearer " + api_key())
    with urllib.request.urlopen(req, timeout=60) as r:
        return json.loads(r.read().decode())


def account() -> str:
    override = os.environ.get("FIREWORKS_ACCOUNT")
    if override:
        return override
    accounts = api_get("/accounts").get("accounts") or []
    if not accounts:
        raise SystemExit("no Fireworks accounts visible to this API key")
    return accounts[0]["name"]


def list_models() -> list[dict[str, Any]]:
    acct = account()
    out: list[dict[str, Any]] = []
    token = ""
    while True:
        suffix = f"?pageSize=200&pageToken={token}" if token else "?pageSize=200"
        page = api_get(f"/{acct}/models{suffix}")
        out.extend(page.get("models") or [])
        token = page.get("nextPageToken") or ""
        if not token:
            return out


def auth_state() -> dict[str, Any]:
    """Report the Cognito session that `firectl model download` actually uses."""
    if not AUTH_INI.exists():
        return {"present": False, "reason": f"{AUTH_INI} missing -- run `firectl signin`"}
    # firectl writes a flat `key = value` file with no [section] headers.
    values: dict[str, str] = {}
    for raw_line in AUTH_INI.read_text().splitlines():
        line = raw_line.strip()
        if not line or line.startswith(("#", ";", "[")) or "=" not in line:
            continue
        key, _, val = line.partition("=")
        values[key.strip()] = val.strip()
    if not values:
        return {"present": False, "reason": "auth.ini has no key=value pairs"}
    id_token = values.get("id_token", "")
    refresh = values.get("refresh_token", "")
    state: dict[str, Any] = {
        "present": True,
        "issuer_url": values.get("issuer_url"),
        "has_refresh_token": bool(refresh),
    }
    if not id_token:
        state.update(expired=True, reason="no id_token stored")
        return state
    try:
        payload = id_token.split(".")[1]
        payload += "=" * (-len(payload) % 4)
        claims = json.loads(base64.urlsafe_b64decode(payload))
    except (IndexError, ValueError) as exc:
        state.update(expired=True, reason=f"unparseable id_token: {exc}")
        return state
    exp = int(claims.get("exp") or 0)
    left = exp - int(time.time())
    state.update(
        email=claims.get("email"),
        fireworks_account=claims.get("fireworks_account"),
        id_token_expiry=time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime(exp)),
        id_token_seconds_left=left,
        expired=left <= 0,
        refreshable=bool(refresh),
    )
    return state


# --------------------------------------------------------------------------
# download
# --------------------------------------------------------------------------


def classify(output: str) -> str:
    text = output.lower()
    if "model downloading is restricted" in text:
        return "restricted"
    if "api keys cannot download" in text or "require signing in" in text:
        return "auth"
    if "not found" in text or "notfound" in text:
        return "not_found"
    return "other"


def download(name: str) -> dict[str, Any]:
    target = MODELS_DIR / name
    target.mkdir(parents=True, exist_ok=True)
    # No --api-key: the key is rejected for downloads by design, the signed-in
    # Cognito session in auth.ini is the only working path.
    cmd = [FIRECTL, "model", "download", name, str(target) + "/", "--quiet"]
    proc = subprocess.run(cmd, capture_output=True, text=True, timeout=DOWNLOAD_TIMEOUT, check=False)
    output = (proc.stdout or "") + (proc.stderr or "")
    if proc.returncode == 0:
        return {"status": "downloaded", "message": ""}
    kind = classify(output)
    line = next(
        (ln.strip() for ln in output.splitlines() if "Failed to execute" in ln),
        output.strip().splitlines()[-1] if output.strip() else "unknown error",
    )
    return {"status": kind, "message": line}


# --------------------------------------------------------------------------
# normalize / verify
# --------------------------------------------------------------------------


def payload_dirs(name: str) -> list[Path]:
    root = MODELS_DIR / name
    if not root.exists():
        return []
    return sorted({p.parent for p in root.rglob("adapter_config.json")})


def normalize(name: str) -> str:
    """Flatten the nested tuned-model/run/.../promoted-step layout into the model dir."""
    root = MODELS_DIR / name
    dirs = payload_dirs(name)
    if not dirs:
        return "no-payload"
    if len(dirs) > 1:
        return f"multi-payload({len(dirs)})"
    src = dirs[0]
    if src == root:
        return "already-flat"
    files = [f for f in src.iterdir() if f.is_file()]
    provenance = {
        "model": name,
        "remote_path": str(src.relative_to(root)),
        "files": sorted(f.name for f in files),
        "normalized_at": now(),
    }
    for f in files:
        shutil.move(str(f), str(root / f.name))
    (root / "provenance.json").write_text(json.dumps(provenance, indent=1) + "\n")
    for d in sorted((p for p in root.rglob("*") if p.is_dir()), key=lambda p: len(p.parts), reverse=True):
        if not any(d.iterdir()):
            d.rmdir()
    return "flattened"


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def read_safetensors_header(path: Path) -> dict[str, Any]:
    size = path.stat().st_size
    with open(path, "rb") as fh:
        (n,) = struct.unpack("<Q", fh.read(8))
        if n <= 0 or n + 8 > size:
            raise ValueError(f"{path.name}: implausible header length {n}")
        header = json.loads(fh.read(n))
    tensors = {k: v for k, v in header.items() if k != "__metadata__"}
    if not tensors:
        raise ValueError(f"{path.name}: no tensors")
    end = max(v["data_offsets"][1] for v in tensors.values())
    expected = 8 + n + end
    if expected != size:
        raise ValueError(f"{path.name}: data size mismatch (header says {expected}, file is {size})")
    return {
        "tensors": len(tensors),
        "dtypes": sorted({v["dtype"] for v in tensors.values()}),
        "shapes": {v["shape"][0] for v in tensors.values()},
    }


def has_payload(name: str) -> bool:
    """A blocked download can leave an empty shell dir; only a payload counts."""
    root = MODELS_DIR / name
    if not root.is_dir():
        return False
    return bool(list(root.glob("*.safetensors")) or list(root.rglob("*.safetensors")))


def verify(name: str) -> dict[str, Any]:
    root = MODELS_DIR / name
    shards = sorted(root.glob("*.safetensors")) or sorted(root.rglob("*.safetensors"))
    if not shards:
        raise ValueError("no safetensors payload")
    index = root / "adapter_model.safetensors.index.json"
    if index.exists():
        wanted = set(json.loads(index.read_text())["weight_map"].values())
        missing = sorted(w for w in wanted if not (root / w).exists())
        if missing:
            raise ValueError(f"index references missing shards: {missing}")
    tensors = 0
    dtypes: set[str] = set()
    for shard in shards:
        header = read_safetensors_header(shard)
        tensors += header["tensors"]
        dtypes.update(header["dtypes"])
    config_path = root / "adapter_config.json"
    config = json.loads(config_path.read_text()) if config_path.exists() else {}
    digest = hashlib.sha256()
    for shard in shards:
        digest.update(f"{shard.name}:{sha256_file(shard)}\n".encode())
    return {
        "sha256": digest.hexdigest(),
        "bytes": sum(s.stat().st_size for s in shards),
        "shards": len(shards),
        "tensors": tensors,
        "dtypes": sorted(dtypes),
        "rank": config.get("r"),
        "alpha": config.get("lora_alpha"),
        "base_model": config.get("base_model_name_or_path") or None,
        "target_modules": len(config.get("target_modules") or []),
    }


# --------------------------------------------------------------------------
# ledger
# --------------------------------------------------------------------------


def load_ledger() -> dict[str, Any]:
    if LEDGER.exists():
        return json.loads(LEDGER.read_text())
    return {"models": {}}


def save_ledger(ledger: dict[str, Any]) -> None:
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    tmp = LEDGER.with_suffix(".json.tmp")
    tmp.write_text(json.dumps(ledger, indent=1, sort_keys=True) + "\n")
    tmp.replace(LEDGER)


def record(ledger: dict[str, Any], name: str, **fields: Any) -> None:
    entry = ledger["models"].setdefault(name, {})
    entry.update(fields)
    entry["checked_at"] = now()
    save_ledger(ledger)


# --------------------------------------------------------------------------
# commands
# --------------------------------------------------------------------------


def cmd_check_auth(_: argparse.Namespace) -> int:
    state = auth_state()
    print(json.dumps(state, indent=1))
    if not state.get("present") or state.get("expired"):
        if state.get("refreshable"):
            print("id_token expired but a refresh_token is stored; firectl refreshes it on use.")
            return 0
        print("SESSION UNAVAILABLE -- run `firectl signin` before syncing.")
        return 1
    return 0


def cmd_status(args: argparse.Namespace) -> int:
    ledger = load_ledger()
    models = list_models()
    rows = []
    for m in models:
        name = m["name"].split("/")[-1]
        if args.only and name not in args.only:
            continue
        entry = ledger["models"].get(name, {})
        prior = entry.get("status", "missing")
        if has_payload(name):
            try:
                info = verify(name)
                state = "ready"
                detail = f"{info['tensors']} tensors r={info['rank']} {info['bytes'] / 1e6:.0f}MB"
            except (ValueError, OSError) as exc:
                # A blocked download can leave an empty shell dir; trust the
                # recorded vendor-side reason over the local parse failure.
                state = prior if prior in BLOCKING else "broken"
                detail = (entry.get("message") or str(exc))[:70]
        else:
            state = prior
            detail = (entry.get("message") or "")[:70]
        rows.append((name, m["kind"], state, detail))
    width = max((len(r[0]) for r in rows), default=10)
    for name, kind, state, detail in rows:
        print(f"{name:<{width}}  {kind:<14}  {state:<10}  {detail}")
    blocked = [r for r in rows if r[2] in BLOCKING]
    if blocked:
        print(f"\n{len(blocked)} model(s) not local. `request` drafts the support email for restricted ones.")
    return 0


def cmd_sync(args: argparse.Namespace) -> int:
    state = auth_state()
    if not state.get("present"):
        print(f"auth: {state.get('reason')}")
        return 1
    ledger = load_ledger()
    models = list_models()
    failures = 0
    for m in models:
        name = m["name"].split("/")[-1]
        if args.only and name not in args.only:
            continue
        if has_payload(name) and not args.force:
            print(f"{name}: present, verifying")
        else:
            print(f"{name}: downloading", flush=True)
            result = download(name)
            if result["status"] != "downloaded":
                print(f"  -> {result['status']}: {result['message']}")
                record(ledger, name, status=result["status"], message=result["message"])
                # Leave no empty shell that would later read as "present".
                shutil.rmtree(MODELS_DIR / name, ignore_errors=True)
                failures += 1
                continue
            record(ledger, name, status="downloaded", message="", downloaded_at=now())
        action = normalize(name)
        if action.startswith("multi-payload"):
            print(f"  -> {action}: left nested, verify walks subdirs")
        try:
            info = verify(name)
        except (ValueError, OSError) as exc:
            print(f"  -> verify failed: {exc}")
            record(ledger, name, status="broken", message=str(exc)[:200])
            failures += 1
            continue
        record(ledger, name, status="ready", message="", **info)
        print(f"  -> ready: {info['tensors']} tensors, r={info['rank']}, {info['bytes'] / 1e6:.1f}MB")
    return 1 if failures else 0


def cmd_normalize(args: argparse.Namespace) -> int:
    names = args.only or sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())
    for name in names:
        print(f"{name}: {normalize(name)}")
    return 0


def cmd_verify(args: argparse.Namespace) -> int:
    ledger = load_ledger()
    names = args.only or sorted(p.name for p in MODELS_DIR.iterdir() if p.is_dir())
    failures = 0
    for name in names:
        if not has_payload(name):
            prior = ledger["models"].get(name, {})
            print(f"{name}: no local payload (ledger: {prior.get('status', 'missing')})")
            continue
        try:
            info = verify(name)
        except (ValueError, OSError) as exc:
            print(f"{name}: BROKEN -- {exc}")
            record(ledger, name, status="broken", message=str(exc)[:200])
            failures += 1
            continue
        record(ledger, name, status="ready", message="", **info)
        print(
            f"{name}: ok  {info['shards']} shard(s)  {info['tensors']} tensors  "
            f"{info['dtypes']}  r={info['rank']} alpha={info['alpha']}  "
            f"{info['bytes'] / 1e6:.1f}MB  sha256:{info['sha256'][:16]}"
        )
    return 1 if failures else 0


def cmd_request(_args: argparse.Namespace) -> int:
    ledger = load_ledger()
    restricted = sorted(n for n, e in ledger["models"].items() if e.get("status") == "restricted")
    if not restricted:
        print("no restricted models in the ledger -- nothing to request")
        return 0
    ready = sorted(n for n, e in ledger["models"].items() if e.get("status") == "ready")
    bullets = "\n".join(f"  - {n}" for n in restricted)
    body = f"""Subject: Re: model download access - {len(restricted)} model(s) still restricted, plus a question about future models

Hi Kat,

Thanks again for turning this on - {len(ready)} of the adapters on my account now export cleanly and everything works on my end. Genuinely appreciate the fast turnaround.

These models still come back restricted when I try to export them:

{bullets}

All of them fail with "model downloading is restricted, please contact Fireworks AI to get access." Could they be added to the allowlist as well?

One question, and apologies if this is already documented somewhere and I simply missed it: will I need to request access individually for each new fine-tuned model I create?

I run a fairly steady cadence of small LoRA experiments - a few a week - and I pull each adapter down to my own hardware for local evaluation, since that's where my eval harness and baseline comparisons live. If every export needs a manual approval, each experiment picks up an extra day or two of waiting, which tends to break the "try it while the idea is fresh" loop and makes it hard to keep momentum on a line of experiments.

Is there a way to enable downloads at the account level, or to set up some kind of standing approval for models I create - with whatever terms, attestation, or responsible-use agreement you need from me attached to it? Happy to sign or fill out anything required.

If per-model approval is genuinely required for policy reasons, that's completely fine - I'd just like to know so I can batch my requests ahead of time instead of asking each round.

Thanks again for the help,

Chad Coburn
"""
    out = MODELS_DIR / ".access-request.txt"
    out.write_text(body)
    print(body)
    print(f"-- written to {out}")
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    sub = parser.add_subparsers(dest="command", required=True)

    def with_only(p: argparse.ArgumentParser) -> argparse.ArgumentParser:
        p.add_argument("--only", nargs="+", metavar="NAME", help="limit to these model names")
        return p

    with_only(sub.add_parser("status", help="table of model export state")).set_defaults(func=cmd_status)
    sync_p = with_only(sub.add_parser("sync", help="download + normalize + verify missing models"))
    sync_p.add_argument("--force", action="store_true", help="re-download models already on disk")
    sync_p.set_defaults(func=cmd_sync)
    with_only(sub.add_parser("normalize", help="flatten nested payload dirs")).set_defaults(func=cmd_normalize)
    with_only(sub.add_parser("verify", help="re-verify local adapters")).set_defaults(func=cmd_verify)
    sub.add_parser("request", help="draft the support email for restricted models").set_defaults(func=cmd_request)
    sub.add_parser("check-auth", help="report the firectl session state").set_defaults(func=cmd_check_auth)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    sys.exit(main())
