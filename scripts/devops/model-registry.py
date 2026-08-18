#!/usr/bin/env python3
import json
import os
from pathlib import Path
from typing import Any

REGISTRY_PATH = Path(os.getenv("MODEL_REGISTRY_PATH", "/tmp/registry/models.json"))
DEFAULT_CHECKPOINT_DIR = Path(os.getenv("MODEL_CHECKPOINT_DIR", "/tmp/checkpoints"))


def _load_manifest() -> dict[str, Any]:
    if not REGISTRY_PATH.exists():
        return {"schema_version": "1.0", "active_run_id": None, "checkpoints": []}
    try:
        with open(REGISTRY_PATH) as f:
            return json.load(f)
    except json.JSONDecodeError:
        raise SystemExit(1)


def _save_manifest(manifest: dict[str, Any]) -> None:
    REGISTRY_PATH.parent.mkdir(parents=True, exist_ok=True)
    with open(REGISTRY_PATH, "w") as f:
        json.dump(manifest, f, indent=2)


def cmd_tag(args: Any) -> None:
    manifest = _load_manifest()

    # Check for duplicates unless forced
    for cp in manifest["checkpoints"]:
        if cp["run_id"] == args.run_id:
            if not getattr(args, "force", False):
                raise SystemExit(1)
            # Remove old duplicate if forced
            manifest["checkpoints"] = [c for c in manifest["checkpoints"] if c["run_id"] != args.run_id]
            break

    checkpoint = {
        "run_id": args.run_id,
        "base_model": args.base_model,
        "dataset_version": args.dataset_version,
        "clinical_validity_score": args.clinical_validity_score,
    }

    if getattr(args, "metrics", None):
        try:
            checkpoint["metrics"] = json.loads(args.metrics)
        except json.JSONDecodeError:
            raise SystemExit(1)

    manifest["checkpoints"].append(checkpoint)

    if getattr(args, "set_active", False):
        manifest["active_run_id"] = args.run_id

    _save_manifest(manifest)


def cmd_show(args: Any) -> None:
    manifest = _load_manifest()
    for cp in manifest["checkpoints"]:
        if cp["run_id"] == args.run_id:
            return
    raise SystemExit(1)


def cmd_list(args: Any) -> None:
    manifest = _load_manifest()
    if not manifest["checkpoints"]:
        return
    for _cp in manifest["checkpoints"]:
        pass


def cmd_rollback(args: Any) -> None:
    manifest = _load_manifest()
    found = False
    for cp in manifest["checkpoints"]:
        if cp["run_id"] == args.run_id:
            found = True
            break

    if not found:
        raise SystemExit(1)

    manifest["active_run_id"] = args.run_id
    _save_manifest(manifest)

    checkpoint_dir = getattr(args, "checkpoint_dir", DEFAULT_CHECKPOINT_DIR)
    target_dir = checkpoint_dir / args.run_id
    active_link = checkpoint_dir / "active"

    if active_link.exists() or active_link.is_symlink():
        active_link.unlink()

    if target_dir.exists():
        # Using a relative symlink is usually better, but for tests creating a file works if symlinks aren't supported
        try:
            active_link.symlink_to(target_dir.name)
        except OSError:
            # Fallback for systems without symlink support
            with open(active_link, "w") as f:
                f.write(str(target_dir))


def main():
    import argparse as _argparse

    parser = _argparse.ArgumentParser(description="Model registry CLI")
    sub = parser.add_subparsers(dest="command")

    tag_p = sub.add_parser("tag")
    tag_p.add_argument("--run-id", required=True)
    tag_p.add_argument("--base-model", required=True)
    tag_p.add_argument("--dataset-version", required=True)
    tag_p.add_argument("--clinical-validity-score", type=float, required=True)
    tag_p.add_argument("--metrics")
    tag_p.add_argument("--set-active", action="store_true")
    tag_p.add_argument("--force", action="store_true")

    show_p = sub.add_parser("show")
    show_p.add_argument("--run-id", required=True)

    sub.add_parser("list")

    rb_p = sub.add_parser("rollback")
    rb_p.add_argument("--run-id", required=True)
    rb_p.add_argument("--checkpoint-dir", type=Path, default=DEFAULT_CHECKPOINT_DIR)

    args = parser.parse_args()
    if args.command == "tag":
        cmd_tag(args)
    elif args.command == "show":
        cmd_show(args)
    elif args.command == "list":
        cmd_list(args)
    elif args.command == "rollback":
        cmd_rollback(args)
    else:
        parser.print_help()
        raise SystemExit(1)


if __name__ == "__main__":
    main()
