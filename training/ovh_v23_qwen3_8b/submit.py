"""Submit the OVH v23 LoRA training job with R2 as both the input store and the result store.

The volume mount is not used: the eval jobs proved it reads fine and fails on
write-back (``SYNC_FAILED : DATASYNC_RETRY_ERROR``), and the adapter is the only
copy of a run that costs GPU time. Instead the job gets one gzipped, base64
environment variable holding a presigned ``GET`` for the export and the trainer,
plus a presigned ``PUT`` for every file the run produces. It pulls its inputs,
trains, and pushes the adapter back before exiting. No credential and no SDK
enters the container.

Usage::

    python -m training.ovh_v23_qwen3_8b.submit stage            # copy the bundle to /tmp/ovh-v23
    python -m training.ovh_v23_qwen3_8b.submit run --dry-run    # show the ovhai command
    python -m training.ovh_v23_qwen3_8b.submit run --execute    # upload, presign, launch
    python -m training.ovh_v23_qwen3_8b.submit collect --tag v23-l40s
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
from pathlib import Path

from training.launch import r2_sync

AI_ROOT = Path(__file__).resolve().parents[2]

DEFAULT_IMAGE = "pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime"
# 8B in bf16 plus activations does not fit comfortably on a 24 GB L4 with the
# 46 GB L40S available, and the v22 run took 584 s of train time on one.
DEFAULT_FLAVOR = "l40s-1-gpu"
DEFAULT_TIMEOUT = "2h"
DEFAULT_STAGE_DIR = Path("/tmp/ovh-v23")
DEFAULT_EXPORT_DIR = AI_ROOT / "training/output/ovh-v23-qwen3-8b"
DEFAULT_RUN_NAME = "ovh-wandb-full-corpus-current-master-20261110-v23-bf16"

# The trainer is the v22 module, unchanged: it reads DATA_DIR/OUT_DIR/RUN_NAME
# from the environment, so the release is a submission parameter rather than a
# forked copy of the training loop.
CODE_FILES = {
    "run_ovh.py": AI_ROOT / "training/ovh_v22_qwen3_8b/run_ovh.py",
    "train.py": AI_ROOT / "training/kaggle_v22_qwen3_8b/train.py",
}
EXPORT_FILES = ("tinker-sft-train.jsonl", "tinker-dataset.json")

# Everything ``model.save_pretrained`` + ``tokenizer.save_pretrained`` write into
# adapter-final, plus the run record. The set is fixed, so each file can be
# signed before the job starts.
ADAPTER_FILES = (
    "adapter_config.json",
    "adapter_model.safetensors",
    "chat_template.jinja",
    "README.md",
    "tokenizer.json",
    "tokenizer_config.json",
)
RESULT_FILES = (*(f"adapter-final/{name}" for name in ADAPTER_FILES), "ovh-run.json")

# Runs inside the container. Kept to the standard library: the image has no
# boto3, and the only thing it needs is HTTP.
BOOTSTRAP = """\
import base64, gzip, json, os, pathlib, shutil, subprocess, sys, urllib.request

plan = json.loads(gzip.decompress(base64.urlsafe_b64decode(os.environ["R2_MAP_B64"].encode())))
root = pathlib.Path(os.environ.get("ROOT", "/tmp/work"))
sizes = plan.get("sizes", {})
downloads = plan.get("download", {})
print(f"bootstrap: {len(downloads)} objects into {root}", flush=True)
for relative, url in sorted(downloads.items()):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size == sizes.get(relative):
        print(f"  cached {relative}", flush=True)
        continue
    request = urllib.request.Request(url, headers={"User-Agent": "pixelated-empathy-r2-sync/1.0"})
    with urllib.request.urlopen(request, timeout=900) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"  fetched {relative} {target.stat().st_size} bytes", flush=True)

code = subprocess.call(["bash", str(root / "code" / "launch.sh")])
print(f"bootstrap: launcher exited {code}", flush=True)

failed = []
for relative, url in sorted(plan.get("upload", {}).items()):
    source = root / "out" / relative
    if not source.is_file():
        print(f"  missing result {relative}", flush=True)
        failed.append(relative)
        continue
    payload = source.read_bytes()
    request = urllib.request.Request(
        url,
        data=payload,
        method="PUT",
        headers={
            "Content-Type": "application/octet-stream",
            "Content-Length": str(len(payload)),
            "User-Agent": "pixelated-empathy-r2-sync/1.0",
        },
    )
    try:
        with urllib.request.urlopen(request, timeout=900) as response:
            print(f"  uploaded {relative} {len(payload)} bytes -> {response.status}", flush=True)
    except Exception as error:
        print(f"  upload failed {relative}: {error}", flush=True)
        failed.append(relative)

if failed:
    print(f"bootstrap: {len(failed)} results did not land: {failed}", flush=True)
raise SystemExit(code if not failed else 1)
"""


def stage(destination: Path = DEFAULT_STAGE_DIR, *, export_dir: Path = DEFAULT_EXPORT_DIR) -> list[str]:
    """Copy the trainer, the launcher and the export into one tree."""
    if destination.exists():
        shutil.rmtree(destination)
    copied: list[str] = []

    for name, source in CODE_FILES.items():
        if not source.is_file():
            raise SystemExit(f"missing trainer file: {source}")
        target = destination / "code" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(f"code/{name}")

    launcher = destination / "code" / "launch.sh"
    shutil.copy2(Path(__file__).with_name("launch.sh"), launcher)
    copied.append("code/launch.sh")

    for name in EXPORT_FILES:
        source = export_dir / name
        if not source.is_file():
            raise SystemExit(f"missing export file: {source} (run export_tinker_release first)")
        target = destination / "data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(source, target)
        copied.append(f"data/{name}")

    return copied


def staged_files(stage_dir: Path) -> list[str]:
    return sorted(path.relative_to(stage_dir).as_posix() for path in stage_dir.rglob("*") if path.is_file())


def result_keys(tag: str) -> dict[str, str]:
    """``out-relative path -> r2 key`` for every file the run produces."""
    return {relative: f"runs/{tag}/{relative}" for relative in RESULT_FILES}


def build_job_env(*, plan: dict[str, object], run_name: str, flavor: str) -> list[str]:
    return [
        f"R2_MAP_B64={r2_sync.encode_map(plan)}",
        f"BOOTSTRAP={BOOTSTRAP}",
        "ROOT=/tmp/work",
        f"RUN_NAME={run_name}",
        f"TRAIN_FLAVOR={flavor}",
    ]


def build_command(*, image: str, name: str, flavor: str, gpu: int, timeout: str, env: list[str]) -> list[str]:
    """The ``ovhai job run`` argv. No volume: R2 carries both directions."""
    argv = [
        "ovhai",
        "job",
        "run",
        "--name",
        name,
        "--flavor",
        flavor,
        "--gpu",
        str(gpu),
        "--timeout",
        timeout,
    ]
    for entry in env:
        argv += ["--env", entry]
    argv += [image, "--", "bash", "-lc", 'python3 -c "$BOOTSTRAP"']
    return argv


def submit(
    *,
    tag: str,
    stage_dir: Path = DEFAULT_STAGE_DIR,
    export_dir: Path = DEFAULT_EXPORT_DIR,
    inputs_prefix: str = "inputs/v23",
    run_name: str = DEFAULT_RUN_NAME,
    image: str = DEFAULT_IMAGE,
    flavor: str = DEFAULT_FLAVOR,
    gpu: int = 1,
    timeout: str = DEFAULT_TIMEOUT,
    name: str | None = None,
    execute: bool = False,
    restage: bool = True,
) -> dict[str, object]:
    """Upload the bundle, sign the map, and launch (or print) the job."""
    staged = stage(stage_dir, export_dir=export_dir) if restage else staged_files(stage_dir)

    s3 = r2_sync.client()
    bucket = r2_sync.credentials()[r2_sync.DEFAULT_BUCKET_ENV]
    uploaded = r2_sync.upload_dir(stage_dir, bucket=bucket, prefix=inputs_prefix, s3_client=s3, skip_unchanged=True)

    download_urls = r2_sync.presign(
        [f"{inputs_prefix}/{relative}" for relative in staged], bucket=bucket, method="get", s3_client=s3
    )
    download = {relative: download_urls[f"{inputs_prefix}/{relative}"] for relative in staged}
    sizes = {relative: (stage_dir / relative).stat().st_size for relative in staged}

    keys = result_keys(tag)
    upload = {
        relative: r2_sync.presign([key], bucket=bucket, method="put", s3_client=s3)[key]
        for relative, key in keys.items()
    }

    job_name = name or f"v23-train-{tag}"
    plan: dict[str, object] = {"tag": tag, "bucket": bucket, "download": download, "sizes": sizes, "upload": upload}
    env = build_job_env(plan=plan, run_name=run_name, flavor=flavor)
    argv = build_command(image=image, name=job_name, flavor=flavor, gpu=gpu, timeout=timeout, env=env)

    result: dict[str, object] = {
        "tag": tag,
        "job_name": job_name,
        "bucket": bucket,
        "inputs_prefix": inputs_prefix,
        "staged": len(staged),
        "uploaded_inputs": len(uploaded),
        "download_objects": len(download),
        "upload_objects": len(upload),
        "map_bytes": len(r2_sync.encode_map(plan)),
        "command": argv,
        "executed": False,
    }
    if execute:
        completed = subprocess.run(argv, check=False, capture_output=True, text=True)
        result["executed"] = True
        result["returncode"] = completed.returncode
        result["stdout"] = completed.stdout.strip()
        result["stderr"] = completed.stderr.strip()
    return result


def collect(*, tag: str, destination: Path | None = None) -> list[Path]:
    """Pull the adapter and run record for a tag out of R2."""
    target = destination or AI_ROOT / f"training/output/ovh-v23-qwen3-8b/run-{tag}"
    bucket = r2_sync.credentials()[r2_sync.DEFAULT_BUCKET_ENV]
    return r2_sync.download_prefix(bucket=bucket, prefix=f"runs/{tag}", destination=target)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    stage_parser = sub.add_parser("stage", help="copy the bundle to the stage directory")
    stage_parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    stage_parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)

    run_parser = sub.add_parser("run", help="upload, presign and (with --execute) launch")
    run_parser.add_argument("--tag", required=True)
    run_parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    run_parser.add_argument("--export-dir", type=Path, default=DEFAULT_EXPORT_DIR)
    run_parser.add_argument("--inputs-prefix", default="inputs/v23")
    run_parser.add_argument("--run-name", default=DEFAULT_RUN_NAME)
    run_parser.add_argument("--image", default=DEFAULT_IMAGE)
    run_parser.add_argument("--flavor", default=DEFAULT_FLAVOR)
    run_parser.add_argument("--gpu", type=int, default=1)
    run_parser.add_argument("--timeout", default=DEFAULT_TIMEOUT)
    run_parser.add_argument("--name", default=None)
    run_parser.add_argument("--no-restage", action="store_true")
    run_parser.add_argument("--execute", action="store_true")

    collect_parser = sub.add_parser("collect", help="download the adapter and run record")
    collect_parser.add_argument("--tag", required=True)
    collect_parser.add_argument("--destination", type=Path, default=None)

    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)

    if args.command == "stage":
        copied = stage(args.stage_dir, export_dir=args.export_dir)
        print(f"staged {len(copied)} files into {args.stage_dir}")
        for relative in copied:
            print(f"  {relative}")
        return 0

    if args.command == "run":
        result = submit(
            tag=args.tag,
            stage_dir=args.stage_dir,
            export_dir=args.export_dir,
            inputs_prefix=args.inputs_prefix,
            run_name=args.run_name,
            image=args.image,
            flavor=args.flavor,
            gpu=args.gpu,
            timeout=args.timeout,
            name=args.name,
            execute=args.execute,
            restage=not args.no_restage,
        )
        print(json.dumps(result, indent=2, sort_keys=True))
        if result.get("executed") and result.get("returncode") != 0:
            return 1
        return 0

    paths = collect(tag=args.tag, destination=args.destination)
    print(f"downloaded {len(paths)} objects")
    for path in paths:
        print(f"  {path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
