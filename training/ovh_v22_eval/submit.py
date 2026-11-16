"""Submit the OVH eval job with R2 as both the input store and the result store.

The OVH volume mount worked for reads and failed for writes: three eval jobs in a
row ended ``SYNC_FAILED : DATASYNC_RETRY_ERROR`` and took every result with them.
So the job no longer mounts a volume at all. It receives one gzipped, base64
environment variable holding a presigned ``GET`` for each input and a presigned
``PUT`` for each result file, pulls its inputs into ``/tmp/work``, runs, and
pushes each finished run directory back to R2. No credential and no SDK enters
the container, and nothing depends on the volume syncing at teardown.

Usage::

    python -m training.ovh_v22_eval.submit stage            # copy the bundle to /tmp/ovh-eval
    python -m training.ovh_v22_eval.submit run --dry-run    # show the ovhai command
    python -m training.ovh_v22_eval.submit run --execute    # upload, presign, launch
    python -m training.ovh_v22_eval.submit collect --tag v23-l4a
"""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from pathlib import Path

from training.launch import r2_sync

AI_ROOT = Path(__file__).resolve().parents[2]
REPO_ROOT = AI_ROOT.parent

DEFAULT_IMAGE = "pytorch/pytorch:2.11.0-cuda12.6-cudnn9-runtime"
DEFAULT_FLAVOR = "l4-1-gpu"
DEFAULT_TIMEOUT = "2h"
DEFAULT_STAGE_DIR = Path("/tmp/ovh-eval")
# 344 MB each: kept out of the repo, staged from the training output directory.
DEFAULT_ADAPTERS = {
    "bf16": AI_ROOT / "training/output/ovh-v22-qwen3-8b/adapter-final",
    "nf4": AI_ROOT / "training/output/kaggle-v22-qwen3-8b/adapter-v6",
}
DEFAULT_LABELS = ("bf16", "nf4")
DEFAULT_SPLITS = ("dev", "sealed")
# Prefix on every run id inside a job, so a stored result says which adapter set
# produced it.
DEFAULT_RUN_LABEL = "ovh-v22"
# What local_paired_eval writes into a results directory. Nothing else is signed,
# so nothing else can be lost.
RESULT_FILES = ("manifest.json", "results.jsonl")

# Modules the runner imports, staged under their package path so PYTHONPATH works
# unchanged inside the container.
CODE_FILES = (
    "training/__init__.py",
    "training/local_paired_eval.py",
    "training/launch/r2_sync.py",
    "qa/validation/__init__.py",
    "qa/validation/therapy_bench/__init__.py",
    "qa/validation/therapy_bench/paired_eval.py",
)
CASES_DIR = REPO_ROOT / "docs/training/execution-2026-10-17"

# Runs inside the container before anything else. Kept to the standard library:
# the image has no boto3, and the only thing it needs is HTTP.
BOOTSTRAP = """\
import base64, gzip, json, os, pathlib, shutil, subprocess, sys, urllib.request

plan = json.loads(gzip.decompress(base64.urlsafe_b64decode(os.environ["R2_MAP_B64"].encode())))
root = pathlib.Path(os.environ.get("ROOT", "/tmp/work"))
downloads = plan.get("download", {})
print(f"bootstrap: {len(downloads)} objects into {root}", flush=True)
for relative, url in sorted(downloads.items()):
    target = root / relative
    target.parent.mkdir(parents=True, exist_ok=True)
    if target.is_file() and target.stat().st_size:
        print(f"  cached {relative}", flush=True)
        continue
    request = urllib.request.Request(url, headers={"User-Agent": "pixelated-empathy-r2-sync/1.0"})
    with urllib.request.urlopen(request, timeout=900) as response, target.open("wb") as handle:
        shutil.copyfileobj(response, handle)
    print(f"  fetched {relative} {target.stat().st_size} bytes", flush=True)
raise SystemExit(subprocess.call(["bash", str(root / "code" / "launch.sh")]))
"""


def stage(destination: Path = DEFAULT_STAGE_DIR, *, adapters: dict[str, Path] | None = None) -> list[str]:
    """Copy the runner bundle, the crisis cases and the adapters into one tree.

    Staging from the repo instead of hand-copying is the difference between a
    job that scores the code under review and a job that scores whatever was in
    /tmp last time.
    """
    adapters = DEFAULT_ADAPTERS if adapters is None else adapters
    if destination.exists():
        shutil.rmtree(destination)
    copied: list[str] = []

    for relative in CODE_FILES:
        target = destination / "code" / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(AI_ROOT / relative, target)
        copied.append(f"code/{relative}")

    launcher = destination / "code" / "launch.sh"
    launcher.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(Path(__file__).with_name("launch.sh"), launcher)
    copied.append("code/launch.sh")

    for split in DEFAULT_SPLITS:
        name = f"scenario-crisis-{split}-v1.jsonl"
        target = destination / "data" / name
        target.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(CASES_DIR / name, target)
        copied.append(f"data/{name}")

    for label, source in adapters.items():
        if not source.is_dir():
            raise SystemExit(f"adapter for {label} is missing: {source}")
        shutil.copytree(source, destination / "adapters" / label)
        copied += [f"adapters/{label}/{path.name}" for path in sorted((destination / "adapters" / label).iterdir())]

    return copied


def run_names(
    *, labels: tuple[str, ...] = DEFAULT_LABELS, splits: tuple[str, ...] = DEFAULT_SPLITS, out_prefix: str = ""
) -> list[str]:
    """The result directory each (adapter, split) pair will produce."""
    return [f"{out_prefix}eval-{label}-{split}" for label in labels for split in splits]


def result_keys(tag: str, *, names: list[str], files: tuple[str, ...] = RESULT_FILES) -> dict[str, str]:
    """``<run>/<file> -> r2 key`` for every file the launcher will upload."""
    return {f"{name}/{file}": f"evals/{tag}/{name}/{file}" for name in names for file in files}


def staged_files(stage_dir: Path) -> list[str]:
    return sorted(path.relative_to(stage_dir).as_posix() for path in stage_dir.rglob("*") if path.is_file())


def build_job_env(
    *,
    tag: str,
    plan: dict[str, object],
    out_prefix: str,
    flavor: str,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    run_label: str = DEFAULT_RUN_LABEL,
) -> list[str]:
    env = [
        f"R2_MAP_B64={r2_sync.encode_map(plan)}",
        f"BOOTSTRAP={BOOTSTRAP}",
        "ROOT=/tmp/work",
        "OUT_ROOT=/tmp/out",
        f"LABELS={' '.join(labels)}",
        f"RUN_LABEL={run_label}",
        f"EVAL_TAG={tag}",
        f"EVAL_FLAVOR={flavor}",
    ]
    if out_prefix:
        # ovhai rejects empty env values; the launcher already defaults to no prefix.
        env.append(f"OUT_PREFIX={out_prefix}")
    return env


def build_command(*, image: str, name: str, flavor: str, gpu: int, timeout: str, env: list[str]) -> list[str]:
    """The ``ovhai job run`` argv. No volume: the whole point of the move to R2."""
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
    inputs_prefix: str = "inputs/v22",
    out_prefix: str = "",
    image: str = DEFAULT_IMAGE,
    flavor: str = DEFAULT_FLAVOR,
    gpu: int = 1,
    timeout: str = DEFAULT_TIMEOUT,
    name: str | None = None,
    execute: bool = False,
    restage: bool = True,
    adapters: dict[str, Path] | None = None,
    labels: tuple[str, ...] = DEFAULT_LABELS,
    run_label: str = DEFAULT_RUN_LABEL,
) -> dict[str, object]:
    """Upload the bundle, sign the map, and launch (or print) the job."""
    adapters = DEFAULT_ADAPTERS if adapters is None else adapters
    staged = stage(stage_dir, adapters=adapters) if restage else staged_files(stage_dir)

    s3 = r2_sync.client()
    bucket = r2_sync.credentials()[r2_sync.DEFAULT_BUCKET_ENV]
    uploaded = r2_sync.upload_dir(stage_dir, bucket=bucket, prefix=inputs_prefix, s3_client=s3, skip_unchanged=True)

    download_urls = r2_sync.presign(
        [f"{inputs_prefix}/{relative}" for relative in staged], bucket=bucket, method="get", s3_client=s3
    )
    download = {relative: download_urls[f"{inputs_prefix}/{relative}"] for relative in staged}

    names = run_names(labels=labels, out_prefix=out_prefix)
    keys = result_keys(tag, names=names)
    upload = {
        relative: r2_sync.presign([key], bucket=bucket, method="put", s3_client=s3)[key]
        for relative, key in keys.items()
    }

    job_name = name or f"v22-eval-{tag}"
    plan: dict[str, object] = {"tag": tag, "bucket": bucket, "download": download, "upload": upload}
    env = build_job_env(tag=tag, plan=plan, out_prefix=out_prefix, flavor=flavor, labels=labels, run_label=run_label)
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
        "runs": names,
        "labels": list(labels),
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
    """Pull every stored result for a tag out of R2."""
    target = destination or AI_ROOT / f"training/output/r2-eval-{tag}"
    bucket = r2_sync.credentials()[r2_sync.DEFAULT_BUCKET_ENV]
    return r2_sync.download_prefix(bucket=bucket, prefix=f"evals/{tag}", destination=target)


def _selected_adapters(parser: argparse.ArgumentParser, args: argparse.Namespace) -> dict[str, Path]:
    """The adapter staged per label, after ``--adapter LABEL=PATH`` overrides.

    Only the labels the job will score are staged: copying an adapter the
    launcher never opens costs 344 MB of upload for nothing.
    """
    labels = tuple(item.strip() for item in args.labels.split(",") if item.strip())
    adapters = dict(DEFAULT_ADAPTERS)
    for pair in args.adapter:
        label, separator, path = pair.partition("=")
        if not separator or not label.strip() or not path.strip():
            parser.error(f"--adapter wants LABEL=PATH, got {pair!r}")
        adapters[label.strip()] = Path(path.strip())
    missing = [label for label in labels if label not in adapters]
    if missing:
        parser.error(f"no adapter staged for {', '.join(missing)}; pass --adapter LABEL=PATH")
    return {label: adapters[label] for label in labels}


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)

    stage_parser = sub.add_parser("stage", help="copy the runner bundle, cases and adapters into one tree")
    stage_parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)

    run_parser = sub.add_parser("run", help="upload the bundle, sign the map and launch the job")
    run_parser.add_argument("--tag", required=True)
    run_parser.add_argument("--stage-dir", type=Path, default=DEFAULT_STAGE_DIR)
    run_parser.add_argument("--inputs-prefix", default="inputs/v22")
    run_parser.add_argument("--out-prefix", default="")
    run_parser.add_argument("--image", default=DEFAULT_IMAGE)
    run_parser.add_argument("--flavor", default=DEFAULT_FLAVOR)
    run_parser.add_argument("--gpu", type=int, default=1)
    run_parser.add_argument("--timeout", default=DEFAULT_TIMEOUT)
    run_parser.add_argument("--name", default=None)
    run_parser.add_argument(
        "--labels",
        default=",".join(DEFAULT_LABELS),
        help="comma-separated adapter labels to score, each needing a staged adapters/<label> directory",
    )
    run_parser.add_argument(
        "--adapter",
        action="append",
        default=[],
        metavar="LABEL=PATH",
        help="override the adapter staged for a label (repeatable); defaults are the two v22 adapters",
    )
    run_parser.add_argument("--run-label", default=DEFAULT_RUN_LABEL)
    run_parser.add_argument("--no-restage", action="store_true")
    run_parser.add_argument("--execute", action="store_true")

    collect_parser = sub.add_parser("collect", help="download every stored result for a tag")
    collect_parser.add_argument("--tag", required=True)
    collect_parser.add_argument("--destination", type=Path, default=None)

    args = parser.parse_args(argv)

    if args.command == "stage":
        copied = stage(args.stage_dir)
        sys.stdout.write(f"staged {len(copied)} files into {args.stage_dir}\n")
        for relative in copied:
            sys.stdout.write(f"  {relative}\n")
        return 0

    if args.command == "collect":
        written = collect(tag=args.tag, destination=args.destination)
        sys.stdout.write(f"downloaded {len(written)} objects\n")
        for path in written:
            sys.stdout.write(f"  {path}\n")
        return 0

    report = submit(
        tag=args.tag,
        stage_dir=args.stage_dir,
        inputs_prefix=args.inputs_prefix,
        out_prefix=args.out_prefix,
        image=args.image,
        flavor=args.flavor,
        gpu=args.gpu,
        timeout=args.timeout,
        name=args.name,
        execute=args.execute,
        restage=not args.no_restage,
        adapters=_selected_adapters(parser, args),
        labels=tuple(item.strip() for item in args.labels.split(",") if item.strip()),
        run_label=args.run_label,
    )
    sys.stdout.write(json.dumps({key: value for key, value in report.items() if key != "command"}, indent=2) + "\n")
    sys.stdout.write("\novhai job run ... " + " ".join(report["command"][-3:]) + "\n")
    if not args.execute:
        sys.stdout.write("(dry run: pass --execute to launch)\n")
    return 0 if report.get("returncode", 0) == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
