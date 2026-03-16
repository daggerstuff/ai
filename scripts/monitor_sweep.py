import os
import subprocess
import sys
from pathlib import Path

import boto3
from dotenv import load_dotenv
from rich import box
from rich.console import Console
from rich.markup import escape
from rich.table import Table
from rich.text import Text

# --- Configuration ---
BUCKET_NAME = "pixel-data"
ENDPOINT_URL = "https://s3.us-east-va.io.cloud.ovh.us"
REGION_NAME = "us-east-va"
UPLOAD_PREFIX = "full_ai_sweep/"
SIZE_THRESHOLD = 100 * 1024 * 1024  # 100 MB

PROJECT_ROOT = Path("/home/vivi/pixelated/ai")
console = Console()

# Load .env from ai/ directory or root
env_paths = [PROJECT_ROOT / ".env", PROJECT_ROOT.parent / ".env"]
for env_path in env_paths:
    if env_path.exists():
        load_dotenv(env_path)
        break


def get_s3_client():
    access_key = os.environ.get("OVH_S3_ACCESS_KEY")
    secret_key = os.environ.get("OVH_S3_SECRET_KEY")
    if not access_key or not secret_key:
        console.print(
            "[red]❌ Error: Missing credentials (OVH_S3_ACCESS_KEY/SECRET_KEY).[/red]"
        )
        sys.exit(1)
    return boto3.client(
        "s3",
        endpoint_url=ENDPOINT_URL,
        region_name=REGION_NAME,
        aws_access_key_id=access_key,
        aws_secret_access_key=secret_key,
    )


def get_sweep_pid():
    try:
        # Look for the python process running the sweep script
        cmd = "ps aux | grep full_ai_sweep_s3.py | grep -v grep | awk '{print $2}'"
        pids = subprocess.check_output(cmd, shell=True).decode().strip().split("\n")
        # Return the most recent one (highest PID usually)
        return pids[-1] if pids[0] else None
    except Exception:
        return None


def get_current_file_progress(pid):
    if not pid:
        return None, 0
    try:
        # Find which large file is currently open
        fds = subprocess.check_output(f"ls -l /proc/{pid}/fd", shell=True).decode()
        for line in fds.split("\n"):
            if (
                "ULTIMATE" in line
                or ".jsonl" in line
                or ".zip" in line
                or ".csv" in line
            ):
                # Extract path
                if " -> " in line:
                    path = line.split(" -> ")[1].strip()
                    fd_num = line.split()[8]
                    # Get position
                    pos_info = subprocess.check_output(
                        f"cat /proc/{pid}/fdinfo/{fd_num} | grep pos", shell=True
                    ).decode()
                    pos = int(pos_info.split(":")[1].strip())
                    return path, pos
    except Exception:
        pass
    return None, 0


def main():
    console.print("[bold blue]📡 Pixelated S3 Migration Monitor[/bold blue]\n")

    s3 = get_s3_client()
    pid = get_sweep_pid()
    active_path, active_pos = get_current_file_progress(pid)

    # 1. Scan Local
    to_upload = []
    seen_sizes = {}
    for path in PROJECT_ROOT.rglob("*"):
        if not path.is_file() or ".venv" in path.parts or ".git" in path.parts:
            continue
        size = path.stat().st_size
        if size > SIZE_THRESHOLD:
            if size not in seen_sizes:
                seen_sizes[size] = path
                to_upload.append(path)

    # 2. Check S3 Status
    table = Table(title="S3 Migration Monitor", box=box.ROUNDED, expand=False)
    table.add_column("FILE", style="cyan")
    table.add_column("MB", justify="right")
    table.add_column("STATUS", justify="center")

    completed_size = 0
    total_size = sum(f.stat().st_size for f in to_upload)
    done_count = 0

    for f in sorted(to_upload, key=lambda x: x.stat().st_size, reverse=True):
        rel = f.relative_to(PROJECT_ROOT.parent)
        s3_key = f"{UPLOAD_PREFIX}{rel}"
        f_size = f.stat().st_size

        status = Text("Waiting...", style="yellow")

        # Check if active
        if active_path and str(f) == active_path:
            pct = (active_pos / f_size) * 100
            status = Text(f"Uploading ({pct:.1f}%)", style="bold green")
        else:
            try:
                head = s3.head_object(Bucket=BUCKET_NAME, Key=s3_key)
                if head["ContentLength"] == f_size:
                    status = Text("DONE", style="bold blue")
                    completed_size += f_size
                    done_count += 1
            except Exception:
                pass

        table.add_row(escape(f.name), f"{f_size / (1024 * 1024):.1f}", status)

    console.print(table)

    # Progress
    overall_pct = (completed_size / total_size) * 100 if total_size > 0 else 0
    console.print(
        f"\nProgress: {done_count}/{len(to_upload)} files ({overall_pct:.1f}%)"
    )

    if not pid:
        console.print(
            "\n[red]⚠️  Sweep script (full_ai_sweep_s3.py) "
            "is NOT currently running.[/red]"
        )


if __name__ == "__main__":
    main()
