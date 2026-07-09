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
ENDPOINT_URL = "https://hel1.your-objectstorage.com"
REGION_NAME = "hel1"
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
    access_key = os.environ.get("HETZNER_S3_ACCESS_KEY")
    secret_key = os.environ.get("HETZNER_S3_SECRET_KEY")
    if not access_key or not secret_key:
        console.print("[red]❌ Error: Missing credentials (HETZNER_S3_ACCESS_KEY/SECRET_KEY).[/red]")
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
        pids = subprocess.check_output(["pgrep", "-f", "full_ai_sweep_s3.py"]).decode().strip().split("\n")
        # Return the most recent one (highest PID usually)
        return pids[-1] if pids and pids[0] else None
    except Exception:
        return None


def get_current_file_progress(pid):
    if not pid:
        return None, 0
    try:
        fd_dir = Path(f"/proc/{pid}/fd")
        if fd_dir.exists():
            for fd_file in fd_dir.iterdir():
                try:
                    path = str(fd_file.readlink())
                    if any(ext in path for ext in ["ULTIMATE", ".jsonl", ".zip", ".csv"]):
                        fd_num = fd_file.name
                        pos_file = Path(f"/proc/{pid}/fdinfo/{fd_num}")
                        pos = 0
                        if pos_file.exists():
                            for pos_line in pos_file.read_text().splitlines():
                                if pos_line.startswith("pos:"):
                                    pos = int(pos_line.split(":")[1].strip())
                                    break
                        return path, pos
                except Exception:
                    continue
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
        if size > SIZE_THRESHOLD and size not in seen_sizes:
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
    console.print(f"\nProgress: {done_count}/{len(to_upload)} files ({overall_pct:.1f}%)")

    if not pid:
        console.print("\n[red]⚠️  Sweep script (full_ai_sweep_s3.py) is NOT currently running.[/red]")


if __name__ == "__main__":
    main()
