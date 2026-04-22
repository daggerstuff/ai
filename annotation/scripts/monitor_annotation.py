import json
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


def monitor(batch_id="batch_001"):
    log_file = Path(f"ai/annotation/results/reddit_5k/{batch_id}.log")
    output_file = Path(f"ai/annotation/results/reddit_5k/{batch_id}_annotated.jsonl")

    _ = _read_last_line(log_file)
    count, last_line = _get_output_stats(output_file)
    if count > 0 and last_line:
        _inspect_last_output(last_line)


def _read_last_line(log_file: Path) -> str | None:
    if not log_file.exists():
        return None
    try:
        file_size = log_file.stat().st_size
        if file_size == 0:
            return None
        with log_file.open("rb") as file:
            file.seek(-min(500, file_size), 2)
            lines = file.readlines()
            return lines[-1].decode().strip() if lines else None
    except OSError:
        logger.exception("Unable to read last log line from %s", log_file)
        return None


def _get_output_stats(output_file: Path) -> tuple[int, str | None]:
    if not output_file.exists():
        return 0, None

    try:
        with output_file.open("rb") as file:
            chunks = iter(lambda: file.read(1024 * 1024), b"")
            count = sum(chunk.count(b"\n") for chunk in chunks)

            file.seek(0, 2)
            file_size = file.tell()

            if file_size > 0:
                file.seek(-1, 2)
                if file.read(1) != b"\n":
                    count += 1

            file.seek(-min(2048, file_size), 2)
            last_lines = file.read().splitlines()
            last_line = last_lines[-1].decode().strip() if last_lines else None

            return count, last_line
    except OSError:
        logger.exception("Unable to read output statistics from %s", output_file)
        return 0, None


def _inspect_last_output(last_line: str):
    try:
        data = json.loads(last_line)
    except json.JSONDecodeError:
        return

    data.get("agreement_metrics", {})

    # Check for "No conversation data" bug again
    first_ann = data.get("individual_annotations", [{}])[0]
    if "No conversation data" in first_ann.get("notes", ""):
        pass
    else:
        pass


if __name__ == "__main__":
    monitor()
