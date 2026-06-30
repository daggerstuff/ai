import argparse
import hashlib
import json
import logging
import random
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")


def _anon_id(raw: str) -> str:
    """Map a raw username to a stable anonymous ID (no PII leakage)."""
    digest = hashlib.sha256(raw.encode()).hexdigest()[:12]
    return f"anon_{digest}"


class RedditConverter:
    """Reddit-to-training converter with anonymization and gate-scoring."""

    def __init__(self, anon: bool = True) -> None:
        self._anon = anon
        self._seen_usernames: dict[str, str] = {}

    def _scrub(self, text: str) -> str:
        """Strip anything that looks like a Reddit /u/ or r/ reference for privacy."""
        import re as _re

        text = _re.sub(r"/u/\w+", "[user]", text)
        return _re.sub(r"/r/\w+", "[subreddit]", text)

    def _canonical_anon(self, username: str | None) -> str:
        if not username:
            return "anon_unknown"
        if username not in self._seen_usernames:
            self._seen_usernames[username] = _anon_id(username)
        return self._seen_usernames[username]

    def convert(self, data: dict) -> list[dict]:
        """Convert a Reddit post dict into training message pairs."""
        if not isinstance(data, dict):
            return []

        if "messages" in data:
            return [data]

        title = self._scrub(data.get("title", ""))
        selftext = self._scrub(data.get("selftext", ""))
        content = self._scrub(data.get("content", ""))
        body = self._scrub(data.get("body", ""))
        author = data.get("author")  # kept raw here; only exposed via anonymous ID

        prompt = ""
        if title or selftext:
            prompt = f"{title}\n\n{selftext}".strip()
        elif content:
            prompt = content
        elif body:
            prompt = body

        pairs: list[dict] = []
        comments = data.get("comments", [])
        if comments and isinstance(comments, list):
            for comment in comments:
                if not isinstance(comment, dict):
                    continue
                comment_body = self._scrub(comment.get("body", ""))
                comment_author = comment.get("author")
                if prompt and comment_body:
                    pairs.append(
                        {
                            "messages": [
                                {"role": "user", "content": prompt},
                                {"role": "assistant", "content": comment_body},
                            ],
                            "meta": {
                                "source": "reddit",
                                "author_anon": self._canonical_anon(author),
                                "commenter_anon": self._canonical_anon(comment_author),
                                "subreddit": self._scrub(data.get("subreddit", "")),
                                "score": data.get("score", 0),
                            },
                        }
                    )
        else:
            completion = self._scrub(data.get("response", data.get("completion", "")))
            if prompt and completion:
                pairs.append(
                    {
                        "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}],
                        "meta": {
                            "source": "reddit",
                            "author_anon": self._canonical_anon(author),
                            "subreddit": self._scrub(data.get("subreddit", "")),
                            "score": data.get("score", 0),
                        },
                    }
                )

        return pairs


def extract_conversations(data):
    """Extract prompt/completion pairs from reddit data."""
    if not isinstance(data, dict):
        return

    # Heuristic 1: If it's already in format
    if "messages" in data:
        yield data
        return

    # Heuristic 2: Reddit post with title/selftext
    title = data.get("title", "")
    selftext = data.get("selftext", "")
    content = data.get("content", "")
    body = data.get("body", "")

    prompt = ""
    if title or selftext:
        prompt = f"{title}\n\n{selftext}".strip()
    elif content:
        prompt = content
    elif body:
        prompt = body

    # Process comments
    comments = data.get("comments", [])
    if comments and isinstance(comments, list):
        for comment in comments:
            if not isinstance(comment, dict):
                continue
            comment_body = comment.get("body", "")
            if prompt and comment_body:
                yield {
                    "messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": comment_body}]
                }
    else:
        # Fallback if there are responses or completion fields
        completion = data.get("response", data.get("completion", ""))
        if prompt and completion:
            yield {"messages": [{"role": "user", "content": prompt}, {"role": "assistant", "content": completion}]}


def process_file(file_path, anon: bool = True):
    """Process a single JSON or JSONL file and yield message pairs."""
    converter = RedditConverter(anon=anon)
    try:
        with open(file_path, encoding="utf-8") as f:
            if str(file_path).endswith(".jsonl"):
                for line_num, line in enumerate(f, 1):
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = json.loads(line)
                        yield from converter.convert(data)
                    except json.JSONDecodeError as e:
                        logging.warning(f"Failed to parse line {line_num} in {file_path}: {e}")
            else:
                try:
                    data = json.load(f)
                    if isinstance(data, list):
                        for item in data:
                            yield from converter.convert(item)
                    else:
                        yield from converter.convert(data)
                except json.JSONDecodeError as e:
                    logging.warning(f"Failed to parse JSON file {file_path}: {e}")
    except Exception as e:
        logging.error(f"Error reading file {file_path}: {e}")


def main():
    parser = argparse.ArgumentParser(description="Convert raw Reddit JSONs to training JSONL format")
    parser.add_argument("--input-dir", "-i", type=str, required=True, help="Input directory containing raw JSONs")
    parser.add_argument(
        "--output-dir", "-o", type=str, required=True, help="Output directory for train.jsonl and val.jsonl"
    )
    parser.add_argument("--val-split", "-v", type=float, default=0.1, help="Validation split ratio")
    parser.add_argument("--no-anon", action="store_true", help="Disable username anonymization")
    args = parser.parse_args()

    input_dir = Path(args.input_dir)
    output_dir = Path(args.output_dir)

    if not input_dir.exists() or not input_dir.is_dir():
        logging.error(f"Input directory does not exist or is not a directory: {input_dir}")
        return

    output_dir.mkdir(parents=True, exist_ok=True)

    train_path = output_dir / "train.jsonl"
    val_path = output_dir / "val.jsonl"

    total_count = 0
    train_count = 0
    val_count = 0
    file_count = 0

    try:
        with open(train_path, "w", encoding="utf-8") as f_train, open(val_path, "w", encoding="utf-8") as f_val:
            for file_path in input_dir.glob("**/*"):
                if file_path.is_file() and file_path.suffix.lower() in [".json", ".jsonl"]:
                    file_count += 1
                    for conv in process_file(file_path, anon=not args.no_anon):
                        line = json.dumps(conv, ensure_ascii=False) + "\n"
                        total_count += 1

                        if random.random() < args.val_split:
                            f_val.write(line)
                            val_count += 1
                        else:
                            f_train.write(line)
                            train_count += 1

        logging.info(f"Processed {file_count} files.")
        logging.info(f"Saved {train_count} samples to {train_path}")
        logging.info(f"Saved {val_count} samples to {val_path}")

        if total_count == 0:
            logging.warning("No conversations extracted. Check input format.")
    except Exception as e:
        logging.error(f"Failed to write output files: {e}")


if __name__ == "__main__":
    main()
