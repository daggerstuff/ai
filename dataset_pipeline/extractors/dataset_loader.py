import csv

from .s3_streamer import S3Streamer


class DatasetLoader:
    """Loads standardized/external JSONL and CSV datasets from S3."""

    def __init__(self, streamer: S3Streamer):
        self.streamer = streamer

    def load_jsonl(self, prefix, source_family, category):
        """Streams a JSONL or JSON dataset from a specific S3 prefix."""
        import json
        files = list(self.streamer.list_files(prefix))
        for f in files:
            if not (f.endswith((".jsonl", ".json"))):
                continue


            # If it's a huge standard JSON array
            if f.endswith(".json") and "combined_dataset" not in f:
                # We have to load the whole file into memory to parse it as JSON array
                content = "".join(list(self.streamer.stream_text(f)))
                try:
                    data = json.loads(content)
                    if isinstance(data, list):
                        for record in data:
                            yield {
                                "raw_data": record,
                                "metadata": {
                                    "source_family": source_family,
                                    "category": category,
                                    "file_key": f
                                }
                            }
                except Exception:
                    pass
            else:
                # Normal JSONL streaming
                for record in self.streamer.stream_jsonl(f):
                    yield {
                        "raw_data": record,
                        "metadata": {
                            "source_family": source_family,
                            "category": category,
                            "file_key": f
                        }
                    }

    def load_csv(self, prefix, source_family, category):
        """Streams a CSV dataset from a specific S3 prefix and yields dicts."""
        files = list(self.streamer.list_files(prefix))
        for f in files:
            if not f.endswith(".csv"):
                continue


            # Since S3Streamer yields lines, we can use csv.DictReader
            # but we need an iterator that returns strings
            line_iterator = self.streamer.stream_text(f)
            reader = csv.DictReader(line_iterator)

            for row in reader:
                yield {
                    "raw_data": row,
                    "metadata": {
                        "source_family": source_family,
                        "category": category,
                        "file_key": f
                    }
                }
