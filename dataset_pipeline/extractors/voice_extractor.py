from .s3_streamer import S3Streamer


class VoiceExtractor:
    """Extracts voice transcripts from S3 and yields raw conversational items."""

    def __init__(self, streamer: S3Streamer):
        self.streamer = streamer
        self.base_prefix = "archive/local_voice_import/"

    def extract_all(self):
        """Yields every voice record found in the S3 voice directory."""
        # Find all JSONL files in the exports/ directories (voice files are 2
        # levels deep: <voice>/exports/*.jsonl — needs recursive listing)
        all_files = list(self.streamer.list_files(self.base_prefix, recursive=True))
        jsonl_files = [f for f in all_files if f.endswith(".jsonl")]

        for file_key in jsonl_files:
            # e.g. archive/local_voice_import/big_think_voice/exports/bigthink_conversations.jsonl
            parts = file_key.split("/")
            # The directory name is like 'big_think_voice'
            category_dir = parts[-3] if len(parts) >= 3 else "unknown_voice"

            for record in self.streamer.stream_jsonl(file_key):
                yield {
                    "raw_data": record,
                    "metadata": {"source_family": "voice_training", "category": category_dir, "file_key": file_key},
                }
