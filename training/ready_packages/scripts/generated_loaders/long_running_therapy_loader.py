
def load_long_running_therapy():
    """Load long-running therapy sessions"""
    from ai.training.ready_packages.utils.s3_dataset_loader import S3DatasetLoader

    loader = S3DatasetLoader()
    # Filter existing therapy datasets for long sessions (>20 turns)
    s3_paths = [
        "s3://pixel-data/gdrive/processed/professional_therapeutic/therapist_sft/...",
        "s3://pixel-data/gdrive/processed/professional_therapeutic/psych8k/..."
    ]

    long_sessions = []
    for s3_path in s3_paths:
        if loader.object_exists(s3_path):
            for conv in loader.stream_jsonl(s3_path):
                if len(conv.get('messages', [])) > 20:
                    long_sessions.append(conv)

    return long_sessions
