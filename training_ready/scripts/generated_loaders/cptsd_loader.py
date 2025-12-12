
def load_cptsd_datasets():
    """Load CPTSD-specific datasets"""
    from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader
    
    loader = S3DatasetLoader()
    # Tim Fletcher transcripts are CPTSD-focused
    s3_paths = [
        "s3://pixelated-training-data/gdrive/processed/voice_persona/tim_fletcher/...",
        "s3://pixelated-training-data/gdrive/processed/edge_cases/cptsd/..."
    ]
    
    cptsd_data = []
    for s3_path in s3_paths:
        if loader.object_exists(s3_path):
            cptsd_data.extend(list(loader.stream_jsonl(s3_path)))
    
    return cptsd_data
