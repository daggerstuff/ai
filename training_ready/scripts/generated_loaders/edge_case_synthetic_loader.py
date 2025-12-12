
def load_edge_case_synthetic():
    """Load synthetic edge case dataset"""
    from ai.training_ready.utils.s3_dataset_loader import S3DatasetLoader
    
    loader = S3DatasetLoader()
    s3_path = "s3://pixelated-training-data/gdrive/processed/edge_cases/synthetic.jsonl"
    
    if loader.object_exists(s3_path):
        return list(loader.stream_jsonl(s3_path))
    else:
        # Generate synthetic edge cases
        raise FileNotFoundError("Synthetic edge cases not found. Generate using edge case generator.")
