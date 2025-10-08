#!/usr/bin/env python3
"""
HuggingFace Upload Script - Model + GGUF Variants
"""

import contextlib
import os
from pathlib import Path

from huggingface_hub import HfApi, create_repo, upload_file, upload_folder


def upload_to_huggingface(model_path, gguf_path, model_name="Wayfarer2-Pixelated"):
    """Upload model and GGUF variants to HuggingFace"""


    # Initialize HF API
    api = HfApi()

    # Get username from token
    try:
        user_info = api.whoami()
        username = user_info["name"]
        repo_id = f"{username}/{model_name}"
    except Exception:
        return False

    # Create repository
    try:
        create_repo(
            repo_id=repo_id,
            exist_ok=True,
            repo_type="model",
            private=False
        )
    except Exception:
        return False

    # Upload main model
    if os.path.exists(model_path):
        try:
            upload_folder(
                folder_path=model_path,
                repo_id=repo_id,
                repo_type="model",
                commit_message="Upload fine-tuned Wayfarer2-Pixelated model"
            )
        except Exception:
            return False
    else:
        pass

    # Upload GGUF variants
    if os.path.exists(gguf_path):

        gguf_files = list(Path(gguf_path).glob("*.gguf"))

        for gguf_file in gguf_files:
            with contextlib.suppress(Exception):
                upload_file(
                    path_or_fileobj=str(gguf_file),
                    path_in_repo=gguf_file.name,
                    repo_id=repo_id,
                    repo_type="model",
                    commit_message=f"Add GGUF variant: {gguf_file.name}"
                )
    else:
        pass

    # Upload model card
    model_card_path = "README.md"
    if os.path.exists(model_card_path):
        with contextlib.suppress(Exception):
            upload_file(
                path_or_fileobj=model_card_path,
                path_in_repo="README.md",
                repo_id=repo_id,
                repo_type="model",
                commit_message="Add comprehensive model card"
            )

    return True

def main():
    import sys

    model_path = sys.argv[1] if len(sys.argv) > 1 else "./wayfarer-finetuned"
    gguf_path = sys.argv[2] if len(sys.argv) > 2 else "./gguf-exports"
    model_name = sys.argv[3] if len(sys.argv) > 3 else "Wayfarer2-Pixelated"

    success = upload_to_huggingface(model_path, gguf_path, model_name)

    if success:
        pass
    else:
        pass

    return success

if __name__ == "__main__":
    main()
