import os
from pathlib import Path


def create_folder_structure(base_path: Path):
    expected_dirs = [
        "configs/stage_configs",
        "configs/model_configs",
        "configs/infrastructure",
        "configs/hyperparameters",
        "datasets/stage1_foundation",
        "datasets/stage2_reasoning",
        "datasets/stage3_edge",
        "datasets/stage4_voice",
        "models/moe",
        "models/base",
        "models/experimental",
        "pipelines/integrated",
        "pipelines/edge",
        "pipelines/voice",
        "infrastructure/kubernetes",
        "infrastructure/helm",
        "infrastructure/docker",
        "tools/data_preparation",
        "tools/validation",
        "tools/monitoring",
        "experimental/research_models",
        "experimental/novel_pipelines",
        "experimental/future_features",
        "scripts/output",
    ]
    created = []
    existing = []
    for d in expected_dirs:
        dir_path = base_path / d
        if dir_path.exists():
            existing.append(d)
        else:
            dir_path.mkdir(parents=True)
            created.append(d)
            (dir_path / ".gitkeep").touch()

    print(f"✅ Created {len(created)} directories. {len(existing)} directories already existed.")
    print("📂 Folder structure:")
    return created, existing

def print_structure(startpath, original_base, level):
    for root, dirs, files in os.walk(startpath):
        level = root.replace(str(original_base), "").count(os.sep)
        indent = " " * 4 * (level)
        dirname = os.path.basename(root)
        if dirname.startswith(".") or dirname == "__pycache__":
            dirs.clear()
            continue
        print(f"{indent}{dirname}/")
        subindent = " " * 4 * (level + 1)
        for f in files:
            if not f.startswith("."):
                print(f"{subindent}{f}")
