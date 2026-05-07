from pathlib import Path

import pytest

from training.scripts.create_folder_structure import create_folder_structure, print_structure


def test_create_folder_structure_new_dirs(tmp_path: Path, capsys):
    """Test creating directories in an empty base path."""
    created, existing = create_folder_structure(tmp_path)

    # Expected directories from the script
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

    assert len(created) == len(expected_dirs)
    assert len(existing) == 0

    # Check that all expected directories were created and contain a .gitkeep file
    for expected_dir in expected_dirs:
        dir_path = tmp_path / expected_dir
        assert dir_path.exists()
        assert dir_path.is_dir()

        gitkeep_path = dir_path / ".gitkeep"
        assert gitkeep_path.exists()
        assert gitkeep_path.is_file()

    # Capture output to ensure no errors were printed
    captured = capsys.readouterr()
    assert "✅ Created" in captured.out
    assert "📂 Folder structure:" in captured.out

def test_create_folder_structure_existing_dirs(tmp_path: Path, capsys):
    """Test creating directories when they already exist."""
    # Run once to create them
    create_folder_structure(tmp_path)

    # Clear captured output from first run
    capsys.readouterr()

    # Run again, everything should be in 'existing'
    created, existing = create_folder_structure(tmp_path)

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

    assert len(created) == 0
    assert len(existing) == len(expected_dirs)

    captured = capsys.readouterr()
    assert "✅ Created 0 directories" in captured.out
    assert f"{len(expected_dirs)} directories already existed" in captured.out

def test_print_structure(tmp_path: Path, capsys):
    """Test the structure printing functionality."""
    # Create a small dummy structure
    dir1 = tmp_path / "dir1"
    dir1.mkdir()
    dir2 = tmp_path / "dir2"
    dir2.mkdir()
    subdir = dir1 / "subdir"
    subdir.mkdir()

    # Also create a hidden dir and __pycache__ that should be ignored
    (tmp_path / ".hidden").mkdir()
    (tmp_path / "__pycache__").mkdir()

    # Call the print function
    print_structure(tmp_path, tmp_path, 0)

    captured = capsys.readouterr()

    # Check that our visible dirs are there
    assert "dir1/" in captured.out
    assert "dir2/" in captured.out
    assert "subdir/" in captured.out

    # Check that ignored dirs are NOT there
    assert ".hidden/" not in captured.out
    assert "__pycache__/" not in captured.out
