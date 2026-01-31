import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Mapping of file paths
# Order matters: Process ai/models first to avoid breaking paths moving INTO ai/models.
REPLACEMENTS = [
    # 1. Existing ai/models content moved to components
    ("ai/models", "ai/models/components"),
    # 2. Specialized Pipelines
    ("ai/pipelines/edge_case_pipeline_standalone", "ai/pipelines/edge_case"),
    ("ai/pipelines/dual_persona_training", "ai/pipelines/dual_persona"),
    ("ai/pipelines/pixel_voice", "ai/pipelines/voice"),
    # 3. Integration Pipeline (Must be before ai/integration)
    ("ai/integration_pipeline", "ai/pipelines/orchestrator/integration_legacy"),
    # 4. Top-level to Pipeline/Infra mappings
    ("ai/dataset_pipeline", "ai/pipelines/orchestrator"),
    ("ai/data_designer", "ai/pipelines/design"),
    ("ai/distributed_processing", "ai/infrastructure/distributed"),
    ("ai/integration", "ai/infrastructure/integration"),
    ("ai/production", "ai/infrastructure/production"),
    ("ai/qa", "ai/infrastructure/qa"),
    # 5. Sourcing
    ("ai/academic_sourcing", "ai/sourcing/academic"),
    ("ai/journal_dataset_research", "ai/sourcing/journal"),
    ("ai/research_system", "ai/sourcing/research_system"),
    # 6. Notebooks/Data
    ("ai/notebooks", "ai/analysis/notebooks"),
    ("ai/voice_data", "ai/data/voice_logs"),
    # 7. Training
    ("ai/training_ready", "ai/training/ready_packages"),
    # 8. Models moves
    # We use ai/pixel/ (with slash) to matching the folder, avoiding ai/pixel_voice
    ("ai/pixel", "ai/models/pixel_core"),
    ("ai/pixel_voice", "ai/pipelines/voice"),
    ("ai/foundation", "ai/models/foundation"),
    # Fix potential double slashes if any
    # ("ai/models/components/", "ai/models/components/"), # No-op
]


def process_file(filepath):
    try:
        with open(filepath, encoding="utf-8") as f:
            content = f.read()
    except UnicodeDecodeError:
        logger.warning(f"Skipping binary file: {filepath}")
        return

    original_content = content

    for old, new in REPLACEMENTS:
        if old in content:
            # Simple string replacement for paths.
            # Risk: "ai/models_extra" -> "ai/models/components_extra" if distinct.
            # But unlikely to have such overlap in this codebase.
            content = content.replace(old, new)

    if content != original_content:
        logger.info(f"Updating paths in {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)


def main():
    root_dir = os.path.abspath("ai")
    logger.info(f"Scanning {root_dir} for path updates...")

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            if file.endswith((".py", ".md", ".sh", ".yaml", ".yml", ".json", ".txt")):
                filepath = os.path.join(root, file)
                if "fix_paths.py" in filepath:
                    continue
                process_file(filepath)


if __name__ == "__main__":
    main()
