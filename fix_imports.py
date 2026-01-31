import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Mapping of old strings to new strings.
# CRITICAL: Order matters. More specific/longer matches should generally come before shorter ones
# to avoid partial replacements (e.g. ai.integration_pipeline before ai.integration).
# Also, ai.models -> ai.models.components must happen BEFORE ai.foundation -> ai.models.foundation
# so that we don't end up with ai.models.foundation.

REPLACEMENTS = [
    # 1. Specialized Pipelines
    ("ai.pipelines.edge_case_pipeline_standalone", "ai.pipelines.edge_case"),
    ("ai.pipelines.dual_persona_training", "ai.pipelines.dual_persona"),
    ("ai.pipelines.pixel_voice", "ai.pipelines.voice"),  # Merged
    # 2. Integration Pipeline (Must be before ai.integration)
    ("ai.integration_pipeline", "ai.pipelines.orchestrator.integration_legacy"),
    # 3. Top-level to Pipeline/Infra mappings
    ("ai.dataset_pipeline", "ai.pipelines.orchestrator"),
    ("ai.data_designer", "ai.pipelines.design"),
    ("ai.distributed_processing", "ai.infrastructure.distributed"),
    ("ai.integration", "ai.infrastructure.integration"),
    ("ai.production", "ai.infrastructure.production"),
    ("ai.qa", "ai.infrastructure.qa"),
    # 4. Sourcing
    ("ai.academic_sourcing", "ai.sourcing.academic"),
    ("ai.journal_dataset_research", "ai.sourcing.journal"),
    ("ai.research_system", "ai.sourcing.research_system"),
    # 5. Notebooks/Data
    ("ai.notebooks", "ai.analysis.notebooks"),
    ("ai.voice_data", "ai.data.voice_logs"),
    # 6. Training
    ("ai.training_ready", "ai.training.ready_packages"),
    # 7. Core Models & Pixel
    # ai.models -> ai.models.components. Must be before foundation move.
    ("ai.models", "ai.models.components"),
    ("ai.pixel_voice", "ai.pipelines.voice"),  # Top level one
    ("ai.pixel", "ai.models.pixel_core"),
    # 8. Foundation (Moved to ai.models.foundation)
    # Applying this after ai.models fix ensures safely.
    ("ai.foundation", "ai.models.foundation"),
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
            content = content.replace(old, new)

    if content != original_content:
        logger.info(f"Updating {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)


def main():
    root_dir = os.path.abspath("ai")

    logger.info(f"Scanning {root_dir}...")

    for root, dirs, files in os.walk(root_dir):
        # Skip hidden folders like .git or __pycache__
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            # Added more extensions
            if file.endswith((".py", ".md", ".sh", ".yaml", ".yml", ".json", ".txt")):
                filepath = os.path.join(root, file)
                # Skip naming conflicts with the script itself if placed there
                if "fix_imports.py" in filepath:
                    continue

                process_file(filepath)


if __name__ == "__main__":
    main()
