import logging
import os

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger(__name__)

# Fix double-application of renaming rules
REPLACEMENTS = [
    ("ai.models.pixel_core", "ai.models.pixel_core"),
    ("ai.models.foundation", "ai.models.foundation"),
    ("ai.models.components", "ai.models.components"),
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
        logger.info(f"Fixing {filepath}")
        with open(filepath, "w", encoding="utf-8") as f:
            f.write(content)


def main():
    root_dir = os.path.abspath("ai")
    logger.info(f"Scanning {root_dir} for double-renaming...")

    for root, dirs, files in os.walk(root_dir):
        dirs[:] = [d for d in dirs if not d.startswith(".")]

        for file in files:
            if file.endswith((".py", ".md", ".sh", ".yaml", ".yml", ".json", ".txt")):
                filepath = os.path.join(root, file)
                process_file(filepath)


if __name__ == "__main__":
    main()
