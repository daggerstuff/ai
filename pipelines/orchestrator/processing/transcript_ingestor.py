import logging
import json
import os
from pathlib import Path
from typing import List, Dict, Optional
import sys

# Adjust import for project structure
try:
    from ai.common.llm_client import LLMClient
except ImportError:
    project_root = Path(__file__).resolve().parents[3]
    sys.path.append(str(project_root))
    from ai.common.llm_client import LLMClient

logger = logging.getLogger(__name__)

class TranscriptIngestor:
    """
    Ingests and filters local transcript files for Stage 4 (Voice/Persona) training.
    """

    # Authors/Folders known to be high-quality therapeutic sources
    ALLOWLIST = [
        "Tim Fletcher",
        "Therapy in a Nutshell",
        "Dr. Daniel Fox",
        "DoctorRamani",
        "Crappy Childhood Fairy",
        "Patrick Teahan",
        "Heidi Priebe",
        "Irene Lyon",
        "Dr. Scott Eilers",
        "Dr. Todd Grande",
        "Psych2Go",
        "Surviving Narcissism"
    ]

    BLOCKLIST = [
        "Jimmy Kimmel Live",
        "The Late Show",
        "LastWeekTonight",
        "The Diary Of A CEO", # Often pop-psych or business
        "Jay Shetty Podcast", # Often pop-psych
        "Tedx Talks",         # Too varied
        "Big Think"           # Too varied
    ]

    def __init__(self, source_path: str = ".notes/transcripts", output_base_path: str = "ai/training/ready_packages/datasets"):
        self.source_path = Path(source_path)
        self.output_path = Path(output_base_path) / "stage4_voice" / "processed_transcripts"
        self.output_path.mkdir(parents=True, exist_ok=True)
        self.llm = LLMClient(driver="mock") # Use mock for speed/cost, switch to OpenAI for real filtering

    def get_files(self) -> List[Path]:
        """Recursively finds .txt files in allowed directories."""
        all_files = []
        for root, dirs, files in os.walk(self.source_path):
            root_path = Path(root)
            # Check if this folder is allowed
            folder_name = root_path.name

            # Simple Allowlist Check (can be relaxed)
            # If the folder is in ALLOWLIST or part of the path is in ALLOWLIST
            is_allowed = False
            for allowed in self.ALLOWLIST:
                if allowed.lower() in str(root_path).lower():
                    is_allowed = True
                    break

            # Start strict: only allow specific Lists
            if not is_allowed:
                # Optional: Check blocklist specifically to exclude
                continue

            for file in files:
                if file.endswith(".txt"):
                    all_files.append(root_path / file)

        return all_files

    def parse_transcript(self, file_path: Path) -> str:
        """Extracts the main body text from the transcript file."""
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                lines = f.readlines()

            content = []
            is_body = False
            for line in lines:
                stripped = line.strip()
                # Heuristic: Skip headers
                if stripped.startswith("## Transcript"):
                    is_body = True
                    continue
                if stripped.startswith("# ") or stripped.startswith("**"):
                    continue

                if is_body or len(lines) < 20: # If short or no header found, take content
                    if stripped:
                        content.append(stripped)

            return " ".join(content)
        except Exception as e:
            logger.error(f"Error reading {file_path}: {e}")
            return ""

    def validate_content(self, text: str, filename: str) -> bool:
        """
        Uses LLM to checking clinical relevance and safety.
        """
        # Quick heuristic check first
        if len(text) < 1000: # Too short
            return False

        # Mock LLM check - In production, use real LLM
        # validation = self.llm.generate_structured(...)
        # For now, blindly accept if it passed allowlist and length check
        return True

    def process_batch(self, batch_size: int = 100):
        """Main execution method."""
        files = self.get_files()
        logger.info(f"Found {len(files)} files in allowlisted directories.")

        processed_data = []
        for i, file_path in enumerate(files):
            if i >= batch_size: break

            logger.info(f"Processing: {file_path.name}")
            raw_text = self.parse_transcript(file_path)

            if self.validate_content(raw_text, file_path.name):
                # Infer author from path
                author = "Unknown"
                is_primary_persona = False

                # Check for Tim Fletcher specifically first
                if "Tim Fletcher" in str(file_path):
                    author = "Tim Fletcher"
                    is_primary_persona = True
                else:
                    for allowed in self.ALLOWLIST:
                        if allowed.lower() in str(file_path).lower():
                            author = allowed
                            break

                processed_data.append({
                    "source_file": file_path.name,
                    "author_persona": author,
                    "is_primary_persona": is_primary_persona,
                    "content": raw_text, # Full text, or chunked
                    "type": "therapeutic_transcript",
                    "validation_status": "filtered_ingested"
                })

        # Export
        if processed_data:
            output_file = self.output_path / "voice_training_data_001.json"
            with open(output_file, "w") as f:
                json.dump(processed_data, f, indent=2)
            logger.info(f"Exported {len(processed_data)} items to {output_file}")
            return str(output_file)

        return None
