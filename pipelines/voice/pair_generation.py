import json
import logging
from pathlib import Path
from typing import Any, cast

import torch
import transformers

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("PairGeneration")


def _redacted(text: str, max_chars: int = 24) -> str:
    """Return a short non-sensitive preview of therapeutic text for logs.

    Full transcript content is sensitive (PHI). Logs must never contain it.
    """
    stripped = text.strip().replace("\n", " ")
    if len(stripped) <= max_chars:
        return stripped
    return stripped[:max_chars] + "..."


class AuthenticityValidator:
    def __init__(self, device: int = 0 if torch.cuda.is_available() else -1):
        logger.info(f"Loading authenticity classification model on device {device}...")
        pipeline_func = cast(Any, transformers.pipeline)
        self.classifier = pipeline_func(
            "zero-shot-classification", model="facebook/bart-large-mnli", device=device
        )

    def is_authentic(self, text: str, threshold: float = 0.5) -> bool:
        if not text.strip():
            return False
        try:
            result = cast(
                dict[str, Any],
                self.classifier(text, candidate_labels=["authentic", "scripted", "robotic"]),
            )
            labels_list = cast(list[str], result["labels"])
            scores_list = cast(list[float], result["scores"])
            auth_score = scores_list[labels_list.index("authentic")]
            return auth_score >= threshold
        except Exception as e:
            logger.warning(f"Authenticity check failed: {e}")
            return False


class TherapeuticPairGenerator:
    def __init__(
        self,
        input_dir: str = "ai/data/features",
        output_dir: str = "ai/data/pairs",
        empathy_threshold: float = 0.5,
    ):
        self.input_dir = Path(input_dir)
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.empathy_threshold = empathy_threshold
        self.authenticity_validator = AuthenticityValidator()

    def extract_pairs(self, segments: list[dict[str, Any]]) -> list[dict[str, Any]]:
        pairs = []
        for i in range(len(segments) - 1):
            current_seg = segments[i]
            next_seg = segments[i + 1]

            if current_seg.get("role") == "Client" and next_seg.get("role") == "Therapist":
                client_text = current_seg.get("text", "")
                therapist_text = next_seg.get("text", "")

                # Validation checks on Therapist response
                features = next_seg.get("features", {})
                empathy_score = features.get("empathy_score", 0.0)

                if empathy_score < self.empathy_threshold:
                    logger.debug(f"Rejecting pair due to low empathy score ({empathy_score:.2f})")
                    continue

                if not self.authenticity_validator.is_authentic(therapist_text):
                    logger.debug(
                        f"Rejecting pair due to failed authenticity check (len={len(therapist_text)}, preview={_redacted(therapist_text)})"
                    )
                    continue

                pairs.append(
                    {
                        "prompt": client_text,
                        "response": therapist_text,
                        "context": {
                            "client_emotion": current_seg.get("features", {}).get("emotion"),
                            "therapist_emotion": features.get("emotion"),
                            "therapist_empathy": empathy_score,
                            "therapist_rhythm": features.get("rhythm_wps"),
                        },
                    }
                )
        return pairs

    def process_all(self) -> None:
        if not self.input_dir.exists():
            logger.error(f"Input directory {self.input_dir} does not exist.")
            return

        feature_files = list(self.input_dir.glob("*_features.json"))
        logger.info(f"Found {len(feature_files)} feature files to process for pairs.")

        for file_path in feature_files:
            out_file = self.output_dir / f"{file_path.stem.replace('_features', '')}_pairs.jsonl"
            if out_file.exists():
                logger.info(f"Skipping {file_path}, pairs already generated.")
                continue

            try:
                with open(file_path, encoding="utf-8") as f:
                    data = json.load(f)
                    segments = data.get("segments", [])

                logger.info(f"Extracting therapeutic pairs for {file_path.name}...")
                valid_pairs = self.extract_pairs(segments)

                with open(out_file, "w", encoding="utf-8") as f:
                    for pair in valid_pairs:
                        f.write(json.dumps(pair) + "\n")

                logger.info(
                    f"Saved {len(valid_pairs)} high-quality therapeutic pairs to {out_file}"
                )

            except Exception as e:
                logger.error(f"Failed to process {file_path}: {e}")


if __name__ == "__main__":
    generator = TherapeuticPairGenerator()
    generator.process_all()
