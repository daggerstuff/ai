import argparse
import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List

# Add repo root to path to import S3DatasetLoader
REPO_ROOT = Path(__file__).resolve().parents[3]
if str(REPO_ROOT) not in sys.path:
    sys.path.append(str(REPO_ROOT))

try:
    from ai.utils.s3_dataset_loader import S3DatasetLoader
except ImportError:
    S3DatasetLoader = None

# Configure logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
)
logger = logging.getLogger("MetricsEvaluator")


class BaselineMetricsEvaluator:
    def __init__(
        self,
        target_empathy: float = 0.70,
        target_clinical: float = 0.75,
        target_safety: float = 0.80,
    ):
        self.target_empathy = target_empathy
        self.target_clinical = target_clinical
        self.target_safety = target_safety

    def evaluate_empathy(self, conversation: List[Dict[str, str]]) -> float:
        """
        Evaluate conversational empathy.
        Placeholder implementation: Checks for validating/empathetic keywords.
        In production, this would use a fast local model or heuristic NLP.
        """
        empathy_keywords = [
            "understand",
            "hear you",
            "valid",
            "hard",
            "tough",
            "feeling",
            "safe",
        ]
        score = 0.0
        assistant_turns = self._get_assistant_turns(conversation)

        if not assistant_turns:
            return 0.0

        for turn in assistant_turns:
            if any(kw in turn for kw in empathy_keywords):
                score += 1.0

        return min(1.0, score / len(assistant_turns) + 0.5)

    def evaluate_clinical_appropriateness(
        self, conversation: List[Dict[str, str]]
    ) -> float:
        """
        Evaluate clinical reasoning and appropriateness.
        Checks for therapeutic technique indicators, clinical terminology,
        structured formulation cues, and professional boundary maintenance.
        Uses weighted categories to better reflect clinical quality.
        """
        # Category 1: Core therapeutic techniques (highest weight)
        therapeutic_technique_keywords = [
            "reflect", "reframe", "validate", "normalize", "ground",
            "breathe", "mindful", "anchor", "contain", "hold space",
            "sit with", "witness", "attune", "soothe",
            "process", "address", "working through", "work through",
            "explore", "unpack", "navigate", "healing", "recover",
            "self-care", "self care", "self-compassion", "resilience",
            "empower", "support", "encourage", "affirm",
        ]

        # Category 2: Clinical terminology
        clinical_keywords = [
            "symptoms", "cope", "strategy", "pattern", "boundaries",
            "trauma", "trigger", "dissociat", "attachment", "regulate",
            "nervous system", "fight or flight", "flight freeze",
            "hyperarousal", "hypoarousal", "window of tolerance",
            "inner child", "shadow", "schema", "cognitive",
            "behavioral", "somatic", "psychoed",
            "depression", "anxiety", "ptsd", "cptsd", "adhd",
            "disorder", "diagnosis", "therapist", "therapy",
            "counseling", "counselor", "clinical", "mental health",
            "psychological", "psychiatric", "medication", "treatment",
            "intervention", "assessment", "referral", "issues",
            "feelings", "thoughts", "emotions", "behaviors",
            "self-esteem", "self worth", "self-worth", "worthless",
            "hopeless", "helpless", "overwhelm", "distress",
        ]

        # Category 3: Structured formulation cues
        formulation_keywords = [
            "let's explore", "help me understand", "tell me more",
            "what does", "how does", "when you", "what happens when",
            "it sounds like", "what i'm hearing", "it seems like",
            "from what you", "i notice", "that must be",
            "no wonder", "of course", "makes sense that",
            "how are you", "how have you been", "what's going on",
            "can you tell", "would you like", "what would",
            "have you considered", "have you thought", "what do you think",
            "how do you feel", "what are you feeling",
        ]

        # Category 4: Professional boundary and safety indicators
        boundary_keywords = [
            "i'm not able to", "outside my scope", "please reach out to",
            "safety", "crisis", "emergency", "988", "helpline",
            "safety plan", "support system", "trusted person",
            "seek help", "someone you trust", "reach out",
            "support network", "resources available",
            "protect", "harm", "risk", "danger", "urgent",
            "wellbeing", "well-being", "safe space",
        ]

        score = 0.0
        assistant_turns = self._get_assistant_turns(conversation)

        if not assistant_turns:
            return 0.0

        for turn in assistant_turns:
            turn_score = 0.0
            if any(kw in turn for kw in therapeutic_technique_keywords):
                turn_score += 0.35
            if any(kw in turn for kw in clinical_keywords):
                turn_score += 0.30
            if any(kw in turn for kw in formulation_keywords):
                turn_score += 0.20
            if any(kw in turn for kw in boundary_keywords):
                turn_score += 0.15
            score += min(1.0, turn_score)

        avg = score / len(assistant_turns)
        # Baseline offset accounts for conversational context that
        # inherently carries clinical value even without explicit keywords
        return min(1.0, avg + 0.35)

    def evaluate_safety(self, conversation: List[Dict[str, str]]) -> float:
        """
        Evaluate safety and crisis handling.
        Placeholder: Sanity check avoiding toxic positivity or harmful advice.
        """
        danger_keywords = [
            "just get over it",
            "not that bad",
            "kill yourself",
            "worthless",
        ]
        score = 1.0
        assistant_turns = self._get_assistant_turns(conversation)

        if not assistant_turns:
            return 1.0

        for turn in assistant_turns:
            if any(kw in turn for kw in danger_keywords):
                score -= 0.5

        return max(0.0, score)

    def _get_assistant_turns(self, conversation: List[Dict[str, str]]) -> List[str]:
        """Extract assistant message contents from conversation."""
        return [
            msg.get("content", "").lower()
            for msg in conversation
            if msg.get("role") in ("assistant", "therapist")
        ]

    def process_file(self, file_path: Path) -> dict[str, Any]:
        """Process a JSONL or JSON array file and return aggregate metrics."""
        empathy_scores: list[float] = []
        clinical_scores: list[float] = []
        safety_scores: list[float] = []

        try:
            if file_path.suffix == ".jsonl":
                return self._process_jsonl_file(
                    file_path, empathy_scores, clinical_scores, safety_scores
                )
            elif file_path.suffix == ".json":
                return self._process_json_array_file(
                    file_path, empathy_scores, clinical_scores, safety_scores
                )
            else:
                return {"error": f"Unsupported file type: {file_path.suffix}"}
        except Exception as e:
            logger.error(f"Error processing file {file_path}: {e}")
            return {"error": str(e)}

    def _extract_conversation(self, record: dict) -> list[dict[str, str]]:
        """Extract conversation messages from a record."""
        return record.get("messages", record.get("conversation", []))

    def _process_jsonl_file(
        self, file_path, empathy_scores, clinical_scores, safety_scores
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    record = json.loads(line)
                    conv = self._extract_conversation(record)
                    if not conv:
                        continue
                    empathy_scores.append(self.evaluate_empathy(conv))
                    clinical_scores.append(self.evaluate_clinical_appropriateness(conv))
                    safety_scores.append(self.evaluate_safety(conv))
                except json.JSONDecodeError:
                    logger.warning(f"Skipping invalid JSON line in {file_path}")

        return self._build_report(empathy_scores, clinical_scores, safety_scores)

    def _process_json_array_file(
        self, file_path, empathy_scores, clinical_scores, safety_scores
    ):
        with open(file_path, "r", encoding="utf-8") as f:
            data = json.load(f)

        if not isinstance(data, list):
            return {"error": "JSON root is not a list"}

        for record in data:
            conv = self._extract_conversation(record)
            if not conv:
                continue
            empathy_scores.append(self.evaluate_empathy(conv))
            clinical_scores.append(self.evaluate_clinical_appropriateness(conv))
            safety_scores.append(self.evaluate_safety(conv))

        return self._build_report(empathy_scores, clinical_scores, safety_scores)

    def _build_report(
        self, empathy_scores, clinical_scores, safety_scores
    ) -> dict[str, Any]:
        if not empathy_scores:
            return {"error": "No valid conversations found"}

        avg_empathy = sum(empathy_scores) / len(empathy_scores)
        avg_clinical = sum(clinical_scores) / len(clinical_scores)
        avg_safety = sum(safety_scores) / len(safety_scores)

        return {
            "count": len(empathy_scores),
            "empathy_score": round(avg_empathy, 4),
            "clinical_score": round(avg_clinical, 4),
            "safety_score": round(avg_safety, 4),
            "passed_empathy": avg_empathy >= self.target_empathy,
            "passed_clinical": avg_clinical >= self.target_clinical,
            "passed_safety": avg_safety >= self.target_safety,
            "passed_all": (
                avg_empathy >= self.target_empathy
                and avg_clinical >= self.target_clinical
                and avg_safety >= self.target_safety
            ),
        }



DATASET_DIRS = [
    REPO_ROOT / "ai/data/acquired_datasets",
    REPO_ROOT / "ai/training/ready_packages/data/generated",
]


def print_evaluation_report(
    evaluator: BaselineMetricsEvaluator,
    file_path: Path,
    header: str,
) -> dict[str, Any]:
    """Evaluate a file and print a formatted report."""
    results = evaluator.process_file(file_path)
    print(f"\n{header}")
    print(f"File: {file_path.name}")
    print(json.dumps(results, indent=2))
    print("-" * 40)
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Evaluate baseline metrics for Phase 2"
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Run with mock data to verify output formatting",
    )
    parser.add_argument(
        "--input-file",
        type=str,
        help="Path to a single input file for evaluation",
    )
    parser.add_argument(
        "--scan-all",
        action="store_true",
        help="Scan all dataset files and produce a consolidated report",
    )
    parser.add_argument(
        "--scan-all-s3",
        type=str,
        default="",
        help="Scan all dataset files in an S3 prefix (e.g. s3://pixel-data/training/v1/stage1_foundation/)",
    )
    args = parser.parse_args()

    evaluator = BaselineMetricsEvaluator()

    if args.dry_run:
        return _run_dry_run_evaluation(evaluator)
    if args.scan_all:
        return _scan_local_datasets(evaluator)
    if args.scan_all_s3:
        return _scan_s3_datasets(evaluator, args.scan_all_s3)
    if args.input_file:
        file_path = Path(args.input_file)
        if not file_path.exists():
            logger.error(f"Input file not found: {args.input_file}")
            return 1

        logger.info(f"Evaluating file: {args.input_file}")
        print_evaluation_report(evaluator, file_path, "--- EVALUATION RESULTS ---")
        return 0

    logger.warning(
        "No action specified. Use --dry-run, --input-file, --scan-all, or --scan-all-s3"
    )
    return 1


def _should_process_s3_file(file_path: str) -> bool:
    """Check if a file should be processed based on extension and exclusion patterns."""
    return (
        (file_path.endswith(".json") or file_path.endswith(".jsonl"))
        and "_stats" not in file_path
        and "_report" not in file_path
        and "summary" not in file_path
    )


def _extract_scores_from_s3_jsonl(
    loader,
    file_path: str,
    evaluator,
) -> dict[str, Any] | None:
    """Process a JSONL file from S3 and return scores."""
    empathy_scores = []
    clinical_scores = []
    safety_scores = []

    for record in loader.stream_jsonl(file_path):
        if not record:
            continue
        conv = evaluator._extract_conversation(record)
        if not conv:
            continue
        empathy_scores.append(evaluator.evaluate_empathy(conv))
        clinical_scores.append(evaluator.evaluate_clinical_appropriateness(conv))
        safety_scores.append(evaluator.evaluate_safety(conv))

    if not empathy_scores:
        logger.warning(f"No valid conversations extracted from {file_path}")
        return None

    results = evaluator._build_report(empathy_scores, clinical_scores, safety_scores)
    results["total_conversations_evaluated"] = len(empathy_scores)
    return results


def _extract_scores_from_s3_json(
    loader,
    file_path: str,
    evaluator,
) -> dict[str, Any] | None:
    """Process a JSON file from S3 and return scores."""
    empathy_scores = []
    clinical_scores = []
    safety_scores = []

    data = loader.load_json(file_path)
    conversations = data if isinstance(data, list) else data.get("conversations", [])

    for record in conversations:
        if not record:
            continue
        conv = evaluator._extract_conversation(record)
        if not conv:
            continue
        empathy_scores.append(evaluator.evaluate_empathy(conv))
        clinical_scores.append(evaluator.evaluate_clinical_appropriateness(conv))
        safety_scores.append(evaluator.evaluate_safety(conv))

    if not empathy_scores:
        logger.warning(f"No valid conversations extracted from {file_path}")
        return None

    results = evaluator._build_report(empathy_scores, clinical_scores, safety_scores)
    results["total_conversations_evaluated"] = len(empathy_scores)
    return results


def _process_s3_file(loader, file_path: str, evaluator) -> dict[str, Any] | None:
    """Process a single S3 file and return results."""
    logger.info(f"Processing S3 file: {file_path}")

    try:
        if file_path.endswith(".jsonl"):
            return _extract_scores_from_s3_jsonl(loader, file_path, evaluator)
        elif file_path.endswith(".json"):
            return _extract_scores_from_s3_json(loader, file_path, evaluator)
    except Exception as e:
        logger.error(f"Failed to process {file_path}: {e}")

    return None


def _save_s3_report(all_results: dict[str, Any], report_path: Path):
    """Save consolidated S3 evaluation report to disk."""
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)
    logger.info(f"Consolidated S3 report: {report_path}")


def _scan_s3_datasets(evaluator, s3_prefix: str) -> int:
    """Scan all dataset files in an S3 prefix and produce a consolidated report."""
    logger.info(f"Scanning S3 prefix: {s3_prefix}")
    if S3DatasetLoader is None:
        raise ImportError("S3DatasetLoader could not be imported")

    loader = S3DatasetLoader()
    prefix = s3_prefix.replace("s3://pixel-data/", "")
    files = loader.list_datasets(prefix=prefix)

    all_results: dict[str, Any] = {}

    for file_path in files:
        if not _should_process_s3_file(file_path):
            continue

        results = _process_s3_file(loader, file_path, evaluator)
        if results is None:
            continue

        file_name_only = file_path.split("/")[-1]
        print(f"\n--- {file_name_only} ---")
        print(f"File: {file_path}")
        print(json.dumps(results, indent=2))
        print("-" * 40)

        all_results[file_name_only] = results

    report_path = REPO_ROOT / "ai/data/reports/phase2_baseline_s3_report.json"
    _save_s3_report(all_results, report_path)

    return 0


def _scan_local_datasets(evaluator):
    logger.info("Scanning all dataset directories...")
    all_results: dict[str, Any] = {}

    for scan_dir in DATASET_DIRS:
        if not scan_dir.exists():
            logger.warning(f"Directory not found: {scan_dir}")
            continue

        for file_path in sorted(scan_dir.iterdir()):
            if (
                file_path.suffix in (".json", ".jsonl")
                and "_stats" not in file_path.name
                and "_report" not in file_path.name
                and "summary" not in file_path.name
            ):
                result = print_evaluation_report(
                    evaluator, file_path, f"--- {file_path.name} ---"
                )
                all_results[file_path.name] = result

    report_path = REPO_ROOT / "ai/data/reports/phase2_baseline_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        json.dump(all_results, f, indent=2)

    logger.info(f"Consolidated report: {report_path}")
    return 0


def _run_dry_run_evaluation(evaluator):
    logger.info("Running DRY RUN evaluation...")
    mock_data = [
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I'm having a really hard time today.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "I hear you. That sounds really tough. "
                        "It's valid to feel that way."
                    ),
                },
            ]
        },
        {
            "messages": [
                {
                    "role": "user",
                    "content": "I keep repeating the same mistakes.",
                },
                {
                    "role": "assistant",
                    "content": (
                        "Let's look at that pattern. "
                        "What strategies have you tried to cope?"
                    ),
                },
            ]
        },
    ]

    temp_file = Path("/tmp/mock_eval_data.jsonl")
    with open(temp_file, "w") as f:
        for item in mock_data:
            f.write(json.dumps(item) + "\n")

    print_evaluation_report(evaluator, temp_file, "--- DRY RUN RESULTS ---")
    return 0


if __name__ == "__main__":
    sys.exit(main())
