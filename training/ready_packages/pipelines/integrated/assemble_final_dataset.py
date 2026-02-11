#!/usr/bin/env python3
"""
Final Dataset Assembly and Stage Balancing

Assembles final training dataset from all processed sources, balancing
across 4 stages (40/25/20/15 target distribution).
"""

import json
import sys
import random
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
import logging

# Add project root to path for imports
project_root = Path(__file__).parent.parent.parent.parent.parent
sys.path.insert(0, str(project_root))

try:
    from ai.pipelines.orchestrator.configs.stages import STAGE1_ID, STAGE2_ID, STAGE3_ID, STAGE4_ID
except ImportError as e:
    logging.error(f"Failed to import pipeline modules: {e}")
    logging.error(f"Project root: {project_root}")
    sys.exit(1)

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

# Target stage distribution (40/25/20/15)
STAGE_TARGETS = {
    STAGE1_ID: 0.40,  # 40%
    STAGE2_ID: 0.25,  # 25%
    STAGE3_ID: 0.20,  # 20%
    STAGE4_ID: 0.15,  # 15%
}


class DatasetAssembler:
    """Assembles final training dataset with stage balancing"""

    def __init__(self, formatting_report_path: Path, target_total: int = 100000):
        self.formatting_report_path = formatting_report_path
        self.formatting_report = self._load_formatting_report()
        self.target_total = target_total
        self.stage_targets = {
            stage: int(target_total * percentage)
            for stage, percentage in STAGE_TARGETS.items()
        }
        self.stats = {
            "total_available": 0,
            "by_stage_available": {stage: 0 for stage in STAGE_TARGETS},
            "by_stage_selected": {stage: 0 for stage in STAGE_TARGETS},
            "sampling_applied": {},
        }

    def _load_formatting_report(self) -> Dict[str, Any]:
        """Load formatting report"""
        if not self.formatting_report_path.exists():
            return {"results": []}
        with open(self.formatting_report_path, "r") as f:
            return json.load(f)

    def load_conversations_from_file(self, file_path: Path) -> List[Dict[str, Any]]:
        """Load conversations from a JSONL file"""
        conversations = []
        try:
            with open(file_path, "r", encoding="utf-8") as f:
                for line in f:
                    if not line.strip():
                        continue
                    try:
                        conv = json.loads(line)
                        conversations.append(conv)
                    except json.JSONDecodeError:
                        continue
        except Exception as e:
            logger.warning(f"Error loading {file_path}: {e}")
        return conversations

    def collect_all_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Collect all conversations by stage"""
        conversations_by_stage = {stage: [] for stage in STAGE_TARGETS}

        for result in self.formatting_report.get("results", []):
            if "error" in result:
                continue

            file_path = Path(result["output_path"])
            if not file_path.exists():
                continue

            stage = result.get("stage", STAGE1_ID)
            conversations = self.load_conversations_from_file(file_path)
            conversations_by_stage[stage].extend(conversations)

            self.stats["by_stage_available"][stage] += len(conversations)
            self.stats["total_available"] += len(conversations)

        return conversations_by_stage

    def sample_to_target(self, conversations: List[Dict[str, Any]], target: int) -> List[Dict[str, Any]]:
        """Sample conversations to meet target count"""
        if len(conversations) <= target:
            return conversations

        # Random sampling
        sampled = random.sample(conversations, target)
        return sampled

    def assemble_final_dataset(self, output_dir: Path) -> Dict[str, Any]:
        """Assemble final dataset with stage balancing"""
        logger.info("📦 Assembling final training dataset...")
        logger.info(f"  Target total: {self.target_total:,}")
        logger.info(f"  Stage targets: {self.stage_targets}")

        # Collect all conversations
        conversations_by_stage = self.collect_all_conversations()

        logger.info(f"\n📊 Available conversations:")
        for stage, count in self.stats["by_stage_available"].items():
            logger.info(f"  {stage}: {count:,}")

        # Sample to meet targets
        final_conversations_by_stage = {}
        for stage, target in self.stage_targets.items():
            available = conversations_by_stage[stage]
            if len(available) < target:
                logger.warning(f"  ⚠️  {stage}: Only {len(available):,} available, target is {target:,}")
                sampled = available  # Use all available
            else:
                sampled = self.sample_to_target(available, target)
                self.stats["sampling_applied"][stage] = len(available) - len(sampled)

            final_conversations_by_stage[stage] = sampled
            self.stats["by_stage_selected"][stage] = len(sampled)

        # Write stage-specific datasets
        stage_outputs = {}
        for stage, conversations in final_conversations_by_stage.items():
            stage_output_path = output_dir / f"{stage}_dataset.jsonl"
            stage_output_path.parent.mkdir(parents=True, exist_ok=True)

            with open(stage_output_path, "w", encoding="utf-8") as f:
                for conv in conversations:
                    f.write(json.dumps(conv, ensure_ascii=False) + "\n")

            stage_outputs[stage] = str(stage_output_path)
            logger.info(f"  ✅ {stage}: {len(conversations):,} conversations -> {stage_output_path}")

        # Write combined dataset
        all_conversations = []
        for stage in [STAGE1_ID, STAGE2_ID, STAGE3_ID, STAGE4_ID]:
            all_conversations.extend(final_conversations_by_stage[stage])

        # Shuffle combined dataset
        random.shuffle(all_conversations)

        combined_output_path = output_dir / "final_training_dataset.jsonl"
        with open(combined_output_path, "w", encoding="utf-8") as f:
            for conv in all_conversations:
                f.write(json.dumps(conv, ensure_ascii=False) + "\n")

        logger.info(f"  ✅ Combined: {len(all_conversations):,} conversations -> {combined_output_path}")

        # Calculate actual distribution
        actual_distribution = {
            stage: len(conversations) / len(all_conversations) * 100
            for stage, conversations in final_conversations_by_stage.items()
        }

        return {
            "timestamp": datetime.now().isoformat(),
            "target_total": self.target_total,
            "actual_total": len(all_conversations),
            "stage_targets": self.stage_targets,
            "stage_outputs": stage_outputs,
            "combined_output": str(combined_output_path),
            "actual_distribution": actual_distribution,
            "target_distribution": {stage: p * 100 for stage, p in STAGE_TARGETS.items()},
            "stats": self.stats,
        }


def main():
    """Main function"""
    base_path = Path.cwd()
    formatting_report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "formatting_report.json"
    output_dir = base_path / "ai" / "training_ready" / "datasets" / "final"

    if not formatting_report_path.exists():
        logger.error(f"Formatting report not found: {formatting_report_path}")
        logger.info("Please run format_for_training.py first")
        return 1

    # Allow target total to be specified
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--target-total", type=int, default=100000, help="Target total conversations")
    args = parser.parse_args()

    logger.info("📦 Starting final dataset assembly...")

    assembler = DatasetAssembler(formatting_report_path, target_total=args.target_total)
    report = assembler.assemble_final_dataset(output_dir)

    # Save report
    report_path = base_path / "ai" / "training_ready" / "scripts" / "output" / "assembly_report.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        json.dump(report, f, indent=2)

    logger.info(f"\n📊 Assembly Summary:")
    logger.info(f"  Target total: {report['target_total']:,}")
    logger.info(f"  Actual total: {report['actual_total']:,}")
    logger.info(f"\n📈 Distribution:")
    for stage in [STAGE1_ID, STAGE2_ID, STAGE3_ID, STAGE4_ID]:
        target_pct = report['target_distribution'][stage]
        actual_pct = report['actual_distribution'][stage]
        logger.info(f"  {stage}: {actual_pct:.1f}% (target: {target_pct:.1f}%)")
    logger.info(f"\n💾 Report saved to: {report_path}")

    return 0


if __name__ == "__main__":
    sys.exit(main())

