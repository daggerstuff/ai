#!/usr/bin/env python3
"""
Orchestration script for Phase 3: Therapeutic Modality Integration.
Generates structured therapeutic content (starting with CBT).
Outputs to ai/training_ready/datasets/stage2_reasoning/cbt_content.
"""

import logging
import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from ai.pipelines.orchestrator.therapies.act_integration import ACTIntegration  # noqa: E402
from ai.pipelines.orchestrator.therapies.cbt_integration import CBTIntegration  # noqa: E402
from ai.pipelines.orchestrator.therapies.dbt_integration import DBTIntegration  # noqa: E402
from ai.pipelines.orchestrator.therapies.emdr_integration import EMDRIntegration  # noqa: E402

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase3_Orchestrator")


def main():
    logger.info("Initializing Phase 3 Therapeutic Modality Integration...")

    # --- Task 3.1: CBT Integration ---
    try:
        logger.info(">>> Launching CBT Integration System")
        cbt_engine = CBTIntegration(
            output_base_path=str(project_root / "ai" / "training_ready" / "datasets")
        )
        # Generate a sample batch
        cbt_data = cbt_engine.generate_batch_content(count=20)
        output_file = cbt_engine.export_data(cbt_data, batch_id="001")
        logger.info(f"CBT Content Generation Result: {output_file}")
    except Exception as e:
        logger.error(f"CBT Integration Failed: {e}")

    # --- Task 3.2: DBT Integration ---
    try:
        logger.info(">>> Launching DBT Integration System")
        dbt_engine = DBTIntegration(
            output_base_path=str(project_root / "ai" / "training_ready" / "datasets")
        )
        # Generate a sample batch
        dbt_data = dbt_engine.generate_batch_content(count=20)
        output_file = dbt_engine.export_data(dbt_data, batch_id="001")
        logger.info(f"DBT Content Generation Result: {output_file}")
    except Exception as e:
        logger.error(f"DBT Integration Failed: {e}")

    # --- Task 3.3: EMDR Integration ---
    try:
        logger.info(">>> Launching EMDR Integration System")
        emdr_engine = EMDRIntegration(
            output_base_path=str(project_root / "ai" / "training_ready" / "datasets")
        )
        # Generate a sample batch
        emdr_data = emdr_engine.generate_batch_content(count=20)
        output_file = emdr_engine.export_data(emdr_data, batch_id="001")
        logger.info(f"EMDR Content Generation Result: {output_file}")
    except Exception as e:
        logger.error(f"EMDR Integration Failed: {e}")

    # --- Task 3.4: ACT Integration ---
    try:
        logger.info(">>> Launching ACT Integration System")
        act_engine = ACTIntegration(
            output_base_path=str(project_root / "ai" / "training_ready" / "datasets")
        )
        # Generate a sample batch
        act_data = act_engine.generate_batch_content(count=20)
        output_file = act_engine.export_data(act_data, batch_id="001")
        logger.info(f"ACT Content Generation Result: {output_file}")
    except Exception as e:
        logger.error(f"ACT Integration Failed: {e}")

    # Future: Task 3.5 Somatic, etc.

    logger.info("Phase 3 Modality Integration (Current Tasks) Completed.")


if __name__ == "__main__":
    main()
