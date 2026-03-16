#!/usr/bin/env python3
"""
Orchestration script for Phase 1: Data Acquisition Infrastructure.
Executes Academic Sourcing and Research Instrument Collection.
Outputs validated data to ai/training_ready/datasets/.
"""

import logging
import sys
from pathlib import Path

# Add project root to path to allow imports
project_root = Path(__file__).resolve().parents[3]
sys.path.append(str(project_root))

from ai.sourcing.academic import AcademicSourcingEngine  # noqa: E402
from ai.pipelines.orchestrator.sourcing.crisis_expansion import (  # noqa: E402
    CrisisScenarioExpander,
)
from ai.pipelines.orchestrator.sourcing.expert_resources import (  # noqa: E402
    ExpertResourceAggregator,
)
from ai.pipelines.orchestrator.sourcing.instruments_collector import (  # noqa: E402
    ResearchInstrumentCollector,
)
from ai.pipelines.orchestrator.sourcing.therapeutic_conversations import (  # noqa: E402
    TherapeuticConversationAcquisition,
)

# Configure Logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)],
)
logger = logging.getLogger("Phase1_Orchestrator")


def main():
    logger.info("Initializing Phase 1 Data Acquisition...")

    # Define Output Base Path (Training Ready Hub)
    # Using absolute path for clarity, but relative to this script's execution
    # context is also fine
    training_ready_datasets = project_root / "ai" / "training_ready" / "datasets"

    # --- 1. Academic Sourcing ---
    try:
        logger.info(">>> Launching Academic Sourcing Engine")
        academic_engine = AcademicSourcingEngine(
            output_base_path=str(training_ready_datasets)
        )
        academic_output = academic_engine.run_sourcing_pipeline()
        logger.info(f"Academic Sourcing Result: {academic_output}")
    except Exception as e:
        logger.error(f"Academic Sourcing Failed: {e}")
        # Continue to next task instead of hard exit, to allow others to run

    # --- 2. Research Instruments Collection ---
    try:
        logger.info(">>> Launching Research Instrument Collector")
        instruments_engine = ResearchInstrumentCollector(
            output_base_path=str(training_ready_datasets)
        )
        instruments_output = instruments_engine.run_collection_pipeline()
        logger.info(f"Instruments Collection Result: {instruments_output}")
    except Exception as e:
        logger.error(f"Instruments Collection Failed: {e}")

    # --- 3. Therapeutic Conversations ---
    try:
        logger.info(">>> Launching Therapeutic Conversation Acquisition")
        conversations_engine = TherapeuticConversationAcquisition(
            output_base_path=str(training_ready_datasets)
        )
        conversations_output = conversations_engine.run_acquisition_pipeline()
        logger.info(f"Therapeutic Conversations Result: {conversations_output}")
    except Exception as e:
        logger.error(f"Therapeutic Conversations Failed: {e}")

    # --- 4. Expert Resource Aggregator ---
    try:
        logger.info(">>> Launching Expert Resource Aggregator")
        expert_engine = ExpertResourceAggregator(
            output_base_path=str(training_ready_datasets)
        )
        expert_output = expert_engine.run_aggregation_pipeline()
        logger.info(f"Expert Resources Result: {expert_output}")
    except Exception as e:
        logger.error(f"Expert Resources Failed: {e}")

    # --- 5. Crisis Scenario Expander ---
    try:
        logger.info(">>> Launching Crisis Scenario Expander")
        crisis_engine = CrisisScenarioExpander(
            output_base_path=str(training_ready_datasets)
        )
        crisis_output = crisis_engine.run_expansion_pipeline()
        logger.info(f"Crisis Expansion Result: {crisis_output}")
    except Exception as e:
        logger.error(f"Crisis Expansion Failed: {e}")

    logger.info("Phase 1 Data Acquisition Completed.")


if __name__ == "__main__":
    main()
