"""
CommandHandler service for API endpoints.

This module provides a service layer that wraps CommandHandler functionality
for use by API endpoints.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional

from ai.journal_dataset_research.cli.commands import CommandHandler
from ai.journal_dataset_research.cli.config import load_config
from ai.journal_dataset_research.models.dataset_models import (
    AcquiredDataset,
    DatasetEvaluation,
    DatasetSource,
    IntegrationPlan,
    ResearchSession,
)
from ai.journal_dataset_research.orchestrator.research_orchestrator import (
    ResearchOrchestrator,
)

logger = logging.getLogger(__name__)


class CommandHandlerService:
    """Service layer for CommandHandler operations."""

    def __init__(self, config: Optional[Dict[str, Any]] = None, dry_run: bool = False):
        """Initialize the service with configuration."""
        self.config = config or load_config()
        self.dry_run = dry_run
        self._command_handler: Optional[CommandHandler] = None

    @property
    def command_handler(self) -> CommandHandler:
        """Get or create CommandHandler instance."""
        if self._command_handler is None:
            self._command_handler = CommandHandler(self.config, self.dry_run)
        return self._command_handler

    @property
    def orchestrator(self) -> ResearchOrchestrator:
        """Get orchestrator instance from command handler."""
        return self.command_handler._get_orchestrator()

    # Session management methods
    def create_session(
        self,
        target_sources: List[str],
        search_keywords: Dict[str, List[str]],
        weekly_targets: Optional[Dict[str, int]] = None,
        session_id: Optional[str] = None,
    ) -> ResearchSession:
        """Create a new research session."""
        orchestrator = self.orchestrator
        session = orchestrator.start_research_session(
            target_sources=target_sources,
            search_keywords=search_keywords,
            weekly_targets=weekly_targets,
            session_id=session_id,
        )
        orchestrator.save_session_state(session.session_id)
        return session

    def list_sessions(self) -> List[ResearchSession]:
        """List all research sessions."""
        orchestrator = self.orchestrator
        sessions_dir = Path(orchestrator._session_storage_path)
        sessions = []

        if sessions_dir.exists():
            for session_file in sessions_dir.glob("*.json"):
                try:
                    session_id = session_file.stem
                    orchestrator.load_session_state(session_id)
                    if session_id in orchestrator.sessions:
                        sessions.append(orchestrator.sessions[session_id])
                except Exception as e:
                    logger.warning(f"Failed to load session {session_file.stem}: {e}")

        return sessions

    def get_session(self, session_id: str) -> ResearchSession:
        """Get session by ID."""
        orchestrator = self.orchestrator
        try:
            orchestrator.load_session_state(session_id)
            if session_id not in orchestrator.sessions:
                raise ValueError(f"Session {session_id} not found")
            return orchestrator.sessions[session_id]
        except FileNotFoundError:
            raise ValueError(f"Session {session_id} not found")

    def update_session(
        self,
        session_id: str,
        target_sources: Optional[List[str]] = None,
        search_keywords: Optional[Dict[str, List[str]]] = None,
        weekly_targets: Optional[Dict[str, int]] = None,
        current_phase: Optional[str] = None,
    ) -> ResearchSession:
        """Update session configuration."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        session = orchestrator.sessions[session_id]

        if target_sources is not None:
            session.target_sources = target_sources
        if search_keywords is not None:
            session.search_keywords = search_keywords
        if weekly_targets is not None:
            session.weekly_targets = weekly_targets
        if current_phase is not None:
            if current_phase not in ResearchOrchestrator.PHASE_ORDER:
                raise ValueError(
                    f"Invalid phase: {current_phase}. Must be one of {ResearchOrchestrator.PHASE_ORDER}"
                )
            session.current_phase = current_phase

        orchestrator.save_session_state(session_id)
        return session

    def delete_session(self, session_id: str) -> None:
        """Delete a research session."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)

        # Remove from memory
        if session_id in orchestrator.sessions:
            del orchestrator.sessions[session_id]
        if session_id in orchestrator.session_states:
            del orchestrator.session_states[session_id]
        if session_id in orchestrator.progress_states:
            del orchestrator.progress_states[session_id]

        # Delete session file
        session_file = Path(orchestrator._session_storage_path) / f"{session_id}.json"
        if session_file.exists():
            session_file.unlink()

    def get_session_state(self, session_id: str) -> Dict[str, Any]:
        """Get session state including sources, evaluations, etc."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)
        session = orchestrator.sessions[session_id]

        return {
            "session": session,
            "sources": state.sources,
            "evaluations": state.evaluations,
            "access_requests": state.access_requests,
            "acquired_datasets": state.acquired_datasets,
            "integration_plans": state.integration_plans,
        }

    def _session_to_dict(self, session: ResearchSession) -> Dict[str, Any]:
        """Convert ResearchSession to dictionary."""
        return {
            "session_id": session.session_id,
            "start_date": session.start_date.isoformat(),
            "target_sources": session.target_sources,
            "search_keywords": session.search_keywords,
            "weekly_targets": session.weekly_targets,
            "current_phase": session.current_phase,
            "progress_metrics": session.progress_metrics,
        }

    def _source_to_dict(self, source: DatasetSource) -> Dict[str, Any]:
        """Convert DatasetSource to dictionary."""
        return {
            "source_id": source.source_id,
            "title": source.title,
            "authors": source.authors,
            "publication_date": source.publication_date,
            "source_type": source.source_type,
            "url": source.url,
            "doi": source.doi,
            "abstract": source.abstract,
            "keywords": source.keywords,
            "open_access": source.open_access,
            "data_availability": source.data_availability,
            "discovery_date": source.discovery_date,
            "discovery_method": source.discovery_method,
        }

    def _evaluation_to_dict(self, evaluation: DatasetEvaluation) -> Dict[str, Any]:
        """Convert DatasetEvaluation to dictionary."""
        return {
            "evaluation_id": f"eval_{evaluation.source_id}",
            "source_id": evaluation.source_id,
            "therapeutic_relevance": evaluation.therapeutic_relevance,
            "data_structure_quality": evaluation.data_structure_quality,
            "training_integration": evaluation.training_integration,
            "ethical_accessibility": evaluation.ethical_accessibility,
            "overall_score": evaluation.overall_score,
            "priority_tier": evaluation.priority_tier,
            "evaluation_date": evaluation.evaluation_date,
            "evaluator": evaluation.evaluator,
        }

    def _acquisition_to_dict(self, acquisition: AcquiredDataset) -> Dict[str, Any]:
        """Convert AcquiredDataset to dictionary."""
        return {
            "acquisition_id": f"acq_{acquisition.source_id}",
            "source_id": acquisition.source_id,
            "status": "completed" if acquisition.storage_path else "pending",
            "download_progress": 100.0 if acquisition.storage_path else 0.0,
            "file_path": acquisition.storage_path,
            "file_size": acquisition.file_size_mb,
            "acquired_date": acquisition.acquisition_date,
        }

    def _integration_plan_to_dict(self, plan: IntegrationPlan) -> Dict[str, Any]:
        """Convert IntegrationPlan to dictionary."""
        return {
            "plan_id": f"plan_{plan.source_id}",
            "source_id": plan.source_id,
            "complexity": plan.complexity,
            "target_format": plan.dataset_format,
            "required_transformations": plan.required_transformations,
            "estimated_effort_hours": plan.estimated_effort_hours,
            "schema_mapping": plan.schema_mapping,
            "created_date": plan.created_date,
        }

    # Discovery methods
    def initiate_discovery(
        self,
        session_id: str,
        keywords: List[str],
        sources: List[str],
    ) -> Dict[str, Any]:
        """Initiate source discovery for a session."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        session = orchestrator.sessions[session_id]

        # Create search keywords dict
        search_keywords = {
            "therapeutic": keywords,
            "dataset": keywords,
        }

        # Update session if needed
        if session.target_sources != sources or session.search_keywords != search_keywords:
            session.target_sources = sources
            session.search_keywords = search_keywords
            orchestrator.save_session_state(session_id)

        # Run discovery
        state = orchestrator.get_session_state(session_id)
        sources_list = []

        if orchestrator.discovery_service:
            try:
                sources_list = orchestrator.discovery_service.discover_sources(session)
                state.sources = sources_list
                orchestrator.update_progress(
                    session_id, {"sources_identified": len(state.sources)}
                )
            except Exception as e:
                logger.exception(f"Discovery error for session {session_id}: {e}")
                raise

        orchestrator.save_session_state(session_id)

        # Broadcast progress update via WebSocket (fire-and-forget)
        self._broadcast_progress_update(session_id)

        return {
            "session_id": session_id,
            "total_sources": len(sources_list),
        }

    def get_sources(self, session_id: str) -> List[DatasetSource]:
        """Get discovered sources for a session."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)
        return state.sources

    def get_source(self, session_id: str, source_id: str) -> DatasetSource:
        """Get source details by ID."""
        sources = self.get_sources(session_id)
        for source in sources:
            if source.source_id == source_id:
                return source
        raise ValueError(f"Source {source_id} not found in session {session_id}")

    # Evaluation methods
    def initiate_evaluation(
        self,
        session_id: str,
        source_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Initiate evaluation for sources."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)

        # Filter sources if source_ids provided
        sources_to_evaluate = (
            [s for s in state.sources if s.source_id in source_ids]
            if source_ids
            else state.sources
        )

        if not sources_to_evaluate:
            return {
                "evaluations": [],
                "session_id": session_id,
            }

        evaluations: List[DatasetEvaluation] = []

        if orchestrator.evaluation_engine:
            for source in sources_to_evaluate:
                try:
                    evaluation = orchestrator.evaluation_engine.evaluate_dataset(
                        source, evaluator="system"
                    )
                    evaluations.append(evaluation)
                except Exception as e:
                    logger.exception(f"Error evaluating {source.source_id}: {e}")
                    raise

        state.evaluations.extend(evaluations)
        orchestrator.update_progress(
            session_id, {"datasets_evaluated": len(state.evaluations)}
        )
        orchestrator.save_session_state(session_id)

        # Broadcast progress update via WebSocket (fire-and-forget)
        self._broadcast_progress_update(session_id)

        return {
            "evaluations": [self._evaluation_to_dict(e) for e in evaluations],
            "session_id": session_id,
        }

    def get_evaluations(self, session_id: str) -> List[DatasetEvaluation]:
        """Get evaluations for a session."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)
        return state.evaluations

    def get_evaluation(self, session_id: str, evaluation_id: str) -> DatasetEvaluation:
        """Get evaluation details by ID."""
        evaluations = self.get_evaluations(session_id)
        # evaluation_id format is "eval_{source_id}"
        source_id = evaluation_id.replace("eval_", "")
        for evaluation in evaluations:
            if evaluation.source_id == source_id:
                return evaluation
        raise ValueError(
            f"Evaluation {evaluation_id} not found in session {session_id}"
        )

    def update_evaluation(
        self,
        session_id: str,
        evaluation_id: str,
        therapeutic_relevance: Optional[int] = None,
        data_structure_quality: Optional[int] = None,
        training_integration: Optional[int] = None,
        ethical_accessibility: Optional[int] = None,
        priority_tier: Optional[str] = None,
    ) -> DatasetEvaluation:
        """Update evaluation scores."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        evaluation = self.get_evaluation(session_id, evaluation_id)

        if therapeutic_relevance is not None:
            evaluation.therapeutic_relevance = therapeutic_relevance
        if data_structure_quality is not None:
            evaluation.data_structure_quality = data_structure_quality
        if training_integration is not None:
            evaluation.training_integration = training_integration
        if ethical_accessibility is not None:
            evaluation.ethical_accessibility = ethical_accessibility
        if priority_tier is not None:
            evaluation.priority_tier = priority_tier

        # Recalculate overall score
        evaluation.overall_score = (
            evaluation.therapeutic_relevance * 0.35
            + evaluation.data_structure_quality * 0.25
            + evaluation.training_integration * 0.20
            + evaluation.ethical_accessibility * 0.20
        )

        orchestrator.save_session_state(session_id)

        # Broadcast progress update via WebSocket (fire-and-forget)
        self._broadcast_progress_update(session_id)

        return evaluation

    # Acquisition methods
    def initiate_acquisition(
        self,
        session_id: str,
        source_ids: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """Initiate acquisition for sources."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)

        # Filter sources if source_ids provided
        sources_to_acquire = (
            [s for s in state.sources if s.source_id in source_ids]
            if source_ids
            else state.sources
        )

        if not sources_to_acquire:
            return {
                "acquired": [],
                "session_id": session_id,
            }

        acquired_count = 0

        if orchestrator.acquisition_manager:
            for source in sources_to_acquire:
                try:
                    # Submit access request
                    access_request = orchestrator.acquisition_manager.submit_access_request(
                        source
                    )
                    state.access_requests.append(access_request)

                    # Download dataset
                    acquired_dataset = orchestrator.acquisition_manager.download_dataset(
                        source, access_request
                    )
                    state.acquired_datasets.append(acquired_dataset)
                    acquired_count += 1
                except Exception as e:
                    logger.exception(f"Error acquiring {source.source_id}: {e}")
                    raise
        else:
            raise ValueError("No acquisition manager configured")

        orchestrator.update_progress(
            session_id, {"datasets_acquired": len(state.acquired_datasets)}
        )
        orchestrator.save_session_state(session_id)

        # Broadcast progress update via WebSocket (fire-and-forget)
        self._broadcast_progress_update(session_id)

        return {
            "acquired": [d.source_id for d in state.acquired_datasets],
            "session_id": session_id,
        }

    def get_acquisitions(self, session_id: str) -> List[AcquiredDataset]:
        """Get acquisitions for a session."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)
        return state.acquired_datasets

    def get_acquisition(self, session_id: str, acquisition_id: str) -> AcquiredDataset:
        """Get acquisition details by ID."""
        acquisitions = self.get_acquisitions(session_id)
        # acquisition_id format is "acq_{source_id}"
        source_id = acquisition_id.replace("acq_", "")
        for acquisition in acquisitions:
            if acquisition.source_id == source_id:
                return acquisition
        raise ValueError(
            f"Acquisition {acquisition_id} not found in session {session_id}"
        )

    def update_acquisition(
        self,
        session_id: str,
        acquisition_id: str,
        status: str,
    ) -> AcquiredDataset:
        """Update acquisition status."""
        # For now, we just return the acquisition
        # Status updates would require modifying the AcquiredDataset model
        acquisition = self.get_acquisition(session_id, acquisition_id)
        orchestrator = self.orchestrator
        orchestrator.save_session_state(session_id)
        return acquisition

    # Integration methods
    def initiate_integration(
        self,
        session_id: str,
        source_ids: Optional[List[str]] = None,
        target_format: str = "chatml",
    ) -> Dict[str, Any]:
        """Initiate integration planning."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)

        # Filter datasets if source_ids provided
        datasets_to_integrate = (
            [d for d in state.acquired_datasets if d.source_id in source_ids]
            if source_ids
            else state.acquired_datasets
        )

        if not datasets_to_integrate:
            return {
                "plans": [],
                "session_id": session_id,
            }

        plans_count = 0

        if orchestrator.integration_engine:
            for dataset in datasets_to_integrate:
                try:
                    plan = orchestrator.integration_engine.create_integration_plan(
                        dataset, target_format
                    )
                    state.integration_plans.append(plan)
                    plans_count += 1
                except Exception as e:
                    logger.exception(f"Error creating plan for {dataset.source_id}: {e}")
                    raise
        else:
            raise ValueError("No integration engine configured")

        orchestrator.update_progress(
            session_id, {"integration_plans_created": len(state.integration_plans)}
        )
        orchestrator.save_session_state(session_id)

        # Broadcast progress update via WebSocket (fire-and-forget)
        self._broadcast_progress_update(session_id)

        return {
            "plans": [p.source_id for p in state.integration_plans],
            "session_id": session_id,
        }

    def get_integration_plans(self, session_id: str) -> List[IntegrationPlan]:
        """Get integration plans for a session."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        state = orchestrator.get_session_state(session_id)
        return state.integration_plans

    def get_integration_plan(
        self, session_id: str, plan_id: str
    ) -> IntegrationPlan:
        """Get integration plan details by ID."""
        plans = self.get_integration_plans(session_id)
        # plan_id format is "plan_{source_id}"
        source_id = plan_id.replace("plan_", "")
        for plan in plans:
            if plan.source_id == source_id:
                return plan
        raise ValueError(f"Integration plan {plan_id} not found in session {session_id}")

    # Progress methods
    def get_progress(self, session_id: str) -> Dict[str, Any]:
        """Get progress metrics for a session."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        session = orchestrator.sessions[session_id]
        state = orchestrator.get_session_state(session_id)
        progress = orchestrator.get_progress(session_id)

        # Calculate progress percentage
        total_metrics = sum(session.progress_metrics.values())
        total_targets = sum(session.weekly_targets.values()) if session.weekly_targets else 1
        progress_percentage = (
            (total_metrics / total_targets * 100) if total_targets > 0 else 0.0
        )

        return {
            "session_id": session_id,
            "current_phase": session.current_phase,
            "progress_metrics": {
                "sources_identified": progress.sources_identified,
                "datasets_evaluated": progress.datasets_evaluated,
                "datasets_acquired": progress.datasets_acquired,
                "integration_plans_created": progress.integration_plans_created,
            },
            "weekly_targets": session.weekly_targets,
            "progress_percentage": min(progress_percentage, 100.0),
        }

    def get_progress_metrics(self, session_id: str) -> Dict[str, Any]:
        """Get detailed progress metrics for a session."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        progress = orchestrator.get_progress(session_id)

        return {
            "session_id": session_id,
            "sources_identified": progress.sources_identified,
            "datasets_evaluated": progress.datasets_evaluated,
            "datasets_acquired": progress.datasets_acquired,
            "integration_plans_created": progress.integration_plans_created,
            "last_updated": progress.last_updated.isoformat() if progress.last_updated else None,
        }

    # Report methods
    def generate_report(
        self,
        session_id: str,
        report_type: str = "session_report",
        format: str = "json",
        date_range: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """Generate a report for a session."""
        orchestrator = self.orchestrator
        orchestrator.load_session_state(session_id)
        session = orchestrator.sessions[session_id]
        state = orchestrator.get_session_state(session_id)

        # Generate report data
        report_data = {
            "session_id": session_id,
            "start_date": session.start_date.isoformat(),
            "current_phase": session.current_phase,
            "progress_metrics": session.progress_metrics,
            "weekly_targets": session.weekly_targets,
            "sources": [self._source_to_dict(s) for s in state.sources],
            "evaluations": [
                self._evaluation_to_dict(e) for e in state.evaluations
            ],
            "acquired_datasets": [
                self._acquisition_to_dict(d) for d in state.acquired_datasets
            ],
            "integration_plans": [
                self._integration_plan_to_dict(p) for p in state.integration_plans
            ],
        }

        # Generate report ID
        report_id = f"report_{session_id}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

        return {
            "report_id": report_id,
            "session_id": session_id,
            "report_type": report_type,
            "format": format,
            "generated_date": datetime.now(),
            "content": report_data,
        }

    def list_reports(self, session_id: str) -> List[Dict[str, Any]]:
        """List reports for a session."""
        # For now, return empty list
        # In production, this would query a report storage system
        return []

    def get_report(self, session_id: str, report_id: str) -> Dict[str, Any]:
        """Get report details by ID."""
        # For now, generate report on the fly
        # In production, this would retrieve from report storage
        return self.generate_report(session_id)

    def _broadcast_progress_update(self, session_id: str) -> None:
        """Broadcast progress update to WebSocket connections (fire-and-forget)."""
        try:
            # Try to get the running event loop (FastAPI context)
            try:
                loop = asyncio.get_running_loop()
                # Schedule broadcast in background if loop is running
                loop.create_task(self._async_broadcast_progress_update(session_id))
            except RuntimeError:
                # No running loop, try to get event loop (may not be running)
                try:
                    loop = asyncio.get_event_loop()
                    if loop.is_running():
                        loop.create_task(self._async_broadcast_progress_update(session_id))
                    else:
                        # No loop running, skip broadcast (can't create task without running loop)
                        logger.debug(f"No running event loop for WebSocket broadcast: {session_id}")
                except RuntimeError:
                    # No event loop at all, skip broadcast
                    logger.debug(f"No event loop for WebSocket broadcast: {session_id}")
        except Exception as e:
            # Don't fail the request if WebSocket broadcast fails
            logger.warning(f"Failed to schedule progress update broadcast: {e}")

    async def _async_broadcast_progress_update(self, session_id: str) -> None:
        """Async helper to broadcast progress update to WebSocket connections."""
        try:
            from ai.journal_dataset_research.api.websocket.manager import manager

            # Get current progress
            progress_data = self.get_progress(session_id)
            metrics_data = self.get_progress_metrics(session_id)

            # Broadcast to all connections for this session
            await manager.broadcast_to_session(
                session_id,
                {
                    "type": "progress_update",
                    "session_id": session_id,
                    "timestamp": datetime.now().isoformat(),
                    "data": {
                        **progress_data,
                        "metrics": metrics_data,
                    },
                },
            )
        except Exception as e:
            # Don't fail the request if WebSocket broadcast fails
            logger.warning(f"Failed to broadcast progress update: {e}")

