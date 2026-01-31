"""
Utility functions for Pixel Voice API and MCP server.
"""

import asyncio
import json
import logging
import os
import subprocess
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .config import config
from .models import JobInfo, JobStatus, LogEntry, PipelineStage, StageResult


class PipelineExecutor:
    """Handles execution of pipeline stages and jobs."""

    def __init__(self):
        self.running_jobs: Dict[str, JobInfo] = {}
        self.completed_jobs: Dict[str, JobInfo] = {}
        self.logger = logging.getLogger(__name__)

    async def execute_stage(
        self,
        stage: PipelineStage,
        input_path: Optional[str] = None,
        output_path: Optional[str] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> StageResult:
        """Execute a single pipeline stage."""
        start_time = datetime.now()
        stage_config = config.get_stage_by_name(stage.value.replace("_", " ").title())

        if not stage_config:
            raise ValueError(f"Unknown pipeline stage: {stage}")

        try:
            # Prepare command
            cmd = ["python3", stage_config.script_path]

            # Add input/output paths if provided
            if input_path:
                cmd.extend(["--input", input_path])
            if output_path:
                cmd.extend(["--output", output_path])

            # Execute stage
            process = await asyncio.create_subprocess_exec(
                *cmd,
                cwd=config.base_dir,
                stdout=asyncio.subprocess.PIPE,
                stderr=asyncio.subprocess.PIPE,
            )

            stdout, stderr = await asyncio.wait_for(
                process.communicate(), timeout=stage_config.timeout_seconds
            )

            execution_time = (datetime.now() - start_time).total_seconds()

            if process.returncode == 0:
                return StageResult(
                    stage=stage,
                    status=JobStatus.COMPLETED,
                    input_path=input_path,
                    output_path=output_path,
                    execution_time=execution_time,
                )
            else:
                error_msg = stderr.decode() if stderr else "Unknown error"
                return StageResult(
                    stage=stage,
                    status=JobStatus.FAILED,
                    input_path=input_path,
                    output_path=output_path,
                    execution_time=execution_time,
                    error_message=error_msg,
                )

        except asyncio.TimeoutError:
            return StageResult(
                stage=stage,
                status=JobStatus.FAILED,
                input_path=input_path,
                output_path=output_path,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=f"Stage execution timed out after {stage_config.timeout_seconds} seconds",
            )
        except Exception as e:
            return StageResult(
                stage=stage,
                status=JobStatus.FAILED,
                input_path=input_path,
                output_path=output_path,
                execution_time=(datetime.now() - start_time).total_seconds(),
                error_message=str(e),
            )

    async def execute_pipeline_job(
        self,
        job_name: str,
        stages: list[PipelineStage],
        input_data: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ) -> str:
        """Execute a complete pipeline job with multiple stages."""
        job_id = str(uuid.uuid4())

        job_info = JobInfo(
            job_id=job_id,
            job_name=job_name,
            status=JobStatus.PENDING,
            stages=stages,
            created_at=datetime.now(),
        )

        self.running_jobs[job_id] = job_info

        # Start job execution in background
        asyncio.create_task(self._execute_job_stages(job_id, stages, input_data, config_overrides))

        return job_id

    async def _execute_job_stages(
        self,
        job_id: str,
        stages: list[PipelineStage],
        input_data: Optional[Dict[str, Any]] = None,
        config_overrides: Optional[Dict[str, Any]] = None,
    ):
        """Internal method to execute job stages sequentially."""
        job_info = self.running_jobs[job_id]
        job_info.status = JobStatus.RUNNING
        job_info.started_at = datetime.now()

        try:
            for i, stage in enumerate(stages):
                job_info.current_stage = stage
                job_info.progress = i / len(stages)

                # Execute stage
                result = await self.execute_stage(stage, config_overrides=config_overrides)

                if result.status == JobStatus.FAILED:
                    job_info.status = JobStatus.FAILED
                    job_info.error_message = result.error_message
                    break

                # Store output path
                if result.output_path:
                    job_info.output_paths[stage.value] = result.output_path

            else:
                # All stages completed successfully
                job_info.status = JobStatus.COMPLETED
                job_info.progress = 1.0

        except Exception as e:
            job_info.status = JobStatus.FAILED
            job_info.error_message = str(e)

        finally:
            job_info.completed_at = datetime.now()
            job_info.current_stage = None

            # Move to completed jobs
            self.completed_jobs[job_id] = self.running_jobs.pop(job_id)

    def get_job_info(self, job_id: str) -> Optional[JobInfo]:
        """Get information about a specific job."""
        if job_id in self.running_jobs:
            return self.running_jobs[job_id]
        elif job_id in self.completed_jobs:
            return self.completed_jobs[job_id]
        return None

    def list_jobs(self, status_filter: Optional[JobStatus] = None) -> List[JobInfo]:
        """List all jobs, optionally filtered by status."""
        all_jobs = list(self.running_jobs.values()) + list(self.completed_jobs.values())

        if status_filter:
            return [job for job in all_jobs if job.status == status_filter]

        return all_jobs

    def cancel_job(self, job_id: str) -> bool:
        """Cancel a running job."""
        if job_id in self.running_jobs:
            job_info = self.running_jobs[job_id]
            job_info.status = JobStatus.CANCELLED
            job_info.completed_at = datetime.now()
            self.completed_jobs[job_id] = self.running_jobs.pop(job_id)
            return True
        return False


class DataManager:
    """Handles data operations and file management."""

    def __init__(self):
        self.logger = logging.getLogger(__name__)

    def get_latest_file(self, directory: str, pattern: str = "*.json") -> Optional[str]:
        """Get the most recently modified file in a directory."""
        try:
            dir_path = Path(config.base_dir) / directory
            if not dir_path.exists():
                return None

            files = list(dir_path.glob(pattern))
            if not files:
                return None

            latest_file = max(files, key=lambda f: f.stat().st_mtime)
            return str(latest_file)

        except Exception as e:
            self.logger.error(f"Error getting latest file from {directory}: {e}")
            return None

    def load_json_data(self, file_path: str) -> Optional[Dict[str, Any]]:
        """Load JSON data from a file."""
        try:
            with open(file_path, "r") as f:
                return json.load(f)
        except Exception as e:
            self.logger.error(f"Error loading JSON from {file_path}: {e}")
            return None

    def save_json_data(self, data: Any, file_path: str) -> bool:
        """Save data as JSON to a file."""
        try:
            os.makedirs(os.path.dirname(file_path), exist_ok=True)
            with open(file_path, "w") as f:
                json.dump(data, f, indent=2, default=str)
            return True
        except Exception as e:
            self.logger.error(f"Error saving JSON to {file_path}: {e}")
            return False

    def list_files(self, directory: str, pattern: str = "*") -> List[str]:
        """List files in a directory matching a pattern."""
        try:
            dir_path = Path(config.base_dir) / directory
            if not dir_path.exists():
                return []

            files = list(dir_path.glob(pattern))
            return [str(f) for f in files if f.is_file()]

        except Exception as e:
            self.logger.error(f"Error listing files in {directory}: {e}")
            return []


# Global instances
pipeline_executor = PipelineExecutor()
data_manager = DataManager()
