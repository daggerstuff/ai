"""
NVIDIA NeMo Data Designer Service

This service provides a high-level interface for generating synthetic datasets
using NVIDIA NeMo Data Designer, specifically tailored for therapeutic and
mental health applications.
"""

import logging
import time
from typing import Any, Optional

try:
    from nemo_microservices.data_designer.essentials import (
        CategorySamplerParams,
        DataDesignerConfigBuilder,
        NeMoDataDesignerClient,
        SamplerColumnConfig,
        SamplerType,
        UniformSamplerParams,
    )
except ImportError as e:
    raise ImportError(
        "nemo-microservices[data-designer] is not installed. "
        "Install it with: uv pip install 'nemo-microservices[data-designer]'"
    ) from e

from ai.core.pipelines.design.config import DataDesignerConfig

logger = logging.getLogger(__name__)


class NeMoDataDesignerService:
    """Service for generating synthetic datasets using NVIDIA NeMo Data Designer."""

    def __init__(self, config: Optional[DataDesignerConfig] = None):
        """
        Initialize the NeMo Data Designer service.

        Args:
            config: Configuration object. If None, loads from environment.
        """
        self.config = config or DataDesignerConfig.from_env()
        self.config.validate()

        self.client = NeMoDataDesignerClient(
            base_url=self.config.base_url,
            default_headers={"Authorization": f"Bearer {self.config.api_key}"},
        )

        # Monkey-patch the client's create method to add column_type to columns
        # This is required for SDK 1.5.0 compatibility with NeMo 25.12 service
        original_create = self.client.create

        def patched_create(config_builder=None, wait_until_done=True, **kwargs):
            # If config_builder is provided, use custom implementation
            if not config_builder:
                # No config_builder, delegate to original
                return original_create(
                    config_builder=config_builder,
                    wait_until_done=wait_until_done,
                    **kwargs,
                )

            import requests

            config = config_builder.build()
            # Patch columns to include column_type
            config_dict = config.model_dump(mode="json")
            for col in config_dict.get("columns", []):
                if "column_type" not in col:
                    col["column_type"] = "sampler"

            # Generate synchronously using the preview API logic since standalone docker
            # lacks a platform job manager
            num_records = kwargs.get("num_records", 5)
            # The preview API might restrict to 10 max, we use 5 to be safe
            all_data = []
            remaining = num_records

            while remaining > 0:
                chunk_size = min(remaining, 5)
                response = requests.post(
                    f"{self.config.base_url}/v1/data-designer/preview",
                    json={
                        "num_records": chunk_size,
                        "config": config_dict,
                    },
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )
                response.raise_for_status()
                try:
                    job_data = response.json()
                except Exception:
                    # Fallback for JSONL responses
                    import json

                    job_data = [
                        json.loads(line)
                        for line in response.text.split("\n")
                        if line.strip()
                    ]

                if "data" in job_data:
                    all_data.extend(job_data["data"])
                elif "preview" in job_data:
                    all_data.extend(job_data["preview"])
                elif isinstance(job_data, list):
                    all_data.extend(job_data)
                else:
                    # Sometimes pydantic responses nest inside other objects
                    all_data.extend(job_data.get("rows", []))

                remaining -= chunk_size

            class JobResult:
                def __init__(self, data):
                    self.data = data
                    self.dataset = data
                    self.id = "synchronous-preview"

                def load_dataset(self):
                    return self.data

                def wait_until_done(self):
                    pass

            return JobResult(all_data)

        self.client.create = patched_create

        logger.info(
            "NeMo Data Designer service initialized"
            f" with base_url: {self.config.base_url}"
        )

    def _build_therapeutic_columns(
        self,
        include_demographics: bool,
        include_symptoms: bool,
        include_treatments: bool,
        include_outcomes: bool,
    ) -> list[SamplerColumnConfig]:
        """Build column configurations for therapeutic datasets."""
        columns = []

        if include_demographics:
            columns.extend(
                [
                    SamplerColumnConfig(
                        name="age",
                        sampler_type="uniform",
                        params=UniformSamplerParams(
                            low=18.0, high=80.0, decimal_places=0
                        ),
                    ),
                    SamplerColumnConfig(
                        name="gender",
                        sampler_type="category",
                        params=CategorySamplerParams(
                            values=[
                                "male",
                                "female",
                                "non-binary",
                                "prefer not to say",
                            ],
                        ),
                    ),
                    SamplerColumnConfig(
                        name="ethnicity",
                        sampler_type="category",
                        params=CategorySamplerParams(
                            values=[
                                "White",
                                "Black or African American",
                                "Hispanic or Latino",
                                "Asian",
                                "Native American",
                                "Pacific Islander",
                                "Other",
                            ],
                        ),
                    ),
                ]
            )

        if include_symptoms:
            columns.extend(
                [
                    SamplerColumnConfig(
                        name="primary_diagnosis",
                        sampler_type="category",
                        params=CategorySamplerParams(
                            values=[
                                "Anxiety Disorders",
                                "Depressive Disorders",
                                "Bipolar Disorders",
                                "PTSD",
                                "OCD",
                                "ADHD",
                                "Personality Disorders",
                                "Eating Disorders",
                                "Substance Use Disorders",
                                "Other",
                            ],
                        ),
                    ),
                    SamplerColumnConfig(
                        name="symptom_severity",
                        sampler_type="uniform",
                        params=UniformSamplerParams(
                            low=1.0, high=10.0, decimal_places=0
                        ),
                    ),
                    SamplerColumnConfig(
                        name="symptom_duration_months",
                        sampler_type="uniform",
                        params=UniformSamplerParams(low=0.5, high=120.0),
                    ),
                ]
            )

        if include_treatments:
            columns.extend(
                [
                    SamplerColumnConfig(
                        name="treatment_type",
                        sampler_type="category",
                        params=CategorySamplerParams(
                            values=[
                                "Cognitive Behavioral Therapy",
                                "Dialectical Behavior Therapy",
                                "Psychodynamic Therapy",
                                "Humanistic Therapy",
                                "Medication Only",
                                "Combined Therapy and Medication",
                                "Group Therapy",
                                "Other",
                            ],
                        ),
                    ),
                    SamplerColumnConfig(
                        name="session_frequency",
                        sampler_type="category",
                        params=CategorySamplerParams(
                            values=["Weekly", "Bi-weekly", "Monthly", "As needed"],
                        ),
                    ),
                    SamplerColumnConfig(
                        name="treatment_duration_weeks",
                        sampler_type="uniform",
                        params=UniformSamplerParams(
                            low=1.0, high=104.0, decimal_places=0
                        ),
                    ),
                ]
            )

        if include_outcomes:
            columns.extend(
                [
                    SamplerColumnConfig(
                        name="improvement_score",
                        sampler_type="uniform",
                        params=UniformSamplerParams(low=0.0, high=10.0),
                    ),
                    SamplerColumnConfig(
                        name="treatment_success",
                        sampler_type="category",
                        params=CategorySamplerParams(
                            values=["Yes", "Partial", "No"],
                        ),
                    ),
                    SamplerColumnConfig(
                        name="client_satisfaction",
                        sampler_type="uniform",
                        params=UniformSamplerParams(
                            low=1.0, high=5.0, decimal_places=0
                        ),
                    ),
                ]
            )

        return columns

    def _extract_dataset_from_result(self, job_result: Any) -> Any:
        """Extract the dataset from a successful job result."""
        if hasattr(job_result, "load_dataset"):
            return job_result.load_dataset()
        if hasattr(job_result, "dataset"):
            return job_result.dataset
        return job_result.data if hasattr(job_result, "data") else None

    def _poll_job_status(
        self, job_result: Any, job_id: str, start_time: float
    ) -> tuple[Any, float]:
        """Manually poll the job for completion."""
        max_wait_time = self.config.timeout
        poll_interval = 5
        elapsed = 0

        while elapsed < max_wait_time:
            time.sleep(poll_interval)
            elapsed += poll_interval

            try:
                if hasattr(job_result, "get_job_status"):
                    status = job_result.get_job_status()
                    if status in ["completed", "done", "success"] and hasattr(
                        job_result, "load_dataset"
                    ):
                        return job_result.load_dataset(), time.time() - start_time
                else:
                    job_status = self.client.get_job_results(job_id=job_id)
                    if hasattr(job_status, "data") and job_status.data:
                        return job_status.data, time.time() - start_time

                    if elapsed % 30 == 0:
                        status = getattr(job_status, "status", "unknown")
                        logger.info(f"⏳ Job status: {status} ({elapsed}s elapsed)")
            except Exception as poll_error:
                logger.warning(f"Error checking job status: {poll_error}")
                if elapsed > 60:
                    raise RuntimeError("Failed to poll job status") from poll_error

        raise TimeoutError(
            f"Job {job_id} did not complete within {max_wait_time} seconds"
        )

    def _execute_job_with_fallback(
        self, config_builder: DataDesignerConfigBuilder, num_samples: int
    ) -> tuple[Any, float]:
        """
        Execute job with wait_until_done and fallback to polling.
        Returns a tuple of (data, elapsed_time).
        """
        start_time = time.time()
        job_result = None

        try:
            job_result = self.client.create(
                config_builder=config_builder,
                num_records=num_samples,
                wait_until_done=True,
            )

            if data := self._extract_dataset_from_result(job_result):
                return data, time.time() - start_time

            job_id = getattr(job_result, "job_id", None) or getattr(
                job_result, "id", None
            )
            if not job_id:
                raise ValueError("Could not extract dataset from job_result")

            logger.info(f"Job completed, fetching dataset for job_id: {job_id}")
            data_response = self.client.get_job_results(job_id=job_id)
            return (
                getattr(data_response, "data", data_response),
                time.time() - start_time,
            )

        except Exception as e:
            logger.warning(f"wait_until_done failed, trying manual polling: {e}")

            if job_result is None:
                raise RuntimeError("Failed to create job") from e

            if hasattr(job_result, "wait_until_done"):
                job_result.wait_until_done()
                if hasattr(job_result, "load_dataset"):
                    return job_result.load_dataset(), time.time() - start_time
                raise ValueError("job_result missing load_dataset method") from e

            job_id = (
                getattr(job_result, "job_id", None)
                or getattr(job_result, "id", None)
                or str(job_result)
            )
            logger.info(f"Job created: {job_id}, polling for completion...")

            try:
                return self._poll_job_status(job_result, job_id, start_time)
            except TimeoutError as timeout_err:
                raise timeout_err from e

    def generate_therapeutic_dataset(
        self,
        num_samples: int = 1000,
        include_demographics: bool = True,
        include_symptoms: bool = True,
        include_treatments: bool = True,
        include_outcomes: bool = True,
    ) -> dict[str, Any]:
        """
        Generate a synthetic therapeutic dataset.

        Args:
            num_samples: Number of samples to generate
            include_demographics: Include demographic information
            include_symptoms: Include mental health symptoms
            include_treatments: Include treatment information
            include_outcomes: Include treatment outcomes

        Returns:
            Dictionary containing generated dataset
        """
        config_builder = DataDesignerConfigBuilder()
        columns = self._build_therapeutic_columns(
            include_demographics,
            include_symptoms,
            include_treatments,
            include_outcomes,
        )
        for col in columns:
            config_builder.add_column(col)
        column_names = [col.name for col in columns]

        logger.info(
            f"Generating {num_samples} synthetic therapeutic dataset samples..."
        )

        try:
            data, elapsed_time = self._execute_job_with_fallback(
                config_builder, num_samples
            )

            return {
                "data": data,
                "num_samples": num_samples,
                "generation_time": elapsed_time,
                "columns": column_names,
                "column_names": column_names,  # Alias for consistency
            }
        except Exception as e:
            logger.error(f"Failed to generate dataset: {e}")
            raise

    def generate_bias_detection_dataset(
        self,
        num_samples: int = 1000,
        protected_attributes: Optional[list[str]] = None,
    ) -> dict[str, Any]:
        """
        Generate a synthetic dataset for bias detection testing.

        Args:
            num_samples: Number of samples to generate
            protected_attributes: List of protected attribute names

        Returns:
            Dictionary containing generated dataset for bias analysis
        """
        if protected_attributes is None:
            protected_attributes = ["gender", "ethnicity", "age_group"]

        config_builder = DataDesignerConfigBuilder()

        # Protected attributes
        if "gender" in protected_attributes:
            config_builder.add_column(
                SamplerColumnConfig(
                    name="gender",
                    sampler_type="category",
                    params=CategorySamplerParams(
                        values=["male", "female", "non-binary", "other"],
                    ),
                )
            )

        if "ethnicity" in protected_attributes:
            config_builder.add_column(
                SamplerColumnConfig(
                    name="ethnicity",
                    sampler_type="category",
                    params=CategorySamplerParams(
                        values=[
                            "White",
                            "Black or African American",
                            "Hispanic or Latino",
                            "Asian",
                            "Native American",
                            "Other",
                        ],
                    ),
                )
            )

        if "age_group" in protected_attributes:
            config_builder.add_column(
                SamplerColumnConfig(
                    name="age_group",
                    sampler_type="category",
                    params=CategorySamplerParams(
                        values=["18-25", "26-35", "36-45", "46-55", "56-65", "65+"],
                    ),
                )
            )

        # Outcome variables
        config_builder.add_column(
            SamplerColumnConfig(
                name="treatment_response",
                sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(low=0.0, high=1.0),
            )
        )

        config_builder.add_column(
            SamplerColumnConfig(
                name="session_attendance_rate",
                sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(low=0.0, high=1.0),
            )
        )

        config_builder.add_column(
            SamplerColumnConfig(
                name="therapist_rating",
                sampler_type=SamplerType.UNIFORM,
                params=UniformSamplerParams(low=1.0, high=5.0, decimal_places=0),
            )
        )

        logger.info(f"Generating {num_samples} bias detection dataset samples...")
        start_time = time.time()

        try:
            # Create the job and wait for it to complete
            job_result = self.client.create(
                config_builder=config_builder,
                num_records=num_samples,
                wait_until_done=True,
            )
            elapsed_time = time.time() - start_time
            logger.info(
                "✅ Bias detection dataset generation completed"
                f" in {elapsed_time:.2f} seconds"
            )

            # Load the dataset using the job_result object
            if hasattr(job_result, "load_dataset"):
                data = job_result.load_dataset()
            elif hasattr(job_result, "dataset"):
                data = job_result.dataset
            elif hasattr(job_result, "data"):
                data = job_result.data
            else:
                data = job_result

            return {
                "data": data,
                "num_samples": num_samples,
                "generation_time": elapsed_time,
                "protected_attributes": protected_attributes,
            }
        except Exception as e:
            logger.error(f"Failed to generate bias detection dataset: {e}")
            raise

    def generate_custom_dataset(
        self,
        column_configs: list[SamplerColumnConfig],
        num_samples: int = 1000,
    ) -> dict[str, Any]:
        """
        Generate a custom dataset with user-defined columns.

        Args:
            column_configs: List of SamplerColumnConfig objects defining columns
            num_samples: Number of samples to generate

        Returns:
            Dictionary containing generated dataset
        """
        config_builder = DataDesignerConfigBuilder()

        for col_config in column_configs:
            config_builder.add_column(col_config)

        logger.info(f"Generating {num_samples} custom dataset samples...")
        start_time = time.time()

        try:
            # Create the job and wait for it to complete
            job_result = self.client.create(
                config_builder=config_builder,
                num_records=num_samples,
                wait_until_done=True,
            )
            elapsed_time = time.time() - start_time
            logger.info(
                f"✅ Custom dataset generation completed in {elapsed_time:.2f} seconds"
            )

            # Load the dataset using the job_result object
            if hasattr(job_result, "load_dataset"):
                data = job_result.load_dataset()
            elif hasattr(job_result, "dataset"):
                data = job_result.dataset
            elif hasattr(job_result, "data"):
                data = job_result.data
            else:
                data = job_result

            return {
                "data": data,
                "num_samples": num_samples,
                "generation_time": elapsed_time,
                "columns": [col.name for col in column_configs],
            }
        except Exception as e:
            logger.error(f"Failed to generate custom dataset: {e}")
            raise
