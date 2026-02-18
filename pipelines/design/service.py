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
        GaussianSamplerParams,
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

from ai.pipelines.design.config import DataDesignerConfig

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
            # If config_builder provided, build and patch it
            if config_builder:
                import requests
                import time
                
                config = config_builder.build()
                # Patch columns to include column_type
                config_dict = config.model_dump(mode='json')
                for col in config_dict.get('columns', []):
                    if 'column_type' not in col:
                        col['column_type'] = 'sampler'
                
                # Create job via API
                response = requests.post(
                    f"{self.config.base_url}/v1/data-designer/jobs",
                    json={"spec": {"num_records": kwargs.get('num_records', 100), "config": config_dict}},
                    headers={"Authorization": f"Bearer {self.config.api_key}"},
                )
                response.raise_for_status()
                job_data = response.json()
                job_id = job_data['id']
                
                logger.info(f"🎨 Creating Data Designer generation job")
                
                # If wait_until_done, poll for completion
                if wait_until_done:
                    max_wait = self.config.timeout
                    poll_interval = 2
                    elapsed = 0
                    
                    while elapsed < max_wait:
                        time.sleep(poll_interval)
                        elapsed += poll_interval
                        
                        # Check job status
                        status_response = requests.get(
                            f"{self.config.base_url}/v1/data-designer/jobs/{job_id}",
                            headers={"Authorization": f"Bearer {self.config.api_key}"},
                        )
                        status_response.raise_for_status()
                        job_status = status_response.json()
                        
                        state = job_status.get('status', 'unknown')
                        
                        if state in ['completed', 'succeeded', 'success']:
                            # Job completed, use SDK to fetch results
                            job_results = self.client.get_job_results(job_id=job_id)
                            dataset = job_results.load_dataset()
                            
                            # Convert DataFrame to list of dicts
                            try:
                                import pandas as pd
                                if isinstance(dataset, pd.DataFrame):
                                    data = dataset.to_dict('records')
                                else:
                                    data = dataset
                            except ImportError:
                                data = dataset
                            
                            # Return result wrapped in object with data attribute
                            class JobResult:
                                def __init__(self, data):
                                    self.data = data
                                    self.dataset = data
                            
                            return JobResult(data)
                        
                        elif state in ['failed', 'error']:
                            raise Exception(f"Job {job_id} failed with status: {state}")
                        
                        # Log progress every 30s
                        if elapsed % 30 == 0:
                            logger.info(f"⏳ Job {job_id} status: {state} ({elapsed}s elapsed)")
                    
                    raise TimeoutError(f"Job {job_id} did not complete within {max_wait} seconds")
                
                # If not waiting, return job ID
                class JobResult:
                    def __init__(self, job_id):
                        self.id = job_id
                        self.job_id = job_id
                return JobResult(job_id)
            
            return original_create(config_builder=config_builder, wait_until_done=wait_until_done, **kwargs)
        
        self.client.create = patched_create

        logger.info(
            f"NeMo Data Designer service initialized with base_url: {self.config.base_url}"
        )

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
        column_names = []

        # Demographic columns
        if include_demographics:
            config_builder.add_column(
                SamplerColumnConfig(
                    name="age",
                    sampler_type="uniform",
                    params=UniformSamplerParams(low=18.0, high=80.0, decimal_places=0),
                )
            )
            column_names.append("age")

            config_builder.add_column(
                SamplerColumnConfig(
                    name="gender",
                    sampler_type="category",
                    params=CategorySamplerParams(
                        values=["male", "female", "non-binary", "prefer not to say"],
                    ),
                )
            )
            column_names.append("gender")

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
                            "Pacific Islander",
                            "Other",
                        ],
                    ),
                )
            )
            column_names.append("ethnicity")

        # Symptom columns
        if include_symptoms:
            config_builder.add_column(
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
                )
            )
            column_names.append("primary_diagnosis")

            config_builder.add_column(
                SamplerColumnConfig(
                    name="symptom_severity",
                    sampler_type="uniform",
                    params=UniformSamplerParams(low=1.0, high=10.0, decimal_places=0),
                )
            )
            column_names.append("symptom_severity")

            config_builder.add_column(
                SamplerColumnConfig(
                    name="symptom_duration_months",
                    sampler_type="uniform",
                    params=UniformSamplerParams(low=0.5, high=120.0),
                )
            )
            column_names.append("symptom_duration_months")

        # Treatment columns
        if include_treatments:
            config_builder.add_column(
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
                )
            )
            column_names.append("treatment_type")

            config_builder.add_column(
                SamplerColumnConfig(
                    name="session_frequency",
                    sampler_type="category",
                    params=CategorySamplerParams(
                        values=["Weekly", "Bi-weekly", "Monthly", "As needed"],
                    ),
                )
            )
            column_names.append("session_frequency")

            config_builder.add_column(
                SamplerColumnConfig(
                    name="treatment_duration_weeks",
                    sampler_type="uniform",
                    params=UniformSamplerParams(low=1.0, high=104.0, decimal_places=0),
                )
            )
            column_names.append("treatment_duration_weeks")

        # Outcome columns
        if include_outcomes:
            config_builder.add_column(
                SamplerColumnConfig(
                    name="improvement_score",
                    sampler_type="uniform",
                    params=UniformSamplerParams(low=0.0, high=10.0),
                )
            )
            column_names.append("improvement_score")

            config_builder.add_column(
                SamplerColumnConfig(
                    name="treatment_success",
                    sampler_type="category",
                    params=CategorySamplerParams(
                        values=["Yes", "Partial", "No"],
                    ),
                )
            )
            column_names.append("treatment_success")

            config_builder.add_column(
                SamplerColumnConfig(
                    name="client_satisfaction",
                    sampler_type="uniform",
                    params=UniformSamplerParams(low=1.0, high=5.0, decimal_places=0),
                )
            )
            column_names.append("client_satisfaction")

        # Generate the dataset
        logger.info(
            f"Generating {num_samples} synthetic therapeutic dataset samples..."
        )
        start_time = time.time()

        try:
            # For small datasets, use preview (fast, no job execution)
            # Preview API only supports up to 10 records
            # For larger datasets, use job creation with progress polling
            if num_samples <= 10:
                logger.info("Using preview API for fast generation...")
                preview_results = self.client.preview(
                    config_builder=config_builder,
                    num_records=num_samples,
                )
                elapsed_time = time.time() - start_time
                logger.info(
                    f"Dataset generation completed in {elapsed_time:.2f} seconds"
                )

                # Extract dataset from preview results
                # PreviewResults has a 'dataset' attribute containing the actual data
                if hasattr(preview_results, "dataset"):
                    data = preview_results.dataset
                elif hasattr(preview_results, "data"):
                    data = preview_results.data
                else:
                    data = preview_results
            else:
                # Create the job - use wait_until_done for simplicity
                # According to docs, job_result has wait_until_done() and load_dataset() methods
                logger.info("Creating generation job...")
                try:
                    # Try using wait_until_done=True first (simplest approach)
                    job_result = self.client.create(
                        config_builder=config_builder,
                        num_records=num_samples,
                        wait_until_done=True,
                    )

                    # Load the dataset using the job_result object
                    if hasattr(job_result, "load_dataset"):
                        data = job_result.load_dataset()
                    elif hasattr(job_result, "dataset"):
                        data = job_result.dataset
                    elif hasattr(job_result, "data"):
                        data = job_result.data
                    else:
                        # Fallback: try to get job ID and retrieve results
                        job_id = getattr(job_result, "job_id", None) or getattr(
                            job_result, "id", None
                        )
                        if job_id:
                            logger.info(
                                f"Job completed, fetching dataset for job_id: {job_id}"
                            )
                            # Try alternative API endpoint
                            data = self.client.get_job_results(job_id=job_id)
                            if hasattr(data, "data"):
                                data = data.data
                        else:
                            raise ValueError(
                                "Could not extract dataset from job_result"
                            )

                    elapsed_time = time.time() - start_time
                    logger.info(f"✅ Job completed in {elapsed_time:.2f} seconds")

                except Exception as e:
                    # If wait_until_done fails, fall back to manual polling
                    logger.warning(
                        f"wait_until_done failed, trying manual polling: {e}"
                    )
                    job_result = self.client.create(
                        config_builder=config_builder,
                        num_records=num_samples,
                        wait_until_done=False,
                    )

                    # Use job_result methods if available
                    if hasattr(job_result, "wait_until_done"):
                        job_result.wait_until_done()
                        if hasattr(job_result, "load_dataset"):
                            data = job_result.load_dataset()
                        else:
                            raise ValueError("job_result missing load_dataset method")
                    else:
                        # Fallback: manual polling using job_result object
                        job_id = (
                            getattr(job_result, "job_id", None)
                            or getattr(job_result, "id", None)
                            or str(job_result)
                        )
                        logger.info(f"Job created: {job_id}, polling for completion...")

                        max_wait_time = self.config.timeout
                        poll_interval = 5
                        elapsed = 0

                        while elapsed < max_wait_time:
                            time.sleep(poll_interval)
                            elapsed += poll_interval

                            try:
                                # Try using job_result methods
                                if hasattr(job_result, "get_job_status"):
                                    status = job_result.get_job_status()
                                    if status in ["completed", "done", "success"]:
                                        if hasattr(job_result, "load_dataset"):
                                            data = job_result.load_dataset()
                                            break
                                else:
                                    # Fallback: use client method
                                    job_status = self.client.get_job_results(
                                        job_id=job_id
                                    )
                                    if hasattr(job_status, "data") and job_status.data:
                                        data = job_status.data
                                        break

                                    # Log progress every 30 seconds
                                    if elapsed % 30 == 0:
                                        status = getattr(
                                            job_status, "status", "unknown"
                                        )
                                        logger.info(
                                            f"⏳ Job status: {status} ({elapsed}s elapsed)"
                                        )
                            except Exception as poll_error:
                                logger.warning(
                                    f"Error checking job status: {poll_error}"
                                )
                                if elapsed > 60:
                                    raise
                        else:
                            raise TimeoutError(
                                f"Job {job_id} did not complete within {max_wait_time} seconds"
                            )

                    elapsed_time = time.time() - start_time

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
                f"✅ Bias detection dataset generation completed in {elapsed_time:.2f} seconds"
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
