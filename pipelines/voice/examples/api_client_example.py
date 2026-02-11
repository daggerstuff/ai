#!/usr/bin/env python3
"""
Example client for the Pixel Voice API server.
Demonstrates how to interact with the API endpoints.
"""
import asyncio
import json
import time
from typing import Dict, Any

import httpx


class PixelVoiceAPIClient:
    """Client for interacting with the Pixel Voice API."""

    def __init__(self, base_url: str = "http://localhost:8000"):
        self.base_url = base_url
        self.client = httpx.AsyncClient(base_url=base_url)

    async def health_check(self) -> Dict[str, Any]:
        """Check API health status."""
        response = await self.client.get("/")
        return response.json()

    async def transcribe_youtube(self, youtube_url: str, **kwargs) -> Dict[str, Any]:
        """Transcribe a YouTube video."""
        data = {"youtube_url": youtube_url, **kwargs}
        response = await self.client.post("/transcribe", json=data)
        return response.json()

    async def create_pipeline_job(
        self,
        job_name: str,
        stages: list,
        input_data: Dict[str, Any] = None,
        config_overrides: Dict[str, Any] = None,
    ) -> str:
        """Create a new pipeline job."""
        data = {
            "job_name": job_name,
            "stages": stages,
            "input_data": input_data or {},
            "config_overrides": config_overrides or {},
        }
        response = await self.client.post("/pipeline/jobs", json=data)
        return response.text.strip('"')

    async def get_job_status(self, job_id: str) -> Dict[str, Any]:
        """Get status of a specific job."""
        response = await self.client.get(f"/pipeline/jobs/{job_id}")
        return response.json()

    async def list_jobs(self, status_filter: str = None) -> list:
        """List all jobs."""
        params = {"status": status_filter} if status_filter else {}
        response = await self.client.get("/pipeline/jobs", params=params)
        return response.json()

    async def cancel_job(self, job_id: str) -> Dict[str, Any]:
        """Cancel a running job."""
        response = await self.client.delete(f"/pipeline/jobs/{job_id}")
        return response.json()

    async def get_pipeline_status(self) -> Dict[str, Any]:
        """Get overall pipeline status."""
        response = await self.client.get("/pipeline/status")
        return response.json()

    async def get_latest_data(self, data_type: str) -> Dict[str, Any]:
        """Get latest data of a specific type."""
        response = await self.client.get(f"/data/{data_type}/latest")
        return response.json()

    async def list_data_files(self, data_type: str) -> Dict[str, Any]:
        """List data files of a specific type."""
        response = await self.client.get(f"/data/{data_type}/files")
        return response.json()

    async def execute_stage(
        self,
        stage: str,
        input_path: str = None,
        output_path: str = None,
        config_overrides: Dict[str, Any] = None,
    ) -> Dict[str, Any]:
        """Execute a single pipeline stage."""
        data = {
            "stage": stage,
            "input_path": input_path,
            "output_path": output_path,
            "config_overrides": config_overrides or {},
        }
        response = await self.client.post("/pipeline/stages/execute", json=data)
        return response.json()

    async def wait_for_job_completion(self, job_id: str, timeout: int = 3600) -> Dict[str, Any]:
        """Wait for a job to complete."""
        start_time = time.time()

        while time.time() - start_time < timeout:
            job_info = await self.get_job_status(job_id)
            status = job_info["status"]

            print(f"Job {job_id}: {status} ({job_info.get('progress', 0):.1%})")

            if status in ["completed", "failed", "cancelled"]:
                return job_info

            await asyncio.sleep(5)  # Check every 5 seconds

        raise TimeoutError(f"Job {job_id} did not complete within {timeout} seconds")

    async def close(self):
        """Close the HTTP client."""
        await self.client.aclose()


async def example_transcription_workflow():
    """Example workflow: Transcribe a YouTube video."""
    client = PixelVoiceAPIClient()

    try:
        # Check API health
        print("Checking API health...")
        health = await client.health_check()
        print(f"API Status: {health['status']}")

        # Transcribe a YouTube video (replace with actual URL)
        youtube_url = "https://www.youtube.com/watch?v=dQw4w9WgXcQ"  # Example URL
        print(f"\nStarting transcription for: {youtube_url}")

        transcription_result = await client.transcribe_youtube(
            youtube_url=youtube_url, language="en", whisper_model="large-v2"
        )

        job_id = transcription_result.get("job_id")
        if job_id:
            print(f"Transcription job created: {job_id}")

            # Wait for completion
            final_result = await client.wait_for_job_completion(job_id)
            print(f"Transcription completed with status: {final_result['status']}")

            if final_result["status"] == "completed":
                # Get the latest transcript data
                transcripts = await client.get_latest_data("transcripts")
                print(f"Transcript data retrieved: {len(transcripts)} segments")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.close()


async def example_full_pipeline_workflow():
    """Example workflow: Run a complete pipeline job."""
    client = PixelVoiceAPIClient()

    try:
        # Create a full pipeline job
        print("Creating full pipeline job...")

        job_id = await client.create_pipeline_job(
            job_name="Complete Voice Processing Example",
            stages=[
                "audio_quality_control",
                "batch_transcription",
                "transcription_filtering",
                "feature_extraction",
                "dialogue_construction",
            ],
            input_data={"source": "existing_audio_files"},
            config_overrides={"whisper_model": "large-v2"},
        )

        print(f"Pipeline job created: {job_id}")

        # Monitor progress
        final_result = await client.wait_for_job_completion(job_id)
        print(f"Pipeline completed with status: {final_result['status']}")

        if final_result["status"] == "completed":
            # Get results from different stages
            print("\nRetrieving results...")

            # Get latest features
            features = await client.get_latest_data("features")
            print(f"Features extracted: {type(features)}")

            # Get dialogue pairs
            dialogue_pairs = await client.get_latest_data("dialogue_pairs")
            print(f"Dialogue pairs created: {type(dialogue_pairs)}")

            # List all available data files
            for data_type in ["transcripts", "features", "dialogue_pairs"]:
                files = await client.list_data_files(data_type)
                print(f"{data_type.title()} files: {files['count']}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.close()


async def example_monitoring_workflow():
    """Example workflow: Monitor system status and jobs."""
    client = PixelVoiceAPIClient()

    try:
        # Get overall pipeline status
        print("Getting pipeline status...")
        status = await client.get_pipeline_status()
        print(f"Total jobs: {status['total_jobs']}")
        print(f"Running jobs: {status['running_jobs']}")
        print(f"System health: {status['system_health']}")

        # List all jobs
        print("\nListing all jobs...")
        all_jobs = await client.list_jobs()
        for job in all_jobs:
            print(f"- {job['job_id']}: {job['job_name']} ({job['status']})")

        # List only running jobs
        print("\nListing running jobs...")
        running_jobs = await client.list_jobs(status_filter="running")
        for job in running_jobs:
            print(f"- {job['job_id']}: {job['job_name']} - {job.get('progress', 0):.1%}")

    except Exception as e:
        print(f"Error: {e}")

    finally:
        await client.close()


async def main():
    """Main function to run examples."""
    print("Pixel Voice API Client Examples")
    print("=" * 40)

    # Choose which example to run
    examples = {
        "1": ("Transcription Workflow", example_transcription_workflow),
        "2": ("Full Pipeline Workflow", example_full_pipeline_workflow),
        "3": ("Monitoring Workflow", example_monitoring_workflow),
    }

    print("\nAvailable examples:")
    for key, (name, _) in examples.items():
        print(f"{key}. {name}")

    choice = input("\nEnter example number (1-3): ").strip()

    if choice in examples:
        name, func = examples[choice]
        print(f"\nRunning: {name}")
        print("-" * 40)
        await func()
    else:
        print("Invalid choice. Running monitoring workflow as default.")
        await example_monitoring_workflow()


if __name__ == "__main__":
    asyncio.run(main())
