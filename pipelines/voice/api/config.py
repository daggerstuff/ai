"""
Configuration management for Pixel Voice API and MCP server.
"""

from pathlib import Path
from typing import List, Optional

from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings


class PipelineStageConfig(BaseModel):
    """Configuration for a single pipeline stage."""

    name: str
    script_path: str
    enabled: bool = True
    timeout_seconds: int = 3600
    retry_count: int = 3
    dependencies: List[str] = Field(default_factory=list)


class DirectoryConfig(BaseModel):
    """Configuration for data directories."""

    voice_raw: str = "data/voice_raw"
    voice_segments: str = "data/voice_segments"
    voice_transcripts: str = "data/voice_transcripts"
    voice_transcripts_filtered: str = "data/voice_transcripts_filtered"
    voice_features: str = "data/voice"
    dialogue_pairs: str = "data/dialogue_pairs"
    therapeutic_pairs: str = "data/therapeutic_pairs"
    voice_consistency: str = "data/voice_consistency"
    voice_optimized: str = "data/voice_optimized"
    logs: str = "logs"
    reports: str = "reports"


class WhisperConfig(BaseModel):
    """Configuration for Whisper transcription."""

    model: str = "large-v2"
    language: str = "en"
    compute_type: str = "float16"
    batch_size: int = 4
    enable_diarization: bool = True


class APIConfig(BaseModel):
    """Configuration for FastAPI server."""

    host: str = "0.0.0.0"
    port: int = 8000
    reload: bool = False
    log_level: str = "info"
    cors_origins: List[str] = Field(default_factory=lambda: ["*"])
    max_concurrent_jobs: int = 5


class MCPConfig(BaseModel):
    """Configuration for MCP server."""

    host: str = "localhost"
    port: int = 8001
    max_connections: int = 10
    timeout_seconds: int = 300


class PixelVoiceConfig(BaseSettings):
    """Main configuration class for Pixel Voice pipeline."""

    # Base paths
    base_dir: str = Field(default_factory=lambda: str(Path(__file__).parent.parent))

    # Directory configuration
    directories: DirectoryConfig = Field(default_factory=DirectoryConfig)

    # Pipeline stages configuration
    pipeline_stages: List[PipelineStageConfig] = Field(
        default_factory=lambda: [
            PipelineStageConfig(
                name="Audio Quality Control",
                script_path="pixel_voice/audio_quality_control.py",
            ),
            PipelineStageConfig(
                name="Batch Transcription",
                script_path="pixel_voice/batch_transcribe.py",
            ),
            PipelineStageConfig(
                name="Transcription Quality Filtering",
                script_path="pixel_voice/transcription_quality_filter.py",
            ),
            PipelineStageConfig(
                name="Feature Extraction",
                script_path="pixel_voice/feature_extraction.py",
            ),
            PipelineStageConfig(
                name="Personality & Emotion Clustering",
                script_path="pixel_voice/personality_emotion_clustering.py",
            ),
            PipelineStageConfig(
                name="Dialogue Pair Construction",
                script_path="pixel_voice/dialogue_pair_constructor.py",
            ),
            PipelineStageConfig(
                name="Dialogue Pair Validation",
                script_path="pixel_voice/dialogue_pair_validation.py",
            ),
            PipelineStageConfig(
                name="Therapeutic Pair Generation",
                script_path="pixel_voice/generate_therapeutic_pairs.py",
            ),
            PipelineStageConfig(
                name="Voice Quality Consistency",
                script_path="pixel_voice/voice_quality_consistency.py",
            ),
            PipelineStageConfig(
                name="Voice Data Filtering",
                script_path="pixel_voice/voice_data_filtering.py",
            ),
        ]
    )

    # Component configurations
    whisper: WhisperConfig = Field(default_factory=WhisperConfig)
    api: APIConfig = Field(default_factory=APIConfig)
    mcp: MCPConfig = Field(default_factory=MCPConfig)

    # Environment settings
    environment: str = Field(default="development", env="PIXEL_VOICE_ENV")
    debug: bool = Field(default=False, env="PIXEL_VOICE_DEBUG")

    class Config:
        env_prefix = "PIXEL_VOICE_"
        env_file = ".env"
        env_file_encoding = "utf-8"

    def get_stage_by_name(self, name: str) -> Optional[PipelineStageConfig]:
        """Get pipeline stage configuration by name."""
        return next(
            (stage for stage in self.pipeline_stages if stage.name == name), None
        )

    def get_enabled_stages(self) -> List[PipelineStageConfig]:
        """Get list of enabled pipeline stages."""
        return [stage for stage in self.pipeline_stages if stage.enabled]

    def ensure_directories(self) -> None:
        """Ensure all configured directories exist."""
        base_path = Path(self.base_dir)
        for dir_name, dir_path in self.directories.model_dump().items():
            full_path = base_path / dir_path
            full_path.mkdir(parents=True, exist_ok=True)


# Global configuration instance
config = PixelVoiceConfig()
