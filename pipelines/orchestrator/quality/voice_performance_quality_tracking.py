"""
Voice Performance Monitoring with Quality Tracking

Adds voice processing performance monitoring with quality tracking (Task 3.20).
"""

import time
from datetime import datetime
from pathlib import Path
from typing import Any

from logger import get_logger

logger = get_logger(__name__)

class VoicePerformanceQualityTracking:
    """Voice processing performance monitoring with quality tracking."""

    def __init__(self, output_dir: str = "./voice_performance"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(exist_ok=True)
        self.logger = get_logger(__name__)
        self.performance_metrics = []
        logger.info("VoicePerformanceQualityTracking initialized")

    def monitor_performance(self, task_name: str, start_time: float, end_time: float,
                          quality_score: float) -> dict[str, Any]:
        """Monitor performance and quality of voice processing tasks."""

        performance_data = {
            "task_name": task_name,
            "processing_time": end_time - start_time,
            "quality_score": quality_score,
            "timestamp": datetime.now().isoformat(),
            "throughput": 1.0 / (end_time - start_time) if end_time > start_time else 0.0
        }

        self.performance_metrics.append(performance_data)
        return performance_data

    def track_quality_metrics(self, voice_file: str, quality_metrics: dict[str, float]) -> dict[str, Any]:
        """Track detailed quality metrics for voice processing."""
        return {
            "voice_file": voice_file,
            "quality_metrics": quality_metrics,
            "overall_quality": sum(quality_metrics.values()) / len(quality_metrics),
            "tracked_at": datetime.now().isoformat()
        }

    def get_performance_summary(self) -> dict[str, Any]:
        """Get summary of performance monitoring."""
        if not self.performance_metrics:
            return {"no_data": True}

        processing_times = [m["processing_time"] for m in self.performance_metrics]
        quality_scores = [m["quality_score"] for m in self.performance_metrics]

        return {
            "total_tasks": len(self.performance_metrics),
            "average_processing_time": sum(processing_times) / len(processing_times),
            "average_quality_score": sum(quality_scores) / len(quality_scores),
            "min_processing_time": min(processing_times),
            "max_processing_time": max(processing_times),
            "performance_trend": "stable"
        }

# Example usage
if __name__ == "__main__":
    monitor = VoicePerformanceQualityTracking()

    # Simulate monitoring
    start = time.time()
    time.sleep(0.1)  # Simulate processing
    end = time.time()

    performance = monitor.monitor_performance("voice_transcription", start, end, 0.89)

    summary = monitor.get_performance_summary()
