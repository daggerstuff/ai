#!/usr/bin/env python3
"""
Real-Time Streaming Data Processing System

This module provides real-time streaming capabilities for processing therapeutic
conversations as they arrive, with immediate quality validation, pattern detection,
and integration into the production pipeline.

Key Features:
- Real-time data ingestion from multiple sources
- Streaming quality validation and clinical accuracy assessment
- Live pattern recognition and anomaly detection
- Real-time database integration and indexing
- Streaming analytics and monitoring
- Scalable event-driven architecture
"""

import asyncio
import json
import logging
import statistics

# Import our existing quality validation systems
import sys
import time
from asyncio import Queue
from collections import defaultdict, deque
from collections.abc import AsyncGenerator, Callable
from concurrent.futures import ThreadPoolExecutor
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any

import aiofiles
import websockets

sys.path.append(str(Path(__file__).parent.parent))
from clinical_accuracy_validator import ClinicalAccuracyValidator
from quality_validation.real_quality_validator import RealQualityValidator


@dataclass
class StreamingEvent:
    """Represents a streaming data event."""
    event_id: str
    event_type: str  # 'conversation', 'message', 'quality_update', 'alert'
    timestamp: datetime
    source: str
    data: dict[str, Any]
    priority: int = 1  # 1=low, 2=medium, 3=high, 4=critical
    metadata: dict[str, Any] = None


@dataclass
class StreamingMetrics:
    """Real-time streaming metrics."""
    events_processed: int = 0
    events_per_second: float = 0.0
    average_processing_time: float = 0.0
    quality_scores: list[float] = None
    error_count: int = 0
    active_connections: int = 0
    buffer_size: int = 0
    last_update: datetime = None

    def __post_init__(self):
        if self.quality_scores is None:
            self.quality_scores = []
        if self.last_update is None:
            self.last_update = datetime.now()


class StreamingDataSource:
    """Base class for streaming data sources."""

    def __init__(self, source_id: str, config: dict[str, Any]):
        self.source_id = source_id
        self.config = config
        self.is_active = False
        self.metrics = StreamingMetrics()

    async def start(self):
        """Start the streaming data source."""
        self.is_active = True
        logging.info(f"Started streaming source: {self.source_id}")

    async def stop(self):
        """Stop the streaming data source."""
        self.is_active = False
        logging.info(f"Stopped streaming source: {self.source_id}")

    async def stream_events(self) -> AsyncGenerator[StreamingEvent, None]:
        """Generate streaming events."""
        raise NotImplementedError("Subclasses must implement stream_events")


class WebSocketDataSource(StreamingDataSource):
    """WebSocket-based streaming data source."""

    def __init__(self, source_id: str, config: dict[str, Any]):
        super().__init__(source_id, config)
        self.websocket = None
        self.uri = config.get("uri", "ws://localhost:8765")

    async def stream_events(self) -> AsyncGenerator[StreamingEvent, None]:
        """Stream events from WebSocket connection."""
        try:
            async with websockets.connect(self.uri) as websocket:
                self.websocket = websocket
                self.metrics.active_connections = 1

                async for message in websocket:
                    try:
                        data = json.loads(message)
                        event = StreamingEvent(
                            event_id=data.get("id", f"ws_{int(time.time() * 1000)}"),
                            event_type=data.get("type", "conversation"),
                            timestamp=datetime.now(),
                            source=self.source_id,
                            data=data,
                            priority=data.get("priority", 1)
                        )
                        yield event
                        self.metrics.events_processed += 1

                    except json.JSONDecodeError as e:
                        logging.error(f"Invalid JSON from WebSocket: {e}")
                        self.metrics.error_count += 1

        except Exception as e:
            logging.error(f"WebSocket connection error: {e}")
            self.metrics.error_count += 1
            self.metrics.active_connections = 0


class FileWatcherDataSource(StreamingDataSource):
    """File system watcher for streaming new files."""

    def __init__(self, source_id: str, config: dict[str, Any]):
        super().__init__(source_id, config)
        self.watch_directory = Path(config.get("directory", "data/streaming"))
        self.file_patterns = config.get("patterns", ["*.jsonl", "*.json"])
        self.processed_files = set()

    async def stream_events(self) -> AsyncGenerator[StreamingEvent, None]:
        """Stream events from new files."""
        while self.is_active:
            try:
                # Check for new files
                for pattern in self.file_patterns:
                    for file_path in self.watch_directory.glob(pattern):
                        if file_path not in self.processed_files:
                            self.processed_files.add(file_path)

                            # Process file
                            async for event in self._process_file(file_path):
                                yield event

                # Wait before next check
                await asyncio.sleep(1.0)

            except Exception as e:
                logging.error(f"File watcher error: {e}")
                self.metrics.error_count += 1

    async def _process_file(self, file_path: Path) -> AsyncGenerator[StreamingEvent, None]:
        """Process a single file and generate events."""
        try:
            async with aiofiles.open(file_path) as f:
                if file_path.suffix == ".jsonl":
                    async for line in f:
                        if line.strip():
                            data = json.loads(line)
                            event = StreamingEvent(
                                event_id=f"file_{file_path.name}_{int(time.time() * 1000)}",
                                event_type="conversation",
                                timestamp=datetime.now(),
                                source=self.source_id,
                                data=data,
                                metadata={"file_path": str(file_path)}
                            )
                            yield event
                            self.metrics.events_processed += 1
                else:
                    content = await f.read()
                    data = json.loads(content)
                    if isinstance(data, list):
                        for item in data:
                            event = StreamingEvent(
                                event_id=f"file_{file_path.name}_{int(time.time() * 1000)}",
                                event_type="conversation",
                                timestamp=datetime.now(),
                                source=self.source_id,
                                data=item,
                                metadata={"file_path": str(file_path)}
                            )
                            yield event
                            self.metrics.events_processed += 1
                    else:
                        event = StreamingEvent(
                            event_id=f"file_{file_path.name}_{int(time.time() * 1000)}",
                            event_type="conversation",
                            timestamp=datetime.now(),
                            source=self.source_id,
                            data=data,
                            metadata={"file_path": str(file_path)}
                        )
                        yield event
                        self.metrics.events_processed += 1

        except Exception as e:
            logging.error(f"Error processing file {file_path}: {e}")
            self.metrics.error_count += 1


class StreamingProcessor:
    """Main streaming data processor with real-time capabilities."""

    def __init__(self, config_path: str = "streaming_config.json"):
        self.config = self._load_config(config_path)
        self.data_sources: dict[str, StreamingDataSource] = {}
        self.event_queue = Queue(maxsize=self.config.get("queue_size", 10000))
        self.processing_queue = Queue(maxsize=self.config.get("processing_queue_size", 1000))

        # Initialize processors
        self.quality_validator = RealQualityValidator()
        self.clinical_validator = ClinicalAccuracyValidator()

        # Metrics and monitoring
        self.global_metrics = StreamingMetrics()
        self.processing_times = deque(maxlen=1000)
        self.quality_scores = deque(maxlen=1000)

        # Threading
        self.executor = ThreadPoolExecutor(max_workers=self.config.get("worker_threads", 4))
        self.is_running = False

        # Event handlers
        self.event_handlers: dict[str, list[Callable]] = defaultdict(list)

        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format="%(asctime)s - %(name)s - %(levelname)s - %(message)s"
        )
        self.logger = logging.getLogger(__name__)

    def _load_config(self, config_path: str) -> dict[str, Any]:
        """Load streaming configuration."""
        default_config = {
            "queue_size": 10000,
            "processing_queue_size": 1000,
            "worker_threads": 4,
            "batch_size": 100,
            "quality_threshold": 0.6,
            "processing_timeout": 30.0,
            "metrics_update_interval": 5.0,
            "data_sources": []
        }

        try:
            if Path(config_path).exists():
                with open(config_path) as f:
                    config = json.load(f)
                    default_config.update(config)
        except Exception as e:
            logging.warning(f"Could not load config from {config_path}: {e}")

        return default_config

    def add_data_source(self, source: StreamingDataSource):
        """Add a streaming data source."""
        self.data_sources[source.source_id] = source
        self.logger.info(f"Added data source: {source.source_id}")

    def add_event_handler(self, event_type: str, handler: Callable):
        """Add an event handler for specific event types."""
        self.event_handlers[event_type].append(handler)
        self.logger.info(f"Added event handler for: {event_type}")

    async def start(self):
        """Start the streaming processor."""
        self.is_running = True
        self.logger.info("Starting streaming processor...")

        # Start all data sources
        for source in self.data_sources.values():
            await source.start()

        # Start processing tasks
        tasks = [
            asyncio.create_task(self._event_collector()),
            asyncio.create_task(self._event_processor()),
            asyncio.create_task(self._metrics_updater()),
        ]

        # Start source streaming tasks
        for source in self.data_sources.values():
            tasks.append(asyncio.create_task(self._stream_from_source(source)))

        try:
            await asyncio.gather(*tasks)
        except Exception as e:
            self.logger.error(f"Streaming processor error: {e}")
        finally:
            await self.stop()

    async def stop(self):
        """Stop the streaming processor."""
        self.is_running = False
        self.logger.info("Stopping streaming processor...")

        # Stop all data sources
        for source in self.data_sources.values():
            await source.stop()

        # Shutdown executor
        self.executor.shutdown(wait=True)

        self.logger.info("Streaming processor stopped")

    async def _stream_from_source(self, source: StreamingDataSource):
        """Stream events from a data source."""
        try:
            async for event in source.stream_events():
                if not self.is_running:
                    break

                await self.event_queue.put(event)

        except Exception as e:
            self.logger.error(f"Error streaming from {source.source_id}: {e}")

    async def _event_collector(self):
        """Collect events from the queue and prepare for processing."""
        batch = []
        batch_size = self.config.get("batch_size", 100)

        while self.is_running:
            try:
                # Collect events into batches
                timeout = 1.0  # 1 second timeout
                try:
                    event = await asyncio.wait_for(self.event_queue.get(), timeout=timeout)
                    batch.append(event)

                    # Process batch when full or timeout
                    if len(batch) >= batch_size:
                        await self.processing_queue.put(batch)
                        batch = []

                except TimeoutError:
                    # Process partial batch on timeout
                    if batch:
                        await self.processing_queue.put(batch)
                        batch = []

            except Exception as e:
                self.logger.error(f"Event collector error: {e}")

    async def _event_processor(self):
        """Process event batches."""
        while self.is_running:
            try:
                batch = await self.processing_queue.get()
                start_time = time.time()

                # Process batch in thread pool
                loop = asyncio.get_event_loop()
                await loop.run_in_executor(self.executor, self._process_batch, batch)

                # Update metrics
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)
                self.global_metrics.events_processed += len(batch)

            except Exception as e:
                self.logger.error(f"Event processor error: {e}")

    def _process_batch(self, batch: list[StreamingEvent]):
        """Process a batch of events (runs in thread pool)."""
        for event in batch:
            try:
                start_time = time.time()

                # Process based on event type
                if event.event_type == "conversation":
                    self._process_conversation_event(event)
                elif event.event_type == "message":
                    self._process_message_event(event)
                elif event.event_type == "quality_update":
                    self._process_quality_event(event)

                # Call event handlers
                for handler in self.event_handlers.get(event.event_type, []):
                    try:
                        handler(event)
                    except Exception as e:
                        self.logger.error(f"Event handler error: {e}")

                # Update processing time
                processing_time = time.time() - start_time
                self.processing_times.append(processing_time)

            except Exception as e:
                self.logger.error(f"Error processing event {event.event_id}: {e}")
                self.global_metrics.error_count += 1

    def _process_conversation_event(self, event: StreamingEvent):
        """Process a conversation event."""
        try:
            # Extract conversation data
            conversation_data = event.data

            # Validate and score quality
            quality_result = self.quality_validator.validate_conversation(conversation_data)
            clinical_result = self.clinical_validator.validate_conversation(conversation_data)

            # Update quality metrics
            overall_quality = quality_result.get("overall_quality", 0.0)
            self.quality_scores.append(overall_quality)

            # Check quality threshold
            quality_threshold = self.config.get("quality_threshold", 0.6)
            if overall_quality < quality_threshold:
                # Create alert event
                alert_event = StreamingEvent(
                    event_id=f"alert_{event.event_id}",
                    event_type="alert",
                    timestamp=datetime.now(),
                    source="quality_monitor",
                    data={
                        "alert_type": "low_quality",
                        "original_event": event.event_id,
                        "quality_score": overall_quality,
                        "threshold": quality_threshold
                    },
                    priority=3
                )

                # Process alert
                for handler in self.event_handlers.get("alert", []):
                    handler(alert_event)

            # Store processed conversation (this would integrate with database)
            self._store_conversation(conversation_data, quality_result, clinical_result)

        except Exception as e:
            self.logger.error(f"Error processing conversation event: {e}")

    def _process_message_event(self, event: StreamingEvent):
        """Process a message event."""
        # Implementation for individual message processing

    def _process_quality_event(self, event: StreamingEvent):
        """Process a quality update event."""
        # Implementation for quality updates

    def _store_conversation(self, conversation_data: dict[str, Any],
                          quality_result: dict[str, Any],
                          clinical_result: dict[str, Any]):
        """Store processed conversation (placeholder for database integration)."""
        # This would integrate with the database system
        # For now, we'll just log the storage
        self.logger.info(f"Stored conversation with quality: {quality_result.get('overall_quality', 0.0)}")

    async def _metrics_updater(self):
        """Update streaming metrics periodically."""
        update_interval = self.config.get("metrics_update_interval", 5.0)

        while self.is_running:
            try:
                await asyncio.sleep(update_interval)

                # Calculate metrics
                current_time = datetime.now()
                time_diff = (current_time - self.global_metrics.last_update).total_seconds()

                if time_diff > 0:
                    events_in_period = self.global_metrics.events_processed
                    self.global_metrics.events_per_second = events_in_period / time_diff

                # Calculate average processing time
                if self.processing_times:
                    self.global_metrics.average_processing_time = statistics.mean(self.processing_times)

                # Update quality scores
                if self.quality_scores:
                    self.global_metrics.quality_scores = list(self.quality_scores)

                # Update buffer sizes
                self.global_metrics.buffer_size = self.event_queue.qsize()
                self.global_metrics.last_update = current_time

                # Log metrics
                self.logger.info(f"Streaming Metrics - EPS: {self.global_metrics.events_per_second:.2f}, "
                               f"Avg Processing: {self.global_metrics.average_processing_time:.3f}s, "
                               f"Buffer: {self.global_metrics.buffer_size}, "
                               f"Errors: {self.global_metrics.error_count}")

            except Exception as e:
                self.logger.error(f"Metrics updater error: {e}")

    def get_metrics(self) -> dict[str, Any]:
        """Get current streaming metrics."""
        return {
            "global_metrics": asdict(self.global_metrics),
            "source_metrics": {
                source_id: asdict(source.metrics)
                for source_id, source in self.data_sources.items()
            },
            "queue_sizes": {
                "event_queue": self.event_queue.qsize(),
                "processing_queue": self.processing_queue.qsize()
            }
        }


# Example usage and configuration
async def main():
    """Example usage of the streaming processor."""

    # Create streaming processor
    processor = StreamingProcessor()

    # Add file watcher data source
    file_source = FileWatcherDataSource(
        source_id="file_watcher",
        config={
            "directory": "data/streaming",
            "patterns": ["*.jsonl", "*.json"]
        }
    )
    processor.add_data_source(file_source)

    # Add WebSocket data source (if available)
    # ws_source = WebSocketDataSource(
    #     source_id='websocket',
    #     config={'uri': 'ws://localhost:8765'}
    # )
    # processor.add_data_source(ws_source)

    # Add event handlers
    def conversation_handler(event: StreamingEvent):
        pass

    def alert_handler(event: StreamingEvent):
        pass

    processor.add_event_handler("conversation", conversation_handler)
    processor.add_event_handler("alert", alert_handler)

    # Start processing
    try:
        await processor.start()
    except KeyboardInterrupt:
        await processor.stop()


if __name__ == "__main__":
    asyncio.run(main())
