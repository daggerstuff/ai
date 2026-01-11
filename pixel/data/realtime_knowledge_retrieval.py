"""
Real-Time Knowledge Retrieval During Training

Provides real-time clinical knowledge retrieval during model training
with caching, batch processing, and training context awareness.
"""

import json
import logging
import threading
import time
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
from datetime import datetime
from enum import Enum
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Union

try:
    import numpy as np
    NUMPY_AVAILABLE = True
except ImportError:
    NUMPY_AVAILABLE = False

from .clinical_similarity_search import (
    ClinicalSimilaritySearch,
    EnhancedSearchResult,
    SearchContext,
    SearchQuery,
)


class RetrievalMode(Enum):
    """Modes for knowledge retrieval during training."""
    SYNCHRONOUS = "synchronous"  # Block training until retrieval complete
    ASYNCHRONOUS = "asynchronous"  # Non-blocking retrieval
    BATCH = "batch"  # Batch multiple requests
    CACHED_ONLY = "cached_only"  # Use only cached results


class TrainingPhase(Enum):
    """Training phases for context-aware retrieval."""
    INITIALIZATION = "initialization"
    FORWARD_PASS = "forward_pass"
    BACKWARD_PASS = "backward_pass"
    VALIDATION = "validation"
    CHECKPOINT = "checkpoint"


@dataclass
class RetrievalRequest:
    """Request for knowledge retrieval during training."""
    query_text: str
    training_phase: TrainingPhase
    batch_id: Optional[str] = None
    step_id: Optional[int] = None
    context_metadata: Dict[str, Any] = field(default_factory=dict)
    max_results: int = 5
    min_relevance: float = 0.6
    timeout_seconds: float = 1.0
    priority: int = 1  # 1=high, 2=medium, 3=low
    callback: Optional[Callable] = None
    request_id: str = field(
        default_factory=lambda: f"req_{int(time.time() * 1000)}")
    created_at: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalResponse:
    """Response from knowledge retrieval."""
    request_id: str
    results: List[EnhancedSearchResult]
    retrieval_time_ms: float
    cache_hit: bool
    error: Optional[str] = None
    metadata: Dict[str, Any] = field(default_factory=dict)
    completed_at: datetime = field(default_factory=datetime.now)


@dataclass
class RetrievalStats:
    """Statistics for knowledge retrieval performance."""
    total_requests: int = 0
    cache_hits: int = 0
    cache_misses: int = 0
    avg_retrieval_time_ms: float = 0.0
    total_retrieval_time_ms: float = 0.0
    requests_by_phase: Dict[str, int] = field(
        default_factory=lambda: defaultdict(int))
    errors: int = 0
    timeouts: int = 0
    last_reset: datetime = field(default_factory=datetime.now)


class RealtimeKnowledgeRetrieval:
    """Real-time knowledge retrieval system for training integration."""

    def __init__(self,
                 similarity_search: Optional[ClinicalSimilaritySearch] = None,
                 mode: RetrievalMode = RetrievalMode.ASYNCHRONOUS,
                 cache_size: int = 1000,
                 batch_size: int = 10,
                 batch_timeout_ms: int = 100,
                 max_workers: int = 4,
                 project_root: Optional[Path] = None):
        """Initialize the real-time knowledge retrieval system."""
        self.project_root = Path(project_root) if project_root else Path.cwd()
        self.mode = mode
        self.cache_size = cache_size
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms
        self.max_workers = max_workers

        # Initialize similarity search
        self.similarity_search = similarity_search
        if self.similarity_search is None:
            self._initialize_similarity_search()

        # Caching system
        self.cache: Dict[str, RetrievalResponse] = {}
        self.cache_access_times: Dict[str, datetime] = {}
        self.cache_lock = threading.RLock()

        # Batch processing
        self.batch_queue: deque = deque()
        self.batch_lock = threading.Lock()
        self.batch_processor_running = False

        # Async processing
        self.executor = ThreadPoolExecutor(max_workers=max_workers)
        self.pending_requests: Dict[str, RetrievalRequest] = {}

        # Statistics
        self.stats = RetrievalStats()
        self.stats_lock = threading.Lock()

        # Setup logging
        logging.basicConfig(level=logging.INFO)
        self.logger = logging.getLogger(__name__)

        # Start background services
        self._start_background_services()

    def _initialize_similarity_search(self):
        """Initialize similarity search if not provided."""
        try:
            self.similarity_search = ClinicalSimilaritySearch(
                project_root=self.project_root)
            self.logger.info(
                "Initialized similarity search for real-time retrieval")
        except Exception as e:
            self.logger.error(f"Failed to initialize similarity search: {e}")
            self.similarity_search = None

    def _start_background_services(self):
        """Start background services for batch processing and cache management."""
        if self.mode == RetrievalMode.BATCH:
            self._start_batch_processor()

        # Start cache cleanup service
        self._start_cache_cleanup()

    def _start_batch_processor(self):
        """Start the batch processor thread."""
        if not self.batch_processor_running:
            self.batch_processor_running = True
            batch_thread = threading.Thread(
                target=self._batch_processor_loop, daemon=True)
            batch_thread.start()
            self.logger.info("Started batch processor")

    def _start_cache_cleanup(self):
        """Start the cache cleanup service."""
        cleanup_thread = threading.Thread(
            target=self._cache_cleanup_loop, daemon=True)
        cleanup_thread.start()
        self.logger.info("Started cache cleanup service")

    def _batch_processor_loop(self):
        """Main loop for batch processing requests."""
        while self.batch_processor_running:
            try:
                # Wait for batch timeout or until batch is full
                time.sleep(self.batch_timeout_ms / 1000.0)

                with self.batch_lock:
                    if len(self.batch_queue) == 0:
                        continue

                    # Extract batch of requests
                    batch_requests = []
                    for _ in range(
                            min(self.batch_size, len(self.batch_queue))):
                        if self.batch_queue:
                            batch_requests.append(self.batch_queue.popleft())

                if batch_requests:
                    self._process_batch(batch_requests)

            except Exception as e:
                self.logger.error(f"Error in batch processor: {e}")

    def _cache_cleanup_loop(self):
        """Main loop for cache cleanup."""
        while True:
            try:
                time.sleep(300)  # Run every 5 minutes
                self._cleanup_cache()
            except Exception as e:
                self.logger.error(f"Error in cache cleanup: {e}")

    def _cleanup_cache(self):
        """Clean up old cache entries."""
        with self.cache_lock:
            if len(self.cache) <= self.cache_size:
                return

            # Remove oldest entries
            current_time = datetime.now()
            entries_to_remove = []

            # Sort by access time and remove oldest
            sorted_entries = sorted(
                self.cache_access_times.items(),
                key=lambda x: x[1]
            )

            num_to_remove = len(self.cache) - self.cache_size
            for cache_key, _ in sorted_entries[:num_to_remove]:
                entries_to_remove.append(cache_key)

            # Remove entries
            for cache_key in entries_to_remove:
                self.cache.pop(cache_key, None)
                self.cache_access_times.pop(cache_key, None)

            self.logger.debug(
                f"Cleaned up {len(entries_to_remove)} cache entries")

    def retrieve(
            self, request: RetrievalRequest) -> Union[RetrievalResponse, str]:
        """Retrieve knowledge based on request mode."""
        if self.mode == RetrievalMode.SYNCHRONOUS:
            return self._retrieve_sync(request)
        elif self.mode == RetrievalMode.ASYNCHRONOUS:
            return self._retrieve_async(request)
        elif self.mode == RetrievalMode.BATCH:
            return self._retrieve_batch(request)
        elif self.mode == RetrievalMode.CACHED_ONLY:
            return self._retrieve_cached_only(request)
        else:
            raise ValueError(f"Unknown retrieval mode: {self.mode}")

    def _retrieve_sync(self, request: RetrievalRequest) -> RetrievalResponse:
        """Synchronous knowledge retrieval."""
        start_time = time.time()

        # Check cache first
        cache_key = self._generate_cache_key(request)
        cached_response = self._get_from_cache(cache_key)

        if cached_response:
            self._update_stats(request, True, time.time() - start_time)
            return cached_response

        # Perform retrieval
        try:
            results = self._perform_search(request)

            retrieval_time = (time.time() - start_time) * 1000
            response = RetrievalResponse(
                request_id=request.request_id,
                results=results,
                retrieval_time_ms=retrieval_time,
                cache_hit=False
            )

            # Cache the response
            self._add_to_cache(cache_key, response)

            # Update statistics
            self._update_stats(request, False, retrieval_time)

            return response

        except Exception as e:
            self.logger.error(
                f"Retrieval error for request {request.request_id}: {e}")

            error_response = RetrievalResponse(
                request_id=request.request_id,
                results=[],
                retrieval_time_ms=(time.time() - start_time) * 1000,
                cache_hit=False,
                error=str(e)
            )

            with self.stats_lock:
                self.stats.errors += 1

            return error_response

    def _retrieve_async(self, request: RetrievalRequest) -> str:
        """Asynchronous knowledge retrieval."""
        # Submit to thread pool
        future = self.executor.submit(self._retrieve_sync, request)

        # Store pending request
        self.pending_requests[request.request_id] = request

        # Set up callback if provided
        if request.callback:
            def callback_wrapper(fut):
                try:
                    response = fut.result()
                    request.callback(response)
                except Exception as e:
                    self.logger.error(
                        f"Callback error for request {request.request_id}: {e}")
                finally:
                    self.pending_requests.pop(request.request_id, None)

            future.add_done_callback(callback_wrapper)

        return request.request_id  # Return request ID for tracking

    def _retrieve_batch(self, request: RetrievalRequest) -> str:
        """Batch knowledge retrieval."""
        with self.batch_lock:
            self.batch_queue.append(request)

        return request.request_id

    def _retrieve_cached_only(
            self,
            request: RetrievalRequest) -> RetrievalResponse:
        """Retrieve only from cache."""
        cache_key = self._generate_cache_key(request)
        cached_response = self._get_from_cache(cache_key)

        if cached_response:
            self._update_stats(request, True, 0)
            return cached_response
        else:
            # Return empty response if not in cache
            return RetrievalResponse(
                request_id=request.request_id,
                results=[],
                retrieval_time_ms=0,
                cache_hit=False,
                error="Not found in cache"
            )

    def _process_batch(self, requests: List[RetrievalRequest]):
        """Process a batch of retrieval requests."""
        start_time = time.time()

        # Group requests by similar queries for efficiency
        query_groups = defaultdict(list)
        for request in requests:
            # Simple grouping by first few words
            group_key = " ".join(request.query_text.split()[:3]).lower()
            query_groups[group_key].append(request)

        # Process each group
        for group_requests in query_groups.values():
            try:
                # Use the first request as representative for the group
                representative_request = group_requests[0]

                # Check cache
                cache_key = self._generate_cache_key(representative_request)
                cached_response = self._get_from_cache(cache_key)

                if cached_response:
                    # Use cached result for all requests in group
                    for request in group_requests:
                        response = RetrievalResponse(
                            request_id=request.request_id,
                            results=cached_response.results,
                            retrieval_time_ms=cached_response.retrieval_time_ms,
                            cache_hit=True)

                        if request.callback:
                            request.callback(response)

                        self._update_stats(request, True, 0)
                else:
                    # Perform search for representative request
                    results = self._perform_search(representative_request)

                    retrieval_time = (time.time() - start_time) * 1000

                    # Create response for each request in group
                    for request in group_requests:
                        response = RetrievalResponse(
                            request_id=request.request_id,
                            results=results,
                            retrieval_time_ms=retrieval_time,
                            cache_hit=False
                        )

                        if request.callback:
                            request.callback(response)

                        self._update_stats(request, False, retrieval_time)

                    # Cache the result
                    cache_response = RetrievalResponse(
                        request_id=representative_request.request_id,
                        results=results,
                        retrieval_time_ms=retrieval_time,
                        cache_hit=False
                    )
                    self._add_to_cache(cache_key, cache_response)

            except Exception as e:
                self.logger.error(f"Batch processing error: {e}")

                # Send error response to all requests in group
                for request in group_requests:
                    error_response = RetrievalResponse(
                        request_id=request.request_id,
                        results=[],
                        retrieval_time_ms=(time.time() - start_time) * 1000,
                        cache_hit=False,
                        error=str(e)
                    )

                    if request.callback:
                        request.callback(error_response)

                    with self.stats_lock:
                        self.stats.errors += 1

    def _perform_search(
            self,
            request: RetrievalRequest) -> List[EnhancedSearchResult]:
        """Perform the actual knowledge search."""
        if self.similarity_search is None:
            raise RuntimeError("Similarity search not available")

        # Create search query based on training context
        search_context = self._map_training_phase_to_search_context(
            request.training_phase)

        search_query = SearchQuery(
            text=request.query_text,
            context=search_context,
            max_results=request.max_results,
            min_relevance_score=request.min_relevance
        )

        # Add context metadata to query if available
        if request.context_metadata:
            if "knowledge_types" in request.context_metadata:
                search_query.knowledge_types = request.context_metadata["knowledge_types"]
            if "clinical_domains" in request.context_metadata:
                search_query.clinical_domains = request.context_metadata["clinical_domains"]

        # Perform search
        results = self.similarity_search.search(search_query)

        return results

    def _map_training_phase_to_search_context(
            self, training_phase: TrainingPhase) -> SearchContext:
        """Map training phase to search context."""
        mapping = {
            TrainingPhase.INITIALIZATION: SearchContext.TRAINING,
            TrainingPhase.FORWARD_PASS: SearchContext.TRAINING,
            TrainingPhase.BACKWARD_PASS: SearchContext.TRAINING,
            TrainingPhase.VALIDATION: SearchContext.VALIDATION,
            TrainingPhase.CHECKPOINT: SearchContext.RESEARCH
        }
        return mapping.get(training_phase, SearchContext.TRAINING)

    def _generate_cache_key(self, request: RetrievalRequest) -> str:
        """Generate cache key for request."""
        # Include query text, max results, and min relevance in key
        key_components = [
            request.query_text.lower().strip(),
            str(request.max_results),
            f"{request.min_relevance:.2f}"
        ]

        # Add context metadata if present
        if request.context_metadata:
            sorted_metadata = sorted(request.context_metadata.items())
            metadata_str = json.dumps(sorted_metadata, sort_keys=True)
            key_components.append(metadata_str)

        return "|".join(key_components)

    def _get_from_cache(self, cache_key: str) -> Optional[RetrievalResponse]:
        """Get response from cache."""
        with self.cache_lock:
            if cache_key in self.cache:
                self.cache_access_times[cache_key] = datetime.now()
                return self.cache[cache_key]
        return None

    def _add_to_cache(self, cache_key: str, response: RetrievalResponse):
        """Add response to cache."""
        with self.cache_lock:
            self.cache[cache_key] = response
            self.cache_access_times[cache_key] = datetime.now()

    def _update_stats(
            self,
            request: RetrievalRequest,
            cache_hit: bool,
            retrieval_time_ms: float):
        """Update retrieval statistics."""
        with self.stats_lock:
            self.stats.total_requests += 1

            if cache_hit:
                self.stats.cache_hits += 1
            else:
                self.stats.cache_misses += 1

            self.stats.total_retrieval_time_ms += retrieval_time_ms
            self.stats.avg_retrieval_time_ms = (
                self.stats.total_retrieval_time_ms / self.stats.total_requests
            )

            phase_key = request.training_phase.value
            self.stats.requests_by_phase[phase_key] += 1

    def get_pending_request(
            self,
            request_id: str) -> Optional[RetrievalRequest]:
        """Get pending request by ID."""
        return self.pending_requests.get(request_id)

    def cancel_request(self, request_id: str) -> bool:
        """Cancel a pending request."""
        if request_id in self.pending_requests:
            self.pending_requests.pop(request_id)
            return True
        return False

    def get_stats(self) -> Dict[str, Any]:
        """Get retrieval statistics."""
        with self.stats_lock:
            cache_hit_rate = (
                self.stats.cache_hits / max(self.stats.total_requests, 1) * 100
            )

            return {
                "total_requests": self.stats.total_requests,
                "cache_hits": self.stats.cache_hits,
                "cache_misses": self.stats.cache_misses,
                "cache_hit_rate_percent": cache_hit_rate,
                "avg_retrieval_time_ms": self.stats.avg_retrieval_time_ms,
                "total_retrieval_time_ms": self.stats.total_retrieval_time_ms,
                "requests_by_phase": dict(self.stats.requests_by_phase),
                "errors": self.stats.errors,
                "timeouts": self.stats.timeouts,
                "cache_size": len(self.cache),
                "pending_requests": len(self.pending_requests),
                "last_reset": self.stats.last_reset.isoformat()
            }

    def reset_stats(self):
        """Reset retrieval statistics."""
        with self.stats_lock:
            self.stats = RetrievalStats()

    def clear_cache(self):
        """Clear the retrieval cache."""
        with self.cache_lock:
            self.cache.clear()
            self.cache_access_times.clear()
        self.logger.info("Cleared retrieval cache")

    def shutdown(self):
        """Shutdown the retrieval system."""
        self.logger.info("Shutting down real-time knowledge retrieval system")

        # Stop batch processor
        self.batch_processor_running = False

        # Shutdown executor
        self.executor.shutdown(wait=True)

        # Clear resources
        self.clear_cache()
        self.pending_requests.clear()


# Convenience functions for training integration
def create_training_retrieval_system(
        project_root: Optional[Path] = None,
        mode: RetrievalMode = RetrievalMode.ASYNCHRONOUS,
        cache_size: int = 1000) -> RealtimeKnowledgeRetrieval:
    """Create a real-time knowledge retrieval system optimized for training."""
    return RealtimeKnowledgeRetrieval(
        mode=mode,
        cache_size=cache_size,
        batch_size=20,  # Larger batches for training
        batch_timeout_ms=50,  # Faster batching for training
        max_workers=8,  # More workers for training workload
        project_root=project_root
    )


def create_inference_retrieval_system(
        project_root: Optional[Path] = None) -> RealtimeKnowledgeRetrieval:
    """Create a real-time knowledge retrieval system optimized for inference."""
    return RealtimeKnowledgeRetrieval(
        mode=RetrievalMode.SYNCHRONOUS,  # Synchronous for inference
        cache_size=500,  # Smaller cache for inference
        batch_size=5,
        batch_timeout_ms=10,  # Very fast for inference
        max_workers=2,  # Fewer workers for inference
        project_root=project_root
    )


def main():
    """Test the real-time knowledge retrieval system."""
    print("Testing Real-Time Knowledge Retrieval")

    # Create retrieval system
    retrieval_system = create_training_retrieval_system()

    if retrieval_system.similarity_search is None:
        print("Similarity search not available, using mock system...")

    # Test synchronous retrieval
    print("\n1. Testing synchronous retrieval...")

    sync_request = RetrievalRequest(
        query_text="depression symptoms and treatment",
        training_phase=TrainingPhase.FORWARD_PASS,
        max_results=3
    )

    # Switch to synchronous mode for testing
    retrieval_system.mode = RetrievalMode.SYNCHRONOUS

    try:
        response = retrieval_system.retrieve(sync_request)
        print(
            f"Sync response: {len(response.results)} results in {response.retrieval_time_ms:.2f}ms")
        print(f"Cache hit: {response.cache_hit}")
    except Exception as e:
        print(f"Sync retrieval error: {e}")

    # Test asynchronous retrieval
    print("\n2. Testing asynchronous retrieval...")

    retrieval_system.mode = RetrievalMode.ASYNCHRONOUS

    def async_callback(response):
        print(
            f"Async callback: {len(response.results)} results in {response.retrieval_time_ms:.2f}ms")

    async_request = RetrievalRequest(
        query_text="anxiety disorders cognitive therapy",
        training_phase=TrainingPhase.VALIDATION,
        callback=async_callback
    )

    request_id = retrieval_system.retrieve(async_request)
    print(f"Async request submitted: {request_id}")

    # Wait a bit for async processing
    time.sleep(0.5)

    # Test batch retrieval
    print("\n3. Testing batch retrieval...")

    retrieval_system.mode = RetrievalMode.BATCH

    batch_requests = []
    for i in range(3):
        request = RetrievalRequest(
            query_text=f"therapy technique {i}",
            training_phase=TrainingPhase.FORWARD_PASS,
            callback=lambda r,
            i=i: print(f"Batch result {i}: {len(r.results)} results"))
        batch_requests.append(request)
        retrieval_system.retrieve(request)

    # Wait for batch processing
    time.sleep(0.2)

    # Test cache-only retrieval
    print("\n4. Testing cache-only retrieval...")

    retrieval_system.mode = RetrievalMode.CACHED_ONLY

    cache_request = RetrievalRequest(
        query_text="depression symptoms and treatment",  # Same as first request
        training_phase=TrainingPhase.FORWARD_PASS
    )

    try:
        cache_response = retrieval_system.retrieve(cache_request)
        print(f"Cache-only response: {len(cache_response.results)} results")
        print(f"Cache hit: {cache_response.cache_hit}")
    except Exception as e:
        print(f"Cache-only retrieval error: {e}")

    # Show statistics
    print("\n5. Retrieval statistics:")
    stats = retrieval_system.get_stats()
    for key, value in stats.items():
        print(f"  {key}: {value}")

    # Cleanup
    retrieval_system.shutdown()

    print("\n✅ Real-Time Knowledge Retrieval testing completed!")


if __name__ == "__main__":
    main()
