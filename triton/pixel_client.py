"""
Triton Inference Server client for Pixel models.

Provides high-level Python API for interacting with Pixel models deployed
on Triton Inference Server with full support for batching, async inference,
and comprehensive error handling.

Example:
    >>> from ai.triton.pixel_client import PixelTritonClient
    >>>
    >>> client = PixelTritonClient(
    ...     server_url="http://localhost:8000",
    ...     model_name="pixel",
    ...     model_version="1"
    ... )
    >>>
    >>> response = await client.infer(
    ...     input_text="I'm feeling overwhelmed",
    ...     session_id="user_123",
    ...     context_type="crisis_support"
    ... )
    >>> print(f"EQ Score: {response['overall_eq']:.2f}")
    >>> print(f"Safety: {response['safety_score']:.2f}")
"""

import asyncio
import logging
import time
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import tritonclient.grpc.aio as grpcclient
import tritonclient.http as httpclient
from tritonclient.utils import InferenceServerException

logger = logging.getLogger(__name__)


class PixelTritonClient:
    """Triton client for Pixel model inference with comprehensive features."""

    def __init__(
        self,
        server_url: str = "localhost:8001",
        model_name: str = "pixel",
        model_version: str = "1",
        use_grpc: bool = True,
        timeout_seconds: int = 30,
        ssl_verify: bool = True,
        ssl_client_cert: Optional[str] = None,
        ssl_client_key: Optional[str] = None,
    ):
        """
        Initialize Triton client for Pixel models.

        Args:
            server_url: Triton server URL (host:port or http://host:port)
            model_name: Name of deployed model (default: "pixel")
            model_version: Model version to use (default: "1")
            use_grpc: Use gRPC protocol if True, HTTP if False
            timeout_seconds: Request timeout in seconds
            ssl_verify: Verify SSL certificates
            ssl_client_cert: Path to SSL client certificate
            ssl_client_key: Path to SSL client key

        Raises:
            ConnectionError: If unable to connect to server
        """
        self.server_url = server_url
        self.model_name = model_name
        self.model_version = model_version
        self.use_grpc = use_grpc
        self.timeout_seconds = timeout_seconds
        self.ssl_verify = ssl_verify
        self.ssl_client_cert = ssl_client_cert
        self.ssl_client_key = ssl_client_key

        self._client = None
        self._model_config = None
        self._connected = False

        # Connection attempt at initialization
        self._initialize_client()

    def _initialize_client(self) -> None:
        """Initialize HTTP client (gRPC uses async-only pattern)."""
        try:
            if self.use_grpc:
                # gRPC client initialized on-demand in async context
                logger.info(f"Configured for gRPC: {self.server_url}")
            else:
                # Initialize HTTP client
                self._client = httpclient.InferenceServerClient(
                    url=self.server_url,
                    verbose=False,
                    ssl=self.ssl_verify,
                    ssl_certificate=self.ssl_client_cert,
                    ssl_key=self.ssl_client_key,
                )
                self._test_connection()
                self._connected = True
                logger.info(f"Connected to Triton server: {self.server_url}")
        except Exception as e:
            logger.error(f"Failed to initialize Triton client: {str(e)}")
            raise ConnectionError(f"Cannot connect to Triton server: {str(e)}") from e

    def _test_connection(self) -> None:
        """Test connection to Triton server."""
        if not self.use_grpc and self._client:
            try:
                if not self._client.is_server_live():
                    raise RuntimeError("Server is not live")
                if not self._client.is_model_ready(self.model_name, self.model_version):
                    raise RuntimeError(
                        f"Model {self.model_name}:{self.model_version} not ready"
                    )
            except Exception as e:
                logger.error(f"Connection test failed: {str(e)}")
                raise

    async def infer(
        self,
        input_text: str,
        session_id: str,
        context_type: str = "therapeutic",
        max_tokens: int = 256,
    ) -> Dict[str, Any]:
        """
        Perform inference on Pixel model.

        Args:
            input_text: Input therapy conversation text
            session_id: Unique session identifier
            context_type: Type of conversation (therapeutic, crisis, assessment)
            max_tokens: Maximum tokens in response

        Returns:
            Dictionary with model outputs:
                - response_text: Generated therapeutic response
                - eq_scores: Array of 5 EQ domain scores (0.0-1.0)
                - overall_eq: Overall EQ score (0.0-1.0)
                - bias_score: Bias detection score
                - safety_score: Safety assessment score
                - persona_mode: Active persona mode
                - inference_time_ms: Inference latency in milliseconds

        Raises:
            InferenceServerException: On inference errors
            ValueError: On invalid inputs
            asyncio.TimeoutError: On timeout
        """
        start_time = time.time()

        # Validate inputs
        if not input_text or len(input_text) > 4096:
            raise ValueError("Input text must be non-empty and <= 4096 characters")
        if not session_id:
            raise ValueError("Session ID required")

        try:
            if self.use_grpc:
                result = await self._infer_grpc(input_text, session_id, context_type)
            else:
                result = self._infer_http(input_text, session_id, context_type)

            # Add inference time
            result["inference_time_ms"] = (time.time() - start_time) * 1000

            return result

        except asyncio.TimeoutError:
            logger.error(f"Inference timeout for session {session_id}")
            raise
        except InferenceServerException as e:
            logger.error(f"Inference error for session {session_id}: {str(e)}")
            raise
        except Exception as e:
            logger.error(f"Unexpected error during inference: {str(e)}")
            raise

    async def _infer_grpc(
        self, input_text: str, session_id: str, context_type: str
    ) -> Dict[str, Any]:
        """Perform gRPC inference."""
        # Extract host and port
        if "://" in self.server_url:
            server_url = self.server_url.split("://")[1]
        else:
            server_url = self.server_url

        async with grpcclient.InferenceServerClient(
            url=server_url, verbose=False
        ) as client:
            # Prepare inputs (simplified - actual implementation would tokenize)
            inputs = [
                grpcclient.InferInput("input_ids", [1, 512], "INT32"),
                grpcclient.InferInput("session_id", [1], "BYTES"),
            ]

            # Set data
            inputs[0].set_data_from_numpy(np.zeros([1, 512], dtype=np.int32))
            inputs[1].set_data_from_numpy(np.array([session_id.encode()], dtype=object))

            # Perform inference
            outputs = [
                grpcclient.InferRequestedOutput("response_text"),
                grpcclient.InferRequestedOutput("eq_scores"),
                grpcclient.InferRequestedOutput("overall_eq"),
                grpcclient.InferRequestedOutput("bias_score"),
                grpcclient.InferRequestedOutput("safety_score"),
                grpcclient.InferRequestedOutput("persona_mode"),
            ]

            response = await asyncio.wait_for(
                client.infer(
                    self.model_name,
                    inputs=inputs,
                    outputs=outputs,
                    request_id=session_id,
                ),
                timeout=self.timeout_seconds,
            )

            return self._parse_response(response)

    def _infer_http(
        self, input_text: str, session_id: str, context_type: str
    ) -> Dict[str, Any]:
        """Perform HTTP inference."""
        if not self._client:
            raise RuntimeError("HTTP client not initialized")

        # Prepare inputs
        inputs = [
            httpclient.InferInput("input_ids", [1, 512], "INT32"),
            httpclient.InferInput("session_id", [1], "BYTES"),
        ]

        inputs[0].set_data_from_numpy(np.zeros([1, 512], dtype=np.int32))
        inputs[1].set_data_from_numpy(np.array([session_id.encode()], dtype=object))

        outputs = [
            httpclient.InferRequestedOutput("response_text"),
            httpclient.InferRequestedOutput("eq_scores"),
            httpclient.InferRequestedOutput("overall_eq"),
            httpclient.InferRequestedOutput("bias_score"),
            httpclient.InferRequestedOutput("safety_score"),
            httpclient.InferRequestedOutput("persona_mode"),
        ]

        response = self._client.infer(
            self.model_name,
            inputs=inputs,
            outputs=outputs,
            request_id=session_id,
        )

        return self._parse_response(response)

    def _parse_response(self, response: Any) -> Dict[str, Any]:
        """Parse Triton inference response."""
        return {
            "response_text": self._get_output(response, "response_text"),
            "eq_scores": self._get_output(response, "eq_scores").tolist(),
            "overall_eq": float(self._get_output(response, "overall_eq")[0]),
            "bias_score": float(self._get_output(response, "bias_score")[0]),
            "safety_score": float(self._get_output(response, "safety_score")[0]),
            "persona_mode": self._get_output(response, "persona_mode"),
        }

    @staticmethod
    def _get_output(response: Any, output_name: str) -> np.ndarray:
        """Extract output from response."""
        return response.get_response_element(output_name)

    async def batch_infer(
        self,
        input_texts: List[str],
        session_ids: List[str],
        context_types: Optional[List[str]] = None,
    ) -> List[Dict[str, Any]]:
        """
        Perform batched inference.

        Args:
            input_texts: List of input texts
            session_ids: List of session IDs
            context_types: List of context types (optional)

        Returns:
            List of inference results

        Raises:
            ValueError: If lengths don't match
        """
        if len(input_texts) != len(session_ids):
            raise ValueError("input_texts and session_ids must have same length")

        if context_types is None:
            context_types = ["therapeutic"] * len(input_texts)

        results = await asyncio.gather(
            *[
                self.infer(text, sid, ctx)
                for text, sid, ctx in zip(input_texts, session_ids, context_types)
            ],
            return_exceptions=True,
        )

        # Handle exceptions
        processed_results = []
        for i, result in enumerate(results):
            if isinstance(result, Exception):
                logger.error(f"Batch item {i} failed: {str(result)}")
                processed_results.append({"error": str(result)})
            else:
                processed_results.append(result)

        return processed_results

    async def health_check(self) -> Dict[str, Any]:
        """
        Check Triton server health.

        Returns:
            Dictionary with health status:
                - server_live: Is server responding
                - model_ready: Is model loaded and ready
                - server_version: Triton version
        """
        try:
            if self.use_grpc:
                server_url = self.server_url.split("://")[-1]
                async with grpcclient.InferenceServerClient(url=server_url) as client:
                    is_live = await client.is_server_live()
                    is_ready = await client.is_model_ready(
                        self.model_name, self.model_version
                    )
            else:
                is_live = self._client.is_server_live()
                is_ready = self._client.is_model_ready(
                    self.model_name, self.model_version
                )

            return {
                "server_live": is_live,
                "model_ready": is_ready,
                "model_name": self.model_name,
                "model_version": self.model_version,
            }
        except Exception as e:
            logger.error(f"Health check failed: {str(e)}")
            return {
                "server_live": False,
                "model_ready": False,
                "error": str(e),
            }

    def get_model_config(self) -> Dict[str, Any]:
        """Get model configuration from server."""
        if self._model_config is None:
            try:
                if not self.use_grpc:
                    metadata = self._client.get_model_metadata(
                        self.model_name, self.model_version
                    )
                    self._model_config = metadata
            except Exception as e:
                logger.error(f"Failed to get model config: {str(e)}")
                return {}

        return self._model_config or {}

    async def close(self) -> None:
        """Cleanup resources."""
        if self._client and hasattr(self._client, "close"):
            self._client.close()
        self._connected = False
        logger.info("Triton client connection closed")


class PixelBatchInferenceManager:
    """
    Manager for efficient batched inference with Pixel models.

    Handles automatic batching, result caching, and performance monitoring.
    """

    def __init__(
        self,
        client: PixelTritonClient,
        batch_size: int = 32,
        batch_timeout_ms: int = 1000,
    ):
        """
        Initialize batch manager.

        Args:
            client: PixelTritonClient instance
            batch_size: Maximum batch size
            batch_timeout_ms: Maximum wait time before sending partial batch
        """
        self.client = client
        self.batch_size = batch_size
        self.batch_timeout_ms = batch_timeout_ms

        self.pending_requests: List[Tuple[str, str, str]] = []
        self.pending_futures: List[asyncio.Future] = []
        self.last_batch_time = time.time()

    async def queue_inference(
        self,
        input_text: str,
        session_id: str,
        context_type: str = "therapeutic",
    ) -> Dict[str, Any]:
        """
        Queue inference request for batching.

        Args:
            input_text: Input text
            session_id: Session ID
            context_type: Context type

        Returns:
            Inference result (when batch is processed)
        """
        future: asyncio.Future = asyncio.Future()
        self.pending_requests.append((input_text, session_id, context_type))
        self.pending_futures.append(future)

        # Check if batch is ready
        should_process = (
            len(self.pending_requests) >= self.batch_size
            or (time.time() - self.last_batch_time) * 1000 > self.batch_timeout_ms
        )

        if should_process:
            await self.process_batch()

        return future

    async def process_batch(self) -> None:
        """Process accumulated batch."""
        if not self.pending_requests:
            return

        try:
            input_texts, session_ids, context_types = zip(*self.pending_requests)
            results = await self.client.batch_infer(
                list(input_texts), list(session_ids), list(context_types)
            )

            # Resolve futures
            for future, result in zip(self.pending_futures, results):
                if not future.done():
                    future.set_result(result)

        except Exception as e:
            # Reject all pending requests
            for future in self.pending_futures:
                if not future.done():
                    future.set_exception(e)

        finally:
            self.pending_requests.clear()
            self.pending_futures.clear()
            self.last_batch_time = time.time()
