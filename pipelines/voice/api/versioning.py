"""
API Versioning Implementation for Pixelated Empathy AI
Provides backward compatibility and smooth API evolution.
"""

from fastapi import FastAPI, Request, HTTPException
from fastapi.routing import APIRoute
from typing import Callable, Dict, Any
import re
from enum import Enum

class APIVersion(str, Enum):
    """Supported API versions."""
    V1 = "v1"
    V2 = "v2"
    LATEST = "v2"  # Points to latest stable version

class VersionedAPIRoute(APIRoute):
    """Custom route class that handles API versioning."""
    
    def __init__(self, path: str, endpoint: Callable, **kwargs):
        # Extract version from path if present
        version_match = re.match(r'^/(v\d+)/', path)
        self.version = version_match.group(1) if version_match else APIVersion.LATEST
        super().__init__(path, endpoint, **kwargs)

def version_header_middleware(request: Request, call_next):
    """Middleware to handle version headers."""
    # Check for version in header
    api_version = request.headers.get("API-Version", APIVersion.LATEST)
    
    # Validate version
    if api_version not in [v.value for v in APIVersion]:
        raise HTTPException(
            status_code=400,
            detail=f"Unsupported API version: {api_version}"
        )
    
    # Add version to request state
    request.state.api_version = api_version
    
    response = await call_next(request)
    
    # Add version info to response headers
    response.headers["API-Version"] = api_version
    response.headers["API-Supported-Versions"] = ",".join([v.value for v in APIVersion])
    
    return response

def create_versioned_app() -> FastAPI:
    """Create FastAPI app with versioning support."""
    app = FastAPI(
        title="Pixelated Empathy AI",
        description="AI-powered empathy and conversation analysis",
        version="2.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add versioning middleware
    app.middleware("http")(version_header_middleware)
    
    return app

# Version-specific endpoint decorators
def v1_endpoint(path: str, **kwargs):
    """Decorator for v1 endpoints."""
    def decorator(func):
        func._api_version = APIVersion.V1
        func._api_path = f"/v1{path}"
        return func
    return decorator

def v2_endpoint(path: str, **kwargs):
    """Decorator for v2 endpoints."""
    def decorator(func):
        func._api_version = APIVersion.V2
        func._api_path = f"/v2{path}"
        return func
    return decorator

# Backward compatibility helpers
class BackwardCompatibility:
    """Handle backward compatibility between API versions."""
    
    @staticmethod
    def transform_v1_to_v2_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform v1 request format to v2."""
        # Example transformation
        if "audio_url" in request_data:
            request_data["input"] = {"url": request_data.pop("audio_url")}
        return request_data
    
    @staticmethod
    def transform_v2_to_v1_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform v2 response format to v1."""
        # Example transformation
        if "result" in response_data and "transcript" in response_data["result"]:
            response_data["transcript"] = response_data["result"]["transcript"]
        return response_data

# Usage example:
# app = create_versioned_app()
# 
# @app.get("/v1/transcribe")
# @v1_endpoint("/transcribe")
# async def transcribe_v1(request: TranscriptRequestV1):
#     # v1 implementation
#     pass
# 
# @app.get("/v2/transcribe")
# @v2_endpoint("/transcribe")
# async def transcribe_v2(request: TranscriptRequestV2):
#     # v2 implementation
#     pass
