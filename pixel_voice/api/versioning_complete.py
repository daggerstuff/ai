"""
Complete API Versioning Implementation for Pixelated Empathy AI
Provides comprehensive backward compatibility and smooth API evolution.
"""

from fastapi import FastAPI, Request, HTTPException, Depends
from fastapi.routing import APIRoute
from fastapi.responses import JSONResponse
from typing import Callable, Dict, Any, Optional, List
import re
from enum import Enum
from datetime import datetime, timedelta
import logging
from pydantic import BaseModel
import semver

logger = logging.getLogger(__name__)

class APIVersion(str, Enum):
    """Supported API versions."""
    V1_0 = "v1.0"
    V1_1 = "v1.1"
    V2_0 = "v2.0"
    V2_1 = "v2.1"
    LATEST = "v2.1"  # Points to latest stable version

class VersionStatus(str, Enum):
    """Version lifecycle status."""
    BETA = "beta"
    STABLE = "stable"
    DEPRECATED = "deprecated"
    SUNSET = "sunset"

class VersionInfo(BaseModel):
    """Version information model."""
    version: str
    status: VersionStatus
    release_date: datetime
    sunset_date: Optional[datetime] = None
    changelog_url: Optional[str] = None
    migration_guide_url: Optional[str] = None

class APIVersionManager:
    """Manages API versions and compatibility."""
    
    def __init__(self):
        self.versions = {
            APIVersion.V1_0: VersionInfo(
                version="v1.0",
                status=VersionStatus.DEPRECATED,
                release_date=datetime(2024, 1, 1),
                sunset_date=datetime(2025, 12, 31),
                changelog_url="/docs/changelog/v1.0",
                migration_guide_url="/docs/migration/v1.0-to-v2.0"
            ),
            APIVersion.V1_1: VersionInfo(
                version="v1.1",
                status=VersionStatus.DEPRECATED,
                release_date=datetime(2024, 6, 1),
                sunset_date=datetime(2025, 12, 31),
                changelog_url="/docs/changelog/v1.1",
                migration_guide_url="/docs/migration/v1.1-to-v2.0"
            ),
            APIVersion.V2_0: VersionInfo(
                version="v2.0",
                status=VersionStatus.STABLE,
                release_date=datetime(2024, 12, 1),
                changelog_url="/docs/changelog/v2.0"
            ),
            APIVersion.V2_1: VersionInfo(
                version="v2.1",
                status=VersionStatus.STABLE,
                release_date=datetime(2025, 8, 1),
                changelog_url="/docs/changelog/v2.1"
            )
        }
    
    def get_version_info(self, version: str) -> Optional[VersionInfo]:
        """Get information about a specific version."""
        return self.versions.get(version)
    
    def is_version_supported(self, version: str) -> bool:
        """Check if a version is still supported."""
        version_info = self.get_version_info(version)
        if not version_info:
            return False
        
        if version_info.status == VersionStatus.SUNSET:
            return False
        
        if version_info.sunset_date and datetime.now() > version_info.sunset_date:
            return False
        
        return True
    
    def get_latest_version(self) -> str:
        """Get the latest stable version."""
        return APIVersion.LATEST
    
    def get_supported_versions(self) -> List[str]:
        """Get all currently supported versions."""
        return [v for v in self.versions.keys() if self.is_version_supported(v)]

class VersionedAPIRoute(APIRoute):
    """Custom route class that handles API versioning."""
    
    def __init__(self, path: str, endpoint: Callable, **kwargs):
        # Extract version from path if present
        version_match = re.match(r'^/(v\d+\.\d+)/', path)
        self.version = version_match.group(1) if version_match else APIVersion.LATEST
        
        # Store original endpoint for version-specific handling
        self.original_endpoint = endpoint
        
        super().__init__(path, endpoint, **kwargs)

def get_api_version(request: Request) -> str:
    """Extract API version from request."""
    # Priority order: URL path > Header > Query param > Default
    
    # 1. Check URL path
    path_match = re.match(r'^/(v\d+\.\d+)/', request.url.path)
    if path_match:
        return path_match.group(1)
    
    # 2. Check Accept header (e.g., "application/vnd.pixelated.v2.1+json")
    accept_header = request.headers.get("Accept", "")
    accept_match = re.search(r'vnd\.pixelated\.(v\d+\.\d+)', accept_header)
    if accept_match:
        return accept_match.group(1)
    
    # 3. Check API-Version header
    version_header = request.headers.get("API-Version")
    if version_header:
        return version_header
    
    # 4. Check query parameter
    version_param = request.query_params.get("version")
    if version_param:
        return version_param
    
    # 5. Default to latest
    return APIVersion.LATEST

async def version_middleware(request: Request, call_next):
    """Middleware to handle API versioning."""
    version_manager = APIVersionManager()
    
    # Get requested version
    requested_version = get_api_version(request)
    
    # Validate version
    if not version_manager.is_version_supported(requested_version):
        version_info = version_manager.get_version_info(requested_version)
        
        if version_info and version_info.status == VersionStatus.DEPRECATED:
            # Allow deprecated versions but add warning header
            logger.warning(f"Deprecated API version {requested_version} used")
        elif version_info and version_info.status == VersionStatus.SUNSET:
            # Reject sunset versions
            return JSONResponse(
                status_code=410,
                content={
                    "error": {
                        "code": "VERSION_SUNSET",
                        "message": f"API version {requested_version} has been sunset",
                        "sunset_date": version_info.sunset_date.isoformat() if version_info.sunset_date else None,
                        "migration_guide": version_info.migration_guide_url,
                        "supported_versions": version_manager.get_supported_versions()
                    }
                }
            )
        else:
            # Unknown version
            return JSONResponse(
                status_code=400,
                content={
                    "error": {
                        "code": "UNSUPPORTED_VERSION",
                        "message": f"API version {requested_version} is not supported",
                        "supported_versions": version_manager.get_supported_versions(),
                        "latest_version": version_manager.get_latest_version()
                    }
                }
            )
    
    # Add version to request state
    request.state.api_version = requested_version
    request.state.version_manager = version_manager
    
    # Process request
    response = await call_next(request)
    
    # Add version info to response headers
    response.headers["API-Version"] = requested_version
    response.headers["API-Supported-Versions"] = ",".join(version_manager.get_supported_versions())
    response.headers["API-Latest-Version"] = version_manager.get_latest_version()
    
    # Add deprecation warning if needed
    version_info = version_manager.get_version_info(requested_version)
    if version_info and version_info.status == VersionStatus.DEPRECATED:
        response.headers["Deprecation"] = "true"
        if version_info.sunset_date:
            response.headers["Sunset"] = version_info.sunset_date.strftime("%a, %d %b %Y %H:%M:%S GMT")
        if version_info.migration_guide_url:
            response.headers["Link"] = f'<{version_info.migration_guide_url}>; rel="migration-guide"'
    
    return response

def create_versioned_app() -> FastAPI:
    """Create FastAPI app with comprehensive versioning support."""
    app = FastAPI(
        title="Pixelated Empathy AI",
        description="AI-powered empathy and conversation analysis with comprehensive API versioning",
        version="2.1.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add versioning middleware
    app.middleware("http")(version_middleware)
    
    return app

# Version-specific endpoint decorators
def versioned_endpoint(versions: List[str], path: str, **kwargs):
    """Decorator for version-specific endpoints."""
    def decorator(func):
        func._api_versions = versions
        func._api_path = path
        func._api_kwargs = kwargs
        return func
    return decorator

def v1_endpoint(path: str, **kwargs):
    """Decorator for v1.x endpoints."""
    return versioned_endpoint([APIVersion.V1_0, APIVersion.V1_1], f"/v1{path}", **kwargs)

def v2_endpoint(path: str, **kwargs):
    """Decorator for v2.x endpoints."""
    return versioned_endpoint([APIVersion.V2_0, APIVersion.V2_1], f"/v2{path}", **kwargs)

def latest_endpoint(path: str, **kwargs):
    """Decorator for latest version endpoints."""
    return versioned_endpoint([APIVersion.LATEST], path, **kwargs)

# Backward compatibility helpers
class BackwardCompatibility:
    """Handle backward compatibility between API versions."""
    
    @staticmethod
    def transform_v1_to_v2_request(request_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform v1 request format to v2."""
        transformed = request_data.copy()
        
        # Example transformations
        if "audio_url" in transformed:
            transformed["input"] = {"url": transformed.pop("audio_url")}
        
        if "user_id" in transformed:
            transformed["user"] = {"id": transformed.pop("user_id")}
        
        # Convert old field names
        field_mappings = {
            "msg": "message",
            "lang": "language",
            "opts": "options"
        }
        
        for old_field, new_field in field_mappings.items():
            if old_field in transformed:
                transformed[new_field] = transformed.pop(old_field)
        
        return transformed
    
    @staticmethod
    def transform_v2_to_v1_response(response_data: Dict[str, Any]) -> Dict[str, Any]:
        """Transform v2 response format to v1."""
        transformed = response_data.copy()
        
        # Example transformations
        if "result" in transformed and isinstance(transformed["result"], dict):
            result = transformed.pop("result")
            transformed.update(result)  # Flatten result object
        
        if "user" in transformed and isinstance(transformed["user"], dict):
            user = transformed.pop("user")
            if "id" in user:
                transformed["user_id"] = user["id"]
        
        # Convert new field names back to old
        field_mappings = {
            "message": "msg",
            "language": "lang",
            "options": "opts"
        }
        
        for new_field, old_field in field_mappings.items():
            if new_field in transformed:
                transformed[old_field] = transformed.pop(new_field)
        
        return transformed
    
    @staticmethod
    def get_version_specific_schema(version: str, base_schema: Dict[str, Any]) -> Dict[str, Any]:
        """Get version-specific OpenAPI schema."""
        schema = base_schema.copy()
        
        if version.startswith("v1"):
            # Modify schema for v1 compatibility
            if "properties" in schema:
                properties = schema["properties"]
                
                # Rename fields for v1
                if "message" in properties:
                    properties["msg"] = properties.pop("message")
                if "language" in properties:
                    properties["lang"] = properties.pop("language")
        
        return schema

# Content negotiation based on version
class VersionedResponse:
    """Handle version-specific response formatting."""
    
    def __init__(self, data: Any, version: str):
        self.data = data
        self.version = version
    
    def format_response(self) -> Dict[str, Any]:
        """Format response based on version."""
        if self.version.startswith("v1"):
            return BackwardCompatibility.transform_v2_to_v1_response(self.data)
        else:
            return self.data

# Version-aware dependency injection
def get_version_manager(request: Request) -> APIVersionManager:
    """Dependency to get version manager."""
    return getattr(request.state, 'version_manager', APIVersionManager())

def get_current_version(request: Request) -> str:
    """Dependency to get current API version."""
    return getattr(request.state, 'api_version', APIVersion.LATEST)

# Migration utilities
class MigrationHelper:
    """Helper for API migrations."""
    
    @staticmethod
    def generate_migration_guide(from_version: str, to_version: str) -> Dict[str, Any]:
        """Generate migration guide between versions."""
        return {
            "from_version": from_version,
            "to_version": to_version,
            "breaking_changes": [
                {
                    "change": "Field renamed",
                    "old": "audio_url",
                    "new": "input.url",
                    "description": "Audio URL is now nested under input object"
                },
                {
                    "change": "Response format changed",
                    "old": "Flat response object",
                    "new": "Nested result object",
                    "description": "Response data is now nested under 'result' key"
                }
            ],
            "new_features": [
                {
                    "feature": "Enhanced error handling",
                    "description": "More detailed error responses with error codes"
                },
                {
                    "feature": "Improved authentication",
                    "description": "Support for multiple authentication methods"
                }
            ],
            "migration_steps": [
                "Update request format to use nested objects",
                "Handle new response format",
                "Update error handling for new error codes",
                "Test with new authentication methods"
            ]
        }

# Example usage with FastAPI app
def setup_versioned_routes(app: FastAPI):
    """Set up version-specific routes."""
    
    @app.get("/versions")
    async def get_api_versions(version_manager: APIVersionManager = Depends(get_version_manager)):
        """Get information about all API versions."""
        return {
            "versions": {v: version_manager.get_version_info(v).dict() for v in version_manager.versions},
            "latest": version_manager.get_latest_version(),
            "supported": version_manager.get_supported_versions()
        }
    
    @app.get("/v1/transcribe")
    @app.get("/v1.0/transcribe")
    @app.get("/v1.1/transcribe")
    async def transcribe_v1(
        request: Request,
        current_version: str = Depends(get_current_version)
    ):
        """V1 transcription endpoint with backward compatibility."""
        # Handle v1 format
        request_data = await request.json() if request.method == "POST" else {}
        
        # Transform to v2 format internally
        v2_data = BackwardCompatibility.transform_v1_to_v2_request(request_data)
        
        # Process with v2 logic
        result = {"transcript": "Hello world", "confidence": 0.95}
        
        # Transform back to v1 format
        v1_result = BackwardCompatibility.transform_v2_to_v1_response(result)
        
        return VersionedResponse(v1_result, current_version).format_response()
    
    @app.get("/v2/transcribe")
    @app.get("/v2.0/transcribe")
    @app.get("/v2.1/transcribe")
    @app.get("/transcribe")  # Latest version
    async def transcribe_v2(
        request: Request,
        current_version: str = Depends(get_current_version)
    ):
        """V2 transcription endpoint with latest features."""
        # Handle v2 format directly
        result = {
            "result": {
                "transcript": "Hello world",
                "confidence": 0.95,
                "metadata": {
                    "processing_time": 1.23,
                    "model_version": "v2.1"
                }
            }
        }
        
        return VersionedResponse(result, current_version).format_response()

# Export main components
__all__ = [
    'APIVersion',
    'VersionStatus',
    'APIVersionManager',
    'create_versioned_app',
    'versioned_endpoint',
    'v1_endpoint',
    'v2_endpoint',
    'latest_endpoint',
    'BackwardCompatibility',
    'VersionedResponse',
    'get_version_manager',
    'get_current_version',
    'MigrationHelper',
    'setup_versioned_routes'
]
