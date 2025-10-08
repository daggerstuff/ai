"""
FastAPI Authentication Middleware Integration
Task 101: API Authentication System - FastAPI Implementation

This module provides FastAPI-specific middleware and decorators for the authentication system.
"""

from fastapi import FastAPI, HTTPException, Depends, Request, status
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials, APIKeyHeader
from starlette.middleware.base import BaseHTTPMiddleware
from typing import Optional, Callable, Dict, Any
import logging
from security.api_authentication import AuthenticationSystem, UserRole, PermissionLevel, User, APIKey

logger = logging.getLogger(__name__)

# Security schemes
bearer_scheme = HTTPBearer()
api_key_header = APIKeyHeader(name="X-API-Key", auto_error=False)

class FastAPIAuthenticationMiddleware(BaseHTTPMiddleware):
    """
    FastAPI middleware for authentication
    """
    
    def __init__(self, app: FastAPI, auth_system: AuthenticationSystem):
        super().__init__(app)
        self.auth_system = auth_system
        
        # Endpoints that don't require authentication
        self.public_endpoints = {
            "/docs", "/redoc", "/openapi.json", "/health", 
            "/auth/login", "/auth/register"
        }
    
    async def dispatch(self, request: Request, call_next: Callable):
        # Skip authentication for public endpoints
        if request.url.path in self.public_endpoints:
            return await call_next(request)
        
        # Extract authentication information
        auth_header = request.headers.get("Authorization", "")
        api_key = request.headers.get("X-API-Key", "")
        
        authenticated_user = None
        authenticated_api_key = None
        
        # Try JWT authentication
        if auth_header.startswith("Bearer "):
            token = auth_header[7:]
            payload = self.auth_system.verify_jwt_token(token)
            
            if payload:
                user_id = payload['user_id']
                user = self.auth_system.users.get(user_id)
                if user and user.is_active:
                    authenticated_user = user
        
        # Try API key authentication
        elif api_key:
            api_key_obj = self.auth_system.authenticate_api_key(api_key)
            if api_key_obj:
                authenticated_api_key = api_key_obj
        
        # Store authentication info in request state
        print(f"DEBUG: Middleware - authenticated_user before state: {authenticated_user}")
        print(f"DEBUG: Middleware - authenticated_api_key before state: {authenticated_api_key}")
        request.state.authenticated_user = authenticated_user
        request.state.authenticated_api_key = authenticated_api_key

        # If no user or API key is authenticated and the endpoint is not public, raise HTTPException
        if not authenticated_user and not authenticated_api_key and request.url.path not in self.public_endpoints:
            print("DEBUG: Middleware - Raising 401: No user or API key authenticated for non-public endpoint")
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Authentication required"
            )
        
        return await call_next(request)

class AuthenticationDependencies:
    """
    FastAPI dependency injection for authentication
    """
    
    def __init__(self, auth_system: AuthenticationSystem):
        self.auth_system = auth_system
    
    async def get_current_user(
        self, 
        credentials: Optional[HTTPAuthorizationCredentials] = Depends(bearer_scheme)
    ) -> Optional[User]:
        """Get current authenticated user from JWT token"""
        print(f"DEBUG: get_current_user called with credentials: {credentials}")
        if not credentials:
            print("DEBUG: No credentials provided for get_current_user")
            return None

        payload = self.auth_system.verify_jwt_token(credentials.credentials)
        if not payload:
            print("DEBUG: Invalid or expired token for get_current_user")
            return None
        
        user_id = payload['user_id']
        user = self.auth_system.users.get(user_id)
        
        if not user or not user.is_active:
            print(f"DEBUG: User {user_id} not found or inactive for get_current_user")
            return None
        
        print(f"DEBUG: User {user.username} authenticated via JWT")
        return user
    
    async def get_api_key(
        self, 
        api_key: Optional[str] = Depends(api_key_header)
    ) -> Optional[APIKey]:
        """Get current authenticated API key"""
        print(f"DEBUG: get_api_key called with api_key: {api_key}")
        if not api_key:
            print("DEBUG: No API key provided for get_api_key")
            return None

        api_key_obj = self.auth_system.authenticate_api_key(api_key)
        if not api_key_obj:
            print("DEBUG: Invalid API key for get_api_key")
            return None
        
        print(f"DEBUG: API key {api_key_obj.name} authenticated")
        return api_key_obj
    
    def require_permission(self, required_permission: PermissionLevel):
        """Dependency to require specific permission level"""
        async def permission_checker(
            request: Request,
            user: Optional[User] = Depends(self.get_current_user),
            api_key: Optional[APIKey] = Depends(self.get_api_key)
        ):
            # Check user permissions
            if user:
                if not self.auth_system.check_permission(user.role, required_permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"Insufficient permissions. Required: {required_permission.value}"
                    )
                return user
            
            # Check API key permissions
            elif api_key:
                if not self.auth_system.check_api_key_permission(api_key, required_permission):
                    raise HTTPException(
                        status_code=status.HTTP_403_FORBIDDEN,
                        detail=f"API key lacks required permission: {required_permission.value}"
                    )
                return api_key
            
            else:
                print("DEBUG: Authentication required block reached!")
                raise HTTPException(
                    status_code=status.HTTP_401_UNAUTHORIZED,
                    detail="Authentication required"
                )
        
        return permission_checker
    
    def require_role(self, required_role: UserRole):
        """Dependency to require specific user role"""
        async def role_checker(user: User = Depends(self.get_current_user)):
            if user.role != required_role:
                raise HTTPException(
                    status_code=status.HTTP_403_FORBIDDEN,
                    detail=f"Required role: {required_role.value}"
                )
            return user
        
        return role_checker

# FastAPI route examples
def create_auth_routes(app: FastAPI, auth_system: AuthenticationSystem):
    """Create authentication routes for FastAPI app"""
    
    auth_deps = AuthenticationDependencies(auth_system)
    
    @app.post("/auth/login")
    async def login(username: str, password: str):
        """User login endpoint"""
        user = auth_system.authenticate_user(username, password)
        if not user:
            raise HTTPException(
                status_code=status.HTTP_401_UNAUTHORIZED,
                detail="Invalid credentials"
            )
        
        token = auth_system.generate_jwt_token(user)
        return {
            "access_token": token,
            "token_type": "bearer",
            "user_id": user.user_id,
            "username": user.username,
            "role": user.role.value
        }
    
    @app.post("/auth/logout")
    async def logout(
        credentials: HTTPAuthorizationCredentials = Depends(bearer_scheme)
    ):
        """User logout endpoint (revoke token)"""
        if credentials:
            auth_system.revoke_token(credentials.credentials)
        return {"message": "Logged out successfully"}
    
    @app.get("/auth/me")
    async def get_current_user_info(
        user: User = Depends(auth_deps.get_current_user)
    ):
        """Get current user information"""
        return {
            "user_id": user.user_id,
            "username": user.username,
            "email": user.email,
            "role": user.role.value,
            "is_active": user.is_active,
            "created_at": user.created_at.isoformat(),
            "last_login": user.last_login.isoformat() if user.last_login else None
        }
    
    @app.post("/auth/api-keys")
    async def create_api_key(
        name: str,
        permissions: list[str],
        expires_in_days: Optional[int] = None,
        user: User = Depends(auth_deps.require_role(UserRole.ADMIN))
    ):
        """Create new API key (admin only)"""
        try:
            permission_objects = [PermissionLevel(p) for p in permissions]
        except ValueError as e:
            raise HTTPException(
                status_code=status.HTTP_400_BAD_REQUEST,
                detail=f"Invalid permission: {e}"
            )
        
        api_key, api_key_obj = auth_system.create_api_key(
            name, permission_objects, expires_in_days
        )
        
        return {
            "api_key": api_key,
            "key_id": api_key_obj.key_id,
            "name": api_key_obj.name,
            "permissions": [p.value for p in api_key_obj.permissions],
            "expires_at": api_key_obj.expires_at.isoformat() if api_key_obj.expires_at else None
        }
    
    @app.get("/auth/api-keys")
    async def list_api_keys(
        user: User = Depends(auth_deps.require_role(UserRole.ADMIN))
    ):
        """List all API keys (admin only)"""
        return [
            {
                "key_id": api_key.key_id,
                "name": api_key.name,
                "permissions": [p.value for p in api_key.permissions],
                "is_active": api_key.is_active,
                "created_at": api_key.created_at.isoformat(),
                "expires_at": api_key.expires_at.isoformat() if api_key.expires_at else None,
                "last_used": api_key.last_used.isoformat() if api_key.last_used else None
            }
            for api_key in auth_system.api_keys.values()
        ]
    
    @app.delete("/auth/api-keys/{key_id}")
    async def revoke_api_key(
        key_id: str,
        user: User = Depends(auth_deps.require_role(UserRole.ADMIN))
    ):
        """Revoke API key (admin only)"""
        if key_id not in auth_system.api_keys:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="API key not found"
            )
        
        auth_system.api_keys[key_id].is_active = False
        return {"message": "API key revoked successfully"}

# Example protected routes
def create_protected_routes(app: FastAPI, auth_system: AuthenticationSystem):
    """Create example protected routes"""
    
    auth_deps = AuthenticationDependencies(auth_system)
    
    @app.get("/protected/read")
    async def protected_read(
        auth_info = Depends(auth_deps.require_permission(PermissionLevel.READ))
    ):
        """Protected endpoint requiring READ permission"""
        return {"message": "You have read access", "auth_info": str(type(auth_info))}
    
    @app.post("/protected/write")
    async def protected_write(
        data: dict,
        auth_info = Depends(auth_deps.require_permission(PermissionLevel.WRITE))
    ):
        """Protected endpoint requiring WRITE permission"""
        return {"message": "Data written successfully", "data": data}
    
    @app.delete("/protected/delete")
    async def protected_delete(
        item_id: str,
        auth_info = Depends(auth_deps.require_permission(PermissionLevel.DELETE))
    ):
        """Protected endpoint requiring DELETE permission"""
        return {"message": f"Item {item_id} deleted successfully"}
    
    @app.get("/admin/users")
    async def admin_list_users(
        user: User = Depends(auth_deps.require_role(UserRole.ADMIN))
    ):
        """Admin-only endpoint to list all users"""
        return [
            {
                "user_id": u.user_id,
                "username": u.username,
                "email": u.email,
                "role": u.role.value,
                "is_active": u.is_active,
                "created_at": u.created_at.isoformat(),
                "last_login": u.last_login.isoformat() if u.last_login else None
            }
            for u in auth_system.users.values()
        ]

# Complete FastAPI app setup
def create_authenticated_app(secret_key: str) -> tuple[FastAPI, AuthenticationSystem]:
    """Create FastAPI app with authentication system"""
    
    # Initialize authentication system
    auth_system = AuthenticationSystem(secret_key)
    
    # Create FastAPI app
    app = FastAPI(
        title="Pixelated Empathy API",
        description="Enterprise API with JWT and API key authentication",
        version="1.0.0"
    )
    
    # Add authentication middleware
    app.add_middleware(FastAPIAuthenticationMiddleware, auth_system=auth_system)
    
    # Create routes
    create_auth_routes(app, auth_system)
    create_protected_routes(app, auth_system)
    
    # Health check endpoint (public)
    @app.get("/health")
    async def health_check():
        return {"status": "healthy", "authentication": "enabled"}
    
    return app, auth_system

# Example usage
if __name__ == "__main__":
    import uvicorn
    
    # Create app with authentication
    app, auth_system = create_authenticated_app("your-secret-key-here")
    
    # Create default admin user
    admin_user = auth_system.create_user(
        "admin", "admin@pixelated.ai", "admin_password", UserRole.ADMIN
    )
    
    # Create test API key
    api_key, _ = auth_system.create_api_key(
        "test_key", [PermissionLevel.READ, PermissionLevel.WRITE]
    )
    
    print(f"Admin user created: admin / admin_password")
    print(f"Test API key: {api_key}")
    print("Starting server on http://localhost:8000")
    print("API docs available at http://localhost:8000/docs")
    
    # Run server
    uvicorn.run(app, host="0.0.0.0", port=8000)
