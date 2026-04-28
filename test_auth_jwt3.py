import pytest
from fastapi.testclient import TestClient
from api.dataset_api import app, auth_system
from security.api_authentication import PermissionLevel, UserRole

client = TestClient(app)

# To test the actual code path without the middleware intercepting, we can use dependency overrides
async def override_get_current_user():
    class MockUser:
        username = "jwt_user"
        role = UserRole.USER
    return MockUser()

from security.fastapi_auth_middleware import AuthenticationDependencies
from api.dataset_api import auth_deps
# Hmm the authentication middleware is not added explicitly to the app in dataset_api.py,
# The issue might be request.state.authenticated_user is not set by any middleware in this app

from api.dataset_api import get_current_active_user_or_api_key

async def mock_get_current_active_user_or_api_key(request):
    # This is what dataset_api.py does:
    # First try to get authenticated user from request state (JWT token auth)
    # So we must mock request.state
    pass
