# AI Security Module

Authentication and authorization module for the Pixelated Empathy AI services.

## Setup

### PYTHONPATH

This module uses relative imports. When running scripts directly or via
pytest, ensure `ai/` is on your `PYTHONPATH`:

```bash
# From repository root
export PYTHONPATH="$PYTHONPATH:$(pwd)/ai"

# Or run tests via uv (recommended — handles PYTHONPATH automatically)
uv run pytest ai/tests/security/
```

### Dependencies

Python packages required (install with `uv`):

```bash
uv pip install pyjwt bcrypt pydantic fastapi uvicorn
```

### Environment Variables

| Variable | Default | Description |
|---|---|---|
| `AUTH_SECRET_KEY` | `super-secret-key-for-dev` | Secret key for JWT token signing. **Must** be set to a strong random value in production. |
| `API_KEY_EXPIRY_DAYS` | `365` | Default API key expiration in days. Set to `0` for no expiry. |

## Database Migrations

Persistent storage for API keys uses SQLite. Run migrations in order:

```bash
# Apply migration
sqlite3 ai/data/conversation_system.db < ai/security/migrations/001_create_api_keys.sql

# Verify tables
sqlite3 ai/data/conversation_system.db ".tables"
# Expected: api_keys, api_key_rate_limits
```

## Module Structure

```
ai/security/
├── __init__.py                  # Module marker
├── api_authentication.py        # Core auth system: JWT, RBAC, API key management
├── fastapi_auth_middleware.py   # FastAPI middleware and dependency injection
├── migrations/
│   └── 001_create_api_keys.sql  # Database migration for persistent key storage
└── README.md                    # This file
```
