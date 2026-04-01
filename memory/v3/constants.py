"""
Shared constants for Claude Subconscious v3.

Centralizes magic numbers to ensure consistency across modules.
"""

# Conversation limits
MAX_CONVERSATION_LENGTH = 3000  # Max chars for reflection prompt
MAX_QUERY_LENGTH = 500  # Max chars for memory query
MAX_TOKENS = 500  # Max tokens for LLM responses

# Database settings
DB_POOL_SIZE = 5
DB_TIMEOUT_MS = 30000  # 30 seconds

# Memory settings
DEFAULT_MAX_MEMORIES = 5
DEFAULT_QUERY_TIMEOUT_MS = 5000  # 5 seconds
DEFAULT_MEMORY_PROVIDER = "local_hindsight"
DEFAULT_BANK_ID = "pixelated"

# Retry settings
DEFAULT_MAX_RETRIES = 3
DEFAULT_RETRY_DELAY_MS = 1000  # 1 second
