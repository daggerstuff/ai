"""Custom exceptions for CLI operations."""


class CLIBaseError(Exception):
    """Base exception for CLI errors."""

    def __init__(self, message: str, details: dict | None = None):
        super().__init__(message)
        self.message = message
        self.details = details or {}

    def __str__(self) -> str:
        if self.details:
            detail_str = ", ".join(f"{k}={v!r}" for k, v in self.details.items())
            return f"{self.message} ({detail_str})"
        return self.message


class CLIConfigError(CLIBaseError):
    """Raised when CLI configuration is invalid or cannot be loaded."""


class CLIValidationError(CLIBaseError):
    """Raised when CLI input validation fails."""


class CLIPipelineError(CLIBaseError):
    """Raised when a pipeline operation fails."""
