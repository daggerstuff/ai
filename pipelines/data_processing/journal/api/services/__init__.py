"""
API services module.

This module provides service layer that wraps CommandHandler functionality
for use by API endpoints.
"""

from ai.pipelines.data_processing.journal.api.services.command_handler_service import (
    CommandHandlerService,
)

__all__ = ["CommandHandlerService"]
