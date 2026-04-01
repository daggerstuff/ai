from __future__ import annotations

import logging
from typing import Any, Dict, List, Optional

import aiohttp

logger = logging.getLogger("hindsight_subconscious.models")


class SubconsciousModelProvider:
    """Encapsulates model discovery and priority-based selection."""

    def __init__(self, model_priority: List[str]) -> None:
        self.model_priority = list(model_priority)

    async def discover_available_models(
        self,
        *,
        base_url: str = "https://api.anthropic.com",
    ) -> List[Dict[str, Any]]:
        try:
            async with aiohttp.ClientSession() as session:
                async with session.get(f"{base_url}/v1/models") as response:
                    if response.status != 200:
                        return []
                    data = await response.json()
                    models = data.get("data", [])
                    prioritized: List[Dict[str, Any]] = []
                    for priority_model in self.model_priority:
                        for model in models:
                            if priority_model in model.get("id", ""):
                                prioritized.append(model)
                    return prioritized
        except Exception as exc:
            logger.warning("Failed to discover models: %s", exc)
            return []

    def select_best_model(self, available_models: List[Dict[str, Any]]) -> Optional[str]:
        if not available_models:
            return None
        available_ids = [model.get("id", "") for model in available_models]
        for priority_model in self.model_priority:
            for model_id in available_ids:
                if priority_model in model_id:
                    return model_id
        return available_ids[0] or None
