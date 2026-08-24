"""
TRULY lazy loader with zero setup from agent perspective.

This module demonstrates how resources can be made available to agents
with zero setup - agents just access them and they load automatically.
"""

import json
import logging
from contextvars import ContextVar
from pathlib import Path
from typing import Any

# Setup logger before used by registry loader
logger = logging.getLogger(__name__)

# Load resource registry from external JSON file
_REGISTRY_FILE = Path(__file__).parent / "resource_registry.json"
try:
    with open(_REGISTRY_FILE) as f:
        _RESOURCE_REGISTRY = json.load(f)
except (FileNotFoundError, json.JSONDecodeError, TypeError) as e:
    logger.error(f"Failed to load resource registry from {_REGISTRY_FILE}: {e}")
    _RESOURCE_REGISTRY = {}


# Cache for loaded resources
_resource_cache: ContextVar[dict[str, Any]] = ContextVar("resource_cache", default={})


def _load_resource(resource_type: str, resource_name: str) -> Any:
    """Actually load a resource from the registry."""
    cache = _resource_cache.get()
    resource_key = f"{resource_type}:{resource_name}"

    if resource_key not in cache:
        # Safe access to registry which might be empty if file load failed
        resource_data = _RESOURCE_REGISTRY.get(resource_type, {}).get(resource_name)

        if resource_data:
            cache[resource_key] = resource_data
            _resource_cache.set(cache)
            return resource_data
        # Return default resource if not found
        default_resource = {
            "name": f"default_{resource_name}",
            "type": resource_type,
        }
        cache[resource_key] = default_resource
        _resource_cache.set(cache)
        return default_resource
    return cache[resource_key]


class LazyResourceProxy:
    """
    Proxy that loads resources lazily when accessed.

    This allows resources to be available as module-level variables
    but only load when actually used.
    """

    def __init__(self, resource_type: str, resource_name: str):
        self.resource_type = resource_type
        self.resource_name = resource_name
        self._loaded_resource = None
        self._accessed = False

    def _ensure_loaded(self):
        """Ensure the resource is loaded."""
        if not self._accessed:
            self._loaded_resource = _load_resource(self.resource_type, self.resource_name)
            self._accessed = True
        return self._loaded_resource

    def __getitem__(self, key):
        """Allow dictionary-style access."""
        resource = self._ensure_loaded()
        return resource[key]

    def __getattr__(self, name):
        """Allow attribute-style access."""
        resource = self._ensure_loaded()
        return getattr(resource, name)

    def __str__(self):
        """String representation."""
        resource = self._ensure_loaded()
        return str(resource)

    def __repr__(self):
        """Representation."""
        if self._accessed:
            return repr(self._loaded_resource)
        return f"LazyResourceProxy({self.resource_type}, {self.resource_name})"


# TRULY LAZY GLOBAL VARIABLES
# These are proxies that only load resources when accessed
# Agents can access them directly without any function calls

# Crisis Expert Agent - loads when accessed
crisis_expert = LazyResourceProxy("agents", "crisis_expert")

# Emotion Analyst Agent - loads when accessed
emotion_analyst = LazyResourceProxy("agents", "emotion_analyst")

# Quality Assurance Agent - loads when accessed
quality_assurance = LazyResourceProxy("agents", "quality_assurance")

# Crisis Detection Skill - loads when accessed
crisis_detection_skill = LazyResourceProxy("skills", "crisis_detection")

# Emotion Analysis Skill - loads when accessed
emotion_analysis_skill = LazyResourceProxy("skills", "emotion_analysis")

# Safety First Rule - loads when accessed
safety_first_rule = LazyResourceProxy("rules", "safety_first")

# Privacy Protection Rule - loads when accessed
privacy_protection_rule = LazyResourceProxy("rules", "privacy_protection")

# Conservative Crisis Direction - loads when accessed
conservative_crisis_direction = LazyResourceProxy("directions", "conservative_crisis")

# Balanced Analysis Direction - loads when accessed
balanced_analysis_direction = LazyResourceProxy("directions", "balanced_analysis")


def demonstrate_true_lazy_usage():
    """
    Demonstrate how agents use resources with TRUE lazy loading.

    Resources are available as variables but only load when accessed.
    """

    # Reset cache for clean demo
    _resource_cache.set({})

    def crisis_agent_first_access():
        """First agent to access resources."""

        # Resources load automatically on first access

        return "First crisis processing completed"

    def crisis_agent_second_access():
        """Second agent accessing same resources."""

        # Same resources - served from cache

        return "Second crisis processing completed"

    def emotion_agent_access():
        """Different agent accessing different resources."""

        # Some resources cached, some new

        return "Emotion processing completed"

    # Run demonstrations
    crisis_agent_first_access()

    crisis_agent_second_access()

    emotion_agent_access()


# Example of how this integrates with agent classes
class ZeroSetupAgent:
    """
    Agent class that uses zero-setup lazy loading.

    No setup required in __init__ - resources are just available.
    """

    def __init__(self, agent_type: str):
        self.agent_type = agent_type
        # NO RESOURCE SETUP NEEDED!

    def process(self, task_content: str):
        """Process task with zero resource setup."""

        # Just access resources directly - they load automatically
        agent = crisis_expert if "crisis" in task_content.lower() else emotion_analyst

        # Resources load automatically when accessed

        return f"Processed by {agent['name']}"


def demo_zero_setup_agents():
    """Demonstrate zero-setup agents."""

    # Reset cache
    _resource_cache.set({})

    # Create agents with NO resource setup
    crisis_agent = ZeroSetupAgent("crisis_expert")
    emotion_agent = ZeroSetupAgent("emotion_analyst")

    # Process tasks - resources load automatically when accessed
    crisis_agent.process("Patient shows signs of crisis")

    emotion_agent.process("Patient discusses emotions")


if __name__ == "__main__":
    demonstrate_true_lazy_usage()
    demo_zero_setup_agents()
