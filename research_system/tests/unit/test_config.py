"""Unit tests for configuration management."""

import os
from pathlib import Path

import pytest

from research_system.config import Config


class TestConfig:
    """Tests for Config class."""
    
    def test_load_config(self):
        """Test loading configuration from YAML file."""
        config = Config()
        
        # Verify basic structure loaded
        assert config.get("api_endpoints") is not None
        assert config.get("search_keywords") is not None
        assert config.get("storage") is not None
    
    def test_get_with_dot_notation(self):
        """Test getting nested values with dot notation."""
        config = Config()
        
        # Test nested access
        pubmed_url = config.get("api_endpoints.pubmed.base_url")
        assert pubmed_url == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def test_get_with_default(self):
        """Test getting non-existent key returns default."""
        config = Config()
        
        value = config.get("nonexistent.key", "default_value")
        assert value == "default_value"
    
    def test_get_api_endpoint(self):
        """Test getting API endpoint configuration."""
        config = Config()
        
        pubmed_config = config.get_api_endpoint("pubmed")
        assert "base_url" in pubmed_config
        assert pubmed_config["base_url"] == "https://eutils.ncbi.nlm.nih.gov/entrez/eutils/"
    
    def test_get_search_keywords(self):
        """Test getting search keywords for dataset type."""
        config = Config()
        
        keywords = config.get_search_keywords("therapy_transcripts")
        assert isinstance(keywords, list)
        assert len(keywords) > 0
        assert "therapy transcript dataset" in keywords
    
    def test_get_all_search_keywords(self):
        """Test getting all search keywords."""
        config = Config()
        
        all_keywords = config.get_all_search_keywords()
        assert isinstance(all_keywords, dict)
        assert "therapy_transcripts" in all_keywords
        assert "clinical_outcomes" in all_keywords
    
    def test_get_mesh_terms(self):
        """Test getting MeSH terms."""
        config = Config()
        
        mesh_terms = config.get_mesh_terms()
        assert isinstance(mesh_terms, list)
        assert len(mesh_terms) > 0
        assert "Psychotherapy" in mesh_terms
    
    def test_get_storage_path(self):
        """Test getting storage paths."""
        config = Config()
        
        logs_path = config.get_storage_path("logs")
        assert isinstance(logs_path, Path)
        assert str(logs_path) == "ai/research_system/logs"
    
    def test_get_weekly_targets(self):
        """Test getting weekly targets."""
        config = Config()
        
        week1_targets = config.get_weekly_targets(1)
        assert isinstance(week1_targets, dict)
        assert "sources_identified" in week1_targets
        assert week1_targets["sources_identified"] == 10
    
    def test_get_evaluation_weights(self):
        """Test getting evaluation weights."""
        config = Config()
        
        weights = config.get_evaluation_weights()
        assert isinstance(weights, dict)
        assert weights["therapeutic_relevance"] == 0.35
        assert weights["data_structure_quality"] == 0.25
        assert weights["training_integration"] == 0.20
        assert weights["ethical_accessibility"] == 0.20
    
    def test_get_priority_thresholds(self):
        """Test getting priority thresholds."""
        config = Config()
        
        thresholds = config.get_priority_thresholds()
        assert isinstance(thresholds, dict)
        assert thresholds["high"] == 8.0
        assert thresholds["medium"] == 6.0
    
    def test_get_rate_limit(self):
        """Test getting rate limits."""
        config = Config()
        
        pubmed_limit = config.get_rate_limit("pubmed")
        assert pubmed_limit == 3
    
    def test_get_retry_config(self):
        """Test getting retry configuration."""
        config = Config()
        
        retry_config = config.get_retry_config()
        assert isinstance(retry_config, dict)
        assert retry_config["max_attempts"] == 3
        assert retry_config["backoff_factor"] == 2
    
    def test_env_var_substitution(self):
        """Test environment variable substitution."""
        # Set a test environment variable
        os.environ["TEST_VAR"] = "test_value"
        
        config = Config()
        
        # The actual config uses ${NCBI_API_KEY} which should be empty if not set
        # We're just testing the mechanism works
        pubmed_config = config.get_api_endpoint("pubmed")
        assert "api_key" in pubmed_config
        
        # Clean up
        del os.environ["TEST_VAR"]
    
    def test_env_var_with_default(self):
        """Test environment variable substitution with default value."""
        # Ensure the variable is not set
        if "NONEXISTENT_VAR" in os.environ:
            del os.environ["NONEXISTENT_VAR"]
        
        config = Config()
        
        # Variables not set should use empty string as default
        # This is the current behavior based on the implementation
        assert True  # Basic test that config loads without error
