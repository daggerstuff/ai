"""
Test suite for ai/dataset_pipeline/tokenize.py

Covers:
- HuggingFace tokenization (basic, multilingual, edge)
- Error handling (missing tokenizer, bad input)
- Config-driven behavior (max length, truncation, padding)
- Tokenization statistics (lengths, overflow, compliance)
- Privacy/bias compliance integration points

London School TDD: All dependencies are mocked/stubbed.
"""

from unittest.mock import MagicMock, patch

import pandas as pd
import pytest

# TDD Anchor: Import the function under test
try:
    from ai.dataset_pipeline.processing.chatml_tokenizer import tokenize_dataset
except ImportError:
    tokenize_dataset = None


class TestTokenizePipeline:
    def test_basic_tokenization(self):
        """Given a DataFrame and a mock tokenizer, when tokenizing, then tokens are added as a new column."""
        df = pd.DataFrame([{"prompt": "Hello", "response": "World"}])
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda x, **kwargs: {"input_ids": [1, 2, 3]}
        with patch(
            "ai.dataset_pipeline.processing.chatml_tokenizer.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            result = tokenize_dataset(
                df, tokenizer_name="mock", text_fields=["prompt", "response"]
            )
            assert "tokens" in result.columns
            assert all(isinstance(t, list) for t in result["tokens"])

    def test_multilingual_tokenization(self):
        """Given multilingual input, when tokenizing, then output tokens for all languages."""
        df = pd.DataFrame([{"prompt": "你好", "response": "世界"}])
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda x, **kwargs: {"input_ids": [42, 43]}
        with patch(
            "ai.dataset_pipeline.processing.chatml_tokenizer.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            result = tokenize_dataset(
                df, tokenizer_name="mock", text_fields=["prompt", "response"]
            )
            assert all(isinstance(t, list) for t in result["tokens"])

    def test_missing_tokenizer_raises(self):
        """Given a missing tokenizer, when tokenizing, then an ImportError or ValueError is raised."""
        df = pd.DataFrame([{"prompt": "test", "response": "test"}])
        with patch(
            "ai.dataset_pipeline.processing.chatml_tokenizer.get_tokenizer",
            side_effect=ImportError("not found"),
        ):
            with pytest.raises(ImportError):
                tokenize_dataset(
                    df, tokenizer_name="notfound", text_fields=["prompt", "response"]
                )

    def test_config_max_length_truncation(self):
        """Given a max_length config, when tokenizing, then tokens are truncated to max_length."""
        df = pd.DataFrame([{"prompt": "a" * 100, "response": "b" * 100}])
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda x, **kwargs: {"input_ids": list(range(50))}
        with patch(
            "ai.dataset_pipeline.processing.chatml_tokenizer.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            result = tokenize_dataset(
                df,
                tokenizer_name="mock",
                text_fields=["prompt", "response"],
                max_length=10,
                truncation=True,
            )
            assert all(len(t) <= 10 for t in result["tokens"])

    def test_tokenization_statistics(self):
        """Given a dataset, when tokenizing, then statistics (mean, max, min) are returned or logged."""
        df = pd.DataFrame([{"prompt": "short", "response": "longer response"}])
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda x, **kwargs: {"input_ids": [1, 2, 3, 4]}
        with patch(
            "ai.dataset_pipeline.processing.chatml_tokenizer.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            with patch(
                "ai.dataset_pipeline.processing.chatml_tokenizer.log_tokenization_stats"
            ) as mock_log:
                tokenize_dataset(
                    df, tokenizer_name="mock", text_fields=["prompt", "response"]
                )
                assert mock_log.called

    def test_error_on_missing_text_fields(self):
        """Given missing text fields, when tokenizing, then a ValueError is raised."""
        df = pd.DataFrame([{"foo": "bar"}])
        mock_tokenizer = MagicMock()
        with patch(
            "ai.dataset_pipeline.processing.chatml_tokenizer.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            with pytest.raises(ValueError):
                tokenize_dataset(
                    df, tokenizer_name="mock", text_fields=["prompt", "response"]
                )

    def test_privacy_bias_hook_integration(self):
        """Given a privacy/bias hook, when tokenizing, then the hook is called for each row."""
        df = pd.DataFrame([{"prompt": "PII: john@example.com", "response": "ok"}])
        mock_tokenizer = MagicMock()
        mock_tokenizer.side_effect = lambda x, **kwargs: {"input_ids": [1, 2]}
        with patch(
            "ai.dataset_pipeline.processing.chatml_tokenizer.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            with patch(
                "ai.dataset_pipeline.processing.chatml_tokenizer.privacy_bias_hook"
            ) as mock_hook:
                tokenize_dataset(
                    df, tokenizer_name="mock", text_fields=["prompt", "response"]
                )
                assert mock_hook.called

    def test_handles_empty_dataframe(self):
        """Given an empty DataFrame, when tokenizing, then returns an empty DataFrame with tokens column."""
        df = pd.DataFrame(columns=["prompt", "response"])
        mock_tokenizer = MagicMock()
        with patch(
            "ai.dataset_pipeline.processing.chatml_tokenizer.get_tokenizer",
            return_value=mock_tokenizer,
        ):
            result = tokenize_dataset(
                df, tokenizer_name="mock", text_fields=["prompt", "response"]
            )
            assert result.empty
            assert "tokens" in result.columns
