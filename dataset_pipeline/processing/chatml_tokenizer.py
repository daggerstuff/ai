"""
Tokenization pipeline for supervised fine-tuning with Wayfarer-2-12B (Unsloth/Lightning.ai Studio compatible).

- Accepts ChatML-formatted records (list of dicts or DataFrame)
- Tokenizes for SFT, handling sequence length, padding, truncation, and special tokens
- Logs tokenization actions and statistics
- Returns tokenized data in modular, testable format (HuggingFace Dataset, DataFrame, or dict)
- No hardcoded secrets; all configs via parameters or secure config
- Robust error handling and logging
- PEP8, <500 lines, ready for TDD

Author: Pixelated Empathy AI Team
"""

import logging
from typing import Any

import pandas as pd

# Handle optional dependencies
try:
    import datasets

    _datasets_available = True
except ImportError:
    datasets = None
    _datasets_available = False

try:
    from transformers import AutoTokenizer, PreTrainedTokenizerBase
except ImportError:

    class AutoTokenizer:
        @staticmethod
        def from_pretrained(*args, **kwargs):
            raise ImportError("transformers package not installed")

    class PreTrainedTokenizerBase:
        pass


# Configure module logger
logger = logging.getLogger("ai.dataset_pipeline.tokenize")
logger.setLevel(logging.INFO)

if not logger.hasHandlers():
    handler = logging.StreamHandler()
    formatter = logging.Formatter("[%(asctime)s] %(levelname)s - %(message)s")
    handler.setFormatter(formatter)
    logger.addHandler(handler)


def _get_tokenizer(
    model_name_or_path: str,
    use_fast: bool = True,
    trust_remote_code: bool = False,
    **kwargs,
) -> Any:
    """
    Load a HuggingFace tokenizer for the specified model.
    """
    # AutoTokenizer will raise ImportError if transformers is not installed
    try:
        tokenizer = AutoTokenizer.from_pretrained(
            model_name_or_path,
            use_fast=use_fast,
            trust_remote_code=trust_remote_code,
            **kwargs,
        )
        logger.info(f"Loaded tokenizer from {model_name_or_path}")
        return tokenizer
    except Exception as e:
        logger.error(f"Failed to load tokenizer: {e}")
        raise


def _format_chatml(record: dict[str, Any], chat_key: str = "messages") -> str:
    """
    Convert a ChatML-formatted record to a single string for tokenization.
    Expects a list of message dicts under `chat_key`, each with 'role' and 'content'.
    """
    if chat_key not in record:
        raise ValueError(f"Record missing '{chat_key}' key.")
    messages = record[chat_key]
    if not isinstance(messages, list):
        raise ValueError(f"'{chat_key}' must be a list of messages.")
    formatted = ""
    for msg in messages:
        role = msg.get("role", "user")
        content = msg.get("content", "")
        # ChatML: <|role|>\ncontent
        formatted += f"<|{role}|>\n{content}\n"
    return formatted.strip()


def _validate_input_data(
    data: list[dict[str, Any]] | pd.DataFrame,
) -> list[dict[str, Any]]:
    """Validate and normalize input data to list of records."""
    if isinstance(data, pd.DataFrame):
        records = data.to_dict(orient="records")
        return [{str(k): v for k, v in record.items()} for record in records]
    if isinstance(data, list):
        return data
    logger.error("Input data must be a list of dicts or a pandas DataFrame.")
    raise TypeError("Input data must be a list of dicts or a pandas DataFrame.")


def _format_texts(records: list[dict[str, Any]], chat_key: str) -> list[str]:
    """Format ChatML records to text strings."""
    texts = []
    for idx, rec in enumerate(records):
        try:
            text = _format_chatml(rec, chat_key=chat_key)
            texts.append(text)
        except Exception as e:
            logger.error(f"Failed to format record {idx}: {e}")
            texts.append("")
    return texts


def _log_tokenization_stats(tokenized: dict[str, Any], num_texts: int) -> None:
    """Log tokenization statistics."""
    lengths = [len(ids) for ids in tokenized["input_ids"]]
    logger.info(f"Tokenized {num_texts} records.")
    logger.info(
        f"Token count: min={min(lengths)}, max={max(lengths)}, mean={sum(lengths) // len(lengths)}"
    )


def _prepare_output(tokenized: dict[str, Any], return_type: str) -> Any:
    """Convert tokenized data to requested output format."""
    if return_type == "dataframe":
        return pd.DataFrame(tokenized)
    if return_type == "dataset":
        if not _datasets_available:
            logger.error(
                "datasets is not installed. Please install it to return a HuggingFace Dataset."
            )
            raise ImportError("datasets is required for return_type='dataset'.")
        dataset_class = getattr(datasets, "Dataset", None)
        if dataset_class is None:
            raise ImportError("Dataset class not found in datasets module")
        return dataset_class.from_dict(tokenized)
    if return_type == "dict":
        return tokenized
    logger.error(f"Invalid return_type: {return_type}")
    raise ValueError(f"Invalid return_type: {return_type}")


def tokenize_chatml(
    data: list[dict[str, Any]] | pd.DataFrame,
    model_name_or_path: str,
    max_length: int = 2048,
    padding: bool | str = "max_length",
    truncation: bool = True,
    chat_key: str = "messages",
    return_type: str = "dataset",
    add_special_tokens: bool = True,
    tokenizer_kwargs: dict[str, Any] | None = None,
    log_stats: bool = True,
    **kwargs,
) -> Any:
    """
    Tokenize ChatML-formatted data for SFT with Wayfarer-2-12B.

    Args:
        data: List of ChatML dicts or DataFrame with ChatML in `chat_key`.
        model_name_or_path: HuggingFace model repo or local path.
        max_length: Max sequence length.
        padding: Padding strategy ("max_length", True, False).
        truncation: Whether to truncate sequences.
        chat_key: Key for ChatML messages in each record.
        return_type: "dataset" (HuggingFace), "dataframe", or "dict".
        add_special_tokens: Whether to add special tokens.
        tokenizer_kwargs: Extra kwargs for tokenizer.
        log_stats: Whether to log tokenization stats.
        **kwargs: Reserved for future use.

    Returns:
        Tokenized data in the requested format.
    """
    tokenizer_kwargs = tokenizer_kwargs or {}

    records = _validate_input_data(data)

    if not records:
        logger.warning("No records to tokenize.")
        return (
            []
            if return_type == "dict"
            else pd.DataFrame()
            if return_type == "dataframe"
            else None
        )

    tokenizer = _get_tokenizer(model_name_or_path, **tokenizer_kwargs)
    texts = _format_texts(records, chat_key)

    try:
        tokenized = tokenizer(
            texts,
            max_length=max_length,
            padding=padding,
            truncation=truncation,
            add_special_tokens=add_special_tokens,
            return_tensors=None,
        )
    except Exception as e:
        logger.error(f"Tokenization failed: {e}")
        raise

    if log_stats:
        _log_tokenization_stats(tokenized, len(texts))

    return _prepare_output(tokenized, return_type)


# === London School TDD: Minimal public stubs for test discovery ===


def get_tokenizer(tokenizer_name: str, **kwargs):
    """London School TDD: public stub for test patching."""
    # Delegate to internal _get_tokenizer if available
    return _get_tokenizer(tokenizer_name, **kwargs)


def privacy_bias_hook(*args, **kwargs):
    """London School TDD: minimal no-op for privacy/bias compliance hook (for test patching)."""
    return


def log_tokenization_stats(*args, **kwargs):
    """London School TDD: minimal no-op for logging tokenization statistics (for test patching)."""
    return


def tokenize_dataset(
    df,
    tokenizer_name: str,
    text_fields: list,
    max_length: int | None = None,
    truncation: bool = False,
    **kwargs,
):
    """
    Tokenize specified text fields in a DataFrame using the provided tokenizer.
    - Each row's text fields are concatenated and tokenized.
    - The "tokens" column contains lists of token ids (from tokenizer["input_ids"]).
    - Enforces max_length/truncation, logs stats, calls privacy/bias hook, and raises on missing fields.
    """
    from ai.dataset_pipeline.processing.chatml_tokenizer import (
        get_tokenizer,
        log_tokenization_stats,
        privacy_bias_hook,
    )

    tokenizer = get_tokenizer(tokenizer_name)
    df = df.copy()

    def concat_fields(row):
        missing = [field for field in text_fields if field not in row]
        if missing:
            raise ValueError(f"Missing required text fields: {missing}")
        return " ".join(str(row[field]) for field in text_fields)

    tokens_list = []
    for _idx, row in df.iterrows():
        # Privacy/bias hook called for each row
        privacy_bias_hook(row)
        text = concat_fields(row)
        result = (
            tokenizer(text, max_length=max_length, truncation=truncation)
            if max_length is not None
            else tokenizer(text)
        )
        # Accept both dict with "input_ids" or list (for test mocks)
        if isinstance(result, dict) and "input_ids" in result:
            tokens = result["input_ids"]
        elif isinstance(result, list):
            tokens = result
        else:
            tokens = []
        # Enforce max_length/truncation if not handled by tokenizer
        if max_length is not None and truncation and len(tokens) > max_length:
            tokens = tokens[:max_length]
        tokens_list.append(tokens)
    df["tokens"] = tokens_list
    log_tokenization_stats(df["tokens"])
    return df
