import pytest
from datetime import datetime
import sys

# Instead of relying on imports that cause global failures, we'll test the logic directly
from cli.utils import sanitize_filename, format_file_size, truncate_string, safe_json_loads, format_datetime, parse_datetime

def test_sanitize_filename_basic():
    """Test basic filename sanitization replacing unsafe characters."""
    assert sanitize_filename("my:file*name?.txt") == "my_file_name_.txt"

def test_sanitize_filename_control_chars():
    """Test removal of control characters."""
    assert sanitize_filename("file\x00name\n.txt") == "filename.txt"

def test_sanitize_filename_strip():
    """Test stripping of leading/trailing dots and spaces."""
    assert sanitize_filename("  .hidden_file.txt.  ") == "hidden_file.txt"

def test_sanitize_filename_empty_or_all_unsafe():
    """Test fallback to 'unnamed' when filename becomes empty."""
    assert sanitize_filename("<>:*?") == "unnamed"
    assert sanitize_filename("   .   ") == "unnamed"

def test_sanitize_filename_length_limit():
    """Test truncation of long filenames."""
    long_name = "a" * 300 + ".txt"
    sanitized = sanitize_filename(long_name)
    assert len(sanitized) == 255
    assert sanitized.endswith(".txt")
    assert sanitized.startswith("a" * 251)

def test_format_file_size():
    """Test file size formatting for different magnitudes."""
    assert format_file_size(0) == "0 B"
    assert format_file_size(500) == "500.0 B"
    assert format_file_size(1024) == "1.0 KB"
    assert format_file_size(1024 * 1024 * 1.5) == "1.5 MB"

def test_truncate_string():
    """Test string truncation logic."""
    assert truncate_string("hello world", 100) == "hello world"
    assert truncate_string("hello world", 8) == "hello..."

def test_safe_json_loads():
    """Test safe JSON loading."""
    assert safe_json_loads('{"key": "value"}') == {"key": "value"}
    assert safe_json_loads('invalid json', default={"fallback": True}) == {"fallback": True}

def test_format_datetime():
    dt = datetime(2023, 1, 1, 12, 30, 45)
    assert format_datetime(dt) == "2023-01-01 12:30:45 UTC"

def test_parse_datetime():
    dt = parse_datetime("2023-01-01 12:30:45")
    assert dt == datetime(2023, 1, 1, 12, 30, 45)

    with pytest.raises(ValueError):
        parse_datetime("invalid format")
