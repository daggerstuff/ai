"""Unit tests for meddies_adapter — turn real Vietnamese record entries into canonical PAL fixtures."""

from __future__ import annotations

import pytest

from meddies_adapter import adapt_record


def test_adapt_basic_vietnamese():
    raw = {
        "demographics": {
            "age": 36,
            "gender": "Nữ",
            "province": "Hà Nội",
        },
        "healthcare_behavior": {
            "health_literacy_level": "Thấp",
            "healthcare_seeking_pattern": "Ưu tiên Đông y",
        },
    }
    adapted = adapt_record(raw)
    assert adapted["demographics"]["age"] == 36
    assert adapted["demographics"]["gender"] == "female"
    assert adapted["demographics"]["location"] == "Hà Nội"
    assert adapted["healthcare_behavior"]["health_literacy"] == "low"
    assert adapted["healthcare_behavior"]["preference"] == "traditional medicine"
    assert adapted["_raw"] is raw


def test_adapt_missing_fields():
    raw = {}
    adapted = adapt_record(raw)
    assert adapted["demographics"]["age"] == "unknown age"
    assert adapted["demographics"]["gender"] == "person"
    assert adapted["demographics"]["location"] == "Vietnam"
    assert adapted["healthcare_behavior"]["health_literacy"] == "average"
    assert adapted["healthcare_behavior"]["preference"] == "standard medicine"


def test_adapt_unknown_vietnamese_string_falls_through():
    """Unknown Vietnamese enum falls through to string passthrough (info-preserving)."""
    raw = {
        "demographics": {"age": 50, "gender": "Nam", "province": "Hồ Chí Minh"},
        "healthcare_behavior": {
            "health_literacy_level": "Bất thường",
            "healthcare_seeking_pattern": "Chữa may rủi",
        },
    }
    adapted = adapt_record(raw)
    assert adapted["demographics"]["gender"] == "male"
    assert adapted["healthcare_behavior"]["health_literacy"] == "Bất thường"
    assert adapted["healthcare_behavior"]["preference"] == "Chữa may rủi"


@pytest.mark.parametrize(
    "vi,en",
    [
        ("Nam", "male"),
        ("Nữ", "female"),
        ("Khác", "non-binary"),
    ],
)
def test_gender_translations(vi, en):
    raw = {"demographics": {"age": 30, "gender": vi}}
    adapted = adapt_record(raw)
    assert adapted["demographics"]["gender"] == en


@pytest.mark.parametrize(
    "vi,en",
    [
        ("Thấp", "low"),
        ("Trung bình", "average"),
        ("Cao", "high"),
    ],
)
def test_health_literacy_translations(vi, en):
    raw = {"healthcare_behavior": {"health_literacy_level": vi}}
    adapted = adapt_record(raw)
    assert adapted["healthcare_behavior"]["health_literacy"] == en


@pytest.mark.parametrize(
    "vi,en",
    [
        ("Ưu tiên Đông y", "traditional medicine"),
        ("Ưu tiên Tây y", "modern medicine"),
        ("Kết hợp", "integrated medicine"),
        ("Kết hợp Đông/Tây y", "integrated medicine"),
        ("Ngay lập tức", "immediate care"),
        ("Tự điều trị", "self-treatment"),
        ("Chưa khám bệnh", "no prior care"),
    ],
)
def test_seeking_pattern_translations(vi, en):
    raw = {"healthcare_behavior": {"healthcare_seeking_pattern": vi}}
    adapted = adapt_record(raw)
    assert adapted["healthcare_behavior"]["preference"] == en


def test_real_meddies_record_shape_round_trips_through_format_persona():
    """format_persona() should produce a readable English string from an adapted real record."""
    from meddies_to_pal import format_persona

    raw = {
        "demographics": {"age": 60, "gender": "Nam", "province": "Đồng Nai"},
        "healthcare_behavior": {
            "health_literacy_level": "Cao",
            "healthcare_seeking_pattern": "Ưu tiên Tây y",
        },
    }
    adapted = adapt_record(raw)
    s = format_persona(adapted)
    assert isinstance(s, str)
    assert "{" not in s and "}" not in s
    assert "60-year-old male" in s
    assert "high health literacy" in s
    assert "prefers modern medicine" in s
