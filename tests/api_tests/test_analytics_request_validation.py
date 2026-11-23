"""Unit tests for the analytics-request validator the route relies on.

The route turns anything the validator raises into a 500 AnalyticsError, so
the validator's contract is: never raise, always answer with errors.
"""

from api.techdeck_integration.utils.validation import (
    validate_analytics_request,
    validate_validation_request,
)


def test_a_mixed_naive_and_aware_range_validates_instead_of_raising():
    """A date-only start against an offset-aware end must compare, not raise."""
    result = validate_analytics_request(
        {"start_date": "2026-01-01", "end_date": "2026-01-02T00:00:00Z", "export_type": "json"}
    )
    assert result == {"valid": True, "errors": []}
    inverted = validate_analytics_request({"start_date": "2026-02-01", "end_date": "2026-01-02T00:00:00Z"})
    assert inverted["valid"] is False
    assert inverted["errors"] == ["start_date must not be later than end_date"]


def test_unhashable_payload_values_are_rejected_not_raised():
    """A JSON list or dict where an enum string belongs is a 400, not a 500."""
    for field, value in [
        ("export_type", ["json"]),
        ("metric_type", {"kind": "usage"}),
    ]:
        result = validate_analytics_request({field: value})
        assert result["valid"] is False
        assert any(field in e for e in result["errors"])


def test_unhashable_check_members_are_rejected_not_raised():
    """A list or dict among the requested checks is a 400, not a 500."""
    result = validate_validation_request(
        {"data": [1], "checks": [["completeness"], {"kind": "consistency"}]}, "quality"
    )
    assert result["valid"] is False
    assert any("unknown checks" in e for e in result["errors"])
