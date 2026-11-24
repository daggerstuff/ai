"""Tests for the security test framework."""

import pytest

from ai.tools.scripts.security_test_framework import (
    SecurityTestFramework,
    SecurityVulnerability,
    TestCategory,
    VulnerabilityLevel,
)


@pytest.fixture
def framework() -> SecurityTestFramework:
    return SecurityTestFramework()


def test_report_of_a_clean_scan_is_structurally_complete(framework: SecurityTestFramework) -> None:
    """A report is produced even when nothing has been scanned yet."""
    report = framework.generate_security_report(0.25)

    for key in ("scan_summary", "test_statistics", "vulnerabilities", "vulnerability_summary", "recommendations"):
        assert key in report

    assert report["vulnerabilities"] == []


def test_recorded_vulnerabilities_reach_the_report(framework: SecurityTestFramework) -> None:
    """A finding added to the framework is reported back."""
    framework.add_vulnerability(
        SecurityVulnerability(
            id="V-1",
            title="Reflected input",
            description="The endpoint echoes user input",
            severity=VulnerabilityLevel.HIGH,
            category=TestCategory.INPUT_VALIDATION,
            affected_endpoint="/api/chat",
        )
    )

    report = framework.generate_security_report(0.25)

    assert len(report["vulnerabilities"]) == 1
    assert report["vulnerabilities"][0]["title"] == "Reflected input"
    assert report["vulnerabilities"][0]["severity"] == "high"


def test_payload_corpora_are_populated(framework: SecurityTestFramework) -> None:
    """The injection payload sets the scan draws from are not empty."""
    assert framework.sql_injection_payloads
    assert framework.command_injection_payloads


def test_the_framework_targets_the_configured_base_url(framework: SecurityTestFramework) -> None:
    """The scan target is configurable and defaults to local."""
    assert framework.base_url
    assert SecurityTestFramework(base_url="http://example.test").base_url == "http://example.test"
