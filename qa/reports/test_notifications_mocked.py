#!/usr/bin/env python3
"""
Notification System Testing Script
Tests all notification channels and priority levels
"""

import asyncio
import os
import sys
from datetime import UTC, datetime, timedelta
from unittest.mock import AsyncMock, patch

from notification_integrations import (
    NotificationChannel,
    NotificationManager,
    NotificationPriority,
)


@patch(
    "notification_integrations.WebhookNotifier.send_notification",
    new_callable=AsyncMock,
)
@patch(
    "notification_integrations.PagerDutyNotifier.send_notification",
    new_callable=AsyncMock,
)
@patch("notification_integrations.SlackNotifier.send_notification", new_callable=AsyncMock)
@patch("notification_integrations.EmailNotifier.send_notification", new_callable=AsyncMock)
async def test_individual_channels(mock_email, mock_slack, mock_pagerduty, mock_webhook):
    """Test each notification channel individually"""
    mock_email.return_value = True
    mock_slack.return_value = True
    mock_pagerduty.return_value = True
    mock_webhook.return_value = True
    manager = NotificationManager()

    # Test Email
    await manager.send_alert(
        title="Email Test - Pixelated AI Monitoring",
        message="This is a test email notification from the Pixelated Empathy AI monitoring system.",
        priority=NotificationPriority.LOW,
        channels=[NotificationChannel.EMAIL],
        metadata={
            "test_type": "email_channel",
            "timestamp": datetime.now(UTC).isoformat(),
            "system_status": "testing",
        },
    )

    # Test Slack
    await manager.send_alert(
        title="Slack Test - Pixelated AI Monitoring",
        message="This is a test Slack notification from the Pixelated Empathy AI monitoring system.",
        priority=NotificationPriority.MEDIUM,
        channels=[NotificationChannel.SLACK],
        metadata={
            "test_type": "slack_channel",
            "timestamp": datetime.now(UTC).isoformat(),
            "system_status": "testing",
        },
    )

    # Test PagerDuty
    await manager.send_alert(
        title="PagerDuty Test - Pixelated AI Monitoring",
        message="This is a test PagerDuty notification from the Pixelated Empathy AI monitoring system.",
        priority=NotificationPriority.HIGH,
        channels=[NotificationChannel.PAGERDUTY],
        metadata={
            "test_type": "pagerduty_channel",
            "timestamp": datetime.now(UTC).isoformat(),
            "system_status": "testing",
        },
    )

    # Test Webhooks
    await manager.send_alert(
        title="Webhook Test - Pixelated AI Monitoring",
        message="This is a test webhook notification from the Pixelated Empathy AI monitoring system.",
        priority=NotificationPriority.MEDIUM,
        channels=[NotificationChannel.WEBHOOK],
        metadata={
            "test_type": "webhook_channel",
            "timestamp": datetime.now(UTC).isoformat(),
            "system_status": "testing",
        },
    )


@patch(
    "notification_integrations.WebhookNotifier.send_notification",
    new_callable=AsyncMock,
)
@patch(
    "notification_integrations.PagerDutyNotifier.send_notification",
    new_callable=AsyncMock,
)
@patch("notification_integrations.SlackNotifier.send_notification", new_callable=AsyncMock)
@patch("notification_integrations.EmailNotifier.send_notification", new_callable=AsyncMock)
async def test_priority_levels(mock_email, mock_slack, mock_pagerduty, mock_webhook):
    """Test different priority levels with appropriate channel routing"""
    mock_email.return_value = True
    mock_slack.return_value = True
    mock_pagerduty.return_value = True
    mock_webhook.return_value = True
    manager = NotificationManager()

    priority_tests = [
        {
            "priority": NotificationPriority.LOW,
            "title": "Low Priority Test - System Information",
            "message": "This is a low priority notification test. System is operating normally.",
            "expected_channels": "Slack only",
        },
        {
            "priority": NotificationPriority.MEDIUM,
            "title": "Medium Priority Test - Performance Warning",
            "message": "This is a medium priority notification test. Performance metrics show elevated usage.",
            "expected_channels": "Email + Slack",
        },
        {
            "priority": NotificationPriority.HIGH,
            "title": "High Priority Test - Service Degradation",
            "message": "This is a high priority notification test. Service degradation detected.",
            "expected_channels": "Email + Slack + PagerDuty",
        },
        {
            "priority": NotificationPriority.CRITICAL,
            "title": "Critical Priority Test - System Failure",
            "message": "This is a critical priority notification test. Immediate attention required.",
            "expected_channels": "All channels",
        },
    ]

    for test in priority_tests:
        results = await manager.send_alert(
            title=test["title"],
            message=test["message"],
            priority=test["priority"],
            metadata={
                "test_type": "priority_routing",
                "priority_level": test["priority"].value,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )

        for _channel, _success in results.items():
            pass


@patch("notification_integrations.NotificationManager.send_alert", new_callable=AsyncMock)
async def test_concurrent_notifications(mock_send_alert):
    """Test sending multiple notifications concurrently"""
    mock_send_alert.return_value = {
        NotificationChannel.EMAIL: True,
        NotificationChannel.SLACK: True,
    }
    manager = NotificationManager()

    # Create multiple notification tasks
    tasks = []
    for i in range(5):
        task = manager.send_alert(
            title=f"Concurrent Test #{i + 1}",
            message=f"This is concurrent notification test #{i + 1} to verify system can handle multiple simultaneous notifications.",
            priority=NotificationPriority.MEDIUM,
            metadata={
                "test_type": "concurrent",
                "test_number": i + 1,
                "timestamp": datetime.now(UTC).isoformat(),
            },
        )
        tasks.append(task)

    # Execute all tasks concurrently
    start_time = datetime.now(UTC)
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = datetime.now(UTC)

    (end_time - start_time).total_seconds()

    # Analyze results
    success_count = 0
    for i, result in enumerate(results):
        if isinstance(result, dict):
            all_successful = all(result.values())
            if all_successful:
                success_count += 1
        else:
            pass


@patch("notification_integrations.EmailNotifier.send_notification", new_callable=AsyncMock)
async def test_error_handling(mock_email):
    """Test error handling with invalid configurations"""
    mock_email.return_value = False

    # Test with invalid configuration
    manager = NotificationManager()

    # Test with invalid email config

    result = await manager.send_alert(
        title="Error Handling Test - Invalid Email",
        message="This should fail gracefully with invalid email configuration.",
        priority=NotificationPriority.LOW,
        channels=[NotificationChannel.EMAIL],
    )

    not result.get(NotificationChannel.EMAIL, True)


@patch(
    "notification_integrations.NotificationManager.send_notification",
    new_callable=AsyncMock,
)
async def test_alert_grouping(mock_send_notification):
    """Test alert grouping functionality"""
    mock_send_notification.return_value = {NotificationChannel.SLACK: True}
    manager = NotificationManager()
    manager.alert_grouper.group_interval = timedelta(seconds=1)

    # Send a burst of similar alerts
    for i in range(5):
        await manager.send_alert(
            title="Grouped Alert Test",
            message=f"This is alert #{i + 1} in the group.",
            priority=NotificationPriority.MEDIUM,
        )

    await asyncio.sleep(2)  # Wait for the group to flush
    await manager.flush_groups()

    # Verify that send_notification was called once with a grouped message
    assert mock_send_notification.call_count == 1
    sent_notification = mock_send_notification.call_args[0][0]
    assert "[GROUPED]" in sent_notification.title
    assert "(x5)" in sent_notification.title


def print_configuration_status():
    """Print current configuration status"""

    config_items = [
        ("Email User", os.getenv("EMAIL_USER", "Not configured")),
        ("Email Recipients", os.getenv("EMAIL_RECIPIENTS", "Not configured")),
        (
            "Slack Webhook",
            "Configured" if os.getenv("SLACK_WEBHOOK_URL") else "Not configured",
        ),
        ("Slack Channel", os.getenv("SLACK_CHANNEL", "#alerts")),
        (
            "PagerDuty Key",
            "Configured" if os.getenv("PAGERDUTY_INTEGRATION_KEY") else "Not configured",
        ),
        (
            "Webhook URLs",
            "Configured" if os.getenv("WEBHOOK_URLS") else "Not configured",
        ),
    ]

    for _item, _status in config_items:
        pass


async def main():
    """Main testing function"""

    # Print configuration status
    print_configuration_status()

    # Run tests based on command line arguments
    if len(sys.argv) > 1:
        test_type = sys.argv[1].lower()

        if test_type == "channels":
            await test_individual_channels()
        elif test_type == "priorities":
            await test_priority_levels()
        elif test_type == "concurrent":
            await test_concurrent_notifications()
        elif test_type == "errors":
            await test_error_handling()
        elif test_type == "grouping":
            await test_alert_grouping()
        elif test_type == "all":
            await test_individual_channels()
            await test_priority_levels()
            await test_concurrent_notifications()
            await test_error_handling()
            await test_alert_grouping()
        else:
            pass
    else:
        # Run all tests by default
        await test_individual_channels()
        await test_priority_levels()
        await test_concurrent_notifications()
        await test_error_handling()
        await test_alert_grouping()


if __name__ == "__main__":
    asyncio.run(main())
