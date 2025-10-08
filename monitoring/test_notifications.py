#!/usr/bin/env python3
"""
Notification System Testing Script
Tests all notification channels and priority levels
"""

import asyncio
import sys
import os
from datetime import datetime
from notification_integrations import (
    NotificationManager, 
    NotificationPriority, 
    NotificationChannel
)

async def test_individual_channels():
    """Test each notification channel individually"""
    manager = NotificationManager()
    
    print("🧪 Testing Individual Notification Channels")
    print("=" * 50)
    
    # Test Email
    print("\n📧 Testing Email Notifications...")
    email_result = await manager.send_alert(
        title="Email Test - Pixelated AI Monitoring",
        message="This is a test email notification from the Pixelated Empathy AI monitoring system.",
        priority=NotificationPriority.LOW,
        channels=[NotificationChannel.EMAIL],
        metadata={
            "test_type": "email_channel",
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "testing"
        }
    )
    print(f"Email Result: {'✅ SUCCESS' if email_result.get(NotificationChannel.EMAIL) else '❌ FAILED'}")
    
    # Test Slack
    print("\n💬 Testing Slack Notifications...")
    slack_result = await manager.send_alert(
        title="Slack Test - Pixelated AI Monitoring",
        message="This is a test Slack notification from the Pixelated Empathy AI monitoring system.",
        priority=NotificationPriority.MEDIUM,
        channels=[NotificationChannel.SLACK],
        metadata={
            "test_type": "slack_channel",
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "testing"
        }
    )
    print(f"Slack Result: {'✅ SUCCESS' if slack_result.get(NotificationChannel.SLACK) else '❌ FAILED'}")
    
    # Test PagerDuty
    print("\n🚨 Testing PagerDuty Notifications...")
    pagerduty_result = await manager.send_alert(
        title="PagerDuty Test - Pixelated AI Monitoring",
        message="This is a test PagerDuty notification from the Pixelated Empathy AI monitoring system.",
        priority=NotificationPriority.HIGH,
        channels=[NotificationChannel.PAGERDUTY],
        metadata={
            "test_type": "pagerduty_channel",
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "testing"
        }
    )
    print(f"PagerDuty Result: {'✅ SUCCESS' if pagerduty_result.get(NotificationChannel.PAGERDUTY) else '❌ FAILED'}")
    
    # Test Webhooks
    print("\n🔗 Testing Webhook Notifications...")
    webhook_result = await manager.send_alert(
        title="Webhook Test - Pixelated AI Monitoring",
        message="This is a test webhook notification from the Pixelated Empathy AI monitoring system.",
        priority=NotificationPriority.MEDIUM,
        channels=[NotificationChannel.WEBHOOK],
        metadata={
            "test_type": "webhook_channel",
            "timestamp": datetime.utcnow().isoformat(),
            "system_status": "testing"
        }
    )
    print(f"Webhook Result: {'✅ SUCCESS' if webhook_result.get(NotificationChannel.WEBHOOK) else '❌ FAILED'}")

async def test_priority_levels():
    """Test different priority levels with appropriate channel routing"""
    manager = NotificationManager()
    
    print("\n\n🎯 Testing Priority Level Routing")
    print("=" * 50)
    
    priority_tests = [
        {
            "priority": NotificationPriority.LOW,
            "title": "Low Priority Test - System Information",
            "message": "This is a low priority notification test. System is operating normally.",
            "expected_channels": "Slack only"
        },
        {
            "priority": NotificationPriority.MEDIUM,
            "title": "Medium Priority Test - Performance Warning",
            "message": "This is a medium priority notification test. Performance metrics show elevated usage.",
            "expected_channels": "Email + Slack"
        },
        {
            "priority": NotificationPriority.HIGH,
            "title": "High Priority Test - Service Degradation",
            "message": "This is a high priority notification test. Service degradation detected.",
            "expected_channels": "Email + Slack + PagerDuty"
        },
        {
            "priority": NotificationPriority.CRITICAL,
            "title": "Critical Priority Test - System Failure",
            "message": "This is a critical priority notification test. Immediate attention required.",
            "expected_channels": "All channels"
        }
    ]
    
    for test in priority_tests:
        print(f"\n🔔 Testing {test['priority'].value.upper()} Priority")
        print(f"Expected Channels: {test['expected_channels']}")
        
        results = await manager.send_alert(
            title=test['title'],
            message=test['message'],
            priority=test['priority'],
            metadata={
                "test_type": "priority_routing",
                "priority_level": test['priority'].value,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        
        print("Results:")
        for channel, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {channel.value}: {status}")

async def test_concurrent_notifications():
    """Test sending multiple notifications concurrently"""
    manager = NotificationManager()
    
    print("\n\n⚡ Testing Concurrent Notifications")
    print("=" * 50)
    
    # Create multiple notification tasks
    tasks = []
    for i in range(5):
        task = manager.send_alert(
            title=f"Concurrent Test #{i+1}",
            message=f"This is concurrent notification test #{i+1} to verify system can handle multiple simultaneous notifications.",
            priority=NotificationPriority.MEDIUM,
            metadata={
                "test_type": "concurrent",
                "test_number": i+1,
                "timestamp": datetime.utcnow().isoformat()
            }
        )
        tasks.append(task)
    
    # Execute all tasks concurrently
    start_time = datetime.utcnow()
    results = await asyncio.gather(*tasks, return_exceptions=True)
    end_time = datetime.utcnow()
    
    duration = (end_time - start_time).total_seconds()
    print(f"Sent 5 concurrent notifications in {duration:.2f} seconds")
    
    # Analyze results
    success_count = 0
    for i, result in enumerate(results):
        if isinstance(result, dict):
            all_successful = all(result.values())
            if all_successful:
                success_count += 1
            print(f"Notification #{i+1}: {'✅ SUCCESS' if all_successful else '❌ FAILED'}")
        else:
            print(f"Notification #{i+1}: ❌ EXCEPTION - {result}")
    
    print(f"\nOverall Success Rate: {success_count}/5 ({success_count/5*100:.1f}%)")

async def test_error_handling():
    """Test error handling with invalid configurations"""
    print("\n\n🛡️ Testing Error Handling")
    print("=" * 50)
    
    # Test with invalid configuration
    manager = NotificationManager()
    
    # Temporarily break configuration to test error handling
    original_config = manager.config
    
    # Test with invalid email config
    print("\n📧 Testing Email Error Handling...")
    manager.config.email_user = "invalid-email"
    manager.config.email_password = "invalid-password"
    
    result = await manager.send_alert(
        title="Error Handling Test - Invalid Email",
        message="This should fail gracefully with invalid email configuration.",
        priority=NotificationPriority.LOW,
        channels=[NotificationChannel.EMAIL]
    )
    
    email_failed = not result.get(NotificationChannel.EMAIL, True)
    print(f"Email Error Handling: {'✅ HANDLED GRACEFULLY' if email_failed else '❌ UNEXPECTED SUCCESS'}")
    
    # Restore original configuration
    manager.config = original_config

def print_configuration_status():
    """Print current configuration status"""
    print("🔧 Configuration Status")
    print("=" * 50)
    
    config_items = [
        ("Email User", os.getenv('EMAIL_USER', 'Not configured')),
        ("Email Recipients", os.getenv('EMAIL_RECIPIENTS', 'Not configured')),
        ("Slack Webhook", "Configured" if os.getenv('SLACK_WEBHOOK_URL') else "Not configured"),
        ("Slack Channel", os.getenv('SLACK_CHANNEL', '#alerts')),
        ("PagerDuty Key", "Configured" if os.getenv('PAGERDUTY_INTEGRATION_KEY') else "Not configured"),
        ("Webhook URLs", "Configured" if os.getenv('WEBHOOK_URLS') else "Not configured")
    ]
    
    for item, status in config_items:
        print(f"{item}: {status}")

async def main():
    """Main testing function"""
    print("🚀 Pixelated Empathy AI - Notification System Test Suite")
    print("=" * 60)
    
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
        elif test_type == "all":
            await test_individual_channels()
            await test_priority_levels()
            await test_concurrent_notifications()
            await test_error_handling()
        else:
            print(f"\nUnknown test type: {test_type}")
            print("Available tests: channels, priorities, concurrent, errors, all")
    else:
        # Run all tests by default
        await test_individual_channels()
        await test_priority_levels()
        await test_concurrent_notifications()
        await test_error_handling()
    
    print("\n✅ Testing Complete!")
    print("\nTo run specific tests:")
    print("  python test_notifications.py channels    # Test individual channels")
    print("  python test_notifications.py priorities  # Test priority routing")
    print("  python test_notifications.py concurrent  # Test concurrent sending")
    print("  python test_notifications.py errors      # Test error handling")
    print("  python test_notifications.py all         # Run all tests")

if __name__ == "__main__":
    asyncio.run(main())
