#!/usr/bin/env python3
"""
Notification Integration System for Pixelated Empathy AI
Supports email, Slack, and PagerDuty notifications with intelligent routing
"""

import os
import json
import smtplib
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart
from dataclasses import dataclass
import requests
import asyncio
import aiohttp
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class NotificationPriority(Enum):
    """Notification priority levels"""
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"
    CRITICAL = "critical"

class NotificationChannel(Enum):
    """Available notification channels"""
    EMAIL = "email"
    SLACK = "slack"
    PAGERDUTY = "pagerduty"
    WEBHOOK = "webhook"

@dataclass
class NotificationConfig:
    """Configuration for notification channels"""
    # Email configuration
    smtp_server: str = "smtp.gmail.com"
    smtp_port: int = 587
    email_user: str = ""
    email_password: str = ""
    default_recipients: List[str] = None
    
    # Slack configuration
    slack_webhook_url: str = ""
    slack_token: str = ""
    slack_channel: str = "#alerts"
    
    # PagerDuty configuration
    pagerduty_integration_key: str = ""
    pagerduty_api_token: str = ""
    
    # Webhook configuration
    webhook_urls: List[str] = None
    
    def __post_init__(self):
        if self.default_recipients is None:
            self.default_recipients = []
        if self.webhook_urls is None:
            self.webhook_urls = []

@dataclass
class NotificationMessage:
    """Notification message structure"""
    title: str
    message: str
    priority: NotificationPriority
    channels: List[NotificationChannel]
    metadata: Dict[str, Any] = None
    timestamp: datetime = None
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.utcnow()
        if self.metadata is None:
            self.metadata = {}

class EmailNotifier:
    """Email notification handler"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        
    async def send_notification(self, message: NotificationMessage) -> bool:
        """Send email notification"""
        try:
            # Create message
            msg = MIMEMultipart()
            msg['From'] = self.config.email_user
            msg['To'] = ', '.join(self.config.default_recipients)
            msg['Subject'] = f"[{message.priority.value.upper()}] {message.title}"
            
            # Create email body
            body = self._create_email_body(message)
            msg.attach(MIMEText(body, 'html'))
            
            # Send email
            server = smtplib.SMTP(self.config.smtp_server, self.config.smtp_port)
            server.starttls()
            server.login(self.config.email_user, self.config.email_password)
            
            text = msg.as_string()
            server.sendmail(self.config.email_user, self.config.default_recipients, text)
            server.quit()
            
            logger.info(f"Email notification sent: {message.title}")
            return True
            
        except Exception as e:
            logger.error(f"Failed to send email notification: {e}")
            return False
    
    def _create_email_body(self, message: NotificationMessage) -> str:
        """Create HTML email body"""
        priority_colors = {
            NotificationPriority.LOW: "#28a745",
            NotificationPriority.MEDIUM: "#ffc107", 
            NotificationPriority.HIGH: "#fd7e14",
            NotificationPriority.CRITICAL: "#dc3545"
        }
        
        color = priority_colors.get(message.priority, "#6c757d")
        
        return f"""
        <html>
        <body>
            <div style="font-family: Arial, sans-serif; max-width: 600px;">
                <div style="background-color: {color}; color: white; padding: 15px; border-radius: 5px 5px 0 0;">
                    <h2 style="margin: 0;">{message.title}</h2>
                    <p style="margin: 5px 0 0 0;">Priority: {message.priority.value.upper()}</p>
                </div>
                <div style="border: 1px solid #ddd; border-top: none; padding: 20px; border-radius: 0 0 5px 5px;">
                    <p><strong>Message:</strong></p>
                    <p>{message.message}</p>
                    
                    <p><strong>Timestamp:</strong> {message.timestamp.isoformat()}</p>
                    
                    {self._format_metadata(message.metadata)}
                </div>
            </div>
        </body>
        </html>
        """
    
    def _format_metadata(self, metadata: Dict[str, Any]) -> str:
        """Format metadata for email display"""
        if not metadata:
            return ""
        
        items = []
        for key, value in metadata.items():
            items.append(f"<li><strong>{key}:</strong> {value}</li>")
        
        return f"<p><strong>Additional Information:</strong></p><ul>{''.join(items)}</ul>"

class SlackNotifier:
    """Slack notification handler"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        
    async def send_notification(self, message: NotificationMessage) -> bool:
        """Send Slack notification"""
        try:
            payload = self._create_slack_payload(message)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    self.config.slack_webhook_url,
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 200:
                        logger.info(f"Slack notification sent: {message.title}")
                        return True
                    else:
                        logger.error(f"Slack notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send Slack notification: {e}")
            return False
    
    def _create_slack_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """Create Slack message payload"""
        priority_colors = {
            NotificationPriority.LOW: "good",
            NotificationPriority.MEDIUM: "warning",
            NotificationPriority.HIGH: "danger",
            NotificationPriority.CRITICAL: "danger"
        }
        
        color = priority_colors.get(message.priority, "good")
        
        fields = [
            {
                "title": "Priority",
                "value": message.priority.value.upper(),
                "short": True
            },
            {
                "title": "Timestamp",
                "value": message.timestamp.strftime("%Y-%m-%d %H:%M:%S UTC"),
                "short": True
            }
        ]
        
        # Add metadata fields
        for key, value in message.metadata.items():
            fields.append({
                "title": key.replace("_", " ").title(),
                "value": str(value),
                "short": True
            })
        
        return {
            "channel": self.config.slack_channel,
            "username": "Pixelated AI Monitor",
            "icon_emoji": ":robot_face:",
            "attachments": [
                {
                    "color": color,
                    "title": message.title,
                    "text": message.message,
                    "fields": fields,
                    "footer": "Pixelated Empathy AI",
                    "ts": int(message.timestamp.timestamp())
                }
            ]
        }

class PagerDutyNotifier:
    """PagerDuty notification handler"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        
    async def send_notification(self, message: NotificationMessage) -> bool:
        """Send PagerDuty notification"""
        try:
            # Only send to PagerDuty for HIGH and CRITICAL priorities
            if message.priority not in [NotificationPriority.HIGH, NotificationPriority.CRITICAL]:
                logger.info(f"Skipping PagerDuty for {message.priority.value} priority")
                return True
            
            payload = self._create_pagerduty_payload(message)
            
            async with aiohttp.ClientSession() as session:
                async with session.post(
                    "https://events.pagerduty.com/v2/enqueue",
                    json=payload,
                    headers={'Content-Type': 'application/json'}
                ) as response:
                    if response.status == 202:
                        logger.info(f"PagerDuty notification sent: {message.title}")
                        return True
                    else:
                        logger.error(f"PagerDuty notification failed: {response.status}")
                        return False
                        
        except Exception as e:
            logger.error(f"Failed to send PagerDuty notification: {e}")
            return False
    
    def _create_pagerduty_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """Create PagerDuty event payload"""
        severity = "critical" if message.priority == NotificationPriority.CRITICAL else "error"
        
        return {
            "routing_key": self.config.pagerduty_integration_key,
            "event_action": "trigger",
            "payload": {
                "summary": message.title,
                "source": "pixelated-empathy-ai",
                "severity": severity,
                "component": "monitoring",
                "group": "ai-processing",
                "class": "system-alert",
                "custom_details": {
                    "message": message.message,
                    "priority": message.priority.value,
                    "timestamp": message.timestamp.isoformat(),
                    **message.metadata
                }
            }
        }

class WebhookNotifier:
    """Generic webhook notification handler"""
    
    def __init__(self, config: NotificationConfig):
        self.config = config
        
    async def send_notification(self, message: NotificationMessage) -> bool:
        """Send webhook notifications"""
        success_count = 0
        
        for webhook_url in self.config.webhook_urls:
            try:
                payload = self._create_webhook_payload(message)
                
                async with aiohttp.ClientSession() as session:
                    async with session.post(
                        webhook_url,
                        json=payload,
                        headers={'Content-Type': 'application/json'}
                    ) as response:
                        if response.status in [200, 201, 202]:
                            success_count += 1
                            logger.info(f"Webhook notification sent to {webhook_url}")
                        else:
                            logger.error(f"Webhook notification failed for {webhook_url}: {response.status}")
                            
            except Exception as e:
                logger.error(f"Failed to send webhook notification to {webhook_url}: {e}")
        
        return success_count > 0
    
    def _create_webhook_payload(self, message: NotificationMessage) -> Dict[str, Any]:
        """Create webhook payload"""
        return {
            "title": message.title,
            "message": message.message,
            "priority": message.priority.value,
            "timestamp": message.timestamp.isoformat(),
            "metadata": message.metadata,
            "source": "pixelated-empathy-ai"
        }

class AlertGrouper:
    """Groups similar alerts to reduce notification fatigue"""

    def __init__(self, notification_manager, group_interval_seconds=300, max_group_size=10):
        self.notification_manager = notification_manager
        self.group_interval = timedelta(seconds=group_interval_seconds)
        self.max_group_size = max_group_size
        self.alert_buffer: Dict[str, List[NotificationMessage]] = {}
        self.group_creation_time: Dict[str, datetime] = {}

    def get_group_key(self, message: NotificationMessage) -> str:
        """Generate a key to group similar alerts"""
        # Group by title and priority
        return f"{message.title}-{message.priority.value}"

    async def add_alert(self, message: NotificationMessage):
        """Add an alert to the buffer and process groups"""
        group_key = self.get_group_key(message)

        if group_key not in self.alert_buffer:
            self.alert_buffer[group_key] = []
            self.group_creation_time[group_key] = datetime.utcnow()

        self.alert_buffer[group_key].append(message)

        if len(self.alert_buffer[group_key]) >= self.max_group_size:
            await self.flush_group(group_key)

    async def flush_expired_groups(self):
        """Flush groups that have exceeded the grouping interval"""
        now = datetime.utcnow()
        expired_groups = [
            key for key, created_time in self.group_creation_time.items()
            if now - created_time > self.group_interval
        ]
        for group_key in expired_groups:
            await self.flush_group(group_key)

    async def flush_group(self, group_key: str):
        """Send a grouped notification for the specified group"""
        if group_key not in self.alert_buffer:
            return

        alerts = self.alert_buffer.pop(group_key)
        self.group_creation_time.pop(group_key)

        if not alerts:
            return

        if len(alerts) == 1:
            # Send single alert as is
            await self.notification_manager.send_notification(alerts[0])
            return

        # Create a summary notification
        first_alert = alerts[0]
        group_size = len(alerts)
        
        summary_title = f"[GROUPED] {first_alert.title} (x{group_size})"
        
        summary_message = f"This alert was triggered {group_size} times in the last {self.group_interval.seconds // 60} minutes.\n\n"
        summary_message += f"First occurrence: {first_alert.timestamp.isoformat()}\n"
        summary_message += f"Last occurrence: {alerts[-1].timestamp.isoformat()}\n\n"
        summary_message += "Summary of alerts:\n"
        for alert in alerts:
            summary_message += f"- {alert.message} at {alert.timestamp.isoformat()}\n"

        summary_notification = NotificationMessage(
            title=summary_title,
            message=summary_message,
            priority=first_alert.priority,
            channels=first_alert.channels,
            metadata={
                "group_key": group_key,
                "group_size": group_size,
                "start_time": first_alert.timestamp.isoformat(),
                "end_time": alerts[-1].timestamp.isoformat(),
            }
        )

        await self.notification_manager.send_notification(summary_notification)
        logger.info(f"Sent grouped notification for: {group_key}")

class NotificationManager:
    """Main notification management system"""
    
    def __init__(self, config_path: Optional[str] = None):
        self.config = self._load_config(config_path)
        self.notifiers = self._initialize_notifiers()
        self.alert_grouper = AlertGrouper(self)
        
    def _load_config(self, config_path: Optional[str]) -> NotificationConfig:
        """Load notification configuration"""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r') as f:
                config_data = json.load(f)
            return NotificationConfig(**config_data)
        
        # Load from environment variables
        return NotificationConfig(
            smtp_server=os.getenv('SMTP_SERVER', 'smtp.gmail.com'),
            smtp_port=int(os.getenv('SMTP_PORT', '587')),
            email_user=os.getenv('EMAIL_USER', ''),
            email_password=os.getenv('EMAIL_PASSWORD', ''),
            default_recipients=os.getenv('EMAIL_RECIPIENTS', '').split(',') if os.getenv('EMAIL_RECIPIENTS') else [],
            slack_webhook_url=os.getenv('SLACK_WEBHOOK_URL', ''),
            slack_token=os.getenv('SLACK_TOKEN', ''),
            slack_channel=os.getenv('SLACK_CHANNEL', '#alerts'),
            pagerduty_integration_key=os.getenv('PAGERDUTY_INTEGRATION_KEY', ''),
            pagerduty_api_token=os.getenv('PAGERDUTY_API_TOKEN', ''),
            webhook_urls=os.getenv('WEBHOOK_URLS', '').split(',') if os.getenv('WEBHOOK_URLS') else []
        )
    
    def _initialize_notifiers(self) -> Dict[NotificationChannel, Any]:
        """Initialize notification handlers"""
        return {
            NotificationChannel.EMAIL: EmailNotifier(self.config),
            NotificationChannel.SLACK: SlackNotifier(self.config),
            NotificationChannel.PAGERDUTY: PagerDutyNotifier(self.config),
            NotificationChannel.WEBHOOK: WebhookNotifier(self.config)
        }
    
    async def send_notification(self, message: NotificationMessage) -> Dict[NotificationChannel, bool]:
        """Send notification through specified channels"""
        results = {}
        
        # Send notifications concurrently
        tasks = []
        for channel in message.channels:
            if channel in self.notifiers:
                notifier = self.notifiers[channel]
                task = asyncio.create_task(notifier.send_notification(message))
                tasks.append((channel, task))
        
        # Wait for all notifications to complete
        for channel, task in tasks:
            try:
                result = await task
                results[channel] = result
            except Exception as e:
                logger.error(f"Notification failed for {channel.value}: {e}")
                results[channel] = False
        
        return results
    
    async def send_alert(self, 
                        title: str, 
                        message: str, 
                        priority: NotificationPriority = NotificationPriority.MEDIUM,
                        channels: Optional[List[NotificationChannel]] = None,
                        metadata: Optional[Dict[str, Any]] = None) -> Dict[NotificationChannel, bool]:
        """Convenience method to send alerts"""
        
        if channels is None:
            # Default channel selection based on priority
            if priority == NotificationPriority.CRITICAL:
                channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY]
            elif priority == NotificationPriority.HIGH:
                channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK, NotificationChannel.PAGERDUTY]
            elif priority == NotificationPriority.MEDIUM:
                channels = [NotificationChannel.EMAIL, NotificationChannel.SLACK]
            else:
                channels = [NotificationChannel.SLACK]
        
        notification = NotificationMessage(
            title=title,
            message=message,
            priority=priority,
            channels=channels,
            metadata=metadata or {}
        )
        
        await self.alert_grouper.add_alert(notification)
        return {channel: True for channel in channels} # Return True since it's handled by the grouper

    async def flush_groups(self):
        """Flush all expired alert groups"""
        await self.alert_grouper.flush_expired_groups()

# Example usage and testing
async def test_notifications():
    """Test notification system"""
    manager = NotificationManager()
    
    # Test different priority levels
    test_cases = [
        {
            "title": "System Health Check",
            "message": "All systems operating normally",
            "priority": NotificationPriority.LOW,
            "metadata": {"cpu_usage": "45%", "memory_usage": "62%"}
        },
        {
            "title": "High Memory Usage Detected",
            "message": "Memory usage has exceeded 85% threshold",
            "priority": NotificationPriority.MEDIUM,
            "metadata": {"memory_usage": "87%", "affected_service": "data_processor"}
        },
        {
            "title": "Processing Queue Backup",
            "message": "Processing queue has over 10,000 pending items",
            "priority": NotificationPriority.HIGH,
            "metadata": {"queue_size": 12543, "estimated_delay": "2 hours"}
        },
        {
            "title": "System Failure - Immediate Action Required",
            "message": "Critical system failure detected. Multiple services are down.",
            "priority": NotificationPriority.CRITICAL,
            "metadata": {"failed_services": ["api", "processor", "database"], "downtime": "5 minutes"}
        }
    ]
    
    for test_case in test_cases:
        print(f"\nTesting {test_case['priority'].value} priority notification...")
        results = await manager.send_alert(**test_case)
        
        for channel, success in results.items():
            status = "✅ SUCCESS" if success else "❌ FAILED"
            print(f"  {channel.value}: {status}")

if __name__ == "__main__":
    # Run tests
    asyncio.run(test_notifications())
