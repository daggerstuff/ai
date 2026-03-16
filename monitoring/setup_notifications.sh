#!/bin/bash

# Notification Integration Setup Script for Pixelated Empathy AI
# This script helps configure notification channels (Email, Slack, PagerDuty, Webhooks)

set -e

echo "🚀 Pixelated Empathy AI - Notification Integration Setup"
echo "======================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration file path
CONFIG_DIR="/home/vivi/pixelated/ai/monitoring"
ENV_FILE="$CONFIG_DIR/.env.notifications"
CONFIG_FILE="$CONFIG_DIR/notification_config.json"

# Create configuration directory if it doesn't exist
mkdir -p "$CONFIG_DIR"

echo -e "${BLUE}This script will help you configure notification integrations.${NC}"
echo -e "${YELLOW}You can skip any integration by pressing Enter without input.${NC}"
echo ""

# Function to prompt for input with default
prompt_input() {
    local prompt="$1"
    local default="$2"
    local secret="$3"
    
    if [ "$secret" = "true" ]; then
        echo -n -e "${BLUE}$prompt${NC}"
        [ -n "$default" ] && echo -n " (default: ***hidden***)"
        echo -n ": "
        read -s input
        echo ""
    else
        echo -n -e "${BLUE}$prompt${NC}"
        [ -n "$default" ] && echo -n " (default: $default)"
        echo -n ": "
        read input
    fi
    
    if [ -z "$input" ] && [ -n "$default" ]; then
        input="$default"
    fi
    
    echo "$input"
}

# Function to validate email
validate_email() {
    local email="$1"
    if [[ "$email" =~ ^[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Za-z]{2,}$ ]]; then
        return 0
    else
        return 1
    fi
}

# Function to validate URL
validate_url() {
    local url="$1"
    if [[ "$url" =~ ^https?:// ]]; then
        return 0
    else
        return 1
    fi
}

echo -e "${GREEN}📧 Email Configuration${NC}"
echo "Configure SMTP settings for email notifications"
echo ""

EMAIL_USER=$(prompt_input "Email address (sender)")
if [ -n "$EMAIL_USER" ]; then
    if ! validate_email "$EMAIL_USER"; then
        echo -e "${RED}Warning: Invalid email format${NC}"
    fi
    
    EMAIL_PASSWORD=$(prompt_input "Email password/app password" "" "true")
    SMTP_SERVER=$(prompt_input "SMTP server" "smtp.gmail.com")
    SMTP_PORT=$(prompt_input "SMTP port" "587")
    EMAIL_RECIPIENTS=$(prompt_input "Recipient emails (comma-separated)")
    
    echo -e "${GREEN}✅ Email configuration saved${NC}"
else
    echo -e "${YELLOW}⏭️  Email configuration skipped${NC}"
fi

echo ""
echo -e "${GREEN}💬 Slack Configuration${NC}"
echo "Configure Slack webhook for notifications"
echo ""

SLACK_WEBHOOK_URL=$(prompt_input "Slack webhook URL")
if [ -n "$SLACK_WEBHOOK_URL" ]; then
    if ! validate_url "$SLACK_WEBHOOK_URL"; then
        echo -e "${RED}Warning: Invalid webhook URL format${NC}"
    fi
    
    SLACK_CHANNEL=$(prompt_input "Slack channel" "#pixelated-ai-alerts")
    SLACK_TOKEN=$(prompt_input "Slack bot token (optional)" "" "true")
    
    echo -e "${GREEN}✅ Slack configuration saved${NC}"
else
    echo -e "${YELLOW}⏭️  Slack configuration skipped${NC}"
fi

echo ""
echo -e "${GREEN}🚨 PagerDuty Configuration${NC}"
echo "Configure PagerDuty for critical alerts"
echo ""

PAGERDUTY_INTEGRATION_KEY=$(prompt_input "PagerDuty integration key" "" "true")
if [ -n "$PAGERDUTY_INTEGRATION_KEY" ]; then
    PAGERDUTY_API_TOKEN=$(prompt_input "PagerDuty API token (optional)" "" "true")
    
    echo -e "${GREEN}✅ PagerDuty configuration saved${NC}"
else
    echo -e "${YELLOW}⏭️  PagerDuty configuration skipped${NC}"
fi

echo ""
echo -e "${GREEN}🔗 Webhook Configuration${NC}"
echo "Configure additional webhook endpoints"
echo ""

WEBHOOK_URLS=""
webhook_count=1
while true; do
    webhook_url=$(prompt_input "Webhook URL #$webhook_count (Enter to finish)")
    
    if [ -z "$webhook_url" ]; then
        break
    fi
    
    if ! validate_url "$webhook_url"; then
        echo -e "${RED}Warning: Invalid webhook URL format${NC}"
    fi
    
    if [ -z "$WEBHOOK_URLS" ]; then
        WEBHOOK_URLS="$webhook_url"
    else
        WEBHOOK_URLS="$WEBHOOK_URLS,$webhook_url"
    fi
    
    webhook_count=$((webhook_count + 1))
done

if [ -n "$WEBHOOK_URLS" ]; then
    echo -e "${GREEN}✅ Webhook configuration saved${NC}"
else
    echo -e "${YELLOW}⏭️  Webhook configuration skipped${NC}"
fi

# Create environment file
echo ""
echo -e "${BLUE}💾 Saving configuration...${NC}"

cat > "$ENV_FILE" << EOF
# Pixelated Empathy AI - Notification Configuration
# Generated on $(date)

# Email Configuration
EMAIL_USER="$EMAIL_USER"
EMAIL_PASSWORD="$EMAIL_PASSWORD"
SMTP_SERVER="$SMTP_SERVER"
SMTP_PORT="$SMTP_PORT"
EMAIL_RECIPIENTS="$EMAIL_RECIPIENTS"

# Slack Configuration
SLACK_WEBHOOK_URL="$SLACK_WEBHOOK_URL"
SLACK_CHANNEL="$SLACK_CHANNEL"
SLACK_TOKEN="$SLACK_TOKEN"

# PagerDuty Configuration
PAGERDUTY_INTEGRATION_KEY="$PAGERDUTY_INTEGRATION_KEY"
PAGERDUTY_API_TOKEN="$PAGERDUTY_API_TOKEN"

# Webhook Configuration
WEBHOOK_URLS="$WEBHOOK_URLS"
EOF

# Create JSON configuration file
cat > "$CONFIG_FILE" << EOF
{
  "smtp_server": "$SMTP_SERVER",
  "smtp_port": ${SMTP_PORT:-587},
  "email_user": "$EMAIL_USER",
  "email_password": "$EMAIL_PASSWORD",
  "default_recipients": [$(echo "$EMAIL_RECIPIENTS" | sed 's/,/", "/g' | sed 's/^/"/' | sed 's/$/"/')],
  
  "slack_webhook_url": "$SLACK_WEBHOOK_URL",
  "slack_token": "$SLACK_TOKEN",
  "slack_channel": "$SLACK_CHANNEL",
  
  "pagerduty_integration_key": "$PAGERDUTY_INTEGRATION_KEY",
  "pagerduty_api_token": "$PAGERDUTY_API_TOKEN",
  
  "webhook_urls": [$(echo "$WEBHOOK_URLS" | sed 's/,/", "/g' | sed 's/^/"/' | sed 's/$/"/')],
  
  "_metadata": {
    "created": "$(date -u +"%Y-%m-%dT%H:%M:%SZ")",
    "version": "1.0"
  }
}
EOF

# Set appropriate permissions
chmod 600 "$ENV_FILE"
chmod 600 "$CONFIG_FILE"

echo -e "${GREEN}✅ Configuration files created:${NC}"
echo "  Environment file: $ENV_FILE"
echo "  JSON config file: $CONFIG_FILE"
echo ""

# Test configuration
echo -e "${BLUE}🧪 Testing configuration...${NC}"

# Source the environment file
set -a
source "$ENV_FILE"
set +a

# Check if Python dependencies are available
if command -v python3 &> /dev/null; then
    if python3 -c "import aiohttp, asyncio" &> /dev/null; then
        echo -e "${GREEN}✅ Python dependencies available${NC}"
        
        # Ask if user wants to run tests
        echo ""
        echo -n -e "${BLUE}Would you like to run notification tests now? (y/N): ${NC}"
        read run_tests
        
        if [[ "$run_tests" =~ ^[Yy]$ ]]; then
            echo ""
            echo -e "${BLUE}Running notification tests...${NC}"
            cd "$CONFIG_DIR"
            python3 test_notifications.py channels
        fi
    else
        echo -e "${YELLOW}⚠️  Python dependencies missing. Install with:${NC}"
        echo "  pip install aiohttp asyncio"
    fi
else
    echo -e "${YELLOW}⚠️  Python3 not found${NC}"
fi

echo ""
echo -e "${GREEN}🎉 Notification setup complete!${NC}"
echo ""
echo -e "${BLUE}Next steps:${NC}"
echo "1. Source the environment file: source $ENV_FILE"
echo "2. Test notifications: python3 $CONFIG_DIR/test_notifications.py"
echo "3. Integrate with monitoring: python3 $CONFIG_DIR/notification_integrations.py"
echo ""
echo -e "${BLUE}Usage examples:${NC}"
echo "  # Test all channels"
echo "  python3 test_notifications.py all"
echo ""
echo "  # Test specific priority levels"
echo "  python3 test_notifications.py priorities"
echo ""
echo "  # Send a test alert"
echo "  python3 -c \"import asyncio; from notification_integrations import *; asyncio.run(NotificationManager().send_alert('Test Alert', 'This is a test message', NotificationPriority.MEDIUM))\""

# Create a quick reference file
cat > "$CONFIG_DIR/notification_quick_reference.md" << EOF
# Notification Integration Quick Reference

## Configuration Files
- Environment: \`$ENV_FILE\`
- JSON Config: \`$CONFIG_FILE\`

## Testing Commands
\`\`\`bash
# Test all notification channels
python3 test_notifications.py all

# Test individual channels
python3 test_notifications.py channels

# Test priority routing
python3 test_notifications.py priorities

# Test concurrent notifications
python3 test_notifications.py concurrent

# Test error handling
python3 test_notifications.py errors
\`\`\`

## Usage in Code
\`\`\`python
from notification_integrations import NotificationManager, NotificationPriority

# Initialize manager
manager = NotificationManager()

# Send alert
await manager.send_alert(
    title="System Alert",
    message="Description of the issue",
    priority=NotificationPriority.HIGH,
    metadata={"service": "api", "error_count": 5}
)
\`\`\`

## Priority Levels and Channel Routing
- **LOW**: Slack only
- **MEDIUM**: Email + Slack  
- **HIGH**: Email + Slack + PagerDuty
- **CRITICAL**: All channels

## Environment Variables
$(cat "$ENV_FILE" | grep -E "^[A-Z_]+=" | sed 's/=.*//' | sort)

Generated on $(date)
EOF

echo -e "${GREEN}📚 Quick reference created: $CONFIG_DIR/notification_quick_reference.md${NC}"
