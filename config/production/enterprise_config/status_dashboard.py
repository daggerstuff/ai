#!/usr/bin/env python3
"""
Enterprise Status Dashboard

Quick status overview of all enterprise components.
"""

import sys
from pathlib import Path

# Add enterprise config to path
sys.path.append(str(Path(__file__).parent))

from enterprise_config import get_config
from enterprise_logging import get_logger
from enterprise_monitoring import get_monitor
from enterprise_error_handling import get_error_handler

def show_status():
    """Show comprehensive system status."""
    print("🏢 PIXELATED AI ENTERPRISE STATUS DASHBOARD")
    print("=" * 60)
    
    # Configuration status
    try:
        config = get_config()
        print(f"⚙️ Configuration: ✅ Loaded ({config.environment})")
    except Exception as e:
        print(f"⚙️ Configuration: ❌ Error - {e}")
    
    # Monitoring status
    try:
        monitor = get_monitor()
        status = monitor.get_system_status()
        print(f"📊 System Status: {status['overall_status'].upper()}")
        
        for name, check in status['health_checks'].items():
            status_icon = "✅" if check['status'] == 'healthy' else "⚠️" if check['status'] == 'degraded' else "❌"
            print(f"   {status_icon} {name}: {check['message']}")
        
        # System metrics
        metrics = status['system_metrics']
        print(f"💻 System Metrics:")
        print(f"   CPU: {metrics.get('cpu_percent', 0):.1f}%")
        print(f"   Memory: {metrics.get('memory_percent', 0):.1f}%")
        print(f"   Disk: {metrics.get('disk_percent', 0):.1f}%")
        
    except Exception as e:
        print(f"📊 Monitoring: ❌ Error - {e}")
    
    # Error handling status
    try:
        error_handler = get_error_handler()
        error_health = error_handler.health_check()
        print(f"🛡️ Error Handling: {error_health['status'].upper()}")
        print(f"   Recent errors: {error_health['statistics']['total_errors']}")
    except Exception as e:
        print(f"🛡️ Error Handling: ❌ Error - {e}")
    
    print("=" * 60)
    print("✅ Enterprise baseline operational")

if __name__ == "__main__":
    show_status()
