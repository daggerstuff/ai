#!/usr/bin/env python3
import sys

sys.path.append("/home/vivi/pixelated/ai")

from monitoring.interactive_dashboard_system import InteractiveDashboardSystem

if __name__ == "__main__":
    dashboard = InteractiveDashboardSystem()
    dashboard.setup_flask_routes()
    dashboard.app.run(host="0.0.0.0", port=5000, debug=False)
