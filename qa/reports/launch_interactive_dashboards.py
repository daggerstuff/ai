#!/usr/bin/env python3
import sys



sys.path.append(str(_PROJECT_ROOT / "ai"))

from monitoring.interactive_dashboard_system import InteractiveDashboardSystem
from pathlib import Path
_PROJECT_ROOT = Path(__file__).resolve().parents[3]

if __name__ == "__main__":
    dashboard = InteractiveDashboardSystem()
    dashboard.setup_flask_routes()
    dashboard.app.run(host="0.0.0.0", port=5000, debug=False)
