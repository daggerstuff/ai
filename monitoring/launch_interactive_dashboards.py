#!/usr/bin/env python3
import sys
sys.path.append('/home/vivi/pixelated/ai')
from monitoring.interactive_dashboard_system import InteractiveDashboardSystem

if __name__ == '__main__':
    dashboard = InteractiveDashboardSystem()
    dashboard.setup_flask_routes()
    print("🌐 Interactive Dashboard Server Starting...")
    print("📊 Access your dashboards at:")
    print("   • Executive Dashboard: http://localhost:5000/executive")
    print("   • Operational Dashboard: http://localhost:5000/operational") 
    print("   • Technical Dashboard: http://localhost:5000/technical")
    print("\n🔄 Dashboards will auto-refresh every 30 seconds")
    print("🛑 Press Ctrl+C to stop the server")
    dashboard.app.run(host='0.0.0.0', port=5000, debug=False)
