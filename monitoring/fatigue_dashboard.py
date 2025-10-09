#!/usr/bin/env python3
"""
Alert Fatigue Dashboard and Management Interface
Web-based dashboard for monitoring and managing alert fatigue prevention
"""

import asyncio
import json
import logging
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Any
from flask import Flask, render_template_string, jsonify, request, redirect, url_for
import sqlite3
import plotly.graph_objs as go
import plotly.utils
from alert_fatigue_prevention import AlertFatiguePreventionSystem, FatigueRule
from intelligent_grouping import IntelligentGroupingEngine

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class FatigueDashboard:
    """Web dashboard for alert fatigue management"""
    
    def __init__(self, afp_system: AlertFatiguePreventionSystem):
        self.afp_system = afp_system
        self.grouping_engine = IntelligentGroupingEngine()
        self.app = Flask(__name__)
        self.setup_routes()
    
    def setup_routes(self):
        """Setup Flask routes for the dashboard"""
        
        @self.app.route('/')
        def dashboard():
            return render_template_string(DASHBOARD_TEMPLATE)
        
        @self.app.route('/api/summary')
        def api_summary():
            """Get dashboard summary data"""
            try:
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                # Get summary data
                summary = loop.run_until_complete(self.afp_system.get_group_summary(hours=24))
                
                # Get additional metrics
                metrics = self.get_dashboard_metrics()
                
                loop.close()
                
                return jsonify({
                    "summary": summary,
                    "metrics": metrics,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error getting summary: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/api/groups')
        def api_groups():
            """Get active alert groups"""
            try:
                hours = request.args.get('hours', 24, type=int)
                groups = self.get_active_groups(hours)
                return jsonify({"groups": groups, "status": "success"})
            except Exception as e:
                logger.error(f"Error getting groups: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/api/rules')
        def api_rules():
            """Get fatigue prevention rules"""
            try:
                rules = self.get_fatigue_rules()
                return jsonify({"rules": rules, "status": "success"})
            except Exception as e:
                logger.error(f"Error getting rules: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/api/rules', methods=['POST'])
        def api_create_rule():
            """Create new fatigue prevention rule"""
            try:
                data = request.json
                rule = FatigueRule(
                    rule_id=data['rule_id'],
                    name=data['name'],
                    description=data.get('description', ''),
                    conditions=data['conditions'],
                    actions=data['actions'],
                    enabled=data.get('enabled', True),
                    priority=data.get('priority', 100)
                )
                
                self.afp_system.add_fatigue_rule(rule)
                return jsonify({"status": "success", "message": "Rule created successfully"})
            except Exception as e:
                logger.error(f"Error creating rule: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/api/rules/<rule_id>', methods=['PUT'])
        def api_update_rule(rule_id):
            """Update fatigue prevention rule"""
            try:
                data = request.json
                
                # Update rule in database
                with sqlite3.connect(self.afp_system.db_path) as conn:
                    conn.execute("""
                        UPDATE fatigue_rules 
                        SET name=?, description=?, conditions=?, actions=?, enabled=?, priority=?, updated_at=?
                        WHERE rule_id=?
                    """, (
                        data['name'], data.get('description', ''),
                        json.dumps(data['conditions']), json.dumps(data['actions']),
                        data.get('enabled', True), data.get('priority', 100),
                        datetime.utcnow().isoformat(), rule_id
                    ))
                
                # Reload rules
                self.afp_system.load_fatigue_rules()
                
                return jsonify({"status": "success", "message": "Rule updated successfully"})
            except Exception as e:
                logger.error(f"Error updating rule: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/api/rules/<rule_id>', methods=['DELETE'])
        def api_delete_rule(rule_id):
            """Delete fatigue prevention rule"""
            try:
                with sqlite3.connect(self.afp_system.db_path) as conn:
                    conn.execute("DELETE FROM fatigue_rules WHERE rule_id=?", (rule_id,))
                
                # Reload rules
                self.afp_system.load_fatigue_rules()
                
                return jsonify({"status": "success", "message": "Rule deleted successfully"})
            except Exception as e:
                logger.error(f"Error deleting rule: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/api/charts/alert_trends')
        def api_alert_trends():
            """Get alert trends chart data"""
            try:
                chart_data = self.generate_alert_trends_chart()
                return jsonify({"chart": chart_data, "status": "success"})
            except Exception as e:
                logger.error(f"Error generating chart: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/api/charts/suppression_stats')
        def api_suppression_stats():
            """Get suppression statistics chart data"""
            try:
                chart_data = self.generate_suppression_stats_chart()
                return jsonify({"chart": chart_data, "status": "success"})
            except Exception as e:
                logger.error(f"Error generating chart: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500
        
        @self.app.route('/api/test_grouping', methods=['POST'])
        def api_test_grouping():
            """Test alert grouping with sample data"""
            try:
                data = request.json
                alerts = data.get('alerts', [])
                algorithm = data.get('algorithm', 'hybrid_approach')
                
                loop = asyncio.new_event_loop()
                asyncio.set_event_loop(loop)
                
                groups = loop.run_until_complete(
                    self.grouping_engine.suggest_groups(alerts, algorithm)
                )
                
                quality = loop.run_until_complete(
                    self.grouping_engine.evaluate_grouping_quality(alerts, groups)
                )
                
                loop.close()
                
                return jsonify({
                    "groups": groups,
                    "quality": quality,
                    "status": "success"
                })
            except Exception as e:
                logger.error(f"Error testing grouping: {e}")
                return jsonify({"error": str(e), "status": "error"}), 500
    
    def get_dashboard_metrics(self) -> Dict[str, Any]:
        """Get key metrics for dashboard"""
        
        with sqlite3.connect(self.afp_system.db_path) as conn:
            # Total groups in last 24 hours
            cutoff_24h = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            total_groups = conn.execute("""
                SELECT COUNT(*) FROM alert_groups WHERE last_seen > ?
            """, (cutoff_24h,)).fetchone()[0]
            
            # Suppressed alerts in last 24 hours
            suppressed_alerts = conn.execute("""
                SELECT SUM(count) FROM alert_groups 
                WHERE state = 'suppressed' AND last_seen > ?
            """, (cutoff_24h,)).fetchone()[0] or 0
            
            # Active rules
            active_rules = conn.execute("""
                SELECT COUNT(*) FROM fatigue_rules WHERE enabled = 1
            """).fetchone()[0]
            
            # Top suppression reasons
            top_suppressions = conn.execute("""
                SELECT rule_id, COUNT(*) as count
                FROM suppression_history 
                WHERE suppressed_at > ?
                GROUP BY rule_id
                ORDER BY count DESC
                LIMIT 5
            """, (cutoff_24h,)).fetchall()
            
            # Average group size
            avg_group_size = conn.execute("""
                SELECT AVG(count) FROM alert_groups WHERE last_seen > ?
            """, (cutoff_24h,)).fetchone()[0] or 0
        
        return {
            "total_groups_24h": total_groups,
            "suppressed_alerts_24h": suppressed_alerts,
            "active_rules": active_rules,
            "top_suppressions": [{"rule": r[0], "count": r[1]} for r in top_suppressions],
            "avg_group_size": round(avg_group_size, 2)
        }
    
    def get_active_groups(self, hours: int = 24) -> List[Dict[str, Any]]:
        """Get active alert groups"""
        
        cutoff_time = (datetime.utcnow() - timedelta(hours=hours)).isoformat()
        
        with sqlite3.connect(self.afp_system.db_path) as conn:
            groups = conn.execute("""
                SELECT group_id, fingerprint_hash, source, alert_type, severity,
                       first_seen, last_seen, count, state, suppression_count,
                       escalation_level, metadata
                FROM alert_groups 
                WHERE last_seen > ?
                ORDER BY last_seen DESC
                LIMIT 100
            """, (cutoff_time,)).fetchall()
            
            result = []
            for group in groups:
                group_dict = {
                    "group_id": group[0],
                    "fingerprint_hash": group[1],
                    "source": group[2],
                    "alert_type": group[3],
                    "severity": group[4],
                    "first_seen": group[5],
                    "last_seen": group[6],
                    "count": group[7],
                    "state": group[8],
                    "suppression_count": group[9],
                    "escalation_level": group[10],
                    "metadata": json.loads(group[11]) if group[11] else {}
                }
                result.append(group_dict)
            
            return result
    
    def get_fatigue_rules(self) -> List[Dict[str, Any]]:
        """Get all fatigue prevention rules"""
        
        with sqlite3.connect(self.afp_system.db_path) as conn:
            rules = conn.execute("""
                SELECT rule_id, name, description, conditions, actions, enabled, priority, created_at
                FROM fatigue_rules
                ORDER BY priority ASC
            """).fetchall()
            
            result = []
            for rule in rules:
                rule_dict = {
                    "rule_id": rule[0],
                    "name": rule[1],
                    "description": rule[2],
                    "conditions": json.loads(rule[3]),
                    "actions": json.loads(rule[4]),
                    "enabled": bool(rule[5]),
                    "priority": rule[6],
                    "created_at": rule[7]
                }
                result.append(rule_dict)
            
            return result
    
    def generate_alert_trends_chart(self) -> str:
        """Generate alert trends chart"""
        
        # Get hourly alert counts for last 24 hours
        with sqlite3.connect(self.afp_system.db_path) as conn:
            cutoff_time = datetime.utcnow() - timedelta(hours=24)
            
            # Generate hourly buckets
            hours = []
            counts = []
            
            for i in range(24):
                hour_start = cutoff_time + timedelta(hours=i)
                hour_end = hour_start + timedelta(hours=1)
                
                count = conn.execute("""
                    SELECT SUM(count) FROM alert_groups 
                    WHERE last_seen >= ? AND last_seen < ?
                """, (hour_start.isoformat(), hour_end.isoformat())).fetchone()[0] or 0
                
                hours.append(hour_start.strftime('%H:%M'))
                counts.append(count)
        
        # Create Plotly chart
        fig = go.Figure()
        fig.add_trace(go.Scatter(
            x=hours,
            y=counts,
            mode='lines+markers',
            name='Alert Count',
            line=dict(color='#1f77b4', width=2),
            marker=dict(size=6)
        ))
        
        fig.update_layout(
            title='Alert Trends (Last 24 Hours)',
            xaxis_title='Time',
            yaxis_title='Alert Count',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def generate_suppression_stats_chart(self) -> str:
        """Generate suppression statistics chart"""
        
        with sqlite3.connect(self.afp_system.db_path) as conn:
            cutoff_time = (datetime.utcnow() - timedelta(hours=24)).isoformat()
            
            # Get suppression counts by rule
            suppressions = conn.execute("""
                SELECT fr.name, COUNT(*) as count
                FROM suppression_history sh
                JOIN fatigue_rules fr ON sh.rule_id = fr.rule_id
                WHERE sh.suppressed_at > ?
                GROUP BY fr.name
                ORDER BY count DESC
                LIMIT 10
            """, (cutoff_time,)).fetchall()
        
        if not suppressions:
            # Return empty chart
            fig = go.Figure()
            fig.add_annotation(
                text="No suppression data available",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
        else:
            names = [s[0] for s in suppressions]
            counts = [s[1] for s in suppressions]
            
            fig = go.Figure(data=[
                go.Bar(x=names, y=counts, marker_color='#ff7f0e')
            ])
        
        fig.update_layout(
            title='Suppression Statistics (Last 24 Hours)',
            xaxis_title='Rule Name',
            yaxis_title='Suppression Count',
            height=400,
            margin=dict(l=50, r=50, t=50, b=50)
        )
        
        return json.dumps(fig, cls=plotly.utils.PlotlyJSONEncoder)
    
    def run(self, host='0.0.0.0', port=5000, debug=False):
        """Run the dashboard server"""
        logger.info(f"Starting Alert Fatigue Dashboard on {host}:{port}")
        self.app.run(host=host, port=port, debug=debug)

# HTML Template for the dashboard
DASHBOARD_TEMPLATE = """
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Alert Fatigue Prevention Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-latest.min.js"></script>
    <script src="https://code.jquery.com/jquery-3.6.0.min.js"></script>
    <style>
        body {
            font-family: -apple-system, BlinkMacSystemFont, 'Segoe UI', Roboto, sans-serif;
            margin: 0;
            padding: 20px;
            background-color: #f5f5f5;
        }
        .header {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            color: white;
            padding: 20px;
            border-radius: 10px;
            margin-bottom: 20px;
        }
        .metrics-grid {
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 20px;
            margin-bottom: 20px;
        }
        .metric-card {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            text-align: center;
        }
        .metric-value {
            font-size: 2em;
            font-weight: bold;
            color: #667eea;
        }
        .metric-label {
            color: #666;
            margin-top: 5px;
        }
        .chart-container {
            background: white;
            padding: 20px;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            margin-bottom: 20px;
        }
        .table-container {
            background: white;
            border-radius: 10px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.1);
            overflow: hidden;
        }
        table {
            width: 100%;
            border-collapse: collapse;
        }
        th, td {
            padding: 12px;
            text-align: left;
            border-bottom: 1px solid #eee;
        }
        th {
            background-color: #f8f9fa;
            font-weight: 600;
        }
        .status-badge {
            padding: 4px 8px;
            border-radius: 4px;
            font-size: 0.8em;
            font-weight: bold;
        }
        .status-new { background-color: #e3f2fd; color: #1976d2; }
        .status-grouped { background-color: #f3e5f5; color: #7b1fa2; }
        .status-suppressed { background-color: #fff3e0; color: #f57c00; }
        .status-escalated { background-color: #ffebee; color: #d32f2f; }
        .loading {
            text-align: center;
            padding: 40px;
            color: #666;
        }
        .error {
            background-color: #ffebee;
            color: #d32f2f;
            padding: 10px;
            border-radius: 5px;
            margin: 10px 0;
        }
    </style>
</head>
<body>
    <div class="header">
        <h1>🛡️ Alert Fatigue Prevention Dashboard</h1>
        <p>Intelligent alert grouping and fatigue prevention for Pixelated Empathy AI</p>
    </div>

    <div class="metrics-grid" id="metrics-grid">
        <div class="loading">Loading metrics...</div>
    </div>

    <div class="chart-container">
        <div id="alert-trends-chart"></div>
    </div>

    <div class="chart-container">
        <div id="suppression-stats-chart"></div>
    </div>

    <div class="table-container">
        <h3 style="padding: 20px; margin: 0; background-color: #f8f9fa;">Recent Alert Groups</h3>
        <div id="groups-table">
            <div class="loading">Loading alert groups...</div>
        </div>
    </div>

    <script>
        // Load dashboard data
        function loadDashboard() {
            // Load summary metrics
            $.get('/api/summary')
                .done(function(data) {
                    if (data.status === 'success') {
                        updateMetrics(data.metrics);
                    } else {
                        showError('Failed to load metrics: ' + data.error);
                    }
                })
                .fail(function() {
                    showError('Failed to connect to API');
                });

            // Load alert groups
            $.get('/api/groups')
                .done(function(data) {
                    if (data.status === 'success') {
                        updateGroupsTable(data.groups);
                    } else {
                        showError('Failed to load groups: ' + data.error);
                    }
                })
                .fail(function() {
                    showError('Failed to load groups');
                });

            // Load charts
            loadCharts();
        }

        function updateMetrics(metrics) {
            const metricsHtml = `
                <div class="metric-card">
                    <div class="metric-value">${metrics.total_groups_24h}</div>
                    <div class="metric-label">Alert Groups (24h)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.suppressed_alerts_24h}</div>
                    <div class="metric-label">Suppressed Alerts (24h)</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.active_rules}</div>
                    <div class="metric-label">Active Rules</div>
                </div>
                <div class="metric-card">
                    <div class="metric-value">${metrics.avg_group_size}</div>
                    <div class="metric-label">Avg Group Size</div>
                </div>
            `;
            $('#metrics-grid').html(metricsHtml);
        }

        function updateGroupsTable(groups) {
            if (groups.length === 0) {
                $('#groups-table').html('<div class="loading">No alert groups found</div>');
                return;
            }

            let tableHtml = `
                <table>
                    <thead>
                        <tr>
                            <th>Group ID</th>
                            <th>Source</th>
                            <th>Alert Type</th>
                            <th>Severity</th>
                            <th>Count</th>
                            <th>State</th>
                            <th>Last Seen</th>
                        </tr>
                    </thead>
                    <tbody>
            `;

            groups.forEach(function(group) {
                const lastSeen = new Date(group.last_seen).toLocaleString();
                const stateClass = 'status-' + group.state;
                
                tableHtml += `
                    <tr>
                        <td>${group.group_id.substring(0, 8)}...</td>
                        <td>${group.source}</td>
                        <td>${group.alert_type}</td>
                        <td>${group.severity}</td>
                        <td>${group.count}</td>
                        <td><span class="status-badge ${stateClass}">${group.state}</span></td>
                        <td>${lastSeen}</td>
                    </tr>
                `;
            });

            tableHtml += '</tbody></table>';
            $('#groups-table').html(tableHtml);
        }

        function loadCharts() {
            // Load alert trends chart
            $.get('/api/charts/alert_trends')
                .done(function(data) {
                    if (data.status === 'success') {
                        Plotly.newPlot('alert-trends-chart', JSON.parse(data.chart).data, JSON.parse(data.chart).layout);
                    }
                });

            // Load suppression stats chart
            $.get('/api/charts/suppression_stats')
                .done(function(data) {
                    if (data.status === 'success') {
                        Plotly.newPlot('suppression-stats-chart', JSON.parse(data.chart).data, JSON.parse(data.chart).layout);
                    }
                });
        }

        function showError(message) {
            const errorHtml = `<div class="error">Error: ${message}</div>`;
            $('body').prepend(errorHtml);
        }

        // Auto-refresh every 30 seconds
        setInterval(loadDashboard, 30000);

        // Initial load
        $(document).ready(function() {
            loadDashboard();
        });
    </script>
</body>
</html>
"""

# Example usage
async def run_dashboard():
    """Run the alert fatigue dashboard"""
    
    # Initialize systems
    afp_system = AlertFatiguePreventionSystem()
    dashboard = FatigueDashboard(afp_system)
    
    # Run dashboard
    dashboard.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    asyncio.run(run_dashboard())
