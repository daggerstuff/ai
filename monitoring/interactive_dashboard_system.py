<<<<<<< HEAD
#!/usr/bin/env python3
"""
Interactive Web Dashboard System
Creates dynamic HTML dashboards with real-time data visualization instead of static images

Features:
- Interactive HTML dashboards with Chart.js
- Real-time data updates via AJAX
- Responsive design for all devices
- Live data feeds from our analytics systems
- No static images - all dynamic content
"""

import json
import os
import sqlite3
from datetime import UTC, datetime, timedelta

import pandas as pd
from flask import Flask, jsonify, render_template


class InteractiveDashboardSystem:
    def __init__(self):
        self.base_dir = "/home/vivi/pixelated/ai"
        self.dashboard_dir = f"{self.base_dir}/monitoring/dashboards"
        self.db_path = f"{self.base_dir}/database/conversations.db"
        self.app = Flask(__name__, template_folder=self.dashboard_dir)

        # Ensure directories exist
        os.makedirs(f"{self.dashboard_dir}/templates", exist_ok=True)
        os.makedirs(f"{self.dashboard_dir}/static/css", exist_ok=True)
        os.makedirs(f"{self.dashboard_dir}/static/js", exist_ok=True)

    def create_base_template(self):
        """Create the base HTML template for all dashboards"""

        base_template = """<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>{% block title %}Pixelated Empathy AI Dashboard{% endblock %}</title>

    <!-- Chart.js for interactive charts -->
    <script src="https://cdn.jsdelivr.net/npm/chart.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/date-fns@2.29.3/index.min.js"></script>
    <script src="https://cdn.jsdelivr.net/npm/chartjs-adapter-date-fns@2.0.0/dist/chartjs-adapter-date-fns.bundle.min.js"></script>

    <!-- Bootstrap for responsive design -->
    <link href="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/css/bootstrap.min.css" rel="stylesheet">
    <script src="https://cdn.jsdelivr.net/npm/bootstrap@5.1.3/dist/js/bootstrap.bundle.min.js"></script>

    <!-- Font Awesome for icons -->
    <link rel="stylesheet" href="https://cdnjs.cloudflare.com/ajax/libs/font-awesome/6.0.0/css/all.min.css">

    <style>
        :root {
            --primary-color: #2c3e50;
            --secondary-color: #3498db;
            --success-color: #27ae60;
            --warning-color: #f39c12;
            --danger-color: #e74c3c;
            --dark-color: #34495e;
            --light-color: #ecf0f1;
        }

        body {
            background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
            font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
            min-height: 100vh;
        }

        .dashboard-header {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border-bottom: 1px solid rgba(255, 255, 255, 0.2);
            padding: 20px 0;
            margin-bottom: 30px;
        }

        .dashboard-card {
            background: rgba(255, 255, 255, 0.95);
            backdrop-filter: blur(10px);
            border: 1px solid rgba(255, 255, 255, 0.2);
            border-radius: 15px;
            box-shadow: 0 8px 32px rgba(0, 0, 0, 0.1);
            margin-bottom: 25px;
            transition: transform 0.3s ease, box-shadow 0.3s ease;
        }

        .dashboard-card:hover {
            transform: translateY(-5px);
            box-shadow: 0 15px 40px rgba(0, 0, 0, 0.15);
        }

        .metric-card {
            text-align: center;
            padding: 25px;
        }

        .metric-value {
            font-size: 2.5rem;
            font-weight: bold;
            margin-bottom: 10px;
        }

        .metric-label {
            color: #7f8c8d;
            font-size: 0.9rem;
            text-transform: uppercase;
            letter-spacing: 1px;
        }

        .chart-container {
            position: relative;
            height: 400px;
            padding: 20px;
        }

        .status-indicator {
            display: inline-block;
            width: 12px;
            height: 12px;
            border-radius: 50%;
            margin-right: 8px;
        }

        .status-online { background-color: var(--success-color); }
        .status-warning { background-color: var(--warning-color); }
        .status-offline { background-color: var(--danger-color); }

        .refresh-indicator {
            position: fixed;
            top: 20px;
            right: 20px;
            background: var(--success-color);
            color: white;
            padding: 10px 15px;
            border-radius: 25px;
            font-size: 0.8rem;
            opacity: 0;
            transition: opacity 0.3s ease;
        }

        .refresh-indicator.show {
            opacity: 1;
        }

        @media (max-width: 768px) {
            .chart-container {
                height: 300px;
                padding: 15px;
            }

            .metric-value {
                font-size: 2rem;
            }
        }
    </style>

    {% block extra_css %}{% endblock %}
</head>
<body>
    <div class="refresh-indicator" id="refreshIndicator">
        <i class="fas fa-sync-alt fa-spin"></i> Updating data...
    </div>

    <div class="dashboard-header">
        <div class="container">
            <div class="row align-items-center">
                <div class="col-md-6">
                    <h1 class="mb-0">
                        <i class="fas fa-chart-line text-primary me-3"></i>
                        {% block header_title %}Pixelated Empathy AI{% endblock %}
                    </h1>
                    <p class="text-muted mb-0">{% block header_subtitle %}Real-time Analytics Dashboard{% endblock %}</p>
                </div>
                <div class="col-md-6 text-end">
                    <div class="d-flex justify-content-end align-items-center">
                        <span class="status-indicator status-online"></span>
                        <span class="me-3">System Online</span>
                        <span class="text-muted" id="lastUpdate">Last updated: {{ current_time }}</span>
                    </div>
                </div>
            </div>
        </div>
    </div>

    <div class="container-fluid">
        {% block content %}{% endblock %}
    </div>

    <script>
        // Auto-refresh functionality
        let refreshInterval;

        function showRefreshIndicator() {
            document.getElementById('refreshIndicator').classList.add('show');
        }

        function hideRefreshIndicator() {
            document.getElementById('refreshIndicator').classList.remove('show');
        }

        function updateLastRefreshTime() {
            document.getElementById('lastUpdate').textContent =
                'Last updated: ' + new Date().toLocaleTimeString();
        }

        function startAutoRefresh(intervalMs = 30000) {
            refreshInterval = setInterval(() => {
                // ⚡ Bolt: Prevent unnecessary API calls and re-renders when tab is inactive
                if (document.hidden) return;

                showRefreshIndicator();
                refreshDashboardData();
            }, intervalMs);
        }

        function refreshDashboardData() {
            // This will be overridden by specific dashboard implementations
            setTimeout(() => {
                updateLastRefreshTime();
                hideRefreshIndicator();
            }, 1000);
        }

        // Start auto-refresh when page loads
        document.addEventListener('DOMContentLoaded', function() {
            startAutoRefresh();
        });

        // Chart.js default configuration
        Chart.defaults.responsive = true;
        Chart.defaults.maintainAspectRatio = false;
        Chart.defaults.plugins.legend.position = 'top';
        Chart.defaults.plugins.title.display = true;
        Chart.defaults.plugins.title.font = {
            size: 16,
            weight: 'bold'
        };
    </script>

    {% block extra_js %}{% endblock %}
</body>
</html>"""

        with open(f"{self.dashboard_dir}/templates/base.html", "w") as f:
            f.write(base_template)

    def create_executive_dashboard(self):
        """Create the executive dashboard HTML template"""

        executive_template = """{% extends "base.html" %}

{% block title %}Executive Dashboard - Pixelated Empathy AI{% endblock %}
{% block header_title %}Executive Dashboard{% endblock %}
{% block header_subtitle %}High-level KPIs and Strategic Metrics{% endblock %}

{% block content %}
<div class="row">
    <!-- Key Metrics Row -->
    <div class="col-lg-3 col-md-6 mb-4">
        <div class="dashboard-card metric-card">
            <div class="metric-value text-primary" id="totalConversations">{{ metrics.total_conversations }}</div>
            <div class="metric-label">Total Conversations</div>
            <small class="text-success">
                <i class="fas fa-arrow-up"></i> +{{ metrics.conversations_growth }}% this month
            </small>
        </div>
    </div>

    <div class="col-lg-3 col-md-6 mb-4">
        <div class="dashboard-card metric-card">
            <div class="metric-value text-success" id="avgQualityScore">{{ metrics.avg_quality_score }}%</div>
            <div class="metric-label">Average Quality Score</div>
            <small class="text-success">
                <i class="fas fa-arrow-up"></i> +{{ metrics.quality_improvement }}% improvement
            </small>
        </div>
    </div>

    <div class="col-lg-3 col-md-6 mb-4">
        <div class="dashboard-card metric-card">
            <div class="metric-value text-warning" id="systemUptime">{{ metrics.system_uptime }}%</div>
            <div class="metric-label">System Uptime</div>
            <small class="text-muted">Last 30 days</small>
        </div>
    </div>

    <div class="col-lg-3 col-md-6 mb-4">
        <div class="dashboard-card metric-card">
            <div class="metric-value text-info" id="activeUsers">{{ metrics.active_users }}</div>
            <div class="metric-label">Active Users</div>
            <small class="text-info">
                <i class="fas fa-users"></i> Currently online
            </small>
        </div>
    </div>
</div>

<div class="row">
    <!-- Conversation Trends Chart -->
    <div class="col-lg-8 mb-4">
        <div class="dashboard-card">
            <div class="chart-container">
                <canvas id="conversationTrendsChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Quality Distribution -->
    <div class="col-lg-4 mb-4">
        <div class="dashboard-card">
            <div class="chart-container">
                <canvas id="qualityDistributionChart"></canvas>
            </div>
        </div>
    </div>
</div>

<div class="row">
    <!-- System Performance -->
    <div class="col-lg-6 mb-4">
        <div class="dashboard-card">
            <div class="chart-container">
                <canvas id="systemPerformanceChart"></canvas>
            </div>
        </div>
    </div>

    <!-- Top Issues -->
    <div class="col-lg-6 mb-4">
        <div class="dashboard-card">
            <div class="card-header">
                <h5 class="mb-0">
                    <i class="fas fa-exclamation-triangle text-warning me-2"></i>
                    Top Issues Requiring Attention
                </h5>
            </div>
            <div class="card-body">
                <div class="list-group list-group-flush">
                    {% for issue in top_issues %}
                    <div class="list-group-item d-flex justify-content-between align-items-center">
                        <div>
                            <strong>{{ issue.title }}</strong>
                            <br>
                            <small class="text-muted">{{ issue.description }}</small>
                        </div>
                        <span class="badge bg-{{ issue.severity }} rounded-pill">{{ issue.count }}</span>
                    </div>
                    {% endfor %}
                </div>
            </div>
        </div>
    </div>
</div>
{% endblock %}

{% block extra_js %}
=======
>>>>>>> origin/staging
<script>
    // Auto-refresh functionality
    let refreshInterval;

    function showRefreshIndicator() {
        document.getElementById('refreshIndicator').classList.add('show');
    }

    function hideRefreshIndicator() {
        document.getElementById('refreshIndicator').classList.remove('show');
    }

    function updateLastRefreshTime() {
        document.getElementById('lastUpdate').textContent =
            'Last updated: ' + new Date().toLocaleTimeString();
    }

    function startAutoRefresh(intervalMs = 30000) {
        refreshInterval = setInterval(() => {
            // ⚡ Bolt: Prevent unnecessary background polling and re-renders when tab is inactive
            if (document.hidden) return;
            // ⚡ Bolt: Trigger immediate refresh when tab becomes visible again
            document.addEventListener('visibilitychange', function() {
                if (!document.hidden) {
                    refreshDashboardData();
                }
            });
            showRefreshIndicator();
            refreshDashboardData();
        }, intervalMs);
    }

    function refreshDashboardData() {
        // This will be overridden by specific dashboard implementations
        setTimeout(() => {
            updateLastRefreshTime();
            hideRefreshIndicator();
        }, 1000);
    }

    // Handle tab visibility changes - refresh immediately when tab becomes visible again
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) {
            // Tab became visible, refresh data immediately to avoid stale data
            refreshDashboardData();
        }
    });

    // Register a new visibilitychange listener to refresh data when tab becomes visible
    document.addEventListener('visibilitychange', function() {
        if (!document.hidden) {
            refreshDashboardData();
            refreshDashboardData(); // Call refreshDashboardData() immediately
        }
    });

    // Start auto-refresh when page loads
    document.addEventListener('DOMContentLoaded', function() {
        startAutoRefresh();
    });

    // Chart.js default configuration
    Chart.defaults.responsive = true;
    Chart.defaults.maintainAspectRatio = false;
    Chart.defaults.plugins.legend.position = 'top';
    Chart.defaults.plugins.title.display = true;
    Chart.defaults.plugins.title.font = {
        size: 16,
        weight: 'bold'
    };
</script>