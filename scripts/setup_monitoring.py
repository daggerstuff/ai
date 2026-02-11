#!/usr/bin/env python3
"""
Production Monitoring Setup
Deploys Prometheus, Grafana, and AlertManager for system monitoring.
"""

import os
import yaml
import json
import subprocess
from pathlib import Path
import logging
from typing import Dict, Any

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('monitoring_setup.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class MonitoringSetup:
    """Production monitoring setup orchestrator."""
    
    def __init__(self):
        self.monitoring_dir = Path("monitoring")
        self.monitoring_dir.mkdir(exist_ok=True)
        
        # Create subdirectories
        (self.monitoring_dir / "prometheus").mkdir(exist_ok=True)
        (self.monitoring_dir / "grafana").mkdir(exist_ok=True)
        (self.monitoring_dir / "alertmanager").mkdir(exist_ok=True)
        (self.monitoring_dir / "docker-compose").mkdir(exist_ok=True)
    
    def create_prometheus_config(self) -> bool:
        """Create Prometheus configuration."""
        try:
            prometheus_config = {
                "global": {
                    "scrape_interval": "15s",
                    "evaluation_interval": "15s"
                },
                "rule_files": [
                    "alert_rules.yml"
                ],
                "alerting": {
                    "alertmanagers": [
                        {
                            "static_configs": [
                                {"targets": ["alertmanager:9093"]}
                            ]
                        }
                    ]
                },
                "scrape_configs": [
                    {
                        "job_name": "prometheus",
                        "static_configs": [
                            {"targets": ["localhost:9090"]}
                        ]
                    },
                    {
                        "job_name": "pixel-api",
                        "static_configs": [
                            {"targets": ["pixel-api:8000"]}
                        ],
                        "metrics_path": "/metrics",
                        "scrape_interval": "10s"
                    },
                    {
                        "job_name": "postgres-exporter",
                        "static_configs": [
                            {"targets": ["postgres-exporter:9187"]}
                        ]
                    },
                    {
                        "job_name": "redis-exporter",
                        "static_configs": [
                            {"targets": ["redis-exporter:9121"]}
                        ]
                    },
                    {
                        "job_name": "node-exporter",
                        "static_configs": [
                            {"targets": ["node-exporter:9100"]}
                        ]
                    }
                ]
            }
            
            config_path = self.monitoring_dir / "prometheus" / "prometheus.yml"
            with open(config_path, 'w') as f:
                yaml.dump(prometheus_config, f, default_flow_style=False)
            
            logger.info("✅ Prometheus configuration created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Prometheus config creation failed: {e}")
            return False
    
    def create_alert_rules(self) -> bool:
        """Create Prometheus alert rules."""
        try:
            alert_rules = {
                "groups": [
                    {
                        "name": "pixel_empathy_alerts",
                        "rules": [
                            {
                                "alert": "HighErrorRate",
                                "expr": "rate(http_requests_total{status=~\"5..\"}[5m]) > 0.1",
                                "for": "5m",
                                "labels": {"severity": "critical"},
                                "annotations": {
                                    "summary": "High error rate detected",
                                    "description": "Error rate is above 10% for 5 minutes"
                                }
                            },
                            {
                                "alert": "DatabaseConnectionFailure",
                                "expr": "up{job=\"postgres-exporter\"} == 0",
                                "for": "1m",
                                "labels": {"severity": "critical"},
                                "annotations": {
                                    "summary": "Database connection failure",
                                    "description": "PostgreSQL is not responding"
                                }
                            },
                            {
                                "alert": "HighMemoryUsage",
                                "expr": "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes < 0.1",
                                "for": "5m",
                                "labels": {"severity": "warning"},
                                "annotations": {
                                    "summary": "High memory usage",
                                    "description": "Memory usage is above 90%"
                                }
                            },
                            {
                                "alert": "DiskSpaceLow",
                                "expr": "node_filesystem_avail_bytes{mountpoint=\"/\"} / node_filesystem_size_bytes{mountpoint=\"/\"} < 0.1",
                                "for": "5m",
                                "labels": {"severity": "warning"},
                                "annotations": {
                                    "summary": "Low disk space",
                                    "description": "Disk space is below 10%"
                                }
                            },
                            {
                                "alert": "ConversationProcessingStalled",
                                "expr": "increase(conversations_processed_total[10m]) == 0",
                                "for": "10m",
                                "labels": {"severity": "warning"},
                                "annotations": {
                                    "summary": "Conversation processing stalled",
                                    "description": "No conversations processed in 10 minutes"
                                }
                            }
                        ]
                    }
                ]
            }
            
            rules_path = self.monitoring_dir / "prometheus" / "alert_rules.yml"
            with open(rules_path, 'w') as f:
                yaml.dump(alert_rules, f, default_flow_style=False)
            
            logger.info("✅ Alert rules created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Alert rules creation failed: {e}")
            return False
    
    def create_grafana_config(self) -> bool:
        """Create Grafana configuration and dashboards."""
        try:
            # Grafana datasource configuration
            datasource_config = {
                "apiVersion": 1,
                "datasources": [
                    {
                        "name": "Prometheus",
                        "type": "prometheus",
                        "access": "proxy",
                        "url": "http://prometheus:9090",
                        "isDefault": True
                    }
                ]
            }
            
            datasource_path = self.monitoring_dir / "grafana" / "datasources.yml"
            with open(datasource_path, 'w') as f:
                yaml.dump(datasource_config, f, default_flow_style=False)
            
            # Create basic dashboard
            dashboard = {
                "dashboard": {
                    "id": None,
                    "title": "Pixel Empathy System Overview",
                    "tags": ["pixel", "empathy", "system"],
                    "timezone": "browser",
                    "panels": [
                        {
                            "id": 1,
                            "title": "API Request Rate",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "rate(http_requests_total[5m])",
                                    "legendFormat": "{{method}} {{status}}"
                                }
                            ],
                            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 0}
                        },
                        {
                            "id": 2,
                            "title": "Database Connections",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "pg_stat_database_numbackends",
                                    "legendFormat": "Active Connections"
                                }
                            ],
                            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 0}
                        },
                        {
                            "id": 3,
                            "title": "Memory Usage",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "node_memory_MemAvailable_bytes / node_memory_MemTotal_bytes * 100",
                                    "legendFormat": "Available Memory %"
                                }
                            ],
                            "gridPos": {"h": 8, "w": 12, "x": 0, "y": 8}
                        },
                        {
                            "id": 4,
                            "title": "Conversation Processing Rate",
                            "type": "graph",
                            "targets": [
                                {
                                    "expr": "rate(conversations_processed_total[5m])",
                                    "legendFormat": "Conversations/sec"
                                }
                            ],
                            "gridPos": {"h": 8, "w": 12, "x": 12, "y": 8}
                        }
                    ],
                    "time": {"from": "now-1h", "to": "now"},
                    "refresh": "5s"
                }
            }
            
            dashboard_path = self.monitoring_dir / "grafana" / "dashboard.json"
            with open(dashboard_path, 'w') as f:
                json.dump(dashboard, f, indent=2)
            
            logger.info("✅ Grafana configuration created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Grafana config creation failed: {e}")
            return False
    
    def create_docker_compose(self) -> bool:
        """Create Docker Compose configuration for monitoring stack."""
        try:
            docker_compose = {
                "version": "3.8",
                "services": {
                    "prometheus": {
                        "image": "prom/prometheus:latest",
                        "container_name": "prometheus",
                        "ports": ["9090:9090"],
                        "volumes": [
                            "./prometheus:/etc/prometheus",
                            "prometheus_data:/prometheus"
                        ],
                        "command": [
                            "--config.file=/etc/prometheus/prometheus.yml",
                            "--storage.tsdb.path=/prometheus",
                            "--web.console.libraries=/etc/prometheus/console_libraries",
                            "--web.console.templates=/etc/prometheus/consoles",
                            "--storage.tsdb.retention.time=200h",
                            "--web.enable-lifecycle"
                        ],
                        "restart": "unless-stopped"
                    },
                    "grafana": {
                        "image": "grafana/grafana:latest",
                        "container_name": "grafana",
                        "ports": ["3000:3000"],
                        "volumes": [
                            "grafana_data:/var/lib/grafana",
                            "./grafana/datasources.yml:/etc/grafana/provisioning/datasources/datasources.yml"
                        ],
                        "environment": [
                            "GF_SECURITY_ADMIN_PASSWORD=admin123",
                            "GF_USERS_ALLOW_SIGN_UP=false"
                        ],
                        "restart": "unless-stopped"
                    },
                    "alertmanager": {
                        "image": "prom/alertmanager:latest",
                        "container_name": "alertmanager",
                        "ports": ["9093:9093"],
                        "volumes": ["./alertmanager:/etc/alertmanager"],
                        "restart": "unless-stopped"
                    },
                    "node-exporter": {
                        "image": "prom/node-exporter:latest",
                        "container_name": "node-exporter",
                        "ports": ["9100:9100"],
                        "volumes": [
                            "/proc:/host/proc:ro",
                            "/sys:/host/sys:ro",
                            "/:/rootfs:ro"
                        ],
                        "command": [
                            "--path.procfs=/host/proc",
                            "--path.rootfs=/rootfs",
                            "--path.sysfs=/host/sys",
                            "--collector.filesystem.mount-points-exclude=^/(sys|proc|dev|host|etc)($$|/)"
                        ],
                        "restart": "unless-stopped"
                    },
                    "postgres-exporter": {
                        "image": "prometheuscommunity/postgres-exporter:latest",
                        "container_name": "postgres-exporter",
                        "ports": ["9187:9187"],
                        "environment": [
                            "DATA_SOURCE_NAME=postgresql://postgres:postgres@host.docker.internal:5432/pixelated_empathy?sslmode=disable"
                        ],
                        "restart": "unless-stopped"
                    }
                },
                "volumes": {
                    "prometheus_data": {},
                    "grafana_data": {}
                }
            }
            
            compose_path = self.monitoring_dir / "docker-compose.yml"
            with open(compose_path, 'w') as f:
                yaml.dump(docker_compose, f, default_flow_style=False)
            
            logger.info("✅ Docker Compose configuration created")
            return True
            
        except Exception as e:
            logger.error(f"❌ Docker Compose creation failed: {e}")
            return False
    
    def deploy_monitoring_stack(self) -> bool:
        """Deploy the monitoring stack using Docker Compose."""
        try:
            logger.info("Deploying monitoring stack...")

            # Try new Docker Compose syntax first, then fall back to old
            compose_commands = [
                ["docker", "compose", "-f", str(self.monitoring_dir / "docker-compose.yml"), "up", "-d"],
                ["docker-compose", "-f", str(self.monitoring_dir / "docker-compose.yml"), "up", "-d"]
            ]

            for cmd in compose_commands:
                try:
                    result = subprocess.run(cmd, capture_output=True, text=True, cwd=self.monitoring_dir)

                    if result.returncode == 0:
                        logger.info("✅ Monitoring stack deployed successfully")
                        logger.info("Prometheus: http://localhost:9090")
                        logger.info("Grafana: http://localhost:3000 (admin/admin123)")
                        logger.info("AlertManager: http://localhost:9093")
                        return True
                    else:
                        logger.warning(f"Command failed: {' '.join(cmd)}")
                        logger.warning(f"Error: {result.stderr}")
                        continue

                except FileNotFoundError:
                    logger.warning(f"Command not found: {cmd[0]}")
                    continue

            logger.error("❌ All Docker Compose commands failed")
            return False

        except Exception as e:
            logger.error(f"❌ Monitoring deployment error: {e}")
            return False

def main():
    """Main monitoring setup orchestrator."""
    logger.info("🔍 STARTING MONITORING SETUP")
    
    setup = MonitoringSetup()
    
    # Create configurations
    if not setup.create_prometheus_config():
        return False
    
    if not setup.create_alert_rules():
        return False
    
    if not setup.create_grafana_config():
        return False
    
    if not setup.create_docker_compose():
        return False
    
    # Deploy stack
    if not setup.deploy_monitoring_stack():
        return False
    
    logger.info("✅ MONITORING SETUP COMPLETED SUCCESSFULLY!")
    logger.info("Access URLs:")
    logger.info("  - Prometheus: http://localhost:9090")
    logger.info("  - Grafana: http://localhost:3000 (admin/admin123)")
    logger.info("  - AlertManager: http://localhost:9093")
    
    return True

if __name__ == "__main__":
    success = main()
    exit(0 if success else 1)
