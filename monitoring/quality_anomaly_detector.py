#!/usr/bin/env python3
"""
Quality Anomaly Detection and Alerting System
Detects anomalies in quality metrics and provides real-time alerts
"""

import json
import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Any
import pandas as pd
import numpy as np
from dataclasses import dataclass
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats
import warnings
warnings.filterwarnings('ignore')

@dataclass
class QualityAnomaly:
    """Quality anomaly detection result"""
    metric: str
    timestamp: datetime
    value: float
    expected_value: float
    deviation: float
    severity: str  # 'low', 'medium', 'high', 'critical'
    anomaly_type: str  # 'spike', 'drop', 'drift', 'outlier'
    confidence: float
    context: Dict[str, Any]

@dataclass
class Alert:
    """Quality alert"""
    alert_id: str
    timestamp: datetime
    severity: str
    title: str
    message: str
    anomalies: List[QualityAnomaly]
    recommended_actions: List[str]
    auto_resolved: bool

class QualityAnomalyDetector:
    """Enterprise-grade quality anomaly detection system"""
    
    def __init__(self, db_path: str = "database/conversations.db"):
        self.db_path = Path(db_path)
        self.output_dir = Path("monitoring/quality_anomalies")
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        # Quality metrics to monitor
        self.quality_metrics = [
            'conversation_length',
            'content_richness',
            'processing_efficiency',
            'tier_quality',
            'dataset_diversity'
        ]
        
        # Anomaly detection parameters
        self.anomaly_thresholds = {
            'low': 2.0,      # 2 standard deviations
            'medium': 2.5,   # 2.5 standard deviations
            'high': 3.0,     # 3 standard deviations
            'critical': 3.5  # 3.5 standard deviations
        }
        
        # Alert configuration
        self.alert_cooldown_hours = 1  # Minimum time between similar alerts
        self.min_data_points = 10      # Minimum data points for anomaly detection
        
    def detect_quality_anomalies(self, hours_back: int = 24) -> List[QualityAnomaly]:
        """Detect quality anomalies in recent data"""
        print(f"🔍 Detecting quality anomalies in last {hours_back} hours...")
        
        try:
            # Get recent data and historical baseline
            recent_data = self._get_recent_data(hours_back)
            baseline_data = self._get_baseline_data(days_back=30)
            
            if not recent_data or not baseline_data:
                print("❌ Insufficient data for anomaly detection")
                return []
            
            # Detect anomalies for each metric
            all_anomalies = []
            
            for metric in self.quality_metrics:
                anomalies = self._detect_metric_anomalies(
                    recent_data, baseline_data, metric
                )
                all_anomalies.extend(anomalies)
            
            # Sort by severity and timestamp
            all_anomalies.sort(key=lambda x: (
                {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
                x.timestamp
            ), reverse=True)
            
            print(f"✅ Detected {len(all_anomalies)} quality anomalies")
            return all_anomalies
            
        except Exception as e:
            print(f"❌ Error detecting anomalies: {e}")
            return []
    
    def _get_recent_data(self, hours_back: int) -> List[Dict]:
        """Get recent data for anomaly detection"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate time threshold
            time_threshold = datetime.now() - timedelta(hours=hours_back)
            
            query = """
            SELECT 
                dataset_source,
                tier,
                turn_count,
                word_count,
                processing_status,
                created_at
            FROM conversations 
            WHERE created_at >= ?
            AND turn_count IS NOT NULL 
            AND word_count IS NOT NULL
            ORDER BY created_at DESC
            LIMIT 1000
            """
            
            cursor = conn.execute(query, (time_threshold.isoformat(),))
            columns = [desc[0] for desc in cursor.description]
            
            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                if record['created_at']:
                    record['created_at'] = datetime.fromisoformat(record['created_at'])
                data.append(record)
            
            conn.close()
            return data
            
        except Exception as e:
            print(f"❌ Error getting recent data: {e}")
            return []
    
    def _get_baseline_data(self, days_back: int) -> List[Dict]:
        """Get baseline data for comparison"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Calculate time range (excluding recent data)
            end_time = datetime.now() - timedelta(hours=24)  # Exclude last 24 hours
            start_time = end_time - timedelta(days=days_back)
            
            query = """
            SELECT 
                dataset_source,
                tier,
                turn_count,
                word_count,
                processing_status,
                created_at
            FROM conversations 
            WHERE created_at >= ? AND created_at <= ?
            AND turn_count IS NOT NULL 
            AND word_count IS NOT NULL
            LIMIT 5000
            """
            
            cursor = conn.execute(query, (start_time.isoformat(), end_time.isoformat()))
            columns = [desc[0] for desc in cursor.description]
            
            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                if record['created_at']:
                    record['created_at'] = datetime.fromisoformat(record['created_at'])
                data.append(record)
            
            conn.close()
            return data
            
        except Exception as e:
            print(f"❌ Error getting baseline data: {e}")
            return []
    
    def _detect_metric_anomalies(self, recent_data: List[Dict], baseline_data: List[Dict],
                               metric: str) -> List[QualityAnomaly]:
        """Detect anomalies for a specific metric"""
        try:
            # Calculate baseline statistics
            baseline_values = self._calculate_metric_values(baseline_data, metric)
            if len(baseline_values) < self.min_data_points:
                return []
            
            baseline_mean = np.mean(baseline_values)
            baseline_std = np.std(baseline_values)
            
            if baseline_std == 0:  # No variation in baseline
                return []
            
            # Calculate recent values
            recent_values = self._calculate_metric_values(recent_data, metric)
            if not recent_values:
                return []
            
            # Detect anomalies
            anomalies = []
            
            for i, value in enumerate(recent_values):
                # Calculate z-score
                z_score = abs(value - baseline_mean) / baseline_std
                
                # Determine severity
                severity = self._determine_severity(z_score)
                if severity is None:  # Not anomalous
                    continue
                
                # Determine anomaly type
                anomaly_type = self._determine_anomaly_type(value, baseline_mean, baseline_std)
                
                # Calculate confidence
                confidence = min(0.99, z_score / 4.0)  # Normalize to 0-1
                
                # Create anomaly
                anomaly = QualityAnomaly(
                    metric=metric,
                    timestamp=datetime.now() - timedelta(minutes=i*10),  # Synthetic timestamps
                    value=value,
                    expected_value=baseline_mean,
                    deviation=value - baseline_mean,
                    severity=severity,
                    anomaly_type=anomaly_type,
                    confidence=confidence,
                    context={
                        'baseline_mean': baseline_mean,
                        'baseline_std': baseline_std,
                        'z_score': z_score,
                        'baseline_samples': len(baseline_values)
                    }
                )
                
                anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            print(f"❌ Error detecting anomalies for {metric}: {e}")
            return []
    
    def _calculate_metric_values(self, data: List[Dict], metric: str) -> List[float]:
        """Calculate metric values for a dataset"""
        try:
            if not data:
                return []
            
            if metric == 'conversation_length':
                return [float(r['turn_count']) for r in data if r['turn_count']]
            elif metric == 'content_richness':
                return [float(r['word_count']) for r in data if r['word_count']]
            elif metric == 'processing_efficiency':
                # Calculate efficiency in batches
                batch_size = max(1, len(data) // 10)  # 10 batches
                efficiencies = []
                for i in range(0, len(data), batch_size):
                    batch = data[i:i+batch_size]
                    total = len(batch)
                    successful = len([r for r in batch if r['processing_status'] == 'processed'])
                    efficiencies.append(100.0 * successful / total if total > 0 else 0)
                return efficiencies
            elif metric == 'tier_quality':
                # Calculate tier quality in batches
                batch_size = max(1, len(data) // 10)
                qualities = []
                for i in range(0, len(data), batch_size):
                    batch = data[i:i+batch_size]
                    total = len(batch)
                    priority = len([r for r in batch if r['tier'] and 'priority' in str(r['tier'])])
                    qualities.append(100.0 * priority / total if total > 0 else 0)
                return qualities
            elif metric == 'dataset_diversity':
                # Calculate diversity in time windows
                window_size = max(1, len(data) // 10)
                diversities = []
                for i in range(0, len(data), window_size):
                    window = data[i:i+window_size]
                    unique_datasets = len(set(r['dataset_source'] for r in window if r['dataset_source']))
                    diversities.append(float(unique_datasets))
                return diversities
            
            return []
            
        except Exception as e:
            print(f"❌ Error calculating {metric}: {e}")
            return []
    
    def _determine_severity(self, z_score: float) -> Optional[str]:
        """Determine anomaly severity based on z-score"""
        if z_score >= self.anomaly_thresholds['critical']:
            return 'critical'
        elif z_score >= self.anomaly_thresholds['high']:
            return 'high'
        elif z_score >= self.anomaly_thresholds['medium']:
            return 'medium'
        elif z_score >= self.anomaly_thresholds['low']:
            return 'low'
        else:
            return None  # Not anomalous
    
    def _determine_anomaly_type(self, value: float, baseline_mean: float, baseline_std: float) -> str:
        """Determine the type of anomaly"""
        deviation = value - baseline_mean
        
        if abs(deviation) > 3 * baseline_std:
            return 'outlier'
        elif deviation > 2 * baseline_std:
            return 'spike'
        elif deviation < -2 * baseline_std:
            return 'drop'
        else:
            return 'drift'
    
    def generate_alerts(self, anomalies: List[QualityAnomaly]) -> List[Alert]:
        """Generate alerts from detected anomalies"""
        print(f"🚨 Generating alerts from {len(anomalies)} anomalies...")
        
        try:
            alerts = []
            
            # Group anomalies by severity and metric
            grouped_anomalies = self._group_anomalies(anomalies)
            
            for group_key, group_anomalies in grouped_anomalies.items():
                alert = self._create_alert(group_key, group_anomalies)
                if alert:
                    alerts.append(alert)
            
            print(f"✅ Generated {len(alerts)} alerts")
            return alerts
            
        except Exception as e:
            print(f"❌ Error generating alerts: {e}")
            return []
    
    def _group_anomalies(self, anomalies: List[QualityAnomaly]) -> Dict[str, List[QualityAnomaly]]:
        """Group anomalies for alert generation"""
        groups = {}
        
        for anomaly in anomalies:
            # Group by severity and metric
            key = f"{anomaly.severity}_{anomaly.metric}"
            if key not in groups:
                groups[key] = []
            groups[key].append(anomaly)
        
        return groups
    
    def _create_alert(self, group_key: str, anomalies: List[QualityAnomaly]) -> Optional[Alert]:
        """Create alert from grouped anomalies"""
        try:
            if not anomalies:
                return None
            
            severity, metric = group_key.split('_', 1)
            
            # Generate alert ID
            alert_id = f"QA_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{group_key}"
            
            # Create title and message
            title = f"{severity.upper()} Quality Anomaly: {metric.replace('_', ' ').title()}"
            
            anomaly_count = len(anomalies)
            avg_deviation = np.mean([abs(a.deviation) for a in anomalies])
            
            message = f"""
Quality anomaly detected in {metric.replace('_', ' ')}.
- Severity: {severity.upper()}
- Anomalies detected: {anomaly_count}
- Average deviation: {avg_deviation:.2f}
- Time range: {anomalies[-1].timestamp.strftime('%H:%M')} - {anomalies[0].timestamp.strftime('%H:%M')}
            """.strip()
            
            # Generate recommended actions
            recommended_actions = self._generate_recommended_actions(metric, severity, anomalies)
            
            return Alert(
                alert_id=alert_id,
                timestamp=datetime.now(),
                severity=severity,
                title=title,
                message=message,
                anomalies=anomalies,
                recommended_actions=recommended_actions,
                auto_resolved=False
            )
            
        except Exception as e:
            print(f"❌ Error creating alert: {e}")
            return None
    
    def _generate_recommended_actions(self, metric: str, severity: str, 
                                    anomalies: List[QualityAnomaly]) -> List[str]:
        """Generate recommended actions for alerts"""
        actions = []
        
        # Severity-based actions
        if severity in ['critical', 'high']:
            actions.append("Immediate investigation required")
            actions.append("Notify quality assurance team")
        
        # Metric-specific actions
        if metric == 'conversation_length':
            actions.extend([
                "Review conversation generation parameters",
                "Check for data truncation issues",
                "Validate conversation templates"
            ])
        elif metric == 'content_richness':
            actions.extend([
                "Review content quality guidelines",
                "Check for data processing errors",
                "Validate word count calculations"
            ])
        elif metric == 'processing_efficiency':
            actions.extend([
                "Check processing pipeline status",
                "Review error logs for failures",
                "Validate system resources"
            ])
        elif metric == 'tier_quality':
            actions.extend([
                "Review tier classification logic",
                "Check data source quality",
                "Validate tier assignment criteria"
            ])
        elif metric == 'dataset_diversity':
            actions.extend([
                "Check data source availability",
                "Review dataset integration status",
                "Validate diversity calculations"
            ])
        
        # Anomaly type specific actions
        anomaly_types = set(a.anomaly_type for a in anomalies)
        if 'spike' in anomaly_types:
            actions.append("Investigate sudden increase in metric values")
        if 'drop' in anomaly_types:
            actions.append("Investigate sudden decrease in metric values")
        if 'outlier' in anomaly_types:
            actions.append("Review extreme outlier values for data quality issues")
        
        return actions
    
    def create_anomaly_visualizations(self, anomalies: List[QualityAnomaly]) -> Dict[str, str]:
        """Create anomaly detection visualizations"""
        print("📈 Creating anomaly detection visualizations...")
        
        viz_files = {}
        
        try:
            if not anomalies:
                print("⚠️ No anomalies to visualize")
                return viz_files
            
            # Set style
            plt.style.use('default')
            sns.set_palette("husl")
            
            # Create anomaly dashboard
            fig, axes = plt.subplots(2, 2, figsize=(16, 12))
            fig.suptitle('Quality Anomaly Detection Dashboard', fontsize=16, fontweight='bold')
            
            # Anomalies by severity
            ax = axes[0, 0]
            severities = [a.severity for a in anomalies]
            severity_counts = pd.Series(severities).value_counts()
            colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'lightblue'}
            pie_colors = [colors.get(s, 'gray') for s in severity_counts.index]
            
            ax.pie(severity_counts.values, labels=severity_counts.index, autopct='%1.1f%%', colors=pie_colors)
            ax.set_title('Anomalies by Severity')
            
            # Anomalies by metric
            ax = axes[0, 1]
            metrics = [a.metric for a in anomalies]
            metric_counts = pd.Series(metrics).value_counts()
            
            bars = ax.bar(range(len(metric_counts)), metric_counts.values, alpha=0.7)
            ax.set_title('Anomalies by Metric')
            ax.set_xlabel('Metrics')
            ax.set_ylabel('Count')
            ax.set_xticks(range(len(metric_counts)))
            ax.set_xticklabels([m.replace('_', ' ').title() for m in metric_counts.index], rotation=45, ha='right')
            ax.grid(True, alpha=0.3)
            
            # Add count labels
            for bar, count in zip(bars, metric_counts.values):
                height = bar.get_height()
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                       str(count), ha='center', va='bottom')
            
            # Deviation distribution
            ax = axes[1, 0]
            deviations = [abs(a.deviation) for a in anomalies]
            ax.hist(deviations, bins=15, alpha=0.7, edgecolor='black')
            ax.set_title('Deviation Magnitude Distribution')
            ax.set_xlabel('Absolute Deviation')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            # Confidence levels
            ax = axes[1, 1]
            confidences = [a.confidence for a in anomalies]
            ax.hist(confidences, bins=15, alpha=0.7, edgecolor='black')
            ax.set_title('Anomaly Confidence Distribution')
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('Frequency')
            ax.grid(True, alpha=0.3)
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_file = self.output_dir / "anomaly_detection_dashboard.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['dashboard'] = str(dashboard_file)
            
            print(f"✅ Created {len(viz_files)} anomaly visualization files")
            return viz_files
            
        except Exception as e:
            print(f"❌ Error creating visualizations: {e}")
            return {}
    
    def export_anomaly_report(self, anomalies: List[QualityAnomaly], alerts: List[Alert],
                            visualizations: Dict[str, str]) -> str:
        """Export comprehensive anomaly detection report"""
        print("📄 Exporting anomaly detection report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"quality_anomaly_report_{timestamp}.json"
            
            # Create summary statistics
            summary_stats = self._create_anomaly_summary(anomalies, alerts)
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'detector_version': '1.0.0',
                    'total_anomalies': len(anomalies),
                    'total_alerts': len(alerts)
                },
                'summary_statistics': summary_stats,
                'anomalies': [
                    {
                        'metric': a.metric,
                        'timestamp': a.timestamp.isoformat(),
                        'value': a.value,
                        'expected_value': a.expected_value,
                        'deviation': a.deviation,
                        'severity': a.severity,
                        'anomaly_type': a.anomaly_type,
                        'confidence': a.confidence,
                        'context': a.context
                    }
                    for a in anomalies
                ],
                'alerts': [
                    {
                        'alert_id': alert.alert_id,
                        'timestamp': alert.timestamp.isoformat(),
                        'severity': alert.severity,
                        'title': alert.title,
                        'message': alert.message,
                        'anomaly_count': len(alert.anomalies),
                        'recommended_actions': alert.recommended_actions,
                        'auto_resolved': alert.auto_resolved
                    }
                    for alert in alerts
                ],
                'visualizations': visualizations
            }
            
            # Save report
            with open(report_file, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            print(f"✅ Exported anomaly report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting anomaly report: {e}")
            return ""
    
    def _create_anomaly_summary(self, anomalies: List[QualityAnomaly], alerts: List[Alert]) -> Dict[str, Any]:
        """Create anomaly detection summary"""
        try:
            if not anomalies:
                return {'status': 'no_anomalies'}
            
            # Severity distribution
            severity_counts = pd.Series([a.severity for a in anomalies]).value_counts().to_dict()
            
            # Metric distribution
            metric_counts = pd.Series([a.metric for a in anomalies]).value_counts().to_dict()
            
            # Anomaly type distribution
            type_counts = pd.Series([a.anomaly_type for a in anomalies]).value_counts().to_dict()
            
            # Statistics
            avg_confidence = np.mean([a.confidence for a in anomalies])
            avg_deviation = np.mean([abs(a.deviation) for a in anomalies])
            
            return {
                'status': 'anomalies_detected',
                'total_anomalies': len(anomalies),
                'total_alerts': len(alerts),
                'severity_distribution': severity_counts,
                'metric_distribution': metric_counts,
                'type_distribution': type_counts,
                'average_confidence': float(avg_confidence),
                'average_deviation': float(avg_deviation),
                'critical_alerts': len([a for a in alerts if a.severity == 'critical']),
                'high_priority_alerts': len([a for a in alerts if a.severity in ['critical', 'high']])
            }
            
        except Exception as e:
            print(f"❌ Error creating anomaly summary: {e}")
            return {'status': 'error', 'message': str(e)}

def main():
    """Main execution function"""
    print("🔍 Quality Anomaly Detection and Alerting System")
    print("=" * 55)
    
    # Initialize detector
    detector = QualityAnomalyDetector()
    
    # Detect anomalies
    anomalies = detector.detect_quality_anomalies(hours_back=24)
    
    if not anomalies:
        print("✅ No quality anomalies detected")
        return
    
    # Generate alerts
    alerts = detector.generate_alerts(anomalies)
    
    # Create visualizations
    visualizations = detector.create_anomaly_visualizations(anomalies)
    
    # Export report
    report_file = detector.export_anomaly_report(anomalies, alerts, visualizations)
    
    # Display summary
    print(f"\n🚨 Anomaly Detection Complete")
    print(f"   - Anomalies detected: {len(anomalies)}")
    print(f"   - Alerts generated: {len(alerts)}")
    print(f"   - Visualizations created: {len(visualizations)}")
    print(f"   - Report saved: {report_file}")
    
    # Show critical alerts
    critical_alerts = [a for a in alerts if a.severity == 'critical']
    if critical_alerts:
        print(f"\n🚨 CRITICAL ALERTS ({len(critical_alerts)}):")
        for alert in critical_alerts[:3]:  # Show top 3
            print(f"   • {alert.title}")
            print(f"     {alert.message.split('.')[0]}")
    
    # Show anomaly summary
    severity_counts = pd.Series([a.severity for a in anomalies]).value_counts()
    print(f"\n📊 Anomaly Summary:")
    for severity, count in severity_counts.items():
        icon = "🔴" if severity == 'critical' else "🟠" if severity == 'high' else "🟡" if severity == 'medium' else "🔵"
        print(f"   {icon} {severity.title()}: {count}")

if __name__ == "__main__":
    main()
