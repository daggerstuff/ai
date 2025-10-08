#!/usr/bin/env python3
"""
Quality Anomaly Detection Demo
Demonstrates anomaly detection with synthetic data
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
import random
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
    severity: str
    anomaly_type: str
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

class QualityAnomalyDemo:
    """Demo quality anomaly detection system"""
    
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
        
    def create_demo_anomalies(self) -> List[QualityAnomaly]:
        """Create demo anomalies with synthetic data"""
        print("🎭 Creating demo quality anomalies...")
        
        try:
            # Get base data for realistic values
            base_data = self._get_base_data()
            
            if not base_data:
                print("❌ No base data found")
                return []
            
            # Create synthetic anomalies
            anomalies = []
            
            for metric in self.quality_metrics:
                metric_anomalies = self._create_metric_anomalies(base_data, metric)
                anomalies.extend(metric_anomalies)
            
            # Sort by severity and timestamp
            anomalies.sort(key=lambda x: (
                {'critical': 4, 'high': 3, 'medium': 2, 'low': 1}[x.severity],
                x.timestamp
            ), reverse=True)
            
            print(f"✅ Created {len(anomalies)} demo anomalies")
            return anomalies
            
        except Exception as e:
            print(f"❌ Error creating demo anomalies: {e}")
            return []
    
    def _get_base_data(self) -> List[Dict]:
        """Get base data from database"""
        try:
            conn = sqlite3.connect(self.db_path)
            
            query = """
            SELECT 
                dataset_source,
                tier,
                turn_count,
                word_count,
                processing_status
            FROM conversations 
            WHERE turn_count IS NOT NULL 
            AND word_count IS NOT NULL
            LIMIT 1000
            """
            
            cursor = conn.execute(query)
            columns = [desc[0] for desc in cursor.description]
            
            data = []
            for row in cursor.fetchall():
                record = dict(zip(columns, row))
                data.append(record)
            
            conn.close()
            return data
            
        except Exception as e:
            print(f"❌ Error getting base data: {e}")
            return []
    
    def _create_metric_anomalies(self, base_data: List[Dict], metric: str) -> List[QualityAnomaly]:
        """Create synthetic anomalies for a metric"""
        try:
            # Calculate baseline statistics
            baseline_value = self._calculate_baseline_value(base_data, metric)
            if baseline_value is None:
                return []
            
            # Create 1-3 anomalies per metric
            num_anomalies = random.randint(1, 3)
            anomalies = []
            
            for i in range(num_anomalies):
                # Create synthetic anomaly
                anomaly = self._create_synthetic_anomaly(metric, baseline_value, i)
                if anomaly:
                    anomalies.append(anomaly)
            
            return anomalies
            
        except Exception as e:
            print(f"❌ Error creating anomalies for {metric}: {e}")
            return []
    
    def _calculate_baseline_value(self, data: List[Dict], metric: str) -> Optional[float]:
        """Calculate baseline value for a metric"""
        try:
            if metric == 'conversation_length':
                values = [r['turn_count'] for r in data if r['turn_count']]
                return np.mean(values) if values else None
            elif metric == 'content_richness':
                values = [r['word_count'] for r in data if r['word_count']]
                return np.mean(values) if values else None
            elif metric == 'processing_efficiency':
                total = len(data)
                successful = len([r for r in data if r['processing_status'] == 'processed'])
                return (successful / total) * 100 if total > 0 else None
            elif metric == 'tier_quality':
                total = len(data)
                priority = len([r for r in data if r['tier'] and 'priority' in str(r['tier'])])
                return (priority / total) * 100 if total > 0 else None
            elif metric == 'dataset_diversity':
                unique_datasets = len(set(r['dataset_source'] for r in data if r['dataset_source']))
                return float(unique_datasets)
            
            return None
            
        except Exception as e:
            print(f"❌ Error calculating baseline for {metric}: {e}")
            return None
    
    def _create_synthetic_anomaly(self, metric: str, baseline_value: float, index: int) -> Optional[QualityAnomaly]:
        """Create a synthetic anomaly"""
        try:
            # Define anomaly scenarios
            scenarios = [
                ('critical', 'spike', 3.5, 4.0),    # Critical spike
                ('high', 'drop', -3.0, -2.5),      # High drop
                ('medium', 'outlier', 2.5, 3.0),   # Medium outlier
                ('low', 'drift', 2.0, 2.5)         # Low drift
            ]
            
            # Choose random scenario
            severity, anomaly_type, min_z, max_z = random.choice(scenarios)
            z_score = random.uniform(min_z, max_z)
            
            # Calculate anomalous value
            std_dev = baseline_value * 0.1  # Assume 10% standard deviation
            anomalous_value = baseline_value + (z_score * std_dev)
            
            # Ensure realistic bounds
            if metric in ['processing_efficiency', 'tier_quality']:
                anomalous_value = max(0, min(100, anomalous_value))
            elif metric == 'conversation_length':
                anomalous_value = max(1, anomalous_value)
            elif metric == 'content_richness':
                anomalous_value = max(10, anomalous_value)
            elif metric == 'dataset_diversity':
                anomalous_value = max(1, anomalous_value)
            
            # Calculate actual deviation
            deviation = anomalous_value - baseline_value
            
            # Create timestamp (recent)
            timestamp = datetime.now() - timedelta(hours=random.randint(1, 24))
            
            # Calculate confidence
            confidence = min(0.99, abs(z_score) / 4.0)
            
            return QualityAnomaly(
                metric=metric,
                timestamp=timestamp,
                value=anomalous_value,
                expected_value=baseline_value,
                deviation=deviation,
                severity=severity,
                anomaly_type=anomaly_type,
                confidence=confidence,
                context={
                    'baseline_value': baseline_value,
                    'z_score': abs(z_score),
                    'std_dev': std_dev,
                    'scenario': f"{severity}_{anomaly_type}"
                }
            )
            
        except Exception as e:
            print(f"❌ Error creating synthetic anomaly: {e}")
            return None
    
    def generate_demo_alerts(self, anomalies: List[QualityAnomaly]) -> List[Alert]:
        """Generate demo alerts from anomalies"""
        print(f"🚨 Generating demo alerts from {len(anomalies)} anomalies...")
        
        try:
            alerts = []
            
            # Group anomalies by severity and metric
            grouped_anomalies = {}
            for anomaly in anomalies:
                key = f"{anomaly.severity}_{anomaly.metric}"
                if key not in grouped_anomalies:
                    grouped_anomalies[key] = []
                grouped_anomalies[key].append(anomaly)
            
            # Create alerts for each group
            for group_key, group_anomalies in grouped_anomalies.items():
                alert = self._create_demo_alert(group_key, group_anomalies)
                if alert:
                    alerts.append(alert)
            
            print(f"✅ Generated {len(alerts)} demo alerts")
            return alerts
            
        except Exception as e:
            print(f"❌ Error generating demo alerts: {e}")
            return []
    
    def _create_demo_alert(self, group_key: str, anomalies: List[QualityAnomaly]) -> Optional[Alert]:
        """Create demo alert from grouped anomalies"""
        try:
            if not anomalies:
                return None
            
            severity, metric = group_key.split('_', 1)
            
            # Generate alert ID
            alert_id = f"DEMO_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{group_key}"
            
            # Create title and message
            title = f"{severity.upper()} Quality Anomaly: {metric.replace('_', ' ').title()}"
            
            anomaly_count = len(anomalies)
            avg_deviation = np.mean([abs(a.deviation) for a in anomalies])
            
            message = f"""
DEMO ALERT: Quality anomaly detected in {metric.replace('_', ' ')}.
- Severity: {severity.upper()}
- Anomalies detected: {anomaly_count}
- Average deviation: {avg_deviation:.2f}
- Detection confidence: {np.mean([a.confidence for a in anomalies]):.2f}
            """.strip()
            
            # Generate recommended actions
            recommended_actions = self._generate_demo_actions(metric, severity)
            
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
            print(f"❌ Error creating demo alert: {e}")
            return None
    
    def _generate_demo_actions(self, metric: str, severity: str) -> List[str]:
        """Generate demo recommended actions"""
        actions = []
        
        # Severity-based actions
        if severity in ['critical', 'high']:
            actions.append("🚨 Immediate investigation required")
            actions.append("📞 Notify quality assurance team")
        else:
            actions.append("📋 Schedule quality review")
            actions.append("📊 Monitor trend closely")
        
        # Metric-specific actions
        if metric == 'conversation_length':
            actions.extend([
                "🔍 Review conversation generation parameters",
                "📝 Check conversation templates",
                "⚙️ Validate length calculation logic"
            ])
        elif metric == 'content_richness':
            actions.extend([
                "📚 Review content quality guidelines",
                "🔧 Check word count calculations",
                "📈 Analyze content complexity trends"
            ])
        elif metric == 'processing_efficiency':
            actions.extend([
                "🖥️ Check processing pipeline status",
                "📋 Review error logs",
                "⚡ Validate system resources"
            ])
        elif metric == 'tier_quality':
            actions.extend([
                "🏷️ Review tier classification logic",
                "📊 Check data source quality",
                "✅ Validate tier assignment criteria"
            ])
        elif metric == 'dataset_diversity':
            actions.extend([
                "📁 Check data source availability",
                "🔄 Review dataset integration status",
                "📈 Validate diversity calculations"
            ])
        
        return actions
    
    def create_demo_visualizations(self, anomalies: List[QualityAnomaly]) -> Dict[str, str]:
        """Create demo anomaly visualizations"""
        print("📈 Creating demo anomaly visualizations...")
        
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
            fig.suptitle('Quality Anomaly Detection Dashboard - Demo', fontsize=16, fontweight='bold')
            
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
                ax.text(bar.get_x() + bar.get_width()/2., height + 0.05,
                       str(count), ha='center', va='bottom')
            
            # Anomaly timeline
            ax = axes[1, 0]
            timestamps = [a.timestamp for a in anomalies]
            severities = [a.severity for a in anomalies]
            
            # Create scatter plot with severity colors
            severity_colors = {'critical': 'red', 'high': 'orange', 'medium': 'yellow', 'low': 'blue'}
            for severity in set(severities):
                severity_times = [t for t, s in zip(timestamps, severities) if s == severity]
                severity_values = [1] * len(severity_times)  # Just for plotting
                ax.scatter(severity_times, severity_values, 
                          c=severity_colors[severity], label=severity, alpha=0.7, s=100)
            
            ax.set_title('Anomaly Timeline')
            ax.set_xlabel('Time')
            ax.set_ylabel('Anomalies')
            ax.legend()
            ax.grid(True, alpha=0.3)
            
            # Confidence vs Deviation
            ax = axes[1, 1]
            confidences = [a.confidence for a in anomalies]
            deviations = [abs(a.deviation) for a in anomalies]
            
            scatter = ax.scatter(confidences, deviations, alpha=0.7, s=100)
            ax.set_title('Confidence vs Deviation')
            ax.set_xlabel('Confidence Level')
            ax.set_ylabel('Absolute Deviation')
            ax.grid(True, alpha=0.3)
            
            # Add trend line
            if len(confidences) > 1:
                z = np.polyfit(confidences, deviations, 1)
                p = np.poly1d(z)
                ax.plot(confidences, p(confidences), "r--", alpha=0.8)
            
            plt.tight_layout()
            
            # Save dashboard
            dashboard_file = self.output_dir / "anomaly_detection_demo.png"
            plt.savefig(dashboard_file, dpi=300, bbox_inches='tight')
            plt.close()
            
            viz_files['dashboard'] = str(dashboard_file)
            
            print(f"✅ Created {len(viz_files)} demo visualization files")
            return viz_files
            
        except Exception as e:
            print(f"❌ Error creating demo visualizations: {e}")
            return {}
    
    def export_demo_report(self, anomalies: List[QualityAnomaly], alerts: List[Alert],
                          visualizations: Dict[str, str]) -> str:
        """Export demo anomaly detection report"""
        print("📄 Exporting demo anomaly detection report...")
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            report_file = self.output_dir / f"quality_anomaly_demo_{timestamp}.json"
            
            # Create summary statistics
            severity_counts = pd.Series([a.severity for a in anomalies]).value_counts().to_dict()
            metric_counts = pd.Series([a.metric for a in anomalies]).value_counts().to_dict()
            type_counts = pd.Series([a.anomaly_type for a in anomalies]).value_counts().to_dict()
            
            avg_confidence = np.mean([a.confidence for a in anomalies]) if anomalies else 0
            avg_deviation = np.mean([abs(a.deviation) for a in anomalies]) if anomalies else 0
            
            # Prepare export data
            export_data = {
                'report_metadata': {
                    'generated_at': datetime.now().isoformat(),
                    'report_type': 'demo',
                    'detector_version': '1.0.0',
                    'total_anomalies': len(anomalies),
                    'total_alerts': len(alerts)
                },
                'summary_statistics': {
                    'status': 'anomalies_detected' if anomalies else 'no_anomalies',
                    'severity_distribution': severity_counts,
                    'metric_distribution': metric_counts,
                    'type_distribution': type_counts,
                    'average_confidence': float(avg_confidence),
                    'average_deviation': float(avg_deviation),
                    'critical_alerts': len([a for a in alerts if a.severity == 'critical']),
                    'high_priority_alerts': len([a for a in alerts if a.severity in ['critical', 'high']])
                },
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
            
            print(f"✅ Exported demo anomaly report to: {report_file}")
            return str(report_file)
            
        except Exception as e:
            print(f"❌ Error exporting demo report: {e}")
            return ""

def main():
    """Main demo execution"""
    print("🎭 Quality Anomaly Detection Demo")
    print("=" * 40)
    
    # Initialize demo
    demo = QualityAnomalyDemo()
    
    # Create demo anomalies
    anomalies = demo.create_demo_anomalies()
    
    if not anomalies:
        print("❌ No demo anomalies created")
        return
    
    # Generate alerts
    alerts = demo.generate_demo_alerts(anomalies)
    
    # Create visualizations
    visualizations = demo.create_demo_visualizations(anomalies)
    
    # Export report
    report_file = demo.export_demo_report(anomalies, alerts, visualizations)
    
    # Display summary
    print(f"\n🚨 Demo Anomaly Detection Complete")
    print(f"   - Anomalies detected: {len(anomalies)}")
    print(f"   - Alerts generated: {len(alerts)}")
    print(f"   - Visualizations created: {len(visualizations)}")
    print(f"   - Report saved: {report_file}")
    
    # Show critical alerts
    critical_alerts = [a for a in alerts if a.severity == 'critical']
    if critical_alerts:
        print(f"\n🚨 CRITICAL ALERTS ({len(critical_alerts)}):")
        for alert in critical_alerts[:2]:  # Show top 2
            print(f"   • {alert.title}")
            print(f"     {alert.message.split('.')[0]}")
    
    # Show anomaly summary
    severity_counts = pd.Series([a.severity for a in anomalies]).value_counts()
    print(f"\n📊 Anomaly Summary:")
    for severity, count in severity_counts.items():
        icon = "🔴" if severity == 'critical' else "🟠" if severity == 'high' else "🟡" if severity == 'medium' else "🔵"
        print(f"   {icon} {severity.title()}: {count}")
    
    # Show top recommendations
    if alerts:
        print(f"\n💡 Top Recommendations:")
        all_actions = []
        for alert in alerts:
            all_actions.extend(alert.recommended_actions[:2])  # Top 2 per alert
        
        for action in list(set(all_actions))[:4]:  # Top 4 unique actions
            print(f"   • {action}")

if __name__ == "__main__":
    main()
