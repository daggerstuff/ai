#!/usr/bin/env python3
"""
Task 5.6.2.1: Quality Analytics Dashboard and Visualization

Enterprise-grade quality analytics dashboard providing comprehensive insights
into conversation quality metrics, trends, and performance across all datasets.
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import sqlite3
import json
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import logging
from typing import Dict, List, Any, Optional, Tuple
import seaborn as sns
import matplotlib.pyplot as plt
from dataclasses import dataclass
import warnings

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QualityAnalytics:
    """Quality analytics data structure."""
    total_conversations: int
    average_quality: float
    quality_distribution: Dict[str, int]
    tier_performance: Dict[str, float]
    trend_data: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]

class QualityAnalyticsDashboard:
    """
    Enterprise-grade quality analytics dashboard.
    
    Provides comprehensive visualization and analysis of conversation quality
    metrics across all datasets with trend analysis and performance insights.
    """
    
    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/database/conversations.db"):
        """Initialize the quality analytics dashboard."""
        self.db_path = db_path
        self.cache_duration = 300  # 5 minutes cache
        self._last_cache_time = None
        self._cached_data = None
        
        # Quality thresholds
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.0
        }
        
        # Color schemes for visualizations
        self.color_schemes = {
            'quality_levels': ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c'],
            'tiers': ['#1976d2', '#7b1fa2', '#d32f2f', '#f57c00', '#388e3c'],
            'trends': '#1976d2',
            'anomalies': '#d32f2f'
        }
        
        logger.info("🎯 Quality Analytics Dashboard initialized")
    
    def load_quality_data(self, force_refresh: bool = False) -> pd.DataFrame:
        """Load quality data from database with caching."""
        current_time = datetime.now()
        
        # Check cache validity
        if (not force_refresh and 
            self._cached_data is not None and 
            self._last_cache_time and 
            (current_time - self._last_cache_time).seconds < self.cache_duration):
            return self._cached_data
        
        try:
            conn = sqlite3.connect(self.db_path)
            
            # Load conversation quality data
            query = """
            SELECT 
                c.id,
                c.tier,
                c.dataset_name,
                c.created_at,
                c.conversation_length,
                q.therapeutic_accuracy,
                q.conversation_coherence,
                q.emotional_authenticity,
                q.clinical_compliance,
                q.personality_consistency,
                q.language_quality,
                q.safety_score,
                q.overall_quality,
                q.validated_at
            FROM conversations c
            LEFT JOIN quality_metrics q ON c.id = q.conversation_id
            WHERE q.overall_quality IS NOT NULL
            ORDER BY c.created_at DESC
            """
            
            df = pd.read_sql_query(query, conn)
            conn.close()
            
            # Convert timestamps
            df['created_at'] = pd.to_datetime(df['created_at'])
            df['validated_at'] = pd.to_datetime(df['validated_at'])
            
            # Cache the data
            self._cached_data = df
            self._last_cache_time = current_time
            
            logger.info(f"✅ Loaded {len(df)} quality records from database")
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading quality data: {e}")
            return pd.DataFrame()
    
    def calculate_quality_analytics(self, df: pd.DataFrame) -> QualityAnalytics:
        """Calculate comprehensive quality analytics."""
        if df.empty:
            return QualityAnalytics(
                total_conversations=0,
                average_quality=0.0,
                quality_distribution={},
                tier_performance={},
                trend_data=[],
                anomalies=[],
                recommendations=["No quality data available"]
            )
        
        # Basic statistics
        total_conversations = len(df)
        average_quality = df['overall_quality'].mean()
        
        # Quality distribution
        quality_bins = pd.cut(
            df['overall_quality'], 
            bins=[0, 0.4, 0.6, 0.8, 1.0], 
            labels=['Poor', 'Fair', 'Good', 'Excellent']
        )
        quality_distribution = quality_bins.value_counts().to_dict()
        
        # Tier performance
        tier_performance = df.groupby('tier')['overall_quality'].mean().to_dict()
        
        # Trend data (last 30 days)
        thirty_days_ago = datetime.now() - timedelta(days=30)
        recent_df = df[df['created_at'] >= thirty_days_ago]
        
        trend_data = []
        if not recent_df.empty:
            daily_quality = recent_df.groupby(recent_df['created_at'].dt.date)['overall_quality'].mean()
            trend_data = [
                {'date': str(date), 'quality': float(quality)}
                for date, quality in daily_quality.items()
            ]
        
        # Anomaly detection (conversations with quality significantly below average)
        quality_std = df['overall_quality'].std()
        anomaly_threshold = average_quality - (2 * quality_std)
        anomalies_df = df[df['overall_quality'] < anomaly_threshold]
        
        anomalies = []
        for _, row in anomalies_df.head(10).iterrows():  # Top 10 anomalies
            anomalies.append({
                'id': row['id'],
                'dataset': row['dataset_name'],
                'tier': row['tier'],
                'quality': float(row['overall_quality']),
                'date': row['created_at'].strftime('%Y-%m-%d')
            })
        
        # Generate recommendations
        recommendations = self._generate_recommendations(df, average_quality, tier_performance)
        
        return QualityAnalytics(
            total_conversations=total_conversations,
            average_quality=float(average_quality),
            quality_distribution=quality_distribution,
            tier_performance=tier_performance,
            trend_data=trend_data,
            anomalies=anomalies,
            recommendations=recommendations
        )
    
    def _generate_recommendations(self, df: pd.DataFrame, avg_quality: float, tier_performance: Dict[str, float]) -> List[str]:
        """Generate quality improvement recommendations."""
        recommendations = []
        
        # Overall quality assessment
        if avg_quality < 0.6:
            recommendations.append("🚨 Overall quality is below acceptable threshold (0.6). Immediate attention required.")
        elif avg_quality < 0.7:
            recommendations.append("⚠️ Overall quality could be improved. Consider quality enhancement measures.")
        else:
            recommendations.append("✅ Overall quality is good. Continue monitoring for consistency.")
        
        # Tier-specific recommendations
        for tier, quality in tier_performance.items():
            if quality < 0.5:
                recommendations.append(f"🔴 Tier {tier} quality ({quality:.3f}) is critically low. Review data sources.")
            elif quality < 0.6:
                recommendations.append(f"🟡 Tier {tier} quality ({quality:.3f}) needs improvement.")
        
        # Component-specific analysis
        component_scores = {
            'therapeutic_accuracy': df['therapeutic_accuracy'].mean(),
            'conversation_coherence': df['conversation_coherence'].mean(),
            'emotional_authenticity': df['emotional_authenticity'].mean(),
            'clinical_compliance': df['clinical_compliance'].mean(),
            'personality_consistency': df['personality_consistency'].mean(),
            'language_quality': df['language_quality'].mean(),
            'safety_score': df['safety_score'].mean()
        }
        
        lowest_component = min(component_scores.items(), key=lambda x: x[1])
        if lowest_component[1] < 0.6:
            recommendations.append(f"📊 Focus on improving {lowest_component[0].replace('_', ' ')} (score: {lowest_component[1]:.3f})")
        
        # Dataset-specific recommendations
        dataset_quality = df.groupby('dataset_name')['overall_quality'].mean()
        worst_dataset = dataset_quality.idxmin()
        if dataset_quality[worst_dataset] < 0.5:
            recommendations.append(f"📁 Dataset '{worst_dataset}' has poor quality ({dataset_quality[worst_dataset]:.3f}). Consider data cleaning.")
        
        return recommendations
    
    def create_quality_overview_chart(self, analytics: QualityAnalytics) -> go.Figure:
        """Create quality overview visualization."""
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('Quality Distribution', 'Tier Performance', 'Quality Trend', 'Component Breakdown'),
            specs=[[{"type": "pie"}, {"type": "bar"}],
                   [{"type": "scatter"}, {"type": "bar"}]]
        )
        
        # Quality distribution pie chart
        if analytics.quality_distribution:
            labels = list(analytics.quality_distribution.keys())
            values = list(analytics.quality_distribution.values())
            
            fig.add_trace(
                go.Pie(
                    labels=labels,
                    values=values,
                    marker_colors=self.color_schemes['quality_levels'],
                    name="Quality Distribution"
                ),
                row=1, col=1
            )
        
        # Tier performance bar chart
        if analytics.tier_performance:
            tiers = list(analytics.tier_performance.keys())
            performance = list(analytics.tier_performance.values())
            
            fig.add_trace(
                go.Bar(
                    x=tiers,
                    y=performance,
                    marker_color=self.color_schemes['tiers'][:len(tiers)],
                    name="Tier Performance"
                ),
                row=1, col=2
            )
        
        # Quality trend line chart
        if analytics.trend_data:
            dates = [item['date'] for item in analytics.trend_data]
            qualities = [item['quality'] for item in analytics.trend_data]
            
            fig.add_trace(
                go.Scatter(
                    x=dates,
                    y=qualities,
                    mode='lines+markers',
                    line=dict(color=self.color_schemes['trends']),
                    name="Quality Trend"
                ),
                row=2, col=1
            )
        
        # Component breakdown (placeholder - would need component data)
        components = ['Therapeutic', 'Coherence', 'Authenticity', 'Clinical', 'Consistency', 'Language', 'Safety']
        scores = [0.7, 0.75, 0.68, 0.72, 0.69, 0.74, 0.8]  # Placeholder values
        
        fig.add_trace(
            go.Bar(
                x=components,
                y=scores,
                marker_color='lightblue',
                name="Component Scores"
            ),
            row=2, col=2
        )
        
        fig.update_layout(
            height=800,
            title_text="Quality Analytics Overview",
            showlegend=False
        )
        
        return fig
    
    def create_detailed_quality_heatmap(self, df: pd.DataFrame) -> go.Figure:
        """Create detailed quality metrics heatmap."""
        if df.empty:
            return go.Figure()
        
        # Quality components
        quality_components = [
            'therapeutic_accuracy',
            'conversation_coherence', 
            'emotional_authenticity',
            'clinical_compliance',
            'personality_consistency',
            'language_quality',
            'safety_score'
        ]
        
        # Calculate correlation matrix
        correlation_data = df[quality_components].corr()
        
        fig = go.Figure(data=go.Heatmap(
            z=correlation_data.values,
            x=correlation_data.columns,
            y=correlation_data.columns,
            colorscale='RdYlBu',
            zmid=0,
            text=correlation_data.round(3).values,
            texttemplate="%{text}",
            textfont={"size": 10},
            hoverongaps=False
        ))
        
        fig.update_layout(
            title="Quality Components Correlation Matrix",
            xaxis_title="Quality Components",
            yaxis_title="Quality Components",
            height=600
        )
        
        return fig
    
    def create_anomaly_detection_chart(self, analytics: QualityAnalytics) -> go.Figure:
        """Create anomaly detection visualization."""
        if not analytics.anomalies:
            return go.Figure()
        
        anomalies_df = pd.DataFrame(analytics.anomalies)
        
        fig = px.scatter(
            anomalies_df,
            x='date',
            y='quality',
            color='tier',
            size=[1] * len(anomalies_df),
            hover_data=['dataset', 'id'],
            title="Quality Anomalies Detection",
            labels={'quality': 'Quality Score', 'date': 'Date'}
        )
        
        # Add threshold line
        fig.add_hline(
            y=0.6,
            line_dash="dash",
            line_color="red",
            annotation_text="Quality Threshold"
        )
        
        fig.update_layout(height=500)
        
        return fig
    
    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Quality Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        st.title("📊 Quality Analytics Dashboard")
        st.markdown("**Enterprise-grade quality analytics and visualization**")
        
        # Sidebar controls
        st.sidebar.header("Dashboard Controls")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data"):
            st.cache_data.clear()
            st.rerun()
        
        # Date range filter
        date_range = st.sidebar.date_input(
            "Date Range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now()
        )
        
        # Tier filter
        available_tiers = ['All', 'priority_1', 'priority_2', 'priority_3', 'professional', 'research']
        selected_tiers = st.sidebar.multiselect(
            "Select Tiers",
            available_tiers,
            default=['All']
        )
        
        # Quality threshold
        quality_threshold = st.sidebar.slider(
            "Quality Threshold",
            min_value=0.0,
            max_value=1.0,
            value=0.6,
            step=0.1
        )
        
        # Load and process data
        with st.spinner("Loading quality data..."):
            df = self.load_quality_data()
            
            if df.empty:
                st.error("❌ No quality data available. Please ensure quality validation has been run.")
                return
            
            # Apply filters
            if 'All' not in selected_tiers:
                df = df[df['tier'].isin(selected_tiers)]
            
            if len(date_range) == 2:
                start_date, end_date = date_range
                df = df[
                    (df['created_at'].dt.date >= start_date) & 
                    (df['created_at'].dt.date <= end_date)
                ]
            
            df = df[df['overall_quality'] >= quality_threshold]
        
        # Calculate analytics
        analytics = self.calculate_quality_analytics(df)
        
        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Conversations",
                f"{analytics.total_conversations:,}",
                delta=None
            )
        
        with col2:
            st.metric(
                "Average Quality",
                f"{analytics.average_quality:.3f}",
                delta=f"{analytics.average_quality - 0.6:.3f}" if analytics.average_quality > 0.6 else None
            )
        
        with col3:
            excellent_count = analytics.quality_distribution.get('Excellent', 0)
            total = sum(analytics.quality_distribution.values()) if analytics.quality_distribution else 1
            excellent_pct = (excellent_count / total) * 100
            st.metric(
                "Excellent Quality %",
                f"{excellent_pct:.1f}%",
                delta=None
            )
        
        with col4:
            anomaly_count = len(analytics.anomalies)
            st.metric(
                "Quality Anomalies",
                anomaly_count,
                delta=None
            )
        
        # Main visualizations
        st.header("📈 Quality Overview")
        overview_chart = self.create_quality_overview_chart(analytics)
        st.plotly_chart(overview_chart, use_container_width=True)
        
        # Detailed analysis
        col1, col2 = st.columns(2)
        
        with col1:
            st.header("🔥 Quality Heatmap")
            heatmap_chart = self.create_detailed_quality_heatmap(df)
            st.plotly_chart(heatmap_chart, use_container_width=True)
        
        with col2:
            st.header("⚠️ Anomaly Detection")
            anomaly_chart = self.create_anomaly_detection_chart(analytics)
            st.plotly_chart(anomaly_chart, use_container_width=True)
        
        # Recommendations
        st.header("💡 Quality Recommendations")
        for i, recommendation in enumerate(analytics.recommendations, 1):
            st.write(f"{i}. {recommendation}")
        
        # Data table
        if st.checkbox("Show Raw Data"):
            st.header("📋 Quality Data Table")
            st.dataframe(
                df[['id', 'tier', 'dataset_name', 'overall_quality', 'created_at']].head(100),
                use_container_width=True
            )
        
        # Export functionality
        st.header("📤 Export Data")
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Export Analytics Report"):
                report_data = {
                    'analytics': analytics.__dict__,
                    'generated_at': datetime.now().isoformat(),
                    'filters_applied': {
                        'tiers': selected_tiers,
                        'date_range': [str(d) for d in date_range] if len(date_range) == 2 else None,
                        'quality_threshold': quality_threshold
                    }
                }
                
                st.download_button(
                    label="Download JSON Report",
                    data=json.dumps(report_data, indent=2),
                    file_name=f"quality_analytics_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("Export Quality Data"):
                csv_data = df.to_csv(index=False)
                st.download_button(
                    label="Download CSV Data",
                    data=csv_data,
                    file_name=f"quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )

def main():
    """Main function to run the dashboard."""
    dashboard = QualityAnalyticsDashboard()
    dashboard.run_dashboard()

if __name__ == "__main__":
    main()
