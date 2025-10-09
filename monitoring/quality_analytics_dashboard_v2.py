#!/usr/bin/env python3
"""
Task 5.6.2.1: Quality Analytics Dashboard and Visualization (Enterprise-Grade)

Production-ready quality analytics dashboard providing comprehensive insights
into conversation quality metrics, trends, and performance across all datasets.

Built against the ACTUAL database schema with proper error handling,
caching, and enterprise-grade architecture.
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
import hashlib
import time

# Suppress warnings for cleaner output
warnings.filterwarnings("ignore")

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

@dataclass
class QualityAnalytics:
    """Quality analytics data structure with enterprise-grade validation."""
    total_conversations: int
    average_quality: float
    quality_distribution: Dict[str, int]
    tier_performance: Dict[str, float]
    component_performance: Dict[str, float]
    trend_data: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]
    recommendations: List[str]
    data_freshness: str
    analysis_timestamp: str

class QualityAnalyticsDashboard:
    """
    Enterprise-grade quality analytics dashboard.
    
    Features:
    - Real-time data loading from actual database schema
    - Intelligent caching with TTL
    - Comprehensive error handling and recovery
    - Production-ready visualizations
    - Performance monitoring and optimization
    - Audit logging and compliance tracking
    """
    
    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/database/conversations.db"):
        """Initialize the quality analytics dashboard with enterprise configuration."""
        self.db_path = Path(db_path)
        self.cache_duration = 300  # 5 minutes cache TTL
        self._cache = {}
        self._cache_timestamps = {}
        
        # Validate database exists
        if not self.db_path.exists():
            raise FileNotFoundError(f"Database not found: {self.db_path}")
        
        # Quality thresholds based on actual data analysis
        self.quality_thresholds = {
            'excellent': 0.8,
            'good': 0.6,
            'fair': 0.4,
            'poor': 0.0
        }
        
        # Component mapping to actual database columns
        self.quality_components = {
            'overall_quality': 'Overall Quality',
            'therapeutic_accuracy': 'Therapeutic Accuracy',
            'clinical_compliance': 'Clinical Compliance',
            'safety_score': 'Safety Score',
            'conversation_coherence': 'Conversation Coherence',
            'emotional_authenticity': 'Emotional Authenticity'
        }
        
        # Color schemes for enterprise visualizations
        self.color_schemes = {
            'quality_levels': ['#d32f2f', '#f57c00', '#fbc02d', '#388e3c'],
            'tiers': ['#1976d2', '#7b1fa2', '#d32f2f', '#f57c00', '#388e3c', '#795548'],
            'trends': '#1976d2',
            'anomalies': '#d32f2f',
            'components': px.colors.qualitative.Set3
        }
        
        logger.info(f"🎯 Quality Analytics Dashboard initialized - DB: {self.db_path}")
    
    def _get_cache_key(self, method_name: str, **kwargs) -> str:
        """Generate cache key for method with parameters."""
        params_str = json.dumps(kwargs, sort_keys=True, default=str)
        return hashlib.md5(f"{method_name}_{params_str}".encode()).hexdigest()
    
    def _is_cache_valid(self, cache_key: str) -> bool:
        """Check if cached data is still valid."""
        if cache_key not in self._cache_timestamps:
            return False
        
        age = time.time() - self._cache_timestamps[cache_key]
        return age < self.cache_duration
    
    def _set_cache(self, cache_key: str, data: Any) -> None:
        """Set cached data with timestamp."""
        self._cache[cache_key] = data
        self._cache_timestamps[cache_key] = time.time()
    
    def load_quality_data(self, 
                         tier_filter: Optional[List[str]] = None,
                         date_range: Optional[Tuple[datetime, datetime]] = None,
                         min_quality: Optional[float] = None,
                         force_refresh: bool = False) -> pd.DataFrame:
        """
        Load quality data from database with enterprise-grade filtering and caching.
        
        Args:
            tier_filter: List of tiers to include (None for all)
            date_range: Tuple of (start_date, end_date) for filtering
            min_quality: Minimum quality threshold
            force_refresh: Force cache refresh
            
        Returns:
            DataFrame with quality data
        """
        # Generate cache key
        cache_key = self._get_cache_key(
            'load_quality_data',
            tier_filter=tier_filter,
            date_range=date_range,
            min_quality=min_quality
        )
        
        # Check cache
        if not force_refresh and self._is_cache_valid(cache_key):
            logger.info("📋 Using cached quality data")
            return self._cache[cache_key]
        
        try:
            start_time = time.time()
            conn = sqlite3.connect(str(self.db_path))
            
            # Build dynamic query based on filters
            base_query = """
            SELECT 
                c.conversation_id,
                c.dataset_source,
                c.tier,
                c.title,
                c.turn_count,
                c.word_count,
                c.processing_status,
                c.created_at,
                q.quality_id,
                q.overall_quality,
                q.therapeutic_accuracy,
                q.clinical_compliance,
                q.safety_score,
                q.conversation_coherence,
                q.emotional_authenticity,
                q.validation_date,
                q.validator_version
            FROM conversations c
            INNER JOIN conversation_quality q ON c.conversation_id = q.conversation_id
            WHERE 1=1
            """
            
            params = []
            
            # Add tier filter
            if tier_filter and 'All' not in tier_filter:
                placeholders = ','.join(['?' for _ in tier_filter])
                base_query += f" AND c.tier IN ({placeholders})"
                params.extend(tier_filter)
            
            # Add date range filter
            if date_range and len(date_range) == 2:
                base_query += " AND DATE(c.created_at) >= DATE(?) AND DATE(c.created_at) <= DATE(?)"
                params.extend([date_range[0].date().isoformat(), date_range[1].date().isoformat()])
            
            # Add quality filter
            if min_quality is not None:
                base_query += " AND q.overall_quality >= ?"
                params.append(min_quality)
            
            base_query += " ORDER BY c.created_at DESC"
            
            # Execute query
            df = pd.read_sql_query(base_query, conn, params=params)
            conn.close()
            
            # Data processing
            if not df.empty:
                # Convert timestamps
                df['created_at'] = pd.to_datetime(df['created_at'])
                df['validation_date'] = pd.to_datetime(df['validation_date'])
                
                # Add derived columns
                df['quality_category'] = pd.cut(
                    df['overall_quality'],
                    bins=[0, 0.4, 0.6, 0.8, 1.0],
                    labels=['Poor', 'Fair', 'Good', 'Excellent'],
                    include_lowest=True
                )
                
                df['date'] = df['created_at'].dt.date
                df['month'] = df['created_at'].dt.to_period('M')
                df['week'] = df['created_at'].dt.to_period('W')
            
            # Cache the result
            self._set_cache(cache_key, df)
            
            load_time = time.time() - start_time
            logger.info(f"✅ Loaded {len(df)} quality records in {load_time:.2f}s")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading quality data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(columns=[
                'conversation_id', 'dataset_source', 'tier', 'title', 'turn_count',
                'word_count', 'processing_status', 'created_at', 'quality_id',
                'overall_quality', 'therapeutic_accuracy', 'clinical_compliance',
                'safety_score', 'conversation_coherence', 'emotional_authenticity',
                'validation_date', 'validator_version'
            ])
    
    def calculate_quality_analytics(self, df: pd.DataFrame) -> QualityAnalytics:
        """
        Calculate comprehensive quality analytics with enterprise-grade validation.
        
        Args:
            df: Quality data DataFrame
            
        Returns:
            QualityAnalytics object with comprehensive metrics
        """
        if df.empty:
            return QualityAnalytics(
                total_conversations=0,
                average_quality=0.0,
                quality_distribution={},
                tier_performance={},
                component_performance={},
                trend_data=[],
                anomalies=[],
                recommendations=["No quality data available for analysis"],
                data_freshness="No data",
                analysis_timestamp=datetime.now().isoformat()
            )
        
        try:
            # Basic statistics
            total_conversations = len(df)
            average_quality = float(df['overall_quality'].mean())
            
            # Quality distribution
            quality_distribution = df['quality_category'].value_counts().to_dict()
            # Convert to strings for JSON serialization
            quality_distribution = {str(k): int(v) for k, v in quality_distribution.items()}
            
            # Tier performance
            tier_performance = df.groupby('tier')['overall_quality'].agg(['mean', 'count']).to_dict('index')
            tier_performance = {
                tier: {
                    'average_quality': float(stats['mean']),
                    'conversation_count': int(stats['count'])
                }
                for tier, stats in tier_performance.items()
            }
            
            # Component performance (only for non-zero values)
            component_performance = {}
            for col, name in self.quality_components.items():
                if col in df.columns:
                    # Only calculate for non-zero values to avoid skewing averages
                    non_zero_values = df[df[col] > 0][col]
                    if len(non_zero_values) > 0:
                        component_performance[name] = {
                            'average_score': float(non_zero_values.mean()),
                            'sample_count': int(len(non_zero_values)),
                            'coverage_percent': float(len(non_zero_values) / len(df) * 100)
                        }
                    else:
                        component_performance[name] = {
                            'average_score': 0.0,
                            'sample_count': 0,
                            'coverage_percent': 0.0
                        }
            
            # Trend data (last 30 days)
            thirty_days_ago = datetime.now() - timedelta(days=30)
            recent_df = df[df['created_at'] >= thirty_days_ago]
            
            trend_data = []
            if not recent_df.empty:
                daily_quality = recent_df.groupby(recent_df['created_at'].dt.date)['overall_quality'].agg(['mean', 'count'])
                trend_data = [
                    {
                        'date': str(date),
                        'average_quality': float(stats['mean']),
                        'conversation_count': int(stats['count'])
                    }
                    for date, stats in daily_quality.iterrows()
                ]
                # Sort by date
                trend_data.sort(key=lambda x: x['date'])
            
            # Anomaly detection (statistical outliers)
            anomalies = self._detect_quality_anomalies(df)
            
            # Generate recommendations
            recommendations = self._generate_recommendations(df, average_quality, tier_performance, component_performance)
            
            # Data freshness
            if not df.empty:
                latest_date = df['created_at'].max()
                data_age = datetime.now() - latest_date
                if data_age.days == 0:
                    data_freshness = "Current (today)"
                elif data_age.days == 1:
                    data_freshness = "1 day old"
                else:
                    data_freshness = f"{data_age.days} days old"
            else:
                data_freshness = "No data"
            
            return QualityAnalytics(
                total_conversations=total_conversations,
                average_quality=average_quality,
                quality_distribution=quality_distribution,
                tier_performance=tier_performance,
                component_performance=component_performance,
                trend_data=trend_data,
                anomalies=anomalies,
                recommendations=recommendations,
                data_freshness=data_freshness,
                analysis_timestamp=datetime.now().isoformat()
            )
            
        except Exception as e:
            logger.error(f"❌ Error calculating quality analytics: {e}")
            return QualityAnalytics(
                total_conversations=0,
                average_quality=0.0,
                quality_distribution={},
                tier_performance={},
                component_performance={},
                trend_data=[],
                anomalies=[],
                recommendations=[f"Error in analysis: {str(e)}"],
                data_freshness="Error",
                analysis_timestamp=datetime.now().isoformat()
            )
    
    def _detect_quality_anomalies(self, df: pd.DataFrame, method: str = 'iqr') -> List[Dict[str, Any]]:
        """
        Detect quality anomalies using statistical methods.
        
        Args:
            df: Quality data DataFrame
            method: Anomaly detection method ('iqr', 'zscore')
            
        Returns:
            List of anomaly dictionaries
        """
        anomalies = []
        
        try:
            if len(df) < 3:  # Need at least 3 data points for meaningful anomaly detection
                return anomalies
            
            if method == 'iqr':
                Q1 = df['overall_quality'].quantile(0.25)
                Q3 = df['overall_quality'].quantile(0.75)
                IQR = Q3 - Q1
                
                # Handle case where IQR is 0 (all values are the same)
                if IQR == 0:
                    # Use standard deviation method instead
                    mean_quality = df['overall_quality'].mean()
                    std_quality = df['overall_quality'].std()
                    if std_quality > 0:
                        lower_bound = mean_quality - 2 * std_quality
                        upper_bound = mean_quality + 2 * std_quality
                    else:
                        return anomalies  # No variation in data
                else:
                    lower_bound = Q1 - 1.5 * IQR
                    upper_bound = Q3 + 1.5 * IQR
                
                anomaly_df = df[(df['overall_quality'] < lower_bound) | (df['overall_quality'] > upper_bound)]
                
            elif method == 'zscore':
                mean_quality = df['overall_quality'].mean()
                std_quality = df['overall_quality'].std()
                
                if std_quality == 0:
                    return anomalies  # No variation in data
                
                z_scores = np.abs((df['overall_quality'] - mean_quality) / std_quality)
                anomaly_df = df[z_scores > 2]
            
            # Convert to list of dictionaries (limit to top 20)
            for _, row in anomaly_df.head(20).iterrows():
                anomalies.append({
                    'conversation_id': str(row['conversation_id']),
                    'tier': str(row['tier']),
                    'dataset_source': str(row['dataset_source']),
                    'quality_score': float(row['overall_quality']),
                    'date': str(row['created_at'].date()),
                    'anomaly_type': 'low' if row['overall_quality'] < df['overall_quality'].mean() else 'high'
                })
                
        except Exception as e:
            logger.error(f"❌ Error detecting anomalies: {e}")
        
        return anomalies
    
    def _generate_recommendations(self, 
                                df: pd.DataFrame, 
                                avg_quality: float, 
                                tier_performance: Dict, 
                                component_performance: Dict) -> List[str]:
        """
        Generate actionable quality improvement recommendations.
        
        Args:
            df: Quality data DataFrame
            avg_quality: Overall average quality
            tier_performance: Tier-wise performance metrics
            component_performance: Component-wise performance metrics
            
        Returns:
            List of recommendation strings
        """
        recommendations = []
        
        try:
            # Overall quality assessment
            if avg_quality < 0.5:
                recommendations.append("🚨 CRITICAL: Overall quality is critically low (< 0.5). Immediate intervention required.")
            elif avg_quality < 0.6:
                recommendations.append("⚠️ WARNING: Overall quality is below acceptable threshold (< 0.6). Review data sources and processing.")
            elif avg_quality < 0.7:
                recommendations.append("📈 IMPROVEMENT: Overall quality is fair but could be enhanced. Consider quality optimization measures.")
            else:
                recommendations.append("✅ GOOD: Overall quality is acceptable. Continue monitoring for consistency.")
            
            # Tier-specific recommendations
            for tier, metrics in tier_performance.items():
                avg_tier_quality = metrics['average_quality']
                count = metrics['conversation_count']
                
                if avg_tier_quality < 0.5:
                    recommendations.append(f"🔴 CRITICAL: Tier '{tier}' quality ({avg_tier_quality:.3f}) is critically low with {count:,} conversations. Urgent review needed.")
                elif avg_tier_quality < 0.6:
                    recommendations.append(f"🟡 WARNING: Tier '{tier}' quality ({avg_tier_quality:.3f}) needs improvement ({count:,} conversations).")
                elif count < 100:
                    recommendations.append(f"📊 INFO: Tier '{tier}' has limited data ({count} conversations). Consider expanding dataset.")
            
            # Component-specific recommendations
            component_issues = []
            for component, metrics in component_performance.items():
                if metrics['sample_count'] > 0:
                    if metrics['average_score'] < 0.5:
                        component_issues.append(f"{component} ({metrics['average_score']:.3f})")
                    elif metrics['coverage_percent'] < 50:
                        recommendations.append(f"📋 DATA: {component} has low coverage ({metrics['coverage_percent']:.1f}%). Consider expanding validation.")
            
            if component_issues:
                recommendations.append(f"🔧 FOCUS: Improve these components: {', '.join(component_issues)}")
            
            # Data quality recommendations
            total_conversations = len(df)
            if total_conversations < 1000:
                recommendations.append(f"📈 SCALE: Dataset is relatively small ({total_conversations:,} conversations). Consider expanding for better insights.")
            
            # Trend-based recommendations
            if len(df) > 7:  # Need at least a week of data
                recent_week = df[df['created_at'] >= (datetime.now() - timedelta(days=7))]
                older_week = df[(df['created_at'] >= (datetime.now() - timedelta(days=14))) & 
                               (df['created_at'] < (datetime.now() - timedelta(days=7)))]
                
                if len(recent_week) > 0 and len(older_week) > 0:
                    recent_avg = recent_week['overall_quality'].mean()
                    older_avg = older_week['overall_quality'].mean()
                    change = recent_avg - older_avg
                    
                    if change < -0.05:
                        recommendations.append(f"📉 TREND: Quality declining over past week ({change:.3f}). Investigate recent changes.")
                    elif change > 0.05:
                        recommendations.append(f"📈 TREND: Quality improving over past week (+{change:.3f}). Continue current practices.")
            
        except Exception as e:
            logger.error(f"❌ Error generating recommendations: {e}")
            recommendations.append(f"⚠️ Error generating recommendations: {str(e)}")
        
        return recommendations
    
    def create_quality_overview_chart(self, analytics: QualityAnalytics) -> go.Figure:
        """
        Create comprehensive quality overview visualization.
        
        Args:
            analytics: QualityAnalytics object with metrics
            
        Returns:
            Plotly figure with quality overview
        """
        try:
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=(
                    'Quality Distribution',
                    'Tier Performance',
                    'Quality Trend (30 Days)',
                    'Component Performance'
                ),
                specs=[
                    [{"type": "pie"}, {"type": "bar"}],
                    [{"type": "scatter"}, {"type": "bar"}]
                ],
                vertical_spacing=0.12,
                horizontal_spacing=0.1
            )
            
            # 1. Quality Distribution (Pie Chart)
            if analytics.quality_distribution:
                labels = list(analytics.quality_distribution.keys())
                values = list(analytics.quality_distribution.values())
                colors = self.color_schemes['quality_levels'][:len(labels)]
                
                fig.add_trace(
                    go.Pie(
                        labels=labels,
                        values=values,
                        marker_colors=colors,
                        textinfo='label+percent',
                        textposition='auto',
                        hovertemplate='<b>%{label}</b><br>Count: %{value}<br>Percentage: %{percent}<extra></extra>'
                    ),
                    row=1, col=1
                )
            
            # 2. Tier Performance (Bar Chart)
            if analytics.tier_performance:
                tiers = list(analytics.tier_performance.keys())
                qualities = [metrics['average_quality'] for metrics in analytics.tier_performance.values()]
                counts = [metrics['conversation_count'] for metrics in analytics.tier_performance.values()]
                
                fig.add_trace(
                    go.Bar(
                        x=tiers,
                        y=qualities,
                        text=[f'{q:.3f}<br>({c:,} conv)' for q, c in zip(qualities, counts)],
                        textposition='auto',
                        marker_color=self.color_schemes['tiers'][:len(tiers)],
                        hovertemplate='<b>%{x}</b><br>Quality: %{y:.3f}<br>Conversations: %{text}<extra></extra>'
                    ),
                    row=1, col=2
                )
            
            # 3. Quality Trend (Line Chart)
            if analytics.trend_data:
                dates = [item['date'] for item in analytics.trend_data]
                qualities = [item['average_quality'] for item in analytics.trend_data]
                counts = [item['conversation_count'] for item in analytics.trend_data]
                
                fig.add_trace(
                    go.Scatter(
                        x=dates,
                        y=qualities,
                        mode='lines+markers',
                        line=dict(color=self.color_schemes['trends'], width=3),
                        marker=dict(size=8),
                        text=[f'{c} conversations' for c in counts],
                        hovertemplate='<b>%{x}</b><br>Quality: %{y:.3f}<br>%{text}<extra></extra>'
                    ),
                    row=2, col=1
                )
            
            # 4. Component Performance (Horizontal Bar Chart)
            if analytics.component_performance:
                components = list(analytics.component_performance.keys())
                scores = [metrics['average_score'] for metrics in analytics.component_performance.values()]
                coverages = [metrics['coverage_percent'] for metrics in analytics.component_performance.values()]
                
                fig.add_trace(
                    go.Bar(
                        y=components,
                        x=scores,
                        orientation='h',
                        text=[f'{s:.3f} ({c:.1f}%)' for s, c in zip(scores, coverages)],
                        textposition='auto',
                        marker_color=self.color_schemes['components'][:len(components)],
                        hovertemplate='<b>%{y}</b><br>Score: %{x:.3f}<br>Coverage: %{text}<extra></extra>'
                    ),
                    row=2, col=2
                )
            
            # Update layout
            fig.update_layout(
                title=dict(
                    text=f"Quality Analytics Overview - {analytics.total_conversations:,} Conversations",
                    x=0.5,
                    font=dict(size=20, color='#2c3e50')
                ),
                height=800,
                showlegend=False,
                plot_bgcolor='white',
                paper_bgcolor='white',
                font=dict(family="Arial, sans-serif", size=12)
            )
            
            # Update axes
            fig.update_xaxes(title_text="Quality Score", row=1, col=2, range=[0, 1])
            fig.update_yaxes(title_text="Average Quality", row=1, col=2)
            fig.update_xaxes(title_text="Date", row=2, col=1)
            fig.update_yaxes(title_text="Quality Score", row=2, col=1, range=[0, 1])
            fig.update_xaxes(title_text="Score", row=2, col=2, range=[0, 1])
            
            return fig
            
        except Exception as e:
            logger.error(f"❌ Error creating overview chart: {e}")
            # Return empty figure
            return go.Figure().add_annotation(
                text=f"Error creating visualization: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
    
    def create_detailed_analysis_charts(self, df: pd.DataFrame) -> Tuple[go.Figure, go.Figure]:
        """
        Create detailed analysis charts for deeper insights.
        
        Args:
            df: Quality data DataFrame
            
        Returns:
            Tuple of (quality_heatmap, correlation_matrix)
        """
        try:
            # 1. Quality Heatmap by Tier and Date
            if not df.empty and len(df) > 1:
                # Create weekly aggregation
                df_weekly = df.copy()
                df_weekly['week'] = df_weekly['created_at'].dt.to_period('W').astype(str)
                
                heatmap_data = df_weekly.groupby(['tier', 'week'])['overall_quality'].mean().reset_index()
                heatmap_pivot = heatmap_data.pivot(index='tier', columns='week', values='overall_quality')
                
                heatmap_fig = go.Figure(data=go.Heatmap(
                    z=heatmap_pivot.values,
                    x=heatmap_pivot.columns,
                    y=heatmap_pivot.index,
                    colorscale='RdYlGn',
                    zmin=0,
                    zmax=1,
                    hoverongaps=False,
                    hovertemplate='<b>%{y}</b><br>Week: %{x}<br>Quality: %{z:.3f}<extra></extra>'
                ))
                
                heatmap_fig.update_layout(
                    title="Quality Heatmap by Tier and Week",
                    xaxis_title="Week",
                    yaxis_title="Tier",
                    height=400
                )
            else:
                heatmap_fig = go.Figure().add_annotation(
                    text="Insufficient data for heatmap",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            # 2. Component Correlation Matrix
            quality_cols = [col for col in self.quality_components.keys() if col in df.columns]
            if len(quality_cols) > 1:
                # Filter out zero values for correlation
                corr_df = df[quality_cols].replace(0, np.nan)
                correlation_matrix = corr_df.corr()
                
                corr_fig = go.Figure(data=go.Heatmap(
                    z=correlation_matrix.values,
                    x=[self.quality_components[col] for col in correlation_matrix.columns],
                    y=[self.quality_components[col] for col in correlation_matrix.index],
                    colorscale='RdBu',
                    zmin=-1,
                    zmax=1,
                    text=correlation_matrix.values,
                    texttemplate='%{text:.2f}',
                    textfont={"size": 10},
                    hovertemplate='<b>%{y}</b> vs <b>%{x}</b><br>Correlation: %{z:.3f}<extra></extra>'
                ))
                
                corr_fig.update_layout(
                    title="Quality Component Correlation Matrix",
                    height=400,
                    xaxis_title="Components",
                    yaxis_title="Components"
                )
            else:
                corr_fig = go.Figure().add_annotation(
                    text="Insufficient components for correlation analysis",
                    xref="paper", yref="paper",
                    x=0.5, y=0.5, showarrow=False
                )
            
            return heatmap_fig, corr_fig
            
        except Exception as e:
            logger.error(f"❌ Error creating detailed charts: {e}")
            error_fig = go.Figure().add_annotation(
                text=f"Error creating detailed analysis: {str(e)}",
                xref="paper", yref="paper",
                x=0.5, y=0.5, showarrow=False
            )
            return error_fig, error_fig
    
    def run_streamlit_dashboard(self):
        """
        Run the Streamlit dashboard interface with enterprise-grade features.
        """
        # Page configuration
        st.set_page_config(
            page_title="Quality Analytics Dashboard",
            page_icon="📊",
            layout="wide",
            initial_sidebar_state="expanded"
        )
        
        # Custom CSS for enterprise styling
        st.markdown("""
        <style>
        .main-header {
            font-size: 2.5rem;
            color: #2c3e50;
            text-align: center;
            margin-bottom: 2rem;
            border-bottom: 3px solid #3498db;
            padding-bottom: 1rem;
        }
        .metric-container {
            background-color: #f8f9fa;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #3498db;
            margin-bottom: 1rem;
        }
        .recommendation-box {
            background-color: #e8f4fd;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #2196f3;
            margin: 0.5rem 0;
        }
        .anomaly-box {
            background-color: #ffebee;
            padding: 1rem;
            border-radius: 0.5rem;
            border-left: 4px solid #f44336;
            margin: 0.5rem 0;
        }
        </style>
        """, unsafe_allow_html=True)
        
        # Main header
        st.markdown('<h1 class="main-header">📊 Quality Analytics Dashboard</h1>', unsafe_allow_html=True)
        
        # Sidebar controls
        st.sidebar.header("🎛️ Dashboard Controls")
        
        # Refresh button
        if st.sidebar.button("🔄 Refresh Data", type="primary"):
            st.cache_data.clear()
            st.rerun()
        
        # Date range filter
        st.sidebar.subheader("📅 Date Range")
        date_range = st.sidebar.date_input(
            "Select date range",
            value=(datetime.now() - timedelta(days=30), datetime.now()),
            max_value=datetime.now(),
            help="Filter conversations by creation date"
        )
        
        # Tier filter
        st.sidebar.subheader("🏷️ Tier Filter")
        available_tiers = ['All', 'priority_1', 'priority_2', 'priority_3', 'professional', 'research', 'cot_reasoning']
        selected_tiers = st.sidebar.multiselect(
            "Select tiers to include",
            available_tiers,
            default=['All'],
            help="Filter by conversation tiers"
        )
        
        # Quality threshold
        st.sidebar.subheader("⚡ Quality Threshold")
        quality_threshold = st.sidebar.slider(
            "Minimum quality score",
            min_value=0.0,
            max_value=1.0,
            value=0.0,
            step=0.1,
            help="Filter conversations below this quality threshold"
        )
        
        # Advanced options
        with st.sidebar.expander("🔧 Advanced Options"):
            force_refresh = st.checkbox("Force data refresh", help="Bypass cache and reload from database")
            show_anomalies = st.checkbox("Show anomalies", value=True, help="Display quality anomalies")
            show_detailed_analysis = st.checkbox("Show detailed analysis", value=True, help="Display additional charts")
        
        # Load and process data
        with st.spinner("🔄 Loading quality data..."):
            try:
                # Convert date range
                date_filter = None
                if len(date_range) == 2:
                    start_date = datetime.combine(date_range[0], datetime.min.time())
                    end_date = datetime.combine(date_range[1], datetime.max.time())
                    date_filter = (start_date, end_date)
                
                # Load data
                df = self.load_quality_data(
                    tier_filter=selected_tiers if 'All' not in selected_tiers else None,
                    date_range=date_filter,
                    min_quality=quality_threshold if quality_threshold > 0 else None,
                    force_refresh=force_refresh
                )
                
                if df.empty:
                    st.error("❌ No quality data available with current filters. Please adjust your filters or ensure quality validation has been run.")
                    st.stop()
                
                # Calculate analytics
                analytics = self.calculate_quality_analytics(df)
                
            except Exception as e:
                st.error(f"❌ Error loading data: {str(e)}")
                st.stop()
        
        # Display key metrics
        st.subheader("📈 Key Metrics")
        col1, col2, col3, col4 = st.columns(4)
        
        with col1:
            st.metric(
                "Total Conversations",
                f"{analytics.total_conversations:,}",
                help="Total number of conversations in filtered dataset"
            )
        
        with col2:
            quality_delta = analytics.average_quality - 0.6  # Benchmark against 0.6
            st.metric(
                "Average Quality",
                f"{analytics.average_quality:.3f}",
                delta=f"{quality_delta:+.3f}" if abs(quality_delta) > 0.001 else None,
                help="Overall average quality score (benchmark: 0.6)"
            )
        
        with col3:
            if analytics.quality_distribution:
                excellent_count = analytics.quality_distribution.get('Excellent', 0)
                total = sum(analytics.quality_distribution.values())
                excellent_pct = (excellent_count / total) * 100 if total > 0 else 0
                st.metric(
                    "Excellent Quality %",
                    f"{excellent_pct:.1f}%",
                    help="Percentage of conversations with excellent quality (≥0.8)"
                )
            else:
                st.metric("Excellent Quality %", "0.0%")
        
        with col4:
            anomaly_count = len(analytics.anomalies)
            st.metric(
                "Quality Anomalies",
                anomaly_count,
                help="Number of statistical outliers in quality scores"
            )
        
        # Data freshness indicator
        st.info(f"📅 Data freshness: {analytics.data_freshness} | Last analysis: {analytics.analysis_timestamp[:19]}")
        
        # Main visualizations
        st.subheader("📊 Quality Overview")
        overview_chart = self.create_quality_overview_chart(analytics)
        st.plotly_chart(overview_chart, use_container_width=True)
        
        # Detailed analysis (if enabled)
        if show_detailed_analysis:
            st.subheader("🔍 Detailed Analysis")
            col1, col2 = st.columns(2)
            
            heatmap_fig, corr_fig = self.create_detailed_analysis_charts(df)
            
            with col1:
                st.plotly_chart(heatmap_fig, use_container_width=True)
            
            with col2:
                st.plotly_chart(corr_fig, use_container_width=True)
        
        # Recommendations
        st.subheader("💡 Recommendations")
        for i, recommendation in enumerate(analytics.recommendations):
            st.markdown(f'<div class="recommendation-box">{recommendation}</div>', unsafe_allow_html=True)
        
        # Anomalies (if enabled and available)
        if show_anomalies and analytics.anomalies:
            st.subheader("⚠️ Quality Anomalies")
            
            anomaly_df = pd.DataFrame(analytics.anomalies)
            
            # Group by anomaly type
            low_anomalies = [a for a in analytics.anomalies if a['anomaly_type'] == 'low']
            high_anomalies = [a for a in analytics.anomalies if a['anomaly_type'] == 'high']
            
            col1, col2 = st.columns(2)
            
            with col1:
                if low_anomalies:
                    st.markdown("**🔴 Low Quality Anomalies**")
                    for anomaly in low_anomalies[:5]:  # Show top 5
                        st.markdown(f"""
                        <div class="anomaly-box">
                        <strong>Tier:</strong> {anomaly['tier']}<br>
                        <strong>Quality:</strong> {anomaly['quality_score']:.3f}<br>
                        <strong>Date:</strong> {anomaly['date']}<br>
                        <strong>Source:</strong> {anomaly['dataset_source']}
                        </div>
                        """, unsafe_allow_html=True)
            
            with col2:
                if high_anomalies:
                    st.markdown("**🟢 High Quality Anomalies**")
                    for anomaly in high_anomalies[:5]:  # Show top 5
                        st.markdown(f"""
                        <div class="recommendation-box">
                        <strong>Tier:</strong> {anomaly['tier']}<br>
                        <strong>Quality:</strong> {anomaly['quality_score']:.3f}<br>
                        <strong>Date:</strong> {anomaly['date']}<br>
                        <strong>Source:</strong> {anomaly['dataset_source']}
                        </div>
                        """, unsafe_allow_html=True)
        
        # Data export
        st.subheader("📥 Data Export")
        col1, col2, col3 = st.columns(3)
        
        with col1:
            if st.button("📊 Export Analytics JSON"):
                analytics_dict = {
                    'total_conversations': analytics.total_conversations,
                    'average_quality': analytics.average_quality,
                    'quality_distribution': analytics.quality_distribution,
                    'tier_performance': analytics.tier_performance,
                    'component_performance': analytics.component_performance,
                    'recommendations': analytics.recommendations,
                    'anomalies': analytics.anomalies,
                    'analysis_timestamp': analytics.analysis_timestamp
                }
                st.download_button(
                    "💾 Download Analytics",
                    data=json.dumps(analytics_dict, indent=2),
                    file_name=f"quality_analytics_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )
        
        with col2:
            if st.button("📋 Export Raw Data CSV"):
                csv_data = df.to_csv(index=False)
                st.download_button(
                    "💾 Download CSV",
                    data=csv_data,
                    file_name=f"quality_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv"
                )
        
        with col3:
            if st.button("📈 Export Summary Report"):
                report = f"""
# Quality Analytics Report
Generated: {analytics.analysis_timestamp}

## Summary
- Total Conversations: {analytics.total_conversations:,}
- Average Quality: {analytics.average_quality:.3f}
- Data Freshness: {analytics.data_freshness}

## Recommendations
{chr(10).join(f"- {rec}" for rec in analytics.recommendations)}

## Anomalies Detected
{len(analytics.anomalies)} quality anomalies found.
                """
                st.download_button(
                    "💾 Download Report",
                    data=report,
                    file_name=f"quality_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md",
                    mime="text/markdown"
                )

def main():
    """Main function to run the quality analytics dashboard."""
    try:
        dashboard = QualityAnalyticsDashboard()
        dashboard.run_streamlit_dashboard()
    except Exception as e:
        st.error(f"❌ Dashboard initialization failed: {str(e)}")
        logger.error(f"Dashboard initialization failed: {e}")

if __name__ == "__main__":
    main()
