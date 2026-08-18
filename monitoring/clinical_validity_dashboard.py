#!/usr/bin/env python3
"""
Clinical Validity Monitoring Dashboard
Provides real-time visualization of clinical validity metrics for the Modern Dataset Project.
"""

import logging
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from typing import Any

import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# Configure logging
logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


class ClinicalValidityDashboard:
    """
    Clinical Validity Monitoring Dashboard for tracking clinical quality metrics
    across the Modern Dataset Project pipeline.
    """

    def __init__(self, db_path: str = "/home/vivi/pixelated/ai/database/conversations.db"):
        """Initialize the clinical validity dashboard."""
        self.db_path = db_path
        self.cache_duration = 60  # 1 minute cache for real-time monitoring
        self._last_cache_time = None
        self._cached_data = None

        # Clinical validity thresholds (from backlog requirements)
        self.quality_thresholds = {
            "excellent": 0.8,
            "good": 0.6,
            "fair": 0.4,
            "poor": 0.0,
        }

        # Color schemes
        self.color_schemes = {
            "clinical_validity": ["#d32f2f", "#f57c00", "#fbc02d", "#388e3c"],
            "trend": "#1976d2",
            "target": "#2e7d32",
            "bottleneck": "#c62828",
        }

        logger.info("🩺 Clinical Validity Dashboard initialized")

    def load_clinical_validity_data(self, force_refresh: bool = False, hours_back: int = 24) -> pd.DataFrame:
        """Load clinical validity data from database with caching."""
        current_time = datetime.now(UTC)

        # Check cache validity
        if (
            not force_refresh
            and self._cached_data is not None
            and self._last_cache_time is not None
            and (current_time - self._last_cache_time).total_seconds() < self.cache_duration
        ):
            return self._cached_data

        try:
            # Connect to database
            conn = sqlite3.connect(self.db_path)

            # Query clinical validity metrics from the last N hours
            query = f"""
            SELECT
                timestamp,
                clinical_validity_score,
                sample_size,
                pipeline_stage,
                data_source,
                notes
            FROM clinical_validity_metrics
            WHERE timestamp >= datetime('now', '-{hours_back} hours')
            ORDER BY timestamp DESC
            """

            df = pd.read_sql_query(query, conn)
            conn.close()

            # Convert timestamp to datetime
            if not df.empty:
                df["timestamp"] = pd.to_datetime(df["timestamp"])

            # Cache the data
            self._cached_data = df
            self._last_cache_time = current_time

            logger.info(f"📊 Loaded {len(df)} clinical validity records from last {hours_back} hours")
            return df

        except Exception as e:
            logger.error(f"❌ Failed to load clinical validity data: {e}")
            # Return empty DataFrame with expected columns
            return pd.DataFrame(
                columns=[
                    "timestamp",
                    "clinical_validity_score",
                    "sample_size",
                    "pipeline_stage",
                    "data_source",
                    "notes",
                ]
            )

    def calculate_current_metrics(self, df: pd.DataFrame) -> dict[str, Any]:
        """Calculate current clinical validity metrics."""
        if df.empty:
            return {
                "current_score": 0.0,
                "sample_size": 0,
                "trend_1h": 0.0,
                "trend_24h": 0.0,
                "valid_samples_24h": 0,
                "total_samples_24h": 0,
                "clinical_validity_rate": 0.0,
                "status": "no_data",
            }

        current_time = datetime.now(UTC)

        # Current score (most recent)
        current_score = df.iloc[0]["clinical_validity_score"] if len(df) > 0 else 0.0

        # Sample size (most recent)
        sample_size = int(df.iloc[0]["sample_size"]) if len(df) > 0 and not pd.isna(df.iloc[0]["sample_size"]) else 0

        # Calculate trends
        trend_1h = 0.0
        trend_24h = 0.0

        if len(df) >= 2:
            # 1-hour trend (compare to 1 hour ago if available)
            one_hour_ago = current_time - timedelta(hours=1)
            recent_data = df[df["timestamp"] >= one_hour_ago]
            if len(recent_data) >= 2:
                oldest_recent = recent_data.iloc[-1]["clinical_validity_score"]
                newest_recent = recent_data.iloc[0]["clinical_validity_score"]
                trend_1h = newest_recent - oldest_recent

            # 24-hour trend
            if len(df) >= 2:
                oldest_24h = df.iloc[-1]["clinical_validity_score"]
                newest_24h = df.iloc[0]["clinical_validity_score"]
                trend_24h = newest_24h - oldest_24h

        # Clinical validity rate (percentage meeting threshold)
        valid_threshold = self.quality_thresholds["good"]  # 0.6
        valid_samples = df[df["clinical_validity_score"] >= valid_threshold].shape[0]
        total_samples = df.shape[0]
        clinical_validity_rate = (valid_samples / total_samples * 100) if total_samples > 0 else 0.0

        # Determine status
        if current_score >= self.quality_thresholds["excellent"]:
            status = "excellent"
        elif current_score >= self.quality_thresholds["good"]:
            status = "good"
        elif current_score >= self.quality_thresholds["fair"]:
            status = "fair"
        else:
            status = "poor"

        return {
            "current_score": float(current_score),
            "sample_size": int(sample_size),
            "trend_1h": float(trend_1h),
            "trend_24h": float(trend_24h),
            "valid_samples_24h": int(valid_samples),
            "total_samples_24h": int(total_samples),
            "clinical_validity_rate": float(clinical_validity_rate),
            "status": status,
        }

    def create_clinical_validity_gauge(self, current_score: float) -> go.Figure:
        """Create a gauge chart showing current clinical validity score."""
        fig = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=current_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "Clinical Validity Score"},
                delta={
                    "reference": self.quality_thresholds["good"],
                    "relative": True,
                    "valueformat": ".1%",
                },
                gauge={
                    "axis": {"range": [None, 1.0], "tickwidth": 1, "tickcolor": "darkblue"},
                    "bar": {"color": "darkblue"},
                    "bgcolor": "white",
                    "borderwidth": 2,
                    "bordercolor": "gray",
                    "steps": [
                        {
                            "range": [0, self.quality_thresholds["poor"]],
                            "color": self.color_schemes["clinical_validity"][0],
                        },
                        {
                            "range": [
                                self.quality_thresholds["poor"],
                                self.quality_thresholds["fair"],
                            ],
                            "color": self.color_schemes["clinical_validity"][1],
                        },
                        {
                            "range": [
                                self.quality_thresholds["fair"],
                                self.quality_thresholds["good"],
                            ],
                            "color": self.color_schemes["clinical_validity"][2],
                        },
                        {
                            "range": [
                                self.quality_thresholds["good"],
                                self.quality_thresholds["excellent"],
                            ],
                            "color": self.color_schemes["clinical_validity"][3],
                        },
                    ],
                    "threshold": {
                        "line": {"color": "red", "width": 4},
                        "thickness": 0.75,
                        "value": self.quality_thresholds["good"],
                    },
                },
            )
        )

        fig.update_layout(
            height=300,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            paper_bgcolor="white",
            font={"color": "darkblue", "family": "Arial"},
        )

        return fig

    def create_trend_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a trend chart showing clinical validity over time."""
        if df.empty:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="No clinical validity data available",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(
                title="Clinical Validity Trend (24 Hours)",
                height=300,
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
            )
            return fig

        # Create trend chart
        fig = go.Figure()

        # Add clinical validity score line
        fig.add_trace(
            go.Scatter(
                x=df["timestamp"],
                y=df["clinical_validity_score"],
                mode="lines+markers",
                name="Clinical Validity Score",
                line={"color": self.color_schemes["trend"], "width": 2},
                marker={"size": 4},
            )
        )

        # Add threshold lines
        fig.add_hline(
            y=self.quality_thresholds["excellent"],
            line_dash="dash",
            line_color=self.color_schemes["clinical_validity"][3],
            annotation_text="Excellent (0.8)",
            annotation_position="top left",
        )

        fig.add_hline(
            y=self.quality_thresholds["good"],
            line_dash="dash",
            line_color=self.color_schemes["clinical_validity"][2],
            annotation_text="Good (0.6) - Target",
            annotation_position="top left",
        )

        fig.add_hline(
            y=self.quality_thresholds["fair"],
            line_dash="dash",
            line_color=self.color_schemes["clinical_validity"][1],
            annotation_text="Fair (0.4)",
            annotation_position="top left",
        )

        fig.add_hline(
            y=self.quality_thresholds["poor"],
            line_dash="dash",
            line_color=self.color_schemes["clinical_validity"][0],
            annotation_text="Poor (0.0)",
            annotation_position="top left",
        )

        fig.update_layout(
            title="Clinical Validity Trend (24 Hours)",
            xaxis_title="Time",
            yaxis_title="Clinical Validity Score",
            yaxis={"range": [0, 1.0]},
            height=300,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            hovermode="x unified",
            legend={"orientation": "h", "yanchor": "bottom", "y": 1.02, "xanchor": "right", "x": 1},
        )

        return fig

    def create_distribution_chart(self, df: pd.DataFrame) -> go.Figure:
        """Create a distribution chart showing clinical validity score distribution."""
        if df.empty or len(df) < 2:
            # Create empty chart with message
            fig = go.Figure()
            fig.add_annotation(
                text="Insufficient data for distribution",
                xref="paper",
                yref="paper",
                x=0.5,
                y=0.5,
                xanchor="center",
                yanchor="middle",
                showarrow=False,
                font_size=16,
            )
            fig.update_layout(
                title="Clinical Validity Score Distribution",
                height=250,
                margin={"l": 20, "r": 20, "t": 50, "b": 20},
            )
            return fig

        # Create histogram
        fig = go.Figure()

        fig.add_trace(
            go.Histogram(
                x=df["clinical_validity_score"],
                nbinsx=20,
                name="Score Distribution",
                marker_color=self.color_schemes["trend"],
                opacity=0.7,
            )
        )

        # Add threshold lines
        fig.add_vline(
            x=self.quality_thresholds["excellent"],
            line_dash="dash",
            line_color=self.color_schemes["clinical_validity"][3],
            annotation_text="Excellent",
            annotation_position="top",
        )

        fig.add_vline(
            x=self.quality_thresholds["good"],
            line_dash="dash",
            line_color=self.color_schemes["clinical_validity"][2],
            annotation_text="Good Target",
            annotation_position="top",
        )

        fig.add_vline(
            x=self.quality_thresholds["fair"],
            line_dash="dash",
            line_color=self.color_schemes["clinical_validity"][1],
            annotation_text="Fair",
            annotation_position="top",
        )

        fig.update_layout(
            title="Clinical Validity Score Distribution",
            xaxis_title="Clinical Validity Score",
            yaxis_title="Frequency",
            height=250,
            margin={"l": 20, "r": 20, "t": 50, "b": 20},
            bargap=0.1,
        )

        return fig

    def run_dashboard(self):
        """Run the Streamlit dashboard."""
        st.set_page_config(
            page_title="Clinical Validity Dashboard",
            page_icon="🩺",
            layout="wide",
            initial_sidebar_state="expanded",
        )

        # Sidebar
        st.sidebar.title("🩺 Clinical Validity Dashboard")
        st.sidebar.markdown("---")

        # Time range selector
        time_range = st.sidebar.selectbox(
            "Time Range",
            options=[1, 6, 12, 24, 48, 168],  # 1h, 6h, 12h, 24h, 48h, 1week
            format_func=lambda x: (
                f"{x} Hour{'s' if x > 1 else ''}"
                if x < 24
                else f"{x // 24} Day{'s' if x > 24 else ''}"
                if x < 168
                else "1 Week"
            ),
            index=3,  # Default to 24 hours
        )

        # Auto-refresh toggle
        auto_refresh = st.sidebar.checkbox("Auto Refresh (30s)", value=True)

        if auto_refresh:
            st.sidebar.write("🔄 Auto-refreshing every 30 seconds")
            # Streamlit's automatic rerun
            st.empty()

        # Manual refresh button
        if st.sidebar.button("🔄 Refresh Now"):
            st.cache_data.clear()
            st.experimental_rerun()

        st.sidebar.markdown("---")
        st.sidebar.markdown(
            """
        ### About
        This dashboard monitors clinical validity metrics
        for the Modern Dataset Project pipeline.

        **Target**: ≥ 0.6 (Good) clinical validity score
        **Current Goal**: Increase from 13.3% to ≥50%
        """
        )

        # Main content
        st.title("🩺 Clinical Validity Monitoring Dashboard")
        st.markdown("*Real-time monitoring of clinical quality metrics for therapeutic AI training data*")

        # Load data
        df = self.load_clinical_validity_data(hours_back=time_range)

        if df.empty:
            st.warning("⚠️ No clinical validity data available. Please check data collection pipeline.")
            st.info("💡 The dashboard will show data once the clinical validity monitoring pipeline is active.")
            return

        # Calculate metrics
        metrics = self.calculate_current_metrics(df)

        # Display key metrics
        col1, col2, col3, col4 = st.columns(4)

        with col1:
            delta_color = "normal" if metrics["trend_1h"] >= 0 else "inverse"
            st.metric(
                label="Current Score",
                value=f"{metrics['current_score']:.3f}",
                delta=f"{metrics['trend_1h']:+.3f} (1h)",
                delta_color=delta_color,
            )

        with col2:
            st.metric(
                label="Sample Size",
                value=f"{metrics['sample_size']:,}",
                help="Number of samples in most recent measurement",
            )

        with col3:
            st.metric(
                label="Validity Rate",
                value=f"{metrics['clinical_validity_rate']:.1f}%",
                help=f"Percentage of samples ≥ {self.quality_thresholds['good']:.1f}",
            )

        with col4:
            status_color = {
                "excellent": "🟢",
                "good": "🟢",
                "fair": "🟡",
                "poor": "🔴",
                "no_data": "⚪",
            }.get(metrics["status"], "⚪")
            st.metric(
                label="Status",
                value=f"{status_color} {metrics['status'].title()}",
                help="Clinical validity status based on current score",
            )

        st.markdown("---")

        # Charts
        chart_col1, chart_col2 = st.columns([2, 1])

        with chart_col1:
            # Trend chart
            trend_fig = self.create_trend_chart(df)
            st.plotly_chart(trend_fig, use_container_width=True)

        with chart_col2:
            # Gauge chart
            gauge_fig = self.create_clinical_validity_gauge(metrics["current_score"])
            st.plotly_chart(gauge_fig, use_container_width=True)

        # Distribution chart
        dist_fig = self.create_distribution_chart(df)
        st.plotly_chart(dist_fig, use_container_width=True)

        # Data table
        st.markdown("---")
        st.subheader("📋 Recent Clinical Validity Data")

        if not df.empty:
            # Format dataframe for display
            display_df = df.copy()
            display_df["timestamp"] = display_df["timestamp"].dt.strftime("%Y-%m-%d %H:%M:%S")
            display_df["clinical_validity_score"] = display_df["clinical_validity_score"].round(3)
            display_df = display_df.rename(
                columns={
                    "timestamp": "Timestamp",
                    "clinical_validity_score": "Score",
                    "sample_size": "Sample Size",
                    "pipeline_stage": "Pipeline Stage",
                    "data_source": "Data Source",
                    "notes": "Notes",
                }
            )

            # Show last 20 records
            st.dataframe(
                display_df.head(20),
                use_container_width=True,
                height=400,
            )
        else:
            st.info("No data to display")

        # Footer
        st.markdown("---")
        st.markdown(
            f"""
        <div style='text-align: center; color: #666; font-size: 0.9em;'>
            Last updated: {datetime.now().strftime("%Y-%m-%d %H:%M:%S")} UTC<br>
            Last updated: {datetime.now(UTC).strftime("%Y-%m-%d %H:%M:%S")} UTC<br>
            Clinical Validity Dashboard v1.0 • Modern Dataset Project
        </div>
        """,
            unsafe_allow_html=True,
        )

        # Auto-refresh every 30 seconds if enabled
        if auto_refresh:
            time.sleep(30)
            st.experimental_rerun()


def main():
    """Main function to run the dashboard."""
    dashboard = ClinicalValidityDashboard()
    dashboard.run_dashboard()


if __name__ == "__main__":
    main()
    main()
    main()
    main()
