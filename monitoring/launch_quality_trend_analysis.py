#!/usr/bin/env python3
"""
Quality Trend Analysis Launcher (Task 5.6.2.2)

Enterprise-grade launcher for the quality trend analysis and reporting system
with comprehensive setup, validation, and execution capabilities.
"""

import sys
import os
import subprocess
import logging
import argparse
from pathlib import Path
from datetime import datetime
import json
import tempfile

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

class QualityTrendAnalysisLauncher:
    """
    Enterprise-grade launcher for quality trend analysis system.
    
    Handles setup, validation, analysis execution, and report generation
    with comprehensive error handling and monitoring.
    """
    
    def __init__(self):
        """Initialize the trend analysis launcher."""
        self.project_root = Path(__file__).parent.parent
        self.monitoring_dir = Path(__file__).parent
        self.db_path = self.project_root / "database" / "conversations.db"
        self.reports_dir = self.monitoring_dir / "reports"
        
        # Ensure reports directory exists
        self.reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Required dependencies
        self.required_packages = [
            'pandas',
            'numpy',
            'sqlite3',
            'scipy',
            'sklearn',
            'plotly',
            'jinja2'
        ]
        
        logger.info("🚀 Quality Trend Analysis Launcher initialized")
    
    def check_dependencies(self) -> bool:
        """Check if all required dependencies are available."""
        logger.info("🔍 Checking dependencies...")
        
        missing_packages = []
        
        for package in self.required_packages:
            try:
                if package == 'sqlite3':
                    import sqlite3
                elif package == 'pandas':
                    import pandas
                elif package == 'numpy':
                    import numpy
                elif package == 'scipy':
                    import scipy
                elif package == 'sklearn':
                    import sklearn
                elif package == 'plotly':
                    import plotly
                elif package == 'jinja2':
                    import jinja2
                
                logger.info(f"  ✅ {package}: Available")
                
            except ImportError:
                missing_packages.append(package)
                logger.error(f"  ❌ {package}: Missing")
        
        if missing_packages:
            logger.error(f"❌ Missing packages: {', '.join(missing_packages)}")
            logger.info("💡 Install missing packages with: pip install " + " ".join(missing_packages))
            return False
        
        logger.info("✅ All dependencies available")
        return True
    
    def validate_database(self) -> bool:
        """Validate that the database exists and has sufficient quality data."""
        logger.info("🔍 Validating database...")
        
        if not self.db_path.exists():
            logger.error(f"❌ Database not found: {self.db_path}")
            logger.info("💡 Run the dataset processing pipeline to create the database")
            return False
        
        try:
            import sqlite3
            conn = sqlite3.connect(str(self.db_path))
            cursor = conn.cursor()
            
            # Check if required tables exist
            cursor.execute("SELECT name FROM sqlite_master WHERE type='table'")
            tables = [row[0] for row in cursor.fetchall()]
            
            required_tables = ['conversations', 'quality_metrics']
            missing_tables = [table for table in required_tables if table not in tables]
            
            if missing_tables:
                logger.error(f"❌ Missing tables: {', '.join(missing_tables)}")
                conn.close()
                return False
            
            # Check if sufficient quality data exists for trend analysis
            cursor.execute("""
                SELECT COUNT(DISTINCT DATE(c.created_at)) as days_with_data,
                       COUNT(*) as total_conversations,
                       MIN(c.created_at) as earliest_date,
                       MAX(c.created_at) as latest_date
                FROM conversations c
                LEFT JOIN quality_metrics q ON c.id = q.conversation_id
                WHERE q.overall_quality IS NOT NULL
            """)
            
            result = cursor.fetchone()
            days_with_data, total_conversations, earliest_date, latest_date = result
            
            if days_with_data < 7:
                logger.error(f"❌ Insufficient data for trend analysis: only {days_with_data} days with quality data")
                logger.info("💡 Need at least 7 days of quality data for meaningful trend analysis")
                conn.close()
                return False
            
            if total_conversations < 50:
                logger.error(f"❌ Insufficient conversations for trend analysis: only {total_conversations} conversations")
                logger.info("💡 Need at least 50 conversations with quality data for trend analysis")
                conn.close()
                return False
            
            logger.info(f"✅ Database validated: {total_conversations} conversations across {days_with_data} days")
            logger.info(f"📅 Data range: {earliest_date} to {latest_date}")
            conn.close()
            return True
            
        except Exception as e:
            logger.error(f"❌ Database validation error: {e}")
            return False
    
    def validate_trend_analysis_files(self) -> bool:
        """Validate that trend analysis files exist and are accessible."""
        logger.info("🔍 Validating trend analysis files...")
        
        required_files = [
            'quality_trend_analyzer.py',
            'quality_trend_reporter.py'
        ]
        
        for filename in required_files:
            filepath = self.monitoring_dir / filename
            
            if not filepath.exists():
                logger.error(f"❌ Required file not found: {filepath}")
                return False
            
            # Check if file has main classes
            try:
                with open(filepath, 'r') as f:
                    content = f.read()
                    
                if filename == 'quality_trend_analyzer.py':
                    if 'QualityTrendAnalyzer' not in content:
                        logger.error(f"❌ QualityTrendAnalyzer class not found in {filename}")
                        return False
                elif filename == 'quality_trend_reporter.py':
                    if 'QualityTrendReporter' not in content:
                        logger.error(f"❌ QualityTrendReporter class not found in {filename}")
                        return False
                
            except Exception as e:
                logger.error(f"❌ Error reading {filename}: {e}")
                return False
        
        logger.info("✅ Trend analysis files validated")
        return True
    
    def run_trend_analysis(self, 
                          days_back: int = 90,
                          include_predictions: bool = True,
                          include_visualizations: bool = True,
                          output_format: str = 'both') -> dict:
        """Run comprehensive trend analysis."""
        logger.info(f"📈 Running trend analysis ({days_back} days back)...")
        
        try:
            # Import trend analysis components
            sys.path.append(str(self.monitoring_dir))
            from quality_trend_analyzer import QualityTrendAnalyzer
            from quality_trend_reporter import QualityTrendReporter
            
            # Initialize components
            analyzer = QualityTrendAnalyzer(db_path=str(self.db_path))
            reporter = QualityTrendReporter(output_dir=str(self.reports_dir))
            
            # Load historical data
            logger.info("📊 Loading historical data...")
            df = analyzer.load_historical_data(days_back=days_back)
            
            if df.empty:
                logger.error("❌ No historical data available for analysis")
                return {'success': False, 'error': 'No historical data available'}
            
            logger.info(f"✅ Loaded {len(df)} conversations for analysis")
            
            # Generate comprehensive report
            logger.info("📋 Generating comprehensive trend report...")
            report = reporter.generate_comprehensive_report(
                days_back=days_back,
                include_predictions=include_predictions,
                include_visualizations=include_visualizations
            )
            
            # Save reports
            saved_files = []
            
            if output_format in ['json', 'both']:
                json_path = reporter.save_report(report, format='json')
                saved_files.append(json_path)
                logger.info(f"📄 JSON report saved: {json_path}")
            
            if output_format in ['html', 'both']:
                html_path = reporter.save_report(report, format='html')
                saved_files.append(html_path)
                logger.info(f"📄 HTML report saved: {html_path}")
            
            # Generate visualizations if requested
            visualization_files = []
            if include_visualizations:
                logger.info("📊 Generating trend visualizations...")
                try:
                    visualizations = reporter.create_trend_visualizations(report)
                    
                    # Save visualizations as HTML files
                    for viz_name, fig in visualizations.items():
                        viz_filename = f"trend_visualization_{viz_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.html"
                        viz_path = self.reports_dir / viz_filename
                        fig.write_html(str(viz_path))
                        visualization_files.append(str(viz_path))
                        logger.info(f"📊 Visualization saved: {viz_path}")
                        
                except Exception as e:
                    logger.warning(f"⚠️ Could not generate visualizations: {e}")
            
            # Create summary
            summary = {
                'success': True,
                'analysis_period': f"{days_back}_days",
                'total_conversations': report.overall_trend.total_conversations,
                'trend_direction': report.overall_trend.trend_direction,
                'trend_strength': report.overall_trend.trend_strength,
                'quality_change': report.overall_trend.quality_change,
                'statistical_significance': report.overall_trend.statistical_significance,
                'predictions_generated': len(report.overall_trend.predictions),
                'anomalies_detected': len(report.overall_trend.anomalies),
                'report_files': saved_files,
                'visualization_files': visualization_files,
                'executive_summary': report.executive_summary,
                'action_items': report.action_items
            }
            
            logger.info("✅ Trend analysis completed successfully")
            return summary
            
        except Exception as e:
            logger.error(f"❌ Error during trend analysis: {e}")
            return {'success': False, 'error': str(e)}
    
    def display_analysis_summary(self, summary: dict):
        """Display analysis summary to console."""
        if not summary.get('success', False):
            print(f"❌ Analysis failed: {summary.get('error', 'Unknown error')}")
            return
        
        print(f"\n📊 Quality Trend Analysis Summary")
        print(f"{'='*50}")
        print(f"Analysis Period: {summary['analysis_period']}")
        print(f"Total Conversations: {summary['total_conversations']:,}")
        print(f"Trend Direction: {summary['trend_direction']}")
        print(f"Trend Strength: {summary['trend_strength']:.3f}")
        print(f"Quality Change: {summary['quality_change']:.3f}")
        print(f"Statistical Significance: {summary['statistical_significance']:.3f}")
        print(f"Predictions Generated: {summary['predictions_generated']}")
        print(f"Anomalies Detected: {summary['anomalies_detected']}")
        
        print(f"\n📄 Generated Files:")
        for file_path in summary['report_files']:
            print(f"  📋 {file_path}")
        
        if summary['visualization_files']:
            print(f"\n📊 Visualization Files:")
            for file_path in summary['visualization_files']:
                print(f"  📈 {file_path}")
        
        print(f"\n💡 Executive Summary:")
        for i, item in enumerate(summary['executive_summary'][:5], 1):
            print(f"  {i}. {item}")
        
        print(f"\n🎯 Action Items:")
        for i, item in enumerate(summary['action_items'][:5], 1):
            print(f"  {i}. {item}")
    
    def launch(self, 
              days_back: int = 90,
              include_predictions: bool = True,
              include_visualizations: bool = True,
              output_format: str = 'both',
              skip_validation: bool = False) -> bool:
        """
        Complete launch sequence for quality trend analysis.
        
        Args:
            days_back: Number of days to analyze
            include_predictions: Include quality predictions
            include_visualizations: Generate trend visualizations
            output_format: Output format ('json', 'html', 'both')
            skip_validation: Skip pre-analysis validation
            
        Returns:
            bool: True if analysis successful, False otherwise
        """
        logger.info("🎯 Starting Quality Trend Analysis Launch Sequence...")
        
        # Step 1: Check dependencies
        if not self.check_dependencies():
            logger.error("❌ Launch aborted: Missing dependencies")
            return False
        
        # Step 2: Validate database (unless skipped)
        if not skip_validation:
            if not self.validate_database():
                logger.error("❌ Launch aborted: Database validation failed")
                return False
        
        # Step 3: Validate trend analysis files
        if not self.validate_trend_analysis_files():
            logger.error("❌ Launch aborted: Trend analysis files validation failed")
            return False
        
        # Step 4: Run trend analysis
        summary = self.run_trend_analysis(
            days_back=days_back,
            include_predictions=include_predictions,
            include_visualizations=include_visualizations,
            output_format=output_format
        )
        
        # Step 5: Display results
        self.display_analysis_summary(summary)
        
        if summary.get('success', False):
            logger.info("🎉 Quality Trend Analysis completed successfully!")
            return True
        else:
            logger.error("❌ Quality Trend Analysis failed")
            return False

def main():
    """Main function to launch quality trend analysis."""
    parser = argparse.ArgumentParser(description='Launch Quality Trend Analysis System')
    parser.add_argument('--days-back', type=int, default=90, help='Number of days to analyze (default: 90)')
    parser.add_argument('--no-predictions', action='store_true', help='Skip quality predictions')
    parser.add_argument('--no-visualizations', action='store_true', help='Skip trend visualizations')
    parser.add_argument('--format', choices=['json', 'html', 'both'], default='both', help='Output format')
    parser.add_argument('--skip-validation', action='store_true', help='Skip pre-analysis validation')
    
    args = parser.parse_args()
    
    launcher = QualityTrendAnalysisLauncher()
    success = launcher.launch(
        days_back=args.days_back,
        include_predictions=not args.no_predictions,
        include_visualizations=not args.no_visualizations,
        output_format=args.format,
        skip_validation=args.skip_validation
    )
    
    if not success:
        sys.exit(1)

if __name__ == "__main__":
    main()
