#!/usr/bin/env python3
"""
Quality Comparison System Launcher (Task 5.6.2.5)

Production launcher for the quality comparison and benchmarking system.
Provides comprehensive comparison analysis across tiers, datasets, and components.
"""

import sys
import os
import argparse
import logging
from datetime import datetime, timedelta
from pathlib import Path
import tempfile
import json

# Add the monitoring directory to the path
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from quality_comparator import QualityComparator
from quality_comparison_reporter import QualityComparisonReporter

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(),
        logging.FileHandler('quality_comparison_launcher.log')
    ]
)
logger = logging.getLogger(__name__)

def parse_arguments():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Launch Quality Comparison System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python launch_quality_comparison.py --days 30
  python launch_quality_comparison.py --days 90 --output /path/to/reports
  python launch_quality_comparison.py --days 7 --format html
  python launch_quality_comparison.py --interactive
        """
    )
    
    parser.add_argument(
        '--days',
        type=int,
        default=30,
        help='Number of days to analyze (default: 30)'
    )
    
    parser.add_argument(
        '--db-path',
        type=str,
        default="/home/vivi/pixelated/ai/database/conversations.db",
        help='Path to the conversations database'
    )
    
    parser.add_argument(
        '--output',
        type=str,
        default=None,
        help='Output directory for reports (default: temp directory)'
    )
    
    parser.add_argument(
        '--format',
        choices=['json', 'html', 'both'],
        default='both',
        help='Report format (default: both)'
    )
    
    parser.add_argument(
        '--interactive',
        action='store_true',
        help='Run in interactive mode with prompts'
    )
    
    parser.add_argument(
        '--tier',
        type=str,
        choices=['priority_1', 'priority_2', 'priority_3'],
        help='Focus analysis on specific tier'
    )
    
    parser.add_argument(
        '--dataset',
        type=str,
        help='Focus analysis on specific dataset'
    )
    
    parser.add_argument(
        '--benchmark',
        action='store_true',
        help='Include benchmark analysis against industry standards'
    )
    
    parser.add_argument(
        '--visualizations',
        action='store_true',
        default=True,
        help='Generate visualizations (default: True)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()

def setup_logging(verbose: bool):
    """Setup logging configuration."""
    if verbose:
        logging.getLogger().setLevel(logging.DEBUG)
        logger.info("🔍 Verbose logging enabled")

def validate_database(db_path: str) -> bool:
    """Validate database exists and is accessible."""
    if not Path(db_path).exists():
        logger.error(f"❌ Database not found: {db_path}")
        return False
    
    try:
        comparator = QualityComparator(db_path=db_path)
        df = comparator.load_comparison_data(days_back=1)
        if df.empty:
            logger.warning("⚠️ No quality data found in database")
            return False
        logger.info(f"✅ Database validated: {len(df)} records available")
        return True
    except Exception as e:
        logger.error(f"❌ Database validation failed: {e}")
        return False

def interactive_mode():
    """Run in interactive mode with user prompts."""
    print("\n🎯 Quality Comparison System - Interactive Mode")
    print("=" * 50)
    
    # Get analysis parameters
    days = input("📅 Analysis period (days) [30]: ").strip()
    days = int(days) if days else 30
    
    db_path = input("🗄️ Database path [/home/vivi/pixelated/ai/database/conversations.db]: ").strip()
    db_path = db_path if db_path else "/home/vivi/pixelated/ai/database/conversations.db"
    
    output_dir = input("📁 Output directory [temp]: ").strip()
    output_dir = output_dir if output_dir else None
    
    format_choice = input("📄 Report format (json/html/both) [both]: ").strip()
    format_choice = format_choice if format_choice in ['json', 'html', 'both'] else 'both'
    
    tier_focus = input("🎯 Focus on specific tier (priority_1/priority_2/priority_3) [all]: ").strip()
    tier_focus = tier_focus if tier_focus in ['priority_1', 'priority_2', 'priority_3'] else None
    
    dataset_focus = input("📊 Focus on specific dataset [all]: ").strip()
    dataset_focus = dataset_focus if dataset_focus else None
    
    include_benchmark = input("📈 Include benchmark analysis? (y/n) [y]: ").strip().lower()
    include_benchmark = include_benchmark != 'n'
    
    return {
        'days': days,
        'db_path': db_path,
        'output': output_dir,
        'format': format_choice,
        'tier': tier_focus,
        'dataset': dataset_focus,
        'benchmark': include_benchmark,
        'visualizations': True
    }

def run_comparison_analysis(args):
    """Run the quality comparison analysis."""
    logger.info("🚀 Starting Quality Comparison System")
    
    try:
        # Initialize comparator
        logger.info(f"🔧 Initializing comparator with database: {args['db_path']}")
        comparator = QualityComparator(db_path=args['db_path'])
        
        # Setup output directory
        if args['output']:
            output_dir = Path(args['output'])
            output_dir.mkdir(parents=True, exist_ok=True)
        else:
            output_dir = Path(tempfile.mkdtemp(prefix='quality_comparison_'))
        
        logger.info(f"📁 Output directory: {output_dir}")
        
        # Initialize reporter
        reporter = QualityComparisonReporter(output_dir=str(output_dir))
        reporter.comparator = comparator
        
        # Load data
        logger.info(f"📊 Loading comparison data ({args['days']} days)")
        df = comparator.load_comparison_data(days_back=args['days'])
        
        if df.empty:
            logger.error("❌ No data available for analysis")
            return False
        
        logger.info(f"✅ Loaded {len(df)} quality records")
        
        # Apply filters if specified
        if args.get('tier'):
            df = df[df['tier'] == args['tier']]
            logger.info(f"🎯 Filtered to tier: {args['tier']} ({len(df)} records)")
        
        if args.get('dataset'):
            df = df[df['dataset_name'] == args['dataset']]
            logger.info(f"📊 Filtered to dataset: {args['dataset']} ({len(df)} records)")
        
        if df.empty:
            logger.error("❌ No data remaining after filtering")
            return False
        
        # Generate comprehensive report
        logger.info("📈 Generating comprehensive comparison report")
        report = reporter.generate_comprehensive_report(days_back=args['days'])
        
        if not report:
            logger.error("❌ Failed to generate report")
            return False
        
        # Save reports
        saved_files = []
        
        if args['format'] in ['json', 'both']:
            json_path = reporter.save_report(report, format='json')
            saved_files.append(json_path)
            logger.info(f"💾 JSON report saved: {json_path}")
        
        if args['format'] in ['html', 'both']:
            html_path = reporter.save_report(report, format='html')
            saved_files.append(html_path)
            logger.info(f"💾 HTML report saved: {html_path}")
        
        # Generate visualizations
        if args.get('visualizations', True):
            logger.info("📊 Creating comparison visualizations")
            visualizations = reporter.create_comparison_visualizations(report)
            logger.info(f"✅ Generated {len(visualizations)} visualizations")
        
        # Display summary
        print("\n" + "="*60)
        print("📊 QUALITY COMPARISON ANALYSIS COMPLETE")
        print("="*60)
        print(f"📅 Analysis Period: {args['days']} days")
        print(f"📊 Records Analyzed: {len(df)}")
        print(f"🎯 Tier Comparisons: {len(report.tier_comparisons)}")
        print(f"📁 Dataset Comparisons: {len(report.dataset_comparisons)}")
        print(f"🧩 Component Comparisons: {len(report.component_comparisons)}")
        print(f"📈 Benchmark Analyses: {len(report.benchmark_analyses)}")
        print(f"🏆 Performance Rankings: {len(report.performance_rankings)}")
        print(f"📋 Executive Summary: {len(report.executive_summary)} items")
        print(f"🎯 Action Items: {len(report.action_items)} items")
        print(f"📁 Output Directory: {output_dir}")
        print(f"💾 Files Generated: {len(saved_files)}")
        
        for file_path in saved_files:
            print(f"   📄 {Path(file_path).name}")
        
        if args.get('visualizations', True):
            print(f"📊 Visualizations: {len(visualizations)} charts created")
        
        print("="*60)
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Analysis failed: {e}")
        return False

def main():
    """Main launcher function."""
    try:
        # Parse arguments
        if len(sys.argv) == 1:
            # No arguments provided, run in interactive mode
            args = interactive_mode()
        else:
            parsed_args = parse_arguments()
            args = vars(parsed_args)
            
            if args['interactive']:
                args.update(interactive_mode())
        
        # Setup logging
        setup_logging(args.get('verbose', False))
        
        # Validate database
        if not validate_database(args['db_path']):
            sys.exit(1)
        
        # Run analysis
        success = run_comparison_analysis(args)
        
        if success:
            logger.info("✅ Quality Comparison System completed successfully")
            sys.exit(0)
        else:
            logger.error("❌ Quality Comparison System failed")
            sys.exit(1)
            
    except KeyboardInterrupt:
        logger.info("⚠️ Analysis interrupted by user")
        sys.exit(1)
    except Exception as e:
        logger.error(f"❌ Unexpected error: {e}")
        sys.exit(1)

if __name__ == "__main__":
    main()
