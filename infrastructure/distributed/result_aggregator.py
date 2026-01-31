#!/usr/bin/env python3
"""
Quality Validation Result Aggregation System for Pixelated Empathy AI
Aggregates and analyzes results from distributed quality validation workers
"""

import os
import sys
import json
import logging
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from dataclasses import dataclass, asdict
from datetime import datetime, timezone
import statistics
import numpy as np
from collections import defaultdict, Counter
import sqlite3
import threading
import time

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)


@dataclass
class AggregatedResult:
    """Aggregated validation results"""
    batch_id: str
    total_files: int
    processed_files: int
    success_rate: float
    overall_quality_score: float
    quality_distribution: Dict[str, int]
    metric_statistics: Dict[str, Dict[str, float]]
    common_issues: List[Dict[str, Any]]
    processing_time_stats: Dict[str, float]
    worker_performance: Dict[str, Dict[str, float]]
    timestamp: str
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary"""
        return asdict(self)


class ResultAggregator:
    """Aggregates quality validation results"""
    
    def __init__(self, db_path: str = None):
        self.db_path = db_path or "quality_results.db"
        self.results_cache: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        self.lock = threading.Lock()
        
        # Initialize database
        self._init_database()
    
    def _init_database(self):
        """Initialize SQLite database for storing results"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS validation_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT NOT NULL,
                    task_id TEXT NOT NULL,
                    file_path TEXT NOT NULL,
                    validation_type TEXT NOT NULL,
                    success BOOLEAN NOT NULL,
                    quality_score REAL,
                    metrics TEXT,  -- JSON
                    issues TEXT,   -- JSON
                    processing_time REAL,
                    worker_id TEXT,
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS aggregated_results (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    batch_id TEXT UNIQUE NOT NULL,
                    total_files INTEGER,
                    processed_files INTEGER,
                    success_rate REAL,
                    overall_quality_score REAL,
                    quality_distribution TEXT,  -- JSON
                    metric_statistics TEXT,     -- JSON
                    common_issues TEXT,         -- JSON
                    processing_time_stats TEXT, -- JSON
                    worker_performance TEXT,    -- JSON
                    timestamp TEXT,
                    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            # Create indexes for better performance
            conn.execute("CREATE INDEX IF NOT EXISTS idx_batch_id ON validation_results(batch_id)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_timestamp ON validation_results(timestamp)")
            conn.execute("CREATE INDEX IF NOT EXISTS idx_worker_id ON validation_results(worker_id)")
    
    def add_result(self, batch_id: str, result: Dict[str, Any]):
        """Add a validation result to the aggregation"""
        with self.lock:
            # Store in cache
            self.results_cache[batch_id].append(result)
            
            # Store in database
            self._store_result_in_db(batch_id, result)
    
    def _store_result_in_db(self, batch_id: str, result: Dict[str, Any]):
        """Store result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT INTO validation_results (
                        batch_id, task_id, file_path, validation_type, success,
                        quality_score, metrics, issues, processing_time,
                        worker_id, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    batch_id,
                    result.get('task_id', ''),
                    result.get('file_path', ''),
                    result.get('validation_type', ''),
                    result.get('success', False),
                    result.get('quality_score', 0.0),
                    json.dumps(result.get('metrics', {})),
                    json.dumps(result.get('issues', [])),
                    result.get('processing_time', 0.0),
                    result.get('worker_id', ''),
                    result.get('timestamp', datetime.now(timezone.utc).isoformat())
                ))
        except Exception as e:
            logger.error(f"Failed to store result in database: {e}")
    
    def aggregate_results(self, batch_id: str, force_refresh: bool = False) -> Optional[AggregatedResult]:
        """Aggregate results for a batch"""
        # Get results from cache or database
        if batch_id in self.results_cache and not force_refresh:
            results = self.results_cache[batch_id]
        else:
            results = self._load_results_from_db(batch_id)
        
        if not results:
            logger.warning(f"No results found for batch: {batch_id}")
            return None
        
        # Perform aggregation
        aggregated = self._perform_aggregation(batch_id, results)
        
        # Store aggregated results
        self._store_aggregated_result(aggregated)
        
        return aggregated
    
    def _load_results_from_db(self, batch_id: str) -> List[Dict[str, Any]]:
        """Load results from database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT task_id, file_path, validation_type, success,
                           quality_score, metrics, issues, processing_time,
                           worker_id, timestamp
                    FROM validation_results
                    WHERE batch_id = ?
                    ORDER BY timestamp
                """, (batch_id,))
                
                results = []
                for row in cursor.fetchall():
                    result = {
                        'task_id': row[0],
                        'file_path': row[1],
                        'validation_type': row[2],
                        'success': bool(row[3]),
                        'quality_score': row[4],
                        'metrics': json.loads(row[5]) if row[5] else {},
                        'issues': json.loads(row[6]) if row[6] else [],
                        'processing_time': row[7],
                        'worker_id': row[8],
                        'timestamp': row[9]
                    }
                    results.append(result)
                
                return results
                
        except Exception as e:
            logger.error(f"Failed to load results from database: {e}")
            return []
    
    def _perform_aggregation(self, batch_id: str, results: List[Dict[str, Any]]) -> AggregatedResult:
        """Perform the actual aggregation of results"""
        total_files = len(results)
        successful_results = [r for r in results if r.get('success', False)]
        processed_files = len(successful_results)
        
        # Calculate success rate
        success_rate = processed_files / total_files if total_files > 0 else 0.0
        
        # Calculate overall quality score
        if successful_results:
            quality_scores = [r.get('quality_score', 0.0) for r in successful_results]
            overall_quality_score = statistics.mean(quality_scores)
        else:
            overall_quality_score = 0.0
        
        # Quality distribution
        quality_distribution = self._calculate_quality_distribution(successful_results)
        
        # Metric statistics
        metric_statistics = self._calculate_metric_statistics(successful_results)
        
        # Common issues
        common_issues = self._analyze_common_issues(results)
        
        # Processing time statistics
        processing_time_stats = self._calculate_processing_time_stats(results)
        
        # Worker performance
        worker_performance = self._analyze_worker_performance(results)
        
        return AggregatedResult(
            batch_id=batch_id,
            total_files=total_files,
            processed_files=processed_files,
            success_rate=success_rate,
            overall_quality_score=overall_quality_score,
            quality_distribution=quality_distribution,
            metric_statistics=metric_statistics,
            common_issues=common_issues,
            processing_time_stats=processing_time_stats,
            worker_performance=worker_performance,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
    
    def _calculate_quality_distribution(self, results: List[Dict[str, Any]]) -> Dict[str, int]:
        """Calculate quality score distribution"""
        distribution = {
            'excellent': 0,  # 0.9-1.0
            'good': 0,       # 0.7-0.9
            'fair': 0,       # 0.5-0.7
            'poor': 0        # 0.0-0.5
        }
        
        for result in results:
            score = result.get('quality_score', 0.0)
            if score >= 0.9:
                distribution['excellent'] += 1
            elif score >= 0.7:
                distribution['good'] += 1
            elif score >= 0.5:
                distribution['fair'] += 1
            else:
                distribution['poor'] += 1
        
        return distribution
    
    def _calculate_metric_statistics(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Calculate statistics for each metric"""
        metric_values = defaultdict(list)
        
        # Collect all metric values
        for result in results:
            metrics = result.get('metrics', {})
            for metric_name, value in metrics.items():
                if isinstance(value, (int, float)):
                    metric_values[metric_name].append(value)
        
        # Calculate statistics for each metric
        metric_stats = {}
        for metric_name, values in metric_values.items():
            if values:
                metric_stats[metric_name] = {
                    'mean': statistics.mean(values),
                    'median': statistics.median(values),
                    'std_dev': statistics.stdev(values) if len(values) > 1 else 0.0,
                    'min': min(values),
                    'max': max(values),
                    'count': len(values)
                }
        
        return metric_stats
    
    def _analyze_common_issues(self, results: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Analyze and identify common issues"""
        issue_counter = Counter()
        issue_details = defaultdict(list)
        
        for result in results:
            issues = result.get('issues', [])
            for issue in issues:
                issue_type = issue.get('type', 'unknown')
                issue_message = issue.get('message', '')
                
                # Create a key for grouping similar issues
                issue_key = f"{issue_type}:{issue_message[:50]}"
                issue_counter[issue_key] += 1
                issue_details[issue_key].append({
                    'file_path': result.get('file_path', ''),
                    'severity': issue.get('severity', 'unknown'),
                    'full_message': issue_message
                })
        
        # Get top 10 most common issues
        common_issues = []
        for issue_key, count in issue_counter.most_common(10):
            issue_type, message_preview = issue_key.split(':', 1)
            
            common_issues.append({
                'type': issue_type,
                'message_preview': message_preview,
                'count': count,
                'percentage': (count / len(results)) * 100,
                'examples': issue_details[issue_key][:3]  # First 3 examples
            })
        
        return common_issues
    
    def _calculate_processing_time_stats(self, results: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate processing time statistics"""
        processing_times = [r.get('processing_time', 0.0) for r in results if r.get('processing_time')]
        
        if not processing_times:
            return {
                'mean': 0.0,
                'median': 0.0,
                'std_dev': 0.0,
                'min': 0.0,
                'max': 0.0,
                'total': 0.0
            }
        
        return {
            'mean': statistics.mean(processing_times),
            'median': statistics.median(processing_times),
            'std_dev': statistics.stdev(processing_times) if len(processing_times) > 1 else 0.0,
            'min': min(processing_times),
            'max': max(processing_times),
            'total': sum(processing_times)
        }
    
    def _analyze_worker_performance(self, results: List[Dict[str, Any]]) -> Dict[str, Dict[str, float]]:
        """Analyze performance by worker"""
        worker_stats = defaultdict(lambda: {
            'tasks_completed': 0,
            'success_rate': 0.0,
            'avg_quality_score': 0.0,
            'avg_processing_time': 0.0,
            'total_processing_time': 0.0
        })
        
        worker_data = defaultdict(lambda: {
            'total_tasks': 0,
            'successful_tasks': 0,
            'quality_scores': [],
            'processing_times': []
        })
        
        # Collect data by worker
        for result in results:
            worker_id = result.get('worker_id', 'unknown')
            worker_data[worker_id]['total_tasks'] += 1
            
            if result.get('success', False):
                worker_data[worker_id]['successful_tasks'] += 1
                worker_data[worker_id]['quality_scores'].append(result.get('quality_score', 0.0))
            
            processing_time = result.get('processing_time', 0.0)
            if processing_time > 0:
                worker_data[worker_id]['processing_times'].append(processing_time)
        
        # Calculate statistics for each worker
        for worker_id, data in worker_data.items():
            worker_stats[worker_id]['tasks_completed'] = data['total_tasks']
            worker_stats[worker_id]['success_rate'] = (
                data['successful_tasks'] / data['total_tasks'] if data['total_tasks'] > 0 else 0.0
            )
            
            if data['quality_scores']:
                worker_stats[worker_id]['avg_quality_score'] = statistics.mean(data['quality_scores'])
            
            if data['processing_times']:
                worker_stats[worker_id]['avg_processing_time'] = statistics.mean(data['processing_times'])
                worker_stats[worker_id]['total_processing_time'] = sum(data['processing_times'])
        
        return dict(worker_stats)
    
    def _store_aggregated_result(self, aggregated: AggregatedResult):
        """Store aggregated result in database"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                conn.execute("""
                    INSERT OR REPLACE INTO aggregated_results (
                        batch_id, total_files, processed_files, success_rate,
                        overall_quality_score, quality_distribution, metric_statistics,
                        common_issues, processing_time_stats, worker_performance, timestamp
                    ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """, (
                    aggregated.batch_id,
                    aggregated.total_files,
                    aggregated.processed_files,
                    aggregated.success_rate,
                    aggregated.overall_quality_score,
                    json.dumps(aggregated.quality_distribution),
                    json.dumps(aggregated.metric_statistics),
                    json.dumps(aggregated.common_issues),
                    json.dumps(aggregated.processing_time_stats),
                    json.dumps(aggregated.worker_performance),
                    aggregated.timestamp
                ))
        except Exception as e:
            logger.error(f"Failed to store aggregated result: {e}")
    
    def get_batch_summary(self, batch_id: str) -> Optional[Dict[str, Any]]:
        """Get summary for a specific batch"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT * FROM aggregated_results WHERE batch_id = ?
                """, (batch_id,))
                
                row = cursor.fetchone()
                if row:
                    return {
                        'batch_id': row[1],
                        'total_files': row[2],
                        'processed_files': row[3],
                        'success_rate': row[4],
                        'overall_quality_score': row[5],
                        'quality_distribution': json.loads(row[6]),
                        'metric_statistics': json.loads(row[7]),
                        'common_issues': json.loads(row[8]),
                        'processing_time_stats': json.loads(row[9]),
                        'worker_performance': json.loads(row[10]),
                        'timestamp': row[11]
                    }
                
                return None
                
        except Exception as e:
            logger.error(f"Failed to get batch summary: {e}")
            return None
    
    def get_all_batches(self) -> List[Dict[str, Any]]:
        """Get summary of all batches"""
        try:
            with sqlite3.connect(self.db_path) as conn:
                cursor = conn.execute("""
                    SELECT batch_id, total_files, processed_files, success_rate,
                           overall_quality_score, timestamp
                    FROM aggregated_results
                    ORDER BY timestamp DESC
                """)
                
                batches = []
                for row in cursor.fetchall():
                    batches.append({
                        'batch_id': row[0],
                        'total_files': row[1],
                        'processed_files': row[2],
                        'success_rate': row[3],
                        'overall_quality_score': row[4],
                        'timestamp': row[5]
                    })
                
                return batches
                
        except Exception as e:
            logger.error(f"Failed to get all batches: {e}")
            return []
    
    def generate_report(self, batch_id: str, output_file: str = None) -> str:
        """Generate a detailed report for a batch"""
        aggregated = self.get_batch_summary(batch_id)
        
        if not aggregated:
            return f"No data found for batch: {batch_id}"
        
        # Generate report content
        report_lines = [
            f"Quality Validation Report - Batch {batch_id}",
            "=" * 60,
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            "",
            "SUMMARY",
            "-" * 20,
            f"Total Files: {aggregated['total_files']}",
            f"Processed Files: {aggregated['processed_files']}",
            f"Success Rate: {aggregated['success_rate']:.2%}",
            f"Overall Quality Score: {aggregated['overall_quality_score']:.3f}",
            "",
            "QUALITY DISTRIBUTION",
            "-" * 20
        ]
        
        for category, count in aggregated['quality_distribution'].items():
            percentage = (count / aggregated['processed_files']) * 100 if aggregated['processed_files'] > 0 else 0
            report_lines.append(f"{category.capitalize()}: {count} ({percentage:.1f}%)")
        
        report_lines.extend([
            "",
            "METRIC STATISTICS",
            "-" * 20
        ])
        
        for metric, stats in aggregated['metric_statistics'].items():
            report_lines.append(f"{metric.capitalize()}:")
            report_lines.append(f"  Mean: {stats['mean']:.3f}")
            report_lines.append(f"  Median: {stats['median']:.3f}")
            report_lines.append(f"  Std Dev: {stats['std_dev']:.3f}")
            report_lines.append(f"  Range: {stats['min']:.3f} - {stats['max']:.3f}")
            report_lines.append("")
        
        report_lines.extend([
            "COMMON ISSUES",
            "-" * 20
        ])
        
        for issue in aggregated['common_issues'][:5]:  # Top 5 issues
            report_lines.append(f"• {issue['type']}: {issue['message_preview']}")
            report_lines.append(f"  Occurrences: {issue['count']} ({issue['percentage']:.1f}%)")
            report_lines.append("")
        
        report_lines.extend([
            "PROCESSING PERFORMANCE",
            "-" * 20,
            f"Total Processing Time: {aggregated['processing_time_stats']['total']:.2f} seconds",
            f"Average Processing Time: {aggregated['processing_time_stats']['mean']:.2f} seconds",
            f"Median Processing Time: {aggregated['processing_time_stats']['median']:.2f} seconds",
            "",
            "WORKER PERFORMANCE",
            "-" * 20
        ])
        
        for worker_id, stats in aggregated['worker_performance'].items():
            report_lines.append(f"Worker: {worker_id}")
            report_lines.append(f"  Tasks: {stats['tasks_completed']}")
            report_lines.append(f"  Success Rate: {stats['success_rate']:.2%}")
            report_lines.append(f"  Avg Quality: {stats['avg_quality_score']:.3f}")
            report_lines.append(f"  Avg Time: {stats['avg_processing_time']:.2f}s")
            report_lines.append("")
        
        report_content = "\n".join(report_lines)
        
        # Save to file if specified
        if output_file:
            with open(output_file, 'w') as f:
                f.write(report_content)
            logger.info(f"Report saved to: {output_file}")
        
        return report_content


def main():
    """Main CLI interface"""
    import argparse
    
    parser = argparse.ArgumentParser(description="Quality Validation Result Aggregator")
    parser.add_argument('--db-path', help="Database path for storing results")
    
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # Aggregate command
    aggregate_parser = subparsers.add_parser('aggregate', help='Aggregate results for a batch')
    aggregate_parser.add_argument('batch_id', help='Batch ID to aggregate')
    aggregate_parser.add_argument('--force-refresh', action='store_true', help='Force refresh from database')
    
    # Report command
    report_parser = subparsers.add_parser('report', help='Generate report for a batch')
    report_parser.add_argument('batch_id', help='Batch ID for report')
    report_parser.add_argument('--output', help='Output file for report')
    
    # List command
    list_parser = subparsers.add_parser('list', help='List all batches')
    
    # Summary command
    summary_parser = subparsers.add_parser('summary', help='Get batch summary')
    summary_parser.add_argument('batch_id', help='Batch ID for summary')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Create aggregator
    aggregator = ResultAggregator(args.db_path)
    
    if args.command == 'aggregate':
        result = aggregator.aggregate_results(args.batch_id, args.force_refresh)
        if result:
            print(json.dumps(result.to_dict(), indent=2))
        else:
            print(f"No results found for batch: {args.batch_id}")
    
    elif args.command == 'report':
        report = aggregator.generate_report(args.batch_id, args.output)
        if not args.output:
            print(report)
    
    elif args.command == 'list':
        batches = aggregator.get_all_batches()
        print(json.dumps(batches, indent=2))
    
    elif args.command == 'summary':
        summary = aggregator.get_batch_summary(args.batch_id)
        if summary:
            print(json.dumps(summary, indent=2))
        else:
            print(f"No summary found for batch: {args.batch_id}")


if __name__ == '__main__':
    main()
