"""
Pipeline Management Commands for Pixelated AI CLI

This module provides commands for managing AI pipelines, including starting, stopping,
monitoring, and configuring pipeline executions.
"""

import click
import json
import time
from typing import Optional, Dict, Any, List
from pathlib import Path

from ..utils import (
    setup_logging, get_logger, validate_environment,
    check_api_health, format_file_size, sanitize_filename
)
from ..config import get_config
from ..auth import AuthManager


logger = get_logger(__name__)


@click.group(name='pipeline')
@click.pass_context
def pipeline_group(ctx):
    """Pipeline management commands for AI operations."""
    setup_logging(ctx.obj.get('verbose', False))
    logger.info("Pipeline management command group initialized")


@pipeline_group.command()
@click.option('--config', '-c', type=click.Path(exists=True), required=True,
              help='Pipeline configuration file')
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input data file or directory')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--name', help='Pipeline execution name')
@click.option('--priority', type=click.Choice(['low', 'normal', 'high']), default='normal',
              help='Execution priority')
@click.option('--tags', help='Comma-separated tags for the pipeline')
@click.pass_context
def start(ctx, config: str, input: str, output: Optional[str], name: Optional[str],
          priority: str, tags: Optional[str]):
    """Start a new pipeline execution."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        # Validate authentication
        if not auth_manager.is_authenticated():
            click.echo("❌ Authentication required. Please login first.", err=True)
            return
        
        # Load pipeline configuration
        with open(config, 'r') as f:
            pipeline_config = json.load(f)
        
        input_path = Path(input)
        output_path = Path(output) if output else Path.cwd() / 'pipeline_output'
        
        # Validate input
        if not input_path.exists():
            click.echo(f"❌ Input path not found: {input}", err=True)
            return
        
        # Parse tags
        tag_list = [tag.strip() for tag in tags.split(',')] if tags else []
        
        click.echo(f"🚀 Starting pipeline execution...")
        click.echo(f"📁 Input: {input_path}")
        click.echo(f"📤 Output: {output_path}")
        click.echo(f"⚡ Priority: {priority}")
        
        if name:
            click.echo(f"🏷️  Name: {name}")
        
        if tag_list:
            click.echo(f"🏷️  Tags: {', '.join(tag_list)}")
        
        # Start pipeline
        pipeline_info = _start_pipeline(
            auth_manager, pipeline_config, input_path, output_path,
            name, priority, tag_list
        )
        
        if pipeline_info['success']:
            pipeline_id = pipeline_info['pipeline_id']
            click.echo(f"✅ Pipeline started successfully!")
            click.echo(f"🆔 Pipeline ID: {pipeline_id}")
            click.echo(f"📊 Monitor with: pixelated pipeline monitor --pipeline-id {pipeline_id}")
            
            # Save pipeline info for later reference
            _save_pipeline_info(pipeline_id, pipeline_info)
            
        else:
            click.echo(f"❌ Failed to start pipeline: {pipeline_info.get('error', 'Unknown error')}", err=True)
            
    except Exception as e:
        logger.error(f"Pipeline start failed: {e}")
        click.echo(f"❌ Pipeline start failed: {e}", err=True)
        raise click.Abort()


@pipeline_group.command()
@click.option('--pipeline-id', help='Specific pipeline to stop')
@click.option('--all', 'stop_all', is_flag=True, help='Stop all running pipelines')
@click.option('--force', is_flag=True, help='Force stop without confirmation')
@click.pass_context
def stop(ctx, pipeline_id: Optional[str], stop_all: bool, force: bool):
    """Stop pipeline execution(s)."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        if stop_all:
            if not force:
                if not click.confirm("⚠️  Are you sure you want to stop ALL running pipelines?"):
                    click.echo("❌ Operation cancelled")
                    return
            
            click.echo("🛑 Stopping all running pipelines...")
            result = _stop_all_pipelines(auth_manager)
            
            if result['success']:
                click.echo(f"✅ Stopped {result['stopped_count']} pipelines")
            else:
                click.echo(f"❌ Failed to stop pipelines: {result.get('error', 'Unknown error')}", err=True)
                
        elif pipeline_id:
            if not force:
                if not click.confirm(f"⚠️  Are you sure you want to stop pipeline {pipeline_id}?"):
                    click.echo("❌ Operation cancelled")
                    return
            
            click.echo(f"🛑 Stopping pipeline: {pipeline_id}")
            result = _stop_pipeline(auth_manager, pipeline_id)
            
            if result['success']:
                click.echo(f"✅ Pipeline {pipeline_id} stopped successfully")
            else:
                click.echo(f"❌ Failed to stop pipeline: {result.get('error', 'Unknown error')}", err=True)
                
        else:
            click.echo("❌ Please specify --pipeline-id or --all")
            
    except Exception as e:
        logger.error(f"Pipeline stop failed: {e}")
        click.echo(f"❌ Pipeline stop failed: {e}", err=True)
        raise click.Abort()


@pipeline_group.command()
@click.option('--pipeline-id', required=True, help='Pipeline ID to monitor')
@click.option('--refresh', '-r', default=5, help='Refresh interval in seconds')
@click.option('--verbose', '-v', is_flag=True, help='Show detailed progress information')
@click.option('--output', type=click.Choice(['simple', 'detailed', 'json']), default='simple',
              help='Output format')
@click.pass_context
def monitor(ctx, pipeline_id: str, refresh: int, verbose: bool, output: str):
    """Monitor pipeline execution in real-time."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        click.echo(f"📊 Monitoring pipeline: {pipeline_id}")
        click.echo(f"🔄 Refresh interval: {refresh}s")
        click.echo(f"📤 Output format: {output}")
        click.echo("-" * 50)
        
        # Monitor loop
        try:
            while True:
                status = _get_pipeline_status(auth_manager, pipeline_id)
                
                if not status:
                    click.echo(f"❌ Pipeline {pipeline_id} not found", err=True)
                    break
                
                # Clear screen for clean display (simple mode)
                if output == 'simple':
                    click.clear()
                
                # Display status based on format
                if output == 'json':
                    click.echo(json.dumps(status, indent=2))
                elif output == 'detailed':
                    _display_detailed_status(status, verbose)
                else:  # simple
                    _display_simple_status(status, verbose)
                
                # Check if pipeline is complete
                if status['status'] in ['completed', 'failed', 'cancelled']:
                    click.echo(f"\n🎯 Pipeline {status['status']}!")
                    
                    if status['status'] == 'completed':
                        click.echo(f"✅ Pipeline completed successfully!")
                        if 'results' in status:
                            click.echo(f"📊 Results: {status['results']}")
                    elif status['status'] == 'failed':
                        click.echo(f"❌ Pipeline failed: {status.get('error', 'Unknown error')}")
                    
                    break
                
                time.sleep(refresh)
                
        except KeyboardInterrupt:
            click.echo("\n⏹️  Monitoring stopped by user")
            
    except Exception as e:
        logger.error(f"Pipeline monitoring failed: {e}")
        click.echo(f"❌ Pipeline monitoring failed: {e}", err=True)
        raise click.Abort()


@pipeline_group.command()
@click.option('--status', type=click.Choice(['running', 'completed', 'failed', 'all']), default='all',
              help='Filter by pipeline status')
@click.option('--limit', default=20, help='Maximum number of results')
@click.option('--detailed', is_flag=True, help='Show detailed information')
@click.pass_context
def list_executions(ctx, status: str, limit: int, detailed: bool):
    """List pipeline executions."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        click.echo("📋 Pipeline Executions:")
        
        executions = _list_pipeline_executions(auth_manager, status, limit)
        
        if not executions:
            click.echo("❌ No pipeline executions found")
            return
        
        for execution in executions:
            _display_execution_summary(execution, detailed)
        
        # Summary
        status_counts = {}
        for execution in executions:
            status_counts[execution['status']] = status_counts.get(execution['status'], 0) + 1
        
        click.echo(f"\n📊 Summary:")
        for status_type, count in status_counts.items():
            status_icon = _get_status_icon(status_type)
            click.echo(f"  {status_icon} {status_type}: {count}")
            
    except Exception as e:
        logger.error(f"Pipeline list failed: {e}")
        click.echo(f"❌ Pipeline list failed: {e}", err=True)
        raise click.Abort()


@pipeline_group.command()
@click.option('--pipeline-id', required=True, help='Pipeline ID to configure')
@click.option('--config-file', type=click.Path(exists=True), help='New configuration file')
@click.option('--param', multiple=True, help='Set parameter (key=value)')
@click.option('--restart', is_flag=True, help='Restart pipeline after configuration')
@click.pass_context
def configure(ctx, pipeline_id: str, config_file: Optional[str], param: tuple, restart: bool):
    """Configure pipeline parameters."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        click.echo(f"⚙️  Configuring pipeline: {pipeline_id}")
        
        # Build configuration updates
        config_updates = {}
        
        if config_file:
            with open(config_file, 'r') as f:
                file_config = json.load(f)
                config_updates.update(file_config)
        
        # Parse individual parameters
        for param_str in param:
            if '=' not in param_str:
                click.echo(f"❌ Invalid parameter format: {param_str}. Use key=value", err=True)
                continue
            
            key, value = param_str.split('=', 1)
            try:
                # Try to parse as JSON for complex values
                config_updates[key] = json.loads(value)
            except json.JSONDecodeError:
                # Use as string if not valid JSON
                config_updates[key] = value
        
        if not config_updates:
            click.echo("❌ No configuration updates provided")
            return
        
        # Apply configuration
        result = _configure_pipeline(auth_manager, pipeline_id, config_updates)
        
        if result['success']:
            click.echo(f"✅ Pipeline configuration updated successfully!")
            
            if restart:
                click.echo("🔄 Restarting pipeline with new configuration...")
                restart_result = _restart_pipeline(auth_manager, pipeline_id)
                
                if restart_result['success']:
                    click.echo("✅ Pipeline restarted successfully!")
                else:
                    click.echo(f"❌ Failed to restart pipeline: {restart_result.get('error', 'Unknown error')}", err=True)
                    
        else:
            click.echo(f"❌ Configuration failed: {result.get('error', 'Unknown error')}", err=True)
            
    except Exception as e:
        logger.error(f"Pipeline configuration failed: {e}")
        click.echo(f"❌ Pipeline configuration failed: {e}", err=True)
        raise click.Abort()


@pipeline_group.command()
@click.option('--pipeline-id', required=True, help='Pipeline ID to analyze')
@click.option('--metric', type=click.Choice(['performance', 'accuracy', 'bias', 'all']), default='all',
              help='Metrics to analyze')
@click.option('--output', '-o', type=click.Path(), help='Output file for analysis results')
@click.pass_context
def analyze(ctx, pipeline_id: str, metric: str, output: Optional[str]):
    """Analyze pipeline performance and results."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        click.echo(f"📊 Analyzing pipeline: {pipeline_id}")
        click.echo(f"🔍 Metrics: {metric}")
        
        analysis_result = _analyze_pipeline(auth_manager, pipeline_id, metric)
        
        if not analysis_result['success']:
            click.echo(f"❌ Analysis failed: {analysis_result.get('error', 'Unknown error')}", err=True)
            return
        
        # Display analysis results
        _display_analysis_results(analysis_result, metric)
        
        # Save results if output file specified
        if output:
            with open(output, 'w') as f:
                json.dump(analysis_result, f, indent=2)
            click.echo(f"📁 Analysis results saved to: {output}")
            
    except Exception as e:
        logger.error(f"Pipeline analysis failed: {e}")
        click.echo(f"❌ Pipeline analysis failed: {e}", err=True)
        raise click.Abort()


@pipeline_group.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input data file')
@click.option('--output', '-o', type=click.Path(), help='Output file')
@click.option('--format', type=click.Choice(['json', 'csv', 'parquet']), default='json',
              help='Output format')
@click.pass_context
def validate_input(ctx, input: str, output: Optional[str], format: str):
    """Validate input data for pipeline compatibility."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"❌ Input file not found: {input}", err=True)
            return
        
        click.echo(f"🔍 Validating input data: {input_path}")
        
        validation_result = _validate_pipeline_input(auth_manager, input_path)
        
        if validation_result['valid']:
            click.echo("✅ Input data is valid for pipeline processing!")
            
            if validation_result.get('warnings'):
                click.echo("⚠️  Warnings:")
                for warning in validation_result['warnings']:
                    click.echo(f"  • {warning}")
                    
        else:
            click.echo("❌ Input data validation failed!")
            
            if validation_result.get('errors'):
                click.echo("❌ Errors:")
                for error in validation_result['errors']:
                    click.echo(f"  • {error}")
        
        # Display validation summary
        click.echo(f"\n📊 Validation Summary:")
        click.echo(f"  Records: {validation_result.get('record_count', 0)}")
        click.echo(f"  Columns: {validation_result.get('column_count', 0)}")
        click.echo(f"  Missing data: {validation_result.get('missing_data_ratio', 0)*100:.1f}%")
        
        # Save detailed results if output specified
        if output:
            output_path = Path(output)
            with open(output_path, 'w') as f:
                json.dump(validation_result, f, indent=2)
            click.echo(f"📁 Validation results saved to: {output_path}")
            
    except Exception as e:
        logger.error(f"Input validation failed: {e}")
        click.echo(f"❌ Input validation failed: {e}", err=True)
        raise click.Abort()


# Helper functions

def _start_pipeline(auth_manager: AuthManager, config: Dict[str, Any], input_path: Path,
                   output_path: Path, name: Optional[str], priority: str,
                   tags: List[str]) -> Dict[str, Any]:
    """Start a new pipeline execution."""
    logger.info(f"Starting pipeline with config: {config}")
    
    # This would call the actual API endpoint
    # For now, return mock result
    return {
        'success': True,
        'pipeline_id': f'pipeline_{int(time.time())}',
        'message': 'Pipeline started successfully'
    }


def _stop_pipeline(auth_manager: AuthManager, pipeline_id: str) -> Dict[str, Any]:
    """Stop a specific pipeline."""
    logger.info(f"Stopping pipeline: {pipeline_id}")
    
    # This would call the actual API endpoint
    return {
        'success': True,
        'message': f'Pipeline {pipeline_id} stopped successfully'
    }


def _stop_all_pipelines(auth_manager: AuthManager) -> Dict[str, Any]:
    """Stop all running pipelines."""
    logger.info("Stopping all pipelines")
    
    # This would call the actual API endpoint
    return {
        'success': True,
        'stopped_count': 3,
        'message': 'All pipelines stopped successfully'
    }


def _get_pipeline_status(auth_manager: AuthManager, pipeline_id: str) -> Optional[Dict[str, Any]]:
    """Get pipeline status information."""
    # This would call the actual API endpoint
    # For now, return mock data
    return {
        'id': pipeline_id,
        'status': 'running',
        'progress': 65,
        'stage': 'data_processing',
        'estimated_completion': '2024-01-01T15:00:00Z',
        'metrics': {
            'items_processed': 150,
            'total_items': 230,
            'errors': 2,
            'warnings': 5
        },
        'performance': {
            'processing_rate': 25.5,  # items per second
            'estimated_remaining_time': 180  # seconds
        }
    }


def _display_simple_status(status: Dict[str, Any], verbose: bool) -> None:
    """Display simple pipeline status."""
    progress_bar = "█" * (status['progress'] // 10) + "░" * (10 - status['progress'] // 10)
    
    click.echo(f"[{progress_bar}] {status['progress']}% | {status['stage']}")
    click.echo(f"Status: {status['status']}")
    
    if verbose:
        click.echo(f"Processed: {status['metrics']['items_processed']}/{status['metrics']['total_items']}")
        click.echo(f"Errors: {status['metrics']['errors']}, Warnings: {status['metrics']['warnings']}")
        
        if 'performance' in status:
            perf = status['performance']
            click.echo(f"Rate: {perf['processing_rate']:.1f} items/sec")
            
            if perf.get('estimated_remaining_time'):
                minutes = perf['estimated_remaining_time'] // 60
                seconds = perf['estimated_remaining_time'] % 60
                click.echo(f"ETA: {minutes}m {seconds}s")


def _display_detailed_status(status: Dict[str, Any], verbose: bool) -> None:
    """Display detailed pipeline status."""
    click.echo(f"Pipeline ID: {status['id']}")
    click.echo(f"Status: {status['status']}")
    click.echo(f"Progress: {status['progress']}%")
    click.echo(f"Current Stage: {status['stage']}")
    
    if status.get('estimated_completion'):
        click.echo(f"Estimated Completion: {status['estimated_completion']}")
    
    click.echo(f"\nMetrics:")
    click.echo(f"  Items Processed: {status['metrics']['items_processed']}/{status['metrics']['total_items']}")
    click.echo(f"  Errors: {status['metrics']['errors']}")
    click.echo(f"  Warnings: {status['metrics']['warnings']}")
    
    if verbose and 'performance' in status:
        perf = status['performance']
        click.echo(f"\nPerformance:")
        click.echo(f"  Processing Rate: {perf['processing_rate']:.1f} items/second")
        
        if perf.get('estimated_remaining_time'):
            minutes = perf['estimated_remaining_time'] // 60
            seconds = perf['estimated_remaining_time'] % 60
            click.echo(f"  Estimated Time Remaining: {minutes}m {seconds}s")


def _list_pipeline_executions(auth_manager: AuthManager, status: str, limit: int) -> List[Dict[str, Any]]:
    """List pipeline executions."""
    # This would call the actual API endpoint
    # For now, return mock data
    executions = [
        {
            'id': 'pipeline_1',
            'name': 'Bias Detection Pipeline',
            'status': 'completed',
            'type': 'bias-detection',
            'created_at': '2024-01-01T12:00:00Z',
            'completed_at': '2024-01-01T12:30:00Z',
            'progress': 100
        },
        {
            'id': 'pipeline_2',
            'name': 'Dialogue Generation Pipeline',
            'status': 'running',
            'type': 'dialogue-generation',
            'created_at': '2024-01-01T13:00:00Z',
            'progress': 65
        }
    ]
    
    if status != 'all':
        executions = [exec for exec in executions if exec['status'] == status]
    
    return executions[:limit]


def _display_execution_summary(execution: Dict[str, Any], detailed: bool) -> None:
    """Display pipeline execution summary."""
    status_icon = _get_status_icon(execution['status'])
    
    click.echo(f"{status_icon} {execution['name']} ({execution['id']})")
    click.echo(f"   Status: {execution['status']}")
    click.echo(f"   Type: {execution['type']}")
    click.echo(f"   Created: {execution['created_at']}")
    
    if execution.get('completed_at'):
        click.echo(f"   Completed: {execution['completed_at']}")
    
    if execution.get('progress') is not None:
        click.echo(f"   Progress: {execution['progress']}%")
    
    if detailed:
        # Add more detailed information
        pass
    
    click.echo()


def _get_status_icon(status: str) -> str:
    """Get status icon."""
    icons = {
        'running': '🟢',
        'completed': '✅',
        'failed': '❌',
        'cancelled': '⏹️',
        'pending': '🟡'
    }
    return icons.get(status, '❓')


def _configure_pipeline(auth_manager: AuthManager, pipeline_id: str,
                       config_updates: Dict[str, Any]) -> Dict[str, Any]:
    """Configure pipeline parameters."""
    logger.info(f"Configuring pipeline {pipeline_id} with updates: {config_updates}")
    
    # This would call the actual API endpoint
    return {
        'success': True,
        'message': 'Pipeline configuration updated successfully'
    }


def _restart_pipeline(auth_manager: AuthManager, pipeline_id: str) -> Dict[str, Any]:
    """Restart a pipeline."""
    logger.info(f"Restarting pipeline: {pipeline_id}")
    
    # This would call the actual API endpoint
    return {
        'success': True,
        'message': f'Pipeline {pipeline_id} restarted successfully'
    }


def _analyze_pipeline(auth_manager: AuthManager, pipeline_id: str,
                     metric: str) -> Dict[str, Any]:
    """Analyze pipeline performance and results."""
    logger.info(f"Analyzing pipeline {pipeline_id} for metrics: {metric}")
    
    # This would call the actual API endpoint
    # For now, return mock analysis results
    return {
        'success': True,
        'pipeline_id': pipeline_id,
        'analysis': {
            'performance': {
                'total_execution_time': 1800,  # seconds
                'average_stage_time': 300,
                'bottleneck_stages': ['data_loading', 'model_inference'],
                'optimization_suggestions': [
                    'Increase batch size for data loading',
                    'Use GPU acceleration for model inference'
                ]
            },
            'accuracy': {
                'overall_accuracy': 0.94,
                'precision': 0.92,
                'recall': 0.96,
                'f1_score': 0.94
            },
            'bias': {
                'demographic_parity': 0.89,
                'equalized_odds': 0.91,
                'bias_score': 0.12,
                'recommendations': [
                    'Consider rebalancing training data',
                    'Implement fairness constraints'
                ]
            }
        }
    }


def _display_analysis_results(analysis_result: Dict[str, Any], metric: str) -> None:
    """Display pipeline analysis results."""
    analysis = analysis_result['analysis']
    
    if metric in ['performance', 'all'] and 'performance' in analysis:
        perf = analysis['performance']
        click.echo("📊 Performance Analysis:")
        click.echo(f"  Total execution time: {perf['total_execution_time']/60:.1f} minutes")
        click.echo(f"  Average stage time: {perf['average_stage_time']/60:.1f} minutes")
        
        if perf.get('bottleneck_stages'):
            click.echo("  Bottleneck stages:")
            for stage in perf['bottleneck_stages']:
                click.echo(f"    • {stage}")
        
        if perf.get('optimization_suggestions'):
            click.echo("  Optimization suggestions:")
            for suggestion in perf['optimization_suggestions']:
                click.echo(f"    • {suggestion}")
        
        click.echo()
    
    if metric in ['accuracy', 'all'] and 'accuracy' in analysis:
        acc = analysis['accuracy']
        click.echo("🎯 Accuracy Analysis:")
        click.echo(f"  Overall accuracy: {acc['overall_accuracy']*100:.1f}%")
        click.echo(f"  Precision: {acc['precision']*100:.1f}%")
        click.echo(f"  Recall: {acc['recall']*100:.1f}%")
        click.echo(f"  F1-score: {acc['f1_score']*100:.1f}%")
        click.echo()
    
    if metric in ['bias', 'all'] and 'bias' in analysis:
        bias = analysis['bias']
        click.echo("⚖️  Bias Analysis:")
        click.echo(f"  Demographic parity: {bias['demographic_parity']*100:.1f}%")
        click.echo(f"  Equalized odds: {bias['equalized_odds']*100:.1f}%")
        click.echo(f"  Bias score: {bias['bias_score']:.3f}")
        
        if bias.get('recommendations'):
            click.echo("  Recommendations:")
            for rec in bias['recommendations']:
                click.echo(f"    • {rec}")
        
        click.echo()


def _validate_pipeline_input(auth_manager: AuthManager, input_path: Path) -> Dict[str, Any]:
    """Validate input data for pipeline compatibility."""
    logger.info(f"Validating pipeline input: {input_path}")
    
    # This would call the actual API endpoint
    # For now, return mock validation results
    return {
        'valid': True,
        'record_count': 1000,
        'column_count': 15,
        'missing_data_ratio': 0.02,
        'warnings': [
            'Column "age" has some outliers that may affect model performance',
            'Missing values detected in optional fields'
        ],
        'errors': [],
        'recommendations': [
            'Consider normalizing numerical features',
            'Check for data leakage between training and validation sets'
        ]
    }


def _save_pipeline_info(pipeline_id: str, pipeline_info: Dict[str, Any]) -> None:
    """Save pipeline information for later reference."""
    # This would save to a local cache or config file
    logger.info(f"Saved pipeline info for {pipeline_id}")