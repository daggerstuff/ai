"""
CLI Interface Commands for Pixelated AI CLI

This module provides commands for direct CLI operations with the Pixelated platform,
including running pipelines, managing data, and performing batch operations.
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


@click.group(name='cli-interface')
@click.pass_context
def cli_interface_group(ctx):
    """Direct CLI operations for Pixelated platform."""
    setup_logging(ctx.obj.get('verbose', False))
    logger.info("CLI interface command group initialized")


@cli_interface_group.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input file or directory')
@click.option('--pipeline', '-p', required=True,
              type=click.Choice(['bias-detection', 'dialogue-generation', 
                               'model-training', 'data-analysis', 'sentiment-analysis']),
              help='Pipeline type to run')
@click.option('--output', '-o', type=click.Path(), help='Output directory')
@click.option('--config', '-c', type=click.Path(exists=True), help='Pipeline configuration file')
@click.option('--batch-size', default=100, help='Batch size for processing')
@click.option('--parallel', default=1, help='Number of parallel workers')
@click.option('--dry-run', is_flag=True, help='Preview without executing')
@click.pass_context
def run(ctx, input: str, pipeline: str, output: Optional[str], config: Optional[str],
        batch_size: int, parallel: int, dry_run: bool):
    """Run a pipeline directly from CLI."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        input_path = Path(input)
        output_path = Path(output) if output else Path.cwd() / 'output'
        output_path.mkdir(exist_ok=True)
        
        # Validate input
        if not input_path.exists():
            click.echo(f"❌ Input path not found: {input}", err=True)
            return
        
        # Load pipeline configuration
        pipeline_config = {}
        if config:
            with open(config, 'r') as f:
                pipeline_config = json.load(f)
        
        click.echo(f"🚀 Running {pipeline} pipeline...")
        click.echo(f"📁 Input: {input_path}")
        click.echo(f"📤 Output: {output_path}")
        click.echo(f"📊 Batch size: {batch_size}")
        click.echo(f"⚡ Parallel workers: {parallel}")
        
        if dry_run:
            click.echo("🔍 DRY RUN MODE - Preview only:")
            _preview_pipeline(input_path, pipeline, pipeline_config)
            return
        
        # Execute pipeline
        pipeline_id = _execute_pipeline(
            auth_manager, input_path, pipeline, pipeline_config,
            output_path, batch_size, parallel
        )
        
        if pipeline_id:
            click.echo(f"✅ Pipeline started successfully!")
            click.echo(f"🆔 Pipeline ID: {pipeline_id}")
            click.echo(f"📊 Monitor progress with: pixelated cli-interface status --pipeline-id {pipeline_id}")
        else:
            click.echo("❌ Pipeline execution failed", err=True)
            
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        click.echo(f"❌ Pipeline execution failed: {e}", err=True)
        raise click.Abort()


@cli_interface_group.command()
@click.option('--pipeline-id', help='Specific pipeline to check')
@click.option('--all', 'list_all', is_flag=True, help='List all pipelines')
@click.option('--status', type=click.Choice(['running', 'completed', 'failed', 'pending']),
              help='Filter by status')
@click.option('--limit', default=10, help='Maximum number of results')
@click.pass_context
def status(ctx, pipeline_id: Optional[str], list_all: bool, status: Optional[str], limit: int):
    """Check pipeline status and results."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        if pipeline_id:
            # Get specific pipeline status
            pipeline_info = _get_pipeline_status(auth_manager, pipeline_id)
            if pipeline_info:
                _display_pipeline_details(pipeline_info)
            else:
                click.echo(f"❌ Pipeline {pipeline_id} not found", err=True)
                
        elif list_all:
            # List all pipelines
            pipelines = _list_pipelines(auth_manager, status, limit)
            _display_pipelines_list(pipelines)
            
        else:
            click.echo("❌ Please specify --pipeline-id or --all")
            
    except Exception as e:
        logger.error(f"Status check failed: {e}")
        click.echo(f"❌ Status check failed: {e}", err=True)
        raise click.Abort()


@cli_interface_group.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Input data file')
@click.option('--output', '-o', type=click.Path(), help='Output file')
@click.option('--format', type=click.Choice(['json', 'csv', 'parquet']), 
              default='json', help='Output format')
@click.option('--validate-only', is_flag=True, help='Only validate, don\'t process')
@click.pass_context
def process(ctx, input: str, output: Optional[str], format: str, validate_only: bool):
    """Process data files through various transformations."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"❌ Input file not found: {input}", err=True)
            return
        
        # Determine output path
        if output:
            output_path = Path(output)
        else:
            suffix = '.json' if format == 'json' else '.csv' if format == 'csv' else '.parquet'
            output_path = input_path.with_suffix(suffix)
        
        click.echo(f"📊 Processing data file...")
        click.echo(f"📁 Input: {input_path}")
        click.echo(f"📤 Output: {output_path}")
        click.echo(f"📋 Format: {format}")
        
        if validate_only:
            click.echo("🔍 Validation mode - checking data integrity...")
            validation_result = _validate_data_file(input_path)
            _display_validation_results(validation_result)
            return
        
        # Process the file
        result = _process_data_file(auth_manager, input_path, output_path, format)
        
        if result['success']:
            click.echo(f"✅ Data processing completed successfully!")
            click.echo(f"📊 Records processed: {result.get('records_processed', 0)}")
            click.echo(f"⚠️  Warnings: {result.get('warnings', 0)}")
            click.echo(f"❌ Errors: {result.get('errors', 0)}")
            
            if result.get('output_size'):
                click.echo(f"📏 Output size: {format_file_size(result['output_size'])}")
        else:
            click.echo(f"❌ Data processing failed: {result.get('error', 'Unknown error')}", err=True)
            
    except Exception as e:
        logger.error(f"Data processing failed: {e}")
        click.echo(f"❌ Data processing failed: {e}", err=True)
        raise click.Abort()


@cli_interface_group.command()
@click.option('--query', '-q', required=True, help='Search query')
@click.option('--type', type=click.Choice(['models', 'datasets', 'pipelines', 'results']),
              default='models', help='Search type')
@click.option('--limit', default=20, help='Maximum results')
@click.option('--json-output', is_flag=True, help='Output in JSON format')
@click.pass_context
def search(ctx, query: str, type: str, limit: int, json_output: bool):
    """Search through models, datasets, and results."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        click.echo(f"🔍 Searching {type} for: {query}")
        
        results = _search_resources(auth_manager, query, type, limit)
        
        if not results:
            click.echo("❌ No results found")
            return
        
        if json_output:
            click.echo(json.dumps(results, indent=2))
        else:
            _display_search_results(results, type)
            
    except Exception as e:
        logger.error(f"Search failed: {e}")
        click.echo(f"❌ Search failed: {e}", err=True)
        raise click.Abort()


@cli_interface_group.command()
@click.option('--models', is_flag=True, help='List available models')
@click.option('--datasets', is_flag=True, help='List available datasets')
@click.option('--pipelines', is_flag=True, help='List available pipelines')
@click.option('--detailed', is_flag=True, help='Show detailed information')
@click.pass_context
def list_resources(ctx, models: bool, datasets: bool, pipelines: bool, detailed: bool):
    """List available resources (models, datasets, pipelines)."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        # If no specific resource type specified, show all
        show_all = not any([models, datasets, pipelines])
        
        if models or show_all:
            _list_models(auth_manager, detailed)
            
        if datasets or show_all:
            _list_datasets(auth_manager, detailed)
            
        if pipelines or show_all:
            _list_pipelines(auth_manager, detailed)
            
    except Exception as e:
        logger.error(f"Resource listing failed: {e}")
        click.echo(f"❌ Resource listing failed: {e}", err=True)
        raise click.Abort()


@cli_interface_group.command()
@click.option('--input', '-i', type=click.Path(exists=True), required=True,
              help='Batch input file with multiple operations')
@click.option('--output', '-o', type=click.Path(), help='Batch results output file')
@click.option('--parallel', default=1, help='Number of parallel operations')
@click.option('--continue-on-error', is_flag=True, help='Continue processing on individual errors')
@click.pass_context
def batch(ctx, input: str, output: Optional[str], parallel: int, continue_on_error: bool):
    """Execute batch operations from a configuration file."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        input_path = Path(input)
        if not input_path.exists():
            click.echo(f"❌ Batch input file not found: {input}", err=True)
            return
        
        # Load batch configuration
        with open(input_path, 'r') as f:
            batch_config = json.load(f)
        
        output_path = Path(output) if output else input_path.with_suffix('.results.json')
        
        click.echo(f"📋 Executing batch operations...")
        click.echo(f"📁 Input: {input_path}")
        click.echo(f"📤 Output: {output_path}")
        click.echo(f"⚡ Parallel workers: {parallel}")
        click.echo(f"🔧 Operations: {len(batch_config.get('operations', []))}")
        
        # Execute batch operations
        results = _execute_batch_operations(
            auth_manager, batch_config, parallel, continue_on_error
        )
        
        # Save results
        with open(output_path, 'w') as f:
            json.dump(results, f, indent=2)
        
        # Display summary
        _display_batch_summary(results)
        
        click.echo(f"✅ Batch operations completed!")
        click.echo(f"📊 Results saved to: {output_path}")
        
    except Exception as e:
        logger.error(f"Batch execution failed: {e}")
        click.echo(f"❌ Batch execution failed: {e}", err=True)
        raise click.Abort()


@cli_interface_group.command()
@click.option('--days', default=7, help='Number of days to show')
@click.option('--type', type=click.Choice(['all', 'pipeline', 'model', 'data']),
              default='all', help='Activity type filter')
@click.pass_context
def activity(ctx, days: int, type: str):
    """Show recent activity and usage statistics."""
    try:
        config_obj = get_config()
        auth_manager = AuthManager(config_obj)
        
        click.echo(f"📊 Activity Report (last {days} days)")
        click.echo("-" * 50)
        
        activity_data = _get_activity_data(auth_manager, days, type)
        
        if not activity_data:
            click.echo("❌ No activity data available")
            return
        
        _display_activity_report(activity_data, type)
        
    except Exception as e:
        logger.error(f"Activity report failed: {e}")
        click.echo(f"❌ Activity report failed: {e}", err=True)
        raise click.Abort()


# Helper functions

def _preview_pipeline(input_path: Path, pipeline: str, config: Dict[str, Any]) -> None:
    """Preview pipeline execution without running it."""
    click.echo("🔍 Pipeline Preview:")
    click.echo(f"  Input: {input_path}")
    click.echo(f"  Type: {pipeline}")
    click.echo(f"  Config: {json.dumps(config, indent=2) if config else 'Default'}")
    
    # Estimate processing time and resources
    if input_path.is_file():
        file_size = input_path.stat().st_size
        click.echo(f"  File size: {format_file_size(file_size)}")
        
        # Rough estimate based on file size and pipeline type
        estimated_time = _estimate_processing_time(file_size, pipeline)
        click.echo(f"  Estimated time: {estimated_time}")


def _execute_pipeline(auth_manager: AuthManager, input_path: Path, pipeline: str,
                     config: Dict[str, Any], output_path: Path, batch_size: int,
                     parallel: int) -> Optional[str]:
    """Execute a pipeline and return pipeline ID."""
    logger.info(f"Executing {pipeline} pipeline for {input_path}")
    
    # This would call the actual API endpoint
    # For now, return a mock pipeline ID
    return f"pipeline_{int(time.time())}"


def _get_pipeline_status(auth_manager: AuthManager, pipeline_id: str) -> Optional[Dict[str, Any]]:
    """Get pipeline status information."""
    # This would call the actual API endpoint
    # For now, return mock data
    return {
        'id': pipeline_id,
        'status': 'completed',
        'progress': 100,
        'created_at': '2024-01-01T12:00:00Z',
        'completed_at': '2024-01-01T12:05:00Z',
        'input_files': 1,
        'output_files': 3,
        'errors': 0
    }


def _list_pipelines(auth_manager: AuthManager, status: Optional[str], limit: int) -> List[Dict[str, Any]]:
    """List pipelines with optional status filter."""
    # This would call the actual API endpoint
    # For now, return mock data
    return [
        {
            'id': 'pipeline_1',
            'type': 'bias-detection',
            'status': 'completed',
            'created_at': '2024-01-01T12:00:00Z'
        },
        {
            'id': 'pipeline_2',
            'type': 'dialogue-generation',
            'status': 'running',
            'created_at': '2024-01-01T13:00:00Z'
        }
    ]


def _display_pipeline_details(pipeline_info: Dict[str, Any]) -> None:
    """Display detailed pipeline information."""
    click.echo(f"Pipeline ID: {pipeline_info['id']}")
    click.echo(f"Status: {pipeline_info['status']}")
    click.echo(f"Progress: {pipeline_info['progress']}%")
    click.echo(f"Created: {pipeline_info['created_at']}")
    
    if pipeline_info.get('completed_at'):
        click.echo(f"Completed: {pipeline_info['completed_at']}")
    
    click.echo(f"Input files: {pipeline_info.get('input_files', 0)}")
    click.echo(f"Output files: {pipeline_info.get('output_files', 0)}")
    click.echo(f"Errors: {pipeline_info.get('errors', 0)}")


def _display_pipelines_list(pipelines: List[Dict[str, Any]]) -> None:
    """Display list of pipelines."""
    if not pipelines:
        click.echo("No pipelines found")
        return
    
    click.echo("📋 Recent Pipelines:")
    for pipeline in pipelines:
        status_icon = "🟢" if pipeline['status'] == 'completed' else "🟡" if pipeline['status'] == 'running' else "🔴"
        click.echo(f"  {status_icon} {pipeline['id']} - {pipeline['type']} - {pipeline['status']}")


def _process_data_file(auth_manager: AuthManager, input_path: Path, output_path: Path,
                      format: str) -> Dict[str, Any]:
    """Process a data file and return results."""
    logger.info(f"Processing data file: {input_path} -> {output_path}")
    
    # This would call the actual API endpoint
    # For now, return mock results
    return {
        'success': True,
        'records_processed': 1000,
        'warnings': 2,
        'errors': 0,
        'output_size': 1024 * 1024  # 1MB
    }


def _validate_data_file(input_path: Path) -> Dict[str, Any]:
    """Validate data file integrity."""
    # This would perform actual validation
    return {
        'valid': True,
        'records': 1000,
        'columns': 15,
        'missing_data': 0.02,  # 2%
        'warnings': ['Column "age" has some outliers']
    }


def _display_validation_results(validation: Dict[str, Any]) -> None:
    """Display validation results."""
    click.echo("🔍 Validation Results:")
    click.echo(f"  Valid: {'✅' if validation['valid'] else '❌'}")
    click.echo(f"  Records: {validation['records']}")
    click.echo(f"  Columns: {validation['columns']}")
    click.echo(f"  Missing data: {validation['missing_data']*100:.1f}%")
    
    if validation.get('warnings'):
        click.echo("  ⚠️  Warnings:")
        for warning in validation['warnings']:
            click.echo(f"    - {warning}")


def _search_resources(auth_manager: AuthManager, query: str, type: str, limit: int) -> List[Dict[str, Any]]:
    """Search for resources."""
    # This would call the actual API endpoint
    # For now, return mock results
    return [
        {
            'id': 'model_1',
            'name': 'Therapeutic Response Model',
            'type': 'model',
            'description': 'AI model for generating therapeutic responses'
        },
        {
            'id': 'dataset_1',
            'name': 'Mental Health Dataset',
            'type': 'dataset',
            'description': 'Dataset of therapeutic conversations'
        }
    ]


def _display_search_results(results: List[Dict[str, Any]], type: str) -> None:
    """Display search results."""
    click.echo(f"🔍 Found {len(results)} results:")
    
    for result in results:
        click.echo(f"  📋 {result['name']} ({result['type']})")
        click.echo(f"     {result['description']}")
        click.echo(f"     ID: {result['id']}")
        click.echo()


def _list_models(auth_manager: AuthManager, detailed: bool) -> None:
    """List available models."""
    click.echo("🤖 Available Models:")
    models = [
        {'id': 'model_1', 'name': 'Therapeutic Response Model', 'status': 'active'},
        {'id': 'model_2', 'name': 'Bias Detection Model', 'status': 'active'}
    ]
    
    for model in models:
        status_icon = "🟢" if model['status'] == 'active' else "🔴"
        click.echo(f"  {status_icon} {model['name']} ({model['id']})")
        
        if detailed:
            click.echo(f"     Status: {model['status']}")


def _list_datasets(auth_manager: AuthManager, detailed: bool) -> None:
    """List available datasets."""
    click.echo("📊 Available Datasets:")
    datasets = [
        {'id': 'dataset_1', 'name': 'Therapeutic Conversations', 'records': 10000},
        {'id': 'dataset_2', 'name': 'Bias Detection Data', 'records': 5000}
    ]
    
    for dataset in datasets:
        click.echo(f"  📋 {dataset['name']} ({dataset['id']})")
        
        if detailed:
            click.echo(f"     Records: {dataset['records']}")


def _list_pipelines(auth_manager: AuthManager, detailed: bool) -> None:
    """List available pipelines."""
    click.echo("🔄 Available Pipelines:")
    pipelines = [
        {'id': 'pipeline_bias', 'name': 'Bias Detection Pipeline', 'stages': 3},
        {'id': 'pipeline_dialogue', 'name': 'Dialogue Generation Pipeline', 'stages': 5}
    ]
    
    for pipeline in pipelines:
        click.echo(f"  🔄 {pipeline['name']} ({pipeline['id']})")
        
        if detailed:
            click.echo(f"     Stages: {pipeline['stages']}")


def _execute_batch_operations(auth_manager: AuthManager, batch_config: Dict[str, Any],
                             parallel: int, continue_on_error: bool) -> Dict[str, Any]:
    """Execute batch operations."""
    logger.info(f"Executing {len(batch_config.get('operations', []))} batch operations")
    
    # This would call the actual API endpoints
    # For now, return mock results
    return {
        'total_operations': 5,
        'successful': 4,
        'failed': 1,
        'results': [
            {'operation': 'process_data', 'status': 'success'},
            {'operation': 'train_model', 'status': 'success'},
            {'operation': 'validate_results', 'status': 'failed', 'error': 'Validation error'},
            {'operation': 'generate_report', 'status': 'success'},
            {'operation': 'cleanup', 'status': 'success'}
        ]
    }


def _display_batch_summary(results: Dict[str, Any]) -> None:
    """Display batch operation summary."""
    click.echo("📊 Batch Operations Summary:")
    click.echo(f"  Total: {results['total_operations']}")
    click.echo(f"  ✅ Successful: {results['successful']}")
    click.echo(f"  ❌ Failed: {results['failed']}")
    
    if results['failed'] > 0:
        click.echo("  ⚠️  Failed operations:")
        for result in results['results']:
            if result['status'] == 'failed':
                click.echo(f"    - {result['operation']}: {result.get('error', 'Unknown error')}")


def _get_activity_data(auth_manager: AuthManager, days: int, type: str) -> Dict[str, Any]:
    """Get activity data for specified period."""
    # This would call the actual API endpoint
    # For now, return mock data
    return {
        'period': f"last_{days}_days",
        'pipelines_run': 15,
        'models_trained': 3,
        'datasets_processed': 8,
        'total_processing_time': 3600,  # seconds
        'average_pipeline_duration': 240,  # seconds
        'success_rate': 0.93
    }


def _display_activity_report(activity_data: Dict[str, Any], type: str) -> None:
    """Display activity report."""
    click.echo(f"📈 Activity Summary ({activity_data['period']}):")
    click.echo(f"  🔄 Pipelines run: {activity_data['pipelines_run']}")
    
    if type in ['all', 'model']:
        click.echo(f"  🤖 Models trained: {activity_data['models_trained']}")
    
    if type in ['all', 'data']:
        click.echo(f"  📊 Datasets processed: {activity_data['datasets_processed']}")
    
    click.echo(f"  ⏱️  Total processing time: {activity_data['total_processing_time']/3600:.1f} hours")
    click.echo(f"  ⚡ Average pipeline duration: {activity_data['average_pipeline_duration']/60:.1f} minutes")
    click.echo(f"  ✅ Success rate: {activity_data['success_rate']*100:.1f}%")


def _estimate_processing_time(file_size: int, pipeline: str) -> str:
    """Estimate processing time based on file size and pipeline type."""
    # Rough estimation logic
    base_time = file_size / (1024 * 1024)  # MB
    pipeline_multipliers = {
        'bias-detection': 2,
        'dialogue-generation': 5,
        'model-training': 10,
        'data-analysis': 1.5,
        'sentiment-analysis': 1
    }
    
    multiplier = pipeline_multipliers.get(pipeline, 3)
    estimated_seconds = base_time * multiplier
    
    if estimated_seconds < 60:
        return f"{estimated_seconds:.0f} seconds"
    elif estimated_seconds < 3600:
        return f"{estimated_seconds/60:.1f} minutes"
    else:
        return f"{estimated_seconds/3600:.1f} hours"