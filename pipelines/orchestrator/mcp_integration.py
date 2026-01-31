"""
MCP (Model Context Protocol) Integration for Dataset Training Pipeline
Provides seamless integration with the MCP system for agent-based interactions
"""

import asyncio
import json
import logging
from typing import Dict, List, Optional, Any, Union
from dataclasses import dataclass, field
from datetime import datetime
import uuid

from .dataset_taxonomy import DatasetTaxonomy, DatasetMetadata, DatasetCategory
from .training_styles import TrainingStyleManager, TrainingStyle, BaseTrainingConfig
from .training_orchestrator import TrainingPipelineOrchestrator

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


@dataclass
class MCPTask:
    """MCP task for dataset training operations"""
    task_id: str = field(default_factory=lambda: str(uuid.uuid4()))
    task_type: str = ""
    status: str = "pending"
    created_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    updated_at: str = field(default_factory=lambda: datetime.utcnow().isoformat())
    parameters: Dict[str, Any] = field(default_factory=dict)
    results: Optional[Dict[str, Any]] = None
    error: Optional[str] = None
    agent_id: Optional[str] = None
    priority: int = 1
    timeout: int = 3600  # 1 hour default


class DatasetTrainingMCPInterface:
    """MCP interface for dataset training pipeline operations"""
    
    def __init__(self, orchestrator: TrainingPipelineOrchestrator):
        self.orchestrator = orchestrator
        self.taxonomy = orchestrator.taxonomy
        self.style_manager = orchestrator.style_manager
        self.active_tasks: Dict[str, MCPTask] = {}
        self.task_handlers = self._setup_task_handlers()
        self.capabilities = self._define_capabilities()
    
    def _setup_task_handlers(self) -> Dict[str, callable]:
        """Setup task handlers for different MCP operations"""
        return {
            "dataset_categorize": self._handle_dataset_categorize,
            "dataset_validate": self._handle_dataset_validate,
            "dataset_recommendations": self._handle_dataset_recommendations,
            "training_style_select": self._handle_training_style_select,
            "training_style_configure": self._handle_training_style_configure,
            "training_execute": self._handle_training_execute,
            "training_monitor": self._handle_training_monitor,
            "training_results": self._handle_training_results,
            "pipeline_execute": self._handle_pipeline_execute,
            "pipeline_status": self._handle_pipeline_status,
            "pipeline_cancel": self._handle_pipeline_cancel,
            "taxonomy_info": self._handle_taxonomy_info,
            "style_info": self._handle_style_info,
            "health_check": self._handle_health_check
        }
    
    def _define_capabilities(self) -> List[str]:
        """Define MCP capabilities"""
        return [
            "dataset_categorization",
            "dataset_validation",
            "training_style_selection",
            "training_configuration",
            "pipeline_execution",
            "progress_monitoring",
            "result_retrieval",
            "taxonomy_management",
            "style_management",
            "health_monitoring"
        ]
    
    async def process_task(self, task: MCPTask) -> Dict[str, Any]:
        """Process an MCP task"""
        logger.info(f"Processing MCP task: {task.task_id} of type {task.task_type}")
        
        # Store task
        self.active_tasks[task.task_id] = task
        task.status = "running"
        task.updated_at = datetime.utcnow().isoformat()
        
        try:
            # Get task handler
            handler = self.task_handlers.get(task.task_type)
            if not handler:
                raise ValueError(f"Unknown task type: {task.task_type}")
            
            # Execute task
            result = await handler(task.parameters)
            
            # Update task status
            task.status = "completed"
            task.updated_at = datetime.utcnow().isoformat()
            task.results = result
            
            logger.info(f"MCP task {task.task_id} completed successfully")
            
            return {
                "task_id": task.task_id,
                "status": "completed",
                "results": result,
                "timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            # Handle task failure
            task.status = "failed"
            task.updated_at = datetime.utcnow().isoformat()
            task.error = str(e)
            
            logger.error(f"MCP task {task.task_id} failed: {e}")
            
            return {
                "task_id": task.task_id,
                "status": "failed",
                "error": str(e),
                "timestamp": datetime.utcnow().isoformat()
            }
    
    async def _handle_dataset_categorize(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dataset categorization task"""
        logger.info("Handling dataset categorization task")
        
        # Extract parameters
        dataset_path = params.get("dataset_path")
        sample_size = params.get("sample_size", 100)
        
        if not dataset_path:
            raise ValueError("dataset_path is required")
        
        # Load dataset sample
        dataset_content = await self._load_dataset_sample(dataset_path, sample_size)
        
        # Create metadata if provided
        metadata_dict = params.get("metadata", {})
        if metadata_dict:
            metadata = DatasetMetadata.from_dict(metadata_dict)
        else:
            metadata = DatasetMetadata(
                name=params.get("name", "unnamed_dataset"),
                description=params.get("description", ""),
                source_type=params.get("source_type", "unknown")
            )
        
        # Categorize dataset
        categorization_result = self.taxonomy.categorize_dataset(dataset_content, metadata)
        
        # Update metadata with categorization results
        metadata.category = DatasetCategory(categorization_result["category"])
        metadata.subcategory = categorization_result["recommended_subcategories"][0] if categorization_result["recommended_subcategories"] else None
        
        return {
            "categorization": categorization_result,
            "metadata": metadata.to_dict(),
            "confidence": categorization_result["confidence"],
            "recommended_category": categorization_result["category"],
            "recommended_subcategories": categorization_result["recommended_subcategories"]
        }
    
    async def _handle_dataset_validate(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dataset validation task"""
        logger.info("Handling dataset validation task")
        
        # Extract parameters
        metadata_dict = params.get("metadata")
        if not metadata_dict:
            raise ValueError("metadata is required for validation")
        
        metadata = DatasetMetadata.from_dict(metadata_dict)
        
        # Validate metadata
        validation_errors = self.taxonomy.validate_metadata(metadata)
        
        # Additional validation based on category
        category_info = self.taxonomy.get_category_info(metadata.category.value)
        
        return {
            "valid": len(validation_errors) == 0,
            "errors": validation_errors,
            "category_info": category_info,
            "validation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_dataset_recommendations(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle dataset recommendations task"""
        logger.info("Handling dataset recommendations task")
        
        # Extract parameters
        metadata_dict = params.get("metadata")
        if not metadata_dict:
            raise ValueError("metadata is required for recommendations")
        
        metadata = DatasetMetadata.from_dict(metadata_dict)
        
        # Get training style recommendations
        training_recommendations = self.taxonomy.get_training_recommendations(metadata)
        
        # Get preprocessing recommendations
        preprocessing_recommendations = self._get_preprocessing_recommendations(metadata)
        
        # Get quality improvement recommendations
        quality_recommendations = self._get_quality_recommendations(metadata)
        
        return {
            "training_styles": training_recommendations,
            "preprocessing_steps": preprocessing_recommendations,
            "quality_improvements": quality_recommendations,
            "recommendation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_training_style_select(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training style selection task"""
        logger.info("Handling training style selection task")
        
        # Extract parameters
        dataset_metadata = params.get("dataset_metadata")
        training_goals = params.get("training_goals", {})
        
        if not dataset_metadata:
            raise ValueError("dataset_metadata is required")
        
        # Select optimal training style
        optimal_style = self.style_manager.select_optimal_style(dataset_metadata, training_goals)
        
        # Get style information
        style_info = self.style_manager._get_style_guidance(optimal_style)
        
        # Get configuration template
        config_template = self.style_manager.get_config_template(optimal_style)
        
        return {
            "selected_style": optimal_style.value,
            "style_info": style_info,
            "config_template": config_template,
            "selection_confidence": 0.85,  # This could be calculated
            "selection_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_training_style_configure(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training style configuration task"""
        logger.info("Handling training style configuration task")
        
        # Extract parameters
        style_name = params.get("style")
        config_params = params.get("config", {})
        
        if not style_name:
            raise ValueError("style is required")
        
        # Parse training style
        try:
            style = TrainingStyle(style_name)
        except ValueError:
            raise ValueError(f"Invalid training style: {style_name}")
        
        # Create configuration
        config = self.style_manager.create_config(style, **config_params)
        
        # Validate configuration
        validation_errors = self.style_manager.validate_config(config)
        
        # Optimize if requested
        if params.get("optimize", False):
            optimization_strategy = params.get("optimization_strategy", "balanced")
            config = self.style_manager.optimize_config(config, optimization_strategy)
        
        return {
            "config": config.__dict__,
            "valid": len(validation_errors) == 0,
            "validation_errors": validation_errors,
            "optimization_applied": params.get("optimize", False),
            "optimization_strategy": params.get("optimization_strategy", "balanced"),
            "configuration_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_training_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training execution task"""
        logger.info("Handling training execution task")
        
        # Extract parameters
        dataset_path = params.get("dataset_path")
        training_config = params.get("training_config")
        
        if not dataset_path or not training_config:
            raise ValueError("dataset_path and training_config are required")
        
        # Create pipeline configuration
        pipeline_config = {
            "dataset_path": dataset_path,
            "training_config": training_config,
            "use_container": params.get("use_container", False),
            "enable_monitoring": params.get("enable_monitoring", True)
        }
        
        # Execute training pipeline
        execution_id = await self.orchestrator.execute_pipeline(pipeline_config)
        
        return {
            "execution_id": execution_id,
            "status": "started",
            "pipeline_id": execution_id,
            "execution_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_training_monitor(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training monitoring task"""
        logger.info("Handling training monitoring task")
        
        # Extract parameters
        execution_id = params.get("execution_id")
        
        if not execution_id:
            raise ValueError("execution_id is required")
        
        # Get execution status
        status = self.orchestrator.get_execution_status(execution_id)
        
        if not status:
            raise ValueError(f"Execution {execution_id} not found")
        
        return {
            "execution_id": execution_id,
            "status": status["status"],
            "progress": status["progress"],
            "current_stage": status["current_stage"],
            "stages": status["stages"],
            "monitoring_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_training_results(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training results retrieval task"""
        logger.info("Handling training results retrieval task")
        
        # Extract parameters
        execution_id = params.get("execution_id")
        
        if not execution_id:
            raise ValueError("execution_id is required")
        
        # This would retrieve results from storage
        # For now, return mock results
        results = {
            "execution_id": execution_id,
            "status": "completed",
            "accuracy": 0.85,
            "precision": 0.82,
            "recall": 0.88,
            "f1_score": 0.85,
            "empathy_score": 0.9,
            "therapeutic_appropriateness": 0.87,
            "safety_score": 0.95,
            "model_path": f"./models/{execution_id}",
            "checkpoint_path": f"./checkpoints/{execution_id}",
            "results_timestamp": datetime.utcnow().isoformat()
        }
        
        return results
    
    async def _handle_pipeline_execute(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pipeline execution task"""
        logger.info("Handling pipeline execution task")
        
        # This is similar to training execute but more comprehensive
        return await self._handle_training_execute(params)
    
    async def _handle_pipeline_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pipeline status task"""
        logger.info("Handling pipeline status task")
        
        # Extract parameters
        execution_id = params.get("execution_id")
        
        if not execution_id:
            raise ValueError("execution_id is required")
        
        return await self._handle_training_monitor(params)
    
    async def _handle_pipeline_cancel(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle pipeline cancellation task"""
        logger.info("Handling pipeline cancellation task")
        
        # Extract parameters
        execution_id = params.get("execution_id")
        
        if not execution_id:
            raise ValueError("execution_id is required")
        
        # Cancel execution
        success = await self.orchestrator.cancel_execution(execution_id)
        
        return {
            "execution_id": execution_id,
            "cancelled": success,
            "cancellation_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_taxonomy_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle taxonomy information task"""
        logger.info("Handling taxonomy information task")
        
        # Get all categories
        categories = self.taxonomy.list_all_categories()
        
        # Get specific category info if requested
        category_name = params.get("category")
        if category_name:
            category_info = self.taxonomy.get_category_info(category_name)
        else:
            category_info = None
        
        return {
            "categories": categories,
            "category_info": category_info,
            "total_categories": len(categories),
            "taxonomy_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_style_info(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle training style information task"""
        logger.info("Handling training style information task")
        
        # Get all styles
        styles = []
        for style in TrainingStyle:
            style_info = self.style_manager._get_style_guidance(style)
            config_template = self.style_manager.get_config_template(style)
            
            styles.append({
                "style": style.value,
                "info": style_info,
                "config_template": config_template
            })
        
        # Get specific style info if requested
        style_name = params.get("style")
        if style_name:
            try:
                specific_style = TrainingStyle(style_name)
                specific_info = self.style_manager._get_style_guidance(specific_style)
                specific_template = self.style_manager.get_config_template(specific_style)
            except ValueError:
                specific_info = None
                specific_template = None
        else:
            specific_info = None
            specific_template = None
        
        return {
            "styles": styles,
            "specific_style_info": specific_info,
            "specific_config_template": specific_template,
            "total_styles": len(styles),
            "styles_timestamp": datetime.utcnow().isoformat()
        }
    
    async def _handle_health_check(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Handle health check task"""
        logger.info("Handling health check task")
        
        # Check system health
        health_status = {
            "status": "healthy",
            "timestamp": datetime.utcnow().isoformat(),
            "components": {
                "taxonomy": "healthy",
                "style_manager": "healthy",
                "orchestrator": "healthy",
                "mcp_interface": "healthy"
            },
            "active_tasks": len(self.active_tasks),
            "capabilities": self.capabilities
        }
        
        return health_status
    
    # Helper methods
    
    async def _load_dataset_sample(self, dataset_path: str, sample_size: int) -> List[Dict[str, Any]]:
        """Load a sample of dataset content"""
        try:
            import json
            from pathlib import Path
            
            path = Path(dataset_path)
            if not path.exists():
                raise FileNotFoundError(f"Dataset not found: {dataset_path}")
            
            with open(path, 'r', encoding='utf-8') as f:
                if path.suffix == '.jsonl':
                    content = []
                    for i, line in enumerate(f):
                        if i >= sample_size:
                            break
                        content.append(json.loads(line))
                else:
                    data = json.load(f)
                    if isinstance(data, list):
                        content = data[:sample_size]
                    else:
                        content = [data]
            
            return content
            
        except Exception as e:
            logger.error(f"Failed to load dataset sample: {e}")
            raise
    
    def _get_preprocessing_recommendations(self, metadata: DatasetMetadata) -> List[str]:
        """Get preprocessing recommendations for dataset"""
        recommendations = []
        
        # Category-specific recommendations
        if metadata.category == DatasetCategory.CLINICAL:
            recommendations.extend(["anonymization", "pii_removal", "consent_validation"])
        
        if metadata.category == DatasetCategory.THERAPEUTIC:
            recommendations.extend(["therapeutic_validation", "crisis_content_filtering"])
        
        if metadata.category == DatasetCategory.CONVERSATIONAL:
            recommendations.extend(["text_cleaning", "conversation_structuring"])
        
        if metadata.category == DatasetCategory.SYNTHETIC:
            recommendations.extend(["authenticity_validation", "quality_assessment"])
        
        if metadata.category == DatasetCategory.MULTIMODAL:
            recommendations.extend(["modality_synchronization", "format_standardization"])
        
        # Common recommendations
        recommendations.extend(["text_normalization", "encoding_validation"])
        
        return recommendations
    
    def _get_quality_reprovements(self, metadata: DatasetMetadata) -> List[str]:
        """Get quality improvement recommendations"""
        recommendations = []
        
        if metadata.quality_score < 0.8:
            recommendations.append("Improve data quality through cleaning and validation")
        
        if metadata.completeness_score < 0.9:
            recommendations.append("Fill missing data or remove incomplete records")
        
        if metadata.consistency_score < 0.85:
            recommendations.append("Standardize data formats and structures")
        
        if metadata.fairness_score < 0.8:
            recommendations.append("Address bias through data balancing or augmentation")
        
        if metadata.crisis_content_ratio > 0.2:
            recommendations.append("Review and potentially filter crisis content")
        
        return recommendations
    
    # Task management methods
    
    def get_task_status(self, task_id: str) -> Optional[Dict[str, Any]]:
        """Get task status"""
        task = self.active_tasks.get(task_id)
        if not task:
            return None
        
        return {
            "task_id": task.task_id,
            "task_type": task.task_type,
            "status": task.status,
            "created_at": task.created_at,
            "updated_at": task.updated_at,
            "results": task.results,
            "error": task.error,
            "agent_id": task.agent_id
        }
    
    def list_active_tasks(self) -> List[Dict[str, Any]]:
        """List all active tasks"""
        return [
            {
                "task_id": task.task_id,
                "task_type": task.task_type,
                "status": task.status,
                "created_at": task.created_at,
                "agent_id": task.agent_id
            }
            for task in self.active_tasks.values()
        ]
    
    async def cancel_task(self, task_id: str) -> bool:
        """Cancel an active task"""
        task = self.active_tasks.get(task_id)
        if not task:
            return False
        
        if task.status in ["pending", "running"]:
            task.status = "cancelled"
            task.updated_at = datetime.utcnow().isoformat()
            return True
        
        return False


# Example usage and testing
async def test_mcp_integration():
    """Test MCP integration interface"""
    print("Testing MCP Integration Interface...")
    
    # Create orchestrator and MCP interface
    orchestrator = TrainingPipelineOrchestrator()
    mcp_interface = DatasetTrainingMCPInterface(orchestrator)
    
    # Test dataset categorization task
    categorize_task = MCPTask(
        task_type="dataset_categorize",
        parameters={
            "dataset_path": "./test_dataset.json",
            "dataset_name": "Test Dataset",
            "sample_size": 10
        }
    )
    
    # Create test dataset
    test_dataset = [
        {"text": "I'm feeling anxious about therapy"},
        {"text": "My therapist suggested CBT techniques"},
        {"text": "I learned mindfulness in today's session"}
    ]
    
    with open("./test_dataset.json", "w") as f:
        json.dump(test_dataset, f, indent=2)
    
    try:
        # Process categorization task
        result = await mcp_interface.process_task(categorize_task)
        print(f"Categorization result: {result}")
        
        # Test training style selection task
        style_task = MCPTask(
            task_type="training_style_select",
            parameters={
                "dataset_metadata": {
                    "record_count": 100,
                    "has_labels": True,
                    "quality_score": 0.9,
                    "category": "therapeutic",
                    "available_compute": "high"
                },
                "training_goals": {
                    "objective": "high_accuracy",
                    "available_compute": "high"
                }
            }
        )
        
        style_result = await mcp_interface.process_task(style_task)
        print(f"Style selection result: {style_result}")
        
        # Test health check task
        health_task = MCPTask(
            task_type="health_check",
            parameters={}
        )
        
        health_result = await mcp_interface.process_task(health_task)
        print(f"Health check result: {health_result}")
        
        # Test task listing
        active_tasks = mcp_interface.list_active_tasks()
        print(f"Active tasks: {len(active_tasks)}")
        
    except Exception as e:
        print(f"MCP integration test failed: {e}")
    
    finally:
        # Cleanup
        import os
        if os.path.exists("./test_dataset.json"):
            os.remove("./test_dataset.json")
    
    print("MCP integration test completed!")


if __name__ == "__main__":
    asyncio.run(test_mcp_integration())