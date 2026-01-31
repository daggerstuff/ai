"""
Dataset Training Pipeline Integration Module
Main entry point for the dataset categorization and training style system
"""

import asyncio
import logging
from typing import Dict, List, Optional, Any, Union
from pathlib import Path

from .dataset_taxonomy import DatasetTaxonomy, DatasetMetadata, DatasetCategory
from .training_styles import TrainingStyleManager, TrainingStyle, BaseTrainingConfig
from .training_orchestrator import TrainingPipelineOrchestrator
from .mcp_integration import DatasetTrainingMCPInterface, MCPTask
from .api_integration import DatasetTrainingAPI

# Set up logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


class DatasetTrainingPipeline:
    """Main dataset training pipeline integration class"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """Initialize the dataset training pipeline"""
        self.config = config or {}
        
        # Initialize core components
        self.taxonomy = DatasetTaxonomy()
        self.style_manager = TrainingStyleManager()
        self.orchestrator = TrainingPipelineOrchestrator(self.config)
        self.mcp_interface = DatasetTrainingMCPInterface(self.orchestrator)
        self.api = DatasetTrainingAPI(self.orchestrator)
        
        logger.info("Dataset Training Pipeline initialized successfully")
    
    async def categorize_dataset(self, dataset_path: str, 
                               dataset_name: str = "unnamed_dataset",
                               sample_size: int = 100) -> Dict[str, Any]:
        """Categorize a dataset automatically"""
        logger.info(f"Categorizing dataset: {dataset_name}")
        
        # Create MCP task for categorization
        task = MCPTask(
            task_type="dataset_categorize",
            parameters={
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "sample_size": sample_size
            }
        )
        
        # Process task
        result = await self.mcp_interface.process_task(task)
        
        if result["status"] == "failed":
            raise RuntimeError(f"Dataset categorization failed: {result['error']}")
        
        return result["results"]
    
    async def select_training_style(self, dataset_metadata: Dict[str, Any], 
                                  training_goals: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
        """Select optimal training style for dataset"""
        logger.info("Selecting optimal training style")
        
        # Create MCP task for style selection
        task = MCPTask(
            task_type="training_style_select",
            parameters={
                "dataset_metadata": dataset_metadata,
                "training_goals": training_goals or {}
            }
        )
        
        # Process task
        result = await self.mcp_interface.process_task(task)
        
        if result["status"] == "failed":
            raise RuntimeError(f"Training style selection failed: {result['error']}")
        
        return result["results"]
    
    async def configure_training(self, style: str, 
                               config_params: Optional[Dict[str, Any]] = None,
                               optimize: bool = False) -> Dict[str, Any]:
        """Configure training parameters"""
        logger.info(f"Configuring training for style: {style}")
        
        # Create MCP task for configuration
        task = MCPTask(
            task_type="training_style_configure",
            parameters={
                "style": style,
                "config": config_params or {},
                "optimize": optimize
            }
        )
        
        # Process task
        result = await self.mcp_interface.process_task(task)
        
        if result["status"] == "failed":
            raise RuntimeError(f"Training configuration failed: {result['error']}")
        
        return result["results"]
    
    async def execute_training_pipeline(self, pipeline_config: Dict[str, Any]) -> str:
        """Execute complete training pipeline"""
        logger.info("Executing training pipeline")
        
        # Create MCP task for pipeline execution
        task = MCPTask(
            task_type="training_execute",
            parameters=pipeline_config
        )
        
        # Process task
        result = await self.mcp_interface.process_task(task)
        
        if result["status"] == "failed":
            raise RuntimeError(f"Pipeline execution failed: {result['error']}")
        
        return result["results"]["execution_id"]
    
    async def get_pipeline_status(self, execution_id: str) -> Dict[str, Any]:
        """Get pipeline execution status"""
        logger.info(f"Getting status for execution: {execution_id}")
        
        # Create MCP task for status monitoring
        task = MCPTask(
            task_type="training_monitor",
            parameters={"execution_id": execution_id}
        )
        
        # Process task
        result = await self.mcp_interface.process_task(task)
        
        if result["status"] == "failed":
            raise RuntimeError(f"Status retrieval failed: {result['error']}")
        
        return result["results"]
    
    async def get_training_results(self, execution_id: str) -> Dict[str, Any]:
        """Get training results"""
        logger.info(f"Getting results for execution: {execution_id}")
        
        # Create MCP task for results retrieval
        task = MCPTask(
            task_type="training_results",
            parameters={"execution_id": execution_id}
        )
        
        # Process task
        result = await self.mcp_interface.process_task(task)
        
        if result["status"] == "failed":
            raise RuntimeError(f"Results retrieval failed: {result['error']}")
        
        return result["results"]
    
    async def cancel_pipeline(self, execution_id: str) -> bool:
        """Cancel pipeline execution"""
        logger.info(f"Cancelling execution: {execution_id}")
        
        # Create MCP task for cancellation
        task = MCPTask(
            task_type="pipeline_cancel",
            parameters={"execution_id": execution_id}
        )
        
        # Process task
        result = await self.mcp_interface.process_task(task)
        
        if result["status"] == "failed":
            raise RuntimeError(f"Cancellation failed: {result['error']}")
        
        return result["results"]["cancelled"]
    
    def get_taxonomy_info(self, category: Optional[str] = None) -> Dict[str, Any]:
        """Get taxonomy information"""
        logger.info("Getting taxonomy information")
        
        if category:
            return self.taxonomy.get_category_info(category)
        else:
            return {
                "categories": self.taxonomy.list_all_categories(),
                "total_categories": len(DatasetCategory)
            }
    
    def get_training_styles_info(self, style: Optional[str] = None) -> Dict[str, Any]:
        """Get training styles information"""
        logger.info("Getting training styles information")
        
        if style:
            try:
                training_style = TrainingStyle(style)
                return {
                    "style_info": self.style_manager._get_style_guidance(training_style),
                    "config_template": self.style_manager.get_config_template(training_style)
                }
            except ValueError:
                raise ValueError(f"Invalid training style: {style}")
        else:
            styles_info = []
            for training_style in TrainingStyle:
                styles_info.append({
                    "style": training_style.value,
                    "info": self.style_manager._get_style_guidance(training_style)
                })
            
            return {
                "styles": styles_info,
                "total_styles": len(TrainingStyle)
            }
    
    def list_active_executions(self) -> List[Dict[str, Any]]:
        """List all active pipeline executions"""
        return self.orchestrator.list_active_executions()
    
    def get_api_app(self) -> Any:
        """Get FastAPI application for serving"""
        return self.api.app
    
    async def start_api_server(self, host: str = "0.0.0.0", port: int = 8000):
        """Start API server"""
        import uvicorn
        
        logger.info(f"Starting API server on {host}:{port}")
        
        config = uvicorn.Config(
            self.api.app,
            host=host,
            port=port,
            log_level="info"
        )
        
        server = uvicorn.Server(config)
        await server.serve()
    
    # Convenience methods for common workflows
    
    async def process_dataset_end_to_end(self, dataset_path: str, 
                                       dataset_name: str,
                                       training_objective: str = "therapeutic_accuracy",
                                       safety_requirements: str = "strict") -> Dict[str, Any]:
        """Process dataset end-to-end: categorize, select style, configure, execute"""
        logger.info(f"Processing dataset end-to-end: {dataset_name}")
        
        try:
            # Step 1: Categorize dataset
            categorization_result = await self.categorize_dataset(dataset_path, dataset_name)
            
            # Step 2: Select training style
            style_selection_result = await self.select_training_style(
                categorization_result["metadata"],
                {"objective": training_objective}
            )
            
            # Step 3: Configure training
            config_result = await self.configure_training(
                style_selection_result["selected_style"],
                optimize=True
            )
            
            # Step 4: Execute training pipeline
            execution_id = await self.execute_training_pipeline({
                "dataset_path": dataset_path,
                "dataset_name": dataset_name,
                "training_config": config_result["config"],
                "safety_requirements": safety_requirements
            })
            
            return {
                "execution_id": execution_id,
                "categorization": categorization_result,
                "style_selection": style_selection_result,
                "configuration": config_result,
                "status": "pipeline_started"
            }
            
        except Exception as e:
            logger.error(f"End-to-end processing failed: {e}")
            raise
    
    async def validate_dataset_pipeline(self, dataset_path: str, 
                                      dataset_name: str) -> Dict[str, Any]:
        """Validate dataset and pipeline configuration"""
        logger.info(f"Validating dataset pipeline: {dataset_name}")
        
        try:
            # Categorize and validate
            categorization_result = await self.categorize_dataset(dataset_path, dataset_name)
            metadata = categorization_result["metadata"]
            
            # Get recommendations
            recommendations = self.taxonomy.get_training_recommendations(
                DatasetMetadata.from_dict(metadata)
            )
            
            # Validate configuration
            if recommendations:
                config_result = await self.configure_training(recommendations[0])
                validation_errors = config_result.get("validation_errors", [])
            else:
                validation_errors = ["No suitable training styles found"]
            
            return {
                "valid": len(validation_errors) == 0,
                "categorization": categorization_result,
                "recommendations": recommendations,
                "validation_errors": validation_errors,
                "validation_timestamp": datetime.utcnow().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Pipeline validation failed: {e}")
            return {
                "valid": False,
                "error": str(e),
                "validation_timestamp": datetime.utcnow().isoformat()
            }


# Example usage and testing
async def test_dataset_training_pipeline():
    """Test the complete dataset training pipeline"""
    print("Testing Dataset Training Pipeline...")
    
    # Initialize pipeline
    pipeline = DatasetTrainingPipeline()
    
    # Create test dataset
    test_dataset = [
        {"text": "I'm feeling anxious about my therapy session tomorrow"},
        {"text": "My therapist suggested trying CBT techniques for my depression"},
        {"text": "I learned some mindfulness exercises in today's session"},
        {"text": "I'm struggling with relationship issues and need support"},
        {"text": "The crisis intervention helped me during a difficult time"}
    ]
    
    test_dataset_path = "./test_therapeutic_dataset.json"
    with open(test_dataset_path, "w") as f:
        json.dump(test_dataset, f, indent=2)
    
    try:
        # Test categorization
        print("Testing dataset categorization...")
        categorization_result = await pipeline.categorize_dataset(
            test_dataset_path,
            "Test Therapeutic Dataset"
        )
        print(f"Recommended category: {categorization_result['recommended_category']}")
        print(f"Confidence: {categorization_result['confidence']}")
        
        # Test style selection
        print("Testing training style selection...")
        style_result = await pipeline.select_training_style(
            categorization_result["metadata"]
        )
        print(f"Selected style: {style_result['selected_style']}")
        
        # Test configuration
        print("Testing training configuration...")
        config_result = await pipeline.configure_training(
            style_result["selected_style"],
            optimize=True
        )
        print(f"Configuration valid: {config_result['valid']}")
        
        # Test taxonomy info
        print("Testing taxonomy information...")
        taxonomy_info = pipeline.get_taxonomy_info()
        print(f"Available categories: {len(taxonomy_info['categories'])}")
        
        # Test styles info
        print("Testing training styles information...")
        styles_info = pipeline.get_training_styles_info()
        print(f"Available styles: {len(styles_info['styles'])}")
        
        # Test pipeline validation
        print("Testing pipeline validation...")
        validation_result = await pipeline.validate_dataset_pipeline(
            test_dataset_path,
            "Test Dataset"
        )
        print(f"Pipeline valid: {validation_result['valid']}")
        
    except Exception as e:
        print(f"Pipeline test failed: {e}")
    
    finally:
        # Cleanup
        import os
        if os.path.exists(test_dataset_path):
            os.remove(test_dataset_path)
    
    print("Dataset training pipeline test completed!")


if __name__ == "__main__":
    asyncio.run(test_dataset_training_pipeline())